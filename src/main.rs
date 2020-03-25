#[macro_use]
extern crate lazy_static;

use std::os::raw::{c_char, c_int};
use std::collections::HashMap;
use std::ffi::{CStr, CString};

fn dtype_to_id(dtype: &str) -> i32 {
    match dtype {
        "float32" => 0,
        "float64" => 1,
        "float16" => 2,
        "uint8" => 3,
        "int32" => 4,
        "int8" => 5,
        "int64" => 6,
        _ => panic!("unknown dtype {}", dtype),
    }
}

lazy_static! {
    static ref SYMBOL_CREATORS: HashMap<String, Creator> = {
        let mut num_symbol_creators = 0;
        let mut symbol_creators = std::ptr::null_mut();
        unsafe {
            assert_eq!(
                mxnet_sys::MXSymbolListAtomicSymbolCreators(
                    &mut num_symbol_creators,
                    &mut symbol_creators
                ),
                0
            );
        }
        let symbol_creators_slice =
            unsafe { std::slice::from_raw_parts(symbol_creators, num_symbol_creators as usize) };
        let mut map = HashMap::with_capacity(num_symbol_creators as usize);
        for symbol_creator in symbol_creators_slice {
            let mut name = std::ptr::null();
            let mut description = std::ptr::null();
            let mut num_args = 0;
            let mut arg_names = std::ptr::null();
            let mut arg_type_infos = std::ptr::null();
            let mut arg_descriptions = std::ptr::null();
            let mut key_var_num_args = std::ptr::null();
            unsafe {
                assert_eq!(
                    mxnet_sys::MXSymbolGetAtomicSymbolInfo(
                        *symbol_creator,
                        &mut name,
                        &mut description,
                        &mut num_args,
                        &mut arg_names,
                        &mut arg_type_infos,
                        &mut arg_descriptions,
                        &mut key_var_num_args,
                        std::ptr::null_mut()
                    ),
                    0
                );
            }
            let arguments = unsafe { std::slice::from_raw_parts(arg_names, num_args as usize) }
                .into_iter()
                .zip(
                    unsafe { std::slice::from_raw_parts(arg_type_infos, num_args as usize) }
                        .into_iter(),
                )
                .filter(|(_, ctype)| {
                    let typ = unsafe { CStr::from_ptr(**ctype) }.to_string_lossy();
                    !(typ.starts_with("NDArray")
                        || typ.starts_with("Symbol")
                        || typ.starts_with("NDArray-or-Symbol"))
                })
                .map(|(name, _)| {
                    unsafe { CStr::from_ptr(*name) }
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
            let name = unsafe { CStr::from_ptr(name) };
            map.insert(
                name.to_string_lossy().into_owned(),
                Creator {
                    inner: *symbol_creator,
                    arguments: arguments,
                },
            );
        }
        map
    };
}

macro_rules! c_try {
    ($x: expr) => {{
        let res = $x;
        if res != 0 {
           panic!( 
                "mxnet error ({} != 0): {}\n{}",
                res,
                stringify!($x),
                std::ffi::CStr::from_ptr(::mxnet_sys::MXGetLastError()).to_string_lossy()
            )
        }
    }};
}

#[derive(Debug)]
struct Creator {
    inner: mxnet_sys::AtomicSymbolCreator,
    arguments: Vec<String>,
}
unsafe impl Sync for Creator {}

pub trait DebugAndToString: std::fmt::Debug + std::string::ToString {}
impl DebugAndToString for str {}
impl DebugAndToString for &str {}
impl DebugAndToString for f32 {}
impl DebugAndToString for u64 {}
impl DebugAndToString for usize {}
impl DebugAndToString for String {}

#[derive(Debug)]
pub enum IntoArgument<'a> {
    ToString(&'a dyn DebugAndToString),
    NDArray(&'a NDArray),
}

pub enum IntoOutput<'a> {
    NDArray(&'a mut NDArray),
    None,
}

impl<'a> IntoArgument<'a> {
    fn is_string(&self) -> bool {
        match *self {
            IntoArgument::ToString(_) => true,
            _ => false,
        }
    }

    fn string_value(&self) -> String {
        match *self {
            IntoArgument::ToString(s) => s.to_string(),
            _ => panic!("unexpected string value request to non-string argument"),
        }
    }
}

#[derive(Debug)]
pub struct NDArray(pub std::ptr::NonNull<mxnet_sys::OpaqueNDArrayHandle>);

impl NDArray {
    pub fn new(ptr: mxnet_sys::NDArrayHandle) -> Self {
        NDArray(std::ptr::NonNull::new(ptr).unwrap())
    }

    pub unsafe fn create(shape: &[usize]) -> NDArray  {
        Self::create_dtype(shape, "float32")
    }

    pub unsafe fn create_dtype(shape: &[usize], dtype: &str) -> NDArray {
        let mut ndarr = std::ptr::null_mut();
        let shape_arr = shape.iter().map(|x| *x as u32).collect::<Vec<_>>();

{c_try!( mxnet_sys::MXNDArrayWaitAll());}
        c_try!(mxnet_sys::MXNDArrayCreateEx(
            shape_arr.as_ptr(),
            shape.len() as u32,
            1,
            0,
            0,
            dtype_to_id(dtype),
            &mut ndarr
        ));

        NDArray::new(ndarr)
    }

    pub fn copy_slice_u8(&mut self, data: &[u8]) {
unsafe {c_try!( mxnet_sys::MXNDArrayWaitAll());}
        unsafe {
            c_try!(mxnet_sys::MXNDArraySyncCopyFromCPU(
                self.0.as_ptr(),
                data.as_ptr() as *const std::ffi::c_void,
                data.len() as usize,
            ));
        }
unsafe {c_try!( mxnet_sys::MXNDArrayWaitAll());}
    }

    pub fn generic_function_invoke<'a, Args: std::iter::Iterator<Item = IntoArgument<'a>>>(
        func_name: &str,
        args: Args,
        kwargs: HashMap<&'a str, IntoArgument<'a>>,
        out: IntoOutput,
    ) -> Option<NDArray> {
        let function = SYMBOL_CREATORS
            .get(func_name).unwrap();
        let (nd_args, pos_args) = args.fold((Vec::new(), Vec::new()), |mut acc, arg| {
            match arg {
                IntoArgument::NDArray(ndarray) => acc.0.push(ndarray),
                IntoArgument::ToString(ts) => acc.1.push(ts.to_string()),
            };
            acc
        });
        assert!(pos_args.len() <= function.arguments.len());

        let updated_kwargs = kwargs
            .iter()
            .filter(|(_, val)| val.is_string())
            .map(|(key, val)| (key.to_string(), val.string_value()))
            .chain(
                function
                    .arguments
                    .iter()
                    .zip(pos_args.into_iter())
                    .map(|(x, y)| (x.clone(), y)),
            )
            .collect::<HashMap<_, _>>();

        let mut outputs = std::ptr::null_mut();
        let outputs_owner = match out {
            IntoOutput::NDArray(ref a) => Some([a.0.as_ptr()]),
            IntoOutput::None => None,
        };
        if let Some(mut o) = outputs_owner {
            outputs = o.as_mut_ptr();
        }
        let mut out_stypes = std::ptr::null_mut();
        let num_outputs_exp = outputs_owner.is_some() as i32;
        let mut num_outputs = num_outputs_exp.clone();

        let updated_kwargs_keys = updated_kwargs
            .keys()
            .map(|k| CString::new(&**k).unwrap())
            .collect::<Vec<_>>();
        let updated_kwargs_keys_ptrs = updated_kwargs_keys
            .iter()
            .map(|x| x.as_ptr())
            .collect::<Vec<_>>();
        let updated_kwargs_vals = updated_kwargs
            .values()
            .map(|v| CString::new(&**v).unwrap())
            .collect::<Vec<_>>();
        let updated_kwargs_vals_ptrs = updated_kwargs_vals
            .iter()
            .map(|x| x.as_ptr())
            .collect::<Vec<_>>();
        let ndargs_ptrs = nd_args
            .into_iter()
            .map(|x| x.0.as_ptr())
            .collect::<Vec<_>>();

        println!("calling {}, outputs {:?}", func_name, outputs_owner);
        unsafe {
            c_try!(MXImperativeInvokeEx(
                function.inner,
                ndargs_ptrs.len() as i32,
                ndargs_ptrs.as_ptr(),
                &mut num_outputs,
                &mut outputs,
                updated_kwargs.len() as i32,
                updated_kwargs_keys_ptrs.as_ptr(),
                updated_kwargs_vals_ptrs.as_ptr(),
                &mut out_stypes
            ));
        }
        if num_outputs != num_outputs_exp {
            if num_outputs == 1 {
                Some(Self::new(
                    unsafe { std::slice::from_raw_parts(outputs, 1) }[0],
                ))
            } else {
                unimplemented!();
            }
        } else {
            None
        }
    }

    pub fn shape(&self) -> Vec<usize> {
unsafe {c_try!( mxnet_sys::MXNDArrayWaitAll());}
        let mut out_pdata = std::ptr::null();
        let mut out_dim = 0;
        unsafe {
            c_try!(mxnet_sys::MXNDArrayGetShape(
                self.0.as_ptr(),
                &mut out_dim,
                &mut out_pdata
            ));
            std::slice::from_raw_parts(out_pdata, out_dim as usize)
                .into_iter()
                .map(|x| *x as usize)
                .collect()
        }
    }

    pub fn copy_to(&self, other: &mut NDArray) {
        Self::generic_function_invoke(
            "_copyto",
            vec![IntoArgument::NDArray(self)].into_iter(),
            HashMap::new(),
            IntoOutput::NDArray(other),
        );
    }
}

#[link(name = "mxnet")]
extern "C" {
    pub fn MXImperativeInvokeEx(
        creator: mxnet_sys::AtomicSymbolCreator,
        num_inputs: c_int,
        inputs: *const mxnet_sys::NDArrayHandle,
        num_outputs: *mut c_int,
        outputs: *mut *mut mxnet_sys::NDArrayHandle,
        num_params: c_int,
        param_keys: *const *const c_char,
        param_vals: *const *const c_char,
        out_stypes: *mut *mut c_int,
    ) -> c_int;
}


fn main() {
    let image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAECAYAAABP2FU6AAAABHNCSVQICAgIfAhkiAAAAB1JREFUCJlj+H+D4T8Dg9r//0z1LowMTAevOTAAAFe6B5o50EKmAAAAAElFTkSuQmCC";
    let bytes = base64::decode(image_b64).unwrap();
 
    let mut bytend = unsafe { NDArray::create_dtype(&[bytes.len()], "uint8") };
    bytend.copy_slice_u8(&bytes);
    let arr_u8 = NDArray::generic_function_invoke(
        "_cvimdecode",
        vec![IntoArgument::NDArray(&bytend)].into_iter(),
        std::collections::HashMap::new(),
        IntoOutput::None,
    ).unwrap();
    let mut arr = unsafe { NDArray::create(&arr_u8.shape()) };
    arr_u8.copy_to(&mut arr);
    println!("{:?}", arr)
}
