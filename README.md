# Maybe problem with Rust FFI on release?

Running `cargo run` works fine. However using `--release` fails with either a segfault or getting
a number of dimensions lower than `-1`.

## gdb debug

```
root@7d3e5c30801c:/workspaces/testing-mxnet# cargo build
...
root@7d3e5c30801c:/workspaces/testing-mxnet# gdb ./target/debug/testing-mxnet
...
(gdb) start
...
(gdb) b /opt/mxnet/src/c_api/c_api_ndarray.cc:155
Breakpoint 2 at 0x7ffff2f5d3a9: file ../src/c_api/c_api_ndarray.cc, line 155.
(gdb) c
...
calling _cvimdecode, outputs None

Thread 1 "testing-mxnet" hit Breakpoint 2, MXImperativeInvokeEx (creator=0x555555610420, num_inputs=1, inputs=0x5555555f9810, num_outputs=0x7fffffffe464, outputs=0x7fffffffe428, num_params=0,
    param_keys=0x8, param_vals=0x8, out_stypes=0x7fffffffe458) at ../src/c_api/c_api_ndarray.cc:155
155	  MXImperativeInvokeImpl(creator, num_inputs, inputs, num_outputs, outputs,
(gdb) c
Continuing.
calling _copyto, outputs Some([0x5555559c2f60])

Thread 1 "testing-mxnet" hit Breakpoint 2, MXImperativeInvokeEx (creator=0x555555615ba0, num_inputs=1, inputs=0x5555559c20a0, num_outputs=0x7fffffffe3a4, outputs=0x7fffffffe368, num_params=0,
    param_keys=0x8, param_vals=0x8, out_stypes=0x7fffffffe398) at ../src/c_api/c_api_ndarray.cc:155
155	  MXImperativeInvokeImpl(creator, num_inputs, inputs, num_outputs, outputs,
(gdb) p **outputs
$1 = (NDArrayHandle) 0x5555559c2f60
...
[Inferior 1 (process 20163) exited normally]
(gdb) quit
```

## gdb release

```
root@7d3e5c30801c:/workspaces/testing-mxnet# cargo build --release
...
root@7d3e5c30801c:/workspaces/testing-mxnet# gdb ./target/release/testing-mxnet
...
(gdb) start
...
(gdb) b /opt/mxnet/src/c_api/c_api_ndarray.cc:155
Breakpoint 2 at 0x7ffff2f5d3a9: file ../src/c_api/c_api_ndarray.cc, line 155.
(gdb) c
...
calling _cvimdecode, outputs None

Thread 1 "testing-mxnet" hit Breakpoint 2, MXImperativeInvokeEx (creator=0x5555555c8420, num_inputs=1, inputs=0x5555555b1810, num_outputs=0x7fffffffe67c, outputs=0x7fffffffe680, num_params=0,
    param_keys=0x8, param_vals=0x8, out_stypes=0x7fffffffe778) at ../src/c_api/c_api_ndarray.cc:155
155	  MXImperativeInvokeImpl(creator, num_inputs, inputs, num_outputs, outputs,
(gdb) c
Continuing.
calling _copyto, outputs Some([0x55555597af60])

Thread 1 "testing-mxnet" hit Breakpoint 2, MXImperativeInvokeEx (creator=0x5555555cdba0, num_inputs=1, inputs=0x55555597a0a0, num_outputs=0x7fffffffe67c, outputs=0x7fffffffe680, num_params=0,
    param_keys=0x8, param_vals=0x8, out_stypes=0x7fffffffe778) at ../src/c_api/c_api_ndarray.cc:155
155	  MXImperativeInvokeImpl(creator, num_inputs, inputs, num_outputs, outputs,
(gdb) p **outputs
$1 = (NDArrayHandle) 0x555555592f68
...
```
