FROM debian:buster-slim AS mxnet-rust

ENV LANG=C.UTF-8

## Dependencies

RUN set -eux && \
  apt-get update && apt-get upgrade -y  && \
  apt-get install -y --no-install-recommends \
  build-essential \
  apt-utils \
  perl \
  libperl-dev \
  libboost-all-dev \
  ca-certificates \
  libclang-dev \
  clang \
  cmake \
  coreutils \
  curl \
  wget \
  libfreetype6 \
  libfreetype6-dev \
  libharfbuzz-bin \
  libharfbuzz-dev \
  ffmpeg \
  libavcodec-dev \
  git \
  gettext \
  liblcms2-dev \
  libavc1394-dev \
  gcc \
  libc6-dbg \
  libffi-dev \
  libturbojpeg0 \
  libturbojpeg0-dev \
  libjpeg62-turbo \
  libjpeg62-turbo-dev \
  libpng-dev \
  libssl-dev \
  libtbb-dev \
  libwebp-dev \
  linux-headers-amd64 \
  make \
  ninja-build \
  libopenblas-dev \
  liblapack-dev \
  python3-dbg \
  python3-pip \
  tesseract-ocr \
  libtiff-dev \
  unzip \
  zlib1g-dbg && \
  apt-get autoremove -y && \
  apt-get clean -y &&  \
  rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN pip3 install --upgrade pip setuptools

RUN pip install -U pip Pillow pytesseract numpy setuptools

## OpenCV

RUN cd /opt && \
    git clone --recursive https://github.com/opencv/opencv && \
    git clone --recursive https://github.com/opencv/opencv_contrib

RUN mkdir -p /opt/opencv/build && cd /opt/opencv/build && \
  cmake -D CMAKE_BUILD_TYPE=RELWITHDEBINFO \
    -D CMAKE_C_COMPILER=/usr/bin/clang \
    -D CMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D WITH_FFMPEG=ON \
    -D WITH_TBB=ON \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
    -D PYTHON_EXECUTABLE=/usr/bin/python3 \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_PC_FILE_NAME=opencv4.pc \
	-D BUILD_WITH_DEBUG_INFO=ON \
    -g \
    .. \
  && make -j$(nproc) 

RUN cd /opt/opencv/build && make install && cd .. \
    && \
  cp -p $(find /usr/local/lib/python3.7/dist-packages -name cv2.*.so) \
     /usr/lib/python3/dist-packages/cv2.so && \
  python -c 'import cv2; print("Python: import cv2 - SUCCESS")'

# MXNet

RUN cd /opt && git clone --recursive https://github.com/apache/incubator-mxnet mxnet

RUN mkdir -p /opt/mxnet/build && cd /opt/mxnet/build && \
  cmake -D USE_CUDA=0 \
      -D ENABLE_CUDA_RTC=0 \
      -D USE_LIBJPEG_TURBO=1 \
      -D USE_CPP_PACKAGE=1 \
      -D BUILD_CPP_EXAMPLES=0 \
      -D USE_MKLDNN=1 \
      -D CMAKE_BUILD_TYPE=RELWITHDEBINFO \
      -D USE_OPENCV=1 \
      -G Ninja \
      -g \
      ..\
  && ninja 

RUN cd /opt/mxnet/build && ninja install && cd ../python && pip install -e . && \
  python -c 'import mxnet; print("Python: import mxnet - SUCCESS")'

## RUST

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN TEMP=`mktemp` && \
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > TEMP && \
  sh TEMP -y --no-modify-path --profile minimal && rm $TEMP

RUN chmod -R a+w $RUSTUP_HOME $CARGO_HOME && \
  rustup --version && \
  cargo --version && \
  rustc --version
