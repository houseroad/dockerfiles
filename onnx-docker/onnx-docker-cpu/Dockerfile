FROM ubuntu:16.04

LABEL com.nvidia.volumes.needed="nvidia_driver"
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \ 
         git \
         ssh \
         vim \
         unzip \
         curl \
         wget \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         libgflags-dev \
         libgoogle-glog-dev \
         liblmdb-dev \
         libprotobuf-dev \
         protobuf-compiler \
         cmake \
         python python-pip python-dev python-setuptools && \
    rm -rf /var/lib/apt/lists/*

RUN pip install wheel && pip install ipython==5.0

RUN mkdir -p /root/programs && \
    git clone --recursive https://github.com/onnxbot/onnx-fb-universe.git /root/programs

RUN cd /root/programs && ./install-develop.sh

WORKDIR /root/programs
