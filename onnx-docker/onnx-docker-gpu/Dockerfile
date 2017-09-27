# Using this script to build your ONNX image.
# Execute "docker build dir", where dir contains this Dockerfile. Make sure you give docker container at least 8GB memory.
# Execute "docker run -i -t image_id", where image_id is the id of the image you just generated.
# Try "python tutorial-without-mobile-part.py".

FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         ssh \
         vim \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda2-4.2.12-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda install -c soumith magma-cuda80 cuda80 && conda clean -ya
RUN conda install -c conda-forge protobuf && conda clean -ya
RUN conda install conda-build numpy pyyaml scipy ipython mkl setuptools cmake gcc cffi && conda clean -ya

RUN mkdir -p /root/programs

RUN cp /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64
RUN git clone --recursive https://github.com/houseroad/caffe2 -b conda_mkl /root/programs/caffe2 && cd /root/programs/caffe2 && conda-build -c conda-forge -c intel conda/ && conda install -c conda-forge --use-local caffe2 && git clean -xfd

ENV TORCH_NVCC_FLAGS "-Xfatbin -compress-all"
ENV TORCH_CUDA_ARCH_LIST "3.0;3.5;5.0;5.2+PTX;6.0;6.1"
RUN git clone --recursive https://github.com/pytorch/pytorch /root/programs/pytorch && cd /root/programs/pytorch && python setup.py install && git clean -xfd

RUN git clone --recursive https://github.com/onnx/onnx /root/programs/onnx && cd /root/programs && pip install -e onnx/
RUN git clone --recursive https://github.com/onnx/onnx-caffe2 /root/programs/onnx-caffe2 && cd /root/programs && pip install -e onnx-caffe2/

RUN conda install torchvision --no-deps
RUN conda install freetype=2.5.5 jbig=2.1 jpeg=9b libgcc=5.2.0 libpng=1.6.30 libtiff=4.0.6 olefile=0.44 pillow=4.2.1 --no-deps
RUN conda clean -ya

WORKDIR /workspace
RUN chmod -R a+w /workspace

RUN curl -o /workspace/tutorial-with-mobile-part.py "http://pytorch.org/tutorials/_downloads/super_resolution_with_caffe2.py"
RUN sed '/assert_almost_equal/q' /workspace/tutorial-with-mobile-part.py > /workspace/tutorial-without-mobile-part.py
