FROM ubuntu:16.04 

LABEL com.nvidia.volumes.needed="nvidia_driver"
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \ 
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

RUN conda install -c soumith magma-cuda80 cuda80
RUN conda install -c conda-forge protobuf
RUN conda install conda-build numpy pyyaml scipy ipython mkl setuptools cmake gcc cffi

RUN mkdir -p /root/programs

RUN git clone --recursive https://github.com/houseroad/caffe2 -b conda_mkl /root/programs/caffe2
RUN cd /root/programs/caffe2 && conda-build -c conda-forge -c intel conda/
RUN cd /root/programs/caffe2 && conda install -c conda-forge --use-local caffe2

RUN git clone --recursive https://github.com/pytorch/pytorch /root/programs/pytorch
RUN cd /root/programs/pytorch && python setup.py install

RUN git clone --recursive https://github.com/onnx/onnx /root/programs/onnx && cd /root/programs && pip install -e onnx/
RUN git clone --recursive https://github.com/onnx/onnx-caffe2 /root/programs/onnx-caffe2 && cd /root/programs && pip install -e onnx-caffe2/

RUN conda install torchvision --no-deps
RUN conda install freetype=2.5.5 jbig=2.1 jpeg=9b libgcc=5.2.0 libpng=1.6.30 libtiff=4.0.6 olefile=0.44 pillow=4.2.1 --no-deps
RUN conda clean -ya

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64


WORKDIR /workspace
RUN chmod -R a+w /workspace

RUN curl -o /workspace/tutorial-with-mobile-part.py "http://pytorch.org/tutorials/_downloads/super_resolution_with_caffe2.py"
RUN sed '/assert_almost_equal/q' /workspace/tutorial-with-mobile-part.py > /workspace/tutorial-without-mobile-part.py
