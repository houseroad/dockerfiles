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

RUN conda install -c conda-forge protobuf
RUN conda install -c soumith magma-cuda80 cuda80
RUN conda install conda-build numpy pyyaml scipy ipython mkl setuptools cmake gcc cffi

RUN mkdir -p /root/programs

RUN git clone --recursive https://github.com/caffe2/caffe2 /root/programs/caffe2
RUN cd /root/programs/caffe2 && conda-build -c conda-forge conda/
RUN cd /root/programs/caffe2 && conda install -c conda-forge --use-local caffe2

RUN git clone --recursive https://github.com/pytorch/pytorch /root/programs/pytorch
RUN cd /root/programs/pytorch && python setup.py install

RUN conda install -c ezyang onnx onnx-caffe2

RUN conda clean -ya

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

WORKDIR /workspace
RUN chmod -R a+w /workspace

# RUN curl -o /workspace/tutorial-with-mobile-part.py "http://pytorch.org/tutorials/_downloads/super_resolution_with_caffe2.py"
RUN curl -o /workspace/tutorial-with-mobile-part.py "https://raw.githubusercontent.com/houseroad/tutorials/6ca0f1f2fd02a6d9405c1fd99fdd6c504d0a9038/advanced_source/super_resolution_with_caffe2.py"
RUN sed '/assert_almost_equal/q' /workspace/tutorial-with-mobile-part.py > /workspace/tutorial-without-mobile-part.py
