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
         liblmdb-dev && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda2-4.2.12-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \     
     rm ~/miniconda.sh

RUN mkdir -p ~/programs/ninja && cd ~/programs/ninja \
    && wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip \
    && unzip ninja-linux.zip

ENV PATH /opt/conda/bin:$PATH

RUN conda install -c soumith magma-cuda80 cuda80
RUN conda install conda-build numpy pyyaml scipy ipython mkl setuptools cmake cffi future six
RUN conda install -c intel mkl-include

RUN mkdir -p /root/programs
RUN pip install --upgrade git+https://github.com/Maratyszcza/PeachPy
RUN pip install --upgrade git+https://github.com/Maratyszcza/confu

#RUN git clone --recursive https://github.com/houseroad/caffe2 -b conda_mkl /root/programs/caffe2
#RUN cd /root/programs/caffe2 && conda-build -c conda-forge -c intel conda/
#RUN cd /root/programs/caffe2 && conda install -c conda-forge --use-local caffe2
RUN conda install -c ezyang/label/gcc5 protobuf
RUN git clone --recursive https://github.com/houseroad/caffe2 /root/programs/caffe2 \
    && cd /root/programs/caffe2 \
    && mkdir build && cd build && cmake .. -DBLAS="MKL" && make -j20 && make install -j20 \
    && cd .. && git clean -xfd

RUN git clone --recursive https://github.com/pytorch/pytorch /root/programs/pytorch
RUN cd /root/programs/pytorch && python setup.py install

ENV CONDA_PREFIX /opt/conda
RUN git clone --recursive https://github.com/onnx/onnx /root/programs/onnx && cd /root/programs && pip install -e onnx/
RUN git clone --recursive https://github.com/onnx/onnx-caffe2 /root/programs/onnx-caffe2 && cd /root/programs && pip install -e onnx-caffe2/

RUN conda install torchvision --no-deps
RUN conda install freetype=2.5.5 jbig=2.1 jpeg=9b libgcc=5.2.0 libpng=1.6.30 libtiff=4.0.6 olefile=0.44 pillow=4.2.1 --no-deps
RUN conda clean -ya

ENV PYTHONPATH /usr/local
ENV LD_LIBRARY_PATH /usr/local/lib

WORKDIR /workspace
RUN chmod -R a+w /workspace

#RUN curl -o /workspace/tutorial-with-mobile-part.py "http://pytorch.org/tutorials/_downloads/super_resolution_with_caffe2.py"
RUN curl -o /workspace/tutorial-with-mobile-part.py "https://raw.githubusercontent.com/houseroad/tutorials/95bf3983baa537bb97e4a069e665fd68b05c616f/advanced_source/super_resolution_with_caffe2.py"
RUN sed '/result looks good/q' /workspace/tutorial-with-mobile-part.py > /workspace/tutorial-without-mobile-part.py
