# GCC support can be specified at major, minor, or micro version
# (e.g. 8, 8.2 or 8.2.0).
# See https://hub.docker.com/r/library/gcc/ for all supported GCC
# tags from Docker Hub.
# See https://docs.docker.com/samples/library/gcc/ for more on how to use this image
FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

# These commands copy your files into the specified directory in the image
# and set that as the working location

RUN apt-get update -q --fix-missing && \
    apt-get install -yq gcc g++

RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    ca-certificates \
    curl \
    environment-modules \
    git \
    build-essential \
    python \
    python-dev \
    python3-dev \
    vim \
    sudo \
    unzip \
    cmake \
    lcov \
    zlib1g-dev \
    libsdl2-dev \
    gfortran \
    graphviz \
    doxygen \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    libelf-dev libdwarf-dev

RUN mkdir -p /etc/apt/keyrings && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

RUN apt-get update -q --fix-missing && \
    apt-get install -yq docker-ce docker-ce-cli containerd.io docker-compose-plugin

COPY . /usr/gpudirect    

RUN mkdir -p /usr/gpudirect/build /usr/exec 

WORKDIR /usr/gpudirect/build

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:/usr/local/cuda-11.4/compat/

# This command compiles your app using GCC, adjust for your source code
RUN pfs=/usr/exec cmake ../ && make -j

# This command runs your application, comment out this line to compile only
CMD ["/bin/bash"]

LABEL Name=gpudirect Version=0.0.1
