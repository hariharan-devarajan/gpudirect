# GCC support can be specified at major, minor, or micro version
# (e.g. 8, 8.2 or 8.2.0).
# See https://hub.docker.com/r/library/gcc/ for all supported GCC
# tags from Docker Hub.
# See https://docs.docker.com/samples/library/gcc/ for more on how to use this image
FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

# These commands copy your files into the specified directory in the image
# and set that as the working location
COPY . /usr/src/myapp

RUN mkdir -p /usr/src/myapp/build /usr/exec 

WORKDIR /usr/src/myapp/build

RUN apt-get update -q --fix-missing && \
    apt-get install -yq gcc g++

RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends cmake

# This command compiles your app using GCC, adjust for your source code
RUN pfs=/usr/exec cmake ../

# This command runs your application, comment out this line to compile only
CMD ["ctest -VV"]

LABEL Name=gpudirect Version=0.0.1
