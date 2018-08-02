# Installing-MXNet-on-JetsonTx2

Build the MXNet core shared library

Step 1 Install build tools and git.
```
$ sudo apt-get update
$ sudo apt-get install -y build-essential git
```
Step 2 Install OpenBLAS.

MXNet uses BLAS and LAPACK libraries for accelerated numerical computations on CPU machine. There are several flavors of BLAS/LAPACK libraries - OpenBLAS, ATLAS and MKL. In this step we install OpenBLAS. You can choose to install ATLAS or MKL.
```
$ sudo apt-get install -y libopenblas-dev liblapack-dev
```
Step 3 Install OpenCV.

MXNet uses OpenCV for efficient image loading and augmentation operations.
```
$ sudo apt-get install -y libopencv-dev
```
Step 4 Download MXNet sources and build MXNet core shared library. You can clone the repository as described in the following code block, or you may try the download links for your desired MXNet version.
```
$ git clone --recursive https://github.com/apache/incubator-mxnet
$ cd incubator-mxnet
$ make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda-9.0 USE_CUDNN=1

```
Note - USE_OPENCV, USE_BLAS, USE_CUDA, USE_CUDA_PATH AND USE_CUDNN are make file flags to set compilation options to use OpenCV, OpenBLAS, CUDA and cuDNN libraries. You can explore and use more compilation options in make/config.mk. Make sure to set USE_CUDA_PATH to right CUDA installation path. In most cases it is - /usr/local/cuda.



Install the MXNet Python binding

Step 1 Install prerequisites - python, setup-tools, python-pip and libfortran (required for Numpy)..
```
$ sudo apt-get install -y python3-dev python3-setuptools python3pip libgfortran3
```
Step 2 Install the MXNet Python binding.
```
$ cd python
$ pip3 install -e .
```
Note that the -e flag is optional. It is equivalent to --editable and means that if you edit the source files, these changes will be reflected in the package installed.

Step 3 Install Graphviz. (Optional, needed for graph visualization using mxnet.viz package).
```
sudo apt-get install graphviz
pip3 install graphviz

```
Step 4 Validate the installation by running simple MXNet code described here.

Validate MXNet Installation
Start the python terminal.
```
$ python3
```
Run a short MXNet python program to create a 2X3 matrix of ones a on a GPU, multiply each element in the matrix by 2 followed by adding 1. We expect the output to be a 2X3 matrix with all elements being 3. We use mx.gpu(), to set MXNet context to be GPUs.
```
>>> import mxnet as mx
>>> a = mx.nd.ones((2, 3), mx.gpu())
>>> b = a * 2 + 1
>>> b.asnumpy()
array([[ 3.,  3.,  3.],
       [ 3.,  3.,  3.]], dtype=float32)
       
```
