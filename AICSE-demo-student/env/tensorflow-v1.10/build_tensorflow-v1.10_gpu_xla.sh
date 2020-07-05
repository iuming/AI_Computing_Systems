#!/bin/bash
set -e

usage () {
    echo
    echo "      -c or --no-build       Configure only, will not build the whl package"
    echo "      null                   Configure and build the whl package"
}

echo "=== env ================================================================="
export WORKSPACE=${PWD}
export PATH_TFVENV_GPU_XLA=${WORKSPACE}/virtualenv_gpu_xla

export PYTHON_BIN_PATH=$(which python || which python3  || true)
export PYTHON_LIB_PATH="/usr/local/lib/python2.7/dist-packages"
export CC_OPT_FLAGS=$([ -z $CC_OPT_FLAGS ] && echo "-march=native" || echo $CC_OPT_FLAGS)
export TF_NEED_JEMALLOC=1
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_NEED_S3=0
export TF_NEED_GDR=0
export TF_ENABLE_XLA=1
export TF_NEED_OPENCL=0
export TF_NEED_MKL=0
export TF_NEED_VERBS=0
export TF_NEED_MPI=0

export TF_NEED_MLU=0

# CUDA config
export TF_NEED_CUDA=1
export TF_CUDA_CLANG=0
export TF_CUDA_VERSION=7.5
export TF_CUDNN_VERSION=6.0.21
export CUDA_TOOLKIT_PATH=/usr/local/cuda
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
export CUDNN_INSTALL_PATH=/usr/local/cuda
export TF_CUDA_COMPUTE_CAPABILITIES=3.5

echo "=== config  ============================================================="
[ ! -e ${PATH_TFVENV_GPU_XLA} ] && virtualenv --system-site-packages ${PATH_TFVENV_GPU_XLA}
source ${PATH_TFVENV_GPU_XLA}/bin/activate

# If missing some whl packages, uncomment these lines to install.
#################################################################################
# if [ ! -e ${HOME}/.pip/pip.conf ]; then
#     mkdir -p ${HOME}/.pip
#     echo "[global]" >> ${HOME}/.pip/pip.conf
#     echo "index-url = http://mirrors.cambricon.com/pypi/web/simple" >> ${HOME}/.pip/pip.conf
#     echo "trusted-host = mirrors.cambricon.com" >> ${HOME}/.pip/pip.conf
# fi
# pip install protobuf scipy opencv-python pillow mock --trusted-host mirrors.cambricon.com
#################################################################################

./configure

while [ $# -gt 0 ]; do
    case "$1" in
        -c | --no-build )
            exit 0
            ;;
        * )
            usage
            exit 1
            ;;
    esac
done

echo "=== build ==============================================================="
bazel build //tensorflow/tools/pip_package:build_pip_package \
    --verbose_failures \
    -c opt \
    --config=cuda \

bazel-bin/tensorflow/tools/pip_package/build_pip_package ${PATH_TFVENV_GPU_XLA} --gpu

# Upgrade the whl package
pip install ${PATH_TFVENV_GPU_XLA}/tensorflow_gpu-1.10*whl -U --no-deps
