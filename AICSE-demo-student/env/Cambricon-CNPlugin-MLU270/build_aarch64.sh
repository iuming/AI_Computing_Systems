#!/bin/bash
set -e
export WORKSPACE=${PWD}

#cross_compile set
export COMPILE_ENV_PATH=/opt/shared/aarch64_linux_lib/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/bin/:$PATH
export TARGET_CHIP_TYPE="armv8a"
export PATH=${COMPILE_ENV_PATH}/bin/:$PATH
CROSS_COMPILE=aarch64-linux-gnu-
which aarch64-linux-gnu-g++

BUILD_TEST="OFF"
BUILD_RELEASE="OFF"

usage () {
    echo "USAGE: build.sh <options>"
    echo
    echo "       If need specify neuware path, please:"
    echo "       Firt, export NEUWARE"
    echo "         export NEUWARE=/path/of/your/neuware"
    echo "       Then, run build.sh --use-stable-neuware"
    echo
    echo "OPTIONS:"
    echo "      -h, --help                     Print usage"
    echo "      --mlu200                       Build for MLU target arch MLU200: NRAM = 512KB, WRAM = 512KB,  __BANG_ARCH__ = 200"
    echo "                                     If not specified, default arch is MLU200"
    echo "      --mlu220                       Build for MLU target arch MLU220: NRAM = 512KB, WRAM = 512KB,  __BANG_ARCH__ = 220"
    echo "      --mlu100                       Build for MLU target arch MLU100: NRAM = 512KB, WRAM = 512KB,  __BANG_ARCH__ = 100"
    echo "      --mlu270                       Build for MLU target arch MLU270: NRAM = 512KB, WRAM = 1024KB, __BANG_ARCH__ = 270"
    echo "      --mlu290                       Build for MLU target arch MLU290: NRAM = 512KB, WRAM = 512KB,  __BANG_ARCH__ = 290"
    echo "      -d, --debug                    Build plugin op with debug mode"
    echo "      -g, --gtest                    Build test case in gtest"
    echo "      -r, --release                  Build with compiled cncc & cnas (for release)"
    echo
}


BUILD_DIR=${PWD}/build
[[ -e ${BUILD_DIR} ]] && rm -rf ${BUILD_DIR}; mkdir ${BUILD_DIR}

prepare_neuware() {
 if [ ${BUILD_RELEASE} == "OFF" ]; then
   if [ -z ${NEUWARE} ]; then
     echo "Caution: env NEUWARE was NULL, please export NEUWARE !"
     echo
     echo "Make sure the path tree in NEUWARE like this:"
     echo "   NEUWARE"
     echo "      |---> bin"
     echo "      |      | ---> cncc"
     echo "      |      | ---> cnas"
     echo "      |---> include"
     echo "      |      | ---> cnml.h"
     echo "      |      | ---> cnrt.h"
     echo "      |---> lib64"
     echo "             | ---> libcnml.so"
     echo "             | ---> libcnrt.so"
     exit 1
   else
     export PATH=${NEUWARE}/bin:$PATH
     CNCC="${NEUWARE}/bin/cncc"
     CNAS="${NEUWARE}/bin/cnas"
     HEAD_CNML="${NEUWARE}/include/cnml.h"
     HEAD_CNRT="${NEUWARE}/include/cnrt.h"
     LIB_CNML="${NEUWARE}/lib64/libcnml.so"
     LIB_CNRT="${NEUWARE}/lib64/libcnrt.so"

     echo "-- check NEUWAEW bin file"
     if [ ! -e ${NEUWARE}/bin ]; then
       echo "-- NEUWARE bin folder not exist"
       exit 1
     else
       if [ ! -f ${CNCC} ]; then
         echo "-- NEUWARE bin/CNCC not exit"
         exit 1
       fi
       if [ ! -f ${CNAS} ]; then
         echo "-- NEUWARE bin/CNAS not exit"
         exit 1
       fi
     fi

     echo "-- check NEUWAEW include file"
     if [ ! -e ${NEUWARE}/include ]; then
       echo "-- NEUWARE include folder not exist"
       exit 1
     else
       if [ ! -f ${HEAD_CNML} ]; then
         echo "-- NEUWARE include/cnml.h not exit"
         exit 1
       fi
       if [ ! -f ${HEAD_CNRT} ]; then
         echo "-- NEUWARE include/cnrt.h not exit"
         exit 1
       fi
     fi
     echo "-- check NEUWARE lib64 file"
     if [ ! -e ${NEUWARE}/lib64 ]; then
       echo "-- NEUWARE lib folder not exist"
       exit 1
     else
       if [ ! -f ${LIB_CNML} ]; then
         echo "-- NEUWARE lib64/libcnml.so not exit"
         exit 1
       fi
       if [ ! -f ${LIB_CNRT} ]; then
         echo "-- NEUWARE lib64/libcnrt.so not exit"
         exit 1
       fi
     fi
   fi
 else
   SOURCE_PATH=$(dirname "${PWD}")
   CNAS="${SOURCE_PATH}/../host-cnas/build/bin"
   CNCC="${SOURCE_PATH}/../host-cncc/build/bin"
   export PATH=${CNAS}:${CNCC}:$PATH
   echo "-- use sopa for release : ${SOURCE_PATH}"
   echo "-- use cncc for release : ${CNAS}"
   echo "-- use cnas for release : ${CNCC}"
   if [ ! -f ${CNAS}/cnas ]; then
     echo "-- cnas not exist!"
     exit 1
   fi
   if [ ! -f ${CNCC}/cncc ]; then
     echo "-- cncc not exist!"
     exit 1
   fi
   HEAD_CNML="${SOURCE_PATH}/out/neuware_home/include/cnml.h"
   HEAD_CNRT="${SOURCE_PATH}/out/neuware_home/include/cnrt.h"
   LIB_CNML="${SOURCE_PATH}/out/neuware_home/lib64/libcnml.so"
   LIB_CNRT="${SOURCE_PATH}/out/neuware_home/lib64/libcnrt.so"
   if [ ! -f ${HEAD_CNML} ]; then
     echo "-- cnml head file not exist!"
     exit 1
   fi
   if [ ! -f ${HEAD_CNRT} ]; then
     echo "-- cnrt head file not exist!"
     exit 1
   fi
   if [ ! -f ${LIB_CNML} ]; then
     echo "-- cnml lib file not exist!"
     exit 1
   fi
   if [ ! -f ${LIB_CNRT} ]; then
     echo "-- cnrt lib file not exist!"
     exit 1
   fi
 fi
}

# default target mlu arch is MLU270
TARGET_MLU_ARCH="MLU270"
BANG_ARCH="270"
MAKE_FLAG=

# build different version
if [ $# != 0 ]; then
  BUILD_MODE="release"
  while [ $# != 0 ]; do
    case "$1" in
      -h | --help)
          usage
          exit 0
          ;;
      --mlu200)
          TARGET_MLU_ARCH="MLU200"
          BANG_ARCH="200"
          FLOAT_MODE="0"
          echo "-- build target arch MLU200."
          shift
          ;;
      --mlu220)
          TARGET_MLU_ARCH="MLU220"
          BANG_ARCH="220"
          BANG_LOG="0"
          FLOAT_MODE="0"
          echo "-- build target arch MLU220."
          shift
          ;;
      --mlu270)
          TARGET_MLU_ARCH="MLU270"
          BANG_ARCH="270"
          BANG_LOG="0"
          FLOAT_MODE="0"
          echo "-- build target arch MLU270."
          shift
          ;;
      --mlu290)
          TARGET_MLU_ARCH="MLU290"
          BANG_ARCH="290"
          BANG_LOG="0"
          FLOAT_MODE="0"
          echo "-- build target arch MLU290."
          shift
          ;;
      --mlu100)
          TARGET_MLU_ARCH="MLU100"
          BANG_ARCH="100"
          BANG_LOG="0"
          FLOAT_MODE="0"
          echo "-- build target arch MLU100."
          shift
          ;;
      -d | --debug)
          BANG_LOG="1"
          BUILD_MODE="debug"
          MAKE_FLAG=-g
          echo "-- use debug mode."
          shift
          ;;
      -g | --gtest)
          BUILD_TEST="ON"
          echo "-- build gtest."
          shift
          ;;
      -r | --release)
          BUILD_RELEASE="ON"
          echo "-- build with release cncc and case."
          shift
          ;;
      *)
          echo "unknown options ${1}, use -h or --help"
          exit -1;
          ;;
    esac
  done
else
  BUILD_MODE="release"
fi
echo "-- build target arch : ${TARGET_MLU_ARCH}"
echo "-- build mode : ${BUILD_MODE}"

prepare_neuware

PLUGIN_OP_DIR=${PWD}/pluginops
BUILD_PLUGINOP_DIR=${BUILD_DIR}/pluginops
[[ -e ${BUILD_PLUGINOP_DIR} ]] && rm -rf ${BUILD_PLUGINOP_DIR}; mkdir ${BUILD_PLUGINOP_DIR}

# build plugin op
for file in `ls ${PLUGIN_OP_DIR}/*/*.mlu`
do
    file_base_name=$(basename $file .mlu)
    echo "cncc build ${file_base_name}"
    # fasterrcnn_detection_output  需要 FLOAT_MODE 宏，roipool 有这个宏，但是不应该有值；
    if [[ $file_base_name =~ "fasterrcnn_detection_output" ||
          $file_base_name =~ "roipool" ||
          $file_base_name =~ "proposal" ||
          $file_base_name =~ "yolov3_detection_output" ||
          $file_base_name =~ "yolov2_detection_output" ||
          $file_base_name =~ "ssd_detection_output" ||
          $file_base_name =~ "resize_yuv_to_rgba" ||
          $file_base_name =~ "resize_convert" ||
          $file_base_name =~ "yuv_to_rgb" ||
          $file_base_name =  "plugin_resize_kernel" ||
          $file_base_name =~ "bert_pre" ]]
    then
      cncc -O3 ${file} -DFLOAT_MODE=${FLOAT_MODE} --target=${TARGET_CHIP_TYPE} --bang-mlu-arch=MLU270 -DBANG_LOG=${BANG_LOG} \
        -o ${BUILD_PLUGINOP_DIR}/${file_base_name}_MLU270.o
      cncc -O3 ${file} -DFLOAT_MODE=${FLOAT_MODE} --target=${TARGET_CHIP_TYPE} --bang-mlu-arch=MLU220 -DBANG_LOG=${BANG_LOG} \
        -o ${BUILD_PLUGINOP_DIR}/${file_base_name}_MLU220.o
    elif [[ $file_base_name =~ "arange" ||
          $file_base_name =~ "init" ||
          $file_base_name =~ "nms" ]]
    then
      cncc -O3 ${file} -DFLOAT_MODE=0 --target=${TARGET_CHIP_TYPE} --bang-mlu-arch=MLU270 -DBANG_LOG=${BANG_LOG} -o \
      ${BUILD_PLUGINOP_DIR}/${file_base_name}_MLU270_half.o
      cncc -O3 ${file} -DFLOAT_MODE=1 --target=${TARGET_CHIP_TYPE} --bang-mlu-arch=MLU270 -DBANG_LOG=${BANG_LOG} -o \
      ${BUILD_PLUGINOP_DIR}/${file_base_name}_MLU220_float.o
    else
      cncc -O3 ${file} --bang-mlu-arch=${TARGET_MLU_ARCH} --target=${TARGET_CHIP_TYPE} -o  \
        ${BUILD_PLUGINOP_DIR}/${file_base_name}.o
    fi
done

cc_files=""
for cc_file in `ls ${PLUGIN_OP_DIR}/*/*.cc`
do
    cc_files+="${cc_file} "
done

for cc_file  in `ls ${PWD}/common/src/*.cc`
do
    cc_files+="${cc_file} "
done

LIB_PATH=$(dirname ${LIB_CNML})
HEAD_PATH=$(dirname ${HEAD_CNML})

aarch64-linux-gnu-g++ ${MAKE_FLAG} -shared -fPIC ${cc_files} -DFLOAT_MODE=${FLOAT_MODE} ${BUILD_PLUGINOP_DIR}/*.o  \
  -L ${LIB_PATH} -I ${HEAD_PATH} -I ./common/include -lcnml -lcnrt --std=c++11 -o ${BUILD_DIR}/libcnplugin.so

if [ ${BUILD_MODE} == "release" ]; then
  aarch64-linux-gnu-strip -s ${BUILD_DIR}/libcnplugin.so
fi

BUILD_PATH="./build"
if [ ! -d "$BUILD_PATH" ]; then
  mkdir "$BUILD_PATH"
fi

pushd ${BUILD_PATH}
  cmake -DVENDOR_NAME="CAMBRICON" \
        -DCROSS_COMPILE="${TOOLCHAIN_PREFIX}" \
        -DCMAKE_C_COMPILER="${CROSS_COMPILE}gcc" \
        -DCMAKE_CXX_COMPILER="${CROSS_COMPILE}g++" \
        -DCMAKE_BUILD_TYPE=${BUILD_MODE} \
        -DBUILD_TEST=${BUILD_TEST} \
        -DLIB_PATH=${LIB_PATH} \
        -DHEAD_PATH=${HEAD_PATH} \
        ..

  make -j
popd

if [ ! -d "input_data" ]; then
	echo "ln -s ../common/input_data build/input_data"
	ln -s ../common/input_data build/input_data
fi
echo "---- build success ----"
