#!/bin/bash

cd $( dirname ${BASH_SOURCE} )
export TENSORFLOW_HOME=$(pwd)
cd - > /dev/null

if [[ ! ${LD_LIBRARY_PATH} =~ ${TENSORFLOW_HOME}/third_party/mlu/lib ]];then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${TENSORFLOW_HOME}/third_party/mlu/lib
fi

## add development environment variables
#if [[ -f ${TENSORFLOW_HOME}/tools/env_dev.sh ]];then
#  source ${TENSORFLOW_HOME}/tools/env_dev.sh
#fi
