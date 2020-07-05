#!/bin/bash

# will do following testing before git push:
#  1. Tensorflow build test
#  2. Gtest build test
#  3. Gtest
#  4. Label image model model build test
#  5. Label image model model test

set -o errexit; # Exit once any command failed

function Usage {
  echo "Usage:"
  echo "    push_branch_to_remote.sh origin branch_name [--force-with-lease]"
  exit 1
}

# Parse parameters
if [[ $# -lt 1 ]];then
  Usage
fi

if [[ ! ${TENSORFLOW_HOME} ]];then
  if type git > /dev/null 2>&1 && git rev-parse --is-inside-work-tree > /dev/null 2>&1;then
    export TENSORFLOW_HOME=$( git rev-parse --show-toplevel )
  else
    1>&2 echo "ERROR: TENSORFLOW_HOME is not set, please set TENSORFLOW_HOME to tensorflow project root"
    exit 1
  fi
fi
source ${TENSORFLOW_HOME}/env.sh

# add available MLU test machines here
MLU270_TEST_MACHINES=($(shuf -e 10.101.114.23 10.101.114.29))
TEST_MACHINE=${MLU270_TEST_MACHINES[0]}

# skip test if current revision already pass the test
REVISION=$( git rev-parse HEAD )
if [[ -f ${TENSORFLOW_HOME}/.pass ]];then
  PASS_REVISIONS=$( cat ${TENSORFLOW_HOME}/.pass )
  for PASS_REVISION in ${PASS_REVISIONS};
  do
    if [ ${REVISION} = ${PASS_REVISION} ];then
      # revision already pass tests, just push it to remote
      git push $@
      exit 0
    fi
  done
fi

# SSH authorization, since HOME is shared, just copy the shared pub key
SSH_PUB_KEY=$HOME/.ssh/id_rsa.pub
if [[ ! -f ${SSH_PUB_KEY} ]];then
  # generate SSH keys
  ssh-keygen -b 2048 -t rsa -f ${SSH_PUB_KEY} -q -N ""
fi

cp -f ${SSH_PUB_KEY} $HOME/.ssh/authorized_keys

# revision not pass the test yet, need to run below tests
# Comile test
bash ${TENSORFLOW_HOME}/build_tensorflow-v1.10_mlu.sh

# Compile gtests
bash ${TENSORFLOW_HOME}/tests/gtest.sh --build-release --output ${TENSORFLOW_HOME}/bin

# Run gtests
bash ${TENSORFLOW_HOME}/tests/gtest.sh --run ${TEST_MACHINE} --path ${TENSORFLOW_HOME}/bin

# Compile label image model test
bash ${TENSORFLOW_HOME}/tests/online_model_test.sh --build-release --output ${TENSORFLOW_HOME}/bin

# Run label image model test
bash ${TENSORFLOW_HOME}/tests/online_model_test.sh --run ${TEST_MACHINE} --path ${TENSORFLOW_HOME}/bin

# Compile label image model test
bash ${TENSORFLOW_HOME}/tests/offline_model_test.sh --build-release --output ${TENSORFLOW_HOME}/bin

# Run label image model test
bash ${TENSORFLOW_HOME}/tests/offline_model_test.sh --run ${TEST_MACHINE} --path ${TENSORFLOW_HOME}/bin

# Run tensorflow_models test
bash ${TENSORFLOW_HOME}/tests/tensorflow_model_test.sh --run ${TEST_MACHINE}

# test pass. record it
echo ${REVISION} >> ${TENSORFLOW_HOME}/.pass

# git push if all tests passed
git push $@

exit 0
