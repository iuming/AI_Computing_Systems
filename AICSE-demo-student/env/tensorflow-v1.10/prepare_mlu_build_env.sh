#!/bin/bash

set -o errexit; # Exit once any command failed

function Usage {
  echo "Usage:"
  echo "    Setup pip config for user: prepare_mlu_build_env.sh"
  echo "    Setup pip config for virtualenv: prepare_mlu_build_env.sh --virtualenv"
  exit 1
}

if [[ ! ${TENSORFLOW_HOME} ]];then
  if type git > /dev/null 2>&1 && git rev-parse --is-inside-work-tree > /dev/null 2>&1;then
      TENSORFLOW_HOME=$( git rev-parse --show-toplevel )
  else
    1>&2 echo "ERROR: TENSORFLOW_HOME is not set, please set TENSORFLOW_HOME to tensorflow project root"
    exit 1
  fi
fi

VIRTUALENV=False
while (( $# )); do
  case $1 in
    --virtualenv)
			VIRTUALENV=True
			shift
    ;;

    *)
      Usage
    ;;
  esac
done

#1. create pip configuration
echo ============================Start Setup Pip Config==========================================


if [[ ${VIRTUALENV} == True ]];then
	mkdir -p ${TENSORFLOW_HOME}/virtualenv_mlu
	PIP_CONFIG=${TENSORFLOW_HOME}/virtualenv_mlu/pip.conf
	rm -f ${PIP_CONFIG}
else
	mkdir -p ${HOME}/.pip
	PIP_CONFIG=${HOME}/.pip/pip.conf
	rm -f ${PIP_CONFIG}
fi

echo "[global]" >> ${PIP_CONFIG}
echo "index-url = http://mirrors.cambricon.com/pypi/web/simple" >> ${PIP_CONFIG}
echo "find-links = /opt/shared/tensorflow/tf-python-pkgs" >> ${PIP_CONFIG}
echo "trusted-host = mirrors.cambricon.com" >> ${PIP_CONFIG}
cat ${PIP_CONFIG}
echo "write to ${PIP_CONFIG}"
echo ============================End Setup Pip Config============================================

#2. install bazel cache
echo ============================Start bazel repo cache install========================================
${TENSORFLOW_HOME}/tools/install_bazel_repository_cache.sh
echo ============================End bazel repo cache install==========================================
