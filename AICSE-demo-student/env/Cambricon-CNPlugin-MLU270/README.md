## Introduction

Cambricon PluginOp library. Maintain and test pluginops developed by Cambricon developers.

## Contributors

Developers from following groups in Cambricon:

* AE
* Compile
* SOPA
* NGPF
* Framworks

## Directory

* common/ : Define common variables and functions.
* pluginops/ : Include all pluginop subfolders arranged by op_name.

## How to start

You can prepare folder like this:

-- mlu

   -- include:  cnml.h, cnrt.h

   -- lib64:  libcnml.so, libcnrt.so

   -- bin: cncc, cnas

-- cnplugin


* Steps:

1. Create directory for .so/.h/bin：makir mlu

2. export NEUWARE=/path/to/mlu

3. Build cnplugin:

   1. cd cnplugin
   2. ./build.sh --platform

   You can run "./build.sh -h" to know more.

5. Test plugin op:

   1. cd samplecode
   2. ./compile_test.sh
   3. ./yolov3_detection_test


