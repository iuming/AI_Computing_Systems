/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/

#include "cnplugin.h"
#include "plugin_init_kernel.h"
#include "plugin_init_cpu.h"

typedef uint16_t half;
#if (FLOAT_MODE == 1)
typedef float DType;
#elif (FLOAT_MODE == 0)     // NOLINT
typedef half DType;
#endif

cnmlStatus_t cnmlCreatePluginInitOpParam(
  cnmlPluginInitOpParam_t *param,
  int size,
  float value,
  int dtype_flag,
  cnmlCoreVersion_t coreVersion
) {
  // CHECK_ENFORCE(param, "param shouldn't be nullptr");
  *param = new cnmlPluginInitOpParam();
  // scalar params
  (*param)->size = size;
  (*param)->value = value;
  (*param)->dtype_flag = dtype_flag;
  (*param)->coreVersion = coreVersion;

  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlDestroyPluginInitOpParam(
  cnmlPluginInitOpParam_t *param
) {
  delete (*param);
  *param = nullptr;

  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlCreatePluginInitOp(
  cnmlBaseOp_t *op,
  cnmlPluginInitOpParam_t param,
  cnmlTensor_t *init_input_tensors,
  cnmlTensor_t *init_output_tensors
) {
  int size = param->size;
  float value = param->value;

  float value_float_;
  half value_half_;
  void** InterfacePtr;
  if (param->dtype_flag == 2) {  // half
    InterfacePtr = reinterpret_cast<void**>(&InitKernel_MLU270_half);
    cnrtConvertFloatToHalf(&value_half_, value);    // NOLINT
  } else if (param->dtype_flag == 4) {  // float
    InterfacePtr = reinterpret_cast<void**>(&InitKernel_MLU270_float);
    value_float_ = value;
  } else {
    std::cout << "MLU Init not support this data type:" << param->dtype_flag;
    return CNML_STATUS_INVALIDARG;
  }
  // Passing param
  cnrtKernelParamsBuffer_t params;
  cnrtGetKernelParamsBuffer(&params);
  cnrtKernelParamsBufferMarkInput(params);   // input 0
  cnrtKernelParamsBufferMarkOutput(params);  // output 0
  if (param->dtype_flag == 2) {
    cnrtKernelParamsBufferAddParam(params, &value_half_, sizeof(half));
  } else {
    cnrtKernelParamsBufferAddParam(params, &value_float_, sizeof(float));
  }
  cnrtKernelParamsBufferAddParam(params, &size, sizeof(int));
  cnmlCreatePluginOp(op,
                     "Init",
                     InterfacePtr,
                     params,
                     init_input_tensors,
                     1,
                     init_output_tensors,
                     1,
                     nullptr,
                     0);
  cnrtDestroyKernelParamsBuffer(params);
  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlComputePluginInitOpForward(
  cnmlBaseOp_t op,
  void **inputs,
  int input_num,
  void **outputs,
  int output_num,
  cnrtQueue_t queue
) {
  cnmlComputePluginOpForward_V4(op,
                                nullptr,
                                inputs,
                                input_num,
                                nullptr,
                                outputs,
                                output_num,
                                queue,
                                nullptr);
  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlCpuComputePluginInitOpForward(
  cnmlPluginInitOpParam_t param,
  float *output
) {
  int size = param->size;
  float value = param->value;
  init_cpu(output, value, size);
  return CNML_STATUS_SUCCESS;
}
