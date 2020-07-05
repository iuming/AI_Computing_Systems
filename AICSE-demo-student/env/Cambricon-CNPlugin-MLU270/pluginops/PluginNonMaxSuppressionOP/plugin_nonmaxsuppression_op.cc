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
#include "plugin_nonmaxsuppression_kernel.h"

cnmlStatus_t cnmlCreatePluginNonMaxSuppressionOpParam(
    cnmlPluginNonMaxSuppressionOpParam_t *param,
    int len,
    int max_num,
    float iou_threshold,
    float score_threshold,
    cnmlCoreVersion_t core_version)
{

  *param = new cnmlPluginNonMaxSuppressionOpParam();

  int static_num = 2;
  (*param)->cnml_static_tensors
    = (cnmlTensor_t *)malloc(sizeof(cnmlTensor_t) * static_num);
  (*param)->cpu_static_init
    = (void **)malloc(sizeof(void*) * static_num);

  int static_tensor0_shape[4] = {16, len, 1, 1};
  cnmlCreateTensor_V2(
      &(*param)->cnml_static_tensors[0],
      CNML_CONST);
  cnmlSetTensorShape(
      (*param)->cnml_static_tensors[0],
      4,
      static_tensor0_shape);
  cnmlSetTensorDataType(
      (*param)->cnml_static_tensors[0],
      CNML_DATA_FLOAT16);

  (*param)->cpu_static_init[0] = malloc(sizeof(half) * 16 * len);
  void* cpu_static0_init = (*param)->cpu_static_init[0];
  memset(cpu_static0_init, 0, 16 * len * sizeof(half));
  for (int i = 0; i < len; i++) {
    ((int*)cpu_static0_init)[i] = i;
  }     
  cnmlBindConstData_V2((*param)->cnml_static_tensors[0], (void*)(*param)->cpu_static_init[0], false);

  int static_tensor1_shape[4] = {1, 16, 1, 1};
  cnmlCreateTensor_V2(
      &(*param)->cnml_static_tensors[1],
      CNML_CONST);
  cnmlSetTensorShape(
      (*param)->cnml_static_tensors[1],
      4,
      static_tensor1_shape);
  cnmlSetTensorDataType(
      (*param)->cnml_static_tensors[1],
      CNML_DATA_FLOAT16);

  (*param)->cpu_static_init[1] = malloc(sizeof(half));
  memset((*param)->cpu_static_init[1], 0,  sizeof(half));
  cnmlBindConstData_V2((*param)->cnml_static_tensors[1], (void*)(*param)->cpu_static_init[1], false);

  (*param)->len = len;
  (*param)->max_num = max_num;
  (*param)->iou_threshold = iou_threshold;
  (*param)->score_threshold = score_threshold;

  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlDestroyPluginNonMaxSuppressionOpParam(
    cnmlPluginNonMaxSuppressionOpParam_t *param,
    int static_num)
{
  for (int i = 0; i < static_num; ++i) {
    cnmlDestroyTensor(&(*param)->cnml_static_tensors[i]);
  }
  free((*param)->cnml_static_tensors);
  free((*param)->cpu_static_init);
  delete (*param);
  *param = nullptr;

  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlCreatePluginNonMaxSuppressionOp(
    cnmlBaseOp_t *op,
    cnmlPluginNonMaxSuppressionOpParam_t param,
    cnmlTensor_t *nms_input_tensors,
    int input_num,
    cnmlTensor_t *nms_output_tensors,
    int output_num,
    int static_num)
{
  // read OpParam
  int len = param->len;
  int max_num = param->max_num;
  float iou_threshold = param->iou_threshold;
  float score_threshold = param->score_threshold;

  // convert from float to half
  half iou_threshold_half;
  half score_threshold_half;

  cnrtConvertFloatToHalf(&iou_threshold_half, iou_threshold);
  cnrtConvertFloatToHalf(&score_threshold_half, score_threshold);

  // prepare op
  cnmlTensor_t *cnml_static_tensors = param->cnml_static_tensors;
  void* *cpu_static_init = param->cpu_static_init;

  // prepare bangC-kernel param
  bool is_int8 = true;
  cnrtKernelParamsBuffer_t params;
  cnrtGetKernelParamsBuffer(&params);
  cnrtKernelParamsBufferMarkOutput(params);  // output 0
  cnrtKernelParamsBufferMarkInput(params);   // input 0
  cnrtKernelParamsBufferMarkInput(params);   // input 1
  cnrtKernelParamsBufferMarkStatic(params);  // temp gdram ptr for rewrite socre input 
  cnrtKernelParamsBufferMarkStatic(params);  // temp gdram ptr for core communication
  cnrtKernelParamsBufferAddParam(params, &len, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &max_num, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &iou_threshold_half, sizeof(half));
  cnrtKernelParamsBufferAddParam(params, &score_threshold_half, sizeof(half));
  cnrtKernelParamsBufferAddParam(params, &is_int8, sizeof(bool));

  //create Plugin op
  void **InterfacePtr = reinterpret_cast<void**>(&NonMaxSuppressionKernel);
  cnmlCreatePluginOp(op,
                     "NonMaxSuppressionKernel",
                     InterfacePtr,
                     params,
                     nms_input_tensors,
                     input_num,
                     nms_output_tensors,
                     output_num,
                     cnml_static_tensors,
                     static_num);

  cnrtDestroyKernelParamsBuffer(params);
  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlComputePluginNonMaxSuppressionOpForward(
    cnmlBaseOp_t op,
    cnmlTensor_t input_tensors[],
    void *inputs[],
    int num_inputs,
    cnmlTensor_t output_tensors[],
    void *outputs[],
    int num_outputs,
    cnrtQueue_t queue,
    void *extra)
{
  cnmlComputePluginOpForward_V4(op,
                                input_tensors,
                                inputs,
                                num_inputs,
                                output_tensors,
                                outputs,
                                num_outputs,
                                queue,
                                extra);

  return CNML_STATUS_SUCCESS;
}
