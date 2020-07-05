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
#include "plugin_onehot_cpu.h"

cnmlStatus_t cnmlCreatePluginOneHotOpParam(
    cnmlPluginOneHotOpParam_t *param,
    cnmlCoreVersion_t core_version,
    int N,
    int H,
    int W,
    int C,
    int depth,
    float onvalue,
    float offvalue,
	int axis)
{
  *param = new cnmlPluginOneHotOpParam();
  (*param)->N = N;
  (*param)->H = H;
  (*param)->W = W;
  (*param)->C = C;
  (*param)->depth = depth;
  (*param)->onvalue = onvalue;
  (*param)->offvalue = offvalue;
  (*param)->axis = axis;
  (*param)->core_version = core_version;
  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlDestroyPluginOneHotOpParam(
    cnmlPluginOneHotOpParam_t *param
)
{
  delete (*param);
  *param = nullptr;
  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlCreatePluginOneHotOp(
    cnmlBaseOp_t *op_ptr,
    cnmlPluginOneHotOpParam_t param,
    cnmlTensor_t *input,
    cnmlTensor_t *output)
{
  int N = param->N;
  int H = param->H;
  int W = param->W;
  int C = param->C;
  int depth = param->depth;
  float onvalue = param->onvalue;
  float offvalue = param->offvalue;
  int axis = param->axis;

  const int input_num = 1;
  const int output_num = 1;

  // create Params
  cnrtKernelParamsBuffer_t params;

	cnrtGetKernelParamsBuffer(&params);
	cnrtKernelParamsBufferMarkInput(params);   // input 0
	cnrtKernelParamsBufferMarkOutput(params);  // output 0
	cnrtKernelParamsBufferAddParam(params, &N, sizeof(int));
	cnrtKernelParamsBufferAddParam(params, &H, sizeof(int));
	cnrtKernelParamsBufferAddParam(params, &W, sizeof(int));
	cnrtKernelParamsBufferAddParam(params, &C, sizeof(int));
	cnrtKernelParamsBufferAddParam(params, &depth, sizeof(int));
	cnrtKernelParamsBufferAddParam(params, &onvalue, sizeof(float));
	cnrtKernelParamsBufferAddParam(params, &offvalue, sizeof(float));
	cnrtKernelParamsBufferAddParam(params, &axis, sizeof(int));

  cnmlStatus_t ret = cnmlCreatePluginOp(op_ptr, "pluginOneHot",
                                          reinterpret_cast<void **>(onehot_kernel),
                                          params,
                                          input, input_num,
                                          output, output_num,
                                          nullptr, 0);
  cnrtDestroyKernelParamsBuffer(params);
  return ret;
}

cnmlStatus_t cnmlCpuComputePluginOneHotOpForward(
   cnmlPluginOneHotOpParam_t param,
   int* indices, float* dst)
{
  onehot_cpu(param, indices, dst);
  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlComputePluginOneHotOpForward(
    cnmlBaseOp_t op,
    void *input_addrs[],
    int num_inputs,
    void *output_addrs[],
    int num_outputs,
    cnrtQueue_t queue)
{
  if (!op)
  {
    printf("op is null!\n");
    assert(0);
  }
  if (!output_addrs)
  {
    printf("output mlu pointer is null!\n");
    assert(0);
  }
	
  cnmlStatus_t ret = cnmlComputePluginOpForward_V4(op,
                                                   nullptr,
                                                   input_addrs,
                                                   num_inputs,
                                                   nullptr,
                                                   output_addrs,
                                                   num_outputs,
                                                   queue, nullptr);
  return ret;
}

