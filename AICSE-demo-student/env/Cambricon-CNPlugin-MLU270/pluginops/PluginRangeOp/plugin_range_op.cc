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
#include "plugin_range_kernel.h"
#include "plugin_range_cpu.h"

cnmlStatus_t cnmlCreatePluginRangeOpParam(
    cnmlPluginRangeOpParam_t *param,
    cnmlCoreVersion_t core_version)
{
  *param = new cnmlPluginRangeOpParam();
  (*param)->core_version = core_version;
  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlDestroyPluginRangeOpParam(
    cnmlPluginRangeOpParam_t *param)
{
  delete (*param);
  *param = nullptr;
  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlCreatePluginRangeOp(
    cnmlBaseOp_t *op_ptr,
    cnmlPluginRangeOpParam_t param,
    cnmlTensor_t *inputs,
    cnmlTensor_t *outputs)
{
  const int input_num = 3;
  const int output_num = 1;

  // create Params
  cnrtKernelParamsBuffer_t params;

	cnrtGetKernelParamsBuffer(&params);
	cnrtKernelParamsBufferMarkInput(params);   // input 0
	cnrtKernelParamsBufferMarkInput(params);   // input 1
	cnrtKernelParamsBufferMarkInput(params);   // input 2
	cnrtKernelParamsBufferMarkOutput(params);  // output 0

	void **InterfacePtr = reinterpret_cast<void **>(&range_kernel);

  cnmlStatus_t ret = cnmlCreatePluginOp(op_ptr, 
		                                   "pluginRange",
                                       InterfacePtr,
                                       params,
                                       inputs, input_num,
                                       outputs, output_num,
                                       nullptr, 0);
  cnrtDestroyKernelParamsBuffer(params);
  return ret;
}

cnmlStatus_t cnmlCpuComputePluginRangeOpForward(
   float start, float limit, float delta, float *output)
{
  range_cpu(start, limit, delta, output);
  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlComputePluginRangeOpForward(
    cnmlBaseOp_t op,
    void *input[],
    int num_inputs,
    void *output[],
    int num_outputs,
    cnrtQueue_t queue)
{
  if (!op)
  {
    printf("op is null!\n");
    assert(0);
  }
  if (!output)
  {
    printf("output mlu pointer is null!\n");
    assert(0);
  }
  const int input_num = 3;
  const int output_num = 1;
  cnmlStatus_t ret = cnmlComputePluginOpForward_V4(op,
		                                               nullptr,
                                                   input,
                                                   num_inputs,
                                                   nullptr,
                                                   output,
                                                   num_outputs,
                                                   queue, nullptr);
  return ret;
}

