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
#include "plugin_bert_squad_kernel.h"

cnmlStatus_t cnmlCreatePluginBertSquadOp(
    cnmlBaseOp_t *op,
    cnmlTensor_t *input_tensors,
    cnmlTensor_t *output_tensors,
    cnmlTensor_t *cnml_static_tensors,
    int static_tensors_num,
    int batch_num,
    int seq_len)
{
  // prepare op
  if (static_tensors_num != 42) {
    std::cout << "bert_squad op expected to have 42 const tensors, but got " << static_tensors_num;
    return CNML_STATUS_LENGTHERR;
  }

  // prepare bangC-kernel param
  cnrtKernelParamsBuffer_t params;
  cnrtGetKernelParamsBuffer(&params);
  cnrtKernelParamsBufferMarkOutput(params);  // output 0 : start_logits
  cnrtKernelParamsBufferMarkOutput(params);  // output 1 : end_logits
  cnrtKernelParamsBufferMarkInput(params);   // input 0 : input_ids 
  cnrtKernelParamsBufferMarkInput(params);   // input 1 : token_ids
  cnrtKernelParamsBufferMarkInput(params);   // input 2 : attention_mask
  for (int i = 0; i < static_tensors_num; ++i) {
    cnrtKernelParamsBufferMarkStatic(params); 
  }
  cnrtKernelParamsBufferAddParam(params, &batch_num, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &seq_len, sizeof(int));

  //create Plugin op
  void **InterfacePtr = reinterpret_cast<void**>(&bertSquadKernel);
  cnmlCreatePluginOp(op,
                     "BertSquadKernel",
                     InterfacePtr,
                     params,
                     input_tensors,
                     3,
                     output_tensors,
                     2,
                     cnml_static_tensors,
                     static_tensors_num);

  cnrtDestroyKernelParamsBuffer(params);
  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlComputePluginBertSquadOpForward(
    cnmlBaseOp_t op,
    cnmlTensor_t* input_tensors, // default as nullptr
    void** inputs,
    cnmlTensor_t* output_tensors, // default as nullptr
    void** outputs,
    cnrtQueue_t queue,
    void *extra)
{
  cnmlComputePluginOpForward_V4(op,
                                input_tensors,
                                inputs,
                                3,
                                output_tensors,
                                outputs,
                                2,
                                queue,
                                extra);
  return CNML_STATUS_SUCCESS;
}
