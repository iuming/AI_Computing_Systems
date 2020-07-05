/*Copyright 2018 Cambricon*/

#include "tensorflow/stream_executor/mlu/mlu_api/lib_ops/mlu_lib_ops.h"
#include "tensorflow/stream_executor/mlu/mlu_api/ops/mlu_ops.h"
#include "tensorflow/stream_executor/mlu/mlu_api/tf_mlu_intf.h"

namespace stream_executor {
namespace mlu {
namespace ops {


Status MLUBatchMatMulV2::CreateMLUOp(std::vector<MLUTensor *> &inputs,
                                     std::vector<MLUTensor *> &outputs,
                                     void *param) {
  TF_PARAMS_CHECK(inputs.size() > 1, "Missing input");
  TF_PARAMS_CHECK(outputs.size() > 0, "Missing output");

  MLUTensor *in0 = inputs.at(0);
  MLUTensor *in1 = inputs.at(1);
  MLUTensor *output = outputs.at(0);

  MLULOG(3) << "CreateBatchMatMulV2Op, input1: "
            << lib::MLUTensorUtil(in0).DebugString()
            << ", input2: " << lib::MLUTensorUtil(in1).DebugString()
            << ", output: " << lib::MLUTensorUtil(output).DebugString();

  MLUBatchMatMulV2OpParam *op_param = static_cast<MLUBatchMatMulV2OpParam *>(param);

  float scale_0 = op_param->scale_0_;
  int pos_0 = op_param->pos_0_;
  float scale_1 = op_param->scale_1_;
  int pos_1 = op_param->pos_1_;
  int dim_0 = op_param->dim_0_;
  int dim_1 = op_param->dim_1_;
  int m = op_param->m_;
  int n = op_param->n_;
  int k = op_param->k_;
  cnmlCoreVersion_t core_version = CNML_MLU270;

  cnmlPluginBatchMatMulV2OpParam_t bm_param;
  TF_CNML_CHECK(cnmlCreatePluginBatchMatMulV2OpParam(&bm_param, scale_0, pos_0,
        scale_1, pos_1, dim_0, dim_1, m, n, k, core_version));

  MLUBaseOp *batch_matmul_op_ptr = nullptr;

  TF_STATUS_CHECK(lib::CreateBatchMatMulV2Op(&batch_matmul_op_ptr, bm_param,
                                             in0, in1, output));

  base_ops_.push_back(batch_matmul_op_ptr);

  return Status::OK();
}

Status MLUBatchMatMulV2::Compute(const std::vector<void *> &inputs,
                               const std::vector<void *> &outputs,
                               cnrtQueue_t queue) {
  MLULOG(3) << "ComputeMLUBatchMatMulV2";

  int input_num = inputs.size();
  int output_num = outputs.size();

  void* real_inputs[input_num];
  void* real_outputs[output_num];

  for (int i = 0; i < input_num; ++i) {
    real_inputs[i] = inputs.at(i);
  }

  for (int i = 0; i < output_num; ++i) {
    real_outputs[i] = outputs.at(i);
  }

  TF_STATUS_CHECK(lib::ComputeBatchMatMulV2Op(base_ops_.at(0), queue,
        real_inputs, input_num, real_outputs, output_num));

  TF_CNRT_CHECK(cnrtSyncQueue(queue));

  return Status::OK();
}

}  // namespace ops
}  // namespace mlu
}  // namespace stream_executor
