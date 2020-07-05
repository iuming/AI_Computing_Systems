#include "cnplugin.h"

int main()
{
  cnmlInit(0);
  unsigned dev_num;
  cnrtGetDeviceCount(&dev_num);
  if (dev_num == 0)
    return CNRT_RET_ERR_NODEV;
  cnrtDev_t dev;
  cnrtGetDeviceHandle(&dev, 0);
  cnrtSetCurrentDevice(dev);

  // prepare and run test
  const int num_batches = 8;
  const int num_classes = 80;
  const int num_anchors = 3;
  const int num_inputs = 3;
  const int num_outputs = 2;
  int num_boxes = 1024 * 2;
  const int mask_size = num_anchors * num_inputs;
  std::vector<int> bias_shape = {1, 2 * mask_size, 1, 1};
  int c_arr_data[3] = {255, 255, 255};
                int im_w = 1 * 416;
                int im_h = 1 * 416;
      int w_arr_data[3] = {1 * 13,
                           1 * 26,
                           1 * 52};
      int h_arr_data[3] = {1 * 13,
                           1 * 26,
                           1 * 52};
  float bias_arr_data[] = {1 * 116,
                           1 * 90,
                           1 * 156,
                           1 * 198,
                           1 * 373,
                           1 * 326,
                           1 * 30,
                           1 * 61,
                           1 * 62,
                           1 * 45,
                           1 * 59,
                           1 * 119,
                           1 * 10,
                           1 * 13,
                           1 * 16,
                           1 * 30,
                           1 * 33,
                           1 * 23};

  float confidence_thresh = 0.5;
  float nms_thresh = 0.45;
  cnmlCoreVersion_t core_version = CNML_MLU270;
  cnmlPluginYolov3DetectionOutputOpParam_t param;
  cnmlCreatePluginYolov3DetectionOutputOpParam(
      &param,
      num_batches,
      num_inputs,
      num_classes,
      num_anchors,
      num_boxes,
      im_w,
      im_h,
      confidence_thresh,
      nms_thresh,
      core_version,
      w_arr_data,
      h_arr_data,
      bias_arr_data);

  int dp = 1;
  std::vector<int> output_shape(4, 1);
  output_shape[0] = num_batches;
  output_shape[1] = 7 * num_boxes + 64;

  std::vector<int> input0_shape(4, 1);
  std::vector<int> input1_shape(4, 1);
  std::vector<int> input2_shape(4, 1);
  input0_shape[0] = num_batches;
  input0_shape[1] = c_arr_data[0];
  input0_shape[2] = h_arr_data[0];
  input0_shape[3] = w_arr_data[0];

  input1_shape[0] = num_batches;
  input1_shape[1] = c_arr_data[1];
  input1_shape[2] = h_arr_data[1];
  input1_shape[3] = w_arr_data[1];

  input2_shape[0] = num_batches;
  input2_shape[1] = c_arr_data[2];
  input2_shape[2] = h_arr_data[2];
  input2_shape[3] = w_arr_data[2];

  // buffer_blob
  int buffer_size = 255 * (h_arr_data[0] * w_arr_data[0] +
                           h_arr_data[1] * w_arr_data[1] +
                           h_arr_data[2] * w_arr_data[2]);
  std::vector<int> buffer_shape = {num_batches, buffer_size, 1, 1};

  cnmlTensor_t *cnml_input_tensor
    = (cnmlTensor_t *)malloc(sizeof(cnmlTensor_t) * 3);
  cnmlTensor_t *cnml_output_tensor
    = (cnmlTensor_t *)malloc(sizeof(cnmlTensor_t) * 2);

  cnmlCreateTensor(
      &cnml_input_tensor[0],
      CNML_TENSOR,
      CNML_DATA_FLOAT16,
      input0_shape[0],
      input0_shape[1],
      input0_shape[2],
      input0_shape[3]);

  cnmlCreateTensor(
      &cnml_input_tensor[1],
      CNML_TENSOR,
      CNML_DATA_FLOAT16,
      input1_shape[0],
      input1_shape[1],
      input1_shape[2],
      input1_shape[3]);

  cnmlCreateTensor(
      &cnml_input_tensor[2],
      CNML_TENSOR,
      CNML_DATA_FLOAT16,
      input2_shape[0],
      input2_shape[1],
      input2_shape[2],
      input2_shape[3]);

  cnmlCreateTensor(
      &cnml_output_tensor[0],
      CNML_TENSOR,
      CNML_DATA_FLOAT16,
      output_shape[0],
      output_shape[1],
      output_shape[2],
      output_shape[3]);

  cnmlCreateTensor(
      &cnml_output_tensor[1],
      CNML_TENSOR,
      CNML_DATA_FLOAT16,
      buffer_shape[0],
      buffer_shape[1],
      buffer_shape[2],
      buffer_shape[3]);

  cnmlCpuTensor_t *cpu_input_tensor
    = (cnmlCpuTensor_t *)malloc(sizeof(cnmlCpuTensor_t) * 3);
  cnmlCreateCpuTensor(
      &cpu_input_tensor[0],
      CNML_TENSOR,
      CNML_DATA_FLOAT32,
      CNML_NHWC,
      input0_shape[0],
      input0_shape[1],
      input0_shape[2],
      input0_shape[3]);

  cnmlCreateCpuTensor(
      &cpu_input_tensor[1],
      CNML_TENSOR,
      CNML_DATA_FLOAT32,
      CNML_NHWC,
      input1_shape[0],
      input1_shape[1],
      input1_shape[2],
      input1_shape[3]);

  cnmlCreateCpuTensor(
      &cpu_input_tensor[2],
      CNML_TENSOR,
      CNML_DATA_FLOAT32,
      CNML_NHWC,
      input2_shape[0],
      input2_shape[1],
      input2_shape[2],
      input2_shape[3]);

  cnmlCpuTensor_t cpu_output_ptr;
  cnmlCreateCpuTensor(
      &cpu_output_ptr,
      CNML_TENSOR,
      CNML_DATA_FLOAT32,
      CNML_NCHW,
      output_shape[0],
      output_shape[1],
      output_shape[2],
      output_shape[3]);

  cnmlBaseOp_t op;
  cnmlCreatePluginYolov3DetectionOutputOp(
      &op,
      param,
      cnml_input_tensor,
      cnml_output_tensor);

  // set op layout
  cnmlSetOperationComputingLayout(op, CNML_NHWC);

  // compile op
  cnmlCompileBaseOp(op, CNML_MLU270, 4);

  // load input data, there are 3 inputs.
  std::ifstream inFile0("scale1.txt");
  std::ifstream inFile1("scale2.txt");
  std::ifstream inFile2("scale3.txt");
  int inCount[num_inputs] = {1};
  inCount[0] = input0_shape[0]*input0_shape[1]*input0_shape[2]*input0_shape[3];
  inCount[1] = input1_shape[0]*input1_shape[1]*input1_shape[2]*input1_shape[3];
  inCount[2] = input2_shape[0]*input2_shape[1]*input2_shape[2]*input2_shape[3];
  std::cout << inCount[0] << std::endl;
  std::cout << inCount[1] << std::endl;
  std::cout << inCount[2] << std::endl;
  int outCount[num_outputs] = {1};
  outCount[0] = output_shape[0]*output_shape[1]*output_shape[2]*output_shape[3];
  outCount[1] = buffer_shape[0]*buffer_shape[1]*buffer_shape[2]*buffer_shape[3];

  float *input0 = (float *)malloc(inCount[0] * sizeof(float));
  float *input1 = (float *)malloc(inCount[1] * sizeof(float));
  float *input2 = (float *)malloc(inCount[2] * sizeof(float));
  float *trans_input0 = (float *)malloc(inCount[0] * sizeof(float));
  float *trans_input1 = (float *)malloc(inCount[1] * sizeof(float));
  float *trans_input2 = (float *)malloc(inCount[2] * sizeof(float));
  float *predicts_cpu = (float *)malloc(outCount[0] * sizeof(float));
  float *predicts_mlu = (float *)malloc(outCount[0] * sizeof(float));
  float *cast_input0 = (float *)malloc(inCount[0] * sizeof(int16_t));
  float *cast_input1 = (float *)malloc(inCount[1] * sizeof(int16_t));
  float *cast_input2 = (float *)malloc(inCount[2] * sizeof(int16_t));
  float *cast_predicts_mlu = (float *)malloc(outCount[0] * sizeof(int16_t));

  double data = 0.0;
  int count = 0;
  while (!inFile0.eof() && count < inCount[0] / num_batches) {
    inFile0 >> data;
    for (int batchIdx = 0; batchIdx < num_batches; batchIdx++) {
      int offset = batchIdx * input0_shape[1] * input0_shape[2] * input0_shape[3] + count;
      input0[offset] = (float)data;
    }
    count++;
  }
  inFile0.close();
  for (int batchIdx = 0; batchIdx < input0_shape[0]; batchIdx++) {
    for (int h = 0; h < input0_shape[2] * input0_shape[3]; h++) {
      for (int w = 0; w < input0_shape[1]; w++) {
        int trans_offset = w * input0_shape[2] * input0_shape[3] + h
                         + batchIdx * input0_shape[2]
                         * input0_shape[3] * input0_shape[1];
        int orig_offset  = h * input0_shape[1] + w
                         + batchIdx * input0_shape[2]
                         * input0_shape[3] * input0_shape[1];
        trans_input0[trans_offset]
          = input0[orig_offset];
      }
    }
  }
  count = 0;
  while (!inFile1.eof() && count < inCount[1] / num_batches) {
    inFile1 >> data;
    for (int batchIdx = 0; batchIdx < num_batches; batchIdx++) {
      input1[batchIdx * input1_shape[1] * input1_shape[2] * input1_shape[3] + count] = (float)data;
    }
    count++;
  }
  inFile1.close();
  for (int batchIdx = 0; batchIdx < input1_shape[0]; batchIdx++) {
    for (int h = 0; h < input1_shape[2] * input1_shape[3]; h++) {
      for (int w = 0; w < input1_shape[1]; w++) {
        int trans_offset = w * input1_shape[2] * input1_shape[3] + h
                         + batchIdx * input1_shape[2]
                         * input1_shape[3] * input1_shape[1];
        int orig_offset  = h * input1_shape[1] + w
                         + batchIdx * input1_shape[2]
                         * input1_shape[3] * input1_shape[1];
        trans_input1[trans_offset]
          = input1[orig_offset];
      }
    }
  }
  count = 0;
  while (!inFile2.eof() && count < inCount[2] / num_batches) {
    inFile2 >> data;
    for (int batchIdx = 0; batchIdx < num_batches; batchIdx++) {
      input2[batchIdx * input2_shape[1] * input2_shape[2] * input2_shape[3] + count] = (float)data;
    }
    count++;
  }
  inFile2.close();
  for (int batchIdx = 0; batchIdx < input2_shape[0]; batchIdx++) {
    for (int h = 0; h < input2_shape[2] * input2_shape[3]; h++) {
      for (int w = 0; w < input2_shape[1]; w++) {
        int trans_offset = w * input2_shape[2] * input2_shape[3] + h
                         + batchIdx * input2_shape[2]
                         * input2_shape[3] * input2_shape[1];
        int orig_offset  = h * input2_shape[1] + w
                         + batchIdx * input2_shape[2]
                         * input2_shape[3] * input2_shape[1];
        trans_input2[trans_offset]
          = input2[orig_offset];
      }
    }
  }

  // cast data for fp16 datatype
  cnrtCastDataType(input0, CNRT_FLOAT32, cast_input0, CNRT_FLOAT16, inCount[0], nullptr);
  cnrtCastDataType(input1, CNRT_FLOAT32, cast_input1, CNRT_FLOAT16, inCount[1], nullptr);
  cnrtCastDataType(input2, CNRT_FLOAT32, cast_input2, CNRT_FLOAT16, inCount[2], nullptr);

  // malloc and memcpy from host to device
  void **cpu_addrs = (void **)malloc(sizeof(void *) * num_inputs);
  void **cpu_org_addrs = (void **)malloc(sizeof(void *) * num_inputs);
  void **input_addrs = (void **)malloc(sizeof(void *) * num_inputs);
  void **output_addrs = (void **)malloc(sizeof(void *) * num_outputs);
  cpu_addrs[0] = (void *)cast_input0;
  cpu_addrs[1] = (void *)cast_input1;
  cpu_addrs[2] = (void *)cast_input2;
  cpu_org_addrs[0] = (void *)trans_input0;
  cpu_org_addrs[1] = (void *)trans_input1;
  cpu_org_addrs[2] = (void *)trans_input2;

  for (int i = 0; i < num_inputs; i++) {
    cnrtMalloc(&(input_addrs[i]), inCount[i] * sizeof(int16_t));
    cnrtMemcpy(input_addrs[i], cpu_addrs[i], (inCount[i] * sizeof(int16_t)), CNRT_MEM_TRANS_DIR_HOST2DEV);
  }
  for (int i = 0; i < num_outputs; i++) {
    cnrtMalloc(&(output_addrs[i]), outCount[i] * sizeof(int16_t));
  }

  // forward cpu
  cnmlCpuComputePluginYolov3DetectionOutputOpForward(
    param,
    cpu_org_addrs,
    (void *)predicts_cpu);
  // forward mlu
  cnrtQueue_t queue;
  cnrtCreateQueue(&queue);
  cnrtInvokeFuncParam_t compute_forw_param;
  u32_t affinity = 0x01;
  compute_forw_param.data_parallelism = &dp;
  compute_forw_param.affinity = &affinity;
  compute_forw_param.end = CNRT_PARAM_END;

  cnmlComputePluginYolov3DetectionOutputOpForward(
    op,
    input_addrs,
    num_inputs,
    output_addrs,
    num_outputs,
    &compute_forw_param,
    queue);
  cnrtSyncQueue(queue);
  cnrtDestroyQueue(queue);

  // memcpy from device to host
  cnrtMemcpy(cast_predicts_mlu, output_addrs[0], (outCount[0] * sizeof(int16_t)), CNRT_MEM_TRANS_DIR_DEV2HOST);
  cnrtCastDataType(cast_predicts_mlu, CNRT_FLOAT16, predicts_mlu, CNRT_FLOAT32, outCount[0], nullptr);

  std::cout << std::endl;
  std::cout << "==================== MLU TEST ===================="
            << std::endl;

  int result_boxes = 0;
  int result_status = 0;
  for (int batchIdx = 0; batchIdx < num_batches; batchIdx++) {
    result_boxes = (int)predicts_mlu[batchIdx * (64 + num_boxes * 7)];
    for (int i = 0; i < result_boxes; i++) {
      for (int j = 0; j < 7; j++) {
        std::cout << (float)predicts_mlu[i * 7 + j + 64 + batchIdx * (64 + num_boxes * 7)] << " ";
      }
      std::cout << std::endl;
      std::cout << std::endl;
    }
    std::cout << "========= Num of valid box from mlu for batch: "
              << batchIdx << " is "
              << result_boxes << " ========="
              << std::endl;

    // compare result
    if (result_boxes != (int)predicts_cpu[batchIdx * (64 + 7 * num_boxes)]) {
      result_status = -1;
    }
  }

  if (result_status < 0) {
    printf("FAILED!\n");
  } else {
    printf("PASSED!\n");
  }

  // free resources
  for (int i = 0; i < num_inputs; i++) {
    cnmlDestroyTensor(&cnml_input_tensor[i]);
    cnmlDestroyCpuTensor(&cpu_input_tensor[i]);
    cnrtFree(input_addrs[i]);
  }
  for (int i = 0; i < num_outputs; i++) {
    cnmlDestroyTensor(&cnml_output_tensor[i]);
    cnrtFree(output_addrs[i]);
  }
  cnmlDestroyCpuTensor(&cpu_output_ptr);
  cnmlDestroyPluginYolov3DetectionOutputOpParam(&param);
  cnmlDestroyBaseOp(&op);

  free(cpu_addrs);
  free(cpu_org_addrs);
  free(input_addrs);
  free(output_addrs);
  free(input0);
  free(input1);
  free(input2);
  free(trans_input0);
  free(trans_input1);
  free(trans_input2);
  free(predicts_mlu);
  free(predicts_cpu);
  free(cast_input0);
  free(cast_input1);
  free(cast_input2);
  free(cast_predicts_mlu);
  free(cnml_input_tensor);
  free(cnml_output_tensor);
  free(cpu_input_tensor);

  cnmlExit();
  return result_status;
}
