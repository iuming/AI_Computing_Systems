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

extern "C" {
void PluginCropFeatureAndResizeKernel(
    half* src_gdram,
    half* boxes_gdram,
    half* box_index_gdram,
    half* dst_gdram,
    int batchNum,
    int depth,
    int image_height,
    int image_width,
    int crop_height,
    int crop_width,
    int box_number,
    int inputDataType,
    int outputDataType,
    int input2half, // not use in .mlu
    int output2uint,
    int pad_size);
}

cnmlStatus_t cnmlCreatePluginCropFeatureAndResizeOpParam(
  cnmlPluginResizeAndColorCvtParam_t* param,
  int s_row,  // H
  int s_col,  // W
  int d_row,  // resize H
  int d_col,  // resize W
  int batchNum,
  int depth,
  int box_number,
  int pad_size,
  cnmlCoreVersion_t core_version) {
  *param = new cnmlPluginResizeAndColorCvtParam();

  (*param)->s_row = s_row;
  (*param)->s_col = s_col;
  (*param)->d_row = d_row;
  (*param)->d_col = d_col;
  (*param)->batchNum = batchNum;
  (*param)->depth = depth;
  (*param)->box_number = box_number;
  (*param)->pad_size = pad_size;
  (*param)->core_version = core_version;
  (*param)->input2half = 1;
  (*param)->output2uint = 1;
}

cnmlStatus_t cnmlDestroyPluginCropFeatureAndResizeOpParam(
    cnmlPluginResizeAndColorCvtParam_t* param){
  delete (*param);
  *param = nullptr;
  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlCreatePluginCropFeatureAndResizeOp(
    cnmlBaseOp_t* op_ptr,
    cnmlPluginResizeAndColorCvtParam_t* param,
    cnmlTensor_t* input_cnml_tensors,
    cnmlTensor_t* output_cnml_tensors) {
    //cnmlTensor_t rgb_mlu, // src
    //cnmlTensor_t box_mlu,  // boxes
    //cnmlTensor_t box_index_mlu, // box_index
    // cnmlTensor_t new_box_mlu) {  // dst
  int image_height = (*param)->s_row;
  int image_width = (*param)->s_col;
  int crop_height = (*param)->d_row;
  int crop_width = (*param)->d_col;
  int batch = (*param)->batchNum;
  int depth = (*param)->depth;
  int box_number = (*param)->box_number;
  int pad_size = (*param)->pad_size;
  int input2half = (*param)->input2half;
  int output2uint = (*param)->output2uint;

  const int input_num = 3;
  const int output_num = 1;
  int inputDataType, outputDataType;
  inputDataType = 2;
  outputDataType = 2;

  // craeete Params
  cnrtKernelParamsBuffer_t params;
  cnrtGetKernelParamsBuffer(&params);

  cnrtKernelParamsBufferMarkInput(params); // src
  cnrtKernelParamsBufferMarkInput(params);  // boxes
  cnrtKernelParamsBufferMarkInput(params);  // box_index
  cnrtKernelParamsBufferMarkOutput(params);  // dst

  //cnrtKernelParamsBufferAddParam(params, &rgb_mlu, sizeof(half*));
  //cnrtKernelParamsBufferAddParam(params, &box_mlu, sizeof(half*));
  //cnrtKernelParamsBufferAddParam(params, &box_index_mlu, sizeof(half*));
  //cnrtKernelParamsBufferAddParam(params, &new_box_mlu, sizeof(half*));
  cnrtKernelParamsBufferAddParam(params, &batch, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &depth, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &image_height, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &image_width, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &crop_height, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &crop_width, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &box_number, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &inputDataType, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &outputDataType, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &input2half, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &output2uint, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &pad_size, sizeof(int));


  //cnmlTensor_t input_cnml_tensors[input_num];
  //cnmlTensor_t output_cnml_tensors[output_num];

  //input_cnml_tensors[0] = rgb_mlu; // src
  //input_cnml_tensors[1] = box_mlu;
  //input_cnml_tensors[2] = box_index_mlu;
  //output_cnml_tensors[0] = new_box_mlu; // dst

  cnmlStatus_t ret = cnmlCreatePluginOp(
      op_ptr, "PluginCropFeatureAndResize",
      reinterpret_cast<void **>(&PluginCropFeatureAndResizeKernel),
      params,
      input_cnml_tensors, input_num,
      output_cnml_tensors, output_num,
      nullptr, 0);
  cnrtDestroyKernelParamsBuffer(params);
  return ret;
}

cnmlStatus_t cnmlComputePluginCropFeatureAndResizeOpForward(
    cnmlBaseOp_t op,
    void* input_addr[],
    void* output_addr[],
    cnrtInvokeFuncParam_t compute_forw_param,
    cnrtQueue_t queue)
{
  //vector<void *> input_addr = {src_addr, box_addr, box_index_addr};
  //vector<void *> output_addr = {dst_addr};
  const int input_num = 3;
  const int output_num = 1;

  cnmlStatus_t ret = cnmlComputePluginOpForward_V3(
      op,
      input_addr,
      input_num,
      output_addr,
      output_num,
      &compute_forw_param,
      queue);
  return ret;
}

unsigned int offset(int image_height, int image_width, int channel,
                    int batch, int height, int width, int depth) {
  unsigned int off = batch * image_height * image_width * channel +
                     height * image_width * channel +
                     width * channel +
                     depth;

  return off;
}

cnmlStatus_t cnmlCpuComputePluginCropFeatureAndResizeOpForward(
    float* src,
    float* boxes,
    float* box_index,
    float* new_box,
    int batchNum,
    int depth,
    int image_height,
    int image_width,
    int crop_height,
    int crop_width,
    int box_number) {
  for (int i = 0; i < box_number; i++) {
    const float y1 = boxes[i * 4 + 0];
    const float x1 = boxes[i * 4 + 1];
    const float y2 = boxes[i * 4 + 2];
    const float x2 = boxes[i * 4 + 3];
    const int b_in = box_index[i];

    const float height_scale = (crop_height > 1) ?
                       (y2 - y1) * (image_height - 1) / (crop_height - 1) : 0;
    const float width_scale = (crop_width > 1) ?
                       (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;

    for (int y = 0; y < crop_height; ++y) {
      const float in_y = (crop_height > 1)
                             ? y1 * (image_height - 1) + y * height_scale
                             : 0.5 * (y1 + y2) * (image_height - 1);

      const int top_y_index = floorf(in_y);
      const int bottom_y_index = ceilf(in_y);
      const float y_lerp = in_y - top_y_index;

      for (int x = 0; x < crop_width; ++x) {
        const float in_x = (crop_width > 1)
                               ? x1 * (image_width - 1) + x * width_scale
                               : 0.5 * (x1 + x2) * (image_width - 1);

        const int left_x_index = floorf(in_x);
        const int right_x_index = ceilf(in_x);
        const float x_lerp = in_x - left_x_index;

        for (int d = 0; d < depth; ++d) {
          const float top_left = *(src + offset(image_height, image_width, depth,
                                                b_in, top_y_index, left_x_index, d));
          const float top_right = *(src + offset(image_height, image_width, depth,
                                                 b_in, top_y_index, right_x_index, d));
          const float bottom_left = *(src + offset(image_height, image_width, depth,
                                                   b_in, bottom_y_index, left_x_index, d));
          const float bottom_right = *(src + offset(image_height, image_width, depth,
                                                    b_in, bottom_y_index, right_x_index, d));
          const float top = top_left + (top_right - top_left) * x_lerp;
          const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
          *(new_box + offset(crop_height, crop_width, depth,
                     i, y, x, d)) = top + (bottom - top) * y_lerp;
        }
      }
    }
  }
}

