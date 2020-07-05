/*************************************************************************
 * Copyright (C) [2018] by Cambricon, Inc.
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
#ifndef BERT_H
#define BERT_H

#ifdef __cplusplus
extern "C" {
#endif
void bertSquadKernel(float* start_logits,
                     float* end_logits,
                     int* input_ids,
                     int* token_ids,
                     uint16_t* attention_mask,
                     float* word_embedding_table,
                     float* segment_embedding_table,
                     float* position_embedding_table,
                     float* embedding_layernorm_beta,
                     float* embedding_layernorm_gamma,
                     short int* post_output_kernel,
                     float* post_output_bias,
                     short int* attr_kernel_Q_ch0,
                     short int* attr_kernel_Q_ch1,
                     short int* attr_kernel_Q_ch2,
                     short int* attr_kernel_Q_ch3,
                     uint16_t* attr_bias_Q,
                     short int* attr_kernel_K_ch0,
                     short int* attr_kernel_K_ch1,
                     short int* attr_kernel_K_ch2,
                     short int* attr_kernel_K_ch3,
                     uint16_t* attr_bias_K,
                     short int* attr_kernel_V_ch0,
                     short int* attr_kernel_V_ch1,
                     short int* attr_kernel_V_ch2,
                     short int* attr_kernel_V_ch3,
                     uint16_t* attr_bias_V,
                     short int* attr_output_kernel_ch0,
                     short int* attr_output_kernel_ch1,
                     short int* attr_output_kernel_ch2,
                     short int* attr_output_kernel_ch3,
                     uint16_t* attr_output_bias,
                     uint16_t* attr_layernorm_beta,
                     uint16_t* attr_layernorm_gamma,
                     short int* inter_kernel_ch0,
                     short int* inter_kernel_ch1,
                     short int* inter_kernel_ch2,
                     short int* inter_kernel_ch3,
                     uint16_t* inter_bias,
                     short int* output_kernel_ch0,
                     short int* output_kernel_ch1,
                     short int* output_kernel_ch2,
                     short int* output_kernel_ch3,
                     uint16_t* output_bias,
                     uint16_t* output_layernorm_beta,
                     uint16_t* output_layernorm_gamma,
                     unsigned char* fix_pos,
                     int batch_num,
                     int seq_len);
#ifdef __cplusplus
}
#endif

#endif // BERT_H
