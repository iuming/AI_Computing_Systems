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
#ifndef KERNEL_CONFIG_H
#define KERNEL_CONFIG_H

// clusterDim can be 1, 2 or 4 on MLU 270
#define CLUSTER_DIM 4

// coreDim is 4 on MLU 270
#define CORE_DIM    4
#define TOTAL_CORES (CLUSTER_DIM * CORE_DIM)

#define ROUND_UP_TO_32(x) ((x + 31) & (~32))
#define ROUND_UP_TO_64(x) ((x + 63) & (~64))

// For hidden_sim = 768, a kernel matrix (Q/K/V) is divided into
// 12 sub matrices with each size equal 64x768. These matrices are spread to 12 IPU cores
// on 4 clusters (each has 3 active IPU cores).
// The following macros are specific to certain BERT-base model (seq_len = 128)
#define MAX_LAYERS 12
#define HIDDEN_DIM 768
#define HEAD_NUM   12
#define HEAD_SIZE  64
#define SEQ_LEN    128
#define MAX_BATCH_SIZE 8
#define SEQ_PER_CORE (SEQ_LEN / TOTAL_CORES)
#define FFD_ITER_NUM (CLUSTER_DIM)

#define NUM_LAYER_FIXPOS 15
#define NUM_FIXPOS (NUM_LAYER_FIXPOS * MAX_LAYERS + 2)

// Unified NRAM buffer size
#define NRAM_SIZE_IN_KB        472
#define NRAM_BUF_SIZE_IN_HALF  (NRAM_SIZE_IN_KB * 512)

// Unified WRAM buffer size
#define WRAM_SIZE_IN_KB        1024
#define WRAM_BUF_SIZE_IN_FIX16 (WRAM_SIZE_IN_KB * 512)

// Unified SRAM buffer size
#define SRAM_SIZE_IN_KB        2048
#define SRAM_BUF_SIZE_IN_HALF  (SRAM_SIZE_IN_KB * 512)

// Reference tensor size
#define TENSOR_SIZE          (SEQ_LEN * HIDDEN_DIM)
#define HALF_TENSOR_SIZE     (SEQ_LEN * HIDDEN_DIM / 2)

// Macro definitions for Word embedding
#define VOCAB_SIZE    30522
#define SEGMENT_SIZE  2
#define POSITION_SIZE 512

#define NRAM_EMLN_BETA_BUF      0
#define NRAM_EMLN_GAMMA_BUF     (NRAM_EMLN_BETA_BUF + HIDDEN_DIM)
#define NRAM_EMLN_TSR_BUF       (NRAM_EMLN_GAMMA_BUF + HIDDEN_DIM)
#define NRAM_EMLN_FP32_BUF      (NRAM_EMLN_TSR_BUF + TENSOR_SIZE/2)
#define NRAM_EMLN_FP32_VAR_BUF  (NRAM_EMLN_TSR_BUF)
#define NRAM_EMLN_MEAN_BUF      (NRAM_EMLN_FP32_BUF + TENSOR_SIZE/2)
#define NRAM_EMLN_SVAR_BUF      (NRAM_EMLN_MEAN_BUF + SEQ_LEN)

#define NRAM_SEG_TABLE          (NRAM_EMLN_MEAN_BUF)
#define NRAM_INPUT_ID_BUF       (NRAM_SEG_TABLE + SEGMENT_SIZE * HIDDEN_DIM)
#define NRAM_TOKEN_ID_BUF       (NRAM_INPUT_ID_BUF + SEQ_PER_CORE)
#define NRAM_WORD_BUF0          (NRAM_TOKEN_ID_BUF + SEQ_PER_CORE)
#define NRAM_WORD_BUF1          (NRAM_WORD_BUF0 + HIDDEN_DIM)
// End of Macro definitions for Word embedding

// NRAM offsets
// Offsets used for QKV layer out
#define NRAM_BIAS_BUF_Q       0
#define NRAM_BIAS_BUF_K       (NRAM_BIAS_BUF_Q + HEAD_SIZE)
#define NRAM_BIAS_BUF_V       (NRAM_BIAS_BUF_K + HEAD_SIZE)
#define NRAM_MASK_BUF         (NRAM_BIAS_BUF_V + HEAD_SIZE)
#define NRAM_BIAS_BUF_ATTOUT  0
#define NRAM_BIAS_BUF_INTER   0
#define NRAM_BIAS_BUF_FFDOUT  (NRAM_BIAS_BUF_INTER + HEAD_SIZE * 3)
#define NRAM_HTSR_BUF0        (NRAM_MASK_BUF + MAX_BATCH_SIZE * SEQ_LEN)
#define NRAM_HTSR_BUF1        (NRAM_HTSR_BUF0 + HALF_TENSOR_SIZE)
#define NRAM_QK_BUF0          (NRAM_HTSR_BUF1 + HALF_TENSOR_SIZE)
#define NRAM_QK_BUF1          (NRAM_QK_BUF0 + SEQ_LEN * HEAD_SIZE)
#define NRAM_V_BUF0           (NRAM_QK_BUF1 + SEQ_LEN * HEAD_SIZE)
#define NRAM_V_BUF1           (NRAM_V_BUF0 + SEQ_LEN * HEAD_SIZE)

// Offsets used for producing attention probs
#define NRAM_CTX_BUF0         (NRAM_MASK_BUF + MAX_BATCH_SIZE * SEQ_LEN)
#define NRAM_CTX_BUF1         (NRAM_CTX_BUF0 + SEQ_LEN * HEAD_SIZE)
#define NRAM_SMAX_BUF0        (NRAM_CTX_BUF1 + SEQ_LEN * HEAD_SIZE)
#define NRAM_SMAX_BUF1        (NRAM_SMAX_BUF0 + SEQ_LEN * SEQ_LEN)
#define NRAM_SMAX_BUF2        (NRAM_SMAX_BUF1 + SEQ_LEN * SEQ_LEN)
#define NRAM_SMAX_REDUCE_BUF  (NRAM_SMAX_BUF2 + SEQ_LEN * SEQ_LEN)

// Offsets used for attention output kernel
#define NRAM_ATT_IN_BUF0      (NRAM_HTSR_BUF0)
#define NRAM_ATT_IN_BUF1      (NRAM_ATT_IN_BUF0 + TENSOR_SIZE)
#define NRAM_ATT_OUT_BUF0     (NRAM_ATT_IN_BUF1 + TENSOR_SIZE)
#define NRAM_ATT_OUT_BUF1     (NRAM_ATT_OUT_BUF0 + SEQ_LEN * HEAD_SIZE)

// Offsets used for layernorm
#define NRAM_LN_BETA_BUF      0
#define NRAM_LN_GAMMA_BUF     (NRAM_LN_BETA_BUF + HIDDEN_DIM)
#define NRAM_LN_PRE_TSR_BUF   (NRAM_LN_GAMMA_BUF + HIDDEN_DIM)
#define NRAM_LN_CUR_TSR_BUF   (NRAM_LN_PRE_TSR_BUF + TENSOR_SIZE/2)
#define NRAM_LN_FP32_BUF      (NRAM_LN_CUR_TSR_BUF + TENSOR_SIZE/2)
#define NRAM_LN_FP32_VAR_BUF  (NRAM_LN_PRE_TSR_BUF)
#define NRAM_LN_MEAN_BUF      (NRAM_LN_FP32_BUF + TENSOR_SIZE)
#define NRAM_LN_SVAR_BUF      (NRAM_LN_MEAN_BUF + SEQ_LEN)

// Offsets used for intermediate output
#define NRAM_INTER_MLP_BUF0    (NRAM_BIAS_BUF_FFDOUT + HEAD_SIZE * 3)
#define NRAM_INTER_MLP_BUF1    (NRAM_INTER_MLP_BUF0 + TENSOR_SIZE)
#define NRAM_INTER_OUT_BUF     (NRAM_INTER_MLP_BUF1 + TENSOR_SIZE)

// Offsets used for partial feedforward conv output
#define NRAM_FFDIN_BUF0       (NRAM_BIAS_BUF_FFDOUT + HEAD_SIZE * 3)
#define NRAM_FFDIN_BUF1       (NRAM_FFDIN_BUF0 + TENSOR_SIZE/2)
#define NRAM_FFDOUT_BUF0      (NRAM_FFDIN_BUF1 + TENSOR_SIZE/2)
#define NRAM_FFDOUT_BUF1      (NRAM_FFDOUT_BUF0 + SEQ_LEN * HEAD_SIZE * 3 / 2)
#define NRAM_FFDSUM_IN_BUF0   (NRAM_FFDOUT_BUF1 + SEQ_LEN * HEAD_SIZE * 3 / 2)
#define NRAM_FFDSUM_IN_BUF1   (NRAM_FFDSUM_IN_BUF0 + SEQ_LEN * HEAD_SIZE * 3 / 2)
#define NRAM_FFDSUM_OUT_BUF0  (NRAM_FFDSUM_IN_BUF1 + SEQ_LEN * HEAD_SIZE * 3 / 2)
#define NRAM_FFDSUM_OUT_BUF1  (NRAM_FFDSUM_OUT_BUF0 + SEQ_LEN * HEAD_SIZE * 3 / 8)
#define NRAM_FFDSUM_F32_BUF0  (NRAM_FFDSUM_OUT_BUF1 + SEQ_LEN * HEAD_SIZE * 3 / 8)
#define NRAM_FFDSUM_F32_BUF1  (NRAM_FFDSUM_F32_BUF0 + SEQ_LEN * HEAD_SIZE * 3 / 8 * 2)
#define NRAM_FFD_BIAS_F32_BUF (NRAM_FFDSUM_F32_BUF1 + SEQ_LEN * HEAD_SIZE * 3 / 8 * 2)

// Offsets used for SQuAD post processing
#define NRAM_BIAS_BUF_POST    (NRAM_LN_SVAR_BUF + SEQ_LEN)

// WRAM offsets (divided by 64 to fit into WRAM banks)
// Offsets used for cache attention mask and intermediate tensor
#define WRAM_MASK_BUF     0
#define WRAM_TENSOR_BUF   (WRAM_MASK_BUF + (MAX_BATCH_SIZE * SEQ_LEN)/64)

// Offsets used for kernels
#define WRAM_KERNEL_Q_BUF      (WRAM_TENSOR_BUF + (SEQ_LEN * HIDDEN_DIM/2)/64)
#define WRAM_KERNEL_K_BUF      (WRAM_KERNEL_Q_BUF + (HEAD_SIZE * HIDDEN_DIM/64))
#define WRAM_KERNEL_V_BUF      (WRAM_KERNEL_K_BUF + (HEAD_SIZE * HIDDEN_DIM/64))
#define WRAM_KERNEL_FFDOUT_BUF (WRAM_KERNEL_V_BUF + (HEAD_SIZE * HIDDEN_DIM/64))
#define WRAM_KERNEL_INTER_BUF  (WRAM_KERNEL_FFDOUT_BUF + (HEAD_SIZE * 3 * HIDDEN_DIM/64))
#define WRAM_KERNEL_ATTOUT_BUF (WRAM_KERNEL_FFDOUT_BUF)

// Offsets used for producing attention probs
#define WRAM_K_BUF0            (WRAM_KERNEL_Q_BUF)
#define WRAM_K_BUF1            (WRAM_K_BUF0 + (SEQ_LEN * HEAD_SIZE/64))
#define WRAM_V_BUF0            (WRAM_K_BUF1 + (SEQ_LEN * HEAD_SIZE)/64)
#define WRAM_V_BUF1            (WRAM_V_BUF0 + (SEQ_LEN * HEAD_SIZE)/64)

// SRAM offsets
#define SRAM_TSR_BUF0    0
#define SRAM_TSR_BUF1    (SRAM_TSR_BUF0 + TENSOR_SIZE)
#define SRAM_TSR_BUF2    (SRAM_TSR_BUF1 + TENSOR_SIZE)
// We need to reserve input tensors for attention layer-norm
#define SRAM_QKV_BUF      (SRAM_TSR_BUF2 + TENSOR_SIZE)
#define SRAM_CTX_BUF0     (SRAM_TSR_BUF2)
#define SRAM_CTX_BUF1     (SRAM_CTX_BUF0 + 4 * SEQ_LEN * HEAD_SIZE)
#define SRAM_ATT_OUT_BUF0 (SRAM_TSR_BUF2 + TENSOR_SIZE)
#define SRAM_ATT_OUT_BUF1 (SRAM_ATT_OUT_BUF0 + SEQ_LEN * HEAD_SIZE * 3)

// Offsets used for feedforward layer (first four batches)
#define SRAM_INTER_TSR_BUFA   (SRAM_TSR_BUF1)
#define SRAM_HINTER_TSR_BUFA0 (SRAM_INTER_TSR_BUFA)
#define SRAM_HINTER_TSR_BUFA1 (SRAM_HINTER_TSR_BUFA0 + TENSOR_SIZE * 4 / 2)
#define SRAM_HINTER_TSR_BUFA2 (SRAM_HINTER_TSR_BUFA1 + TENSOR_SIZE * 4 / 2)
#define SRAM_HFFD_OUT_BUFA0   (SRAM_HINTER_TSR_BUFA2 + TENSOR_SIZE * 4 / 2)
#define SRAM_HFFD_OUT_BUFA1   (SRAM_HFFD_OUT_BUFA0 + TENSOR_SIZE / 2)
#define SRAM_HFFD_SUM_BUFA0   (SRAM_HFFD_OUT_BUFA1 + TENSOR_SIZE / 2)
#define SRAM_HFFD_SUM_BUFA1   (SRAM_HFFD_SUM_BUFA0 + SEQ_LEN * HEAD_SIZE * 3 / 2)
// Offsets for intermediate out
#define SRAM_INTER_OUT_BUFA0  (SRAM_HINTER_TSR_BUFA2)
#define SRAM_INTER_OUT_BUFA1  (SRAM_INTER_OUT_BUFA0 + SEQ_LEN * HIDDEN_DIM)

// Offsets used for feedforward layer (first four batches)
#define SRAM_INTER_TSR_BUFB   (SRAM_TSR_BUF2)
#define SRAM_HINTER_TSR_BUFB0 (SRAM_INTER_TSR_BUFB)
#define SRAM_HINTER_TSR_BUFB1 (SRAM_HINTER_TSR_BUFB0 + TENSOR_SIZE * 4 / 2)
#define SRAM_HINTER_TSR_BUFB2 (SRAM_HINTER_TSR_BUFB1 + TENSOR_SIZE * 4 / 2)
#define SRAM_HFFD_OUT_BUFB0   (SRAM_HINTER_TSR_BUFB2 + TENSOR_SIZE * 4 / 2)
#define SRAM_HFFD_OUT_BUFB1   (SRAM_HFFD_OUT_BUFB0 + TENSOR_SIZE / 2)
#define SRAM_HFFD_SUM_BUFB0   (SRAM_HFFD_OUT_BUFB1 + TENSOR_SIZE / 2)
#define SRAM_HFFD_SUM_BUFB1   (SRAM_HFFD_SUM_BUFB0 + SEQ_LEN * HEAD_SIZE * 3 / 2)
// Offsets for intermediate out
#define SRAM_INTER_OUT_BUFB0  (SRAM_HINTER_TSR_BUFB2)
#define SRAM_INTER_OUT_BUFB1  (SRAM_INTER_OUT_BUFB0 + SEQ_LEN * HIDDEN_DIM)

// Offsets used for temporary attention output and reshape
#define SRAM_TSR_BUF3 (SRAM_HFFD_SUM_BUFA0)
#define SRAM_TSR_BUF4 (SRAM_TSR_BUF3 + TENSOR_SIZE)

#endif // KERNEL_CONFIG_H
