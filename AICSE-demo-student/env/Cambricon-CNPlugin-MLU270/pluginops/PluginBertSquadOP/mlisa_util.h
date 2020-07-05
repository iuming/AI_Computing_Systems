#ifndef MLISA_UTIL_H
#define MLISA_UTIL_H

#include "bert_squad_config.h"

#define MEM_CORE 0x80

__mlu_func__ uint32_t mlisa_get_tck_low(void) {
  uint32_t tck_low = 0;
  __asm__ volatile("mv.sreg.gpr %%perf_start, 1;\n\t"
                   "mv.sreg.gpr %%perf_read, 1;\n\t"
                   "mv.gpr.sreg %[time_low], %%perf_time_stample_low;\n\t"
                   :[time_low]"=r"(tck_low));
  return tck_low * 40; // Assuming the tick frequence is 25MHz
}
__mlu_func__ int mlisa_get_icache_miss_start(void) {
  int icache_miss_start = 0;
  __asm__ volatile("mv.sreg.gpr %%perf_start, 1;\n\t"
                   "mv.sreg.gpr %%perf_read, 1;\n\t"
                   "mv.gpr.sreg %[miss_start], %%perf_cache_miss_num;\n\t"
                   :[miss_start]"=r"(icache_miss_start));
  return icache_miss_start;
}
__mlu_func__ int mlisa_get_icache_miss_stop(void) {
  int icache_miss_stop = 0;
  __asm__ volatile("mv.sreg.gpr %%perf_stop, 1;\n\t"
                   "mv.sreg.gpr %%perf_read, 1;\n\t"
                   "mv.gpr.sreg %[miss_stop], %%perf_cache_miss_num;\n\t"
                   :[miss_stop]"=r"(icache_miss_stop));
  return icache_miss_stop;
}
__mlu_func__ int mlisa_get_exec_inst_start(void) {
  int inst_count_start = 0;
  __asm__ volatile("mv.sreg.gpr %%perf_start, 1;\n\t"
                   "mv.sreg.gpr %%perf_read, 1;\n\t"
                   "mv.gpr.sreg %[count_start], %%perf_excuted_inst_num;\n\t"
                   :[count_start]"=r"(inst_count_start));
  return inst_count_start;
}
__mlu_func__ int mlisa_get_exec_inst_stop(void) {
  int inst_count_stop = 0;
  __asm__ volatile("mv.sreg.gpr %%perf_stop, 1;\n\t"
                   "mv.sreg.gpr %%perf_read, 1;\n\t"
                   "mv.gpr.sreg %[count_stop], %%perf_excuted_inst_num;\n\t"
                   :[count_stop]"=r"(inst_count_stop));
  return inst_count_stop;
}
__mlu_func__ void mlisa_pv_lock_ipu(void) {
  __asm__ volatile("pv.p.dma0_sync.dma1_sync 0, 0;\n\t");
}
__mlu_func__ void mlisa_pv_unlock_ipu(void) {
  __asm__ volatile("pv.v.dma0_sync.dma1_sync 0, 0;\n\t");
}
__mlu_func__ void mlisa_sync (void) {
  __asm__ volatile("sync;\n\t");
}
__mlu_func__ void mlisa_barrier_cluster(void) {
  __asm__ volatile("barrier.sync.local 1, 5;\n\t");
}
__mlu_func__ void mlisa_barrier_all(void) {
  __asm__ volatile("barrier.sync.global 0, 20;\n\t");
}
__mlu_func__ void mlisa_barrier_cluster_PR0(void) {
  __asm__ volatile("@%pr0 barrier.sync.local 1, 5;\n\t");
}
__mlu_func__ void mlisa_barrier_all_PR0(void) {
  __asm__ volatile("@%pr0 barrier.sync.global 0, 20;\n\t");
}
// Reshape tensor from NRAM to SRAM
__mlu_func__ void mlisa_attr_reshape_NtoS_async(half* sram_dst, half* nram_src)
{
  __asm__ volatile(
      "st.async.stride.sram.nram [%[sram_dst]], [%[nram_src]], 128, 1536, 128, 127;\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src));
}
__mlu_func__ void mlisa_attr_reshape_NtoS_async(half* sram_dst, half* nram_src, int num_segs)
{
  __asm__ volatile(
      "st.async.stride.sram.nram [%[sram_dst]], [%[nram_src]], 128, 1536, 128, %[num_segs];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src), [num_segs] "r"(num_segs));
}
__mlu_func__ void mlisa_attr_reshape_NtoS_async_PR0(half* sram_dst, half* nram_src,
                                                    int num_segs)
{
  __asm__ volatile(
      "@%%pr0 st.async.stride.sram.nram [%[sram_dst]], [%[nram_src]],"
      " 128, 1536, 128, %[num_segs];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src), [num_segs] "r"(num_segs));
}
__mlu_func__ void mlisa_attr_reshape_NtoS_async_PR0(half* sram_dst, half* nram_src)
{
  __asm__ volatile(
      "@%%pr0 st.async.stride.sram.nram [%[sram_dst]], [%[nram_src]], 128, 1536, 128, 127;\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src));
}
__mlu_func__ void mlisa_attr_reshape_NtoS_async_PR1(half* sram_dst, half* nram_src)
{
  __asm__ volatile(
      "@%%pr1 st.async.stride.sram.nram [%[sram_dst]], [%[nram_src]], 128, 1536, 128, 127;\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src));
}
__mlu_func__ void mlisa_attr_reshape_NtoS_async_PR1(half* sram_dst, half* nram_src, int num_segs)
{
  __asm__ volatile(
      "@%%pr1 st.async.stride.sram.nram [%[sram_dst]], [%[nram_src]],"
      " 128, 1536, 128, %[num_segs];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src), [num_segs] "r"(num_segs));
}
__mlu_func__ void mlisa_attr_reshape_NtoS_async_PR5(half* sram_dst, half* nram_src,
                                                    int num_segs)
{
  __asm__ volatile(
      "@%%pr5 st.async.stride.sram.nram [%[sram_dst]], [%[nram_src]],"
      " 128, 1536, 128, %[num_segs];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src), [num_segs] "r"(num_segs));
}
__mlu_func__ void mlisa_reshape_key_NtoS_async(half* sram_dst, half* nram_src)
{
  __asm__ volatile(
      "st.async.stride.sram.nram [%[sram_dst]], [%[nram_src]], 128, 256, 128, 63;\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src));
}
__mlu_func__ void mlisa_reshape_key_NtoS_async_PR2(half* sram_dst, half* nram_src)
{
  __asm__ volatile(
      "@%%pr2 st.async.stride.sram.nram [%[sram_dst]], [%[nram_src]], 128, 256, 128, 63;\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src));
}
__mlu_func__ void mlisa_reshape_key_NtoS_async_PR3(half* sram_dst, half* nram_src)
{
  __asm__ volatile(
      "@%%pr3 st.async.stride.sram.nram [%[sram_dst]], [%[nram_src]], 128, 256, 128, 63;\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src));
}
// Reshape tensor from SRAM to SRAM
__mlu_func__ void mlisa_attr_reshape_StoS_async(half* sram_dst, half* sram_src,
                                                       int dst_cluster_id)
{
  __asm__ volatile(
      "mv.async.stride.sram.sram [%[sram_dst]], [%[sram_src]], 128, 1536, 128, 127, %[dst_id];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [sram_src] "r"(sram_src), [dst_id] "r"(dst_cluster_id));
}
__mlu_func__ void mlisa_attr_reshape_StoS_async(half* sram_dst, half* sram_src, int num_segs,
                                                       int dst_cluster_id)
{
  __asm__ volatile(
      "mv.async.stride.sram.sram [%[sram_dst]], [%[sram_src]], 128, 1536, 128,"
      " %[num_segs], %[dst_id];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [sram_src] "r"(sram_src), [num_segs] "i"(num_segs), [dst_id] "r"(dst_cluster_id));
}
__mlu_func__ void mlisa_attr_reshape_StoS_async_PR1(half* sram_dst, half* sram_src,
                                                    int num_segs, int dst_cluster_id)
{
  __asm__ volatile(
      "@%%pr1 mv.async.stride.sram.sram [%[sram_dst]], [%[sram_src]], 128, 1536, 128,"
      " %[num_segs], %[dst_id];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [sram_src] "r"(sram_src), [num_segs] "r"(num_segs), [dst_id] "r"(dst_cluster_id));
}
__mlu_func__ void mlisa_attr_reshape_StoS_async_PR2(half* sram_dst, half* sram_src,
                                                    int num_segs, int dst_cluster_id)
{
  __asm__ volatile(
      "@%%pr2 mv.async.stride.sram.sram [%[sram_dst]], [%[sram_src]], 128, 1536, 128,"
      " %[num_segs], %[dst_id];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [sram_src] "r"(sram_src), [num_segs] "r"(num_segs), [dst_id] "r"(dst_cluster_id));
}
// Used for reshaping keys for WRAM
__mlu_func__ void mlisa_attr_query_reshape(half* nram_dst, half* nram_src)
{
  __asm__ volatile(
    "tiling3d.nram.b1024 [%[dst]], [%[src]],\n\t" // dst, src
                        "2, 128,\n\t"   // n3, s3 (src)
                        "64, 2,\n\t"    // n4, s4 (src)
                        "1, 1,\n\t"     // n5, s5 (src)
                        "2, 2,\n\t"     // n8, s8 (dst)
                        "64, 4,\n\t"    // n9, s9 (dst)
                        "1, 1, 1;\n\t"  // n10, s10 (dst), n
    : : [dst] "r" (nram_dst), [src] "r" (nram_src));
}
__mlu_func__ void mlisa_inter_reshape_NtoS_async(half* sram_dst, half* nram_src)
{
  __asm__ volatile(
      "st.async.stride.sram.nram [%[sram_dst]], [%[nram_src]], 384, 1536, 384, 63;\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src));
}
__mlu_func__ void mlisa_inter_reshape_NtoS_async_PR1(half* sram_dst, half* nram_src)
{
  __asm__ volatile(
      "@%%pr1 st.async.stride.sram.nram [%[sram_dst]], [%[nram_src]], 384, 1536, 384, 63;\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src));
}
__mlu_func__ void mlisa_inter_reshape_NtoS_async_PR1(half* sram_dst,
                                                     half* nram_src, int seg_num)
{
  __asm__ volatile(
      "@%%pr1 st.async.stride.sram.nram [%[sram_dst]], [%[nram_src]], "
      "384, 1536, 384, %[seg_num];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src), [seg_num] "r"(seg_num));
}
__mlu_func__ void mlisa_inter_reshape_NtoS_async_PR2(half* sram_dst,
                                                     half* nram_src, int seg_num)
{
  __asm__ volatile(
      "@%%pr2 st.async.stride.sram.nram [%[sram_dst]], [%[nram_src]], "
      "384, 1536, 384, %[seg_num];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src), [seg_num] "r"(seg_num));
}
__mlu_func__ void mlisa_inter_reshape_NtoS_async_PR2(half* sram_dst, half* nram_src)
{
  __asm__ volatile(
      "@%%pr2 st.async.stride.sram.nram [%[sram_dst]], [%[nram_src]], 384, 1536, 384, 63;\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src));
}
__mlu_func__ void mlisa_ffd_reshape_NtoS_async(half* sram_dst, half* nram_src)
{
  __asm__ volatile(
      "st.async.stride.sram.nram [%[sram_dst]], [%[nram_src]], 384, 1536, 384, 15;\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src));
}
__mlu_func__ void mlisa_ffd_reshape_NtoS_async_PR4(half* sram_dst, half* nram_src, int seg_num)
{
  __asm__ volatile(
      "@%%pr4 st.async.stride.sram.nram [%[sram_dst]], [%[nram_src]],"
      " 384, 1536, 384, %[seg_num];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src), [seg_num]"r"(seg_num));
}
__mlu_func__ void mlisa_ffd_reshape_NtoS_async_PR5(half* sram_dst, half* nram_src, int seg_num)
{
  __asm__ volatile(
      "@%%pr5 st.async.stride.sram.nram [%[sram_dst]], [%[nram_src]],"
      " 384, 1536, 384, %[seg_num];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src), [seg_num]"r"(seg_num));
}
__mlu_func__ void mlisa_ffd_reshape_StoS_async(half* sram_dst, half* sram_src,
                                               int dst_cluster_id)
{
  __asm__ volatile(
      "mv.async.stride.sram.sram [%[sram_dst]], [%[sram_src]], 384, 1536, 384, 63, %[dst_id];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [sram_src] "r"(sram_src), [dst_id] "r"(dst_cluster_id));
}
__mlu_func__ void mlisa_ffd_reshape_StoS_async_PR1(half* sram_dst, half* sram_src,
                                                   int dst_cluster_id)
{
  __asm__ volatile(
      "@%%pr1 mv.async.stride.sram.sram [%[sram_dst]], [%[sram_src]],"
      " 384, 1536, 384, 63, %[dst_id];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [sram_src] "r"(sram_src), [dst_id] "r"(dst_cluster_id));
}
__mlu_func__ void mlisa_ffd_reshape_StoS_async(half* sram_dst, half* sram_src, int seg_num,
                                               int dst_cluster_id)
{
  __asm__ volatile(
      "mv.async.stride.sram.sram [%[sram_dst]], [%[sram_src]],"
      " 384, 1536, 384, %[seg_num], %[dst_id];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [sram_src] "r"(sram_src), [dst_id] "r"(dst_cluster_id),
          [seg_num] "r"(seg_num));
}
__mlu_func__ void mlisa_ffd_reshape_StoS_async_PR1(half* sram_dst, half* sram_src, int seg_num,
                                                   int dst_cluster_id)
{
  __asm__ volatile(
      "@%%pr1 mv.async.stride.sram.sram [%[sram_dst]], [%[sram_src]],"
      " 384, 1536, 384, %[seg_num], %[dst_id];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [sram_src] "r"(sram_src), [dst_id] "r"(dst_cluster_id),
          [seg_num] "r"(seg_num));
}
__mlu_func__ void mlisa_attr_mlp(half* dst, int16* src, int16* kernel, half* bias, int fix_pos)
{
  __asm__ volatile(
      "conv.nram.f16fix16fix16 [%[dst]], [%[src]], [%[kernel]], [%[bias]],\n\t"
                               "768, 1, 64,\n\t"     //channel_input, src_height, src_width
                               "1, 1,\n\t"           // kernel_height, kernel_width
                               "1, 1,\n\t"           // stride_x, stride_y
                               "64, %[fix_pos];\n\t" // channel_output, fix_pos
      : : [dst]"r"(dst), [src]"r"(src), [kernel]"r"(kernel), [bias]"r"(bias),
          [fix_pos]"r"(fix_pos));
}
__mlu_func__ void mlisa_attr_mlp_PR0(half* dst, int16* src, int16* kernel, half* bias, int fix_pos)
{
  __asm__ volatile(
      "@%%pr0 conv.nram.f16fix16fix16 [%[dst]], [%[src]], [%[kernel]], [%[bias]],\n\t"
                               "768, 1, 64,\n\t"     //channel_input, src_height, src_width
                               "1, 1,\n\t"           // kernel_height, kernel_width
                               "1, 1,\n\t"           // stride_x, stride_y
                               "64, %[fix_pos];\n\t" // channel_output, fix_pos
      : : [dst]"r"(dst), [src]"r"(src), [kernel]"r"(kernel), [bias]"r"(bias),
          [fix_pos]"r"(fix_pos));
}
// Used for producing intermediate output
__mlu_func__ void mlisa_inter_mlp(half* dst, int16* src, int16* kernel, half* bias, int fix_pos)
{
  __asm__ volatile(
      "conv.nram.f16fix16fix16 [%[dst]], [%[src]], [%[kernel]], [%[bias]],\n\t"
                               "768, 1, 128,\n\t"    //channel_input, src_height, src_width
                               "1, 1,\n\t"           // kernel_height, kernel_width
                               "1, 1,\n\t"           // stride_x, stride_y
                               "192, %[fix_pos];\n\t" // channel_output, fix_pos
      : : [dst]"r"(dst), [src]"r"(src), [kernel]"r"(kernel), [bias]"r"(bias),
          [fix_pos]"r"(fix_pos));
}
__mlu_func__ void mlisa_inter_mlp_PR3(half* dst, int16* src, int16* kernel, half* bias, int fix_pos)
{
  __asm__ volatile(
      "@%%pr3 conv.nram.f16fix16fix16 [%[dst]], [%[src]], [%[kernel]], [%[bias]],\n\t"
                               "768, 1, 128,\n\t"    //channel_input, src_height, src_width
                               "1, 1,\n\t"           // kernel_height, kernel_width
                               "1, 1,\n\t"           // stride_x, stride_y
                               "192, %[fix_pos];\n\t" // channel_output, fix_pos
      : : [dst]"r"(dst), [src]"r"(src), [kernel]"r"(kernel), [bias]"r"(bias),
          [fix_pos]"r"(fix_pos));
}
__mlu_func__ void mlisa_inter_mlp_PR3(half* dst, int16* src, int16* kernel,
                                      half* bias, int src_width, int fix_pos)
{
  __asm__ volatile(
      "@%%pr3 conv.nram.f16fix16fix16 [%[dst]], [%[src]], [%[kernel]], [%[bias]],\n\t"
                               "768, 1, %[src_width],\n\t"    //channel_input, src_height, src_width
                               "1, 1,\n\t"           // kernel_height, kernel_width
                               "1, 1,\n\t"           // stride_x, stride_y
                               "192, %[fix_pos];\n\t" // channel_output, fix_pos
      : : [dst]"r"(dst), [src]"r"(src), [kernel]"r"(kernel), [bias]"r"(bias),
          [src_width]"r"(src_width), [fix_pos]"r"(fix_pos));
}
__mlu_func__ void mlisa_inter_mlp_PR3(float* dst, int16* src, int16* kernel,
                                      half* bias, int src_width, int fix_pos)
{
  __asm__ volatile(
      "@%%pr3 conv.nram.f32fix16fix16 [%[dst]], [%[src]], [%[kernel]], [%[bias]],\n\t"
                               "768, 1, %[src_width],\n\t"    //channel_input, src_height, src_width
                               "1, 1,\n\t"           // kernel_height, kernel_width
                               "1, 1,\n\t"           // stride_x, stride_y
                               "192, %[fix_pos];\n\t" // channel_output, fix_pos
      : : [dst]"r"(dst), [src]"r"(src), [kernel]"r"(kernel), [bias]"r"(bias),
          [src_width]"r"(src_width), [fix_pos]"r"(fix_pos));
}
// Used for producing feedforward output
__mlu_func__ void mlisa_feedforward_mlp(half* dst, int16* src, int16* kernel, int fix_pos)
{
  __asm__ volatile(
      "conv.nram.f16fix16fix16 [%[dst]], [%[src]], [%[kernel]],\n\t"
                               "768, 1, 64,\n\t"    //channel_input, src_height, src_width
                               "1, 1,\n\t"           // kernel_height, kernel_width
                               "1, 1,\n\t"           // stride_x, stride_y
                               "192, %[fix_pos];\n\t" // channel_output, fix_pos
      : : [dst]"r"(dst), [src]"r"(src), [kernel]"r"(kernel),
          [fix_pos]"r"(fix_pos));
}
__mlu_func__ void mlisa_feedforward_mlp_PR7(half* dst, int16* src, int16* kernel,
                                            int src_width, int fix_pos)
{
  __asm__ volatile(
      "@%%pr7 conv.nram.f16fix16fix16 [%[dst]], [%[src]], [%[kernel]],\n\t"
                               "768, 1, %[src_width],\n\t" //channel_input, src_height, src_width
                               "1, 1,\n\t"           // kernel_height, kernel_width
                               "1, 1,\n\t"           // stride_x, stride_y
                               "192, %[fix_pos];\n\t" // channel_output, fix_pos
      : : [dst]"r"(dst), [src]"r"(src), [kernel]"r"(kernel),
          [src_width]"r" (src_width), [fix_pos]"r"(fix_pos));
}
__mlu_func__ void mlisa_attr_qk_matmul(half* dst, int16* query, int16* key, int fix_pos)
{
  __asm__ volatile(
    "conv.nram.f16fix16fix16 [%[dst]], [%[query]], [%[key]],\n\t"
                             "64, 1, 128,\n\t" //channel_input, src_height, src_width
                             "1, 1,\n\t" // kernel_height, kernel_width
                             "1, 1,\n\t" // stride_x, stride_y
                             "128, %[fix_pos];\n\t" // channel_output, fix_pos
    : : [dst]"r"(dst), [query]"r"(query), [key]"r"(key), [fix_pos]"r"(fix_pos));
}
__mlu_func__ void mlisa_attr_qkv_matmul(half* dst, int16* qk, int16* value, int fix_pos)
{
  __asm__ volatile(
    "conv.nram.f16fix16fix16 [%[dst]], [%[qk]], [%[value]],\n\t"
                             "128, 1, 128,\n\t" //channel_input, src_height, src_width
                             "1, 1,\n\t" // kernel_height, kernel_width
                             "1, 1,\n\t" // stride_x, stride_y
                             "64, %[fix_pos];\n\t" // channel_output, fix_pos
    : : [dst]"r"(dst), [qk]"r"(qk), [value]"r"(value), [fix_pos]"r"(fix_pos));
}
__mlu_func__ void mlisa_attr_qkv_matmul_fixout(half* dst, int16* qk, int16* value, int fix_pos)
{
  __asm__ volatile(
    "conv.nram.fix16fix16fix16 [%[dst]], [%[qk]], [%[value]],\n\t"
                               "128, 1, 128,\n\t" //channel_input, src_height, src_width
                               "1, 1,\n\t" // kernel_height, kernel_width
                               "1, 1,\n\t" // stride_x, stride_y
                               "64, %[fix_pos];\n\t" // channel_output, fix_pos
    : : [dst]"r"(dst), [qk]"r"(qk), [value]"r"(value), [fix_pos]"r"(fix_pos));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __OPTIMIZE__ // if compiled in optimized mode
__mlu_func__ void mlisa_stream_add_f16_PR3(half* dst, half* src0, half* src1, int num_elems)
{
  __asm__ volatile(
    "@%%pr3 add.nram.f16 [%[dst]], [%[src0]], [%[src1]], %[num_elems];\n\t"
    : : [dst]"r"(dst), [src0]"r"(src0), [src1]"r"(src1), [num_elems]"i"(num_elems)
  );
}
__mlu_func__ void mlisa_stream_add_f32_PR3(float* dst, float* src0, float* src1, int num_elems)
{
  __asm__ volatile(
    "@%%pr3 add.nram.f32 [%[dst]], [%[src0]], [%[src1]], %[num_elems];\n\t"
    : : [dst]"r"(dst), [src0]"r"(src0), [src1]"r"(src1), [num_elems]"r"(num_elems)
  );
}
__mlu_func__ void mlisa_stream_cycle_add_f16_PR3(half* dst, half* src0, half* src1,
                                                 int src_elem_count, int seg_elem_count)
{
  __asm__ volatile(
    "@%%pr3 add.cycle.nram.f16 [%[dst]], [%[src0]], [%[src1]],"
    " %[num_src_elems], %[num_seg_elems];\n\t"
    : : [dst]"r"(dst), [src0]"r"(src0), [src1]"r"(src1),
        [num_src_elems]"r"(src_elem_count), [num_seg_elems]"i"(seg_elem_count)
  );
}
__mlu_func__ void mlisa_stream_cycle_add_f32_PR3(float* dst, float* src0, float* src1,
                                                 int src_elem_count, int seg_elem_count)
{
  __asm__ volatile(
    "@%%pr3 add.cycle.nram.f32 [%[dst]], [%[src0]], [%[src1]],"
    " %[num_src_elems], %[num_seg_elems];\n\t"
    : : [dst]"r"(dst), [src0]"r"(src0), [src1]"r"(src1),
        [num_src_elems]"r"(src_elem_count), [num_seg_elems]"i"(seg_elem_count)
  );
}
__mlu_func__ void mlisa_mem_load_StoN_async_PR1(half* nram_dst, half* sram_src, int size)
{
  __asm__ volatile(
      "@%%pr1 ld.async.nram.sram [%[nram_dst]], [%[sram_src]], %[size];\n\t"
      : : [nram_dst] "r"(nram_dst),
          [sram_src] "r"(sram_src),
          [size] "r"(size));
}
// Load kernel from SRAM to WRAM
__mlu_func__ void mlisa_mem_load_StoW_async_PR0(int16* wram_dst, half* sram_src, int size)
{
  __asm__ volatile(
      "@%%pr0 ld.async.wram.sram [%[wram_dst]], [%[sram_src]], %[size];\n\t"
      : : [wram_dst] "r"(wram_dst),
          [sram_src] "r"(sram_src),
          [size] "i"(size));
}
__mlu_func__ void mlisa_mem_load_StoW_async_PR1(int16* wram_dst, half* sram_src, int size)
{
  __asm__ volatile(
      "@%%pr1 ld.async.wram.sram [%[wram_dst]], [%[sram_src]], %[size];\n\t"
      : : [wram_dst] "r"(wram_dst),
          [sram_src] "r"(sram_src),
          [size] "i"(size));
}
__mlu_func__ void mlisa_mem_load_StoW_async_PR3(int16* wram_dst, half* sram_src, int size)
{
  __asm__ volatile(
      "@%%pr3 ld.async.wram.sram [%[wram_dst]], [%[sram_src]], %[size];\n\t"
      : : [wram_dst] "r"(wram_dst),
          [sram_src] "r"(sram_src),
          [size] "i"(size));
}
__mlu_func__ void mlisa_transpose_PR1(half* dst, half* src, int h, int w)
{
  __asm__ volatile(
    "@%%pr1 trans.nram.f16 [%[dst]], [%[src]], %[high], %[width];\n\t"
    : : [dst]"r"(dst), [src]"r"(src), [high]"i"(h), [width]"i"(w));
}
__mlu_func__ void mlisa_half2int16_rd_PR0(int16* dst, half* src, int num_elems, int fixpos)
{
  __asm__ volatile(
    "@%%pr0 cvtfix16.nram.rd.f16 [%[dst]], [%[src]], %[num_elems], %[fix_pos];\n\t"
    : : [dst]"r"(dst), [src]"r"(src), [num_elems]"i"(num_elems), [fix_pos]"r"(fixpos)
  );
}
__mlu_func__ void mlisa_half2int16_rd_PR1(int16* dst, half* src, int num_elems, int fixpos)
{
  __asm__ volatile(
    "@%%pr1 cvtfix16.nram.rd.f16 [%[dst]], [%[src]], %[num_elems], %[fix_pos];\n\t"
    : : [dst]"r"(dst), [src]"r"(src), [num_elems]"r"(num_elems), [fix_pos]"r"(fixpos)
  );
}
__mlu_func__ void mlisa_half2int16_rd_PR2(int16* dst, half* src, int num_elems, int fixpos)
{
  __asm__ volatile(
    "@%%pr2 cvtfix16.nram.rd.f16 [%[dst]], [%[src]], %[num_elems], %[fix_pos];\n\t"
    : : [dst]"r"(dst), [src]"r"(src), [num_elems]"i"(num_elems), [fix_pos]"r"(fixpos)
  );
}
__mlu_func__ void mlisa_half2int16_rd_PR3(int16* dst, half* src, int num_elems, int fixpos)
{
  __asm__ volatile(
    "@%%pr3 cvtfix16.nram.rd.f16 [%[dst]], [%[src]], %[num_elems], %[fix_pos];\n\t"
    : : [dst]"r"(dst), [src]"r"(src), [num_elems]"r"(num_elems), [fix_pos]"r"(fixpos)
  );
}
__mlu_func__ void mlisa_half2float_PR3(float* dst, half* src, int num_elems)
{
  __asm__ volatile(
    "@%%pr3 cvtf32.nram.f16 [%[dst]], [%[src]], %[num_elems];\n\t"
    : : [dst]"r"(dst), [src]"r"(src), [num_elems]"r"(num_elems)
  );
}
__mlu_func__ void mlisa_float2half_rd_PR3(half* dst, float* src, int num_elems)
{
  __asm__ volatile(
    "@%%pr3 cvtf16.nram.rd.f32 [%[dst]], [%[src]], %[num_elems];\n\t"
    : : [dst]"r"(dst), [src]"r"(src), [num_elems]"r"(num_elems)
  );
}
__mlu_func__ void mlisa_setpred_eqi_PR0(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.eq.pred.u32 %%pr0, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_eqi_PR1(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.eq.pred.u32 %%pr1, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_eqi_PR2(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.eq.pred.u32 %%pr2, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_eqi_PR3(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.eq.pred.u32 %%pr3, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_eqi_PR4(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.eq.pred.u32 %%pr4, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_eqi_PR5(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.eq.pred.u32 %%pr5, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_eqi_PR6(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.eq.pred.u32 %%pr6, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_eqi_PR8(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.eq.pred.u32 %%pr8, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_gei_PR0(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.ge.pred.u32 %%pr0, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_gei_PR1(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.ge.pred.u32 %%pr1, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_gti_PR1(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.gt.pred.u32 %%pr1, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_gti_PR2(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.gt.pred.u32 %%pr2, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_lti_PR0(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.lt.pred.u32 %%pr0, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_lti_PR1(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.lt.pred.u32 %%pr1, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_lti_PR2(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.lt.pred.u32 %%pr2, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_lti_PR3(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.lt.pred.u32 %%pr3, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_lt_PR3(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.lt.pred.u32 %%pr3, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "r" (imm)
  );
}
__mlu_func__ void mlisa_setpred_lti_PR4(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.lt.pred.u32 %%pr4, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_lti_PR5(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.lt.pred.u32 %%pr5, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_lti_PR7(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.lt.pred.u32 %%pr7, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_lei_PR0(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.le.pred.u32 %%pr0, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "i" (imm)
  );
}
__mlu_func__ void mlisa_setpred_le_PR0(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.le.pred.u32 %%pr0, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "r" (imm)
  );
}
__mlu_func__ void mlisa_active_gelu_PR3(half* dst, half* src, int num_elems)
{
  __asm__ volatile(
    "@%%pr3 active.gelu.nram.f16 [%[dst]], [%[src]], %[num_elems];\n\t"
    : : [dst]"r"(dst), [src]"r"(src), [num_elems]"r"(num_elems)
  );
}
__mlu_func__ void mlisa_active_gelu_PR3(float* dst, float* src, int num_elems)
{
  __asm__ volatile(
    "@%%pr3 active.gelu.nram.f32 [%[dst]], [%[src]], %[num_elems];\n\t"
    : : [dst]"r"(dst), [src]"r"(src), [num_elems]"r"(num_elems)
  );
}
// Load kernel from GDRAM to WRAM
__mlu_func__ void mlisa_load_kernel_GtoW_async(int16* wram_dst, int16* gdram_src, int size)
{
  __asm__ volatile(
      "ld.async.wram.gdram [%[wdst]], [%[gsrc]], %[size];\n\t"
      : : [wdst] "r"(wram_dst),
          [gsrc] "r"(gdram_src),
          [size] "i"(size));
}
__mlu_func__ void mlisa_load_kernel_GtoW_async_PR0(int16* wram_dst, int16* gdram_src, int size)
{
  __asm__ volatile(
      "@%%pr0 ld.async.wram.gdram [%[wdst]], [%[gsrc]], %[size];\n\t"
      : : [wdst] "r"(wram_dst),
          [gsrc] "r"(gdram_src),
          [size] "i"(size));
}
__mlu_func__ void mlisa_load_kernel_GtoW_async_PR3(int16* wram_dst, int16* gdram_src, int size)
{
  __asm__ volatile(
      "@%%pr3 ld.async.wram.gdram [%[wdst]], [%[gsrc]], %[size];\n\t"
      : : [wdst] "r"(wram_dst),
          [gsrc] "r"(gdram_src),
          [size] "i"(size));
}
__mlu_func__ void mlisa_load_kernel_GtoW_async_PR4(int16* wram_dst, int16* gdram_src, int size)
{
  __asm__ volatile(
      "@%%pr4 ld.async.wram.gdram [%[wdst]], [%[gsrc]], %[size];\n\t"
      : : [wdst] "r"(wram_dst),
          [gsrc] "r"(gdram_src),
          [size] "i"(size));
}
__mlu_func__ void mlisa_load_kernel_GtoW_async_PR5(int16* wram_dst, int16* gdram_src, int size)
{
  __asm__ volatile(
      "@%%pr5 ld.async.wram.gdram [%[wdst]], [%[gsrc]], %[size];\n\t"
      : : [wdst] "r"(wram_dst),
          [gsrc] "r"(gdram_src),
          [size] "i"(size));
}
__mlu_func__ void mlisa_load_kernel_GtoW_async_PR6(int16* wram_dst, int16* gdram_src, int size)
{
  __asm__ volatile(
      "@%%pr6 ld.async.wram.gdram [%[wdst]], [%[gsrc]], %[size];\n\t"
      : : [wdst] "r"(wram_dst),
          [gsrc] "r"(gdram_src),
          [size] "i"(size));
}
__mlu_func__ void mlisa_load_kernel_GtoW_async_PR8(int16* wram_dst, int16* gdram_src, int size)
{
  __asm__ volatile(
      "@%%pr8 ld.async.wram.gdram [%[wdst]], [%[gsrc]], %[size];\n\t"
      : : [wdst] "r"(wram_dst),
          [gsrc] "r"(gdram_src),
          [size] "i"(size));
}
// Load bias from GDRAM to NRAM
__mlu_func__ void mlisa_mem_mv_NtoW_async_PR1(int16* wram_dst, half* nram_src, int size)
{
  __asm__ volatile(
      "@%%pr1 mv.async.wram.nram [%[wram_dst]], [%[nram_src]], %[size];\n\t"
      : : [wram_dst] "r"(wram_dst),
          [nram_src] "r"(nram_src),
          [size] "i"(size));
}
__mlu_func__ void mlisa_mem_load_GtoN_async_PR1(half* nram_dst, float* gdram_src, int size)
{
  __asm__ volatile(
      "@%%pr1 ld.async.nram.gdram [%[nram_dst]], [%[gdram_src]], %[size];\n\t"
      : : [nram_dst] "r"(nram_dst),
          [gdram_src] "r"(gdram_src),
          [size] "r"(size));
}
__mlu_func__ void mlisa_mem_load_StoN_async_PR0(half* nram_dst, half* sram_src, int size)
{
  __asm__ volatile(
      "@%%pr0 ld.async.nram.sram [%[nram_dst]], [%[sram_src]], %[size];\n\t"
      : : [nram_dst] "r"(nram_dst),
          [sram_src] "r"(sram_src),
          [size] "r"(size));
}
__mlu_func__ void mlisa_mem_load_StoN_async_PR3(half* nram_dst, half* sram_src, int size)
{
  __asm__ volatile(
      "@%%pr3 ld.async.nram.sram [%[nram_dst]], [%[sram_src]], %[size];\n\t"
      : : [nram_dst] "r"(nram_dst),
          [sram_src] "r"(sram_src),
          [size] "i"(size));
}
// Save tensor from NRAM to SRAM
__mlu_func__ void mlisa_mem_store_NtoS_async(half* sram_dst, half* nram_src, int size)
{
  __asm__ volatile(
      "st.async.sram.nram [%[sram_dst]], [%[nram_src]], %[size];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src),
          [size]"r"(size));
}
__mlu_func__ void mlisa_mem_store_NtoS_async_PR0(half* sram_dst, half* nram_src, int size)
{
  __asm__ volatile(
      "@%%pr0 st.async.sram.nram [%[sram_dst]], [%[nram_src]], %[size];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src),
          [size]"i"(size));
}
__mlu_func__ void mlisa_mem_store_NtoS_async_PR1(half* sram_dst, half* nram_src, int size)
{
  __asm__ volatile(
      "@%%pr1 st.async.sram.nram [%[sram_dst]], [%[nram_src]], %[size];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src),
          [size]"i"(size));
}
__mlu_func__ void mlisa_mem_store_NtoS_async_PR2(half* sram_dst, half* nram_src, int size)
{
  __asm__ volatile(
      "@%%pr2 st.async.sram.nram [%[sram_dst]], [%[nram_src]], %[size];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src),
          [size]"r"(size));
}
__mlu_func__ void mlisa_ctx_store_NtoS_async_PR1(half* sram_dst, half* nram_src, int size)
{
  __asm__ volatile(
      "@%%pr1 st.async.sram.nram [%[sram_dst]], [%[nram_src]], %[size];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src),
          [size]"r"(size));
}
__mlu_func__ void mlisa_ctx_store_NtoS_async_PR2(half* sram_dst, half* nram_src, int size)
{
  __asm__ volatile(
      "@%%pr2 st.async.sram.nram [%[sram_dst]], [%[nram_src]], %[size];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src),
          [size]"r"(size));
}
__mlu_func__ void mlisa_mem_store_NtoS_async_PR3(half* sram_dst, half* nram_src, int size)
{
  __asm__ volatile(
      "@%%pr3 st.async.sram.nram [%[sram_dst]], [%[nram_src]], %[size];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src),
          [size]"i"(size));
}
__mlu_func__ void mlisa_mem_store_NtoS_async_PR4(half* sram_dst, half* nram_src, int size)
{
  __asm__ volatile(
      "@%%pr4 st.async.sram.nram [%[sram_dst]], [%[nram_src]], %[size];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src),
          [size]"i"(size));
}
__mlu_func__ void mlisa_mem_store_NtoS_async_PR6(half* sram_dst, half* nram_src, int size)
{
  __asm__ volatile(
      "@%%pr6 st.async.sram.nram [%[sram_dst]], [%[nram_src]], %[size];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src),
          [size]"r"(size));
}
// Exchange tensor between clusters' SRAM
__mlu_func__ void mlisa_tensor_exchange_StoS_async_PR0(half* sram_dst, half* sram_src,
                                                       int size, int dst_cluster_id)
{
  __asm__ volatile(
      "@%%pr0 mv.async.sram.sram [%[sram_dst]], [%[sram_src]], %[size], %[dst_id];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [sram_src] "r"(sram_src),
          [size] "r"(size), [dst_id] "r"(dst_cluster_id));
}
__mlu_func__ void mlisa_tensor_exchange_StoS_async_PR1(half* sram_dst, half* sram_src,
                                                       int size, int dst_cluster_id)
{
  __asm__ volatile(
      "@%%pr1 mv.async.sram.sram [%[sram_dst]], [%[sram_src]], %[size], %[dst_id];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [sram_src] "r"(sram_src),
          [size] "r"(size), [dst_id] "r"(dst_cluster_id));
}
// Broadcast tensor from SRAM to NRAM on different cores
__mlu_func__ void mlisa_mem_multicast_StoN_all(half* nram_dst, half* sram_src, int size)
{
  __asm__ volatile(
        "ld.multicast.nram.sram [%[nram_buf]], [%[sram_buf]], %[size], 0xF;\n\t"
        : : [nram_buf] "r"(nram_dst),
            [sram_buf] "r"(sram_src),
            [size] "r"(size));
}
__mlu_func__ void mlisa_mem_multicast_StoN_all_PR0(half* nram_dst, half* sram_src, int size)
{
  __asm__ volatile(
        "@%%pr0 ld.multicast.nram.sram [%[nram_buf]], [%[sram_buf]], %[size], 0xF;\n\t"
        : : [nram_buf] "r"(nram_dst),
            [sram_buf] "r"(sram_src),
            [size] "r"(size));
}
__mlu_func__ void mlisa_attr_multicast_tensor(half* nram_dst, half* sram_src, int size)
{
  __asm__ volatile(
        "ld.multicast.nram.sram [%[nram_buf]], [%[sram_buf]], %[size], 0x7;\n\t"
        : : [nram_buf] "r"(nram_dst),
            [sram_buf] "r"(sram_src),
            [size] "i"(size));
}
__mlu_func__ void mlisa_attr_multicast_tensor_PR0(half* nram_dst, half* sram_src, int size)
{
  __asm__ volatile(
        "@%%pr0 ld.multicast.nram.sram [%[nram_buf]], [%[sram_buf]], %[size], 0x7;\n\t"
        : : [nram_buf] "r"(nram_dst),
            [sram_buf] "r"(sram_src),
            [size] "r"(size));
}
__mlu_func__ void mlisa_attr_multicast_tensor_PR1(half* nram_dst, half* sram_src, int size)
{
  __asm__ volatile(
        "@%%pr1 ld.multicast.nram.sram [%[nram_buf]], [%[sram_buf]], %[size], 0x7;\n\t"
        : : [nram_buf] "r"(nram_dst),
            [sram_buf] "r"(sram_src),
            [size] "r"(size));
}
////////////////////////////////////////////////////////////////////////////////////////////////////
#else // if compiled in -O0 mode
__mlu_func__ void mlisa_stream_add_f16_PR3(half* dst, half* src0, half* src1, int num_elems)
{
  __asm__ volatile(
    "@%%pr3 add.nram.f16 [%[dst]], [%[src0]], [%[src1]], %[num_elems];\n\t"
    : : [dst]"r"(dst), [src0]"r"(src0), [src1]"r"(src1), [num_elems]"r"(num_elems)
  );
}
__mlu_func__ void mlisa_stream_add_f32_PR3(float* dst, float* src0, float* src1, int num_elems)
{
  __asm__ volatile(
    "@%%pr3 add.nram.f32 [%[dst]], [%[src0]], [%[src1]], %[num_elems];\n\t"
    : : [dst]"r"(dst), [src0]"r"(src0), [src1]"r"(src1), [num_elems]"r"(num_elems)
  );
}
__mlu_func__ void mlisa_stream_cycle_add_f16_PR3(half* dst, half* src0, half* src1,
                                                 int src_elem_count, int seg_elem_count)
{
  __asm__ volatile(
    "@%%pr3 add.cycle.nram.f16 [%[dst]], [%[src0]], [%[src1]],"
    " %[num_src_elems], %[num_seg_elems];\n\t"
    : : [dst]"r"(dst), [src0]"r"(src0), [src1]"r"(src1),
        [num_src_elems]"r"(src_elem_count), [num_seg_elems]"r"(seg_elem_count)
  );
}
__mlu_func__ void mlisa_stream_cycle_add_f32_PR3(float* dst, float* src0, float* src1,
                                                 int src_elem_count, int seg_elem_count)
{
  __asm__ volatile(
    "@%%pr3 add.cycle.nram.f32 [%[dst]], [%[src0]], [%[src1]],"
    " %[num_src_elems], %[num_seg_elems];\n\t"
    : : [dst]"r"(dst), [src0]"r"(src0), [src1]"r"(src1),
        [num_src_elems]"r"(src_elem_count), [num_seg_elems]"r"(seg_elem_count)
  );
}
__mlu_func__ void mlisa_mem_load_StoN_async_PR1(half* nram_dst, half* sram_src, int size)
{
  __asm__ volatile(
      "@%%pr1 ld.async.nram.sram [%[nram_dst]], [%[sram_src]], %[size];\n\t"
      : : [nram_dst] "r"(nram_dst),
          [sram_src] "r"(sram_src),
          [size] "r"(size));
}
// Load kernel from SRAM to WRAM
__mlu_func__ void mlisa_mem_load_StoW_async_PR0(int16* wram_dst, half* sram_src, int size)
{
  __asm__ volatile(
      "@%%pr0 ld.async.wram.sram [%[wram_dst]], [%[sram_src]], %[size];\n\t"
      : : [wram_dst] "r"(wram_dst),
          [sram_src] "r"(sram_src),
          [size] "r"(size));
}
// Save tensor from NRAM to SRAM
__mlu_func__ void mlisa_store_tensor_NtoS_async_PR0(half* sram_dst, half* nram_src, int size)
{
  __asm__ volatile(
      "@%%pr0 st.async.sram.nram [%[sram_dst]], [%[nram_src]], %[size];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src),
          [size]"r"(size));
}
__mlu_func__ void mlisa_half2int16_rd_PR3(int16* dst, half* src, int num_elems, int fixpos)
{
  __asm__ volatile(
    "@%%pr3 cvtfix16.nram.rd.f16 [%[dst]], [%[src]], %[num_elems], %[fix_pos];\n\t"
    : : [dst]"r"(dst), [src]"r"(src), [num_elems]"r"(num_elems), [fix_pos]"i"(fixpos)
  );
}
__mlu_func__ void mlisa_half2float_PR3(float* dst, half* src, int num_elems)
{
  __asm__ volatile(
    "@%%pr3 cvtf32.nram.f16 [%[dst]], [%[src]], %[num_elems];\n\t"
    : : [dst]"r"(dst), [src]"r"(src), [num_elems]"r"(num_elems)
  );
}
__mlu_func__ void mlisa_float2half_rd_PR3(half* dst, float* src, int num_elems)
{
  __asm__ volatile(
    "@%%pr3 cvtf16.nram.rd.f32 [%[dst]], [%[src]], %[num_elems];\n\t"
    : : [dst]"r"(dst), [src]"r"(src), [num_elems]"r"(num_elems)
  );
}
__mlu_func__ void mlisa_setpred_eqi_PR0(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.eq.pred.u32 %%pr0, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "r" (imm)
  );
}
__mlu_func__ void mlisa_setpred_eqi_PR1(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.eq.pred.u32 %%pr1, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "r" (imm)
  );
}
__mlu_func__ void mlisa_setpred_eqi_PR2(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.eq.pred.u32 %%pr2, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "r" (imm)
  );
}
__mlu_func__ void mlisa_setpred_eqi_PR3(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.eq.pred.u32 %%pr3, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "r" (imm)
  );
}
__mlu_func__ void mlisa_setpred_eqi_PR4(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.eq.pred.u32 %%pr4, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "r" (imm)
  );
}
__mlu_func__ void mlisa_setpred_eqi_PR5(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.eq.pred.u32 %%pr5, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "r" (imm)
  );
}
__mlu_func__ void mlisa_setpred_eqi_PR6(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.eq.pred.u32 %%pr6, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "r" (imm)
  );
}
__mlu_func__ void mlisa_setpred_gei_PR0(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.ge.pred.u32 %%pr0, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "r" (imm)
  );
}
__mlu_func__ void mlisa_setpred_gei_PR1(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.ge.pred.u32 %%pr1, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "r" (imm)
  );
}
__mlu_func__ void mlisa_setpred_gti_PR2(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.gt.pred.u32 %%pr2, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "r" (imm)
  );
}
__mlu_func__ void mlisa_setpred_lti_PR0(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.lt.pred.u32 %%pr0, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "r" (imm)
  );
}
__mlu_func__ void mlisa_setpred_lti_PR3(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.lt.pred.u32 %%pr3, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "r" (imm)
  );
}
__mlu_func__ void mlisa_setpred_lti_PR7(uint32_t a, uint32_t imm)
{
  __asm__ volatile(
      "setp.lt.pred.u32 %%pr7, %[in_a], %[in_imm];\n\t"
      : : [in_a] "r" (a), [in_imm] "r" (imm)
  );
}
__mlu_func__ void mlisa_active_gelu_PR3(half* dst, half* src, int num_elems)
{
  __asm__ volatile(
    "@%%pr3 active.gelu.nram.f16 [%[dst]], [%[src]], %[num_elems];\n\t"
    : : [dst]"r"(dst), [src]"r"(src), [num_elems]"r"(num_elems)
  );
}
// Load kernel from GDRAM to WRAM
__mlu_func__ void mlisa_load_kernel_GtoW_async_PR3(int16* wram_dst, int16* gdram_src, int size)
{
  __asm__ volatile(
      "@%%pr3 ld.async.wram.gdram [%[wdst]], [%[gsrc]], %[size];\n\t"
      : : [wdst] "r"(wram_dst),
          [gsrc] "r"(gdram_src),
          [size] "r"(size));
}
__mlu_func__ void mlisa_mem_load_StoN_async_PR0(half* nram_dst, half* sram_src, int size)
{
  __asm__ volatile(
      "@%%pr0 ld.async.nram.sram [%[nram_dst]], [%[sram_src]], %[size];\n\t"
      : : [nram_dst] "r"(nram_dst),
          [sram_src] "r"(sram_src),
          [size] "r"(size));
}
// Save tensor from NRAM to SRAM
__mlu_func__ void mlisa_mem_store_NtoS_async_PR0(half* sram_dst, half* nram_src, int size)
{
  __asm__ volatile(
      "@%%pr0 st.async.sram.nram [%[sram_dst]], [%[nram_src]], %[size];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src),
          [size]"r"(size));
}
__mlu_func__ void mlisa_mem_store_NtoS_async_PR1(half* sram_dst, half* nram_src, int size)
{
  __asm__ volatile(
      "@%%pr1 st.async.sram.nram [%[sram_dst]], [%[nram_src]], %[size];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src),
          [size]"r"(size));
}
__mlu_func__ void mlisa_mem_store_NtoS_async_PR2(half* sram_dst, half* nram_src, int size)
{
  __asm__ volatile(
      "@%%pr2 st.async.sram.nram [%[sram_dst]], [%[nram_src]], %[size];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src),
          [size]"r"(size));
}
__mlu_func__ void mlisa_mem_store_NtoS_async_PR6(half* sram_dst, half* nram_src, int size)
{
  __asm__ volatile(
      "@%%pr6 st.async.sram.nram [%[sram_dst]], [%[nram_src]], %[size];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [nram_src] "r"(nram_src),
          [size]"r"(size));
}
// Exchange tensor between clusters' SRAM
__mlu_func__ void mlisa_tensor_exchange_StoS_async_PR0(half* sram_dst, half* sram_src,
                                                       int size, int dst_cluster_id)
{
  __asm__ volatile(
      "@%%pr0 mv.async.sram.sram [%[sram_dst]], [%[sram_src]], %[size], %[dst_id];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [sram_src] "r"(sram_src),
          [size] "r"(size), [dst_id] "r"(dst_cluster_id));
}
__mlu_func__ void mlisa_tensor_exchange_StoS_async_PR1(half* sram_dst, half* sram_src,
                                                       int size, int dst_cluster_id)
{
  __asm__ volatile(
      "@%%pr1 mv.async.sram.sram [%[sram_dst]], [%[sram_src]], %[size], %[dst_id];\n\t"
      : : [sram_dst] "r"(sram_dst),
          [sram_src] "r"(sram_src),
          [size] "r"(size), [dst_id] "r"(dst_cluster_id));
}
// Broadcast tensor from SRAM to NRAM on different cores
__mlu_func__ void mlisa_mem_multicast_StoN_all(half* nram_dst, half* sram_src, int size)
{
  __asm__ volatile(
        "ld.multicast.nram.sram [%[nram_buf]], [%[sram_buf]], %[size], 0xF;\n\t"
        : : [nram_buf] "r"(nram_dst),
            [sram_buf] "r"(sram_src),
            [size] "r"(size));
}
__mlu_func__ void mlisa_mem_multicast_StoN_all_PR0(half* nram_dst, half* sram_src, int size)
{
  __asm__ volatile(
        "@%%pr0 ld.multicast.nram.sram [%[nram_buf]], [%[sram_buf]], %[size], 0xF;\n\t"
        : : [nram_buf] "r"(nram_dst),
            [sram_buf] "r"(sram_src),
            [size] "r"(size));
}
__mlu_func__ void mlisa_attr_multicast_tensor(half* nram_dst, half* sram_src, int size)
{
  __asm__ volatile(
        "ld.multicast.nram.sram [%[nram_buf]], [%[sram_buf]], %[size], 0x7;\n\t"
        : : [nram_buf] "r"(nram_dst),
            [sram_buf] "r"(sram_src),
            [size] "r"(size));
}
__mlu_func__ void mlisa_attr_multicast_tensor_PR0(half* nram_dst, half* sram_src, int size)
{
  __asm__ volatile(
        "@%%pr0 ld.multicast.nram.sram [%[nram_buf]], [%[sram_buf]], %[size], 0x7;\n\t"
        : : [nram_buf] "r"(nram_dst),
            [sram_buf] "r"(sram_src),
            [size] "r"(size));
}
__mlu_func__ void mlisa_attr_multicast_tensor_PR1(half* nram_dst, half* sram_src, int size)
{
  __asm__ volatile(
        "@%%pr1 ld.multicast.nram.sram [%[nram_buf]], [%[sram_buf]], %[size], 0x7;\n\t"
        : : [nram_buf] "r"(nram_dst),
            [sram_buf] "r"(sram_src),
            [size] "r"(size));
}
#endif // End of defined OPTIMIZE

#endif // MLISA_UTIL_H