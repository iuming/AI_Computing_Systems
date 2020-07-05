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
#ifndef MLU100_PLUGIN_OPS_RANGEOP_PLUGIN_MACRO_H_

#define BLOCK 256
#define ALIGN_UP_TO(x, n) ((((x)-1) / (n) + 1) * (n))
#define ALIGN_DOWN_TO(x, n) ((((x)-1) / (n)) * (n))


// ***********************************mlu200***********************************
#if __BANG_ARCH__ >= 200

// [CHANGE200] datatype on cpu
#define VALUE_DTYPE float

// for range _ops.cpp
// [CHANGE200] mlu tensor datatype
#define MLUDtype CNML_DATA_FLOAT32
// [CHANGE200] cpu tensor datatype
#define CPUDtype CNML_DATA_FLOAT32

// Dtype      : int8_t int16_t int32_t half float
// DtypeIndex : 0      1       2       3    4
// [CHANGE200] datatype index on mlu
#define DtypeIndex 4
// [CHANGE200] datatype on mlu
#define Dtype float

// int8_t
#if DtypeIndex == 0

#define ALIGN_SIZE 64
#define LIMIT 32
#define NRAM_ELEM_CNT (480 * 1024)  // NRAM_ELEM_CNT*2 <= nram.bytes

// int16_t
#elif DtypeIndex == 1

#define ALIGN_SIZE 64
#define LIMIT 16
#define NRAM_ELEM_CNT (240 * 1024)  // NRAM_ELEM_CNT*2 <= nram.bytes

// int32_t
#elif DtypeIndex == 2

#define ALIGN_SIZE 64
#define LIMIT 8
#define NRAM_ELEM_CNT (120 * 1024)  // NRAM_ELEM_CNT*2 <= nram.bytes
// #define NRAMSETVALUE __nramset_half

// half
#elif DtypeIndex == 3

#define ALIGN_SIZE 64
#define LIMIT 16
#define NRAM_ELEM_CNT (240 * 1024)  // NRAM_ELEM_CNT*2 <= nram.bytes
// #define NRAMSETVALUE __nramset_half

// float
#elif DtypeIndex == 4
// mlu100 not support float32

#define ALIGN_SIZE 64
#define LIMIT 8
#define NRAM_ELEM_CNT (120 * 1024)  // NRAM_ELEM_CNT*4 <= nram.bytes
// #define NRAMSETVALUE __mlvm_memcpy_nramset_float

#endif


// ************************************mlu100***********************************
#else

// [CHANGE100] datatype on cpu
#define VALUE_DTYPE int32_t

// for range_ops.cpp
// [CHANGE100] mlu tensor datatype
#define MLUDtype CNML_DATA_INT32
// [CHANGE100] cpu tensor datatype
#define CPUDtype CNML_DATA_INT32

// Dtype      : int8_t int16_t int32_t half float
// DtypeIndex : 0      1       2       3    4

// [CHANGE100] datatype index on mlu
#define DtypeIndex 2
// [CHANGE100] datatype on mlu
#define Dtype int32_t
// int8_t
#if DtypeIndex == 0

#define ALIGN_SIZE 32
#define LIMIT 32
#define NRAM_ELEM_CNT (480 * 1024)  // NRAM_ELEM_CNT*2 <= nram.bytes

// int16_t
#elif DtypeIndex == 1

#define ALIGN_SIZE 16
#define LIMIT 16
#define NRAM_ELEM_CNT (240 * 1024)  // NRAM_ELEM_CNT*2 <= nram.bytes

// int32_t
#elif DtypeIndex == 2

#define ALIGN_SIZE 8
#define LIMIT 8
#define NRAM_ELEM_CNT (120 * 1024)  // NRAM_ELEM_CNT*2 <= nram.bytes
// #define NRAMSETVALUE __nramset_half

// half
#elif DtypeIndex == 3

#define ALIGN_SIZE 16
#define LIMIT 16
#define NRAM_ELEM_CNT (240 * 1024)  // NRAM_ELEM_CNT*2 <= nram.bytes
// #define NRAMSETVALUE __nramset_half

// float
#elif DtypeIndex == 4
// mlu100 not support float32

#define ALIGN_SIZE 8
#define LIMIT 8
#define NRAM_ELEM_CNT (120 * 1024)  // NRAM_ELEM_CNT*4 <= nram.bytes
// #define NRAMSETVALUE __mlvm_memcpy_nramset_float

#endif
#endif // __BANG_ARCH__

#define MLU100_PLUGIN_OPS_RANGEOP_PLUGIN_MACRO_H_
#endif  // MLU100_PLUGIN_OPS_RANGEOP_PLUGIN_MACRO_H_
