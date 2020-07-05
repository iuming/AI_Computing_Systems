/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc.
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/
#ifndef _CROP_AND_RESIZE_Kernel_HPP_
#define _CROP_AND_RESIZE_Kernel_HPP_

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* __cplusplus */

// 16 for fp16, 32 for int8, 64 for mlu200
#if __BANG_ARCH__ >= 200
#define PAD_SIZE 64
#else
#define PAD_SIZE 16
#endif
//#define PAD_SIZE 64

void PluginCropFeatureAndResizeKernel(uint16_t* src_gdram, uint16_t* boxes_gdram,
                         uint16_t* box_index_gdram, uint16_t* dst_gdram,
                         int batchNum, int depth, int image_height, int image_width,
                         int crop_height, int crop_width, int box_number,
                         int inputDataType, int outputDataType, int input2half,
                         int output2uint, int pad_size);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */

#endif  // _CROP_AND_RESIZE_Kernel_HPP_
