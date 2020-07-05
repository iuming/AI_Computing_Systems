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
#include <iostream>
#include <stdio.h>
#include <cmath>
#include "cnplugin.h"
#include "plugin_onehot_cpu.h"

void onehot_cpu(cnmlPluginOneHotOpParam_t param,
                int* indices,
                float* dst){
    int N = param->N;
    int H = param->H;
    int W = param->W;
    int C = param->C;
    int depth = param->depth;
    float onvalue = param->onvalue;
    float offvalue = param->offvalue;
    int axis = param->axis;
	
	int size = N*C*depth*H*W;
	for(int i = 0; i<size;i++)
		dst[i] = offvalue;
	
	if(axis==-1) {
	  for(int16_t i=0;i<N;i++) {
		for(int16_t j=0;j<H;j++){
		  for(int16_t k=0;k<W;k++) {
			for(int16_t m=0;m<C;m++) {
			  if(indices[i*W*H*C + j*W*C + k*C + m]>=0 && indices[i*W*H*C + j*W*C + k*C + m]<depth)
			    dst[i*H*W*C*depth + j*W*C*depth + k*C*depth + m*depth + indices[i*W*H*C + j*W*C + k*C + m]]=onvalue;
			}
		  }
		}
	  }
	  return ;
    }
	if (axis == 3){
      for(int16_t i=0;i<N;i++) {
		for(int16_t j=0;j<H;j++){
		  for(int16_t k=0;k<W;k++) {
			for(int16_t m=0;m<C;m++) {
			  if(indices[i*W*H*C + j*W*C + k*C + m]>=0 && indices[i*W*H*C + j*W*C + k*C + m]<depth){
				dst[i*H*W*C*depth + j*W*C*depth + k*C*depth + C*indices[i*W*H*C + j*W*C + k*C + m] + m]=onvalue;
			  }
			}
		  }
		}
	  }
	  return ;
    }
	if(axis == 2) {
      for(int16_t i=0;i<N;i++) {
		for(int16_t j=0;j<H;j++){
		  for(int16_t k=0;k<W;k++) {
			for(int16_t m=0;m<C;m++) {
			  if(indices[i*W*H*C + j*W*C + k*C + m]>=0 && indices[i*W*H*C + j*W*C + k*C + m]<depth){
				dst[i*H*W*C*depth + j*W*C*depth + W*C*indices[i*W*H*C + j*W*C + k*C + m] + C*k + m]=onvalue;
			  }
			}
		  }
		}
	  }
	  return ;
    }
    if(axis == 1) {
      for(int16_t i=0;i<N;i++) {
		for(int16_t j=0;j<H;j++){
		  for(int16_t k=0;k<W;k++) {
			for(int16_t m=0;m<C;m++) {
			  if(indices[i*W*H*C + j*W*C + k*C + m]>=0 && indices[i*W*H*C + j*W*C + k*C + m]<depth){
				dst[i*H*W*C*depth + H*W*C*indices[i*W*H*C + j*W*C + k*C + m] + W*C*j + C*k + m]=onvalue;
			  }
			}
		  }
	    }
	  }
	  return ;
    }
    if(axis == 0) {
      for(int16_t i=0;i<N;i++) {
		for(int16_t j=0;j<H;j++){
		  for(int16_t k=0;k<W;k++) {
			for(int16_t m=0;m<C;m++) {
			  if(indices[i*W*H*C + j*W*C + k*C + m]>=0 && indices[i*W*H*C + j*W*C + k*C + m]<depth){				 	
				  dst[H*W*N*C*indices[i*W*H*C + j*W*C + k*C + m] + H*W*C*i + W*C*j + C*k + m]=onvalue;
			  }
			}
		  }
		}
	  }
	  return ;
    }
}