/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_TILING_DATA_ADDLAYERNORM_H_
#define ATT_TILING_DATA_ADDLAYERNORM_H_
#include <stdint.h>
#include <vector>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(graph_normalTilingData)
  // definitions of BaseParams
    TILING_DATA_FIELD_DEF(uint32_t, nbo_size)
    TILING_DATA_FIELD_DEF(uint32_t, nio_size)
    TILING_DATA_FIELD_DEF(uint32_t, sbo_size)
    TILING_DATA_FIELD_DEF(uint32_t, sio_size)
    TILING_DATA_FIELD_DEF(uint32_t, wbo_size)
    TILING_DATA_FIELD_DEF(uint32_t, wio_size)

  // definitions of HardWareParams
    TILING_DATA_FIELD_DEF(uint32_t, block_dim)
    TILING_DATA_FIELD_DEF(uint32_t, ub_size)
    TILING_DATA_FIELD_DEF(uint32_t, workspaceSize)

  // definitions of InputParams
    TILING_DATA_FIELD_DEF(uint32_t, A)
    TILING_DATA_FIELD_DEF(uint32_t, BL)
    TILING_DATA_FIELD_DEF(uint32_t, R)


  // definitions of CoreParams
  // 参数：{轴}_aligned_size
  // 含义：本轴对齐后的大小
  // 约束：仅Ascend IR表达了对齐时，才会生成该参数
  // 计算公式：{轴}_aligned_size = ({轴}_size - 1) / ALIGN_SIZE * ALIGN_SIZE + ALIGN_SIZE
    TILING_DATA_FIELD_DEF(uint32_t, A_aligned_size)
  // 参数：{轴}_aligned_size
  // 含义：本轴对齐后的大小
  // 约束：仅Ascend IR表达了对齐时，才会生成该参数
  // 计算公式：{轴}_aligned_size = ({轴}_size - 1) / ALIGN_SIZE * ALIGN_SIZE + ALIGN_SIZE
    TILING_DATA_FIELD_DEF(uint32_t, R_aligned_size)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, nio_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, nio_loop_num)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_tail_size
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次最内轴元素
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_tail_size = ({轴的父轴}_tail_size % {轴}_size) == 0 ? {轴}_size : 
  //         ({轴的父轴}_tail_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, nbo_tail_tile_nio_tail_size)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_loop_num
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次本轴
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_loop_num = Ceil({轴的父轴}_tail_size / {轴}_size)
  //                                                  = (({轴的父轴}_tail_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, nbo_tail_tile_nio_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, nbo_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, nbo_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, sbo_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, sbo_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, sio_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, sio_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, wbo_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, wbo_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, wio_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, wio_loop_num)

  // definitions of MemoryParams
  // 含义：输出的gm大小
    TILING_DATA_FIELD_DEF(uint32_t, output3_total_size)
  // 含义：单核输出的gm大小
    TILING_DATA_FIELD_DEF(uint32_t, output2_single_core_size)
  // 含义：输出的gm大小
    TILING_DATA_FIELD_DEF(uint32_t, output2_total_size)
  // 含义：输出的gm大小
    TILING_DATA_FIELD_DEF(uint32_t, output1_total_size)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, Q0)
  // 含义：单核输出的gm大小
    TILING_DATA_FIELD_DEF(uint32_t, output1_single_core_size)
  // 含义：GM的总大小
    TILING_DATA_FIELD_DEF(uint32_t, gm_size)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, Q1)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, Q3)
  // 含义：单核输出的gm大小
    TILING_DATA_FIELD_DEF(uint32_t, output3_single_core_size)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, Q8)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, Q2)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, Q4)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, Q5)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, Q9)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, Q6)
  // 含义：输出的gm大小
    TILING_DATA_FIELD_DEF(uint32_t, output0_total_size)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, Q7)
  // 含义：单核输出的gm大小
    TILING_DATA_FIELD_DEF(uint32_t, output0_single_core_size)

  // definitions of TilingKeyParms
    TILING_DATA_FIELD_DEF(uint32_t, tiling_key)


END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(AddLayerNorm, graph_normalTilingData)
bool GetTiling(graph_normalTilingData &tiling_data, int32_t tilingCaseId = -1);
} // namespace optiling
#endif

