/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_TILING_DATA_H_
#define ATT_TILING_DATA_H_
#include <stdint.h>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "exe_graph/runtime/tiling_context.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
  // definitions of BaseParams
    TILING_DATA_FIELD_DEF(uint32_t, bngs1Tb_size)
    TILING_DATA_FIELD_DEF(uint32_t, bngs1Tbt_size)
    TILING_DATA_FIELD_DEF(uint32_t, s1t_size)
    TILING_DATA_FIELD_DEF(uint32_t, s1tt2_size)
    TILING_DATA_FIELD_DEF(uint32_t, s1tt_size)
    TILING_DATA_FIELD_DEF(uint32_t, s2t_size)

  // definitions of HardWareParams
    TILING_DATA_FIELD_DEF(uint32_t, block_dim)
    TILING_DATA_FIELD_DEF(uint32_t, corenum)
    TILING_DATA_FIELD_DEF(uint32_t, hbm_size)
    TILING_DATA_FIELD_DEF(uint32_t, ub_size)

  // definitions of InputParams
    TILING_DATA_FIELD_DEF(uint32_t, B)
    TILING_DATA_FIELD_DEF(uint32_t, BL)
    TILING_DATA_FIELD_DEF(uint32_t, D)
    TILING_DATA_FIELD_DEF(uint32_t, G)
    TILING_DATA_FIELD_DEF(uint32_t, N)
    TILING_DATA_FIELD_DEF(uint32_t, S1)
    TILING_DATA_FIELD_DEF(uint32_t, S2)


  // definitions of AttrParams
  // 含义:属性参数定义
    TILING_DATA_FIELD_DEF(uint32_t, head_num)

  // definitions of CoreParams
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, s1tt2_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size % {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, s1tt2_loop_num)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_tail_size
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次最内轴元素
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_tail_size = ({轴的父轴}_tail_size % {轴}_size) == 0 ? {轴}_size :   //         ({轴的父轴}_tail_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, s1t_tail_tile_s1tt2_tail_size)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_loop_num
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次本轴
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //       2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_loop_num = Ceil({轴的父轴}_tail_size / {轴}_size)
  //                                                  = (({轴的父轴}_tail_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, s1t_tail_tile_s1tt2_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, s1tt_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size % {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, s1tt_loop_num)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_tail_size
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次最内轴元素
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_tail_size = ({轴的父轴}_tail_size % {轴}_size) == 0 ? {轴}_size :   //         ({轴的父轴}_tail_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, s1t_tail_tile_s1tt_tail_size)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_loop_num
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次本轴
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //       2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_loop_num = Ceil({轴的父轴}_tail_size / {轴}_size)
  //                                                  = (({轴的父轴}_tail_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, s1t_tail_tile_s1tt_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, s2t_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size % {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, s2t_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, bngs1Tbt_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size % {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, bngs1Tbt_loop_num)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_tail_size
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次最内轴元素
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_tail_size = ({轴的父轴}_tail_size % {轴}_size) == 0 ? {轴}_size :   //         ({轴的父轴}_tail_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, bngs1Tb_tail_tile_bngs1Tbt_tail_size)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_loop_num
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次本轴
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //       2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_loop_num = Ceil({轴的父轴}_tail_size / {轴}_size)
  //                                                  = (({轴的父轴}_tail_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, bngs1Tb_tail_tile_bngs1Tbt_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, bngs1Tb_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size % {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, bngs1Tb_loop_num)
  // 参数：{轴}_aligned_size
  // 含义：本轴对其后的大小
  // 约束：仅Ascend IR表达了对齐时，才会生成该参数
  // 计算公式：{轴}_aligned_size = ({轴}_size - 1) * ALIGN_SIZE / ALIGN_SIZE + ALIGN_SIZE
    TILING_DATA_FIELD_DEF(uint32_t, b_aligned_size)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, s1t_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size % {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, s1t_loop_num)
  // 参数：{轴}_aligned_size
  // 含义：本轴对其后的大小
  // 约束：仅Ascend IR表达了对齐时，才会生成该参数
  // 计算公式：{轴}_aligned_size = ({轴}_size - 1) * ALIGN_SIZE / ALIGN_SIZE + ALIGN_SIZE
    TILING_DATA_FIELD_DEF(uint32_t, d_aligned_size)
  // 参数：{轴}_aligned_size
  // 含义：本轴对其后的大小
  // 约束：仅Ascend IR表达了对齐时，才会生成该参数
  // 计算公式：{轴}_aligned_size = ({轴}_size - 1) * ALIGN_SIZE / ALIGN_SIZE + ALIGN_SIZE
    TILING_DATA_FIELD_DEF(uint32_t, s2_aligned_size)
  // 参数：{轴}_aligned_size
  // 含义：本轴对其后的大小
  // 约束：仅Ascend IR表达了对齐时，才会生成该参数
  // 计算公式：{轴}_aligned_size = ({轴}_size - 1) * ALIGN_SIZE / ALIGN_SIZE + ALIGN_SIZE
    TILING_DATA_FIELD_DEF(uint32_t, s1_aligned_size)
  // 参数：{轴}_aligned_size
  // 含义：本轴对其后的大小
  // 约束：仅Ascend IR表达了对齐时，才会生成该参数
  // 计算公式：{轴}_aligned_size = ({轴}_size - 1) * ALIGN_SIZE / ALIGN_SIZE + ALIGN_SIZE
    TILING_DATA_FIELD_DEF(uint32_t, g_aligned_size)
  // 参数：{轴}_aligned_size
  // 含义：本轴对其后的大小
  // 约束：仅Ascend IR表达了对齐时，才会生成该参数
  // 计算公式：{轴}_aligned_size = ({轴}_size - 1) * ALIGN_SIZE / ALIGN_SIZE + ALIGN_SIZE
    TILING_DATA_FIELD_DEF(uint32_t, n_aligned_size)

  // definitions of MemoryParams
  // 含义：输出的gm大小
    TILING_DATA_FIELD_DEF(uint32_t, output1_total_size)
  // 含义：单核输出的gm大小
    TILING_DATA_FIELD_DEF(uint32_t, output0_single_core_size)
  // 含义：输出的gm大小
    TILING_DATA_FIELD_DEF(uint32_t, output0_total_size)
  // 含义：单核输出的gm大小
    TILING_DATA_FIELD_DEF(uint32_t, output1_single_core_size)
  // 含义：GM的总大小
    TILING_DATA_FIELD_DEF(uint32_t, gm_size)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, Q3)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, BUF2)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, Q5)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, BUF0)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, BUF3)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, BUF1)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, BUF5)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, Q0)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, Q1)
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
    TILING_DATA_FIELD_DEF(uint32_t, BUF4)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, Q6)

  // definitions of TilingKeyParms
    TILING_DATA_FIELD_DEF(uint32_t, tiling_key)


END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(OpTest, TilingData)
bool GetTiling(TilingData &tiling_data, gert::TilingContext *context, int32_t tilingCaseId = -1);
ge::graphStatus GetCtxTiling(gert::TilingContext *context);
} // namespace optiling
#endif

