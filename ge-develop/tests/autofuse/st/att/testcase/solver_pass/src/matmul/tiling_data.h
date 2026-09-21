/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_TILING_DATA_MATMUL_H_
#define ATT_TILING_DATA_MATMUL_H_
#include <stdint.h>
#include <vector>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(MMTilingData)
  // definitions of BaseParams
    TILING_DATA_FIELD_DEF(uint32_t, basek_size)
    TILING_DATA_FIELD_DEF(uint32_t, basem_size)
    TILING_DATA_FIELD_DEF(uint32_t, basen_size)
    TILING_DATA_FIELD_DEF(uint32_t, stepka_size)
    TILING_DATA_FIELD_DEF(uint32_t, stepkb_size)
    TILING_DATA_FIELD_DEF(uint32_t, tilem_size)
    TILING_DATA_FIELD_DEF(uint32_t, tilen_size)

  // definitions of HardWareParams
    TILING_DATA_FIELD_DEF(uint32_t, l0a_size)
    TILING_DATA_FIELD_DEF(uint32_t, l0b_size)
    TILING_DATA_FIELD_DEF(uint32_t, l0c_size)
    TILING_DATA_FIELD_DEF(uint32_t, l1_size)
    TILING_DATA_FIELD_DEF(uint32_t, l2_size)
    TILING_DATA_FIELD_DEF(uint32_t, workspaceSize)

  // definitions of InputParams
    TILING_DATA_FIELD_DEF(uint32_t, block_dim)
    TILING_DATA_FIELD_DEF(uint32_t, k_size)
    TILING_DATA_FIELD_DEF(uint32_t, m_size)
    TILING_DATA_FIELD_DEF(uint32_t, n_size)


  // definitions of CoreParams
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, basem_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, basem_loop_num)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_tail_size
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次最内轴元素
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_tail_size = ({轴的父轴}_tail_size % {轴}_size) == 0 ? {轴}_size : 
  //         ({轴的父轴}_tail_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, tilem_tail_tile_basem_tail_size)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_loop_num
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次本轴
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_loop_num = Ceil({轴的父轴}_tail_size / {轴}_size)
  //                                                  = (({轴的父轴}_tail_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, tilem_tail_tile_basem_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, tilem_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, tilem_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, basen_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, basen_loop_num)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_tail_size
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次最内轴元素
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_tail_size = ({轴的父轴}_tail_size % {轴}_size) == 0 ? {轴}_size : 
  //         ({轴的父轴}_tail_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, tilen_tail_tile_basen_tail_size)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_loop_num
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次本轴
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_loop_num = Ceil({轴的父轴}_tail_size / {轴}_size)
  //                                                  = (({轴的父轴}_tail_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, tilen_tail_tile_basen_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, tilen_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, tilen_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, basek_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, basek_loop_num)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_tail_size
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次最内轴元素
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_tail_size = ({轴的父轴}_tail_size % {轴}_size) == 0 ? {轴}_size : 
  //         ({轴的父轴}_tail_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, stepkb_tail_tile_basek_tail_size)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_loop_num
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次本轴
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_loop_num = Ceil({轴的父轴}_tail_size / {轴}_size)
  //                                                  = (({轴的父轴}_tail_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, stepkb_tail_tile_basek_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, stepkb_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, stepkb_loop_num)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_tail_size
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次最内轴元素
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_tail_size = ({轴的父轴}_tail_size % {轴}_size) == 0 ? {轴}_size : 
  //         ({轴的父轴}_tail_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, stepka_tail_tile_stepkb_tail_size)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_loop_num
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次本轴
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_loop_num = Ceil({轴的父轴}_tail_size / {轴}_size)
  //                                                  = (({轴的父轴}_tail_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, stepka_tail_tile_stepkb_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, stepka_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, stepka_loop_num)

  // definitions of MemoryParams
  // 含义：单核输出的gm大小
    TILING_DATA_FIELD_DEF(uint32_t, output0_single_core_size)
  // 含义：输出的gm大小
    TILING_DATA_FIELD_DEF(uint32_t, output0_total_size)
  // 含义：GM的总大小
    TILING_DATA_FIELD_DEF(uint32_t, gm_size)
  // 含义：Tensor所占用内存大小
    TILING_DATA_FIELD_DEF(uint32_t, MATMUL_OUTPUT1)
  // 含义：Tbuf或者Queue使用的内存大小
  // 约束：定义了Tbuf或者Queue则会生成该参数
  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
    TILING_DATA_FIELD_DEF(uint32_t, Q1)

  // definitions of TilingKeyParms
    TILING_DATA_FIELD_DEF(uint32_t, tiling_key)


END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(Matmul, MMTilingData)
bool GetTiling(MMTilingData &tiling_data, int32_t tilingCaseId = -1);
} // namespace optiling
#endif

