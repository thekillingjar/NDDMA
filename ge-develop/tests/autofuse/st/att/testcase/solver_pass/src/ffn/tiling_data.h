/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_TILING_DATA_FFN_H_
#define ATT_TILING_DATA_FFN_H_
#include <stdint.h>
#include <vector>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(FFNTilingData)
  // definitions of BaseParams
    TILING_DATA_FIELD_DEF(uint32_t, base_m1)
    TILING_DATA_FIELD_DEF(uint32_t, base_m2)
    TILING_DATA_FIELD_DEF(uint32_t, base_n1)
    TILING_DATA_FIELD_DEF(uint32_t, base_n2)
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_m1)
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_m2)
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_n1)
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_n2)
    TILING_DATA_FIELD_DEF(uint32_t, ub_m)

  // definitions of HardWareParams
    TILING_DATA_FIELD_DEF(uint32_t, block_dim)
    TILING_DATA_FIELD_DEF(uint32_t, btbuf_size)
    TILING_DATA_FIELD_DEF(uint32_t, l0c_size)
    TILING_DATA_FIELD_DEF(uint32_t, ub_size)
    TILING_DATA_FIELD_DEF(uint32_t, workspaceSize)

  // definitions of InputParams
    TILING_DATA_FIELD_DEF(uint32_t, K1)
    TILING_DATA_FIELD_DEF(uint32_t, N1)
    TILING_DATA_FIELD_DEF(uint32_t, N2)
    TILING_DATA_FIELD_DEF(uint32_t, maxTokens)


  // definitions of CoreParams
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, ub_m_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, ub_m_loop_num)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_tail_size
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次最内轴元素
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_tail_size = ({轴的父轴}_tail_size % {轴}_size) == 0 ? {轴}_size : 
  //         ({轴的父轴}_tail_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, base_m1_tail_tile_ub_m_tail_size)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_loop_num
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次本轴
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_loop_num = Ceil({轴的父轴}_tail_size / {轴}_size)
  //                                                  = (({轴的父轴}_tail_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, base_m1_tail_tile_ub_m_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, base_m1_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, base_m1_loop_num)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_tail_size
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次最内轴元素
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_tail_size = ({轴的父轴}_tail_size % {轴}_size) == 0 ? {轴}_size : 
  //         ({轴的父轴}_tail_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_m1_tail_tile_base_m1_tail_size)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_loop_num
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次本轴
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_loop_num = Ceil({轴的父轴}_tail_size / {轴}_size)
  //                                                  = (({轴的父轴}_tail_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_m1_tail_tile_base_m1_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_m1_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_m1_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, base_m2_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, base_m2_loop_num)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_tail_size
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次最内轴元素
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_tail_size = ({轴的父轴}_tail_size % {轴}_size) == 0 ? {轴}_size : 
  //         ({轴的父轴}_tail_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_m2_tail_tile_base_m2_tail_size)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_loop_num
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次本轴
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_loop_num = Ceil({轴的父轴}_tail_size / {轴}_size)
  //                                                  = (({轴的父轴}_tail_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_m2_tail_tile_base_m2_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_m2_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_m2_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, base_n1_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, base_n1_loop_num)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_tail_size
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次最内轴元素
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_tail_size = ({轴的父轴}_tail_size % {轴}_size) == 0 ? {轴}_size : 
  //         ({轴的父轴}_tail_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_n1_tail_tile_base_n1_tail_size)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_loop_num
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次本轴
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_loop_num = Ceil({轴的父轴}_tail_size / {轴}_size)
  //                                                  = (({轴的父轴}_tail_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_n1_tail_tile_base_n1_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_n1_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_n1_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, base_n2_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, base_n2_loop_num)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_tail_size
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次最内轴元素
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_tail_size = ({轴的父轴}_tail_size % {轴}_size) == 0 ? {轴}_size : 
  //         ({轴的父轴}_tail_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_n2_tail_tile_base_n2_tail_size)
  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_loop_num
  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次本轴
  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；
  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);
  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_loop_num = Ceil({轴的父轴}_tail_size / {轴}_size)
  //                                                  = (({轴的父轴}_tail_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_n2_tail_tile_base_n2_loop_num)
  // 参数：{轴}_tail_size
  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_n2_tail_size)
  // 参数：{轴}_loop_num
  // 含义：本轴的父轴需要循环多少次本轴
  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
    TILING_DATA_FIELD_DEF(uint32_t, singlecore_n2_loop_num)

  // definitions of MemoryParams
  // 含义：单核输出的gm大小
    TILING_DATA_FIELD_DEF(uint32_t, output0_single_core_size)
  // 含义：输出的gm大小
    TILING_DATA_FIELD_DEF(uint32_t, output0_total_size)
  // 含义：GM的总大小
    TILING_DATA_FIELD_DEF(uint32_t, gm_size)

  // definitions of TilingKeyParms
    TILING_DATA_FIELD_DEF(uint32_t, tiling_key)


END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FFN, FFNTilingData)
bool GetTiling(FFNTilingData &tiling_data, int32_t tilingCaseId = -1);
} // namespace optiling
#endif

