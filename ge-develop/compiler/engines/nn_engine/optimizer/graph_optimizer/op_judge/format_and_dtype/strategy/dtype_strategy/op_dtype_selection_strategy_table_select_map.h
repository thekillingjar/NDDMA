/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_TABLE_SELECT_MAP_H_
 #define AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_TABLE_SELECT_MAP_H_

 namespace fe {
 /*
 origin_dtype       match_dtype_vec
 hif8               hif8,fp16,fp32
 bf16               hif8,bf16,fp32
 fp16               hif8,fp16,fp32
 fp32               hif8,fp16,fp32
 other              keep origin_dtype
 */
 const std::map<ge::DataType, std::vector<ge::DataType>> dtype_match_map_white_hif8 = {
   {ge::DT_HIFLOAT8, {ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_FLOAT}},
   {ge::DT_BF16, {ge::DT_HIFLOAT8, ge::DT_BF16, ge::DT_FLOAT}},
   {ge::DT_FLOAT16, {ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_FLOAT}},
   {ge::DT_FLOAT, {ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_FLOAT}}
 };
 
 /*
 origin_dtype       match_dtype_vec
 bf16               fp16,fp32
 fp16               fp16,fp32
 fp32               fp32
 other              keep origin_dtype
 */
 const std::map<ge::DataType, std::vector<ge::DataType>> dtype_match_map_black_hif8 = {
   {ge::DT_BF16, {ge::DT_BF16, ge::DT_FLOAT}},
   {ge::DT_FLOAT16, {ge::DT_FLOAT16, ge::DT_FLOAT}},
   {ge::DT_FLOAT, {ge::DT_FLOAT}}
 };
 
 /*
 father_dtype        origin_dtype        match_dtype_vec
 hif8                other               hif8,fp16,fp32
                     bf16                hif8,bf16,fp32(need to replace fp16 with bf16, after get list)
 bf16                any                 bf16,fp32
 fp16                any                 fp16,fp32
 fp32                any                 fp32
 other               any                 keep father_dtype
 */
 const std::map<ge::DataType, std::vector<ge::DataType>> dtype_match_map_gray_hif8 = {
   {ge::DT_HIFLOAT8, {ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_FLOAT}},
   {ge::DT_BF16, {ge::DT_BF16, ge::DT_FLOAT}},
   {ge::DT_FLOAT16, {ge::DT_FLOAT16, ge::DT_FLOAT}},
   {ge::DT_FLOAT, {ge::DT_FLOAT}}
 };
 }  // namespace fe
 #endif  // AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_TABLE_SELECT_MAP_H_
 