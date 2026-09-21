/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __AUTOFUSE_REDUCE_API_CALL_BASE_H__
#define __AUTOFUSE_REDUCE_API_CALL_BASE_H__

#include <sstream>
#include "codegen_kernel.h"

namespace reduce_base {
using namespace codegen;

static std::map<std::string, std::pair<int, std::string>> reduce_type_map = {
  {"Min", {ReduceOpType::kMin, "Min"}},  {"Max", {ReduceOpType::kMax, "Max"}},
  {"Any", {ReduceOpType::kAny, "Max"}},  {"All", {ReduceOpType::kAll, "Min"}},
  {"Sum", {ReduceOpType::kSum, "Add"}},  {"Prod", {ReduceOpType::kProd, "Mul"}},
  {"Mean", {ReduceOpType::kMean, "Add"}}
};

void GetIsArAndPattern(const Tensor &y, bool &isAr, std::string &reduce_pattern);
void ReduceMergedSizeCodeGen(const TPipe &tpipe, std::stringstream &ss, const Tensor &src, const Tensor &dst,
                             bool is_tail = false);
bool IsNeedMultiReduce(const Tiler &tiler, const Tensor &input, const Tensor &output, ascir::AxisId axis_id);
void ReduceMeanCodeGen(std::string &dtype_name, const TPipe &tpipe, const Tensor &dst,
                       std::stringstream &ss);
void ReduceInitCodeGen(const Tensor &x, const Tensor &y, const int &type_value,
                       std::stringstream &ss, const TPipe &tpipe);
void ReduceDimACodeGen(const Tensor &x, const std::string &apiName, std::stringstream &ss);
}  // namespace codegen
#endif // __AUTOFUSE_REDUCE_API_CALL_BASE_H__