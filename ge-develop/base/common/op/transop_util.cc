/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/op/transop_util.h"

#include "common/types.h"
#include "graph/utils/type_utils.h"
#include "framework/common/debug/ge_log.h"

namespace {
constexpr int32_t kInvalidTransopDataIndex = -1;
constexpr int32_t kTransOpOutIndex = 0;
}  // namespace

namespace ge {
TransOpUtil::TransOpUtil() {
  transop_index_map_ = {{TRANSDATA, 0},   {TRANSPOSE, 0},  {TRANSPOSED, 0}, {RESHAPE, 0},
                        {REFORMAT, 0},    {CAST, 0},       {SQUEEZE, 0},    {SQUEEZEV2, 0},
                        {UNSQUEEZEV2, 0}, {EXPANDDIMS, 0}, {SQUEEZEV3, 0},  {UNSQUEEZEV3, 0}};

  for (size_t i = 0U; i < static_cast<size_t>(DataType::DT_MAX); ++i) {
    if (i == DT_BOOL) {
      continue;
    }
    (void)precision_loss_table_.Add(i, DT_BOOL, true);
  }

  // floating point to int cause precisoin loss
  for (const auto f_type : {DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_BF16}) {
    for (const auto i_type : {DT_INT32, DT_INT64}) {
      (void)precision_loss_table_.Add(f_type, i_type, true);
    }
  }

  // todo high precision to lower precision cause precision loss
  // transop_without_reshape_fusion_pass should check precison loss along trans road
  // currently if add high to lower precision loss may cause preformance problem
}

TransOpUtil &TransOpUtil::Instance() {
  static TransOpUtil inst;
  return inst;
}

bool TransOpUtil::IsTransOp(const NodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  return IsTransOp(node->GetType());
}

bool TransOpUtil::IsTransOp(const std::string &type) {
  return Instance().transop_index_map_.find(type) != Instance().transop_index_map_.end();
}

int32_t TransOpUtil::GetTransOpDataIndex(const NodePtr &node) {
  if (node == nullptr) {
    return kInvalidTransopDataIndex;
  }
  return GetTransOpDataIndex(node->GetType());
}

int32_t TransOpUtil::GetTransOpDataIndex(const std::string &type) {
  const auto it = Instance().transop_index_map_.find(type);
  if (it != Instance().transop_index_map_.end()) {
    return it->second;
  }
  return kInvalidTransopDataIndex;
}

bool TransOpUtil::IsPrecisionLoss(const ge::NodePtr &cast_node) {
  const auto idx = TransOpUtil::GetTransOpDataIndex(cast_node);
  const auto input_desc = cast_node->GetOpDesc()->GetInputDesc(static_cast<uint32_t> (idx));
  const auto output_desc = cast_node->GetOpDesc()->GetOutputDesc(static_cast<uint32_t> (kTransOpOutIndex));
  const auto src_dtype = input_desc.GetDataType();
  const auto dst_dtype = output_desc.GetDataType();
  // ge::DT_BOOL only supports 0/1 ?
  if (Instance().precision_loss_table_.Find(src_dtype, dst_dtype)) {
    GELOGW("Node %s transfer data type from %s to %s ,it will cause precision loss. ignore pass.",
           cast_node->GetName().c_str(), TypeUtils::DataTypeToSerialString(src_dtype).c_str(),
           TypeUtils::DataTypeToSerialString(dst_dtype).c_str());
    return true;
  }
  return false;
}

std::string TransOpUtil::TransopMapToString() {
  std::string buffer;
  for (auto &key : Instance().transop_index_map_) {
    buffer += key.first + " ";
  }
  return buffer;
}
}  // namespace ge
