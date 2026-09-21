/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/dev_mem_value_holder.h"

#include "common/checker.h"
#include "common/util/mem_utils.h"
#include "exe_graph/lowering/builtin_node_types.h"
#include "exe_graph/lowering/exe_graph_attrs.h"

namespace gert {
namespace bg {
DevMemValueHolderPtr DevMemValueHolder::CreateError(int64_t logic_stream_id, const char *fmt, va_list arg) {
  auto value_holder = ge::MakeShared<DevMemValueHolder>(logic_stream_id);
  GE_ASSERT_NOTNULL(value_holder);
  value_holder->SetErrorMsg(fmt, arg);
  return value_holder;
}

DevMemValueHolderPtr DevMemValueHolder::CreateError(int64_t logic_stream_id, const char *fmt, ...) {
  va_list arg;
  va_start(arg, fmt);
  auto holder = DevMemValueHolder::CreateError(logic_stream_id, fmt, arg);
  va_end(arg);
  return holder;
}

std::vector<DevMemValueHolderPtr> DevMemValueHolder::CreateDataOutput(const char *node_type,
                                                                      const std::vector<ValueHolderPtr> &inputs,
                                                                      size_t out_count, int64_t logic_stream_id) {
  auto node = CreateNode(node_type, inputs, out_count);
  if (node == nullptr) {
    return {out_count, nullptr};
  }
  return ValueHolder::CreateFromNode<DevMemValueHolder>(node, 0, out_count, logic_stream_id);
}

DevMemValueHolderPtr DevMemValueHolder::CreateSingleDataOutput(const char *node_type,
                                                               const std::vector<ValueHolderPtr> &inputs,
                                                               int64_t logic_stream_id) {
  auto node = CreateNode(node_type, inputs, 1U);
  if (node == nullptr) {
    return nullptr;
  }
  return ValueHolder::CreateFromNode<DevMemValueHolder>(node, 0, ValueHolderType::kOutput, logic_stream_id);
}


/**
 * @param data const数据
 * @param size const数据的长度
 * @param is_string 此const是否是个字符串, todo: 当前对string支持的不好
 * @return
 */
DevMemValueHolderPtr DevMemValueHolder::CreateConst(const void *data, size_t size, int64_t logic_stream_id,
                                                    bool is_string) {
  GE_ASSERT_NOTNULL(data);
  auto node = CreateNode(kConst, {}, 1);
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_SUCCESS(op_desc->SetAttr("is_string", ge::AnyValue::CreateFrom(is_string)));
  GE_ASSERT_TRUE(ge::AttrUtils::SetZeroCopyBytes(op_desc, kConstValue,
                                                 ge::Buffer::CopyFrom(ge::PtrToPtr<void, uint8_t>(data), size)));
  return CreateFromNode<DevMemValueHolder>(node, 0, ValueHolderType::kConst, logic_stream_id);
}

ValueHolderPtr DevMemValueHolder::CreateMateFromNode(ge::FastNode *node, int32_t index, ValueHolderType type) {
  return ValueHolder::CreateFromNode<DevMemValueHolder>(node, index, type, logic_stream_id_);
}

int64_t DevMemValueHolder::GetLogicStream() const { return logic_stream_id_; }
}  // namespace bg
}  // namespace gert
