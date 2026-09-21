/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_CUSTOM_PASS_CONTEXT_IMPL_H_
#define METADEF_CXX_CUSTOM_PASS_CONTEXT_IMPL_H_
#include "common/checker.h"
#include "graph/utils/node_adapter.h"
#include "graph/ascend_string.h"
#include "graph/ge_context.h"
namespace ge {
class StreamPassContextImpl {
 public:
  explicit StreamPassContextImpl(int64_t current_max_stream_id) : current_stream_id_(current_max_stream_id) {
  }

  ~StreamPassContextImpl() = default;

  int64_t GetCurrentMaxStreamId() const {
    return current_stream_id_;
  }

  int64_t AllocateNextStreamId() {
    return ++current_stream_id_;
  }

  graphStatus SetStreamId(const GNode &node, int64_t stream_id) const {
    if (stream_id < 0) {
      GELOGE(PARAM_INVALID, "Failed to set unassigned stream id %ld, stream id should be positive integer.", stream_id);
      return FAILED;
    }
    if (stream_id > current_stream_id_) {
      GELOGE(PARAM_INVALID, "Failed to set unassigned stream id %ld, current_stream_id is %ld.", stream_id,
             current_stream_id_);
      return FAILED;
    }
    const auto compute_node = NodeAdapter::GNode2Node(node);
    GE_ASSERT_NOTNULL(compute_node);
    const auto *op_desc = compute_node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    GELOGI("Set node %s stream id from %ld to %ld by custom pass", op_desc->GetNamePtr(), op_desc->GetStreamId(),
           stream_id);
    compute_node->GetOpDesc()->SetStreamId(stream_id);
    return GRAPH_SUCCESS;
  }

  static graphStatus GetStreamId(const GNode &node, int64_t &stream_id) {
    const auto compute_node = NodeAdapter::GNode2Node(node);
    GE_ASSERT_NOTNULL(compute_node);
    GE_ASSERT_NOTNULL(compute_node->GetOpDesc());
    stream_id = compute_node->GetOpDesc()->GetStreamId();
    return GRAPH_SUCCESS;
  }
 private:
    int64_t current_stream_id_ = 0L;
};

class CustomPassContextImpl {
 public:
  CustomPassContextImpl() = default;
  ~CustomPassContextImpl() = default;

  void SetErrorMessage(const AscendString &error_message) {
    error_message_ = error_message;
  }

  void SetPassName(const AscendString &pass_name) {
    pass_name_ = pass_name;
  }

  AscendString GetErrorMessage() const {
    return error_message_;
  }

  AscendString GetPassName() const {
    return pass_name_;
  }

  graphStatus GetOptionValue(const AscendString &option_key, AscendString &option_value) const {
    std::string option_value_str;
    const auto ret = GetContext().GetOption(option_key.GetString(), option_value_str);
    option_value = AscendString(option_value_str.c_str());
    return ret;
  }

 private:
  AscendString error_message_{""};
  AscendString pass_name_{""};
};
}  // namespace ge

#endif  // METADEF_CXX_CUSTOM_PASS_CONTEXT_IMPL_H_
