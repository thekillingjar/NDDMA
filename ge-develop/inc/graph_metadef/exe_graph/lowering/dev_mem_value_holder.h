/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_DEV_MEM_VALUE_HOLDER_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_DEV_MEM_VALUE_HOLDER_H_

#include <cstddef>
#include "value_holder.h"
#include "graph/fast_graph/fast_node.h"
#include "exe_graph/lowering/value_holder.h"

namespace gert {
namespace bg {
constexpr int64_t kMainStream = 0;
class DevMemValueHolder;
using DevMemValueHolderPtr = std::shared_ptr<DevMemValueHolder>;
/**
 * Value holder with stream
 *
 */
class DevMemValueHolder : public ValueHolder {
 public:
  explicit DevMemValueHolder(const int64_t logic_stream_id) : logic_stream_id_(logic_stream_id){};

  DevMemValueHolder() = delete;
  DevMemValueHolder(const DevMemValueHolder &other) = delete;
  DevMemValueHolder &operator=(const DevMemValueHolder &other) = delete;
  ~DevMemValueHolder() override = default;

  ValueHolderPtr CreateMateFromNode(ge::FastNode *node, int32_t index, ValueHolderType type) override;

  static DevMemValueHolderPtr CreateSingleDataOutput(const ge::char_t *node_type,
                                                     const std::vector<ValueHolderPtr> &inputs,
                                                     int64_t logic_stream_id);

  static std::vector<DevMemValueHolderPtr> CreateDataOutput(const ge::char_t *node_type,
                                                            const std::vector<ValueHolderPtr> &inputs,
                                                            size_t out_count, int64_t logic_stream_id);

  static DevMemValueHolderPtr CreateConst(const void *data, size_t size, int64_t logic_stream_id,
                                          bool is_string = false);

  static DevMemValueHolderPtr CreateError(int64_t logic_stream_id, const char *fmt, va_list arg);
  static DevMemValueHolderPtr CreateError(int64_t logic_stream_id, const char *fmt, ...);

  int64_t GetLogicStream() const;

 private:
  int64_t logic_stream_id_{kMainStream};
};
}  // namespace bg
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_DEV_MEM_VALUE_HOLDER_H_
