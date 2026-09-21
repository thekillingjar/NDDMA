/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_COMPATIBLE_UTILS_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_COMPATIBLE_UTILS_H_
#include <vector>
#include "graph/node.h"
#include "graph/buffer.h"
#include "graph/detail/model_serialize_imp.h"
#include "proto/ge_ir.pb.h"
#include "exe_graph/lowering/value_holder.h"
#include "common/debug/ge_log.h"

namespace gert {
namespace bg {
class CompatibleUtils {
 public:
  static ge::Buffer SerializeNode(const ge::NodePtr &node) {
    ge::ModelSerializeImp model_serialize_imp;
    ge::proto::OpDef op_def_proto;
    if (!model_serialize_imp.SerializeNode(node, &op_def_proto)) {
      GELOGW("Serialize node[name:%s] not success.", node->GetName().c_str());
      return ge::Buffer();
    }
    ge::Buffer buffer(op_def_proto.ByteSizeLong());
    const auto ret = op_def_proto.SerializeToArray(buffer.GetData(), buffer.GetSize());
    if (!ret) {
      GELOGW("Fail to serialize node %s to array.", node->GetName().c_str());
      return ge::Buffer();
    }
    return buffer;
  }

  static std::vector<ValueHolderPtr> BuildOpDescBufferConst(const ge::NodePtr &node) {
    auto buffer = SerializeNode(node);
    auto buffer_size = buffer.GetSize();
    if (buffer_size == 0) {
      return {};
    }
    auto op_buffer = ValueHolder::CreateConst(buffer.GetData(), buffer_size);
    auto op_buffer_size = ValueHolder::CreateConst(&buffer_size, sizeof(buffer_size));
    return {op_buffer, op_buffer_size};
  }
};
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_COMPATIBLE_UTILS_H_
