/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "buffer_serializer.h"
#include <string>
#include "proto/ge_ir.pb.h"
#include "graph/buffer.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
graphStatus BufferSerializer::Serialize(const AnyValue &av, proto::AttrDef &def) {
  Buffer val;
  const graphStatus ret = av.GetValue(val);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get buffer attr.");
    return GRAPH_FAILED;
  }
  if ((val.data()!= nullptr) && (val.size() > 0U)) {
    def.set_bt(val.GetData(), val.GetSize());
  }
  return GRAPH_SUCCESS;
}

graphStatus BufferSerializer::Deserialize(const proto::AttrDef &def, AnyValue &av) {
  Buffer buffer = Buffer::CopyFrom(reinterpret_cast<const uint8_t *>(def.bt().data()), def.bt().size());
  return av.SetValue(std::move(buffer));
}

REG_GEIR_SERIALIZER(buffer_serializer, BufferSerializer, GetTypeId<ge::Buffer>(), proto::AttrDef::kBt);
}  // namespace ge
