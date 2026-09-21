/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_serializer.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/detail/model_serialize_imp.h"

namespace ge {
graphStatus GraphSerializer::Serialize(const AnyValue &av, proto::AttrDef &def) {
  const auto graph = def.mutable_g();
  GE_CHECK_NOTNULL(graph);

  if (av.GetValue(*graph) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Serialize graph failed");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

graphStatus GraphSerializer::Deserialize(const proto::AttrDef &def, AnyValue &av) {
  return av.SetValue(def.g());
}

REG_GEIR_SERIALIZER(graph_serializer, GraphSerializer, GetTypeId<proto::GraphDef>(), proto::AttrDef::kG);
}  // namespace ge
