/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_type_serializer.h"
#include "proto/ge_ir.pb.h"
#include "framework/common/debug/ge_log.h"
#include "graph/types.h"

namespace ge {
graphStatus DataTypeSerializer::Serialize(const AnyValue &av, proto::AttrDef &def) {
  ge::DataType value;
  const graphStatus ret = av.GetValue(value);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get datatype attr.");
    return GRAPH_FAILED;
  }
  def.set_dt(static_cast<proto::DataType>(value));
  return GRAPH_SUCCESS;
}

graphStatus DataTypeSerializer::Deserialize(const proto::AttrDef &def, AnyValue &av) {
  return av.SetValue(static_cast<DataType>(def.dt()));
}

REG_GEIR_SERIALIZER(data_type_serializer, DataTypeSerializer, GetTypeId<ge::DataType>(), proto::AttrDef::kDt);
}  // namespace ge
