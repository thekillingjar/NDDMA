/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensor_desc_serializer.h"

#include "framework/common/debug/ge_log.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/attr_utils.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/ge_tensor.h"

namespace ge {
graphStatus TensorDescSerializer::Serialize(const AnyValue &av, proto::AttrDef &def) {
  GeTensorDesc tensor_desc;
  const graphStatus ret = av.GetValue(tensor_desc);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get tensor_desc attr.");
    return GRAPH_FAILED;
  }
  GeTensorSerializeUtils::GeTensorDescAsProto(tensor_desc, def.mutable_td());
  return GRAPH_SUCCESS;
}

graphStatus TensorDescSerializer::Deserialize(const proto::AttrDef &def, AnyValue &av) {
  GeTensorDesc tensor_desc;
  const proto::TensorDescriptor &descriptor = def.td();
  GeTensorSerializeUtils::AssembleGeTensorDescFromProto(&descriptor, tensor_desc);
  return av.SetValue(std::move(tensor_desc));
}

REG_GEIR_SERIALIZER(tensor_desc_serialzier, TensorDescSerializer, GetTypeId<GeTensorDesc>(), proto::AttrDef::kTd);
}  // namespace ge
