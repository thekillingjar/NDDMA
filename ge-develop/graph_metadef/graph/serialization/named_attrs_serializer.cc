/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "named_attrs_serializer.h"
#include "framework/common/debug/ge_log.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/attr_utils.h"

namespace ge {
graphStatus NamedAttrsSerializer::Serialize(const AnyValue &av, proto::AttrDef &def) {
  ge::NamedAttrs named_attrs;
  const graphStatus ret = av.GetValue(named_attrs);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get named attrs.");
    return GRAPH_FAILED;
  }
  auto func = def.mutable_func();

  return Serialize(named_attrs, func);
}

graphStatus NamedAttrsSerializer::Serialize(const ge::NamedAttrs &named_attr, proto::NamedAttrs* proto_attr) const {
  GE_CHECK_NOTNULL(proto_attr);
  proto_attr->set_name(named_attr.GetName().c_str());
  const auto mutable_attr = proto_attr->mutable_attr();
  GE_CHECK_NOTNULL(mutable_attr);

  const auto attrs = AttrUtils::GetAllAttrs(named_attr);
  for (const auto &attr : attrs) {
    const AnyValue attr_value = attr.second;
    const auto serializer = AttrSerializerRegistry::GetInstance().GetSerializer(attr_value.GetValueTypeId());
    GE_CHECK_NOTNULL(serializer);
    proto::AttrDef attr_def;
    if (serializer->Serialize(attr_value, attr_def) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Attr serialized failed, name:[%s].", attr.first.c_str());
      return FAILED;
    }
    (*mutable_attr)[attr.first] = attr_def;
  }
  return GRAPH_SUCCESS;
}

graphStatus NamedAttrsSerializer::Deserialize(const proto::AttrDef &def, AnyValue &av) {
  ge::NamedAttrs value;
  if (Deserialize(def.func(), value) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  return av.SetValue(std::move(value));
}

graphStatus NamedAttrsSerializer::Deserialize(const proto::NamedAttrs &proto_attr, ge::NamedAttrs &named_attrs) const {
  named_attrs.SetName(proto_attr.name());
  const auto proto_attr_map = proto_attr.attr();
  for (const auto &sub_proto_attr : proto_attr_map) {
    const auto deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(sub_proto_attr.second.value_case());
    GE_CHECK_NOTNULL(deserializer);
    AnyValue attr_value;
    if (deserializer->Deserialize(sub_proto_attr.second, attr_value) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Attr deserialized failed, name:[%s].", sub_proto_attr.first.c_str());
      return FAILED;
    }
    if (named_attrs.SetAttr(sub_proto_attr.first, attr_value) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "NamedAttrs [%s] set attr [%s] failed.",
             named_attrs.GetName().c_str(), sub_proto_attr.first.c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

REG_GEIR_SERIALIZER(named_attr_serializer, NamedAttrsSerializer, GetTypeId<ge::NamedAttrs>(), proto::AttrDef::kFunc);
}  // namespace ge
