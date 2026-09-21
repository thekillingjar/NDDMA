/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/debug/ge_log.h"
#include "attribute_group/attr_group_serialize.h"
#include "attribute_group/attr_group_serializer_registry.h"
#include "common/checker.h"

namespace ge {
graphStatus AttrGroupSerialize::SerializeAllAttr(proto::AttrGroups &attr_groups, const AttrStore &attr_store) {
  GE_ASSERT_GRAPH_SUCCESS(OtherGroupSerialize(attr_groups, attr_store));

  auto& id_2_ptr = attr_store.GetAttrsGroupPtr();
  for (const auto& ptr : id_2_ptr) {
    if (ptr.second != nullptr) {
      GE_ASSERT_GRAPH_SUCCESS(ptr.second->Serialize(*attr_groups.add_attr_group_def()));
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus AttrGroupSerialize::DeserializeAllAttr(const proto::AttrGroups &attr_group, AttrHolder *attr_holder) {
  GE_ASSERT_NOTNULL(attr_holder);
  auto &attr_store = attr_holder->MutableAttrMap();
  GE_ASSERT_GRAPH_SUCCESS(OtherGroupDeserialize(attr_group, attr_store));
  for (const auto &attr_group_def : attr_group.attr_group_def()) {
    auto deserializer = AttrGroupSerializerRegistry::GetInstance()
        .GetDeserializer(attr_group_def.attr_group_case());
    if (deserializer.impl == nullptr) {
      continue;
    }
    GE_ASSERT_GRAPH_SUCCESS(deserializer.impl->Deserialize(attr_group_def, attr_holder));
    attr_store.MutableAttrsGroupPtr()[deserializer.id] = std::move(deserializer.impl);
  }
  return ge::GRAPH_SUCCESS;
}
// todo: other group计划是需要再主线替换掉当前Ge IR上的的map<string, AttrDef> attr = 5字段
//       这个在分支上暂时不需要切换，等主线切换后再做替换，当前先做属性组的序列化和反序列化
graphStatus AttrGroupSerialize::OtherGroupSerialize(proto::AttrGroups &attr_groups, const AttrStore &attr_store) {
  (void)attr_store;
  (void)attr_groups;
  return GRAPH_SUCCESS;
}

graphStatus AttrGroupSerialize::OtherGroupDeserialize(const proto::AttrGroups &attr_groups, AttrStore &attr_store) {
  (void)attr_store;
  (void)attr_groups;
  return GRAPH_SUCCESS;
}
}