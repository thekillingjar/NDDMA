/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/attribute_group/attr_group_serializer_registry.h"
#include "framework/common/debug/ge_log.h"
namespace ge {

AttrGroupSerializerRegistry &AttrGroupSerializerRegistry::GetInstance() {
  static AttrGroupSerializerRegistry instance;
  return instance;
}

void AttrGroupSerializerRegistry::RegisterAttrGroupSerialize(const AttrGroupSerializeBuilder &builder,
                                                             const TypeId obj_type,
                                                             const proto::AttrGroupDef::AttrGroupCase proto_type) {
  const std::lock_guard<std::mutex> lck_guard(mutex_);
  std::unique_ptr<AttrGroupsBase> serializer = builder();
  const auto ptr = serializer.get();
  if (ptr == nullptr) {
    GELOGE(FAILED, "SerializerBuilder is invalid.");
    return;
  }
  if (serializer_builder_map_.count(obj_type) > 0U) {
    GELOGW("Serializer %s for type %s has been registered",
           typeid(*ptr).name(), HashedPointer<void>(obj_type).ToString().c_str());
    return;
  }
  GELOGD("Serializer %s for type %s register successfully",
         typeid(*ptr).name(), HashedPointer<void>(obj_type).ToString().c_str());
  serializer_builder_map_[obj_type] = builder;
  deserializer_builder_map_[proto_type] = std::make_pair(builder, obj_type);
}

std::unique_ptr<AttrGroupsBase> AttrGroupSerializerRegistry::GetSerializer(const TypeId obj_type) {
  const auto iter = serializer_builder_map_.find(obj_type);
  if (iter == serializer_builder_map_.cend()) {
    GELOGW("Serializer for type %s has not been registered", HashedPointer<void>(obj_type).ToString().c_str());
    return nullptr;
  }
  return iter->second();
}

AttrGroupDeserializer AttrGroupSerializerRegistry::GetDeserializer(const proto::AttrGroupDef::AttrGroupCase proto_type) {
  const auto iter =
      deserializer_builder_map_.find(proto_type);
  if (iter == deserializer_builder_map_.cend()) {
    GELOGW("Deserializer for type [%d] has not been registered", static_cast<int32_t>(proto_type));
    return AttrGroupDeserializer(nullptr, nullptr);
  }
  return AttrGroupDeserializer(iter->second.first(), iter->second.second);
}

AttrGroupSerializerRegister::AttrGroupSerializerRegister(const AttrGroupSerializeBuilder builder,
                                                         const TypeId obj_type,
                                                         const proto::AttrGroupDef::AttrGroupCase proto_type) noexcept {
  if (builder == nullptr) {
    GELOGE(FAILED, "SerializerBuilder is nullptr.");
    return;
  }
  AttrGroupSerializerRegistry::GetInstance().RegisterAttrGroupSerialize(builder, obj_type, proto_type);
}
}