/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_GRAPH_ATTRIBUTE_GROUP_ATTR_GROUP_SERIALIZER_REGISTRY_H_
#define METADEF_CXX_INC_GRAPH_ATTRIBUTE_GROUP_ATTR_GROUP_SERIALIZER_REGISTRY_H_
#include <functional>
#include <memory>
#include <mutex>
#include <map>

#include "graph/attribute_group/attr_group_base.h"
#include "proto/ge_ir.pb.h"

#define REG_ATTR_GROUP_SERIALIZER(serializer_name, cls, obj_type, bin_type)                              \
    REG_ATTR_GROUP_SERIALIZER_BUILDER_UNIQ_HELPER(serializer_name, __COUNTER__, cls, obj_type, bin_type)

#define REG_ATTR_GROUP_SERIALIZER_BUILDER_UNIQ_HELPER(name, ctr, cls, obj_type, bin_type)                \
    REG_ATTR_GROUP_SERIALIZER_BUILDER_UNIQ(name, ctr, cls, obj_type, bin_type)

#define REG_ATTR_GROUP_SERIALIZER_BUILDER_UNIQ(name, ctr, cls, obj_type, bin_type)               \
  static ::ge::AttrGroupSerializerRegister register_serialize_##name##ctr                      \
      __attribute__((unused)) =                                                            \
          ::ge::AttrGroupSerializerRegister([]()->std::unique_ptr<ge::AttrGroupsBase>{     \
               return std::unique_ptr<ge::AttrGroupsBase>(new(std::nothrow)cls());     \
          }, obj_type, bin_type)

namespace ge {
template<typename T>
struct HashedPointer {
  explicit HashedPointer(const T *ptr) : hash_value(std::hash<const T *>{}(ptr)) {}
  size_t hash_value;
  std::string ToString() const {
    return "Hashed_" + std::to_string(hash_value);
  }
};
using AttrGroupSerializeBuilder = std::function<std::unique_ptr<AttrGroupsBase>()>;
struct AttrGroupDeserializer {
  AttrGroupDeserializer(std::unique_ptr<AttrGroupsBase> impl_obj, TypeId id_obj)
      : impl(std::move(impl_obj)), id(id_obj) {}
  std::unique_ptr<AttrGroupsBase> impl{nullptr};
  TypeId id{nullptr};
};
class AttrGroupSerializerRegistry {
 public:
  AttrGroupSerializerRegistry(const AttrGroupSerializerRegistry &) = delete;
  AttrGroupSerializerRegistry(AttrGroupSerializerRegistry &&) = delete;
  AttrGroupSerializerRegistry &operator=(const AttrGroupSerializerRegistry &) = delete;
  AttrGroupSerializerRegistry &operator=(AttrGroupSerializerRegistry &&) = delete;

  ~AttrGroupSerializerRegistry() = default;

  static AttrGroupSerializerRegistry &GetInstance();
  /**
   * 注册一个Attr Group的序列化、反序列化handler
   * @param builder 调用该builder时，返回一个handler的实例
   * @param obj_type 内存中的数据类型，可以通过`GetTypeId<T>`函数获得
   * @param proto_type protobuf数据类型枚举值
   */
  void RegisterAttrGroupSerialize(const AttrGroupSerializeBuilder &builder,
                                  const TypeId obj_type,
                                  const proto::AttrGroupDef::AttrGroupCase proto_type);

  std::unique_ptr<AttrGroupsBase> GetSerializer(const TypeId obj_type);
  AttrGroupDeserializer GetDeserializer(const proto::AttrGroupDef::AttrGroupCase proto_type);

 private:
  AttrGroupSerializerRegistry() = default;

  std::mutex mutex_;
  std::map<TypeId, AttrGroupSerializeBuilder> serializer_builder_map_;
  std::map<proto::AttrGroupDef::AttrGroupCase, std::pair<AttrGroupSerializeBuilder, TypeId>> deserializer_builder_map_;
};

class AttrGroupSerializerRegister {
 public:
  AttrGroupSerializerRegister(const AttrGroupSerializeBuilder builder,
                              const TypeId obj_type,
                              const proto::AttrGroupDef::AttrGroupCase proto_type) noexcept;
  ~AttrGroupSerializerRegister() = default;
};
}  // namespace ge
#endif // METADEF_CXX_INC_GRAPH_ATTRIBUTE_GROUP_ATTR_GROUP_SERIALIZER_REGISTRY_H_
