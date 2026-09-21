/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_MODEL_ENDPOINT_H_
#define GE_COMMON_MODEL_ENDPOINT_H_

#include <string>
#include "graph/node.h"
#include "graph/attr_store.h"
#include "ge/ge_api_types.h"
#include "proto/flow_model.pb.h"

namespace ge {
using AttrValueMap = ::google::protobuf::Map<string, ge::flow_model::proto::ModelRelationDef::AttrValue>;
enum class EndpointType : std::uint32_t { kQueue = 0, kEvent, kDummyQueue, kMaxEndpointTypeNum };

class Endpoint {
 public:
  Endpoint(const string &name, const EndpointType endpoint_type) : name_(name), endpoint_type_(endpoint_type){};
  ~Endpoint() = default;
  EndpointType GetEndpointType() const;
  const std::string &GetName() const;
  std::string &MutableName();
  Endpoint &SetName(const std::string &name);
  Endpoint &SetEndpointType(const EndpointType endpoint_type);
  void SetAnyValueByName(const std::string &name, const AnyValue &value);

  template <typename T>
  void SetAttr(const std::string &name, const T &value) {
    (void)attrs_.SetByName(name, value);
  }

  template <typename T>
  const T GetAttr(const std::string &name, const T &default_value) const {
    const auto result_attr = attrs_.GetByName<T>(name);
    return (result_attr != nullptr) ? *result_attr : default_value;
  }

  // function for serialize
  std::map<std::string, AnyValue> GetAllAttrs() const;
  using SetAttrFunc = Status (*)(const AnyValue &value, ge::flow_model::proto::ModelRelationDef::AttrValue &attr);
  static const std::map<AnyValue::ValueType, SetAttrFunc> set_attr_funcs_;
  static Status SetStringAttr(const AnyValue &value, ge::flow_model::proto::ModelRelationDef::AttrValue &attr);
  static Status SetIntAttr(const AnyValue &value, ge::flow_model::proto::ModelRelationDef::AttrValue &attr);
  static Status SetBoolAttr(const AnyValue &value, ge::flow_model::proto::ModelRelationDef::AttrValue &attr);

  // function for deserialize
  using SetAnyValueFunc = Status (*)(const ge::flow_model::proto::ModelRelationDef::AttrValue &attr, AnyValue &value);
  static const std::map<ge::flow_model::proto::ModelRelationDef::AttrValue::ValueCase, SetAnyValueFunc>
      set_any_value_funcs_;
  static Status SetStringAnyValue(const ge::flow_model::proto::ModelRelationDef::AttrValue &attr, AnyValue &value);
  static Status SetIntAnyValue(const ge::flow_model::proto::ModelRelationDef::AttrValue &attr, AnyValue &value);
  static Status SetBoolAnyValue(const ge::flow_model::proto::ModelRelationDef::AttrValue &attr, AnyValue &value);

  Status Serialize(ge::flow_model::proto::ModelRelationDef_Endpoint *proto_endpoint) const;
  std::string SerializeToString() const;
  Status Deserialize(const ge::flow_model::proto::ModelRelationDef_Endpoint &proto_endpoint);
  bool ParseFromString(const std::string &data);

 private:
  std::string name_;
  EndpointType endpoint_type_;
  AttrStore attrs_;
};

constexpr int64_t kQueueActionDefault = 0;
constexpr int64_t kQueueActionControl = 1;
constexpr int64_t kQueueActionStatus = 2;
constexpr int64_t kQueueActionSched = 3;

class QueueNodeUtils {
 public:
  explicit QueueNodeUtils(Endpoint &endpoint) : endpoint_(endpoint){};
  QueueNodeUtils &SetDepth(const int64_t depth);
  QueueNodeUtils &SetEnqueuePolicy(const std::string &enqueue_policy);
  QueueNodeUtils &SetNodeAction(const int64_t action = kQueueActionDefault);
  static int64_t GetDepth(const Endpoint &endpoint);
  int64_t GetDepth() const;
  static std::string GetEnqueuePolicy(const Endpoint &endpoint);
  std::string GetEnqueuePolicy() const;
  static bool GetIsControl(const Endpoint &endpoint);
  bool GetIsControl() const;
  static bool GetIsStatus(const Endpoint &endpoint);
  static bool GetIsSched(const Endpoint &endpoint);

 private:
  Endpoint &endpoint_;
};
}  // namespace ge
#endif  // GE_COMMON_MODEL_ENDPOINT_H_
