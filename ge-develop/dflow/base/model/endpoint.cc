/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "endpoint.h"
#include <iomanip>
#include <google/protobuf/text_format.h>
#include "common/types.h"
#include "common/util.h"
#include "common/util/mem_utils.h"
#include "dflow/inc/data_flow/model/pne_model.h"
#include "common/checker.h"

namespace ge {
namespace {
/**
 * QueueDef 属性的索引，若需要扩展，请新增枚举类型
 */
constexpr const char_t kQueueDepth[] = "QueueDepth";
constexpr const char_t kQueueEnqueuePolicy[] = "QueueEnqueuePolicy";
constexpr const char_t kQueueNodeAction[] = "QueueNodeAction";

constexpr char_t kDefaultStrVal[] = "";
constexpr int64_t kDefaultInt64Val = -1;
constexpr char_t kDefaultEnqueuePolicy[] = "FIFO";
constexpr int64_t kDefaultDepth = 128L;
}  // namespace

const std::map<AnyValue::ValueType, Endpoint::SetAttrFunc> Endpoint::set_attr_funcs_ = {
    {AnyValue::ValueType::VT_STRING, &Endpoint::SetStringAttr},
    {AnyValue::ValueType::VT_INT, &Endpoint::SetIntAttr},
    {AnyValue::ValueType::VT_BOOL, &Endpoint::SetBoolAttr}};

const std::map<ge::flow_model::proto::ModelRelationDef::AttrValue::ValueCase, Endpoint::SetAnyValueFunc>
    Endpoint::set_any_value_funcs_ = {
        {ge::flow_model::proto::ModelRelationDef::AttrValue::kS, &Endpoint::SetStringAnyValue},
        {ge::flow_model::proto::ModelRelationDef::AttrValue::kI, &Endpoint::SetIntAnyValue},
        {ge::flow_model::proto::ModelRelationDef::AttrValue::kB, &Endpoint::SetBoolAnyValue}};

EndpointType Endpoint::GetEndpointType() const {
  return endpoint_type_;
}
const std::string &Endpoint::GetName() const {
  return name_;
}
std::string &Endpoint::MutableName() {
  return name_;
}

Endpoint &Endpoint::SetName(const std::string &name) {
  name_ = name;
  return *this;
}

Endpoint &Endpoint::SetEndpointType(const EndpointType endpoint_type) {
  endpoint_type_ = endpoint_type;
  return *this;
}

void Endpoint::SetAnyValueByName(const std::string &name, const AnyValue &value) {
  (void)attrs_.SetAnyValueByName(name, value);
}

std::map<std::string, AnyValue> Endpoint::GetAllAttrs() const {
  return attrs_.GetAllAttrs();
}

Status Endpoint::SetStringAttr(const AnyValue &value, ge::flow_model::proto::ModelRelationDef::AttrValue &attr) {
  std::string attr_value;
  GE_CHK_STATUS_RET(value.GetValue(attr_value), "Get string from AnyValue failed.");
  attr.set_s(attr_value);
  return SUCCESS;
}

Status Endpoint::SetIntAttr(const AnyValue &value, ge::flow_model::proto::ModelRelationDef::AttrValue &attr) {
  int64_t attr_value = 0;
  GE_CHK_STATUS_RET(value.GetValue(attr_value), "Get int64_t from AnyValue failed.");
  attr.set_i(attr_value);
  return SUCCESS;
}

Status Endpoint::SetBoolAttr(const AnyValue &value, ge::flow_model::proto::ModelRelationDef::AttrValue &attr) {
  bool attr_value = false;
  GE_CHK_STATUS_RET(value.GetValue(attr_value), "Get bool from AnyValue failed.");
  attr.set_b(attr_value);
  return SUCCESS;
}

Status Endpoint::SetStringAnyValue(const ge::flow_model::proto::ModelRelationDef::AttrValue &attr, AnyValue &value) {
  GE_CHK_STATUS_RET(value.SetValue(attr.s()), "Get string from AttrValue failed.");
  return SUCCESS;
}

Status Endpoint::SetIntAnyValue(const ge::flow_model::proto::ModelRelationDef::AttrValue &attr, AnyValue &value) {
  GE_CHK_STATUS_RET(value.SetValue(attr.i()), "Get int from AttrValue failed.");
  return SUCCESS;
}

Status Endpoint::SetBoolAnyValue(const ge::flow_model::proto::ModelRelationDef::AttrValue &attr, AnyValue &value) {
  GE_CHK_STATUS_RET(value.SetValue(attr.b()), "Get bool from AttrValue failed.");
  return SUCCESS;
}

Status Endpoint::Serialize(ge::flow_model::proto::ModelRelationDef_Endpoint *proto_endpoint) const {
  // intrinsic attr
  proto_endpoint->set_name(GetName());
  proto_endpoint->set_endpoint_type(static_cast<int32_t>(GetEndpointType()));

  // attrs attr
  const std::map<std::string, AnyValue> name_to_value = GetAllAttrs();
  google::protobuf::Map<std::string, ge::flow_model::proto::ModelRelationDef::AttrValue> *attrs_iter =
      proto_endpoint->mutable_attrs();
  for (const auto &iter : name_to_value) {
    const auto &attr_name = iter.first;
    const auto &any_value = iter.second;
    const auto &attr_type = any_value.GetValueType();
    const auto &func_iter = set_attr_funcs_.find(attr_type);
    GE_ASSERT_TRUE(func_iter != set_attr_funcs_.cend(), "Endpoint attr type(%u) does not has process function.",
                   static_cast<int64_t>(attr_type));

    ge::flow_model::proto::ModelRelationDef::AttrValue attr_value;
    GE_CHK_STATUS_RET(func_iter->second(any_value, attr_value), "Failed to set %s attr.", attr_name.c_str());
    const auto value_type = AttrValueMap::value_type(attr_name, attr_value);
    (void)attrs_iter->insert(value_type);
  }
  return SUCCESS;
}

std::string Endpoint::SerializeToString() const {
  auto proto_endpoint = MakeShared<ge::flow_model::proto::ModelRelationDef_Endpoint>();
  GE_ASSERT_NOTNULL(proto_endpoint, "[New][EndpointProto] failed, endpoint name = %s", GetName().c_str());
  GE_ASSERT_SUCCESS(Serialize(proto_endpoint.get()), "[Serialize][Endpoint] failed, name = %s", GetName().c_str());
  std::string output;
  GE_ASSERT_TRUE(google::protobuf::TextFormat::PrintToString(*proto_endpoint, &output),
                 "[Print][ToString] failed, endpoint name = %s", GetName().c_str());
  return output;
}

bool Endpoint::ParseFromString(const std::string &data) {
  auto proto_endpoint = MakeShared<ge::flow_model::proto::ModelRelationDef_Endpoint>();
  GE_ASSERT_NOTNULL(proto_endpoint, "[New][EndpointProto] failed, endpoint name = %s", GetName().c_str());
  GE_ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(data, proto_endpoint.get()),
                 "[Parse][FromString] failed, endpoint name = %s", GetName().c_str());
  GE_ASSERT_SUCCESS(Deserialize(*proto_endpoint), "[Deserialize][Endpoint] failed, endpoint name = %s",
                    GetName().c_str());
  return true;
}

Status Endpoint::Deserialize(const ge::flow_model::proto::ModelRelationDef_Endpoint &proto_endpoint) {
  // intrinsic attr
  (void)SetName(proto_endpoint.name());
  (void)SetEndpointType(static_cast<EndpointType>(proto_endpoint.endpoint_type()));

  // attrs attr
  const auto attrs_iter = proto_endpoint.attrs();
  for (const auto &iter : attrs_iter) {
    const auto &attr_name = iter.first;
    const auto &attr_value = iter.second;
    const auto &attr_value_case = attr_value.value_case();
    const auto &func_iter = set_any_value_funcs_.find(attr_value_case);
    GE_ASSERT_TRUE(func_iter != set_any_value_funcs_.cend(), "Endpoint attr type(%u) does not has process function.",
                   static_cast<int64_t>(attr_value_case));

    AnyValue any_value;
    GE_CHK_STATUS_RET(func_iter->second(attr_value, any_value), "Failed to get %s 's any_value.", attr_name.c_str());
    this->SetAnyValueByName(attr_name, any_value);
  }
  return SUCCESS;
}

QueueNodeUtils &QueueNodeUtils::SetDepth(const int64_t depth) {
  endpoint_.SetAttr(kQueueDepth, depth);
  return *this;
}

QueueNodeUtils &QueueNodeUtils::SetEnqueuePolicy(const std::string &enqueue_policy) {
  endpoint_.SetAttr(kQueueEnqueuePolicy, enqueue_policy);
  return *this;
}

QueueNodeUtils &QueueNodeUtils::SetNodeAction(const int64_t action) {
  endpoint_.SetAttr(kQueueNodeAction, action);
  return *this;
}

int64_t QueueNodeUtils::GetDepth() const {
  const int64_t depth = endpoint_.GetAttr<int64_t>(kQueueDepth, kDefaultInt64Val);
  return (depth == -1L) ? kDefaultDepth : depth;
}

int64_t QueueNodeUtils::GetDepth(const Endpoint &endpoint) {
  const int64_t depth = endpoint.GetAttr<int64_t>(kQueueDepth, kDefaultInt64Val);
  return (depth == -1L) ? kDefaultDepth : depth;
}

std::string QueueNodeUtils::GetEnqueuePolicy() const {
  const std::string enqueue_policy = endpoint_.GetAttr<std::string>(kQueueEnqueuePolicy, kDefaultStrVal);
  return (enqueue_policy.empty()) ? kDefaultEnqueuePolicy : enqueue_policy;
}

std::string QueueNodeUtils::GetEnqueuePolicy(const Endpoint &endpoint) {
  const std::string enqueue_policy = endpoint.GetAttr<std::string>(kQueueEnqueuePolicy, kDefaultStrVal);
  return (enqueue_policy.empty()) ? kDefaultEnqueuePolicy : enqueue_policy;
}

bool QueueNodeUtils::GetIsControl() const {
  const int64_t action = endpoint_.GetAttr<int64_t>(kQueueNodeAction, kQueueActionDefault);
  return action == kQueueActionControl;
}

bool QueueNodeUtils::GetIsControl(const Endpoint &endpoint) {
  const int64_t action = endpoint.GetAttr<int64_t>(kQueueNodeAction, kQueueActionDefault);
  return action == kQueueActionControl;
}

bool QueueNodeUtils::GetIsStatus(const Endpoint &endpoint) {
  const int64_t action = endpoint.GetAttr<int64_t>(kQueueNodeAction, kQueueActionDefault);
  return action == kQueueActionStatus;
}

bool QueueNodeUtils::GetIsSched(const Endpoint &endpoint) {
  const int64_t action = endpoint.GetAttr<int64_t>(kQueueNodeAction, kQueueActionDefault);
  return action == kQueueActionSched;
}
}  // namespace ge
