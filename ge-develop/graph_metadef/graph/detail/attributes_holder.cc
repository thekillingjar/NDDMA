/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/detail/attributes_holder.h"

#include "framework/common/debug/ge_log.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/ge_attr_value.h"
#include "proto/ge_ir.pb.h"

namespace ge {
void AttrHolder::CopyAttrsFrom(const AttrHolder &holder) {
  MutableAttrMap() = holder.GetAttrMap();
}
void AttrHolder::CopyFrom(const AttrHolder &holder) {
  required_attrs_and_type_ = holder.required_attrs_and_type_;
  ext_attrs_ = holder.ext_attrs_;
}

graphStatus AttrHolder::SetAttr(const std::string &name, const AnyValue &value) {
  if (value.IsEmpty()) {
    REPORT_INNER_ERR_MSG("E18888", "param value is empty, check invalid, key of the attr:%s", name.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] value is empty, key of the attr is %s", name.c_str());
    return GRAPH_FAILED;
  }
  if (!MutableAttrMap().SetAnyValueByName(name, value)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
graphStatus AttrHolder::TrySetAttr(const std::string &name, const AnyValue &value) {
  if (value.IsEmpty()) {
    REPORT_INNER_ERR_MSG("E18888", "param value is empty, check invalid, key of the attr:%s", name.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] value is empty, key of the attr is %s", name.c_str());
    return GRAPH_FAILED;
  }
  if (MutableAttrMap().Exists(name)) {
    GELOGW("attr %s already existed, skip update", name.c_str());
  } else {
    if (!MutableAttrMap().SetAnyValueByName(name, value)) {
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}
graphStatus AttrHolder::AddRequiredAttr(const std::string &name) {
  return AddRequiredAttrWithType(name, "");
}

graphStatus AttrHolder::AddRequiredAttrWithType(const std::string &name, const std::string &type) {
  if (HasAttr(name)) {
    return GRAPH_FAILED;
  }
  required_attrs_and_type_.emplace(name, type);
  return GRAPH_SUCCESS;
}

graphStatus AttrHolder::GetAttr(const std::string &name, AnyValue &value) const {
  const auto av = GetAttrMap().GetAnyValue(name);
  if (av == nullptr) {
    return GRAPH_FAILED;
  }
  value = *av;
  return GRAPH_SUCCESS;
}

bool AttrHolder::HasAttr(const std::string &name) const {
  if (GetAttrMap().Exists(name)) {
    return true;
  }
  return required_attrs_and_type_.find(name) != required_attrs_and_type_.end();
}

graphStatus AttrHolder::DelAttr(const std::string &name) {
  return MutableAttrMap().Delete(name) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

const std::map<std::string, AnyValue> AttrHolder::GetAllAttrs() const {
  return GetAttrMap().GetAllAttrs();
}

const std::map<std::string, AnyValue> AttrHolder::GetAllAttrsWithFilter(const AttrNameFilter &attr_filter) const {
  return GetAttrMap().GetAllAttrsWithFilter(attr_filter);
}

const std::set<std::string> AttrHolder::GetAllAttrNames() const {
  return GetAttrMap().GetAllAttrNames();
}

template <>
void GeIrProtoHelper<proto::AttrDef>::InitDefault() {
  std::shared_ptr<proto::AttrDef> proto_owner;
  proto_owner = ComGraphMakeShared<proto::AttrDef>();
  if (proto_owner == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "create AttrDef failed.");
    GELOGE(GRAPH_FAILED, "[Create][AttrDef] proto::AttrDef make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::TensorDef>::InitDefault() {
  std::shared_ptr<proto::TensorDef> proto_owner;
  proto_owner = ComGraphMakeShared<proto::TensorDef>();
  if (proto_owner == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "create TensorDef failed.");
    GELOGE(GRAPH_FAILED, "[Create][TensorDef] proto::TensorDef make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::TensorDescriptor>::InitDefault() {
  std::shared_ptr<proto::TensorDescriptor> proto_owner;
  proto_owner = ComGraphMakeShared<proto::TensorDescriptor>();
  if (proto_owner == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "create TensorDescriptor failed.");
    GELOGE(GRAPH_FAILED, "[Create][TensorDescriptor] proto::TensorDescriptor make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::ShapeDef>::InitDefault() {
  std::shared_ptr<proto::ShapeDef> proto_owner;
  proto_owner = ComGraphMakeShared<proto::ShapeDef>();
  if (proto_owner == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "create ShapeDef failed.");
    GELOGE(GRAPH_FAILED, "[Create][ShapeDef] proto::ShapeDef make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::NamedAttrs>::InitDefault() {
  std::shared_ptr<proto::NamedAttrs> proto_owner;
  proto_owner = ComGraphMakeShared<proto::NamedAttrs>();
  if (proto_owner == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "create NamedAttrs failed.");
    GELOGE(GRAPH_FAILED, "[Create][NamedAttrs] proto::NamedAttrs make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::ModelDef>::InitDefault() {
  std::shared_ptr<proto::ModelDef> proto_owner;
  proto_owner = ComGraphMakeShared<proto::ModelDef>();
  if (proto_owner == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "create ModelDef failed.");
    GELOGE(GRAPH_FAILED, "[Create][ModelDef] proto::ModelDef make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::OpDef>::InitDefault() {
  std::shared_ptr<proto::OpDef> proto_owner;
  proto_owner = ComGraphMakeShared<proto::OpDef>();
  if (proto_owner == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "create OpDef failed.");
    GELOGE(GRAPH_FAILED, "[Create][OpDef] proto::OpDef make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::GraphDef>::InitDefault() {
  std::shared_ptr<proto::GraphDef> proto_owner;
  proto_owner = ComGraphMakeShared<proto::GraphDef>();
  if (proto_owner == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "create GraphDef failed.");
    GELOGE(GRAPH_FAILED, "[Create][GraphDef] proto::GraphDef make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}
}  // namespace ge
