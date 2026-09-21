/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "exe_graph/runtime/exe_res_generation_context.h"
#include "common/checker.h"
#include "graph/any_value.h"
#include "graph/node.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "exe_graph/lowering/exe_res_generation_ctx_builder.h"
#include "exe_graph/lowering/bg_kernel_context_extend.h"
namespace gert {
namespace {
enum class InputType : int32_t {
  kNode,
  kInputNum,
  kOutputNum,
  kShapeStart
};
void GeShapeToGertShape(const ge::GeShape &ge_shape, gert::Shape &gert_shape) {
  gert_shape.SetDimNum(ge_shape.GetDimNum());
  for (size_t i = 0; i < ge_shape.GetDimNum(); ++i) {
    GELOGD("Dim[%zu] val[%ld].", i, ge_shape.GetDim(i));
    gert_shape.SetDim(i, ge_shape.GetDim(i));
  }
}
}

void ExeResGenerationCtxBuilder::CreateShapesInputs(const ge::Node &node, std::vector<void *> &inputs) {
  auto op_desc = node.GetOpDesc();
  input_shapes_.reserve(op_desc->GetInputsSize());
  output_shapes_.reserve(op_desc->GetOutputsSize());
  for (const auto &in_data_anchor : node.GetAllInDataAnchors()) {
    if ((in_data_anchor == nullptr) || (in_data_anchor->GetPeerOutAnchor() == nullptr)) {
      GELOGD("In anchor is unused, get next.");
      continue;
    }
    const auto input_desc = op_desc->GetInputDescPtr(static_cast<uint32_t>(in_data_anchor->GetIdx()));
    if (input_desc == nullptr) {
      continue;
    }
    GELOGD("In anchor[%ld] push in shape.", in_data_anchor->GetIdx());
    StorageShape shape;
    GeShapeToGertShape(input_desc->GetShape(), shape.MutableStorageShape());
    GeShapeToGertShape(input_desc->GetOriginShape(), shape.MutableOriginShape());
    input_shapes_.emplace_back(std::move(shape));
  }
  for (const auto &out_data_anchor : node.GetAllOutDataAnchors()) {
    if (out_data_anchor == nullptr || out_data_anchor->GetPeerInDataNodesSize() == 0) {
      GELOGD("Node[%s] out anchor is null or peer in size is zero.", op_desc->GetNamePtr());
      continue;
    }
    const auto &output_desc = op_desc->GetOutputDescPtr(static_cast<uint32_t>(out_data_anchor->GetIdx()));
    if (output_desc == nullptr) {
      continue;
    }
    StorageShape shape;
    GeShapeToGertShape(output_desc->GetShape(), shape.MutableStorageShape());
    GeShapeToGertShape(output_desc->GetOriginShape(), shape.MutableOriginShape());
    output_shapes_.emplace_back(std::move(shape));
  }
  GELOGD("Node[%s] input size[%zu], output size[%zu].", op_desc->GetNamePtr(), input_shapes_.size(),
         output_shapes_.size());
  inputs.emplace_back(reinterpret_cast<void *>(input_shapes_.size()));
  inputs.emplace_back(reinterpret_cast<void *>(output_shapes_.size()));
  for (auto &in_shape : input_shapes_) {
    inputs.emplace_back(&in_shape);
  }
  for (auto &out_shape : output_shapes_) {
    inputs.emplace_back(&out_shape);
  }
  return;
}

ExeResGenerationCtxHolderPtr ExeResGenerationCtxBuilder::CreateOpExeContext(ge::Node &node) {
  std::vector<void *> inputs;
  inputs.emplace_back(&node);
  CreateShapesInputs(node, inputs);
  auto exe_res_ctx_holder = gert::KernelRunContextBuilder().Inputs(inputs).Build(node.GetOpDesc());
  ctx_holder_ptr_ = ge::ComGraphMakeShared<KernelContextHolder>(std::move(exe_res_ctx_holder));
  if (ctx_holder_ptr_ == nullptr || ctx_holder_ptr_->context_ == nullptr) {
    GE_LOGE("Op[%s][%s] create context holder failed.", node.GetNamePtr(), node.GetTypePtr());
    return nullptr;
  }
  auto op_exe_res_ctx = reinterpret_cast<ExeResGenerationContext *>(ctx_holder_ptr_->context_);
  if (!op_exe_res_ctx->CheckContextValid()) {
    GE_LOGE("Op[%s][%s] create context is invalid.", node.GetNamePtr(), node.GetTypePtr());
    return nullptr;
  }
  const auto input_num = op_exe_res_ctx->GetInputValue<size_t>(static_cast<uint32_t>(InputType::kInputNum));
  GELOGD("Input num:%zu.", input_num);
  for (size_t i = 0; i < input_num; ++i) {
    auto shape = op_exe_res_ctx->GetInputPointer<StorageShape>(static_cast<uint32_t>(InputType::kShapeStart) + i);
    GE_ASSERT_NOTNULL(shape);
    GELOGD("shape[%lx] Dim num[%zu].", shape, shape->GetStorageShape().GetDimNum());
  }
  return ctx_holder_ptr_;
}

ExeResGenerationCtxHolderPtr ExeResGenerationCtxBuilder::CreateOpCheckContext(ge::Node &node) {
  std::vector<void *> inputs;
  inputs.emplace_back(&node);
  CreateShapesInputs(node, inputs);
  auto kernel_ctx_holder = gert::KernelRunContextBuilder().Inputs(inputs).Build(node.GetOpDesc());
  ctx_holder_ptr_ = ge::ComGraphMakeShared<KernelContextHolder>(std::move(kernel_ctx_holder));
  if (ctx_holder_ptr_ == nullptr || ctx_holder_ptr_->context_ == nullptr) {
    GE_LOGE("Op[%s][%s] create context holder failed.", node.GetNamePtr(), node.GetTypePtr());
    return nullptr;
  }
  auto check_ctx_ptr_ = reinterpret_cast<OpCheckContext *>(ctx_holder_ptr_->context_);
  if (!check_ctx_ptr_->CheckContextValid()) {
    GE_LOGE("Op[%s][%s] create context is invalid.", node.GetNamePtr(), node.GetTypePtr());
    return nullptr;
  }
  return ctx_holder_ptr_;
}

bool ExeResGenerationContext::CheckContextValid() const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  if (node_ptr == nullptr) {
    GE_LOGE("Op exe res context node is null.");
    REPORT_INNER_ERR_MSG("E29999", "Node create exe context failed with null node.");
    return false;
  }
  return true;
}

ExecuteMode ExeResGenerationContext::GetExecuteMode() const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  GE_ASSERT_NOTNULL(node_ptr);
  const auto own_graph = node_ptr->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(own_graph);
  const auto ret = own_graph->GetGraphUnknownFlag() ? ExecuteMode::kDynamicExecute : ExecuteMode::kStaticOffloadExecute;
  GELOGD("Node[%s] exe mode is %d.", node_ptr->GetNamePtr(), ret);
  return ret;
}

// GetInputConstData is inaccurate(do not judge subgraph), need change new interface form GE later.
bool ExeResGenerationContext::IsConstInput(const ge::AscendString &name) const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  GE_ASSERT_NOTNULL(node_ptr);
  auto op_desc = node_ptr->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node_ptr->shared_from_this());
  const auto index = op_desc->GetInputIndexByName(name.GetString());
  if (index < 0) {
    GE_LOGE("Op[%s][%s] get invalid index[%d] by ir name[%s].", node_ptr->GetNamePtr(), node_ptr->GetTypePtr(), index,
            name.GetString());
    REPORT_INNER_ERR_MSG("E29999", "Node[%s][%s] get invalid index[%d] by ir name[%s].", node_ptr->GetNamePtr(),
                         node_ptr->GetTypePtr(), index, name.GetString());
    return false;
  }
  const bool ret = (op_desc->GetInputDesc(static_cast<uint32_t>(index)).IsValid() == ge::GRAPH_SUCCESS) &&
      (ge::OpDescUtils::GetInputConstData(op, static_cast<uint32_t>(index)) != nullptr);
  GELOGD("Node[%s] input[%d] is const flag:%d.", op_desc->GetNamePtr(), index, ret);
  return ret;
}

const gert::StorageShape* ExeResGenerationContext::GetInputShape(int64_t index) const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  GE_ASSERT_NOTNULL(node_ptr);
  const auto input_num = GetInputValue<size_t>(static_cast<uint32_t>(InputType::kInputNum));
  GELOGD("Node[%s] input index[%ld] with input num:%zu.", node_ptr->GetNamePtr(), index, input_num);
  if (index < 0 || static_cast<size_t>(index) >= input_num) {
    GE_LOGE("Op[%s] input index %ld is invalid, input num is %zu.", node_ptr->GetNamePtr(), index, input_num);
    REPORT_INNER_ERR_MSG("E29999", "Node[%s][%s] input index %ld is invalid, input num is %zu.", node_ptr->GetNamePtr(),
                         node_ptr->GetTypePtr(), index, input_num);
    return nullptr;
  }
  return GetInputPointer<StorageShape>(static_cast<size_t>(InputType::kShapeStart) + static_cast<size_t>(index));
}

const gert::StorageShape* ExeResGenerationContext::GetOutputShape(int64_t index) const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  GE_ASSERT_NOTNULL(node_ptr);
  const auto input_num = GetInputValue<size_t>(static_cast<uint32_t>(InputType::kInputNum));
  const auto output_num = GetInputValue<size_t>(static_cast<uint32_t>(InputType::kOutputNum));
  GELOGD("Node[%s] output index[%ld] with input num:%zu, out num:%zu.", node_ptr->GetNamePtr(), index, input_num,
         output_num);
  if (index < 0 || static_cast<size_t>(index) >= output_num) {
    GE_LOGE("Op[%s] output index %ld is invalid, output num is %zu.", node_ptr->GetNamePtr(), index, output_num);
    REPORT_INNER_ERR_MSG("E29999", "Node[%s][%s] output index %ld is invalid, output num is %zu.",
                         node_ptr->GetNamePtr(), node_ptr->GetTypePtr(), index, output_num);
    return nullptr;
  }
  return GetInputPointer<StorageShape>(static_cast<size_t>(InputType::kShapeStart) + input_num + static_cast<size_t>(index));
}

ge::graphStatus ExeResGenerationContext::SetAttachedStreamInfos(std::vector<StreamInfo> &stream_info_vec) const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  GE_ASSERT_NOTNULL(node_ptr);
  if (stream_info_vec.empty()) {
    GELOGW("Node[%s] set empty stream info vector.", node_ptr->GetNamePtr());
    return ge::GRAPH_FAILED;
  }
  GELOGD("Node[%s] set stream info vector size:%zu.", node_ptr->GetNamePtr(), stream_info_vec.size());
  std::vector<ge::GeAttrValue::NAMED_ATTRS> attached_stream_info;
  for (const auto &stream_info : stream_info_vec) {
    if (stream_info.name.GetLength() == 0) {
      REPORT_INNER_ERR_MSG("E29999", "Node[%s][%s] stream info using name is empty.", node_ptr->GetNamePtr(),
                           node_ptr->GetTypePtr());
      return ge::GRAPH_FAILED;
    }
    ge::GeAttrValue::NAMED_ATTRS attached_stream;
    (void)ge::AttrUtils::SetStr(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_NAME,
                                stream_info.name.GetString());
    (void)ge::AttrUtils::SetStr(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY,
                                stream_info.reuse_key.GetString());
    // ge::ATTR_NAME_ATTACHED_STREAM_DEPEND_VALUE_LIST
    (void)ge::AttrUtils::SetListInt(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_DEPEND_VALUE_LIST_INT,
                                    stream_info.depend_value_input_indices);
    (void)ge::AttrUtils::SetBool(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, stream_info.required);
    GELOGD("Stream info: name[%s], reuse_key[%s], required[%d].", stream_info.name.GetString(),
           stream_info.reuse_key.GetString(), stream_info.required);
    attached_stream_info.emplace_back(attached_stream);
  }
  (void)ge::AttrUtils::SetListNamedAttrs(node_ptr->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         attached_stream_info);
  return ge::GRAPH_SUCCESS;
}

std::vector<StreamInfo> ExeResGenerationContext::GetAttachedStreamInfos() const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  GE_ASSERT_NOTNULL(node_ptr);
  GELOGD("Node[%s] get stream info vector.", node_ptr->GetNamePtr());
  std::vector<ge::GeAttrValue::NAMED_ATTRS> stream_info_attrs;
  (void)ge::AttrUtils::GetListNamedAttrs(node_ptr->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         stream_info_attrs);
  GELOGD("Node[%s] get stream info vector size:%zu.", node_ptr->GetNamePtr(), stream_info_attrs.size());
  if (stream_info_attrs.empty()) {
    GELOGD("Node[%s] get empty stream info vector.", node_ptr->GetNamePtr());
    return {};
  }
  std::vector<StreamInfo> stream_info_vec;
  stream_info_vec.reserve(stream_info_attrs.size());
  std::string tmp_str;
  for (auto &stream_info_attr : stream_info_attrs) {
    StreamInfo stream_info;
    (void)ge::AttrUtils::GetStr(stream_info_attr, ge::ATTR_NAME_ATTACHED_RESOURCE_NAME, tmp_str);
    stream_info.name = tmp_str.c_str();
    (void)ge::AttrUtils::GetStr(stream_info_attr, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, tmp_str);
    stream_info.reuse_key = tmp_str.c_str();
    (void)ge::AttrUtils::GetListInt(stream_info_attr, ge::ATTR_NAME_ATTACHED_RESOURCE_DEPEND_VALUE_LIST_INT,
                                    stream_info.depend_value_input_indices);
    (void)ge::AttrUtils::GetBool(stream_info_attr, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG,
                                 stream_info.required);
    (void)ge::AttrUtils::GetBool(stream_info_attr, ge::ATTR_NAME_ATTACHED_RESOURCE_IS_VALID, stream_info.is_valid);
    (void)ge::AttrUtils::GetInt(stream_info_attr, ge::ATTR_NAME_ATTACHED_RESOURCE_ID, stream_info.stream_id);
    GELOGD("Get stream info:name[%s], reuse_key[%s], stream_id[%ld], required[%d], is_valid[%d].",
           stream_info.name.GetString(), stream_info.reuse_key.GetString(), stream_info.stream_id,
           stream_info.required, stream_info.is_valid);
    stream_info_vec.emplace_back(stream_info);
  }
  return stream_info_vec;
}

ge::graphStatus ExeResGenerationContext::SetListStr(const std::string &attr_name,
                                                    const std::vector<std::string> &list) const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  GE_ASSERT_NOTNULL(node_ptr);
  GELOGD("Node[%s] set list str:%s, size:%zu.", node_ptr->GetNamePtr(), attr_name.c_str(), list.size());
  (void)ge::AttrUtils::SetListStr(node_ptr->GetOpDesc(), attr_name, list);
  return ge::GRAPH_SUCCESS;
}

bool ExeResGenerationContext::GetStrAttrVal(const char *attr_name,
    ge::AscendString &val) const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  GE_ASSERT_NOTNULL(node_ptr);
  std::string val_str;
  auto res = ge::AttrUtils::GetStr(node_ptr->GetOpDesc(), attr_name, val_str);
  ge::AscendString val_tmp(val_str.c_str());
  val = val_tmp;
  return res;
}

bool ExeResGenerationContext::GetIntAttrVal(const char *attr_name, int64_t &val) const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  GE_ASSERT_NOTNULL(node_ptr);
  return ge::AttrUtils::GetInt(node_ptr->GetOpDesc(), attr_name, val);
}

bool ExeResGenerationContext::SetStrAttrVal(const char *attr_name, const char *val) const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  GE_ASSERT_NOTNULL(node_ptr);
  return ge::AttrUtils::SetStr(node_ptr->GetOpDesc(), attr_name, val);
}

bool ExeResGenerationContext::SetIntAttrVal(const char *attr_name, const int64_t val) const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  GE_ASSERT_NOTNULL(node_ptr);
  return ge::AttrUtils::SetInt(node_ptr->GetOpDesc(), attr_name, val);
}

ge::graphStatus ExeResGenerationContext::SetSyncResInfos(std::vector<SyncResInfo> &sync_info_vec) const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  GE_ASSERT_NOTNULL(node_ptr);
  GELOGD("Node[%s] set sync info vector size:%zu.", node_ptr->GetNamePtr(), sync_info_vec.size());
  if (sync_info_vec.empty()) {
    GELOGW("Node[%s] set empty sync info vector.", node_ptr->GetNamePtr());
    return ge::GRAPH_FAILED;
  }
  std::vector<ge::GeAttrValue::NAMED_ATTRS> sync_info_attrs;
  for (const auto &sync_info : sync_info_vec) {
    if (sync_info.name.GetLength() == 0) {
      REPORT_INNER_ERR_MSG("E29999", "Node[%s][%s] sync info using name is empty.", node_ptr->GetNamePtr(),
                           node_ptr->GetTypePtr());
      return ge::GRAPH_FAILED;
    }
    ge::GeAttrValue::NAMED_ATTRS sync_info_attr;
    (void)ge::AttrUtils::SetInt(sync_info_attr, ge::ATTR_NAME_ATTACHED_RESOURCE_TYPE,
                                static_cast<int64_t>(sync_info.type));
    (void)ge::AttrUtils::SetStr(sync_info_attr, ge::ATTR_NAME_ATTACHED_RESOURCE_NAME, sync_info.name.GetString());
    (void)ge::AttrUtils::SetStr(sync_info_attr, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY,
                                sync_info.reuse_key.GetString());
    (void)ge::AttrUtils::SetBool(sync_info_attr, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, sync_info.required);
    GELOGD("Sync info:name[%s], reuse_key[%s], type[%d], required[%d].",
           sync_info.name.GetString(), sync_info.reuse_key.GetString(), sync_info.type,
           sync_info.required);
    sync_info_attrs.emplace_back(sync_info_attr);
  }
  // ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO
  (void)ge::AttrUtils::SetListNamedAttrs(node_ptr->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         sync_info_attrs);
  return ge::GRAPH_SUCCESS;
}

std::vector<SyncResInfo> ExeResGenerationContext::GetSyncResInfos() const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  GE_ASSERT_NOTNULL(node_ptr);
  GELOGD("Node[%s] get sync info vector.", node_ptr->GetNamePtr());
  std::vector<ge::GeAttrValue::NAMED_ATTRS> sync_info_attrs;
  (void)ge::AttrUtils::GetListNamedAttrs(node_ptr->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         sync_info_attrs);
  GELOGD("Node[%s] Get sync info vector size:%zu.", node_ptr->GetNamePtr(), sync_info_attrs.size());
  if (sync_info_attrs.empty()) {
    GELOGD("Node[%s] get empty sync info vector.", node_ptr->GetNamePtr());
    return {};
  }
  std::vector<SyncResInfo> sync_info_vec;
  sync_info_vec.reserve(sync_info_attrs.size());
  std::string tmp_str;
  for (auto &sync_info_attr : sync_info_attrs) {
    SyncResInfo sync_info;
    int64_t type = 2;
    (void)ge::AttrUtils::GetInt(sync_info_attr, ge::ATTR_NAME_ATTACHED_RESOURCE_TYPE, type);
    sync_info.type = static_cast<SyncResType>(type);
    (void)ge::AttrUtils::GetStr(sync_info_attr, ge::ATTR_NAME_ATTACHED_RESOURCE_NAME, tmp_str);
    sync_info.name = tmp_str.c_str();
    (void)ge::AttrUtils::GetStr(sync_info_attr, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, tmp_str);
    sync_info.reuse_key = tmp_str.c_str();
    (void)ge::AttrUtils::GetBool(sync_info_attr, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, sync_info.required);
    (void)ge::AttrUtils::GetBool(sync_info_attr, ge::ATTR_NAME_ATTACHED_RESOURCE_IS_VALID, sync_info.is_valid);
    (void)ge::AttrUtils::GetInt(sync_info_attr, ge::ATTR_NAME_ATTACHED_RESOURCE_ID, sync_info.sync_res_id);
    sync_info_vec.emplace_back(sync_info);
    GELOGD("Sync info:name[%s], reuse_key[%s], type[%d], required[%d], is_valid[%d], sync_id[%ld].",
           sync_info.name.GetString(), sync_info.reuse_key.GetString(), sync_info.type,
           sync_info.required, sync_info.is_valid, sync_info.sync_res_id);
  }
  return sync_info_vec;
}

std::vector<int64_t> ExeResGenerationContext::GetWorkspaceBytes() const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  GE_ASSERT_NOTNULL(node_ptr);
  GE_ASSERT_NOTNULL(node_ptr->GetOpDesc());
  return node_ptr->GetOpDesc()->GetWorkspaceBytes();
}

void ExeResGenerationContext::SetWorkspaceBytes(const std::vector<int64_t> &workspace_bytes) const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  if (node_ptr != nullptr && node_ptr->GetOpDesc() != nullptr) {
    return node_ptr->GetOpDesc()->SetWorkspaceBytes(workspace_bytes);
  }
}

int64_t ExeResGenerationContext::GetStreamId() const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  GE_ASSERT_NOTNULL(node_ptr);
  GE_ASSERT_NOTNULL(node_ptr->GetOpDesc());
  return node_ptr->GetOpDesc()->GetStreamId();
}

int64_t ExeResGenerationContext::GetOpId() const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  GE_ASSERT_NOTNULL(node_ptr);
  GE_ASSERT_NOTNULL(node_ptr->GetOpDesc());
  return node_ptr->GetOpDesc()->GetId();
}

// GetInputConstData is inaccurate(do not judge subgraph), need change new interface form GE later.
bool OpCheckContext::IsConstInput(const ge::AscendString &name) const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  GE_ASSERT_NOTNULL(node_ptr);
  auto op_desc = node_ptr->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node_ptr->shared_from_this());
  const auto index = op_desc->GetInputIndexByName(name.GetString());
  if (index < 0) {
    GE_LOGE("Op[%s][%s] get invalid index[%d] by ir name[%s].", node_ptr->GetNamePtr(), node_ptr->GetTypePtr(), index,
            name.GetString());
    REPORT_INNER_ERR_MSG("E29999", "Node[%s][%s] get invalid index[%d] by ir name[%s].", node_ptr->GetNamePtr(),
                         node_ptr->GetTypePtr(), index, name.GetString());
    return false;
  }
  const bool ret = (op_desc->GetInputDesc(static_cast<uint32_t>(index)).IsValid() == ge::GRAPH_SUCCESS) &&
      (ge::OpDescUtils::GetInputConstData(op, static_cast<uint32_t>(index)) != nullptr);
  GELOGD("Node[%s] input[%d] is const flag:%d.", op_desc->GetNamePtr(), index, ret);
  return ret;
}

const StorageShape* OpCheckContext::GetInputShape(int64_t index) const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  GE_ASSERT_NOTNULL(node_ptr);
  const auto input_num = GetInputValue<size_t>(static_cast<uint32_t>(InputType::kInputNum));
  GELOGD("Node[%s] input index[%ld] with input num:%zu.", node_ptr->GetNamePtr(), index, input_num);
  if (index < 0 || static_cast<size_t>(index) >= input_num) {
    GE_LOGE("Op[%s] input index %ld is invalid, input num is %zu.", node_ptr->GetNamePtr(), index, input_num);
    REPORT_INNER_ERR_MSG("E29999", "Node[%s][%s] input index %ld is invalid, input num is %zu.", node_ptr->GetNamePtr(),
                         node_ptr->GetTypePtr(), index, input_num);
    return nullptr;
  }
  return GetInputPointer<StorageShape>(static_cast<size_t>(InputType::kShapeStart) + static_cast<size_t>(index));
}

const StorageShape* OpCheckContext::GetOutputShape(int64_t index) const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  GE_ASSERT_NOTNULL(node_ptr);
  const auto input_num = GetInputValue<size_t>(static_cast<uint32_t>(InputType::kInputNum));
  const auto output_num = GetInputValue<size_t>(static_cast<uint32_t>(InputType::kOutputNum));
  GELOGD("Node[%s] output index[%ld] with input num:%zu, out num:%zu.", node_ptr->GetNamePtr(), index, input_num,
         output_num);
  if (index < 0 || static_cast<size_t>(index) >= output_num) {
    GE_LOGE("Op[%s] output index %ld is invalid, output num is %zu.", node_ptr->GetNamePtr(), index, output_num);
    REPORT_INNER_ERR_MSG("E29999", "Node[%s][%s] output index %ld is invalid, output num is %zu.",
                         node_ptr->GetNamePtr(), node_ptr->GetTypePtr(), index, output_num);
    return nullptr;
  }
  return GetInputPointer<StorageShape>(static_cast<size_t>(InputType::kShapeStart) + input_num + static_cast<size_t>(index));
}

bool OpCheckContext::CheckContextValid() const {
  auto node_ptr = MutableInputPointer<ge::Node>(0);
  if (node_ptr == nullptr) {
    GE_LOGE("Op exe res context node is null.");
    REPORT_INNER_ERR_MSG("E29999", "Node create exe context failed with null node.");
    return false;
  }
  return true;
}

} // namespace gert
