/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/model/model_compress_manager.h"
#include "graph/utils/enum_attr_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
constexpr int64_t kNoNeedCompress = 0xFFFFFFFFFFFFFFFF;
constexpr int64_t kNeedCompressVersion = 0x1;
const string kKernelName = "_kernelname";
const string kAtomicKernelName = "_atomic_kernelname";
const std::set<std::string> kUnusedAttrNames = { "_l2fusion_ToOpStruct", "_op_slice_info", "_ir_attr_names" };

int64_t ModelCompressManager::om_compress_version_;
vector<string> ModelCompressManager::enum_attr_names_;
vector<string> ModelCompressManager::enum_attr_values_;
vector_bit_t ModelCompressManager::name_use_string_values_;
std::mutex ModelCompressManager::mutex_;

Status ModelCompressManager::Compress(const GeModelPtr &ge_model) {
  GELOGD("begin to compress model.");
  GE_ASSERT_NOTNULL(ge_model);
  const std::lock_guard<std::mutex> lock(mutex_);
  CacheClear();
  if (!CheckNeedCompress(ge_model)) {
    GELOGI("no need to compress model.");
    return SUCCESS;
  }
  const auto &compute_graph = ge_model->GetGraph();
  GE_ASSERT_NOTNULL(compute_graph);
  GE_DUMP(compute_graph, "BeforeAttrsCompress");
  for (const auto &node : compute_graph->GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    GE_ASSERT_NOTNULL(node->GetOpDesc());
    PrintOpInfo(node->GetOpDesc());
    GE_ASSERT_SUCCESS(ProcessKernelNameAttrsForOp(node->GetOpDesc()));
    GE_ASSERT_SUCCESS(ProcessAttrsForOp(node->GetOpDesc(), true));
    GE_ASSERT_SUCCESS(ProcessAttrsForTensor(node->GetOpDesc(), true));
  }

  GE_ASSERT_SUCCESS(AddEnumAttrsToModel(ge_model));
  GE_DUMP(compute_graph, "AfterAttrsCompress");
  GELOGD("success compress model.");
  return SUCCESS;
}

Status ModelCompressManager::Decompress(const GeModelPtr &ge_model) {
  GELOGD("begin to decompress model.");
  GE_ASSERT_NOTNULL(ge_model);
  const std::lock_guard<std::mutex> lock(mutex_);
  CacheClear();
  GE_ASSERT_SUCCESS(GetEnumAttrsFromModel(ge_model), "failed get enum attrs from model.");
  const auto &compute_graph = ge_model->GetGraph();
  GE_ASSERT_NOTNULL(compute_graph);
  if (om_compress_version_ == kNoNeedCompress) {
    GELOGI("no need to decompress model.");
    return SUCCESS;
  }
  GE_DUMP(compute_graph, "BeforeAttrsDecompress");
  for (const auto &node : compute_graph->GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    GE_ASSERT_NOTNULL(node->GetOpDesc());
    GE_ASSERT_SUCCESS(ProcessAttrsForOp(node->GetOpDesc(), false));
    GE_ASSERT_SUCCESS(ProcessAttrsForTensor(node->GetOpDesc(), false));
    PrintOpInfo(node->GetOpDesc());
  }

  GE_DUMP(compute_graph, "AfterAttrsDecompress");
  GELOGD("success decompress model.");
  return SUCCESS;
}

Status ModelCompressManager::ProcessAttrsForOp(const OpDescPtr &op_desc, const bool is_compress) {
  const auto &attrs_map = op_desc->GetAllAttrs();
  for (const auto &iter : attrs_map) {
    string attr_name = "";
    GeAttrValue attr_value;
    if (is_compress) {
      if (DeleteUnusedAttrs(op_desc, iter.first)) {
        continue;
      }
      GE_ASSERT_SUCCESS(EnumAttrs(iter, attr_name, attr_value), "EnumAttrs failed.");
    } else {
      GE_ASSERT_SUCCESS(DenumAttrs(iter, attr_name, attr_value), "DenumAttrs failed.");
    }
    GE_ASSERT_GRAPH_SUCCESS(op_desc->DelAttr(iter.first), "op_desc delete attr failed.");
    GE_ASSERT_GRAPH_SUCCESS(op_desc->SetAttr(attr_name, attr_value), "op_desc set attr failed.");
  }
  return SUCCESS;
}

Status ModelCompressManager::ProcessKernelNameAttrsForOp(const OpDescPtr &op_desc) {
  GE_ASSERT_GRAPH_SUCCESS(ProcessKernelName(op_desc));
  GE_ASSERT_GRAPH_SUCCESS(ProcessAtomicKernelName(op_desc));
  return SUCCESS;
}

Status ModelCompressManager::ProcessKernelName(const OpDescPtr &op_desc) {
  string attr_value;
  const string pre_kernel_name = op_desc->GetName() + kKernelName;
  if (AttrUtils::GetStr(op_desc, kKernelName, attr_value)) {
    (void)op_desc->DelAttr(pre_kernel_name);
    (void)op_desc->DelAttr(ATTR_NAME_TBE_KERNEL_NAME_FOR_LOAD);
  } else if (AttrUtils::GetStr(op_desc, pre_kernel_name, attr_value) ||
             AttrUtils::GetStr(op_desc, ATTR_NAME_TBE_KERNEL_NAME_FOR_LOAD, attr_value)) {
    GE_ASSERT_TRUE(AttrUtils::SetStr(op_desc, kKernelName, attr_value),
                   "op_desc[%s] set kKernelName[%s] value[%s] failed.",
                   op_desc->GetName().c_str(), kKernelName.c_str(), attr_value.c_str());
    (void)op_desc->DelAttr(pre_kernel_name);
    (void)op_desc->DelAttr(ATTR_NAME_TBE_KERNEL_NAME_FOR_LOAD);
  } else {
    // do nothing
  }
  return SUCCESS;
}

Status ModelCompressManager::ProcessAtomicKernelName(const OpDescPtr &op_desc) {
  string attr_value;
  const string pre_atomic_kernel_name = op_desc->GetName() + kAtomicKernelName;
  if (AttrUtils::GetStr(op_desc, kAtomicKernelName, attr_value)) {
    (void)op_desc->DelAttr(pre_atomic_kernel_name);
  } else if (AttrUtils::GetStr(op_desc, pre_atomic_kernel_name, attr_value)) {
    GE_ASSERT_TRUE(AttrUtils::SetStr(op_desc, kAtomicKernelName, attr_value),
                   "op_desc[%s] set kAtomicKernelName[%s] value[%s] failed.",
                   op_desc->GetName().c_str(), kAtomicKernelName.c_str(), attr_value.c_str());
    (void)op_desc->DelAttr(pre_atomic_kernel_name);
  } else {
    // do nothing
  }
  return SUCCESS;
}

Status ModelCompressManager::ProcessAttrsForTensor(const OpDescPtr &op_desc, const bool is_compress) {
  for (size_t i = 0U; i < op_desc->GetInputsSize(); i++) {
    const auto &input_tensor = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    GE_ASSERT_SUCCESS(ProcessForTensor(input_tensor, is_compress));
  }

  for (size_t i = 0U; i < op_desc->GetOutputsSize(); i++) {
    const auto &output_tensor = op_desc->MutableOutputDesc(static_cast<uint32_t>(i));
    GE_ASSERT_SUCCESS(ProcessForTensor(output_tensor, is_compress));
  }
  return SUCCESS;
}

Status ModelCompressManager::ProcessForTensor(const GeTensorDescPtr &tensor, const bool is_compress) {
  if (tensor == nullptr) {
    return SUCCESS;
  }
  const auto &attrs_map = tensor->GetAllAttrs();
  for (const auto &iter : attrs_map) {
    string attr_name = "";
    GeAttrValue attr_value;
    if (is_compress) {
      GE_ASSERT_SUCCESS(EnumAttrs(iter, attr_name, attr_value), "EnumAttrs failed.");
    } else {
      GE_ASSERT_SUCCESS(DenumAttrs(iter, attr_name, attr_value), "DenumAttrs failed.");
    }
    GE_ASSERT_GRAPH_SUCCESS(tensor->DelAttr(iter.first), "tensor delete attr failed.");
    GE_ASSERT_GRAPH_SUCCESS(tensor->SetAttr(attr_name, attr_value), "tensor set attr failed.");
  }
  return SUCCESS;
}

Status ModelCompressManager::AddEnumAttrsToModel(const GeModelPtr &ge_model) {
  if (enum_attr_names_.size() == 0U) {
    GELOGI("no need to add enum attrs to model.");
    return SUCCESS;
  }
  om_compress_version_ = kNeedCompressVersion;
  GE_ASSERT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_OM_COMPRESS_VERSION, om_compress_version_),
                 "set om_compress_version_ failed.");
  GE_ASSERT_TRUE(AttrUtils::SetListStr(ge_model, ATTR_MODEL_ATTR_NAME_ENUM, enum_attr_names_),
                 "set enum_attr_names_ failed.");
  GE_ASSERT_TRUE(AttrUtils::SetListStr(ge_model, ATTR_MODEL_ATTR_VALUE_ENUM, enum_attr_values_),
                 "set enum_attr_values_ failed.");
  GE_ASSERT_TRUE(AttrUtils::SetListBool(ge_model, ATTR_MODEL_ATTRS_USE_STRING_VALUE, name_use_string_values_),
                 "set name_use_string_values_ failed.");
  GELOGD("success add enum attrs to model, version:%lld, enum_attr_names_ size:%zu, enum_attr_values_ size:%zu.",
         om_compress_version_, enum_attr_names_.size(), enum_attr_values_.size());
  return SUCCESS;
}

Status ModelCompressManager::GetEnumAttrsFromModel(const GeModelPtr &ge_model) {
  if (!AttrUtils::GetInt(ge_model, ATTR_MODEL_OM_COMPRESS_VERSION, om_compress_version_)) {
    om_compress_version_ = kNoNeedCompress;
    return SUCCESS;
  }
  GE_ASSERT_TRUE(AttrUtils::GetListStr(ge_model, ATTR_MODEL_ATTR_NAME_ENUM, enum_attr_names_),
                 "get enum_attr_names_ failed.");
  GE_ASSERT_TRUE(AttrUtils::GetListStr(ge_model, ATTR_MODEL_ATTR_VALUE_ENUM, enum_attr_values_),
                 "get enum_attr_values_ failed.");
  GE_ASSERT_TRUE(AttrUtils::GetListBool(ge_model, ATTR_MODEL_ATTRS_USE_STRING_VALUE, name_use_string_values_),
                 "get name_use_string_values_ failed.");
  GE_ASSERT_TRUE((enum_attr_names_.size() == name_use_string_values_.size()),
                 "enum_attr_names size[%zu] name_use_string_values size[%zu].",
                 enum_attr_names_.size(), name_use_string_values_.size());
  GELOGD("success get enum attrs from model.");
  return SUCCESS;
}

bool ModelCompressManager::CheckNeedCompress(const GeModelPtr &ge_model) {
  if (!AttrUtils::GetInt(ge_model, ATTR_MODEL_OM_COMPRESS_VERSION, om_compress_version_)) {
    return true;
  }
  return false;
}

Status ModelCompressManager::CpyModelAttrs2Dst(const GeModelPtr &src_ge_model, const GeModelPtr &dst_ge_model) {
  const std::lock_guard<std::mutex> lock(mutex_);
  GE_ASSERT_SUCCESS(GetEnumAttrsFromModel(src_ge_model));
  if (om_compress_version_ == kNoNeedCompress) {
    GELOGI("no need to cpy model attrs.");
    return SUCCESS;
  }
  GE_ASSERT_SUCCESS(AddEnumAttrsToModel(dst_ge_model));
  GELOGI("success cpy model attrs to dst.");
  return SUCCESS;
}

void ModelCompressManager::DeleteModelAttrs(const GeModelPtr &ge_model) {
  const std::lock_guard<std::mutex> lock(mutex_);
  (void)ge_model->DelAttr(ATTR_MODEL_OM_COMPRESS_VERSION);
  (void)ge_model->DelAttr(ATTR_MODEL_ATTR_NAME_ENUM);
  (void)ge_model->DelAttr(ATTR_MODEL_ATTR_VALUE_ENUM);
  (void)ge_model->DelAttr(ATTR_MODEL_ATTRS_USE_STRING_VALUE);
  GELOGI("success delete model attrs.");
}

bool ModelCompressManager::DeleteUnusedAttrs(const OpDescPtr &op_desc, const string &attr_name) {
  if (kUnusedAttrNames.find(attr_name) != kUnusedAttrNames.cend()) {
    (void)op_desc->DelAttr(attr_name);
    GELOGD("success delete op_desc name[%s] attr name[%s].", op_desc->GetName().c_str(), attr_name.c_str());
    return true;
  }
  return false;
}

Status ModelCompressManager::EnumAttrs(const pair<const string, GeAttrValue> &name_to_value, string &enum_attr_name,
                                       GeAttrValue &enum_attr_value) {
  // 1. compress attr name
  const auto &attr_name = name_to_value.first;
  bool is_new_attr = false;
  EnumAttrUtils::GetEnumAttrName(enum_attr_names_, attr_name, enum_attr_name, is_new_attr);

  // 2. compress attr value
  const auto &attr_value = name_to_value.second;
  const auto &value_type = attr_value.GetValueType();
  if (value_type == GeAttrValue::VT_STRING) {
    string data_s;
    GE_ASSERT_GRAPH_SUCCESS(attr_value.GetValue(data_s), "get value failed.");
    int64_t data_i = 0;
    EnumAttrUtils::GetEnumAttrValue(enum_attr_values_, data_s, data_i);
    GE_ASSERT_GRAPH_SUCCESS(enum_attr_value.SetValue(data_i), "set value failed.");
    UpdateStatus(is_new_attr, true);
  } else if (value_type == GeAttrValue::VT_LIST_STRING) {
    vector<string> data_ss = {};
    GE_ASSERT_GRAPH_SUCCESS(attr_value.GetValue(data_ss), "get value failed.");
    vector<int64_t> data_is = {};
    EnumAttrUtils::GetEnumAttrValues(enum_attr_values_, data_ss, data_is);
    GE_ASSERT_GRAPH_SUCCESS(enum_attr_value.SetValue(data_is), "set value failed.");
    UpdateStatus(is_new_attr, true);
  } else {
    enum_attr_value = attr_value;
    UpdateStatus(is_new_attr, false);
  }
  return SUCCESS;
}

Status ModelCompressManager::DenumAttrs(const pair<const string, GeAttrValue> &name_to_value, string &attr_name,
                                        GeAttrValue &attr_value) {
  // 1. decompress attr name
  const auto &enum_attr_name = name_to_value.first;
  bool is_value_string = false;
  GE_ASSERT_GRAPH_SUCCESS(EnumAttrUtils::GetAttrName(enum_attr_names_, name_use_string_values_, enum_attr_name,
                                                     attr_name, is_value_string), "[GetAttrName] failed");

  // 2. decompress attr value
  if (is_value_string) {
    const auto &enum_attr_value = name_to_value.second;
    const auto &value_type = enum_attr_value.GetValueType();
    if (value_type == GeAttrValue::VT_INT) {
      int64_t data_i = 0;
      GE_ASSERT_GRAPH_SUCCESS(enum_attr_value.GetValue(data_i), "get value failed.");
      string data_s;
      GE_ASSERT_GRAPH_SUCCESS(EnumAttrUtils::GetAttrValue(enum_attr_values_, data_i, data_s),
                              "[GetAttrValue] failed");
      GE_ASSERT_GRAPH_SUCCESS(attr_value.SetValue(data_s), "set value failed.");
    } else if (value_type == GeAttrValue::VT_LIST_INT) {
      vector<int64_t> data_is = {};
      GE_ASSERT_GRAPH_SUCCESS(enum_attr_value.GetValue(data_is), "get value failed.");
      vector<string> data_ss = {};
      GE_ASSERT_GRAPH_SUCCESS(EnumAttrUtils::GetAttrValues(enum_attr_values_, data_is, data_ss),
                              "[GetAttrValues] failed");
      GE_ASSERT_GRAPH_SUCCESS(attr_value.SetValue(data_ss), "set value failed.");
    } else {
      GELOGE(FAILED, "has unsupport value_type[%u].", static_cast<uint32_t>(value_type));
      return FAILED;
    }
  } else {
    attr_value = name_to_value.second;
  }
  return SUCCESS;
}

void ModelCompressManager::UpdateStatus(const bool is_new_attr, const bool is_string_type) {
  if (is_new_attr) {
    name_use_string_values_.push_back(is_string_type);
  }
}

void ModelCompressManager::PrintOpInfo(const OpDescPtr &op_desc) {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    return;
  }
  stringstream ss;
  const auto &op_attrs = op_desc->GetAllAttrs();
  ss << "op name: " << op_desc->GetName() << "; op_attrs size: " << op_attrs.size() << "; ";
  for (size_t i = 0U; i < op_desc->GetInputsSize(); i++) {
    const auto &input_tensor = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (input_tensor == nullptr) {
      continue;
    }
    const auto &input_tensor_attrs = input_tensor->GetAllAttrs();
    ss << "input_tensor" << i << " input_tensor_attrs size: " << input_tensor_attrs.size() << "; ";
  }
  for (size_t i = 0U; i < op_desc->GetOutputsSize(); i++) {
    const auto &output_tensor = op_desc->MutableOutputDesc(static_cast<uint32_t>(i));
    if (output_tensor == nullptr) {
      continue;
    }
    const auto &output_tensor_attrs = output_tensor->GetAllAttrs();
    ss << "output_tensor" << i << " output_tensor_attrs size: " << output_tensor_attrs.size() << "; ";
  }
  GELOGD("%s", ss.str().c_str());
}

void ModelCompressManager::CacheClear() {
  om_compress_version_ = kNoNeedCompress;
  enum_attr_names_.clear();
  enum_attr_values_.clear();
  name_use_string_values_.clear();
}
}  // namespace ge
