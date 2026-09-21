/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_kernel_store/sub_ops_store.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <utility>
#include <unordered_map>
#include <cmath>
#include "common/fe_inner_error_codes.h"
#include "common/graph/fe_graph_utils.h"
#include "common/string_utils.h"
#include "common/fe_type_utils.h"
#include "common/configuration.h"
#include "common/fe_context_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "common/math_util.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "graph/ge_tensor.h"
#include "external/op_common/data_type_utils.h"
#include "transfer_shape_utils.h"
#include "graph_optimizer/dynamic_shape_optimizer/fuzzy_compiler/input_node_generalize.h"

using ge::GeAttrValue;
using std::map;
using std::string;
using std::vector;
using fe::StringUtils;

namespace fe {
namespace {
bool GetTensorName(SupportedFormatAndDtype &info, const string &input_or_output, uint32_t index,
                   const IndexNameMap &index_name_map, string &name) {
  std::ostringstream reason_oss;
  auto index_name_map_iter = index_name_map.find(index);
  if (index_name_map_iter == index_name_map.end()) {
    reason_oss << "the name of " << input_or_output << " [" << index << "] is not found";
    info.reason = reason_oss.str();
    REPORT_FE_ERROR("[ChkSpt][FEChk][GetTensorNm]Can not find the %s index %u in map whose size is %zu",
                    input_or_output.c_str(), index, info.input_index_name_map.size());
    return false;
  }
  name = index_name_map_iter->second;
  return true;
}

void SetReason(string reason_str, OpNotSupportedReasonID id, UnSupportedReason &reason) {
  reason.reason += reason_str;
  reason.reason_id = static_cast<uint64_t>(id);
}
} // namespace

/*
 *  @namespace fe
 *  @brief   check specified GeAttrValue::ValueType of op_desc_attr.Value() is in info_op_attr.vector<Values>
 *  @param   [in] value      : used to specified GeAttrValue::ValueType for template
 *  @param   [in] op_desc_attr : GeAttrValue from OpDesc
 *  @param   [in] info_op_attr : vector<GeAttrValue> from OpKernelInfo
 *  @return  true or false
 */
template <typename T>
bool FindValueInVector(T &value, const GeAttrValue &op_desc_attr, const vector<GeAttrValue> &info_op_attr);

SubOpsStore::SubOpsStore(const std::string &engine_name)
    : init_flag_(false),
      engine_name_(engine_name),
      sub_store_info_(),
      format_dtype_querier_ptr_(nullptr) {}

SubOpsStore::~SubOpsStore() {}

void SubOpsStore::SetSubStoreInfo(const FEOpsStoreInfo &sub_store_info) { sub_store_info_ = (sub_store_info); }

Status SubOpsStore::InitializeSubStore() {
  if (init_flag_) {
    FE_LOGD("SubOpsStore has already been initialized, directly return.");
    return SUCCESS;
  }

  FE_LOGD("Initialize %s SubOpsStore begin.", sub_store_info_.fe_ops_store_name.c_str());

  FE_MAKE_SHARED(format_dtype_querier_ptr_ = std::make_shared<FormatDtypeQuerier>(engine_name_),
      return FAILED);
  FE_CHECK_NOTNULL(format_dtype_querier_ptr_);

  FE_LOGI("Initialize %s SubOpsStore finished.", sub_store_info_.fe_ops_store_name.c_str());

  init_flag_ = true;
  return SUCCESS;
}

Status SubOpsStore::FinalizeSubStore() {
  FE_LOGD("Finalizing the SubOpsKernelInfoStore begin.");
  if (!init_flag_) {
    FE_LOGD("SubOpsStore has not been initialized, Finalize is not allowed, directly return.");
    return SUCCESS;
  }

  init_flag_ = false;
  return SUCCESS;
}

PrecisionReduceType SubOpsStore::GetPrecisionReduceType(const ge::DataType &desc_dtype,
                                                        const ge::DataType &stored_dtype) const {
  if (stored_dtype == ge::DT_FLOAT16 && desc_dtype == ge::DT_FLOAT) {
    return PrecisionReduceType::FP32_TO_FP16;
  }

  if (stored_dtype == ge::DT_FLOAT16 && desc_dtype == ge::DT_BF16) {
    return PrecisionReduceType::BF16_TO_FP16;
  }

  if (stored_dtype == ge::DT_BF16 && desc_dtype == ge::DT_FLOAT) {
    return PrecisionReduceType::FP32_TO_BF16;
  }

  return PrecisionReduceType::INVALID;
}

bool SubOpsStore::PrecisionReduceToFp16(const bool &black_list_op, const fe::PrecisionMode &precision_mode) const {
  if (((precision_mode == fe::PrecisionMode::ENUM_FORCE_FP32) ||
       (precision_mode == fe::PrecisionMode::ENUM_ALLOW_MIX_PRECISION_FP16)) &&
      black_list_op) {
    return false;
  }
  if (precision_mode == fe::PrecisionMode::ENUM_ALLOW_FP32_TO_BF16 ||
      precision_mode == fe::PrecisionMode::ENUM_ALLOW_MIX_PRECISION_BF16) {
    return false;
  }

  return true;
}

bool SubOpsStore::PrecisionReduceToBf16(const bool &black_list_op, const fe::PrecisionMode &precision_mode) const {
  if (((precision_mode == fe::PrecisionMode::ENUM_FORCE_FP32) ||
       (precision_mode == fe::PrecisionMode::ENUM_ALLOW_MIX_PRECISION_BF16)) &&
      black_list_op) {
    return false;
  }

  if (precision_mode == fe::PrecisionMode::ENUM_ALLOW_FP32_TO_FP16 ||
      precision_mode == fe::PrecisionMode::ENUM_ALLOW_MIX_PRECISION_FP16) {
    return false;
  }
  return true;
}

bool SubOpsStore::CheckPrecisionReduce(const ge::DataType &desc_dtype, const ge::DataType &stored_dtype,
    const int64_t &keep_dtype, const bool &black_list_op, const fe::PrecisionMode &precision_mode) const {
  // dtype match
  if (desc_dtype == stored_dtype) {
    return true;
  }

  // improve precision
  if ((desc_dtype == ge::DT_FLOAT16 && stored_dtype == ge::DT_FLOAT) ||
      (desc_dtype == ge::DT_BF16 && stored_dtype == ge::DT_FLOAT)) {
    return true;
  }

  // reduce precision
  if (keep_dtype || precision_mode == fe::PrecisionMode::ENUM_MUST_KEEP_ORIGIN_DTYPE) {
    return false;
  }

  PrecisionReduceType reduce_type = GetPrecisionReduceType(desc_dtype, stored_dtype);
  size_t index = static_cast<size_t>(reduce_type);
  return (index >= static_cast<size_t>(PrecisionReduceType::INVALID))
             ? false
             : precison_reduce_checkfuncs_[index](black_list_op, precision_mode);
}

void SubOpsStore::JudgeNeedUpdateDtype(const fe::PrecisionMode &precision_mode,
    const ge::GeTensorDescPtr &tensor_desc_ptr, const vector<ge::DataType> &support_data_types,
    const ge::OpDescPtr &op_desc_ptr, const InputOrOutputInfoPtr &input_or_output_info_ptr) const {
  if (precision_mode == fe::PrecisionMode::ENUM_ALLOW_FP32_TO_FP16) {
    FE_LOGD("Current precision mode is allow_fp32_tofp16, start checking.");
    if (tensor_desc_ptr->GetDataType() == ge::DT_FLOAT &&
        IsAllDtypeExpected(ge::DT_FLOAT16, ge::DT_FLOAT, support_data_types)) {
      FE_LOGD("Node[%s, %s]: current precision mode is allow_fp32_tofp16.", op_desc_ptr->GetName().c_str(),
              op_desc_ptr->GetType().c_str());
      FE_LOGD("Node[%s, %s], Tensor[%s]: dtype supports fp16. Need to update dtype when doing checksupport.",
              op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(),
              input_or_output_info_ptr->GetName().c_str());
      (void)ge::AttrUtils::SetBool(tensor_desc_ptr, NEED_UPDATE_DTYPE_WHEN_OP_CHECKSUPPORT, true);
    } else {
      (void)ge::AttrUtils::SetBool(tensor_desc_ptr, NEED_UPDATE_DTYPE_WHEN_OP_CHECKSUPPORT, false);
    }
  }
  if (precision_mode == fe::PrecisionMode::ENUM_ALLOW_MIX_PRECISION_BF16) {
    FE_LOGD("Current precision mode is allow_mix_precision_bf16. Start to do checksupport.");
    if (tensor_desc_ptr->GetDataType() == ge::DT_FLOAT && IsSupportExpectedDtype(ge::DT_BF16, support_data_types)) {
      FE_LOGD("Node[%s, %s], Tensor[%s]: dtype supports bf16. Need to update dtype when doing checksupport.",
              op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(),
              input_or_output_info_ptr->GetName().c_str());
      (void)ge::AttrUtils::SetBool(tensor_desc_ptr, NEED_UPDATE_DTYPE_WHEN_OP_CHECKSUPPORT, true);
    } else {
      FE_LOGD("Node[%s, %s], Tensor[%s]: no need to update dtype when doing checksupport.",
              op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(),
              input_or_output_info_ptr->GetName().c_str());
      (void)ge::AttrUtils::SetBool(tensor_desc_ptr, NEED_UPDATE_DTYPE_WHEN_OP_CHECKSUPPORT, false);
    }
  }
}

bool SubOpsStore::CheckPromoteTypeSupport(const vector<ge::DataType> &support_data_types,
                                          const SupportedFormatAndDtype &info) {
  if (!info.promote_flag || !info.is_input || info.promote_input_list.empty() ||
      (info.promote_input_list.size() != info.promote_target_type.size())) {
    return false;
  }
  ge::DataType target_type = ge::DT_UNDEFINED;
  for (size_t i = 0; i < info.promote_input_list.size(); ++i) {
    if (std::find(info.promote_input_list[i].begin(), info.promote_input_list[i].end(),
        info.cur_idx) != info.promote_input_list[i].end()) {
      target_type = info.promote_target_type[i];
      break;
    }
  }
  if (target_type == ge::DT_UNDEFINED) {
    return false;
  }
  for (const auto &stored_dtype : support_data_types) {
    auto promoted_dtype = opcommon::PromoteType(target_type, stored_dtype);
    if (promoted_dtype == stored_dtype) {
      return true;
    }
  }
  return false;                                          
}

bool SubOpsStore::CheckDtypeSupported(const ge::NodePtr &node, const ge::GeTensorDescPtr &tensor_desc_ptr,
                                      InputOrOutputInfoPtr input_or_output_info_ptr,
                                      const vector<ge::DataType> &support_data_types,
                                      const SupportedFormatAndDtype &info) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  const std::string &op_type = op_desc_ptr->GetType();
  FE_CHECK(tensor_desc_ptr == nullptr,
           REPORT_FE_ERROR("[ChkSpt][FEChk][ChkTensor]cTensorDescPtr is nullptr."), return false);
  FE_CHECK(input_or_output_info_ptr == nullptr,
           REPORT_FE_ERROR("[ChkSpt][FEChk][ChkTensor] InputOrOutputInfoPtr is nullptr."), return false);
  ge::DataType desc_dtype = tensor_desc_ptr->GetDataType();
  /* For those configurated as keep-dtype, we will not allow precision loss. */
  int64_t keep_dtype = 0;
  (void)ge::AttrUtils::GetInt(*(op_desc_ptr.get()), KEEP_DTYPE, keep_dtype);
  fe::PrecisionMode precision_mode = fe::PrecisionMode::ENUM_UNDEFINED;
  FeGraphUtils::GetPrecisionModeFromGraph(*(node->GetOwnerComputeGraph()), precision_mode);
  PrecisionPolicy op_kernel_policy = info.op_kernel_info_ptr->GetPrecisionPolicy();
  PrecisionPolicy op_precision_policy = Configuration::Instance(engine_name_).GetPrecisionPolicy(
      op_desc_ptr->GetType(), op_kernel_policy);
  bool black_list_op = op_precision_policy == BLACK;

  JudgeNeedUpdateDtype(precision_mode, tensor_desc_ptr, support_data_types, op_desc_ptr, input_or_output_info_ptr);
  bool is_check_true = false;
  for (auto &store_dtype : support_data_types) {
    is_check_true = CheckPrecisionReduce(desc_dtype, store_dtype, keep_dtype, black_list_op, precision_mode);
    if (is_check_true) {
      return true;
    }
  }
  if (CheckPromoteTypeSupport(support_data_types, info)) {
    FE_LOGD("Check promote dtype support for op[%s:%s] success, cur input tensor [%s].",
      op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr(), input_or_output_info_ptr->GetUniqueName().c_str());
    (void)ge::AttrUtils::SetBool(op_desc_ptr, kMustPromoteFlag, true);
    return true;
  }
  FE_LOGD("Check dtype for op[%s:%s] tensor %s, dtype %s, support data types is %s, precision_mode is %u unsuccessful",
      op_desc_ptr->GetNamePtr(),
      op_desc_ptr->GetTypePtr(),
      input_or_output_info_ptr->GetUniqueName().c_str(),
      ge::TypeUtils::DataTypeToSerialString(desc_dtype).c_str(),
      GetStrByDataTypeVec(support_data_types).c_str(),
      precision_mode);
  return false;
}

bool SubOpsStore::CheckAccuracyDtypeSupported(ge::ConstGeTensorDescPtr c_tensor_desc_ptr,
                                              uint32_t type_index, InputOrOutputInfoPtr input_or_output_info_ptr,
                                              const vector<ge::DataType> &support_data_types) const {
  FE_CHECK(c_tensor_desc_ptr == nullptr,
           REPORT_FE_ERROR("[ChkSpt][FEChk][ChkAccurDtypeSup] cTensorDescPtr is nullptr."), return false);
  FE_CHECK(input_or_output_info_ptr == nullptr,
           REPORT_FE_ERROR("[ChkSpt][FEChk][ChkAccurDTypeSup] InputOrOutputInfoPtr is nullptr."), return false);
  ge::DataType desc_dtype = c_tensor_desc_ptr->GetDataType();
  if (support_data_types.size() < type_index + 1) {
    FE_LOGW("supportDataTypes.size() less than type_index.");
    return false;
  }

  if (desc_dtype == support_data_types[type_index]) {
    return true;
  } else {
    return false;
  }
}

bool SubOpsStore::CheckFormatSupported(const ge::NodePtr &node, ge::ConstGeTensorDescPtr c_tensor_desc_ptr,
                                       InputOrOutputInfoPtr input_or_output_info_ptr,
                                       const vector<ge::Format> &support_formats) const {
  FE_CHECK(c_tensor_desc_ptr == nullptr,
           REPORT_FE_ERROR("[ChkSpt][FEChk][ChkFmtSpt] cTensorDescPtr of %s is nullptr.",
                           node->GetName().c_str()), return false);
  FE_CHECK(input_or_output_info_ptr == nullptr,
           REPORT_FE_ERROR("[ChkSpt][FEChk][ChkFmtSpt] InputOrOutputInfoPtr of %s is nullptr.",
                           node->GetName().c_str()), return false);
  uint32_t tensor_index = input_or_output_info_ptr->GetIndex();
  // TBE plugin: the op_info_format_vec may be empty
  if (support_formats.empty()) {
    FE_LOGI("The op_info_format is empty, check success, return true.");
    return true;
  }

  ge::Format desc_format = c_tensor_desc_ptr->GetOriginFormat();
  if (!IsNd(desc_format)) {
    return true;
  }

  FE_LOGD("The format in tensor [named %s, index %u] of node %s is %s.",
          input_or_output_info_ptr->GetName().c_str(), tensor_index,
          node->GetName().c_str(), FormatToStr(desc_format).c_str());
  for (auto &op_info_format : support_formats) {
    if (IsSupportedTransType(desc_format, op_info_format)) {
      FE_LOGD("Check format successfully! Formats %s in op info lib supports this tensor [named %s, index %u].",
              ge::TypeUtils::FormatToSerialString(op_info_format).c_str(), input_or_output_info_ptr->GetName().c_str(),
              tensor_index);
      return true;
    }
  }
  FE_LOGD("Formats in op info lib does not match with format in tensor.");
  return false;
}

bool SubOpsStore::CheckSubformatSupported(const ge::NodePtr &node, int64_t groups,
                                          const InputOrOutputInfoPtr &input_or_output_info_ptr,
                                          const string &tensor_name,
                                          SupportedFormatAndDtype &info) const {
  FE_CHECK(input_or_output_info_ptr == nullptr,
           REPORT_FE_ERROR("[ChkSpt][FEChk][ChkSubfmtSpt] InputOrOutputInfoPtr of %s is nullptr.",
                           node->GetName().c_str()), return false);
  if (groups > UINT16_MAX) {
    REPORT_FE_ERROR("[ChkSpt][FEChk][ChkSubfmtSpt] Node[%s]'s groups[%ld] exceeds the maximum supported value.",
                    node->GetName().c_str(), groups);
    return false;
  }
  auto support_format_vec = info.suppport_formats_map[tensor_name];
  auto support_sub_format_vec =  info.suppport_sub_formats_map[tensor_name];
  for (auto &format : support_format_vec) {
    if (std::find(FE_GROUP_RELA_FORMAT_VECTOR.begin(), FE_GROUP_RELA_FORMAT_VECTOR.end(), format) ==
      FE_GROUP_RELA_FORMAT_VECTOR.end()) {
        return true;
    }
  }
  if (support_sub_format_vec.empty()) {
    FE_LOGI("[ChkSpt][FEChk][ChkSubfmtSpt] Node[%s] support sub_format is empty, not support groups %ld",
            node->GetName().c_str(), groups);
    return false;
  }
  if (std::find(support_sub_format_vec.begin(), support_sub_format_vec.end(), SUPPORT_ALL_SUB_FORMAT) !=
      support_sub_format_vec.end()) {
    return true;
  }
  uint32_t tensor_index = input_or_output_info_ptr->GetIndex();
  FE_LOGD("[ChkSpt][FEChk][ChkSubfmtSpt] The subformat support in tensor [named %s, index %u] of node %s is %s.",
          input_or_output_info_ptr->GetName().c_str(), tensor_index, node->GetName().c_str(),
          StringUtils::IntegerVecToString(support_sub_format_vec).c_str());
  if (std::find(support_sub_format_vec.begin(), support_sub_format_vec.end(), static_cast<uint32_t>(groups)) ==
      support_sub_format_vec.end()) {
    FE_LOGD("[ChkSubfmtSpt] Node[%s] subformat[%s] in op info lib does not match with groups[%ld] in tensor.",
            node->GetName().c_str(), StringUtils::IntegerVecToString(support_sub_format_vec).c_str(), groups);
    return false;
  }
  FE_LOGD("Check subformat successfully! Groups %ld in op info lib supports this tensor [named %s, index %u].",
          groups, input_or_output_info_ptr->GetName().c_str(), tensor_index);
  return true;
}

bool SubOpsStore::CheckAccuracyFormatSupported(ge::ConstGeTensorDescPtr c_tensor_desc_ptr,
                                               uint32_t type_index, InputOrOutputInfoPtr input_or_output_info_ptr,
                                               const vector<ge::Format> &support_formats) const {
  FE_CHECK(c_tensor_desc_ptr == nullptr,
           REPORT_FE_ERROR("[ChkSpt][FEChk][ChkAccurFmtSpt] cTensorDescPtr is nullptr."), return false);
  FE_CHECK(input_or_output_info_ptr == nullptr,
           REPORT_FE_ERROR("[ChkSpt][FEChk][ChkAccurFmtSpt] InputOrOutputInfoPtr is nullptr."), return false);
  ge::Format desc_format = static_cast<ge::Format>(ge::GetPrimaryFormat(c_tensor_desc_ptr->GetFormat()));
  if (support_formats.size() >= type_index + 1 &&
      (support_formats[type_index] == ge::FORMAT_ND || desc_format == support_formats[type_index])) {
    return true;
  }
  return false;
}

bool SubOpsStore::VerifyFormatC0Val(const ge::OpDescPtr &op_desc_ptr, const ge::ConstGeTensorDescPtr &tensor_desc_ptr) {
  FE_CHECK(op_desc_ptr == nullptr,
           FE_LOGW("[ChkSpt][FEChk][ChkAccurC0Val] op_desc_ptr is nullptr."), return false);
  // only check C0 for cast node
  // ge may try to swap position of transdata and cast, and check whether they can do it.
  // for fp16 and fp32, the c0 for cast is always 16. cast from 5HD,fp16(c0=16) to 5HD,fp32(c0=8) is not support
  if (op_desc_ptr->GetType() != CAST) {
    return true;
  }
  FE_CHECK(tensor_desc_ptr == nullptr,
           FE_LOGW("[ChkSpt][FEChk][ChkAccurC0Val] tensor_desc_ptr is nullptr."), return false);
  // only check while data type is fp32
  if (tensor_desc_ptr->GetDataType() != ge::DT_FLOAT || !tensor_desc_ptr->GetShape().IsUnknownShape()) {
    return true;
  }
  ge::Format format = tensor_desc_ptr->GetFormat();
  if (!ge::HasC0Format(static_cast<int32_t>(format))) {
    return true;
  }
  int64_t current_c0_val = ge::GetC0Value(static_cast<int32_t>(format));
  int64_t c0_val = transformer::TransferShapeUtils::GetC0ByDtype(tensor_desc_ptr->GetDataType());
  if (current_c0_val != c0_val) {
    FE_LOGW("C0 val of data type[%s] should be [%ld], but current c0 val from format is [%ld]. Cast op is [%s].",
            ge::TypeUtils::DataTypeToSerialString(tensor_desc_ptr->GetDataType()).c_str(),
            c0_val, current_c0_val, op_desc_ptr->GetNamePtr());
  }
  return current_c0_val == c0_val;
}

template <typename T>
bool FindValueInVector(T &value, const GeAttrValue &op_desc_attr, const vector<GeAttrValue> &info_op_attr) {
  if (op_desc_attr.GetValue<T>(value) != ge::GRAPH_SUCCESS) {
    FE_LOGD("Can not get value from OpDescAttr.");
    return false;
  }

  for (auto &info_op_attr_value : info_op_attr) {
    T info_value;
    if (info_op_attr_value.GetValue<T>(info_value) != ge::GRAPH_SUCCESS) {
      FE_LOGD("Get value from OpDescAttr not successfully");
      return false;
    }
    if (CompareValue(value, info_value)) {
      return true;
    }
  }
  return false;
}

bool SubOpsStore::CheckAttrValue(const GeAttrValue &op_desc_attr, const vector<GeAttrValue> &info_op_attr,
                                 GeAttrValue::ValueType attr_type) const {
  switch (attr_type) {
    case GeAttrValue::VT_INT: {
      int64_t attr_value;
      return FindValueInVector(attr_value, op_desc_attr, info_op_attr);
    }
    case GeAttrValue::VT_FLOAT: {
      float attr_value;
      return FindValueInVector(attr_value, op_desc_attr, info_op_attr);
    }
    case GeAttrValue::VT_BOOL: {
      bool attr_value = false;
      return FindValueInVector(attr_value, op_desc_attr, info_op_attr);
    }
    case GeAttrValue::VT_STRING: {
      string attr_value;
      bool ret = FindValueInVector(attr_value, op_desc_attr, info_op_attr);
      return ret;
    }
    case GeAttrValue::VT_LIST_INT: {
      vector<int64_t> attr_value;
      return FindValueInVector(attr_value, op_desc_attr, info_op_attr);
    }
    case GeAttrValue::VT_LIST_FLOAT: {
      vector<float> attr_value;
      return FindValueInVector(attr_value, op_desc_attr, info_op_attr);
    }
    case GeAttrValue::VT_LIST_BOOL: {
      vector<bool> attr_value;
      return FindValueInVector(attr_value, op_desc_attr, info_op_attr);
    }
    case GeAttrValue::VT_LIST_STRING: {
      vector<string> attr_value;
      return FindValueInVector(attr_value, op_desc_attr, info_op_attr);
    }
    case GeAttrValue::VT_LIST_LIST_INT: {
      vector<vector<int64_t>> attr_value;
      return FindValueInVector(attr_value, op_desc_attr, info_op_attr);
    }
    default:
      FE_LOGW("Not supported ValueType %d.", attr_type);
      break;
  }
  return false;
}

bool SubOpsStore::CheckAttrSupport(const ge::NodePtr &node, const OpKernelInfo &op_kernel_info,
                                   std::string &reason) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  ge::OpDesc &op_desc = *(op_desc_ptr.get());
  // get all attrs from input OpDesc
  map<string, GeAttrValue> op_desc_attrs = op_desc.GetAllAttrs();

  // get all attributes information from op kernel
  const vector<AttrInfoPtr> &attrs_info = op_kernel_info.GetVecAttrInfo();
  if (attrs_info.empty()) {
    FE_LOGD("The Op: [%s] contains no attribute.", op_desc.GetName().c_str());
    return true;
  }

  std::ostringstream reason_oss;
  // check attr support, check each OpKernelInfo specified Attr should in OpDesc
  for (auto &iter : attrs_info) {
    string key = iter->GetAttrName();
    // check OpDesc should have each attr specified in OpKernelInfo
    if (op_desc_attrs.count(key) == 0) {
      if (iter->GetIsRequired()) {
        reason_oss << ("the attribute[" + key + "] which is required is not found in op information library");
        reason = reason_oss.str();
        FE_LOGW("Can not find AttrValue:[%s] in OpDesc (name %s)", key.c_str(), op_desc.GetName().c_str());
        return false;
      } else {
        continue;
      }
    }
    // if attr "isSupportAllValue_" flag is true, then pass this attr value check
    if (iter->GetSupportAllValue()) {
      continue;
    }

    // check attr value match
    // get Attr Value in OpDesc: GeAttrValue
    map<string, GeAttrValue>::const_iterator op_desc_attr_iter = op_desc_attrs.find(key);
    if (op_desc_attr_iter == op_desc_attrs.cend()) {
      if (iter->GetIsRequired()) {
        reason_oss << ("the attribute[" + key + "] which is required is not found in op information library");
        reason = reason_oss.str();
        FE_LOGW("Can not find Attr Value (key [%s]) in OpDesc (name [%s]) and it is required!", key.c_str(),
                op_desc.GetName().c_str());
        return false;
      } else {
        continue;
      }
    }
    GeAttrValue op_desc_attr_value = op_desc_attr_iter->second;

    // get Attr Value Vector in Op Kernel Info: vector<GeAttrValue>
    vector<GeAttrValue> attr_value_vector;
    uint32_t ret = op_kernel_info.GetAttrValueByAttrName(key, attr_value_vector);
    if (ret != SUCCESS) {
      reason_oss << ("the value of attribute[" + key + "] is not found in op information library");
      reason = reason_oss.str();
      FE_LOGW("Can not find Attr Value (key [%s]) in OpKernelInfo of op (name [%s]). Ret [%u]", key.c_str(),
              op_desc.GetName().c_str(), ret);
      return false;
    }

    // get GeAttrValue::ValueType of Attr from op_kernel_info
    GeAttrValue::ValueType attr_value_type;
    ret = op_kernel_info.GetAttrTypeByAttrName(key, attr_value_type);
    if (ret != SUCCESS) {
      reason_oss << ("the data type of attribute[" + key + "] is not found in op information library");
      reason = reason_oss.str();
      FE_LOGW("Can not find the attr value type of Attr [%s] of op (name [%s]) in OpKernelInfo; Ret value [%u]",
              key.c_str(), op_desc.GetName().c_str(), ret);
      if (ret == OP_ATTR_EMPTY_IN_OP_KERNEL_INFO) {
        return true;
      }
      return false;
    }

    // check if op_desc_attr_value is in info_attr_value_list
    if (!CheckAttrValue(op_desc_attr_value, attr_value_vector, attr_value_type)) {
      reason_oss << ("the matched value of attribute[" + key + "] is not found in op desc");
      reason = reason_oss.str();
      FE_LOGD("Match attr value of attr name %s type [%u] from OpKernelInfo not successfully.", key.c_str(),
              attr_value_type);
      return false;
    }
  }
  FE_LOGD("check OpDesc: %s attr value supported", op_desc.GetName().c_str());
  return true;
}

bool SubOpsStore::CheckParamType(const ge::NodePtr &node, OpKernelInfoPtr op_kernel_info_ptr,
                                 const map<uint32_t, string> &input_index_name_map,
                                 const map<uint32_t, string> &output_index_name_map, std::string &reason) const {
  FE_CHECK(op_kernel_info_ptr == nullptr,
           REPORT_FE_ERROR("[ChkSpt][FEChk][ChkParmType] opKernelInfoPtr is nullptr."), return false);
  auto &all_input_info_vector = op_kernel_info_ptr->GetAllInputInfo();
  auto &all_output_info_vector = op_kernel_info_ptr->GetAllOutputInfo();
  if (!CheckAllTensorsParamType(node, true, all_input_info_vector, input_index_name_map, reason)) {
    return false;
  }

  return CheckAllTensorsParamType(node, false, all_output_info_vector, output_index_name_map, reason);
}

bool SubOpsStore::CheckAllTensorsParamType(const ge::NodePtr &node, bool is_input,
                                           const vector<InputOrOutputInfoPtr> &all_tesnors_info,
                                           const map<uint32_t, string> &index_name_map, std::string &reason) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  ge::OpDesc &op_desc = *(op_desc_ptr.get());
  int index = 0;
  std::ostringstream reason_oss;
  for (auto &tensor_info : all_tesnors_info) {
    int32_t counter = 0;
    for (auto &index_name : index_name_map) {
      if (index_name.second == tensor_info->GetName()) {
        counter++;
      }
    }
    if ((tensor_info->GetParamType() == REQUIRED) && (counter != 1)) {
      reason_oss << "the quantity of required " << IS_INPUT_TO_STRING(is_input);
      reason_oss <<  "[" << tensor_info->GetName() << "] is not one";
      reason = reason_oss.str();
      FE_LOGD("%s [%d] which named [%s] is required but its quantity is [%d] instead of 1. node is [%s] [%s].",
              IS_INPUT_TO_STRING(is_input), index, tensor_info->GetName().c_str(), counter, op_desc.GetName().c_str(),
              op_desc.GetType().c_str());
      return false;
    }
    if ((tensor_info->GetParamType() == DYNAMIC) && (counter == 0)) {
      reason_oss << "the quantity of dynamic " << IS_INPUT_TO_STRING(is_input);
      reason_oss << "[" << tensor_info->GetName() << "] should be larger than zero";
      reason = reason_oss.str();
      FE_LOGD("%s [%d] which named [%s] is dynamic but its quantity is 0 (should be larger than 0). node is [%s] [%s]",
              IS_INPUT_TO_STRING(is_input), index, tensor_info->GetName().c_str(), op_desc.GetName().c_str(),
              op_desc.GetType().c_str());
      return false;
    }
    index++;
  }
  return true;
}

void SubOpsStore::LogAllFormatAndDtype(const SupportedFormatAndDtype &info, const string &tensor_name,
                                       std::ostringstream &reason_oss, string &reason) const {
  auto all_data_type = StringUtils::IntegerVecToString(info.support_data_types_map.at(tensor_name));
  auto all_formats = StringUtils::IntegerVecToString(info.suppport_formats_map.at(tensor_name));
  reason_oss << "All supported data type and format of tensor " << tensor_name << " is: "
             << "Data Type: " << all_data_type
             << "Format:" << all_formats;
  reason = reason_oss.str();
}

bool SubOpsStore::CheckFormatAndDtypeNormalMode(const ge::NodePtr &node, const std::string &name,
                                                const ge::GeTensorDescPtr &tensor_desc,
                                                SupportedFormatAndDtype &info,
                                                const bool &is_force_dtype_support,
                                                std::string &reason) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  ge::OpDesc &op_desc = *(op_desc_ptr.get());
  std::ostringstream reason_oss;
  InputOrOutputInfoPtr tensor_kernel;
  string in_or_out = IS_INPUT_TO_STRING(info.is_input);
  if (info.op_kernel_info_ptr->GetTensorInfoByName(info.is_input, name, tensor_kernel) != SUCCESS) {
    reason_oss << "the detailed info of " << in_or_out << " [" << name
               << "] of this op is not found in op information library.";
    reason = reason_oss.str();
    FE_LOGW("Get %s kernel info of Op (name [%s] type [%s] not success!", in_or_out.c_str(), op_desc.GetName().c_str(),
            op_desc.GetType().c_str());
    return false;
  }

  string tensor_name = tensor_kernel->GetUniqueName();
  // if is_force_dtype_support, do not need to check dtype
  if (!is_force_dtype_support) {
    auto iter_types_map = info.support_data_types_map.find(tensor_name);
    if (iter_types_map == info.support_data_types_map.end()) {
      REPORT_FE_ERROR("[ChkSpt][FEChk][ChkTensor][%s, %s] Failed to find the support data_types for the %s [%s].",
                      op_desc.GetName().c_str(), op_desc.GetType().c_str(), in_or_out.c_str(), tensor_name.c_str());
      return false;
    }

    if (!CheckDtypeSupported(node, tensor_desc, tensor_kernel, iter_types_map->second, info)) {
      string dtype = ge::TypeUtils::DataTypeToSerialString(tensor_desc->GetDataType());
      reason_oss << "data type " << dtype << " of " << in_or_out << " [" << name << "] is not supported. ";

      FE_LOGD("[ChkSpt][FEChk][ChkTensor][Node %s, type %s] %s %s's data type [%s] is not supported by ops kernel.",
              op_desc.GetName().c_str(), op_desc.GetType().c_str(), in_or_out.c_str(), tensor_name.c_str(),
              dtype.c_str());
      LogAllFormatAndDtype(info, tensor_name, reason_oss, reason);
      return false;
    }
  }

  auto iter_formats_map = info.suppport_formats_map.find(tensor_name);
  if (iter_formats_map == info.suppport_formats_map.end()) {
    REPORT_FE_ERROR("[ChkSpt][FEChk][ChkTensor][Node %s, type %s] Failed to find the support formats for the %s [%s].",
                    op_desc.GetName().c_str(), op_desc.GetType().c_str(), in_or_out.c_str(), tensor_name.c_str());
    return false;
  }

  if (!CheckFormatSupported(node, tensor_desc, tensor_kernel, iter_formats_map->second)) {
    string format = FormatToStr(tensor_desc->GetOriginFormat());
    reason_oss << "origin format " << format << " of " << in_or_out << " [" << name << "] is not supported. ";

    FE_LOGD("[ChkSpt][FEChk][ChkTensor][Node %s, type %s] %s %s's origin format [%s] is not supported by ops kernel.",
            op_desc.GetName().c_str(), op_desc.GetType().c_str(), in_or_out.c_str(), tensor_name.c_str(),
            format.c_str());
    LogAllFormatAndDtype(info, tensor_name, reason_oss, reason);
    return false;
  }

  // check sub_format whether support
  auto iter_sub_formats_map = info.suppport_sub_formats_map.find(tensor_name);
  if (iter_sub_formats_map == info.suppport_sub_formats_map.end()) {
    FE_LOGD("[ChkSpt][FEChk][ChkTensor][%s, %s] not find the support sub_formats for the %s [%s].",
            op_desc.GetName().c_str(), op_desc.GetType().c_str(), in_or_out.c_str(), tensor_name.c_str());
    info.suppport_sub_formats_map.emplace(std::make_pair(tensor_name, vector<uint32_t>{DEFAULT_SUB_FORMAT}));
  }
  int64_t groups = GROUPS_DEFAULT_VALUE;
  if (GetGroupAttributeWithVerify(op_desc_ptr, groups) != SUCCESS || groups > UINT32_MAX) {
    REPORT_FE_ERROR("[ChkSpt][FEChk][ChkTensor] Node[%s, %s]: The groups[%ld] value is invalid.",
                    op_desc.GetName().c_str(), op_desc.GetType().c_str(), groups);
    return false;
  }
  if (groups > GROUPS_DEFAULT_VALUE && !CheckSubformatSupported(node, groups, tensor_kernel, tensor_name, info)) {
    FE_LOGD("[ChkSpt][FEChk][ChkTensor][Node %s, type %s] %s %s's group [%ld] is not supported by ops kernel.",
            op_desc.GetName().c_str(), op_desc.GetType().c_str(), in_or_out.c_str(), tensor_name.c_str(),
            groups);
    return false;
  }
  return true;
}

bool SubOpsStore::CheckInputConstValueDepend(const ge::NodePtr &node,
                                             const string &input_name,
                                             const uint32_t &in_index,
                                             SupportedFormatAndDtype &info) const {
  std::ostringstream reason_oss;
  InputOrOutputInfoPtr tensor_kernel;
  if (info.op_kernel_info_ptr->GetTensorInfoByName(info.is_input, input_name, tensor_kernel) != SUCCESS) {
    reason_oss << "the detail info of input[" << input_name << "] of this op is not found in op information library";
    info.reason = reason_oss.str();
    FE_LOGW("Failed to get the tensor kernel of input[%s].", input_name.c_str());
    return false;
  }

  if (tensor_kernel->GetConstValueDepend() == CONST_REQUIRED) {
    auto tensor_desc = node->GetOpDesc()->MutableInputDesc(in_index);
    bool is_input_const = false;
    if (ge::AttrUtils::HasAttr(tensor_desc, ge::ATTR_NAME_VALUE)) {
      // fuzz build, GE set const value to op_desc by tensor attr, change const node to data node
      FE_LOGD("The index[%u] of node[%s] is const input.", in_index, node->GetName().c_str());
      is_input_const = true;
    } else {
      is_input_const = IsInputConst(node, in_index);
    }

    if (!is_input_const) {
      reason_oss << "The value depend of input[" << input_name << "] is required,";
      reason_oss << " but this input does not linked to a const or constant node";
      info.reason = reason_oss.str();
      FE_LOGI("%s", info.reason.c_str());
      return false;
    }
  }
  return true;
}

bool SubOpsStore::IsInputConst(const ge::NodePtr &node, const int32_t &index) const {
  auto in_anchor = node->GetInDataAnchor(index);
  if (in_anchor == nullptr) {
    return false;
  }
  auto out_anchor = in_anchor->GetPeerOutAnchor();
  if (out_anchor == nullptr || out_anchor->GetOwnerNode() == nullptr) {
    return false;
  }
  auto pre_node = out_anchor->GetOwnerNode();
  std::string node_type = pre_node->GetType();
  std::unordered_set<string> types = {CONSTANT, CONSTANTOP};
  bool matched = false;
  FeGraphUtils::IsNodeSpecificType(types, pre_node, matched);
  return matched;
}

bool SubOpsStore::CheckInputSupported(const ge::NodePtr &node, uint32_t input_size, const bool &is_force_dtype_support,
                                      SupportedFormatAndDtype &info) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  std::ostringstream reason_oss;
  for (uint32_t in_index = 0; in_index < input_size; ++in_index) {
    auto tensor_desc = op_desc_ptr->MutableInputDesc(in_index);
    if (tensor_desc == nullptr) {
      continue;
    }
    IndexNameMap::const_iterator input_index_name_map_iter = info.input_index_name_map.find(in_index);
    if (input_index_name_map_iter == info.input_index_name_map.cend()) {
      reason_oss << "the name of input[" << in_index << "] is not found";
      info.reason = reason_oss.str();
      REPORT_FE_ERROR("[ChkSpt][FEChk][ChkInSpt] Can not find the input index %u in map whose size is %zu",
                      in_index, info.input_index_name_map.size());
      return false;
    }
    string input_name = input_index_name_map_iter->second;
    info.is_input = true;
    info.cur_idx = in_index;
    if (!CheckFormatAndDtypeNormalMode(node, input_name, tensor_desc, info, is_force_dtype_support, info.reason)) {
      return false;
    }

    if (!CheckInputConstValueDepend(node, input_name, in_index, info)) {
      return false;
    }
  }
  return true;
}

bool SubOpsStore::CheckOutputSupported(const ge::NodePtr &node, uint32_t output_size,
                                       const bool &is_force_dtype_support, SupportedFormatAndDtype &info) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  ge::OpDesc &op_desc = *(op_desc_ptr.get());
  std::ostringstream reason_oss;
  // check output%d.dtype & output%d.shape supported or not
  for (uint32_t out_index = 0; out_index < output_size; ++out_index) {
    ge::GeTensorDescPtr output_desc = op_desc.MutableOutputDesc(out_index);
    if (output_desc == nullptr) {
      REPORT_FE_ERROR("[ChkSpt][FEChk][ChkOutSpt] output_desc is null.");
      return false;
    }
    const ge::GeShape &out_shape = output_desc->GetShape();
    if (!CheckTensorNotNull(out_shape)) {
      REPORT_FE_WARN("[ChkSpt][FEChk][ChkOutSpt] Node[%s, %s] output index %u shape:%s not valid.",
                     node->GetName().c_str(), node->GetType().c_str(), out_index,
                     ShapeToString(out_shape.GetDims()).c_str());
    }
    IndexNameMap::const_iterator output_index_name_map_iter = info.output_index_name_map.find(out_index);
    if (output_index_name_map_iter == info.output_index_name_map.cend()) {
      reason_oss << "the name of output[" << out_index << "] is not found";
      info.reason = reason_oss.str();
      REPORT_FE_ERROR("[ChkSpt][FEChk][ChkOutSpt] Can not find the output index %u in map.", out_index);
      return false;
    }
    string output_name = output_index_name_map_iter->second;
    info.is_input = false;
    if (!CheckFormatAndDtypeNormalMode(node, output_name, output_desc, info, is_force_dtype_support, info.reason)) {
      return false;
    }
  }
  return true;
}

bool SubOpsStore::CheckSingleTensorSupportedAccurateMode(const ge::NodePtr &node, uint32_t index, uint32_t type_index,
                                                         SupportedFormatAndDtype &info, bool &check_flag) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  ge::OpDesc &op_desc = *(op_desc_ptr.get());
  string input_or_output = IS_INPUT_TO_STRING(info.is_input);
  std::ostringstream reason_oss;
  ge::ConstGeTensorDescPtr tensor_desc;
  string name;
  if (info.is_input) {
    tensor_desc = op_desc.GetInputDescPtr(index);
    if (!GetTensorName(info, input_or_output, index, info.input_index_name_map, name)) {
      return false;
    }
  } else {
    tensor_desc = op_desc.GetOutputDescPtr(index);
    if (!GetTensorName(info, input_or_output, index, info.output_index_name_map, name)) {
      return false;
    }
  }
  if (tensor_desc == nullptr) {
    REPORT_FE_ERROR("[ChkSpt][FEChk][ChkSigTensSptAccrMode] tensor_desc is null.");
    return false;
  }
  InputOrOutputInfoPtr tensor_info_ptr;
  if (info.op_kernel_info_ptr->GetTensorInfoByName(info.is_input, name, tensor_info_ptr) != SUCCESS) {
    reason_oss << "the detail info of " << input_or_output << " [" << name
               << "] of this op is not found in op information library";
    info.reason = reason_oss.str();
    FE_LOGW("Get %s kernel info of Op (name [%s] type [%s]) not success!", input_or_output.c_str(),
            op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return false;
  }

  string unique_name = tensor_info_ptr->GetUniqueName();
  if (info.support_data_types_map.find(unique_name) == info.support_data_types_map.end()) {
    REPORT_FE_ERROR("[ChkSpt][FEChk][ChkSigTensSptAccrMode] Node[%s, %s]: failed to find supported dtypes of %s[%s].",
                    op_desc.GetName().c_str(), op_desc.GetType().c_str(), input_or_output.c_str(), unique_name.c_str());
    return false;
  }
  if (info.suppport_formats_map.find(unique_name) == info.suppport_formats_map.end()) {
    REPORT_FE_ERROR("[ChkSpt][FEChk][ChkSigTensSptAccrMode] Node[%s, %s]: failed to find supported formats of %s[%s]",
                    op_desc.GetName().c_str(), op_desc.GetType().c_str(), input_or_output.c_str(), unique_name.c_str());
    return false;
  }

  check_flag = !CheckAccuracyDtypeSupported(tensor_desc, type_index, tensor_info_ptr,
                                            info.support_data_types_map.at(unique_name)) ||
               !CheckAccuracyFormatSupported(tensor_desc, type_index, tensor_info_ptr,
                                             info.suppport_formats_map.at(unique_name)) ||
               !VerifyFormatC0Val(op_desc_ptr, tensor_desc);
  return true;
}

bool SubOpsStore::CheckSingleTensorAccurateMode(uint32_t size, uint32_t type_index, const ge::NodePtr &node,
                                                SupportedFormatAndDtype &info, bool &need_continue) const {
  for (uint32_t in_index = 0; in_index < size; ++in_index) {
    bool check_flag = false;
    if (!CheckSingleTensorSupportedAccurateMode(node, in_index, type_index, info, check_flag)) {
      return false;
    }
    if (check_flag) {
      need_continue = true;
      break;
    }
  }
  return true;
}

bool SubOpsStore::CheckAllTensorsSupportedAccurateMode(const ge::NodePtr &node, SupportedFormatAndDtype &info) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  ge::OpDesc &op_desc = *(op_desc_ptr.get());
  ge::OpDesc::Vistor<ge::GeTensorDescPtr> all_input_desc_ptr_vistor = op_desc.GetAllInputsDescPtr();
  ge::OpDesc::Vistor<ge::GeTensorDescPtr> all_output_desc_ptr_vistor = op_desc.GetAllOutputsDescPtr();
  auto input_size = static_cast<uint32_t>(all_input_desc_ptr_vistor.size());
  auto output_size = static_cast<uint32_t>(all_output_desc_ptr_vistor.size());

  UnSupportedReason reason_tmp;
  bool ret = GetInputOutputNameMap(node, info.op_kernel_info_ptr, info.input_index_name_map,
                                   info.output_index_name_map, reason_tmp);
  if (!ret) {
    info.reason = reason_tmp.reason;
    return false;
  }

  if (info.op_kernel_info_ptr->GetAllInputInfo().size() == 0) {
    info.reason = "inputs size of op_kernel_info is 0";
    return false;
  }

  uint32_t type_size = static_cast<uint32_t>(info.support_data_types_map.cbegin()->second.size());
  for (uint32_t type_index = 0; type_index < type_size; type_index++) {
    bool need_continue = false;
    need_continue = false;
    // check input%d.dtype & input%d.shape supported or not
    info.is_input = true;
    if (!CheckSingleTensorAccurateMode(input_size, type_index, node, info, need_continue)) {
      return false;
    }
    if (need_continue) {
      continue;
    }
    info.is_input = false;
    // check output%d.dtype & output%d.shape supported or not
    if (!CheckSingleTensorAccurateMode(output_size, type_index, node, info, need_continue)) {
      return false;
    }
    if (need_continue) {
      continue;
    }
    return true;
  }
  info.reason = "The format and dtype is not precisely equivalent to format and dtype in op information library.";
  return false;
}

bool SubOpsStore::CheckOpSupported(const ge::NodePtr &node_ptr, const bool &is_dynamic_impl,
                                   CheckSupportParam &check_param) const {
  if (!check_param.op_kernel_ptr->IsNeedCheckSupport()) {
    FE_LOGD("[ChkSpt][OpChk][Node %s, type %s] This op is not configure to check it's implementation.",
            node_ptr->GetName().c_str(), node_ptr->GetType().c_str());
    return true;
  }
  OpStoreAdapterPtr op_store_adapter =
          OpStoreAdapterManager::Instance(engine_name_).GetOpStoreAdapter(sub_store_info_.op_impl_type);
  if (op_store_adapter == nullptr) {
    REPORT_FE_ERROR("[GraphOpt][SubChk][GetTbeAdap] Failed to get op adapter by op impl type[%ld].",
                    sub_store_info_.op_impl_type);
    return false;
  }

  std::ostringstream reason_oss;
  std::string unsupported_reason_in_op;
  if (!op_store_adapter->CheckSupport(node_ptr, check_param, is_dynamic_impl, unsupported_reason_in_op)) {
    OpNotSupportedReasonID reason_id = OpNotSupportedReasonID::EN_OPERATOR_NOT_SUPPORT;
    FE_LOGI("[ChkSpt][OpChk][Node %s, type %s] This op is not supported inside its implementation. Reason is %s",
            node_ptr->GetName().c_str(), node_ptr->GetType().c_str(), unsupported_reason_in_op.c_str());
    reason_oss << "Op " << node_ptr->GetName() << " type " << node_ptr->GetType().c_str()
               << " is not supported in the operator's implementation. Reason is: " << unsupported_reason_in_op << ".";
    uint64_t offset_num = sub_store_info_.op_impl_type * NOT_SUPPORTED_REASON_OFFSET_BIT_NUM;
    check_param.unsupport_reason.reason_id += (static_cast<uint64_t>(reason_id) << offset_num);
    check_param.unsupport_reason.reason += reason_oss.str();
    return false;
  }

  return true;
}

void SubOpsStore::SetAttrParamTypeList(const ge::OpDescPtr &op_desc, const OpKernelInfoPtr &op_kernel_info_ptr,
    const SupportedFormatAndDtype &info) const {
  std::vector<uint32_t> input_type_list;
  std::vector<std::string> input_name_list;
  for (const auto &index_name : info.input_index_name_map) {
    std::string input_name = index_name.second;
    InputOrOutputInfoPtr input_info;
    if (op_kernel_info_ptr->GetInputInfoByName(input_name, input_info) != fe::SUCCESS) {
      FE_LOGW("[ChkSpt][SubChk][SetAttrParamType] Get kernel in info for op[%s, %s] failed.",
              op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return;
    }
    input_type_list.emplace_back(static_cast<uint32_t>(input_info->GetParamType()));
    input_name_list.emplace_back(input_info->GetName());
  }
  (void)ge::AttrUtils::SetListInt(op_desc, kInputParaTypeList, input_type_list);
  (void)ge::AttrUtils::SetListStr(op_desc, kInputNameList, input_name_list);
  std::vector<uint32_t> output_type_list;
  std::vector<std::string> output_name_list;
  for (const auto &index_name : info.output_index_name_map) {
    std::string output_name = index_name.second;
    InputOrOutputInfoPtr output_info;
    if (op_kernel_info_ptr->GetOutputInfoByName(output_name, output_info) != fe::SUCCESS) {
      FE_LOGW("[ChkSpt][SubChk][SetAttrParamType] Get kernel out info for op[%s, %s] failed.",
              op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return;
    }
    output_type_list.emplace_back(static_cast<uint32_t>(output_info->GetParamType()));
    output_name_list.emplace_back(output_info->GetName());
  }
  (void)ge::AttrUtils::SetListInt(op_desc, kOutputParaTypeList, output_type_list);
  (void)ge::AttrUtils::SetListStr(op_desc, kOutputNameList, output_name_list);
  FE_LOGD("[GraphOpt][SetAttrParamType] Finish to set attr para_type_list for op[%s, %s].",
          op_desc->GetName().c_str(), op_desc->GetType().c_str());
  return;
}

void SubOpsStore::TransInputNameToIdx(const std::vector<std::string> &promote_item, const ge::NodePtr &node,
                                      std::vector<std::vector<int>> &promote_list) const {
  std::vector<int> promote_item_vec;
  for (const auto &input_name : promote_item) {
    int idx = node->GetOpDesc()->GetInputIndexByName(input_name);
    FE_LOGD("Trans input name [%s] to index[%d] for node[%s, %s].",
            input_name.c_str(), idx, node->GetNamePtr(), node->GetTypePtr());
    promote_item_vec.emplace_back(idx);
  }
  if (promote_item_vec.size() > 1) {
    promote_list.emplace_back(promote_item_vec);
  }
}


bool SubOpsStore::ParsePromoteStr(const PromoteTypeVal &promote_type_val, const ge::NodePtr &node,
    const OpKernelInfoPtr &op_kernel_info_ptr, std::vector<std::vector<int>> &promote_list) const {
  if (!promote_type_val.is_dynamic) {
    for (const auto &promote_item : promote_type_val.promote_val) {
      TransInputNameToIdx(promote_item, node, promote_list);
    }
    return true;
  }
  OpStoreAdapterPtr op_store_adapter =
          OpStoreAdapterManager::Instance(engine_name_).GetOpStoreAdapter(sub_store_info_.op_impl_type);
  if (op_store_adapter == nullptr) {
    REPORT_FE_ERROR("[GraphOpt][SubChk][GetTbeAdap] Failed to get op adapter by op impl type[%ld].",
                    sub_store_info_.op_impl_type);
    return false;
  }
  std::string promote_str_dynamic;
  (void)op_store_adapter->GetDynamicPromoteType(node, op_kernel_info_ptr, promote_str_dynamic);
  if (promote_str_dynamic.empty()) {
    return false;
  }
  std::vector<std::string> promote_val_vec;
  auto pos = promote_str_dynamic.find(';');
  if (pos == string::npos) {
    promote_val_vec.emplace_back(promote_str_dynamic);
  } else {
    StringUtils::SplitWithTrim(promote_str_dynamic, ';', promote_val_vec);
  }
  for (const std::string &promote_val : promote_val_vec) {
    auto pos_tmp = promote_val.find(',');
    if (pos_tmp == string::npos) {
      continue;
    }
    std::vector<std::string> promote_item;
    StringUtils::SplitWithTrim(promote_val, ',', promote_item);
    if (promote_item.empty() || promote_item.size() < kMinPromoteSize) {
      continue;
    }
    TransInputNameToIdx(promote_item, node, promote_list);
  }
  return true;
}

void SubOpsStore::GetPromoteInputList(const ge::NodePtr &node, SupportedFormatAndDtype &info) const {
  if (info.op_kernel_info_ptr == nullptr) {
    FE_LOGW("Node[%s, %s] op kernel info ptr is nullptr, unexpected.",
            node->GetNamePtr(), node->GetTypePtr());
    return;
  }
  PromoteTypeVal promote_type_val;
  info.op_kernel_info_ptr->GetPromoteInfo(promote_type_val);
  if (!promote_type_val.promote_flag) {
    return;
  }
  if (!ParsePromoteStr(promote_type_val, node, info.op_kernel_info_ptr, info.promote_input_list)) {
    FE_LOGW("Node[%s, %s] get promote type val from get_op_specific_info is null.",
            node->GetNamePtr(), node->GetTypePtr());
    return;
  }
}

void SubOpsStore::GetPromoteFlag(const ge::NodePtr &node, SupportedFormatAndDtype &info) const {
  for (const auto &promote_index : info.promote_input_list) {
    if (promote_index.size() < kMinPromoteSize) {
      ge::DataType dfl_dtype = ge::DT_UNDEFINED;
      info.promote_target_type.emplace_back(dfl_dtype);
      continue;
    }

    const auto &input_desc_0 = node->GetOpDesc()->MutableInputDesc(promote_index[0]);
    const auto &input_desc_1 = node->GetOpDesc()->MutableInputDesc(promote_index[1]);
    if (input_desc_0 == nullptr || input_desc_1 == nullptr) {
      FE_LOGW("Node[%s, %s] get inputDesc by promote input idx [%u, %u] is nullptr.",
              node->GetNamePtr(), node->GetTypePtr(), promote_index[0], promote_index[1]);
      return;
    }

    auto promote_dtype_0 = input_desc_0->GetDataType();
    auto promote_dtype_1 = input_desc_1->GetDataType();
    ge::DataType target_dtype = opcommon::PromoteType(promote_dtype_0, promote_dtype_1);
    if (target_dtype != promote_dtype_0 || target_dtype != promote_dtype_1) {
      info.promote_flag = true;
    }

    size_t inputs_size = node->GetOpDesc()->GetAllInputsSize();
    for (size_t i = 2; i < promote_index.size(); ++i) {
      if (static_cast<size_t>(promote_index[i]) >= inputs_size) {
        FE_LOGW("Node[%s, %s] promote input idx [%u] is invalid, while inputs size is %zu.",
                node->GetNamePtr(), node->GetTypePtr(), promote_index[i], inputs_size);
        return;
      }
      const auto &input_desc = node->GetOpDesc()->MutableInputDesc(promote_index[i]);
      if (input_desc == nullptr) {
        FE_LOGW("Node[%s, %s] get inputDesc by promote input idx [%u] is nullptr.",
                node->GetNamePtr(), node->GetTypePtr(), promote_index[i]);
        return;
      }
      auto cur_dtype = input_desc->GetDataType();
      if (target_dtype != cur_dtype) {
        info.promote_flag = true;
        target_dtype = opcommon::PromoteType(cur_dtype, target_dtype);
      }
    }
    info.promote_target_type.emplace_back(target_dtype);
  }
}

void SubOpsStore::FeedPromoteInfo(const ge::NodePtr &node, SupportedFormatAndDtype &info) const {
  GetPromoteInputList(node, info);
  if (info.promote_input_list.empty()) {
    FE_LOGW("Node[%s, %s] get promote input list from ge is empty.",
            node->GetNamePtr(), node->GetTypePtr());
    return;
  }
  GetPromoteFlag(node, info);
  if (info.promote_flag) {
    (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kMustPromoteFlag, info.promote_flag);
    node->GetOpDesc()->SetExtAttr(kPromoteInfo, info.promote_input_list);
  }
  FE_LOGD("Node[%s, %s] promote flag is %d, target dtype vec size is %zu.",
          node->GetNamePtr(), node->GetTypePtr(), info.promote_flag, info.promote_target_type.size());
}

bool SubOpsStore::CheckSubStoreSupported(const ge::NodePtr &node, const CheckSupportMode &check_mode,
                                         const bool &check_unknown_shape, CheckSupportParam &check_param) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  ge::OpDesc &op_desc = *(op_desc_ptr.get());
  string check_type = check_unknown_shape ? "[Dynamic shape check]: " : "[Static shape check]:";
  check_param.unsupport_reason.reason += check_type;
  FE_LOGD("[ChkSpt][FEChk][SubChk][Node %s, type %s] %s in lib %s by check_mode %u.", op_desc.GetName().c_str(),
          op_desc.GetType().c_str(), check_type.c_str(), sub_store_info_.fe_ops_store_name.c_str(),
          static_cast<uint32_t>(check_mode));

  auto input_size = static_cast<uint32_t>(op_desc.GetAllInputsSize());
  auto output_size = static_cast<uint32_t>(op_desc.GetOutputsSize());

  SupportedFormatAndDtype info(check_param.op_kernel_ptr, "");
  if (!GetInputOutputNameMap(node, check_param.op_kernel_ptr, info.input_index_name_map,
                             info.output_index_name_map, check_param.unsupport_reason)) {
    return false;
  }
  FeedPromoteInfo(node, info);
  SetAttrParamTypeList(node->GetOpDesc(), check_param.op_kernel_ptr, info);
  FormatDtypeInfo format_dtype_info;
  if (GetSupportFormatAndDtype(node, check_param.op_kernel_ptr,
                               check_unknown_shape, format_dtype_info) != SUCCESS) {
    FE_LOGW("[GraphOpt][Setcheck][CheckSubSupt] Failed to get the GetSupportFormatAndDtype, return false.");
    return false;
  }
  if (!format_dtype_info.format_map.empty() && !format_dtype_info.data_type_map.empty()) {
    info.suppport_formats_map = std::move(format_dtype_info.format_map);
    info.support_data_types_map = std::move(format_dtype_info.data_type_map);
    info.suppport_sub_formats_map = std::move(format_dtype_info.sub_format_map);
  }

  bool is_force_dtype_support = false;
  if (!CheckCustomizeDtype(node->GetOpDesc(), info, is_force_dtype_support)) {
    FE_LOGI("[GraphOpt][Setcheck][CheckSubSupt] The custom dtypes for op[%s, %s] is not support by its op kernel.",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
    SetReason(info.reason, OpNotSupportedReasonID::EN_INPUTS_AND_OUTPUTS_NOT_ACCURACY_SUPPORT,
              check_param.unsupport_reason);
    return false;
  }

  if (check_mode == CheckSupportMode::ACCURACY_MODE) {
    if (!CheckAllTensorsSupportedAccurateMode(node, info)) {
      SetReason(info.reason, OpNotSupportedReasonID::EN_INPUTS_AND_OUTPUTS_NOT_ACCURACY_SUPPORT,
                check_param.unsupport_reason);
      return false;
    }
  } else {
    if (!CheckInputSupported(node, input_size, is_force_dtype_support, info)) {
      SetReason(info.reason, OpNotSupportedReasonID::EN_INPUTS_NOT_SUPPORT, check_param.unsupport_reason);
      FE_LOGD("Node %s, type %s's input is not supported in %s.", op_desc.GetName().c_str(), op_desc.GetType().c_str(),
              sub_store_info_.fe_ops_store_name.c_str());
      return false;
    }
    if (!CheckOutputSupported(node, output_size, is_force_dtype_support, info)) {
      SetReason(info.reason, OpNotSupportedReasonID::EN_OUTPUTS_NOT_SUPPORT, check_param.unsupport_reason);
      FE_LOGD("Node %s, type %s's output is not supported in %s.", op_desc.GetName().c_str(),
              op_desc.GetType().c_str(), sub_store_info_.fe_ops_store_name.c_str());
      return false;
    }
  }
  // check attrs are supported or not
  if (!CheckAttrSupport(node, *(check_param.op_kernel_ptr.get()),
      check_param.unsupport_reason.reason)) {
    check_param.unsupport_reason.reason_id = static_cast<uint64_t>(OpNotSupportedReasonID::EN_ATTRS_NOT_SUPPORT);
    FE_LOGD("Attr is not supported in %s.", sub_store_info_.fe_ops_store_name.c_str());
    return false;
  }

  // input or output dynamic can have more than 1, required must equal 1,optional not check;
  if (!CheckParamType(node, check_param.op_kernel_ptr, info.input_index_name_map,
      info.output_index_name_map, check_param.unsupport_reason.reason)) {
    check_param.unsupport_reason.reason_id = static_cast<uint64_t>(OpNotSupportedReasonID::EN_PARAM_TYPE_NOT_SUPPORT);
    FE_LOGD("Parameter type is not supported in %s.", sub_store_info_.fe_ops_store_name.c_str());
    return false;
  }

  // check op supported
  if (!CheckOpSupported(node, check_unknown_shape, check_param)) {
    return false;
  }

  // set is_ge_op attr
  if (check_param.op_kernel_ptr->IsOpFileNull()) {
    (void)ge::AttrUtils::SetBool(op_desc_ptr, IS_GE_OP, true);
  }

  FE_LOGD("[ChkSpt][FEChk][%s][Node %s, type %s] is support by op store [%s] by check_mode %u.",
          check_type.c_str(), op_desc.GetName().c_str(), op_desc.GetType().c_str(),
          sub_store_info_.fe_ops_store_name.c_str(),
          static_cast<uint32_t>(check_mode));
  return true;
}

bool SubOpsStore::CheckCustomizeDtype(const ge::OpDescPtr &op_desc, const SupportedFormatAndDtype &info,
                                      bool &is_force_dtype_support) const {
  if (info.support_data_types_map.empty()) {
    return true;
  }

  OpCustomizeDtype op_cust_dtypes;
  if (!Configuration::Instance(engine_name_).GetCustomizeDtypeByOpName(op_desc->GetName(), op_cust_dtypes)) {
    if (!Configuration::Instance(engine_name_).GetCustomizeDtypeByOpType(op_desc->GetType(), op_cust_dtypes)) {
      return true;
    }
  }

  FE_LOGD("Op customize dtype is found for op[%s, %s]. Input dtype size is %zu, output dtype size is %zu.",
          op_desc->GetName().c_str(), op_desc->GetType().c_str(),
          op_cust_dtypes.input_dtypes.size(), op_cust_dtypes.output_dtypes.size());
  vector<bool> filer_index(info.support_data_types_map.begin()->second.size(), true);
  FilterDtypesByCustom(op_cust_dtypes.input_dtypes, info.input_index_name_map, info, true, filer_index);
  FilterDtypesByCustom(op_cust_dtypes.output_dtypes, info.output_index_name_map, info, false, filer_index);

  is_force_dtype_support = std::find(filer_index.begin(), filer_index.end(), true) != filer_index.end();
  return is_force_dtype_support;
}

void SubOpsStore::FilterDtypesByCustom(const vector<ge::DataType> &cust_dtypes,
                                       const IndexNameMap &index_map,
                                       const SupportedFormatAndDtype &info,
                                       const bool &is_input,
                                       vector<bool> &filer_index) const {
  for (size_t i = 0; i < cust_dtypes.size(); i++) {
    if (cust_dtypes[i] == ge::DT_UNDEFINED) {
      continue;
    }
    IndexNameMap::const_iterator iter_index = index_map.find(i);
    if (iter_index == index_map.cend()) {
      continue;
    }

    InputOrOutputInfoPtr tensor_kernel = nullptr;
    if (info.op_kernel_info_ptr->GetTensorInfoByName(is_input, iter_index->second, tensor_kernel) != SUCCESS) {
      continue;
    }

    map<string, vector<ge::DataType>>::const_iterator iter_name =
            info.support_data_types_map.find(tensor_kernel->GetUniqueName());
    if (iter_name == info.support_data_types_map.cend()) {
      continue;
    }
    for (size_t j = 0; j < iter_name->second.size(); j++) {
      if (cust_dtypes[i] != iter_name->second[j]) {
        filer_index[j] = false;
      }
    }
  }
}

Status SubOpsStore::GetSupportFormatAndDtype(const ge::NodePtr &node,
                                             const OpKernelInfoPtr &op_kernel_info_ptr,
                                             const bool &is_dynamic_check,
                                             FormatDtypeInfo &format_dtype_info) const {
  return this->format_dtype_querier_ptr_->GetSupportFormatAndDtype(op_kernel_info_ptr, node, is_dynamic_check,
                                                                   format_dtype_info);
}

bool SubOpsStore::IsAllDtypeExpected(ge::DataType dtype_expected, ge::DataType dtype_unexpected,
                                     const vector<ge::DataType> &dtypes) const {
  for (const auto &dtype : dtypes) {
    if (dtype == dtype_unexpected) {
      return false;
    }
  }
  for (const auto &dtype : dtypes) {
    if (dtype == dtype_expected) {
      return true;
    }
  }
  return false;
}

bool SubOpsStore::IsSupportExpectedDtype(ge::DataType dtype_expected, const vector<ge::DataType> &dtypes) const {
  for (const auto &dtype : dtypes) {
    if (dtype == dtype_expected) {
      return true;
    }
  }
  return false;
}

bool SubOpsStore::CheckTensorNotNull(const ge::GeShape &ge_shape) {
  for (const auto &dim : ge_shape.GetDims()) {
    if (dim == SHAPE_NUMBER_0) {
      return false;
    }
  }
  return true;
}
}  // namespace fe
