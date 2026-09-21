/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_store/op_kernel_info_constructor.h"
#include "register/op_registry.h"
#include "common/fe_log.h"
#include "common/aicore_util_types.h"
#include "common/aicore_util_constants.h"
#include "common/fe_type_utils.h"
#include "common/string_utils.h"
#include "ops_store/ops_kernel_constants.h"
#include "ops_store/ops_kernel_utils.h"
#include "ops_store/ops_kernel_error_codes.h"
#include "base/err_msg.h"
#include "common/fe_error_code.h"
#include "transfer_shape_utils.h"
#include "common/configuration.h"

namespace fe {
using std::string;
using ge::GeAttrValue;
namespace {
const std::string kStageInitOpKernel = "[InitOpInfoLib][InitOpKernel]";
const std::map<std::string, ge::GeAttrValue::ValueType> kAttrTypeMap {
        {STR_INT, ge::GeAttrValue::VT_INT},
        {STR_FLOAT, ge::GeAttrValue::VT_FLOAT},
        {STR_BOOL, ge::GeAttrValue::VT_BOOL},
        {STR_STR, ge::GeAttrValue::VT_STRING},
        {STR_LIST_INT, ge::GeAttrValue::VT_LIST_INT},
        {STR_LIST_FLOAT, ge::GeAttrValue::VT_LIST_FLOAT},
        {STR_LIST_BOOL, ge::GeAttrValue::VT_LIST_BOOL},
        {STR_LIST_STR, ge::GeAttrValue::VT_LIST_STRING},
        {STR_LIST_LIST_INT, ge::GeAttrValue::VT_LIST_LIST_INT}};

// tuple<bool, string, string> default value, item_name, item_key
const std::map<OP_KERNEL_FLAG, std::tuple<bool, string, string>> kOpkernelFlagMap {
        {OP_KERNEL_FLAG::NeedCheckSupport, {false, kStrNeedCheckSupport, STR_FLAG}},
        {OP_KERNEL_FLAG::DynamicFormat, {false, kStrDynamicFormat, STR_FLAG}},
        {OP_KERNEL_FLAG::HeavyOp, {false, STR_HEAVY_OP, STR_FLAG}},
        {OP_KERNEL_FLAG::SoftSyncOp, {false, kSoftSync, STR_FLAG}},
        {OP_KERNEL_FLAG::SupportDynamicShape, {false, kStrDynamicShapeSupport, STR_FLAG}},
        {OP_KERNEL_FLAG::InputMemContinues, {false, kStrInputMemContinues, STR_FLAG}},
        {OP_KERNEL_FLAG::OutputMemContinues, {false, kStrOutputMemContinues, STR_FLAG}}
};

// tuple<int64_t, string, string, map<string, int64_t>> default value, item_name, item_key, str_item_map
const std::map<OP_KERNEL_PARAM, std::tuple<int64_t, string, string, std::map<string, int64_t>>> kOpkernelParamMap {
        {OP_KERNEL_PARAM::OpPattern,
         {static_cast<int64_t>(OP_PATTERN_OP_KERNEL), STR_OP, STR_PATTERN, kOpPatternMap}},
        {OP_KERNEL_PARAM::PrecisionPolicy,
         {static_cast<int64_t>(GRAY), STR_PRECISION_POLICY, STR_FLAG, kStrPrecisionPolicyMap}},
        {OP_KERNEL_PARAM::OpSlicePattern,
         {static_cast<int64_t>(PATTERN_RESERVED), STR_SLICE_PATTERN, STR_VALUE, kStrSlicePatternMap}},
        {OP_KERNEL_PARAM::JitCompileType,
         {static_cast<int64_t>(JitCompile::DEFAULT), kStrJitCompile, STR_FLAG, kStrJitCompileMap}},
        {OP_KERNEL_PARAM::RangeLimitType,
         {static_cast<int64_t>(RangeLimitType::DEFAULT), kStrRangeLimit, STR_VALUE, kStrRangeLimitTypeMap}},
        {OP_KERNEL_PARAM::DynamicRankType,
         {static_cast<int64_t>(DynamicRankType::NOT_SUPPORT), kStrDynamicRankSupport, STR_FLAG,
          kStrDynamicRankTypeMap}},
        {OP_KERNEL_PARAM::DynamicCompileStatic,
         {static_cast<int64_t>(DynamicCompileStatic::NOT_SUPPORT), kStrDynamicCompileStatic, STR_FLAG,
          kStrDynamicCompileStaticMap}},
        {OP_KERNEL_PARAM::AclnnSupportType,
         {static_cast<int64_t>(AclnnSupportType::DEFAULT), kAclnnSupport, STR_VALUE,
          kAclnnSupportStrParamToEnumMap}},
        {OP_KERNEL_PARAM::MultiKernelSupportType,
         {static_cast<int64_t>(MultiKernelSupportType::DEFAULT), kMultiKernelSupportDynamicGraph, STR_VALUE,
          kMultiKernelSupportStrParamToEnumMap}},
        {OP_KERNEL_PARAM::EnableVectorCore,
         {static_cast<int64_t>(VectorCoreType::DEFAULT), kStrEnableVectorCore, STR_FLAG,
          kEnableVectorCoreStrParamToEnumMap}}
};

// tuple<bool, string, string> need_split, item_name, item_key
const std::map<OP_KERNEL_STR_PARAM, std::tuple<string, string>> kOpkernelStrParamMap {
        {OP_KERNEL_STR_PARAM::OpImpPath, {STR_IMP_PATH, STR_PATH}},
        {OP_KERNEL_STR_PARAM::CoreType, {kCoreType, STR_VALUE}},
        {OP_KERNEL_STR_PARAM::OpImplSwitch, {kStrOpImplSwitch, STR_VALUE}},
        {OP_KERNEL_STR_PARAM::PrebuildPattern, {kStrPrebuildPattern, STR_VALUE}},
        {OP_KERNEL_STR_PARAM::PromoteType, {kStrPromoteType, STR_VALUE}}
};
}

OpKernelInfoConstructor::OpKernelInfoConstructor() {}

OpKernelInfoConstructor::~OpKernelInfoConstructor() {}

Status OpKernelInfoConstructor::ParseBasicParameter(const OpContent &op_content, OpKernelInfoPtr op_kernel_info) const {
  // init op kernel flag
  op_kernel_info->op_flag_vec_.fill(false);
  for (const auto &flag_item : kOpkernelFlagMap) {
    bool flag_value = std::get<0>(flag_item.second);
    Status flag_ret = InitOpItemValueByMaping(std::get<1>(flag_item.second), std::get<2>(flag_item.second), op_content,
                                              kStrBoolMap, flag_value);
    if (flag_ret != SUCCESS) {
      return flag_ret;
    }
    op_kernel_info->op_flag_vec_[static_cast<size_t>(flag_item.first)] = flag_value;
  }

  op_kernel_info->op_param_vec_.fill(0);
  for (const auto &param_item : kOpkernelParamMap) {
    int64_t param_value = std::get<0>(param_item.second);
    Status ret = InitOpItemValueByMaping(std::get<1>(param_item.second), std::get<2>(param_item.second), op_content,
                                         std::get<3>(param_item.second), param_value);
    if (ret != SUCCESS) {
      return ret;
    }
    op_kernel_info->op_param_vec_[static_cast<size_t>(param_item.first)] = param_value;
  }

  // if dynamicFormat is true, op pattern should be op customize
  if (op_kernel_info->op_flag_vec_[static_cast<size_t>(OP_KERNEL_FLAG::DynamicFormat)]) {
    op_kernel_info->op_param_vec_[static_cast<size_t>(OP_KERNEL_PARAM::OpPattern)] =
            static_cast<int64_t>(OP_PATTERN_OP_CUSTOMIZE);
  }

  for (const auto &str_param_item : kOpkernelStrParamMap) {
    std::string str_value;
    std::vector<std::string> str_value_vec;
    Status status = InitOpStrItem(std::get<0>(str_param_item.second), std::get<1>(str_param_item.second),
                                  op_content, str_value);
    if (status != SUCCESS) {
      return status;
    }
    op_kernel_info->op_str_param_vec_[static_cast<size_t>(str_param_item.first)] = str_value;
    StringUtils::SplitWithTrim(str_value, ',', str_value_vec);
    FE_LOGD("%s of operation type [%s] is [%s], vector is [%s].",
            (std::get<0>(str_param_item.second)).c_str(), op_kernel_info->GetOpType().c_str(),
            str_value.c_str(), StringVecToString(str_value_vec).c_str());
    op_kernel_info->op_str_param_2_vec_[static_cast<size_t>(str_param_item.first)] = str_value_vec;
  }
  ParsePromoteStr(op_kernel_info);
  ParseOpOutputInplaceAbility(op_content, op_kernel_info);
  return SUCCESS;
}

void OpKernelInfoConstructor::ParseOpOutputInplaceAbility(const OpContent &op_content,
                                                          OpKernelInfoPtr &op_kernel_info) const {
  std::string str_value;
  Status status = InitOpStrItem(kStrOutputInplaceAbility, STR_FLAG, op_content, str_value);
  if (status != SUCCESS) {
    return;
  }
  // tbe str__value: {{0,0},{0,1}.....}
  vector<vector<int64_t>> output_inplace_ability;
  StringUtils::TransStringToListListInt(str_value, output_inplace_ability);
  op_kernel_info->SetOutputIplaceInfo(output_inplace_ability);
}

void OpKernelInfoConstructor::ParsePromoteStr(OpKernelInfoPtr op_kernel_info) const {
  PromoteTypeVal promote_type_val;
  std::string promote_str =
      op_kernel_info->op_str_param_vec_[static_cast<size_t>(OP_KERNEL_STR_PARAM::PromoteType)];
  if (promote_str.empty()) {
    return;
  }
  promote_type_val.promote_flag = true;
  if (promote_str == "dynamic") {
    promote_type_val.is_dynamic = true;
    op_kernel_info->SetPromoteInfo(promote_type_val);
    return;
  }
  promote_type_val.is_dynamic = false;
  std::vector<std::string> promote_val_vec;
  auto pos = promote_str.find(';');
  if (pos == string::npos) {
    promote_val_vec.emplace_back(promote_str);
  } else {
    StringUtils::SplitWithTrim(promote_str, ';', promote_val_vec);
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
    promote_type_val.promote_val.emplace_back(promote_item);
    FE_LOGD("Parse promoteType value of operation type [%s] is [%s], vector is [%s].",
            op_kernel_info->GetOpType().c_str(),
            promote_str.c_str(), StringVecToString(promote_item).c_str());
  }
  if (promote_type_val.promote_val.empty()) {
    promote_type_val.promote_flag = false;
    return;
  }
  op_kernel_info->SetPromoteInfo(promote_type_val);
  return;
}

Status OpKernelInfoConstructor::InitializeOpKernelInfo(const std::string &engine_name, const OpContent &op_content,
                                                       OpKernelInfoPtr op_kernel_info) {
  FE_CHECK(op_kernel_info == nullptr,
           FE_LOGW("OpKernelInfo is a null pointer in the InitializeOpKernelInfo function!"),
           return SUCCESS);

  if (op_kernel_info->init_flag_) {
    FE_LOGD("OpKernelInfo has been initialized.");
    return SUCCESS;
  }

  op_kernel_info->op_type_ = op_content.op_type_;
  op_kernel_info->ops_path_name_prefix_ = op_content.ops_path_name_prefix_;
  FE_LOGD("[InitOpInfoLib][InitOpKernel] Set ops_path_name_prefix[%s] of op[%s]",
          op_kernel_info->ops_path_name_prefix_.c_str(), op_kernel_info->op_type_.c_str());

  if (ParseBasicParameter(op_content, op_kernel_info) != SUCCESS) {
    return FAILED;
  }

  // parse the input and output tensor_desc_info
  if (ParseInputAndOutputFromOpContent(op_content, op_kernel_info) != SUCCESS) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Failed to initialize inputs and outputs for Op %s.",
                    op_kernel_info->op_type_.c_str());
    return FAILED;
  }

  if (CheckFormatAgnosticOp(op_kernel_info) != SUCCESS) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Op %s check the format agnostic failed.",
                    op_kernel_info->op_type_.c_str());
    return FAILED;
  }

  // parse refer tensor name, input tensor and output has the same name
  ParserReferTensorNameVec(op_kernel_info);

  // parse the attr_info
  if (InitAttrValue(op_content, op_kernel_info) != SUCCESS) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Failed to initialize AttrValue for Op %s.", op_kernel_info->op_type_.c_str());
    return FAILED;
  }

  // parse OpInfo
  if (InitOpInfo(engine_name, op_content, op_kernel_info) != SUCCESS) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Failed to initialize OpInfo for Op %s.", op_kernel_info->op_type_.c_str());
    return FAILED;
  }
  if (InitFallback(op_content, op_kernel_info) != SUCCESS) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Op %s init fallback flags failed.",
                    op_kernel_info->op_type_.c_str());
    return FAILED;
  }
  op_kernel_info->init_flag_ = true;
  return SUCCESS;
}

Status OpKernelInfoConstructor::CheckFormatAgnosticOp(OpKernelInfoPtr op_kernel_info) const {
  if (op_kernel_info->GetOpPattern() != OP_PATTERN_FORMAT_AGNOSTIC) {
    return SUCCESS;
  }
  std::vector<InputOrOutputInfoPtr> input_infos = op_kernel_info->input_infos_;
  std::vector<InputOrOutputInfoPtr> output_infos = op_kernel_info->output_infos_;
  if (input_infos.size() != 1 || output_infos.size() != 1) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Node[%s]: input_infos size %zu or output_infos size %zu is not equal to 1.",
                    op_kernel_info->op_type_.c_str(), input_infos.size(), output_infos.size());
    return FAILED;
  }
  return SUCCESS;
}

void OpKernelInfoConstructor::ParserReferTensorNameVec(OpKernelInfoPtr op_kernel_info) const {
  for (const InputOrOutputInfoPtr &output_ptr : op_kernel_info->output_infos_) {
    if (output_ptr->op_param_type_ == DYNAMIC || output_ptr->op_param_type_ == RESERVED) {
      continue;
    }
    for (const InputOrOutputInfoPtr &input_ptr : op_kernel_info->input_infos_) {
      if (input_ptr->op_param_type_ == DYNAMIC || input_ptr->op_param_type_ == RESERVED) {
        continue;
      }
      if (output_ptr->GetName() == input_ptr->GetName()) {
        op_kernel_info->refer_tensor_name_vec_.push_back(output_ptr->GetName());
        break;
      }
    }
  }
  if (!op_kernel_info->refer_tensor_name_vec_.empty()) {
    FE_LOGD("Node[%s]: referenced tensor is [%s].",
            op_kernel_info->GetOpType().c_str(), StringVecToString(op_kernel_info->refer_tensor_name_vec_).c_str());
  }
}

Status OpKernelInfoConstructor::FinalizeOpKernelInfo(OpKernelInfoPtr op_kernel_info) const {
  if (op_kernel_info == nullptr) {
    FE_LOGW("OpKernelInfo is null pointer and it does not need to be finalized");
    return SUCCESS;
  }

  op_kernel_info->op_type_ = "";
  /* The following two loop can be ommited because when a vector
   * is cleared, the shared pointers in this vector will automatically
   * destroy themselves. */
  for (size_t i = 0; i < op_kernel_info->input_infos_.size(); i++) {
    if (op_kernel_info->input_infos_[i] == nullptr) {
      FE_LOGW("The finalize_op_kernel_info pointer at index %zu in input_infos_ should not be nullptr.", i);
      continue;
    }
    FinalizeInputAndOutput(op_kernel_info->input_infos_[i]);
  }

  for (size_t i = 0; i < op_kernel_info->output_infos_.size(); i++) {
    if (op_kernel_info->output_infos_[i] == nullptr) {
      FE_LOGW("FinalizeOpKernelInfo pointer at index %zu in output_infos_ should not be nullptr!", i);
      continue;
    }
    FinalizeInputAndOutput(op_kernel_info->output_infos_[i]);
  }

  op_kernel_info->input_infos_.clear();
  op_kernel_info->output_infos_.clear();
  op_kernel_info->attrs_info_.clear();
  op_kernel_info->init_flag_ = false;
  return SUCCESS;
}

void GetIntputAndOutputFlag(const OpContent &op_content, vector<string> &input_vec, vector<string> &output_vec,
                            bool &input_found, bool &output_found) {
  for (auto &content : op_content.map_kernel_info_) {
    if (CheckInputSubStr(content.first, STR_INPUT_LOWERCASE) ||
        CheckInputSubStr(content.first, STR_INPUT_FIRST_LETTER_UPPERCASE)) {
      input_vec.push_back(content.first);
      input_found = true;
    }
    if (CheckInputSubStr(content.first, STR_OUTPUT_LOWERCASE)||
        CheckInputSubStr(content.first, STR_OUTPUT_FIRST_LETTER_UPPERCASE)) {
      output_vec.push_back(content.first);
      output_found = true;
    }
  }
}

template <typename ContainerOfIter>
void AllNext(ContainerOfIter &iters) {
  for (size_t i = 0; i < iters.size(); i++) {
    iters[i] = next(iters[i]);
  }
}

template <typename ContainerOfContainerPtr, typename ContainerOfIter>
void AllErase(ContainerOfContainerPtr &containers, ContainerOfIter &iters) {
  for (size_t i = 0; i < containers.size(); i++) {
    iters[i] = containers[i]->erase(iters[i]);
  }
}

bool RemoveHeavyFormatSupportWhenC0Change(vector<vector<ge::DataType> *>& all_support_dtypes,
                                          vector<vector<ge::Format> *>& all_support_format) {
  vector<vector<ge::DataType>::iterator> all_support_dtypes_iters;
  vector<vector<ge::Format>::iterator> all_support_format_iters;

  for (auto &supported_dtypes: all_support_dtypes) {
      all_support_dtypes_iters.emplace_back(supported_dtypes->begin());
  }

  for (auto &supported_formats: all_support_format) {
      all_support_format_iters.emplace_back(supported_formats->begin());
  }

  bool do_remove = false;
  int case_num = all_support_dtypes[0]->size();
  for (int i = 0; i < case_num; i++) {
    bool has_heavy_format_output = false;
    for (auto &format_iter : all_support_format_iters) {
      if (std::find(FE_HEAVY_FORMAT_VECTOR.begin(), FE_HEAVY_FORMAT_VECTOR.end(), *format_iter) !=
          FE_HEAVY_FORMAT_VECTOR.end()) {
        has_heavy_format_output = true;
        break;
      }
    }

    bool C0_size_change = false;
    int64_t C0_size = transformer::TransferShapeUtils::GetC0ByDtype(*all_support_dtypes_iters[0]);
    for (auto &dtype_iter : all_support_dtypes_iters) {
      if (!has_heavy_format_output) {
        break;
      }

      if (C0_size != transformer::TransferShapeUtils::GetC0ByDtype(*dtype_iter)) {
        C0_size_change = true;
        break;
      }
    }

    if (C0_size_change) {
      do_remove = true;
      AllErase(all_support_dtypes, all_support_dtypes_iters);
      AllErase(all_support_format, all_support_format_iters);
    } else {
      AllNext(all_support_dtypes_iters);
      AllNext(all_support_format_iters);
    }
  }

  return do_remove;
}

bool OpKernelInfoConstructor::RemoveUnknownShapeAndShapeHeavyFormatSupportWhenC0Change(
    vector<InputOrOutputInfoPtr>& inputs, vector<InputOrOutputInfoPtr>& outputs) const {
  vector<vector<ge::DataType> *> all_support_dtypes;
  vector<vector<ge::Format> *> all_support_format;

  for (auto &list_of_input_or_output : {inputs, outputs}) {
    for (auto &input_or_output : list_of_input_or_output) {
      all_support_dtypes.emplace_back(&input_or_output->supported_dtypes_);
      all_support_format.emplace_back(&input_or_output->supported_formats_);
    }
  }

  bool is_remove = RemoveHeavyFormatSupportWhenC0Change(all_support_dtypes, all_support_format);

  vector<vector<ge::DataType> *> all_support_unknown_shape_dtypes;
  vector<vector<ge::Format> *> all_support_unknown_shape_format;

  for (auto &list_of_input_or_output : {inputs, outputs}) {
    for (auto &input_or_output : list_of_input_or_output) {
      all_support_unknown_shape_dtypes.emplace_back(&input_or_output->supported_unknown_shape_dtypes_);
      all_support_unknown_shape_format.emplace_back(&input_or_output->supported_unknown_shape_formats_);
    }
  }

  bool is_remove_unknown_shape = RemoveHeavyFormatSupportWhenC0Change(all_support_unknown_shape_dtypes,
                                                                      all_support_unknown_shape_format);
  return is_remove || is_remove_unknown_shape;
}

void OpKernelInfoConstructor::LogSupportFormatAndType(const char* message, OpKernelInfoPtr op_kernel_info) {
    for (auto &input: op_kernel_info->input_infos_) {
          FE_LOGD("[%s] format %s.", message, GetStrByFormatVec(input->supported_formats_).c_str());
          FE_LOGD("[%s] dtype %s.", message, GetStrByDataTypeVec(input->supported_dtypes_).c_str());
    }

    for (auto &output: op_kernel_info->output_infos_) {
          FE_LOGD("[%s] format %s.", message, GetStrByFormatVec(output->supported_formats_).c_str());
          FE_LOGD("[%s] dtype %s.", message, GetStrByDataTypeVec(output->supported_dtypes_).c_str());
    }
}

Status OpKernelInfoConstructor::ParseInputAndOutputFromOpContent(const OpContent &op_content,
                                                                 OpKernelInfoPtr op_kernel_info) {
  vector<string> input_vec;
  vector<string> output_vec;
  bool input_found = false;
  bool output_found = false;
  GetIntputAndOutputFlag(op_content, input_vec, output_vec, input_found, output_found);

  sort(input_vec.begin(), input_vec.end(), CmpInputsNum);
  sort(output_vec.begin(), output_vec.end(), CmpOutputsNum);

  if (!input_found || !output_found) {
    FE_LOGW("input[%u] or output[%u] is empty in op type [%s]!", input_found, output_found,
            op_content.op_type_.c_str());
  }
  uint32_t input_idx = 0;
  uint32_t dtype_and_format_size_of_first_input = INVALID_DTYPE_AND_FORMAT_SIZE;
  for (auto &input_index : input_vec) {
    auto input_pos = op_content.map_kernel_info_.find(input_index);
    if (input_pos == op_content.map_kernel_info_.end()) {
      return OP_STORE_MAP_KEY_FIND_FAILED;
    }
    string input_name;
    if (GetStrFromOpContent(op_content, input_index, STR_NAME, input_name) != SUCCESS) {
      REPORT_FE_ERROR(
          "[InitOpInfoLib][InitOpKernel] Failed to get InputName %s for op %s, the name of input %u for op %s does not exist.",
          input_index.c_str(), op_kernel_info->op_type_.c_str(), input_idx, op_content.op_type_.c_str());
      return FAILED;
    }
    InputOrOutputInfoPtr input_info_ptr = nullptr;
    FE_MAKE_SHARED(input_info_ptr = std::make_shared<InputOrOutputInfo>(input_name), return FAILED);
    FE_CHECK_NOTNULL(input_info_ptr);
    input_info_ptr->is_input_ = true;
    if (InitializeInputAndOutput(op_kernel_info->GetOpPattern(), op_kernel_info->op_type_, input_pos->second,
                                 input_idx++, input_info_ptr, dtype_and_format_size_of_first_input,
                                 op_kernel_info) != SUCCESS) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Initialization of Op %s with input_desc_info %s failed.",
                      op_kernel_info->op_type_.c_str(), input_index.c_str());
      return FAILED;
    }

    op_kernel_info->input_infos_.push_back(input_info_ptr);
  }
  uint32_t output_idx = 0;
  for (auto &output_index : output_vec) {
    auto output_pos = op_content.map_kernel_info_.find(output_index);
    if (output_pos == op_content.map_kernel_info_.end()) {
      return OP_STORE_MAP_KEY_FIND_FAILED;
    }
    string output_name;
    if (GetStrFromOpContent(op_content, output_index, STR_NAME, output_name) != SUCCESS) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Op %s get OutputName %s failed.",
                      op_kernel_info->op_type_.c_str(), output_index.c_str());
      return FAILED;
    }
    InputOrOutputInfoPtr output_info_ptr = nullptr;
    FE_MAKE_SHARED(output_info_ptr = std::make_shared<InputOrOutputInfo>(output_name), return FAILED);
    FE_CHECK_NOTNULL(output_info_ptr);

    output_info_ptr->is_input_ = false;
    if (InitializeInputAndOutput(op_kernel_info->GetOpPattern(), op_kernel_info->op_type_, output_pos->second,
                                 output_idx++, output_info_ptr, dtype_and_format_size_of_first_input,
                                 op_kernel_info) != SUCCESS) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Failed to initialize output_desc_info %s for Op %s.",
                      output_index.c_str(), op_kernel_info->op_type_.c_str());
      return FAILED;
    }
    op_kernel_info->output_infos_.push_back(output_info_ptr);
  }
  if (op_kernel_info->GetOpPattern() == OP_PATTERN_FORMAT_AGNOSTIC) {
      bool is_remove = RemoveUnknownShapeAndShapeHeavyFormatSupportWhenC0Change(op_kernel_info->input_infos_,
          op_kernel_info->output_infos_);
      if (is_remove) {
          LogSupportFormatAndType("After remove heavy format change C0 size.", op_kernel_info);
      }
  }
  return SUCCESS;
}

Status OpKernelInfoConstructor::InitFormatAndDtypeForSingleInputAndOutput(
    OpPattern op_pattren, const map<string, string> &map_info,
    const InputOrOutputInfoPtr &input_or_output_info,
    OpKernelInfoPtr op_kernel_info, uint32_t &dtype_and_format_size_of_first_input) {
  string format_str;
  string unknown_format_str;
  Status format_stat = GetStrFromMap(map_info, STR_FORMAT, format_str);
  Status unknown_format_stat = GetStrFromMap(map_info, STR_UNKNOWN_SHAPE_FORMAT, unknown_format_str);

  string unique_name = input_or_output_info->GetUniqueName();
  string op_type = op_kernel_info->GetOpType();
  bool init_by_op_kernel =
      op_pattren == OP_PATTERN_OP_KERNEL || format_stat == SUCCESS || unknown_format_stat == SUCCESS;
  if (init_by_op_kernel) {
    if (format_stat == SUCCESS &&
        InitDtypeAndFormat(map_info, input_or_output_info, dtype_and_format_size_of_first_input) != SUCCESS) {
      REPORT_FE_ERROR("%s Node[%s]: failed to initialize data type and format. Input or output is %s.", kStageInitOpKernel.c_str(),
                      op_type.c_str(), unique_name.c_str());
      return FAILED;
    }
    if (unknown_format_stat == SUCCESS && InitDtypeByPattern(map_info, input_or_output_info,
        dtype_and_format_size_of_first_input, op_pattren) != SUCCESS) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Node[%s]: failed to init dtypes. Input or output is %s.",
                      op_type.c_str(), unique_name.c_str());
      return FAILED;
    }
    if (InitUnknownFormatAndDtype(map_info, input_or_output_info, dtype_and_format_size_of_first_input) != SUCCESS) {
      REPORT_FE_ERROR("%s Node[%s]: failed to initialize unknown shape dtype and format. The input or output is %s.",
                      kStageInitOpKernel.c_str(), op_type.c_str(), unique_name.c_str());
      return FAILED;
    }
  } else if (op_pattren == OP_PATTERN_OP_CUSTOMIZE) {
    FE_LOGD("Node[%s]: the dynamic_format.flag is true. No need to init dtype or format. Input or output is %s.",
            op_type.c_str(), unique_name.c_str());
  } else if (op_pattren == OP_PATTERN_FORMAT_AGNOSTIC) {
    FE_LOGD("Node[%s]: The operation pattern is format-agnostic. It needs to initialize the data type and all formats. The input or output is %s.",
            op_type.c_str(), unique_name.c_str());
    if (InitDtypeAndAllFormat(map_info, input_or_output_info, dtype_and_format_size_of_first_input, op_kernel_info) !=
        SUCCESS) {
      REPORT_FE_ERROR("%s Node[%s]: failed to init dtype and all formats. Input or output is %s.",
                      kStageInitOpKernel.c_str(), op_type.c_str(), unique_name.c_str());
      return FAILED;
    }
  } else {
    FE_LOGD("Node[%s]: operation pattern is %s. Initialization of dtypes is required. Input or output type is %s.",
            op_type.c_str(), GetOpPatternString(op_pattren).c_str(), unique_name.c_str());
    if (InitDtype(map_info, input_or_output_info, dtype_and_format_size_of_first_input) != SUCCESS) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Node[%s]: failed to init dtypes. Input or output is %s.",
                      op_type.c_str(), unique_name.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status OpKernelInfoConstructor::InitializeInputAndOutput(OpPattern op_pattren, const string &op_type,
                                                         const map<string, string> &map_info, uint32_t index,
                                                         InputOrOutputInfoPtr input_or_output_info,
                                                         uint32_t &dtype_and_format_size_of_first_input,
                                                         OpKernelInfoPtr op_kernel_info) {
  FE_CHECK(input_or_output_info == nullptr, FE_LOGW("Parameter input_or_output_info is null."),
           return OP_KERNEL_INFO_NULL_PTR);

  // 1. set unique_name
  input_or_output_info->index_ = index;
  SetUniqueName(input_or_output_info);

  // 2. init formats and dtypes
  if (InitFormatAndDtypeForSingleInputAndOutput(op_pattren, map_info, input_or_output_info,
                                                op_kernel_info, dtype_and_format_size_of_first_input) != SUCCESS) {
    return FAILED;
  }
  // 3. init param_type
  string param_type_str;
  if (GetStrFromMap(map_info, STR_PARAM_TYPE, param_type_str) != SUCCESS) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] TensorDescInfo::Initialize param_type_str key invalid.");
    return FAILED;
  }
  auto param_type_iter = kParamTypeMap.find(StringUtils::Trim(param_type_str));
  if (param_type_iter == kParamTypeMap.end()) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] invalid param_type.");
    return OP_STORE_MAP_KEY_FIND_FAILED;
  }
  input_or_output_info->op_param_type_ = param_type_iter->second;

  // 4. init tuneformat_switch
  if (InitTuneFormatSwitch(map_info, input_or_output_info, op_type) != SUCCESS) {
    FE_LOGE("op(type[%s]) fail to init tuneformat_switch.", op_type.c_str());
    return FAILED;
  }

  // 5. init reshape_type
  string reshape_type_str;
  if (GetStrFromMap(map_info, STR_RESHAPE_TYPE, reshape_type_str) == SUCCESS) {
    FE_LOGD("Node[%s], Tensor[%u, %s]: reshape type is %s. The is_input_ flag value is %u.",
        op_type.c_str(), index, input_or_output_info->name_.c_str(), reshape_type_str.c_str(),
        input_or_output_info->is_input_);
    input_or_output_info->reshape_type_ = reshape_type_str;
  }

  // 6. init const_value_depend
  string const_value_depend_str;
  if (GetStrFromMap(map_info, STR_CONST_VALUE_DEPEND, const_value_depend_str) == SUCCESS) {
    FE_LOGD("Node[%s], Tensor[%u, %s]: It has a constant value dependency on %s.",
            op_type.c_str(), index, input_or_output_info->name_.c_str(), const_value_depend_str.c_str());
    // const depend param is optional
    auto const_depend_iter = kConstValueDependMap.find(StringUtils::Trim(const_value_depend_str));
    if (const_depend_iter != kConstValueDependMap.end()) {
      input_or_output_info->op_const_value_depend_ = const_depend_iter->second;
    }
  }

  return SUCCESS;
}

Status OpKernelInfoConstructor::FinalizeInputAndOutput(InputOrOutputInfoPtr input_or_output_info) const {
  /* null pointer means it is already finalized, so return SUCCESS here. */
  if (input_or_output_info == nullptr) {
    FE_LOGW("InputOrOutputInfo is a null pointer in the FinalizeInputAndOutput function.");
    return SUCCESS;
  }
  input_or_output_info->supported_dtypes_.clear();
  input_or_output_info->supported_formats_.clear();
  return SUCCESS;
}

template <typename T>
Status ConvertStr2VecNumber(const string &attr_str, vector<T> &temp_plate_vec) {
  vector<string> template_str_vec = StringUtils::Split(attr_str, ',');
  for (auto el : template_str_vec) {
    // std::remove_cv: get rid of prefix like const or volatile
    T attr_elem;
    using DT = typename std::remove_cv<T>::type;
    if (std::is_same<int64_t, DT>::value) {
      try {
        attr_elem = static_cast<int64_t>(stol(el));
      } catch (...) {
        REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Invalid int attribute: %s.", attr_str.c_str());
        return OP_STORE_STRING_CONVERT_FAILED;
      }
    } else if (std::is_same<bool, DT>::value) {
      attr_elem = StringUtils::MatchString(kStrTrue, el);
    } else {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Attribute %s is invalid. The attribute type is not one of int or bool.",
                      attr_str.c_str());
      return OP_STORE_STRING_CONVERT_FAILED;
    }
    temp_plate_vec.push_back(attr_elem);
  }
  return SUCCESS;
}

Status ConvertStr2VecNumber(const string &attr_str, vector<float> &temp_plate_vec) {
  vector<string> template_str_vec = StringUtils::Split(attr_str, ',');
  for (auto &el : template_str_vec) {
    float attr_elem;
    try {
      attr_elem = std::stof(el);
    } catch (...) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Float attribute %s is invalid.", attr_str.c_str());
      return OP_STORE_STRING_CONVERT_FAILED;
    }
    temp_plate_vec.emplace_back(attr_elem);
  }
  return SUCCESS;
}

/* string type */
Status ConvertStr2VecNumber(const string &attr_str, vector<string> &temp_plate_vec) {
  vector<string> template_str_vec = StringUtils::Split(attr_str, ',');
  for (auto const &el : template_str_vec) {
    temp_plate_vec.push_back(el);
  }
  return SUCCESS;
}

template <typename T>
Status ConvertVecStr2VecNumber(string &attr_str, vector<T> &temp_plate_vec) {
  // [1,2,3]
  attr_str.erase(std::remove(attr_str.begin(), attr_str.end(), '['), attr_str.end());
  attr_str.erase(std::remove(attr_str.begin(), attr_str.end(), ']'), attr_str.end());
  if (ConvertStr2VecNumber(attr_str, temp_plate_vec) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

template <typename T>
Status ConvertVecListStr2VecNumber(string &attr_str, vector<vector<T>> &temp_plate_vec) {
  // [[1,2,3],[2,3,4]]
  vector<string> attr_list_vec = StringUtils::Split(attr_str, "],");
  for (auto &attr_item : attr_list_vec) {
    vector<T> temp_attr_vec;
    if (ConvertVecStr2VecNumber(attr_item, temp_attr_vec) != SUCCESS) {
      return FAILED;
    }
    temp_plate_vec.emplace_back(temp_attr_vec);
  }
  return SUCCESS;
}

Status OpKernelInfoConstructor::InitDtype(const map<string, string> &map_info,
                                          InputOrOutputInfoPtr input_or_output_info,
                                          uint32_t &dtype_and_format_size_of_first_input) {
  string dtype_str;
  if (GetStrFromMap(map_info, STR_DTYPE, dtype_str) != SUCCESS) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] InputOrOutputInfo::Initialize failed due to invalid data_type key.");
    return FAILED;
  }
  vector<string> dtype_str_vec = StringUtils::Split(StringUtils::Trim(dtype_str), ',');
  uint32_t dtype_size = static_cast<uint32_t>(dtype_str_vec.size());

  // check th dtype size and format size of inputs or outputs should be the same as
  // the size of the first input or output
  if (dtype_and_format_size_of_first_input == INVALID_DTYPE_AND_FORMAT_SIZE) {
    dtype_and_format_size_of_first_input = dtype_size;
  }
  if (dtype_size != dtype_and_format_size_of_first_input) {
    REPORT_FE_ERROR(
        "[InitOpInfoLib][InitOpKernel] "
        "The dtype size of input [%u] named [%s] is [%u], which is not as same as other input or output [%u].",
        input_or_output_info->GetIndex(), input_or_output_info->GetName().c_str(), dtype_size,
        dtype_and_format_size_of_first_input);
    return FAILED;
  }

  for (uint32_t i = 0; i < dtype_size; i++) {
    ge::DataType dtype;
    if (String2DataType(StringUtils::Trim(dtype_str_vec[i]), dtype) != SUCCESS) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] String2DataType conversion for %s failed due to illegal dtype!",
                      dtype_str_vec[i].c_str());
      return FAILED;
    }
    input_or_output_info->supported_dtypes_.push_back(dtype);
  }
  return SUCCESS;
}

Status OpKernelInfoConstructor::InitDtypeAndAllFormat(const map<string, string> &map_info,
                                                      InputOrOutputInfoPtr input_or_output_info,
                                                      uint32_t &dtype_of_first_input, OpKernelInfoPtr op_kernel_info) {
  string dtype_str;
  if (GetStrFromMap(map_info, STR_DTYPE, dtype_str) != SUCCESS) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] InputOrOutputInfo::Initialize failed due to invalid data_type key.");
    return FAILED;
  }
  vector<string> dtype_str_vec = StringUtils::Split(StringUtils::Trim(dtype_str), ',');
  size_t dtype_size = dtype_str_vec.size();

  // 1. check th dtype size of inputs or outputs should be the same as
  // the size of the first input or output
  if (dtype_of_first_input == INVALID_DTYPE_AND_FORMAT_SIZE) {
    dtype_of_first_input = static_cast<uint32_t >(dtype_size);
  }
  if (dtype_size != dtype_of_first_input) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Tensor[%u, %s]: the dtype size %zu does not match %u.",
                    input_or_output_info->GetIndex(), input_or_output_info->GetName().c_str(), dtype_size,
                    dtype_of_first_input);
    return FAILED;
  }

  // 2. generate the new data_types and the new formats
  vector<ge::DataType> old_data_types;
  for (size_t i = 0; i < dtype_size; i++) {
    ge::DataType dtype;
    if (String2DataType(StringUtils::Trim(dtype_str_vec[i]), dtype) != SUCCESS) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] String2DataType conversion for %s failed due to illegal dtype!",
                      dtype_str_vec[i].c_str());
      return FAILED;
    }
    old_data_types.push_back(dtype);
  }
  vector<ge::Format> old_formats(FE_HEAVY_FORMAT_VECTOR);
  old_formats.insert(old_formats.cend(), FE_ORIGIN_FORMAT_VECTOR.cbegin(), FE_ORIGIN_FORMAT_VECTOR.cend());
  vector<ge::Format> new_formats;
  vector<ge::DataType> new_data_types;
  if (GenerateUnionFormatAndDtype(old_formats, old_data_types, new_formats, new_data_types) != SUCCESS) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Failed to GenerateUnionFormatDtype");
    return FAILED;
  }
  input_or_output_info->supported_formats_.insert(input_or_output_info->supported_formats_.cend(),
                                                  new_formats.cbegin(), new_formats.cend());
  input_or_output_info->supported_dtypes_.insert(input_or_output_info->supported_dtypes_.cend(),
                                                 new_data_types.cbegin(), new_data_types.cend());
  input_or_output_info->supported_sub_formats_.emplace_back(SUPPORT_ALL_SUB_FORMAT);
  // unknown shape format and dtype
  if (op_kernel_info->IsSupportDynamicShape()) {
    FE_LOGD("Node [%s]: It supports an unknown shape. Set its format and data type.", op_kernel_info->GetOpType().c_str());
    input_or_output_info->supported_unknown_shape_formats_.insert(
        input_or_output_info->supported_unknown_shape_formats_.cend(), new_formats.cbegin(), new_formats.cend());
    input_or_output_info->supported_unknown_shape_dtypes_.insert(
        input_or_output_info->supported_unknown_shape_dtypes_.cend(), new_data_types.cbegin(), new_data_types.cend());
  }
  FE_LOGD("The new_formats is %s, and the new_data_types is %s.", GetStrByFormatVec(new_formats).c_str(),
          GetStrByDataTypeVec(new_data_types).c_str());
  return SUCCESS;
}

Status OpKernelInfoConstructor::InitFallback(const OpContent &op_content, const OpKernelInfoPtr &op_kernel_info) const {
  if (ParseFallbackFlags(op_content, STR_UNKNOWN_SHAPE_ENABLE,
      op_kernel_info->fallback_flags_.unknown_shape_fallbacks) != SUCCESS) {
    return FAILED;
  }
  if (ParseFallbackFlags(op_content, STR_ENABLE, op_kernel_info->fallback_flags_.fallbacks) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status OpKernelInfoConstructor::ParseFallbackFlags(const OpContent &op_content, const string &fallback_key,
                                                   std::vector<bool> &cfg_fallbacks) const {
  string fallback_str;
  if (GetStrFromOpContent(op_content, STR_FALLBACK, fallback_key, fallback_str) != SUCCESS) {
    return SUCCESS;
  }
  vector<string> fallback_str_vec;
  StringUtils::SplitWithTrim(fallback_str, ',', fallback_str_vec);
  size_t fallback_size = fallback_str_vec.size();
  vector<bool> fallback_flags;
  for (size_t i = 0; i < fallback_size; ++i) {
    bool fallback_flag;
    if (String2Bool(fallback_str_vec[i], fallback_flag) != SUCCESS) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] String2Bool conversion of %s failed. The fallback_flag value is invalid!",
                      fallback_str_vec[i].c_str());
      return FAILED;
    }
    fallback_flags.emplace_back(fallback_flag);
  }
  cfg_fallbacks = std::move(fallback_flags);
  return SUCCESS;
}

Status OpKernelInfoConstructor::InitDtypeAndFormat(const map<string, string> &map_info,
                                                   InputOrOutputInfoPtr input_or_output_info,
                                                   uint32_t &dtype_and_format_size_of_first_input) {
  string format_str;
  (void)GetStrFromMap(map_info, STR_FORMAT, format_str);
  vector<string> format_str_vec = StringUtils::Split(format_str, ',');

  string dtype_str;
  if (GetStrFromMap(map_info, STR_DTYPE, dtype_str) != SUCCESS) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] InputOrOutputInfo::Initialize failed due to invalid data_type key.");
    return FAILED;
  }
  vector<string> dtype_str_vec = StringUtils::Split(StringUtils::Trim(dtype_str), ',');

  string sub_format_str = "";
  (void)GetStrFromMap(map_info, STR_SUB_FORMAT, sub_format_str);
  vector<string> sub_format_str_vec = StringUtils::Split(StringUtils::Trim(sub_format_str), ',');

  size_t format_size = format_str_vec.size();
  size_t dtype_size = dtype_str_vec.size();
  if (dtype_size != format_size) {
    REPORT_FE_ERROR(
        "[GraphOpt][InitDtypeAndFormat][InitDtypeAndFormat] Inconsistent format size [%zu] and dtype size [%zu]!",
        format_size, dtype_size);
    return FAILED;
  }
  /* The Following dtype size and format size of inputs or outputs should be the
   * same as
   * the size of the first input or output */
  if (dtype_and_format_size_of_first_input == INVALID_DTYPE_AND_FORMAT_SIZE) {
    dtype_and_format_size_of_first_input = static_cast<uint32_t>(format_size);
  }
  if (format_size != dtype_and_format_size_of_first_input) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Tensor[%u, %s]: the format and dtype size %zu does not equal %u.",
                    input_or_output_info->GetIndex(), input_or_output_info->GetName().c_str(), format_size,
                    dtype_and_format_size_of_first_input);
    return FAILED;
  }

  for (size_t i = 0; i < format_size; i++) {
    ge::Format format = ge::TypeUtils::SerialStringToFormat(StringUtils::Trim(format_str_vec[i]));
    if (format == ge::FORMAT_RESERVED) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Failed to convert format %zu from string %s to enum.",
                      i, format_str_vec[i].c_str());
      return FAILED;
    }
    input_or_output_info->supported_formats_.push_back(format);

    ge::DataType dtype;
    if (String2DataType(StringUtils::Trim(dtype_str_vec[i]), dtype) != SUCCESS) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] String2DataType conversion for %s failed due to illegal dtype!",
                      dtype_str_vec[i].c_str());
      return FAILED;
    }
    input_or_output_info->supported_dtypes_.push_back(dtype);
  }

  for (auto &one_sub_format_str : sub_format_str_vec) {
    int64_t subformat = DEFAULT_SUB_FORMAT;
    try {
      subformat = std::stoll(one_sub_format_str);
      if (subformat > UINT16_MAX || subformat < SUPPORT_ALL_SUB_FORMAT) {
        REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Subformat %s value is illegal!", one_sub_format_str.c_str());
        return FAILED;
      }
    } catch (const std::exception &e) {
      FE_LOGW("[InitOpInfoLib][InitOpKernel] Sub_format:%s transform failed, use default value.",
              one_sub_format_str.c_str());
      return FAILED;
    }
    input_or_output_info->supported_sub_formats_.emplace_back(static_cast<uint32_t>(subformat));
  }
  return SUCCESS;
}

Status OpKernelInfoConstructor::InitUnknownFormatAndDtype(const map<string, string> &map_info,
                                                          InputOrOutputInfoPtr input_or_output_info,
                                                          uint32_t &dtype_and_format_size_of_first_input) {
  string format_str;
  Status status = GetStrFromMap(map_info, STR_UNKNOWN_SHAPE_FORMAT, format_str);
  if (status != SUCCESS) {
    FE_LOGD("Can not find unknownshape_format. Set its format and dtype.");
    if (GetStrFromMap(map_info, STR_FORMAT, format_str) != SUCCESS) {
      FE_LOGW("[InitOpInfoLib][InitOpKernel] InputOrOutputInfo::Initialize format key invalid.");
      return SUCCESS;
    }
  }
  vector<string> format_str_vec = StringUtils::Split(format_str, ',');

  string dtype_str;
  if (GetStrFromMap(map_info, STR_DTYPE, dtype_str) != SUCCESS) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] InputOrOutputInfo::Initialize failed due to invalid data_type key.");
    return FAILED;
  }
  vector<string> dtype_str_vec = StringUtils::Split(StringUtils::Trim(dtype_str), ',');

  size_t format_size = format_str_vec.size();
  size_t dtype_size = dtype_str_vec.size();
  if (dtype_size != format_size) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Inconsistent format size [%zu] and data type size [%zu]!",
        format_size, dtype_size);
    return FAILED;
  }
  /* The Following dtype size and format size of inputs or outputs should be the
   * same as
   * the size of the first input or output */
  if (dtype_and_format_size_of_first_input == INVALID_DTYPE_AND_FORMAT_SIZE) {
    dtype_and_format_size_of_first_input = static_cast<uint32_t>(format_size);
  }
  if (format_size != dtype_and_format_size_of_first_input) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Tensor[%u, %s]: the format and dtype size %zu does not equal %u.",
                    input_or_output_info->GetIndex(), input_or_output_info->GetName().c_str(), format_size,
                    dtype_and_format_size_of_first_input);
    return FAILED;
  }
  for (size_t i = 0; i < format_size; i++) {
    ge::Format format = ge::TypeUtils::SerialStringToFormat(StringUtils::Trim(format_str_vec[i]));
    if (format == ge::FORMAT_NULL) {
      FE_LOGD("Format %zu is null.", i);
      continue;
    }
    if (format == ge::FORMAT_RESERVED) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Failed to convert format %zu from string %s to enum.",
                      i, format_str_vec[i].c_str());
      return FAILED;
    }
    input_or_output_info->supported_unknown_shape_formats_.push_back(format);
    ge::DataType dtype;
    if (String2DataType(StringUtils::Trim(dtype_str_vec[i]), dtype) != SUCCESS) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] String2DataType conversion for %s failed due to illegal dtype!",
                      dtype_str_vec[i].c_str());
      return FAILED;
    }
    input_or_output_info->supported_unknown_shape_dtypes_.push_back(dtype);
  }
  return SUCCESS;
}

Status OpKernelInfoConstructor::GetStrFromMap(const map<string, string> &map_info, string key, string &value) const {
  std::map<string, string>::const_iterator find = map_info.find(key);
  if (find == map_info.end()) {
    return OP_STORE_MAP_KEY_FIND_FAILED;
  }
  value = find->second;
  return SUCCESS;
}

template <typename T>
Status OpKernelInfoConstructor::ConvertListListAttrValue(const OpContent &op_content, const std::string &attr_name,
                                                         const std::string &key_name,
                                                         vector<vector<vector<T>>> &list_list_attr_vec) const{
  string value;
  Status status = GetStrFromOpContent(op_content, attr_name, key_name, value);
  if (status != SUCCESS) {
    if (key_name == STR_VALUE) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Op [%s] Get ListList attr[%s:%s] value failed.",
                      op_content.op_type_.c_str(), attr_name.c_str(), key_name.c_str());
      return FAILED;
    } else {
      FE_LOGD("Getting attribute %s for [%s:%s] was not successfully.", op_content.op_type_.c_str(), attr_name.c_str(), key_name.c_str());
      return NOT_CHANGED;
    }
  }
  vector<string> value_str_list_list = StringUtils::Split(value, ';');
  for (auto &value_str_list : value_str_list_list) {
    list_list_attr_vec.emplace_back(vector<vector<T>>());
    vector<vector<T>> &value_list = list_list_attr_vec.back();
    if (ConvertVecListStr2VecNumber(value_str_list, value_list) != SUCCESS) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Failed to convert attribute [%s:%s] value for operator [%s].",
                      attr_name.c_str(), key_name.c_str(), op_content.op_type_.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

template <typename T>
Status OpKernelInfoConstructor::FillUpListListAttrValue(const OpContent &op_content, const std::string &attr_name,
                                                        vector<vector<vector<T>>> &list_list_attr_vec,
                                                        const AttrInfoPtr &attr_info) const {
  Status status = SUCCESS;
  if (!attr_info->is_required_) {
    status = ConvertListListAttrValue(op_content, attr_name, STR_DEFAULT_VALUE, list_list_attr_vec);
    if (status == NOT_CHANGED) {
      attr_info->is_default_value_defined_ = false;
    } else if (status == FAILED) {
      return FAILED;
    } else {
      if (!list_list_attr_vec.empty()) {
        attr_info->default_value_ = GeAttrValue::CreateFrom<vector<vector<T>>>(list_list_attr_vec[0]);
        attr_info->is_default_value_defined_ = true;
      }
    }
  }
  if (attr_info->is_support_all_value_) {
    return SUCCESS;
  }
  status = ConvertListListAttrValue(op_content, attr_name, STR_VALUE, list_list_attr_vec);
  if (status != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

template <typename T>
Status OpKernelInfoConstructor::ConvertListAttrValue(const OpContent &op_content, const std::string &attr_name,
                                                     const std::string &key_name,
                                                     std::vector<std::vector<T>> &list_attr_vec) const {
  string value;
  Status status = GetStrFromOpContent(op_content, attr_name, key_name, value);
  if (status != SUCCESS) {
    if (key_name == STR_VALUE) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Op [%s] Get ListList attr[%s:%s] value failed.",
                      op_content.op_type_.c_str(), attr_name.c_str(), key_name.c_str());
      return FAILED;
    } else {
      FE_LOGD("Getting attribute %s for [%s:%s] was not successfully.", op_content.op_type_.c_str(), attr_name.c_str(), key_name.c_str());
      return NOT_CHANGED;
    }
  }
  vector<string> value_str_list = StringUtils::Split(value, ';');
  for (auto &value_str : value_str_list) {
    list_attr_vec.emplace_back(vector<T>());
    vector<T> &value_list = list_attr_vec.back();
    if (ConvertVecStr2VecNumber(value_str, value_list) != SUCCESS) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Failed to convert attribute [%s:%s] value for operator [%s].",
                      op_content.op_type_.c_str(), attr_name.c_str(), key_name.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

template <typename T>
Status OpKernelInfoConstructor::FillUpListAttrValue(const OpContent &op_content, const std::string &attr_name,
                                                    std::vector<std::vector<T>> &list_attr_vec,
                                                    const AttrInfoPtr &attr_info) const {
  Status status = SUCCESS;
  if (!attr_info->is_required_) {
    status = ConvertListAttrValue(op_content, attr_name, STR_DEFAULT_VALUE, list_attr_vec);
    if (status == NOT_CHANGED) {
      attr_info->is_default_value_defined_ = false;
    } else if (status == FAILED) {
      return FAILED;
    } else {
      if (!list_attr_vec.empty()) {
        attr_info->default_value_ = GeAttrValue::CreateFrom<vector<T>>(list_attr_vec[0]);
        attr_info->is_default_value_defined_ = true;
      }
    }
  }
  if (attr_info->is_support_all_value_) {
    return SUCCESS;
  }
  status = ConvertListAttrValue(op_content, attr_name, STR_VALUE, list_attr_vec);
  if (status != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

template <typename T>
Status OpKernelInfoConstructor::InitAttrTemplate(const OpContent &op_content, const std::string &attr,
                                                 const AttrInfoPtr &attr_info) const {
  string attr_and_prefix = STR_ATTR_PREFIX + attr;
  vector<T> template_vec;
  string template_str;
  string default_int_str;

  /* Get Default Value for optional Attr */
  if (!attr_info->is_required_) {
    if (GetStrFromOpContent(op_content, attr_and_prefix, STR_DEFAULT_VALUE, default_int_str) == SUCCESS) {
      if (ConvertStr2VecNumber(default_int_str, template_vec) == SUCCESS) {
        /* we only takes the first default value. Here in function
         * GetStrFromOpContent we eliminate cases that default value string is
         * empty
         */
        if (!template_vec.empty()) {
          attr_info->default_value_ = GeAttrValue::CreateFrom<T>(template_vec[0]);
          attr_info->is_default_value_defined_ = true;
        }
      } else {
        FE_LOGW("Op %s failed to convert DefaultValue string to int %s.", op_content.op_type_.c_str(),
                default_int_str.c_str());
      }
    } else {
      attr_info->is_default_value_defined_ = false;
      FE_LOGW("Failed to get default_value of attribute %s in op %s.", attr_and_prefix.c_str(),
              op_content.op_type_.c_str());
    }
  }

  if (attr_info->is_support_all_value_) {
    return SUCCESS;
  }

  if (GetStrFromOpContent(op_content, attr_and_prefix, STR_VALUE, template_str) != SUCCESS) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Failed to get integer attribute %s for operation %s.",
                    attr_and_prefix.c_str(), op_content.op_type_.c_str());
    return FAILED;
  }
  if (ConvertStr2VecNumber(template_str, template_vec) != SUCCESS) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Failed to convert Op %s to int %s.",
                    op_content.op_type_.c_str(), template_str.c_str());
    return FAILED;
  }

  for (const auto &it : template_vec) {
    ge::GeAttrValue attr_value = GeAttrValue::CreateFrom<T>(it);
    attr_info->supported_values_.push_back(attr_value);
  }
  return SUCCESS;
}

template <typename T>
Status OpKernelInfoConstructor::InitAttrListTemplate(const OpContent &op_content, const std::string &attr,
                                                     const AttrInfoPtr &attr_info) const {
  string attr_and_prefix = STR_ATTR_PREFIX + attr;
  vector<vector<T>> list_attr_vec;
  vector<GeAttrValue> attr_value_vec;
  if (FillUpListAttrValue<T>(op_content, attr_and_prefix, list_attr_vec, attr_info) != SUCCESS) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Failed to convert Op %s to ListInt %s.",
                    op_content.op_type_.c_str(), attr_and_prefix.c_str());
    return FAILED;
  }
  for (auto &el : list_attr_vec) {
    GeAttrValue attr_value = GeAttrValue::CreateFrom<vector<T>>(el);
    attr_info->supported_values_.push_back(attr_value);
  }
  return SUCCESS;
}

template <typename T>
Status OpKernelInfoConstructor::InitAttrListListTemplate(const OpContent &op_content, const std::string &attr,
                                                         const AttrInfoPtr &attr_info) const {
  string attr_and_prefix = STR_ATTR_PREFIX + attr;
  vector<vector<vector<T>>> list_attr_vec;
  vector<GeAttrValue> attr_value_vec;
  if (FillUpListListAttrValue<T>(op_content, attr_and_prefix, list_attr_vec, attr_info) != SUCCESS) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Conversion of Op %s to ListListInt %s failed.",
                    op_content.op_type_.c_str(), attr_and_prefix.c_str());
    return FAILED;
  }
  for (auto &el : list_attr_vec) {
    GeAttrValue attr_value = GeAttrValue::CreateFrom<vector<vector<T>>>(el);
    attr_info->supported_values_.push_back(attr_value);
  }
  return SUCCESS;
}

Status OpKernelInfoConstructor::InitAttrValueSub(const OpContent &op_content, OpKernelInfoPtr op_kernel_info) {
  for (auto &attr : op_kernel_info->attrs_info_) {
    Status status = SUCCESS;
    switch (attr->dtype_) {
      case GeAttrValue::VT_INT: {
        status = InitAttrTemplate<int64_t>(op_content, attr->attr_name_, attr);
        break;
      }
      case GeAttrValue::VT_FLOAT: {
        status = InitAttrTemplate<float>(op_content, attr->attr_name_, attr);
        break;
      }
      case GeAttrValue::VT_BOOL: {
        status = InitAttrTemplate<bool>(op_content, attr->attr_name_, attr);
        break;
      }
      case GeAttrValue::VT_STRING: {
        status = InitAttrTemplate<string>(op_content, attr->attr_name_, attr);
        break;
      }
      case GeAttrValue::VT_LIST_INT: {
        status = InitAttrListTemplate<int64_t>(op_content, attr->attr_name_, attr);
        break;
      }
      case GeAttrValue::VT_LIST_FLOAT: {
        status = InitAttrListTemplate<float>(op_content, attr->attr_name_, attr);
        break;
      }
      case GeAttrValue::VT_LIST_BOOL: {
        status = InitAttrListTemplate<bool>(op_content, attr->attr_name_, attr);
        break;
      }
      case GeAttrValue::VT_LIST_LIST_INT: {
        status = InitAttrListListTemplate<int64_t>(op_content, attr->attr_name_, attr);
        break;
      }
      default: {
        status = InitAttrListTemplate<string>(op_content, attr->attr_name_, attr);
        break;
      }
    }
    if (status != SUCCESS) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Op [%s]: Failed to initialize attribute %s.",
                      op_content.op_type_.c_str(), attr->attr_name_.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status OpKernelInfoConstructor::InitAttrValue(const OpContent &op_content, OpKernelInfoPtr op_kernel_info) {
  vector<string> attr_vec;
  auto attr_pos = op_content.map_kernel_info_.find(STR_ATTR);
  if (op_content.map_kernel_info_.end() == attr_pos) {
    return SUCCESS;
  }
  string attr_str;
  if (GetStrFromOpContent(op_content, STR_ATTR, STR_LIST, attr_str) != SUCCESS) {
    FE_LOGW("Operation %s get attr.list was not successfully.", op_content.op_type_.c_str());
    return SUCCESS;
  }
  attr_vec = StringUtils::Split(attr_str, ',');
  for (auto &attr : attr_vec) {
    string attr_type_str;
    string attr_value_str;
    string param_type_str;

    string attr_name = STR_ATTR_PREFIX + StringUtils::Trim(attr);
    if (GetStrFromOpContent(op_content, attr_name, STR_TYPE, attr_type_str) != SUCCESS) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel][InitAttrValue] Node[%s]: unable to obtain type of attribute %s.",
                      op_content.op_type_.c_str(), attr_name.c_str());
      return FAILED;
    }
    if (GetStrFromOpContent(op_content, attr_name, STR_VALUE, attr_value_str) != SUCCESS) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel][InitAttrValue] Node[%s]: failed to get value of attr %s.",
                      op_content.op_type_.c_str(), attr_name.c_str());
      return FAILED;
    }
    if (GetStrFromOpContent(op_content, attr_name, STR_PARAM_TYPE, param_type_str) != SUCCESS) {
      FE_LOGD("OpKernelInfoConstructor::InitAttrValue, Op %s get %s.param_type_str unsuccess, check the ops_kernel_info.",
          op_content.op_type_.c_str(), attr_name.c_str());
      continue;
    }
    std::shared_ptr<AttrInfo> new_attr_info = nullptr;
    FE_MAKE_SHARED(new_attr_info = std::make_shared<AttrInfo>(StringUtils::Trim(attr)), return FAILED);
    if (new_attr_info == nullptr) {
      FE_LOGW("Operation %s failed to create a shared pointer for new_attr_info", op_content.op_type_.c_str());
      continue;
    }

    if (param_type_str == STR_REQUIRED) {
      new_attr_info->is_required_ = true;
    } else {
      new_attr_info->is_required_ = false;
    }

    std::map<std::string, ge::GeAttrValue::ValueType>::const_iterator find_attr_type =
            kAttrTypeMap.find(attr_type_str);
    if (find_attr_type == kAttrTypeMap.end()) {
      FE_LOGW("Attr type [%s] of Attr [%s] is invalid for op [%s].", attr_type_str.c_str(), attr_name.c_str(),
              op_content.op_type_.c_str());
      continue;
    }

    new_attr_info->dtype_ = find_attr_type->second;
    bool is_all = StringUtils::MatchString(STR_ALL, attr_value_str);
    new_attr_info->is_support_all_value_ = is_all;
    op_kernel_info->attrs_info_.push_back(new_attr_info);
  }

  if (InitAttrValueSub(op_content, op_kernel_info) != SUCCESS) {
    REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] OpKernelInfoConstructor::InitAttrValueSub failed.");
    return FAILED;
  }
  return SUCCESS;
}

Status OpKernelInfoConstructor::GetStrFromOpContent(const OpContent &op_content, const string &key1,
                                                    const string &key2, string &value) const {
  vector<string> attr_vec;
  auto key1_pos = op_content.map_kernel_info_.find(key1);
  if (op_content.map_kernel_info_.end() == key1_pos) {
    if (key1 != STR_RESHAPE_TYPE && key2 != STR_DEFAULT_VALUE) {
      FE_LOGD("Op %s not found %s in OpContent!", op_content.op_type_.c_str(), key1.c_str());
    }
    return OP_STORE_MAP_KEY_FIND_FAILED;
  }

  auto key2_pos = key1_pos->second.find(key2);
  if (key1_pos->second.end() == key2_pos) {
    if (key1 != STR_RESHAPE_TYPE && key2 != STR_DEFAULT_VALUE) {
      FE_LOGD("Operation %s not found in %s.%s of OpContent!", op_content.op_type_.c_str(), key1.c_str(), key2.c_str());
    }
    return OP_STORE_MAP_KEY_FIND_FAILED;
  }

  value = key2_pos->second;
  if (value.empty()) {
    return FAILED;
  }
  return SUCCESS;
}

Status OpKernelInfoConstructor::InitOpStrItem(const std::string &item_name, const std::string &item_key,
                                              const OpContent &op_content, std::string &value) const {
  std::set<std::string> empty_value_range;
  return InitOpStrItem(item_name, item_key, op_content, empty_value_range, value);
}

Status OpKernelInfoConstructor::InitOpStrItem(const std::string &item_name, const std::string &item_key,
                                              const OpContent &op_content, const std::set<std::string> &value_range,
                                              std::string &value) const {
  string item_value_str;
  Status status = GetStrFromOpContent(op_content, item_name, item_key, item_value_str);
  StringUtils::Trim(item_value_str);
  if (status == SUCCESS && !item_value_str.empty()) {
    if (!value_range.empty()) {
      if (value_range.count(item_value_str) == 0) {
        REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] The parameter value [%s] for [%s] in operation [%s] is invalid.",
                        item_value_str.c_str(), item_name.c_str(), op_content.op_type_.c_str());
        return PARAM_INVALID;
      }
    }
    value = item_value_str;
    FE_LOGD("The [%s] for operation [%s] is [%s].", item_name.c_str(), op_content.op_type_.c_str(), item_value_str.c_str());
  }
  return SUCCESS;
}

template <typename T>
Status OpKernelInfoConstructor::InitOpItemValueByMaping(const std::string &item_name, const std::string &item_key,
                                                        const OpContent &op_content,
                                                        const std::map<std::string, T> &str_item_map, T &value) const {
  string item_value_str;
  Status status = GetStrFromOpContent(op_content, item_name, item_key, item_value_str);
  StringUtils::Trim(item_value_str);
  if (status == SUCCESS && !item_value_str.empty()) {
    StringUtils::ToLowerString(item_value_str);
    auto iter = str_item_map.find(item_value_str);
    if (iter == str_item_map.end()) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] The parameter value [%s] for [%s] in operation [%s] is invalid.",
                      item_value_str.c_str(), item_name.c_str(), op_content.op_type_.c_str());
      return PARAM_INVALID;
    }
    value = iter->second;
    FE_LOGD("The [%s][%s] of op [%s] is [%s].",
            item_name.c_str(), item_key.c_str(), op_content.op_type_.c_str(), item_value_str.c_str());
  }
  return SUCCESS;
}

Status OpKernelInfoConstructor::InitOpInfo(const std::string &engine_name, const OpContent &op_content,
                                           OpKernelInfoPtr op_kernel_info) const {
  op_kernel_info->op_info_.engine = engine_name;
  op_kernel_info->op_info_.flagPartial = false;
  op_kernel_info->op_info_.flagAsync = false;
  op_kernel_info->op_info_.computeCost = COMPUTE_COST_DEFAULT;

  Status status = InitOpStrItem(STR_OP_FILE, STR_VALUE, op_content, op_kernel_info->op_info_.opFileName);
  if (status != SUCCESS) {
    return status;
  }

  status = InitOpStrItem(STR_OP_INTERFACE, STR_VALUE, op_content, op_kernel_info->op_info_.opFuncName);
  if (status != SUCCESS) {
    return status;
  }
  return SUCCESS;
}

void OpKernelInfoConstructor::SetUniqueName(InputOrOutputInfoPtr input_or_output_info_ptr) const {
  string unique_name_prefix = input_or_output_info_ptr->is_input_ ? STR_INPUT_LOWERCASE : STR_OUTPUT_LOWERCASE;
  input_or_output_info_ptr->unique_name_ =
      unique_name_prefix + to_string(input_or_output_info_ptr->index_) + "." + input_or_output_info_ptr->name_;
}

Status OpKernelInfoConstructor::InitDtypeByPattern(const map<string, string> &map_info,
                                                   InputOrOutputInfoPtr input_or_output_info,
                                                   uint32_t &dtype_and_format_size_of_first_input,
                                                   const OpPattern &op_pattern) {
  if (op_pattern == OP_PATTERN_BROADCAST || op_pattern == OP_PATTERN_BROADCAST_ENHANCED ||
      op_pattern == OP_PATTERN_REDUCE) {
    if (InitDtype(map_info, input_or_output_info, dtype_and_format_size_of_first_input) != SUCCESS) {
      REPORT_FE_ERROR("[InitOpInfoLib][InitOpKernel] Init dtypes failed by pattern.");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status OpKernelInfoConstructor::InitTuneFormatSwitch(const map<string, string> &map_info,
                                                     InputOrOutputInfoPtr &input_or_output_info,
                                                     const string &op_type) {
  string tuneformat_switch_str;
  if (GetStrFromMap(map_info, STR_TUNEFORMAT_SWITCH, tuneformat_switch_str) != SUCCESS) {
    FE_LOGD("[InitOpInfoLib][InitOpKernel] op(type:[%s]) not find tuneformat_switch_str key.", op_type.c_str());
    input_or_output_info->op_tune_format_switch_ = false;
  } else {
    auto tuneformat_switch_iter = kBoolToStringMap.find(StringUtils::Trim(tuneformat_switch_str));
    if (tuneformat_switch_iter == kBoolToStringMap.end()) {
      FE_LOGE("[InitOpInfoLib][InitOpKernel] op(type:[%s]) invalid tuneformat_switch.", op_type.c_str());
      return OP_STORE_MAP_KEY_FIND_FAILED;
    }
    input_or_output_info->op_tune_format_switch_ = tuneformat_switch_iter->second;
  }
  return SUCCESS;
}

}  // namespace fe
