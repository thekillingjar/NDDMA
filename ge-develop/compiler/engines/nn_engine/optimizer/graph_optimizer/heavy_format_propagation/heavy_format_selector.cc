/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "heavy_format_selector.h"
#include <sstream>
#include "common/math_util.h"

namespace fe {
namespace {
const uint32_t UNSUPPORT_FORMAT = 64;

bool CheckSubformatSupport(const vector<uint32_t> &kernel_sub_formats, const uint32_t &real_sub_format) {
  bool no_need_check_support = std::find(kernel_sub_formats.begin(), kernel_sub_formats.end(),
      SUPPORT_ALL_SUB_FORMAT) != kernel_sub_formats.end();
  bool subformat_support = std::find(kernel_sub_formats.begin(), kernel_sub_formats.end(),
      real_sub_format) != kernel_sub_formats.end();
  return real_sub_format == DEFAULT_SUB_FORMAT || no_need_check_support || subformat_support;
}

FormatScore GetFormatScore(const ge::Format &distributed_heavy_format, const uint32_t &real_sub_format,
                           const ge::Format &current_original_format,
                           const ge::Format &kernel_format, const vector<uint32_t> &kernel_sub_formats) {
  if (kernel_format == distributed_heavy_format) {
    if ((kernel_format == ge::FORMAT_FRACTAL_Z || kernel_format == ge::FORMAT_FRACTAL_Z_3D) && !CheckSubformatSupport(
        kernel_sub_formats, real_sub_format)) {
      return FormatScore::SUB_FORMAT_MISMATCH_SCORE;
    }
    return FormatScore::DISTRIBUTED_HEAVY_FORMAT_SCORE;
  }
  if (kernel_format == ge::FORMAT_NC1HWC0 || kernel_format == ge::FORMAT_FRACTAL_Z ||
      kernel_format == ge::FORMAT_FRACTAL_Z_3D || kernel_format == ge::FORMAT_C1HWNCoC0 ||
      kernel_format == ge::FORMAT_FRACTAL_NZ) {
      if ((kernel_format == ge::FORMAT_FRACTAL_Z || kernel_format == ge::FORMAT_FRACTAL_Z_3D) && !CheckSubformatSupport(
        kernel_sub_formats, real_sub_format)) {
        return FormatScore::SUB_FORMAT_MISMATCH_SCORE;
      }
      return FormatScore::OTHER_HEAVY_FORMAT_SCORE;
  }
  if (kernel_format == current_original_format || kernel_format == ge::FORMAT_ND) {
    return FormatScore::ORIGINAL_FORMAT_SCORE;
  }
  return FormatScore::OTHER_FORMAT_SCORE;
}

/* Get the largest element's index in vec */
template <typename T>
int32_t GetLargestElement(std::vector<T> vec) {
  T max = 0;
  int32_t matched_index = INVALID_KERNEL_INDEX;
  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec.at(i) > max) {
      max = vec.at(i);
      matched_index = static_cast<int32_t>(i);
    }
  }
  return matched_index;
}
} // namespace

HeavyFormatSelector::HeavyFormatSelector(FormatDtypeQuerierPtr format_dtype_querier_ptr)
    : format_dtype_querier_ptr_(format_dtype_querier_ptr),
      precise_dtype_matcher_ptr_(nullptr),
      input_and_output_kernel_(),
      matched_index_() {}

HeavyFormatSelector::~HeavyFormatSelector() {}

Status HeavyFormatSelector::Initalize() {
  FE_MAKE_SHARED(precise_dtype_matcher_ptr_ = std::make_shared<OpDtypePreciseMatcher>(), return FAILED);
  input_and_output_kernel_.emplace_back();
  input_and_output_kernel_.emplace_back();
  return SUCCESS;
}

Status HeavyFormatSelector::CalcFormatScore(const ge::OpDesc::Vistor<ge::GeTensorDescPtr> &all_tensors,
                                            const fe::OpKernelInfoPtr &op_kernel_info_ptr,
                                            const ge::NodePtr &node, uint32_t kernel_format_index,
                                            const HeavyFormatInfo &heavy_format_info, InputOrOutputIndex in_or_out,
                                            uint64_t &score) {
  auto op_desc_ptr = node->GetOpDesc();
  auto size = all_tensors.size();
  for (size_t index = 0; index < size; ++index) {
    auto tensor = all_tensors.at(index);
    /* Here we just skip the array size and empty check because we have
     * done that in last function SelectQualifiedFormat. */
    auto tensor_info = input_and_output_kernel_[in_or_out].at(index);
    vector<ge::Format> kernel_formats;
    vector<uint32_t> kernel_sub_formats;
    uint32_t real_sub_format = (static_cast<uint32_t>(heavy_format_info.sub_format) < DEFAULT_SUB_FORMAT) ?
        DEFAULT_SUB_FORMAT : static_cast<uint32_t>(heavy_format_info.sub_format);
    if (format_dtype_querier_ptr_->GetSupportFormatSubFormat(op_kernel_info_ptr, tensor_info, node,
                                                             kernel_formats, kernel_sub_formats,
                                                             real_sub_format) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOptJdgInst][FmtPropagate][CalFmtScore] Failed to get sub_format for node %s.",
                      op_desc_ptr->GetName().c_str());
      return FAILED;
    }

    vector<ge::DataType> kernel_data_types;
    if (format_dtype_querier_ptr_->GetSupportDataTypes(op_kernel_info_ptr, tensor_info, node,
                                                       kernel_data_types) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOptJdgInst][FmtPropagate][CalFmtScore] Failed to obtain the supported data_types for node %s.",
                      op_desc_ptr->GetName().c_str());
      return FAILED;
    }
    if (kernel_format_index >= kernel_formats.size() || kernel_format_index >= kernel_data_types.size()) {
      FE_LOGW("Node[%s, %s]: Format index %u is larger than the kernel format size %zu or the kernel data type size %zu.",
              op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr(), kernel_format_index, kernel_formats.size(),
              kernel_data_types.size());
      return FAILED;
    }
  /* Here a abnormal case will show up due to we stop propagation through all
   * inconsistent edges. The case is :
   *       input0(NCHW)     input1(ND)
   *              \        /
   *               \      /
   *                \    /
   *                 \  /
   *                  op
   *                  |
   *                 Conv2D
   * The first input will be inferred as 5HD and the second will still be ND.
   * We consider if op is a normal op if the original format is NCHW and ND,
   * it supports 5HD and ND as two inputs. If op is function op, it is also support two
   * inputs as 5HD and ND. */
    auto kernel_dst_format = kernel_formats.at(kernel_format_index);
    auto kernel_dst_dtype = kernel_data_types.at(kernel_format_index);
    if (!IsHeavyFormatConsistentWithOriFormat(tensor, kernel_dst_format, kernel_dst_dtype, op_desc_ptr)) {
        FE_LOGW("Original format %u is inconsistent with the kernel's dst format %u; format_index[%zu] will be erased.",
                tensor->GetOriginFormat(), kernel_dst_format, index);
        return UNSUPPORT_FORMAT;
    }
    uint64_t format_score = static_cast<uint64_t>(GetFormatScore(
        heavy_format_info.expected_heavy_format, real_sub_format, tensor->GetOriginFormat(),
        kernel_dst_format, kernel_sub_formats));
    FE_UINT64_ADDCHECK(score, format_score);
    score += format_score;
    if (format_score == static_cast<uint64_t>(FormatScore::SUB_FORMAT_MISMATCH_SCORE)) {
      return SUCCESS;
    }
  }
  return SUCCESS;
}

Status HeavyFormatSelector::GetMostSuitableFormatIndex(const fe::OpKernelInfoPtr &op_kernel_info_ptr,
                                                       const ge::NodePtr &current_node,
                                                       const HeavyFormatInfo &heavy_format_info,
                                                       const std::vector<IndexNameMap> &tensor_map,
                                                       int32_t &most_suitable_index) {
  /* First Clear input_and_output_kernel_ */
  for (auto &ele : input_and_output_kernel_) {
    auto new_vec = std::vector<InputOrOutputInfoPtr>();
    ele.swap(new_vec);
  }

  std::vector<uint32_t> new_matched_index = std::vector<uint32_t>();
  matched_index_.swap(new_matched_index);

  Status ret = SelectQualifiedFormat(op_kernel_info_ptr, current_node, heavy_format_info, tensor_map);
  if (ret != SUCCESS) {
    return FAILED;
  }

  ge::OpDescPtr op_desc_ptr = current_node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  auto all_input_tensor = op_desc_ptr->GetAllInputsDescPtr();
  auto all_output_tensor = op_desc_ptr->GetAllOutputsDescPtr();
  /* Here we define a vector which store the corresponding score of current
   * combination of format.
   * The format score is define as below:
   * Same as the heavy format which is distributed right now : 2000;
   * Other heavy format : 200
   * Same as the tensor's original format(ND included) : 100
   * Other format: 1 */
  /* For each index value in matched_index, we calculate its score and
   * store the score in score_record_vec */
  std::vector<uint64_t> score_record_vec;
  for (const auto &kernel_format_index : matched_index_) {
    uint64_t score = 0;
    Status in_ret = CalcFormatScore(all_input_tensor, op_kernel_info_ptr, current_node, kernel_format_index,
                                    heavy_format_info, INPUT_INDEX, score);
    if (in_ret == FAILED) {
      return FAILED;
    }
    if (in_ret == UNSUPPORT_FORMAT) {
      score_record_vec.emplace_back(0);
      continue;
    }
    in_ret = CalcFormatScore(all_output_tensor, op_kernel_info_ptr, current_node, kernel_format_index,
                             heavy_format_info, OUTPUT_INDEX, score);
    if (in_ret == FAILED) {
      return FAILED;
    }
    if (in_ret == UNSUPPORT_FORMAT) {
      score_record_vec.emplace_back(0);
      continue;
    }
    score_record_vec.emplace_back(score);
  }

  FE_LOGD("The score is %s.", StringUtils::IntegerVecToString(score_record_vec).c_str());
  auto largest_index = GetLargestElement(score_record_vec);
  if (largest_index != INVALID_KERNEL_INDEX) {
    most_suitable_index = matched_index_[largest_index];
    return SUCCESS;
  } else {
    return FAILED;
  }
}

Status HeavyFormatSelector::Match(const OpKernelInfoPtr &op_kernel_info_ptr,
                                  const ge::NodePtr &node,
                                  const ge::OpDesc::Vistor<ge::GeTensorDescPtr> &all_tensors,
                                  InputOrOutputIndex in_or_out) {
  Status ret;
  auto op_desc_ptr = node->GetOpDesc();
  for (size_t index = 0; index < all_tensors.size(); ++index) {
    auto &tensor = all_tensors.at(index);
    if (index >= input_and_output_kernel_[in_or_out].size()) {
      FE_LOGW("Output tensor index %zu exceeds kernel size %zu for operation %s.", index,
              input_and_output_kernel_[in_or_out].size(), op_desc_ptr->GetName().c_str());
      return FAILED;
    }
    auto output_info = input_and_output_kernel_[in_or_out].at(index);
    vector<ge::DataType> kernel_data_types;
    if (format_dtype_querier_ptr_->GetSupportDataTypes(op_kernel_info_ptr, output_info, node,
                                                       kernel_data_types) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOptJdgInst][FmtPropagate][Match] Failed to get the support data_types for node %s.",
                      op_desc_ptr->GetName().c_str());
      return FAILED;
    }
    ge::DataType current_data_type = tensor->GetDataType();
    ret =
        precise_dtype_matcher_ptr_->FindSuitableDtype(kernel_data_types, current_data_type, matched_index_, ge::DT_MAX);
    if (ret == FAILED) {
      FE_LOGD("FindSuitableDtype for node %s unsuccessful.", op_desc_ptr->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status HeavyFormatSelector::MatchDtypeForAllInputAndOutput(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                           const ge::NodePtr &current_node) {
  ge::OpDescPtr op_desc_ptr = current_node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  if (input_and_output_kernel_.empty() || input_and_output_kernel_.size() < INPUT_OUTPUT_INDEX_BOTTOM) {
    FE_LOGW("Input tensor vector for node %s is empty; its size is %zu.", current_node->GetName().c_str(),
            input_and_output_kernel_.size());
    return FAILED;
  }
  Status ret;
  auto all_input_tensor = op_desc_ptr->GetAllInputsDescPtr();
  if (all_input_tensor.empty()) {
    FE_LOGW("Input tensor vector of node %s is empty", current_node->GetName().c_str());
    return FAILED;
  }
  ret = Match(op_kernel_info_ptr, current_node, all_input_tensor, INPUT_INDEX);
  if (ret != SUCCESS) {
    return ret;
  }

  auto all_output_tensor = op_desc_ptr->GetAllOutputsDescPtr();
  if (all_output_tensor.empty()) {
    FE_LOGW("Output tensor vector for node %s is empty.", current_node->GetName().c_str());
    return FAILED;
  }
  ret = Match(op_kernel_info_ptr, current_node, all_output_tensor, OUTPUT_INDEX);
  if (ret != SUCCESS) {
    return ret;
  }

  FE_LOGD("After matching dtype, matched index is %s for node %s.",
          StringUtils::IntegerVecToString(matched_index_).c_str(),
          current_node->GetName().c_str());
  return SUCCESS;
}

Status HeavyFormatSelector::SelectQualifiedFormat(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                  const ge::NodePtr &current_node,
                                                  const HeavyFormatInfo &heavy_format_info,
                                                  const std::vector<IndexNameMap> &tensor_map) {
  InputOrOutputInfoPtr input_or_output_info;

  Status ret = GetAllInputAndOutputKernelInfo(op_kernel_info_ptr, current_node, tensor_map, input_and_output_kernel_);
  if (ret != SUCCESS) {
    return FAILED;
  }

  if (input_and_output_kernel_.empty() || input_and_output_kernel_.size() < INPUT_OUTPUT_INDEX_BOTTOM) {
    FE_LOGW("Input tensor vector for node %s is empty; its size is %zu.", current_node->GetName().c_str(),
            input_and_output_kernel_.size());
    return FAILED;
  }

  ret = SearchHeavyFormatInKernel(op_kernel_info_ptr, current_node, heavy_format_info);
  if (ret != SUCCESS) {
    return FAILED;
  }

  ret = MatchDtypeForAllInputAndOutput(op_kernel_info_ptr, current_node);
  if (ret != SUCCESS) {
    FE_LOGD("MatchDtypeForAllInputAndOutput failed.");
    return FAILED;
  }
  return SUCCESS;
}

Status HeavyFormatSelector::SearchHeavyFormatInKernel(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                      const ge::NodePtr& current_node,
                                                      const HeavyFormatInfo &heavy_format_info) {
  auto op_desc_ptr = current_node->GetOpDesc();
  InputOrOutputInfoPtr input_or_output_info;
  if (heavy_format_info.is_input) {
    if (static_cast<size_t>(heavy_format_info.anchor_index) >= input_and_output_kernel_[INPUT_INDEX].size()) {
      FE_LOGW("Input tensor at index %u exceeds kernel size %zu. Operation: %s.", heavy_format_info.anchor_index,
              input_and_output_kernel_[INPUT_INDEX].size(), op_desc_ptr->GetName().c_str());
      return FAILED;
    }
    input_or_output_info = input_and_output_kernel_[INPUT_INDEX].at(heavy_format_info.anchor_index);
  } else {
    if (static_cast<size_t>(heavy_format_info.anchor_index) >= input_and_output_kernel_[OUTPUT_INDEX].size()) {
      FE_LOGW("Output tensor index %u exceeds kernel size %zu. Operation: %s.", heavy_format_info.anchor_index,
              input_and_output_kernel_[OUTPUT_INDEX].size(), op_desc_ptr->GetName().c_str());
      return FAILED;
    }
    input_or_output_info = input_and_output_kernel_[OUTPUT_INDEX].at(heavy_format_info.anchor_index);
  }
  vector<ge::Format> kernel_formats;
  if (format_dtype_querier_ptr_->GetSupportFormats(op_kernel_info_ptr, input_or_output_info, current_node,
                                                   kernel_formats) != SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOptJdgInst][FmtPropagate][SearchHvyFmtInKernel] Fail to get the supported formats for node %s.",
        op_desc_ptr->GetName().c_str());
    return FAILED;
  }

  vector<ge::DataType> kernel_data_types;
  if (format_dtype_querier_ptr_->GetSupportDataTypes(op_kernel_info_ptr, input_or_output_info, current_node,
                                                     kernel_data_types) != SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOptJdgInst][FmtPropagate][SearchHvyFmtInKernel] Failed to obtain the supported data_types for node %s.",
        op_desc_ptr->GetName().c_str());
    return FAILED;
  }

  for (size_t i = 0; i < kernel_formats.size(); ++i) {
    /* Heavy format or ALL */
    if (kernel_formats[i] == heavy_format_info.expected_heavy_format) {
      matched_index_.emplace_back(i);
    }
  }

  FE_LOGD("Format index %s meets the requirements for node %s.",
          StringUtils::IntegerVecToString(matched_index_).c_str(), op_desc_ptr->GetName().c_str());
  return SUCCESS;
}

bool HeavyFormatSelector::IsDtypeSensitiveOpForHeavyFormatConsistentWithOriFormat(
    const ge::OpDescPtr &op_desc_ptr) {
  if (op_desc_ptr == nullptr || op_desc_ptr->GetInputDescPtr(0) == nullptr ||
      op_desc_ptr->GetOutputDescPtr(0) == nullptr) {
    return false;
  }
  return (IsDtypeSensitiveOp(op_desc_ptr->GetType()) &&
          (kDtypeSensitiveOpNotHeavy.count(op_desc_ptr->GetInputDescPtr(0)->GetDataType()) > 0 ||
           kDtypeSensitiveOpNotHeavy.count(op_desc_ptr->GetOutputDescPtr(0)->GetDataType()) > 0));
}

/* In current stage: HWCN and NC1HWC0, NHWC and FRACTAL_Z are two
 * inconsistent cases. */
bool HeavyFormatSelector::IsHeavyFormatConsistentWithOriFormat(const ge::GeTensorDescPtr &current_tensor,
                                                               const ge::Format &heavy_format,
                                                               const ge::DataType &dst_dtype,
                                                               const ge::OpDescPtr &op_desc_ptr) {
  if (current_tensor == nullptr) {
    return false;
  }
  if (current_tensor->GetOriginFormat() == ge::FORMAT_HWCN && heavy_format == ge::FORMAT_NC1HWC0) {
    FE_LOGD("Node[%s, %s] ori format HWCN is inconsistent with heavy format 5HD.",
            op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
    return false;
  }
  if (current_tensor->GetOriginFormat() == ge::FORMAT_ND) {
    if (heavy_format == ge::FORMAT_NC1HWC0 || heavy_format == ge::FORMAT_NDC1HWC0) {
      FE_LOGD("Node [%s, %s] original format ND is inconsistent with heavy format 5HD.",
              op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
      return false;
    }
  }
  if (current_tensor->GetOriginFormat() == ge::FORMAT_NHWC && heavy_format == ge::FORMAT_FRACTAL_Z) {
    FE_LOGD("Node [%s, %s] original format NHWC is inconsistent with the expected format FZ.",
            op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
    return false;
  }
  if (heavy_format == ge::FORMAT_FRACTAL_NZ) {
    if (current_tensor->GetOriginShape().GetDimNum() < MINIMUM_NZ_SHAPE_DIM_NUM &&
        !current_tensor->GetOriginShape().IsUnknownDimNum()) {
      FE_LOGD("Node[%s, %s] ori shape dim num is inconsistent with heavy format NZ.",
              op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
      return false;
    }
  }
  auto cur_dtype = current_tensor->GetDataType();
  if (cur_dtype == ge::DT_INT64 && cur_dtype != dst_dtype) {
    FE_LOGD("Node [%s, %s] does not support int64 dtype propagation.",
            op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
    return false;
  }
  if (current_tensor->GetOriginFormat() == ge::FORMAT_ND) {
    if ((heavy_format == ge::FORMAT_FRACTAL_Z || heavy_format == ge::FORMAT_FRACTAL_Z_3D) &&
        current_tensor->GetOriginShape().GetDimNum() > MINIMUM_NZ_SHAPE_DIM_NUM) {
      FE_LOGD("Node[%s, %s] ori format ND is inconsistent with heavy format FZ when shape dim is bigger than 2.",
              op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
      return false;
    }
  }
  if (IsDtypeSensitiveOpForHeavyFormatConsistentWithOriFormat(op_desc_ptr)) {
    FE_LOGD("Node[%s, %s] is dtype sensitive op, which is unsupported.",
            op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
    return false;
  }
  return true;
}
}  // namespace fe
