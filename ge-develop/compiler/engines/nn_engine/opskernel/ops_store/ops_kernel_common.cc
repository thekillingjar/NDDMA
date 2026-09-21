/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_store/ops_kernel_common.h"
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "common/configuration.h"
#include "common/fe_type_utils.h"
#include "platform/platform_info.h"
#include "common/platform_utils.h"
#include "common/scope_allocator.h"
#include "common/math_util.h"
#include "graph/ge_context.h"
#include "graph/utils/op_desc_utils.h"
#include "ops_store/ops_kernel_manager.h"
#include "graph/utils/op_type_utils.h"
#include "common/aicore_util_attr_define.h"
#include "common/string_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"

namespace fe {
bool IsHeavyOp(ge::OpDescPtr op_desc_ptr) {
  std::string engine_name;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, kAttrEngineType, engine_name);
  OpKernelInfoPtr current_op_kernel_info = OpsKernelManager::Instance(engine_name.c_str()).GetOpKernelInfoByOpDesc(
      op_desc_ptr);
  if (current_op_kernel_info == nullptr) {
    FE_LOGW("current node[%s, %s] without kernel info", op_desc_ptr->GetName().c_str(),
            op_desc_ptr->GetType().c_str());
    return false;
  }
  return current_op_kernel_info->IsHeavyOp();
}

bool IsAllowNzMatmul(const ge::NodePtr &node_ptr) {
  const std::string &op_type = node_ptr->GetOpDesc()->GetType();
  if (KFeFormatModeFilterOp.count(op_type) == 0) {
    return false;
  }
  bool ffts_switch = false;
  (void)ge::AttrUtils::GetBool(node_ptr->GetOwnerComputeGraph(), "_ffts_switch", ffts_switch);
  if (ffts_switch) {
    return false;
  }
  bool allow_nz = false;
  (void) ge::AttrUtils::GetBool(node_ptr->GetOpDesc(), "allow_nz", allow_nz);
  return allow_nz;
}

bool IsHeavyOp(ge::NodePtr node_ptr) {
  return IsAllowNzMatmul(node_ptr) || IsHeavyOp(node_ptr->GetOpDesc());
}

bool IsDynamicInputOrOutput(const InputOrOutputInfoPtr& input_output_info) {
  return input_output_info->GetParamType() == OpParamType::DYNAMIC;
}

std::string GetOpParamTypeStr(OpParamType op_param_type) {
  auto iter = kParamTypeStrMap.find(op_param_type);
  if (iter == kParamTypeStrMap.end()) {
    return "unknow-op-param-type";
  } else {
    return iter->second;
  }
}

bool CheckInputSubString(const std::string& op_desc_input_name, const std::string& info_input_name) {
  size_t length_of_info_input_name = info_input_name.length();
  size_t length_of_op_desc_input_name = op_desc_input_name.length();
  if (length_of_info_input_name > length_of_op_desc_input_name) {
    return false;
  } else {
    /* LengthOfInfoInputName less than length_of_op_desc_input_name */
    if (op_desc_input_name.substr(0, length_of_info_input_name) == info_input_name) {
      /* Get from the first char after "infoInputName"
       * to the end of op_desc_input_name */
      std::string rest = op_desc_input_name.substr(length_of_info_input_name);
      if (rest.empty()) {
        return true;
      }
      if (StringUtils::IsInteger(rest)) {
        return true;
      } else {
        /* In other cases, we consider this input name of op_desc is illegal.
         * Digits should only appears at the end of name
         * as index. */
        FE_LOGW("Illegal input name [%s] in opdesc during comparison with input name [%s].", op_desc_input_name.c_str(),
                info_input_name.c_str());
        return false;
      }
    } else {
      return false;
    }
  }
}

void CheckSpecialCases(const std::vector<InputOrOutputInfoPtr>& input_or_output_info, IndexNameMap& index_name_map,
                       uint32_t index, uint32_t op_desc_input_or_output_size, bool& has_found) {
  if (input_or_output_info.size() == 1 && input_or_output_info[0]->GetParamType() == DYNAMIC) {
    has_found = true;
    index_name_map[index] = input_or_output_info[0]->GetName();
  }
  if (!has_found) {
    size_t optional_count = 0;
    // Find the count of input or output whose parameter type is optional
    for (size_t loop = 0; loop < input_or_output_info.size(); loop++) {
      if (input_or_output_info[loop]->GetParamType() == OPTIONAL) {
        if (CheckSizetAddOverFlow(optional_count, 1) != SUCCESS) {
          REPORT_FE_ERROR("[GraphOpt][Setcheck][ChkSplCases] The addition between [%lu] and 1 is overflow.",
                          optional_count);
          return;
        }
        optional_count++;
      }
    }
    // If more than one optional input or output is found, can not decide which
    // one should be choose.
    if ((op_desc_input_or_output_size >= input_or_output_info.size() - optional_count) &&
        (op_desc_input_or_output_size <= input_or_output_info.size())) {
      for (auto const& ele : input_or_output_info) {
        uint32_t index_in_op_kernel = ele->GetIndex();
        if (index == index_in_op_kernel) {
          has_found = true;
          index_name_map[index] = ele->GetName();
          break;
        }
      }
    }
  }
}

Status GetInputIndexNameMap(const ge::OpDesc &op_desc, const OpKernelInfo &op_kernel_info,
                            IndexNameMap &input_map) {
  const std::vector<InputOrOutputInfoPtr>& input_info = op_kernel_info.GetAllInputInfo();
  size_t input_size_in_op_kernel = input_info.size();
  if (input_size_in_op_kernel == 0) {
    return fe::SUCCESS;
  }
  auto input_desc_size = op_desc.GetAllInputsSize();
  for (size_t i = 0; i < input_desc_size; i++) {
    std::string op_desc_input_name = op_desc.GetInputNameByIndex(i);
    FE_LOGD("Op[name:%s,type:%s] op desc index is %zu, desc name is %s.",
            op_desc.GetName().c_str(),
            op_desc.GetType().c_str(), i, op_desc_input_name.c_str());
    bool has_found = false;

    for (auto const& ele : input_info) {
      std::string info_input_name = ele->GetName();
      if ((!IsDynamicInputOrOutput(ele) && op_desc_input_name == info_input_name) ||
          (IsDynamicInputOrOutput(ele) && CheckInputSubString(op_desc_input_name, info_input_name))) {
        has_found = true;
        input_map[i] = info_input_name;
        break;
      }
    }
    // Now op node info is not created by IR, so many node name is none or is
    // wrong. Fix this problem by match the input count with the input count in op kernel info
    bool input_size_match_flag =
        (input_desc_size == input_size_in_op_kernel ||
         (input_desc_size == input_size_in_op_kernel - 1 && op_kernel_info.GetOpStoreImplType() == EN_IMPL_PLUGIN_TBE));
    if (!has_found && (input_size_match_flag)) {
      for (auto const& ele : input_info) {
        uint32_t index = ele->GetIndex();
        if (index == i) {
          has_found = true;
          input_map[i] = ele->GetName();
          break;
        }
      }
    }
    if (!has_found) {
      CheckSpecialCases(input_info, input_map, i, op_desc.GetAllInputsSize(), has_found);
    }
    if (!has_found) {
      FE_LOGI("Input name[%s] index %zu is not found in kernel of [%s]. "
              "Size in Opdesc is [%zu] and in kernel is [%zu].",
              op_desc_input_name.c_str(), i, op_kernel_info.GetOpType().c_str(), op_desc.GetAllInputsSize(),
              input_info.size());
      return fe::FAILED;
    }
  }
  return fe::SUCCESS;
}

Status GetOutputIndexNameMap(const ge::OpDesc &op_desc, const OpKernelInfo &op_kernel_info, IndexNameMap &output_map) {
  const std::vector<InputOrOutputInfoPtr>& output_info = op_kernel_info.GetAllOutputInfo();
  size_t output_size_in_op_kernel = output_info.size();
  if (output_size_in_op_kernel == 0) {
    return fe::SUCCESS;
  }
  for (size_t i = 0; i < op_desc.GetAllOutputsDescSize(); i++) {
    std::string op_desc_output_name = op_desc.GetOutputNameByIndex(i);
    FE_LOGD("Op[name:%s,type:%s] op desc index is %zu, desc name is %s.",
            op_desc.GetName().c_str(), op_desc.GetType().c_str(), i, op_desc_output_name.c_str());
    bool has_found = false;
    auto output0_op_kernel_info = output_info[0];
    if (output_size_in_op_kernel == 1 && output_info[0]->GetParamType() == DYNAMIC) {
      has_found = true;
      output_map[i] = output0_op_kernel_info->GetName();
      continue;
    }
    for (auto const& ele : output_info) {
      std::string info_output_name = ele->GetName();
      if ((!IsDynamicInputOrOutput(ele) && op_desc_output_name == info_output_name) ||
          (IsDynamicInputOrOutput(ele) && CheckInputSubString(op_desc_output_name, info_output_name))) {
        has_found = true;
        output_map[i] = info_output_name;
        break;
      }
    }
    // Now op node info is not created by IR, so many node name is none or is wrong.
    // Fix this problem by match the input count with the input count in op kernel info
    if (!has_found && op_desc.GetOutputsSize() == output_info.size()) {
      for (auto const& ele : output_info) {
        uint32_t index = ele->GetIndex();
        if (index == i) {
          has_found = true;
          output_map[i] = ele->GetName();
          break;
        }
      }
    }
    if (!has_found) {
      CheckSpecialCases(output_info, output_map, i, op_desc.GetAllOutputsDescSize(), has_found);
    }
    if (!has_found) {
      FE_LOGI("Output name[%s] index %zu is not found in kernel of [%s]. "
              "Size in Opdesc is [%u] and in kernel is [%zu].",
              op_desc_output_name.c_str(), i, op_kernel_info.GetOpType().c_str(), op_desc.GetAllOutputsDescSize(),
              output_info.size());
      return fe::FAILED;
    }
  }
  return fe::SUCCESS;
}

Status GetInputOutputNameMap(const ge::OpDesc &op_desc, const OpKernelInfoPtr &op_kernel_info_ptr,
                             IndexNameMap &input_map, IndexNameMap &output_map) {
  // feed all inputs to TbeOpInfo
  FE_CHECK(op_kernel_info_ptr == nullptr,
           REPORT_FE_ERROR("[GraphOpt][Setcheck][GetInOutNm] opKernelInfoPtr is nullptr."),
           return FAILED);

  if (GetInputIndexNameMap(op_desc, *op_kernel_info_ptr, input_map) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Setcheck][GetInOutNm] Failed to get input index name map for op %s.",
                    op_desc.GetName().c_str());
    return FAILED;
  }

  if (GetOutputIndexNameMap(op_desc, *op_kernel_info_ptr, output_map) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Setcheck][GetInOutNm] Failed to get output index name map for op %s.",
                    op_desc.GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status GetOutputNameMap(const ge::OpDesc& op_desc, const OpKernelInfoPtr& op_kernel_info_ptr,
                        IndexNameMap& output_map) {
  FE_CHECK(op_kernel_info_ptr == nullptr, REPORT_FE_ERROR("[GraphOpt][Setcheck][GetOutNm] opKernelInfoPtr is nullptr"),
           return FAILED);
  if (GetOutputIndexNameMap(op_desc, *op_kernel_info_ptr, output_map) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Setcheck][GetOutNm] Failed to GetOutputIndexNameMap for op %s.",
                    op_desc.GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

bool GetInputOutputNameMap(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                           IndexNameMap &input_map, IndexNameMap &output_map, UnSupportedReason &reason) {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  ge::OpDesc &op_desc = *(op_desc_ptr.get());
  FE_CHECK(op_kernel_info_ptr == nullptr, FE_LOGI("opKernelInfoPtr must not be nullptr"), return false);
  if (GetInputIndexNameMap(op_desc, *(op_kernel_info_ptr.get()), input_map) != SUCCESS) {
    reason.reason = "the inputs in op desc and inputs in op information library are not matched";
    reason.reason_id = static_cast<uint64_t>(OpNotSupportedReasonID::EN_INPUTS_NUM_NOT_MATCH);
    FE_LOGW("[ChkSpt][GetInputName]Failed to GetInputIndexNameMap for op %s.", op_desc.GetName().c_str());
    return false;
  }

  if (GetOutputIndexNameMap(op_desc, *(op_kernel_info_ptr.get()), output_map) != SUCCESS) {
    reason.reason = "the outputs in op desc and outputs in op information library are not matched";
    reason.reason_id = static_cast<uint64_t>(OpNotSupportedReasonID::EN_OUTPUTS_NUM_NOT_MATCH);
    FE_LOGW("[ChkSpt][GetOutputName]Failed to GetOutputIndexNameMap for op %s.", op_desc.GetName().c_str());
    return false;
  }
  return true;
}

bool CheckVirtualSoftsyncOp(const OpKernelInfoPtr &op_kernel_ptr, const ge::OpDescPtr &op_desc_ptr) {
  bool virtual_env = Configuration::Instance(AI_CORE_NAME).IsEnableVirtualType();
  bool is_soft_sync_op = op_kernel_ptr->IsSoftSyncOp();
  bool is_dynamic_shape = UnknownShapeUtils::IsUnknownShapeOp(*op_desc_ptr);
  return (virtual_env && !is_dynamic_shape && is_soft_sync_op);
}

Status GetAllInputAndOutputKernelInfo(const OpKernelInfoPtr& op_kernel_info_ptr, const ge::NodePtr& current_node,
                                      const std::vector<IndexNameMap>& tensor_map,
                                      std::vector<std::vector<InputOrOutputInfoPtr>>& input_and_output_kernel) {
  ge::OpDescPtr op_desc_ptr = current_node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  auto all_input_tensor = op_desc_ptr->GetAllInputsDescPtr();
  if (all_input_tensor.empty()) {
    FE_LOGW("Input tensor vector of node %s is empty", current_node->GetName().c_str());
    return FAILED;
  }

  if (input_and_output_kernel.size() != INPUT_OUTPUT_INDEX_BOTTOM) {
    FE_LOGW("Size of input kernel vector %zu is not correct for node %s.", input_and_output_kernel.size(),
            current_node->GetName().c_str());
    return FAILED;
  }

  for (size_t index = 0; index < op_desc_ptr->GetAllInputsSize(); index++) {
    auto input_desc = op_desc_ptr->GetInputDescPtr(index);
    if (input_desc == nullptr) {
      continue;
    }
    InputOrOutputInfoPtr input_info;
    auto iter = tensor_map[INPUT_INDEX].find(index);
    if (iter == tensor_map[INPUT_INDEX].end()) {
      FE_LOGW("Can not find input %zu in tensor map!", index);
      continue;
    }
    FE_LOGD("Valid input index %zu for node[%s].", index, op_desc_ptr->GetName().c_str());
    Status ret = op_kernel_info_ptr->GetInputInfoByName(iter->second, input_info);
    if (ret == FAILED) {
      return FAILED;
    }
    input_and_output_kernel[INPUT_INDEX].emplace_back(input_info);
  }

  for (size_t index = 0; index < op_desc_ptr->GetAllOutputsDescSize(); index++) {
    auto output_desc = op_desc_ptr->GetOutputDescPtr(index);
    if (output_desc == nullptr) {
      continue;
    }
    InputOrOutputInfoPtr output_info;
    auto iter = tensor_map[OUTPUT_INDEX].find(index);
    if (iter == tensor_map[OUTPUT_INDEX].end()) {
      FE_LOGW("Can not find output %zu in tensor map!", index);
      continue;
    }
    FE_LOGD("Valid output index %zu for node[%s].", index, op_desc_ptr->GetName().c_str());
    Status ret = op_kernel_info_ptr->GetOutputInfoByName(iter->second, output_info);
    if (ret == FAILED) {
      return FAILED;
    }
    input_and_output_kernel[OUTPUT_INDEX].emplace_back(output_info);
  }
  return SUCCESS;
}

void GenerateOpSupportInfo(const OpKernelInfoPtr &op_kernel_info_ptr, const bool &is_dynamic_impl,
                           const std::map<string, vector<ge::Format>> &format_map,
                           const std::map<string, vector<ge::DataType>> &datatype_map,
                           std::string &op_support_info) {
  nlohmann::json op_info_json;
  op_info_json["is_dynamic_impl"] = is_dynamic_impl ? kStrTrue : kStrFalse;
  for (const InputOrOutputInfoPtr &input_info_ptr : op_kernel_info_ptr->GetAllInputInfo()) {
    nlohmann::json input_json;
    input_json["name"] = input_info_ptr->GetName();
    input_json["param_type"] = GetOpParamTypeStr(input_info_ptr->GetParamType());
    std::string unique_name = input_info_ptr->GetUniqueName();
    auto format_iter = format_map.find(unique_name);
    if (format_iter != format_map.end()) {
      input_json["format"] = GetStrByFormatVec(format_iter->second);
    }
    auto dtype_iter = datatype_map.find(unique_name);
    if (dtype_iter != datatype_map.end()) {
      input_json["dtype"] = GetStrByDataTypeVec(dtype_iter->second);
    }
    op_info_json[unique_name.substr(0, unique_name.find("."))] = input_json;
  }
  for (const InputOrOutputInfoPtr &output_info_ptr : op_kernel_info_ptr->GetAllOutputInfo()) {
    nlohmann::json output_json;
    output_json["name"] = output_info_ptr->GetName();
    output_json["param_type"] = GetOpParamTypeStr(output_info_ptr->GetParamType());
    std::string unique_name = output_info_ptr->GetUniqueName();
    auto format_iter = format_map.find(unique_name);
    if (format_iter != format_map.end()) {
      output_json["format"] = GetStrByFormatVec(format_iter->second);
    }
    auto dtype_iter = datatype_map.find(unique_name);
    if (dtype_iter != datatype_map.end()) {
      output_json["dtype"] = GetStrByDataTypeVec(dtype_iter->second);
    }
    op_info_json[unique_name.substr(0, unique_name.find("."))] = output_json;
  }
  op_support_info = op_info_json.dump();
}
}  // namespace fe
