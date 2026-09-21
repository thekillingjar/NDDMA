/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dvpp_ops_checker.h"
#include "common/dvpp_chip_capability.h"
#include "common/dvpp_ops_lib.h"
#include "util/dvpp_constexpr.h"
#include "util/dvpp_define.h"
#include "util/dvpp_log.h"
#include "util/util.h"

namespace dvpp {
bool DvppOpsChecker::CheckInputSupported(const DvppOpInfo& dvpp_op_info,
    const ge::OpDescPtr& op_desc_ptr, std::string& unsupported_reason) {
  // check input num
  size_t intput_size = op_desc_ptr->GetInputsSize();
  DVPP_CHECK_IF_THEN_DO(
      intput_size != dvpp_op_info.inputs.size(),
      unsupported_reason = "op[" + op_desc_ptr->GetType() +
          "]: inputs num[" + std::to_string(intput_size) +
          "] should be " + std::to_string(dvpp_op_info.inputs.size());
      DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str());
      return false);

  // check every input
  for (uint32_t input_index = 0; input_index < intput_size; ++input_index) {
    auto input_desc = op_desc_ptr->GetInputDesc(input_index);
    // check input exist
    // 这里不能校验name 因为OptimizeGraphPrepare阶段还没有name
    std::string input_name = "input" + std::to_string(input_index);
    const auto iter = dvpp_op_info.inputs.find(input_name);
    DVPP_CHECK_IF_THEN_DO(iter == dvpp_op_info.inputs.end(),
        unsupported_reason = "can not find op[" + op_desc_ptr->GetType() +
            "]: input[" + input_name + "] in dvpp ops info store";
        DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str());
        return false);

    // check data type
    struct InOutInfo in_out_info = iter->second;
    std::set<ge::DataType> data_type_set;
    StringToDataType(in_out_info.dataType, data_type_set);
    auto data_type = input_desc.GetDataType();
    DVPP_CHECK_IF_THEN_DO(data_type_set.find(data_type) == data_type_set.end(),
        std::string data_type_str;
        (void)DataTypeToString(data_type, data_type_str);
        unsupported_reason = "can not find op[" + op_desc_ptr->GetType() +
            "]: input[" + input_name + "] data type[" + data_type_str +
            "] in dvpp ops info store";
        DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str());
        return false);

    // check data format
    // 如果信息库里的数据为空 则表示没有限制 无需校验
    DVPP_CHECK_IF_THEN_DO(in_out_info.dataFormat == "", continue);

    std::set<ge::Format> data_format_set;
    StringToDataFormat(in_out_info.dataFormat, data_format_set);
    auto data_format = static_cast<ge::Format>(
        ge::GetPrimaryFormat(static_cast<int32_t>(input_desc.GetFormat())));
    DVPP_CHECK_IF_THEN_DO(
        data_format_set.find(data_format) == data_format_set.end(),
        std::string data_format_str;
        (void)DataFormatToString(data_format, data_format_str);
        unsupported_reason = "can not find op[" + op_desc_ptr->GetType() +
            "]: input[" + input_name + "] data format[" + data_format_str +
            "] in dvpp ops info store";
        DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str());
        return false);
  }

  return true;
}

bool DvppOpsChecker::CheckOutputSupported(const DvppOpInfo& dvpp_op_info,
    const ge::OpDescPtr& op_desc_ptr, std::string& unsupported_reason) {
  // check output num
  size_t output_size = op_desc_ptr->GetOutputsSize();
  DVPP_CHECK_IF_THEN_DO(
      output_size != dvpp_op_info.outputs.size(),
      unsupported_reason = "op[" + op_desc_ptr->GetType() +
          "]: outputs num[" + std::to_string(output_size) +
          "] should be " + std::to_string(dvpp_op_info.outputs.size());
      DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str());
      return false);

  // check every output
  for (uint32_t output_index = 0; output_index < output_size; ++output_index) {
    auto output_desc = op_desc_ptr->GetOutputDesc(output_index);
    // check output exist
    // 这里不能校验name 因为OptimizeGraphPrepare阶段还没有name
    std::string output_name = "output" + std::to_string(output_index);
    const auto iter = dvpp_op_info.outputs.find(output_name);
    DVPP_CHECK_IF_THEN_DO(iter == dvpp_op_info.outputs.end(),
        unsupported_reason = "can not find op[" + op_desc_ptr->GetType() +
            "]: output[" + output_name + "] in dvpp ops info store";
        DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str());
        return false);

    // check data type
    struct InOutInfo in_out_info = iter->second;
    std::set<ge::DataType> data_type_set;
    StringToDataType(in_out_info.dataType, data_type_set);
    auto data_type = output_desc.GetDataType();
    DVPP_CHECK_IF_THEN_DO(data_type_set.find(data_type) == data_type_set.end(),
        std::string data_type_str;
        (void)DataTypeToString(data_type, data_type_str);
        unsupported_reason = "can not find op[" + op_desc_ptr->GetType() +
            "]: output[" + output_name + "] data type[" + data_type_str +
            "] in dvpp ops info store";
        DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str());
        return false);

    // check data format
    // 如果信息库里的数据为空 则表示没有限制 无需校验
    DVPP_CHECK_IF_THEN_DO(in_out_info.dataFormat == "", continue);

    std::set<ge::Format> data_format_set;
    StringToDataFormat(in_out_info.dataFormat, data_format_set);
    auto data_format = static_cast<ge::Format>(
        ge::GetPrimaryFormat(static_cast<int32_t>(output_desc.GetFormat())));
    DVPP_CHECK_IF_THEN_DO(
        data_format_set.find(data_format) == data_format_set.end(),
        std::string data_format_str;
        (void)DataFormatToString(data_format, data_format_str);
        unsupported_reason = "can not find op[" + op_desc_ptr->GetType() +
            "]: output[" + output_name + "] data format[" + data_format_str +
            "] in dvpp ops info store";
        DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str());
        return false);
  }

  return true;
}

bool DvppOpsChecker::CheckAttrSupported(const DvppOpInfo& dvpp_op_info,
    const ge::OpDescPtr& op_desc_ptr, std::string& unsupported_reason) {
  // 通过信息库反向去校验算子 信息库如果没有attr 表示没有限制 无需校验
  auto op_desc_attrs = op_desc_ptr->GetAllAttrs();
  for (auto& attr : dvpp_op_info.attrs) {
    // check attr exist
    std::string attr_name = attr.first;
    const auto iter = op_desc_attrs.find(attr_name);
    DVPP_CHECK_IF_THEN_DO(iter == op_desc_attrs.end(),
        unsupported_reason = "op[" + op_desc_ptr->GetType() +
            "]: does not have attr[" + attr_name + "]";
        DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str());
        return false);

    // check attr value
    const auto ge_value_type = iter->second.GetValueType();
    std::string attr_value = attr.second.value;
    bool ret = CheckAttrValue(
        op_desc_ptr, attr_name, ge_value_type, attr_value);
    DVPP_CHECK_IF_THEN_DO(!ret,
        unsupported_reason = "do not support op[" + op_desc_ptr->GetType() +
            "] attr[" + attr_name + "] value";
        DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str());
        return false);
  }

  return true;
}

bool DvppOpsChecker::CheckParameterSupported(const DvppOpInfo& dvpp_op_info,
    const ge::NodePtr& node, std::string& unsupported_reason) {
  auto op_desc_ptr = node->GetOpDesc();

  // check input
  bool ret = CheckInputSupported(dvpp_op_info, op_desc_ptr, unsupported_reason);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  // check output
  ret = CheckOutputSupported(dvpp_op_info, op_desc_ptr, unsupported_reason);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  // check attr
  ret = CheckAttrSupported(dvpp_op_info, op_desc_ptr, unsupported_reason);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  // check chip capability
  ret = DvppChipCapability::CheckSupported(
      dvpp_op_info, node, unsupported_reason);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  return true;
}

bool DvppOpsChecker::CheckSupported(
    const ge::NodePtr& node, std::string& unsupported_reason) {
  // check node
  DVPP_CHECK_IF_THEN_DO(node == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("node is nullptr");
      return false);

  // check op_desc_ptr
  auto op_desc_ptr = node->GetOpDesc();
  DVPP_CHECK_IF_THEN_DO(op_desc_ptr == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("op_desc_ptr is nullptr");
      return false);

  // check op_type
  std::string op_type = op_desc_ptr->GetType();
  DVPP_CHECK_IF_THEN_DO(op_type.empty(),
      DVPP_REPORT_INNER_ERR_MSG("op[%s] op type is empty",
                              op_desc_ptr->GetName().c_str());
      return false);

  // get dvpp op info
  DvppOpInfo dvpp_op_info;
  bool ret = DvppOpsLib::Instance().GetDvppOpInfo(op_type, dvpp_op_info);
  DVPP_CHECK_IF_THEN_DO(!ret,
      unsupported_reason = "dvpp does not support op[" + op_type + "]";
      DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str());
      return false);

  // 检查是否已校验过 如果能获取ATTR_DVPP_CHECK_RESULT 则说明已校验过
  bool result = false;
  ret = ge::AttrUtils::GetBool(op_desc_ptr, ATTR_DVPP_CHECK_RESULT, result);
  DVPP_CHECK_IF_THEN_DO(ret,
      DVPP_ENGINE_LOG_EVENT("dvpp already checked this op[%s]",
                            op_desc_ptr->GetName().c_str());
      (void)ge::AttrUtils::GetStr(
          op_desc_ptr, ATTR_DVPP_CHECK_UNSUPPORT_REASON, unsupported_reason);
      return result);

  // check patameter
  ret = CheckParameterSupported(dvpp_op_info, node, unsupported_reason);
  // 记录校验结果
  (void)ge::AttrUtils::SetBool(op_desc_ptr, ATTR_DVPP_CHECK_RESULT, ret);
  (void)ge::AttrUtils::SetStr(
      op_desc_ptr, ATTR_DVPP_CHECK_UNSUPPORT_REASON, unsupported_reason);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  DVPP_ENGINE_LOG_EVENT("dvpp support op[%s]", op_desc_ptr->GetName().c_str());
  return true;
}

bool DvppOpsChecker::CheckAxisC(const ge::NodePtr& node) {
  // check op_desc_ptr
  auto op_desc_ptr = node->GetOpDesc();
  DVPP_CHECK_IF_THEN_DO(op_desc_ptr == nullptr, return false);

  // check input0 和 input1 维度是否相同 且为3或者4
  auto input_0 = op_desc_ptr->GetInputDesc(kNum0);
  auto input_0_dims = input_0.GetShape().GetDims();
  auto input_1 = op_desc_ptr->GetInputDesc(kNum1);
  auto input_1_dims = input_1.GetShape().GetDims();
  DVPP_CHECK_IF_THEN_DO(input_0_dims.size() != input_1_dims.size(),
                        return false);
  DVPP_CHECK_IF_THEN_DO(
      (input_1_dims.size() != kNum3) && (input_1_dims.size() != kNum4),
      return false);

  // find c-axis position in input1
  uint32_t c_index = kNum4;
  for (uint32_t index = 0; index < input_1_dims.size(); ++index) {
    DVPP_CHECK_IF_THEN_DO(input_1_dims[index] == kNum3,
        c_index = index;
        break);
  }
  DVPP_CHECK_IF_THEN_DO(c_index == kNum4, return false);

  // check c-axis value in input0
  DVPP_CHECK_IF_THEN_DO(input_0_dims[c_index] != kNum3, return false);

  return true;
}

// ge:Maximum  ge:Constant
//        \     /
//          Sub  ge:Constant
//            \   /
//             Mul
// 当前仅支持特定场景下的Sub和Mul算子融合为Normalize算子
bool DvppOpsChecker::CheckSubAndMulFusedIntoNormalizeV2(
    const ge::NodePtr& mul_node) {
  // check node
  DVPP_CHECK_IF_THEN_DO(mul_node == nullptr, return false);

  // check 当前node 是否为 Mul
  DVPP_CHECK_IF_THEN_DO(mul_node->GetType() != kNodeMul, return false);

  // check Mul 输入0 是否为 Sub
  const size_t mul_in_size = 2; // 两个输入
  DVPP_CHECK_IF_THEN_DO(mul_node->GetInDataNodes().size() != mul_in_size,
      return false);
  auto sub_node = mul_node->GetInDataNodes().at(kNum0);
  DVPP_CHECK_IF_THEN_DO(sub_node == nullptr, return false);
  DVPP_CHECK_IF_THEN_DO(sub_node->GetType() != kNodeSub, return false);

  // check Mul 输入1 是否为 Const或Constant
  const size_t sub_in_size = 2; // 两个输入
  DVPP_CHECK_IF_THEN_DO(sub_node->GetInDataNodes().size() != sub_in_size,
      return false);
  auto const_node = mul_node->GetInDataNodes().at(kNum1);
  DVPP_CHECK_IF_THEN_DO(const_node == nullptr, return false);
  DVPP_CHECK_IF_THEN_DO((const_node->GetType() != kNodeConst) &&
                        (const_node->GetType() != kNodeConstant),
      return false);

  // check mul 两个输入的C轴
  bool ret = CheckAxisC(mul_node);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  // check Sub 输入0 不能为 Const或Constant
  auto in_node = sub_node->GetInDataNodes().at(kNum0);
  DVPP_CHECK_IF_THEN_DO(in_node == nullptr, return false);
  DVPP_CHECK_IF_THEN_DO((in_node->GetType() == kNodeConst) ||
                        (in_node->GetType() == kNodeConstant),
      return false);

  // check Sub 输入1 是否为 Const或Constant
  const_node = sub_node->GetInDataNodes().at(kNum1);
  DVPP_CHECK_IF_THEN_DO(const_node == nullptr, return false);
  DVPP_CHECK_IF_THEN_DO((const_node->GetType() != kNodeConst) &&
                        (const_node->GetType() != kNodeConstant),
      return false);

  // check sub 两个输入的C轴
  ret = CheckAxisC(sub_node);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  return true;
}
} // namespace dvpp
