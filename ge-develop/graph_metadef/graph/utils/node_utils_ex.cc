/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/node_utils_ex.h"
#include "graph_metadef/common/ge_common/util.h"
#include "common/util/trace_manager/trace_manager.h"
#include "graph/refiner/format_refiner.h"
#include "graph/shape_refiner.h"
#include "graph/normal_graph/operator_impl.h"
#include "graph/operator_factory_impl.h"
#include "graph/common_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/util/mem_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "base/err_msg.h"

namespace ge {
namespace {
bool NeedUpdateIOName(const OpDescPtr &op_desc) {
  const auto &input_name_2_idx = op_desc->GetAllInputName();
  const bool is_input_names_empty = (op_desc->GetInputsSize() > 0U) && input_name_2_idx.empty();
  const bool is_default_input_name = !input_name_2_idx.empty() &&
    StringUtils::StartWith(input_name_2_idx.cbegin()->first, "__input");
  if (is_input_names_empty || is_default_input_name) {
    return true;
  }

  const auto &output_name_2_idx = op_desc->GetAllOutputName();
  const bool is_output_names_empty = (op_desc->GetOutputsSize() > 0U) && output_name_2_idx.empty();
  const bool is_default_output_name = !output_name_2_idx.empty() &&
    StringUtils::StartWith(output_name_2_idx.cbegin()->first, "__output");
  if (is_output_names_empty || is_default_output_name) {
    return true;
  }
  return false;
}
std::string IoNameToString(const std::string &prefix, const std::map<std::string, uint32_t> &io_names) {
  std::stringstream ss;
  ss << prefix << ":";
  if (io_names.empty()) {
    ss << "empty";
    return ss.str();
  }
  for (const auto &pair : io_names) {
    ss << "[" << pair.second << "," << pair.first << "]";
  }
  return ss.str();
}
} // namespace
graphStatus NodeUtilsEx::InferShapeAndType(const NodePtr &node) {
  GE_CHECK_NOTNULL(node, ", Node is null for Infer Shape.");
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  return ShapeRefiner::InferShapeAndType(node, op);
}

graphStatus NodeUtilsEx::InferOriginFormat(const NodePtr &node) {
  GE_CHECK_NOTNULL(node, ", Node is null for Infer Format.");
  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc, ", Op is null for Infer Format.");
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  return OpDescUtilsEx::CallInferFormatFunc(op_desc, op);
}

graphStatus NodeUtilsEx::IsInputsValid(const NodePtr &node) {
  const auto &op_desc = node->GetOpDesc();
  for (const auto &in_anchor : node->GetAllInDataAnchorsPtr()) {
    if (in_anchor == nullptr) {
      GELOGW("[Verify][CheckParam] In data anchor is null");
      continue;
    }
    const bool valid_anchor = OpTypeUtils::IsDataNode(node->GetType()) ||
                              (node->GetType() == CONSTANT) || (node->GetType() == VARIABLE) ||
                              (node->GetType() == CONSTANTOP) ||
                              (op_desc->MutableInputDesc(static_cast<uint32_t>(in_anchor->GetIdx())) == nullptr) ||
                              (in_anchor->GetPeerAnchorsSize() > 0UL);
    if (!valid_anchor) {
      REPORT_PREDEFINED_ERR_MSG(
          "E11019", std::vector<const char *>({"opname", "index"}),
          std::vector<const char *>({node->GetName().c_str(), std::to_string(in_anchor->GetIdx()).c_str()}));
      GELOGE(GRAPH_FAILED, "[Check][Param] operator %s's input %d is not linked.",
             node->GetName().c_str(), in_anchor->GetIdx());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus NodeUtilsEx::Verify(const NodePtr &node) {
  GE_CHECK_NOTNULL(node, ", Node is null for Infer Verify.");
  const bool is_unknown_graph = node->GetOwnerComputeGraph()->GetGraphUnknownFlag();
  if (is_unknown_graph) {
    return GRAPH_SUCCESS;
  }

  GE_CHK_STATUS_RET_NOLOG(IsInputsValid(node));

/*
  临时方案：
  如下代码使用原型库注册的creator构造临时op_desc，获取其input_names设置到当前op desc上有缺陷。
  1.只能恢复靠前的必选输入
  2.不能恢复dynamic input
  3.不能区分传入了哪几个可选输入，全部恢复

  且该行为归属parser, 不应该由infershape干预。但因为tf parser等前端没有正确设置input names。直接去掉会导致部分算子infershape失败。
  因此判断若input names以'__input'打头才需要刷新，作为临时方案。

  正式方案：
  tf、caffee、onnx parser要将op desc的必备字段设置完整
  */
  const auto op_desc = node->GetOpDesc();
  const bool need_update_name = (node->GetType() != FRAMEWORKOP) && NeedUpdateIOName(op_desc);
  GELOGD("Before update %s(%s) io name, input size %zu, %s, output size %zu, %s", op_desc->GetNamePtr(),
         op_desc->GetTypePtr(), op_desc->GetInputsSize(),
         IoNameToString("Input names", op_desc->GetAllInputName()).c_str(),
         op_desc->GetOutputsSize(), IoNameToString("Output names", op_desc->GetAllOutputName()).c_str());
  if (need_update_name) {
    const auto node_op = ge::OperatorFactoryImpl::CreateOperator("node_op", node->GetType());
    if (node_op.IsEmpty()) {
      GELOGW("[Verify][CheckParam] Get op from OperatorFactory failed, type: %s", node->GetType().c_str());
    } else {
      GELOGD("get op from OperatorFactory success. opType: %s", node->GetType().c_str());
      const auto temp_op_desc = ge::OpDescUtils::GetOpDescFromOperator(node_op);
      if (temp_op_desc == nullptr) {
        REPORT_INNER_ERR_MSG("E18888", "GetOpDescFromOperator failed, as return nullptr, type:%s",
                             node->GetType().c_str());
        GELOGE(GRAPH_FAILED, "[Get][OpDesc] temp op desc is null, type:%s", node->GetType().c_str());
        return GRAPH_FAILED;
      }
      if (!op_desc->UpdateInputName(temp_op_desc->GetAllInputName())) {
        GELOGW("[Verify][Update] Update input name failed");
      }
      if (!op_desc->UpdateOutputName(temp_op_desc->GetAllOutputName())) {
        GELOGW("[Verify][Update] Update output name failed");
      }
      GELOGD("After update %s(%s) io name, input size %zu, %s, output size %zu, %s", op_desc->GetNamePtr(),
             op_desc->GetTypePtr(), op_desc->GetInputsSize(),
             IoNameToString("Input names", op_desc->GetAllInputName()).c_str(),
             op_desc->GetOutputsSize(), IoNameToString("Output names", op_desc->GetAllOutputName()).c_str());
    }
    node_op.BreakConnect();
  }

  if (op_desc->CommonVerify() == GRAPH_SUCCESS) {
    Operator op = OpDescUtils::CreateOperatorFromNode(node);
    auto verify_func = op_desc->GetVerifyFunc();
    if (verify_func == nullptr) {
      verify_func = OperatorFactoryImpl::GetVerifyFunc(node->GetType());
    }
    if (verify_func != nullptr) {
      return static_cast<graphStatus>(verify_func(op));
    }
    return GRAPH_SUCCESS;
  } else {
    REPORT_INNER_ERR_MSG("E18888", "%s(%s) Verify failed.", node->GetName().c_str(), node->GetType().c_str());
    GELOGE(GRAPH_FAILED, "[Call][CommonVerify] %s(%s) failed.", node->GetName().c_str(), node->GetType().c_str());
    return GRAPH_FAILED;
  }
}
} // namespace ge
