/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter/tbe_adapter/tbe_info/tbe_single_op_info_assembler.h"
#include "adapter/tbe_adapter/tbe_info/tbe_info_assembler.h"
#include "common/fe_log.h"
#include "common/fe_type_utils.h"
#include "common/math_util.h"
#include "common/fe_op_info_common.h"
#include "common/string_utils.h"
#include "ge/ge_api_types.h"
#include "graph/debug/ge_attr_define.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "graph/ge_local_context.h"

namespace fe {
static size_t OP_ATTR_TYPE_SIZE = 2;

TbeSingleOpInfoAssembler::TbeSingleOpInfoAssembler() : tbe_info_assembler_ptr_(nullptr) {}

TbeSingleOpInfoAssembler::~TbeSingleOpInfoAssembler() {}

Status TbeSingleOpInfoAssembler::Initialize() {
  FE_MAKE_SHARED(tbe_info_assembler_ptr_ = std::make_shared<TbeInfoAssembler>(), return FAILED);
  FE_CHECK_NOTNULL(tbe_info_assembler_ptr_);
  tbe_info_assembler_ptr_->Initialize();
  return SUCCESS;
}

static Status ParseInputInfoFromOp(const ge::Node *node, const ge::OpDescPtr &op_desc_ptr,
                                   vector<OpTensorStruct> &op_input_tensor_struct) {
  FE_LOGD("Node[Name:%s,Type:%s] has %zu input.", op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(),
          node->GetOpDesc()->GetAllInputsSize());
  for (unsigned int i = 0; i < node->GetOpDesc()->GetAllInputsSize(); i++) {
    OpTensorStruct op_input_tensor_info = {i, REQUIRED, false};
    if (node->GetInDataAnchor(i) == nullptr || node->GetInDataAnchor(i)->GetPeerOutAnchor() == nullptr) {
      auto input_desc = node->GetOpDesc()->GetInputDesc(i);
      if ((input_desc.GetDataType() == ge::DT_UNDEFINED || input_desc.GetDataType() >= ge::DT_MAX) &&
          ge::GetPrimaryFormat(input_desc.GetFormat()) == ge::FORMAT_RESERVED) {
        op_input_tensor_info.op_param_type = OPTIONAL;
      }
    }
    FE_LOGD("Node[Name:%s,Type:%s]'s input index is:%u, input type is:%d.", op_desc_ptr->GetName().c_str(),
            op_desc_ptr->GetType().c_str(), op_input_tensor_info.index, op_input_tensor_info.op_param_type);
    op_input_tensor_struct.push_back(op_input_tensor_info);
  }

  FE_LOGD("The size of node[Name:%s,Type:%s] parsed input info struct is %zu", op_desc_ptr->GetName().c_str(),
          op_desc_ptr->GetType().c_str(), op_input_tensor_struct.size());
  bool has_dynamic_input_attr = op_desc_ptr->HasAttr(ge::ATTR_NAME_DYNAMIC_INPUT_START) &&
                                op_desc_ptr->HasAttr(ge::ATTR_NAME_DYNAMIC_INPUT_END);
  if (has_dynamic_input_attr) {
    vector<uint32_t> dynamic_input_start_idx;
    vector<uint32_t> dynamic_input_end_idx;
    (void)ge::AttrUtils::GetListInt(op_desc_ptr, ge::ATTR_NAME_DYNAMIC_INPUT_START, dynamic_input_start_idx);
    (void)ge::AttrUtils::GetListInt(op_desc_ptr, ge::ATTR_NAME_DYNAMIC_INPUT_END, dynamic_input_end_idx);
    if (dynamic_input_start_idx.size() != dynamic_input_end_idx.size()) {
      REPORT_FE_ERROR(
          "[SubGraphOpt][Compile][ParInInfoFromOp] Node[%s, %s]: the size of attr _dynamic_input_index is not equal.",
          op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
      return FAILED;
    }
    for (unsigned int i = 0; i < dynamic_input_start_idx.size(); i++) {
      if (dynamic_input_start_idx[i] > dynamic_input_end_idx[i]) {
        REPORT_FE_ERROR(
            "[SubGraphOpt][Compile][ParInInfoFromOp] Node[%s, %s]: the start number is more than end number",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
        return FAILED;
      }
      if (dynamic_input_end_idx[i] >= node->GetOpDesc()->GetAllInputsSize()) {
        REPORT_FE_ERROR(
            "[SubGraphOpt][Compile][ParInInfoFromOp] Node[%s, %s]: the end number is more than it's input size.",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
        return FAILED;
      }
      if (op_input_tensor_struct.size() <= i) {
        REPORT_FE_ERROR(
            "[SubGraphOpt][Compile][ParInInfoFromOp] Node[%s, %s]: the end number is more than it's input tensor size",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
        return FAILED;
      }
      for (uint32_t j = dynamic_input_start_idx[i]; j <= dynamic_input_end_idx[i]; j++) {
        op_input_tensor_struct[i].op_param_type = DYNAMIC;
        if (j == dynamic_input_end_idx[i]) {
          op_input_tensor_struct[i].is_last_dynamic_tensor = true;
        }
        FE_LOGD("Update [Name:%s,Type:%s]'s input %u to input type %d, this input is the last dynamic input:%d",
                op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), op_input_tensor_struct[i].index,
                op_input_tensor_struct[i].op_param_type, op_input_tensor_struct[i].is_last_dynamic_tensor);
      }
    }
  }
  return SUCCESS;
}

Status TbeSingleOpInfoAssembler::FeedInputInfoToSingleTbeInfo(const ge::OpDescPtr &op_desc_ptr,
                                                              const vector<OpTensorStruct> op_input_tensor_struct,
                                                              te::TbeOpInfo &tbe_op_info) {
  for (unsigned int i = 0; i < op_desc_ptr->GetAllInputsSize(); i++) {
    auto each_op_input = op_desc_ptr->MutableInputDesc(i);
    FE_CHECK_NOTNULL(each_op_input);
    te::TbeOpTensor input_tensor(each_op_input->GetName(), each_op_input->MutableShape().GetDims(), "", "");
    if (tbe_info_assembler_ptr_->SetInputTensorBaseInfo(op_desc_ptr, i, input_tensor) != SUCCESS) {
      REPORT_FE_ERROR(
          "[SubGraphOpt][Compile][FdInInfoToSingleInfo] SetInputTensorBaseInfo for node[Name:%s Type:%s] failed.",
          op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
      return FAILED;
    }
    input_tensor.SetOriginFormat(ge::TypeUtils::FormatToSerialString(each_op_input->GetOriginFormat()));
    std::vector<int64_t> dyn_ori_shape_vec;
    std::vector<int64_t> ori_shape_vec = each_op_input->GetOriginShape().GetDims();
    (void)ge::AttrUtils::GetListInt(each_op_input, kSoftSyncDynOriShape, dyn_ori_shape_vec);
    if (!dyn_ori_shape_vec.empty()) {
      ori_shape_vec = dyn_ori_shape_vec;
    }
    input_tensor.SetOriginShape(ori_shape_vec);
    FE_LOGD("Node[Name:%s Type:%s]: origin input format is [%s].", op_desc_ptr->GetName().c_str(),
            op_desc_ptr->GetType().c_str(),
            ge::TypeUtils::FormatToSerialString(each_op_input->GetOriginFormat()).c_str());
    string input_origin_shape_dims = GetShapeDims(ori_shape_vec);
    FE_LOGD("Node[Name:%s Type:%s]: origin input shape is [%s].", op_desc_ptr->GetName().c_str(),
            op_desc_ptr->GetType().c_str(), input_origin_shape_dims.c_str());
    te::TbeOpParam input;
    if (op_input_tensor_struct.size() <= i) {
      REPORT_FE_ERROR(
          "[SubGraphOpt][Compile][FdInInfoToSingleInfo] Node[%s, %s]: input desc size is more than input tensor size",
          op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
      return FAILED;
    }
    input.SetType(te::TensorType(op_input_tensor_struct[i].op_param_type));
    if (op_input_tensor_struct[i].op_param_type == REQUIRED) {
      input.SetTensor(input_tensor);
      tbe_op_info.AddInput(input);
      input.Clear();
    } else if (op_input_tensor_struct[i].op_param_type == OPTIONAL) {
      tbe_op_info.AddInput(input);
      input.Clear();
    } else {
      input.SetTensor(input_tensor);
      if (op_input_tensor_struct[i].is_last_dynamic_tensor) {
        tbe_op_info.AddInput(input);
        input.Clear();
      }
    }
  }
  return SUCCESS;
}

static Status ParseOutputInfoFromOp(const ge::Node *node, const ge::OpDescPtr &op_desc_ptr,
                                    vector<OpTensorStruct> &op_output_tensor_struct) {
  FE_LOGI("Node[Name:%s,Type:%s] has %d output.", op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(),
          node->GetOpDesc()->GetAllOutputsDescSize());
  for (unsigned int i = 0; i < node->GetOpDesc()->GetAllOutputsDescSize(); i++) {
    OpTensorStruct op_output_tensor_info = {i, REQUIRED, false};
    if (node->GetOutDataAnchor(i) == nullptr || node->GetOutDataAnchor(i)->GetPeerInDataNodesSize() == 0) {
      auto output_desc = node->GetOpDesc()->GetOutputDesc(i);
      if ((output_desc.GetDataType() == ge::DT_UNDEFINED || output_desc.GetDataType() >= ge::DT_MAX) &&
          ge::GetPrimaryFormat(output_desc.GetFormat()) == ge::FORMAT_RESERVED) {
        op_output_tensor_info.op_param_type = OPTIONAL;
      }
    }
    FE_LOGD("Node[Name:%s,Type:%s]'s output index is %u, param type is %d.", op_desc_ptr->GetName().c_str(),
            op_desc_ptr->GetType().c_str(), op_output_tensor_info.index, op_output_tensor_info.op_param_type);
    op_output_tensor_struct.push_back(op_output_tensor_info);
  }
  bool has_dynamic_output_attr = op_desc_ptr->HasAttr("_dynamic_output_index_start") &&
                                 op_desc_ptr->HasAttr("_dynamic_output_index_end");
  if (has_dynamic_output_attr) {
    vector<uint32_t> dynamic_output_start_idx;
    vector<uint32_t> dynamic_output_end_idx;
    (void)ge::AttrUtils::GetListInt(op_desc_ptr, "_dynamic_output_index_start", dynamic_output_start_idx);
    (void)ge::AttrUtils::GetListInt(op_desc_ptr, "_dynamic_output_index_end", dynamic_output_end_idx);
    if (dynamic_output_start_idx.size() != dynamic_output_end_idx.size()) {
      REPORT_FE_ERROR(
          "[SubGraphOpt][Compile][ParOutInfoFromOp] Node[%s, %s]: the size of attr _dynamic_output_index is not equal.",
          op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
      return FAILED;
    }
    for (unsigned int i = 0; i < dynamic_output_start_idx.size(); i++) {
      if (dynamic_output_start_idx[i] > dynamic_output_end_idx[i]) {
        REPORT_FE_ERROR(
            "[SubGraphOpt][Compile][ParOutInfoFromOp] Node[%s, %s]: the start number is more than end number.",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
        return FAILED;
      }
      if (dynamic_output_end_idx[i] >= node->GetOpDesc()->GetAllOutputsDescSize()) {
        REPORT_FE_ERROR(
            "[SubGraphOpt][Compile][ParOutInfoFromOp] Node[%s, %s]: the end number is more than it's output size.",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
        return FAILED;
      }
      if (op_output_tensor_struct.size() <= i) {
        REPORT_FE_ERROR(
            "[SubGraphOpt][Compile][ParOutInfoFromOp] Node[%s, %s]: The end number is more than output tensor size",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
        return FAILED;
      }
      for (uint32_t j = dynamic_output_start_idx[i]; j <= dynamic_output_end_idx[i]; j++) {
        op_output_tensor_struct[i].op_param_type = DYNAMIC;
        if (j == dynamic_output_end_idx[i]) {
          op_output_tensor_struct[i].is_last_dynamic_tensor = true;
        }
        FE_LOGD("Update node[Name:%s,Type:%s]'s output index %u to param type %d, is_last_dynamic_input: %d.",
                op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), op_output_tensor_struct[i].index,
                op_output_tensor_struct[i].op_param_type, op_output_tensor_struct[i].is_last_dynamic_tensor);
      }
    }
  }
  return SUCCESS;
}

Status TbeSingleOpInfoAssembler::FeedOutputInfoToSingleTbeInfo(const ge::OpDescPtr &op_desc_ptr,
                                                               const vector<OpTensorStruct> op_output_tensor_struct,
                                                               te::TbeOpInfo &tbe_op_info) {
  for (unsigned int i = 0; i < op_desc_ptr->GetAllOutputsDesc().size(); i++) {
    auto each_op_output = op_desc_ptr->GetAllOutputsDesc().at(i);
    te::TbeOpTensor output_tensor(each_op_output.GetName(), each_op_output.MutableShape().GetDims(), "", "");
    if (tbe_info_assembler_ptr_->SetInputTensorBaseInfo(op_desc_ptr, i, output_tensor) != SUCCESS) {
      REPORT_FE_ERROR(
          "[SubGraphOpt][Compile][FdOutInfoToSingleInfo] SetInputTensorBaseInfo for node[Name:%s Type:%s] failed.",
          op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
      return FAILED;
    }
    output_tensor.SetOriginFormat(ge::TypeUtils::FormatToSerialString(each_op_output.GetOriginFormat()));
    std::vector<int64_t> dyn_ori_shape_vec;
    std::vector<int64_t> ori_shape_vec = each_op_output.GetOriginShape().GetDims();
    (void)ge::AttrUtils::GetListInt(each_op_output, kSoftSyncDynOriShape, dyn_ori_shape_vec);
    if (!dyn_ori_shape_vec.empty()) {
      ori_shape_vec = dyn_ori_shape_vec;
    }
    output_tensor.SetOriginShape(ori_shape_vec);
    FE_LOGD("Node[Name:%s Type:%s]: origin output format is [%s].", op_desc_ptr->GetName().c_str(),
            op_desc_ptr->GetType().c_str(),
            ge::TypeUtils::FormatToSerialString(each_op_output.GetOriginFormat()).c_str());
    string output_origin_shape_dims = GetShapeDims(ori_shape_vec);
    FE_LOGD("Node[Name:%s Type:%s]: origin output shape is [%s].", op_desc_ptr->GetName().c_str(),
            op_desc_ptr->GetType().c_str(), output_origin_shape_dims.c_str());
    te::TbeOpParam output;
    if (op_output_tensor_struct.size() <= i) {
      REPORT_FE_ERROR(
          "[SubGraphOpt][Compile][FdOutInfoToSingleInfo] Node[%s, %s]: output desc size is more than tensor size",
          op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
      return FAILED;
    }
    output.SetType(te::TensorType(op_output_tensor_struct[i].op_param_type));
    if (op_output_tensor_struct[i].op_param_type == REQUIRED) {
      output.SetTensor(output_tensor);
      tbe_op_info.AddOutput(output);
      output.Clear();
    } else if (op_output_tensor_struct[i].op_param_type == OPTIONAL) {
      tbe_op_info.AddOutput(output);
      output.Clear();
    } else {
      output.SetTensor(output_tensor);
      if (op_output_tensor_struct[i].is_last_dynamic_tensor) {
        tbe_op_info.AddOutput(output);
        output.Clear();
      }
    }
  }
  return SUCCESS;
}

static Status parse_attr_list_from_op(const ge::OpDescPtr &op_desc_ptr,
                                      std::map<std::string, ge::GeAttrValue::ValueType> &op_attr_map) {
  std::string op_attr_list;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, ge::ATTR_NAME_UNREGST_ATTRLIST, op_attr_list);
  std::vector<std::string> each_op_attr_list = StringUtils::Split(op_attr_list, ';');
  for (const std::string &each_op_attr : each_op_attr_list) {
    FE_LOGD("Node[%s]'s each_op_attr_list is %s.", op_desc_ptr->GetName().c_str(), each_op_attr.c_str());
    std::vector<std::string> each_op_attr_type = StringUtils::Split(each_op_attr, ':');
    if (each_op_attr_type.size() < OP_ATTR_TYPE_SIZE) {
      continue;
    }
    auto iter = ATTR_STRING_TO_VALUETYPE_MAP.find(each_op_attr_type[1]);
    if (iter == ATTR_STRING_TO_VALUETYPE_MAP.end()) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][ParAttrFromOp] Not support attr[%s] in node[%s].",
                      each_op_attr_type[1].c_str(), op_desc_ptr->GetName().c_str());
      return FAILED;
    }
    ge::GeAttrValue::ValueType op_attr_value = iter->second;
    op_attr_map.emplace(each_op_attr_type[0], op_attr_value);
  }
  return SUCCESS;
}

Status TbeSingleOpInfoAssembler::FeedAttrsToSingleTbeOpInfo(const ge::OpDescPtr &op_desc_ptr,
                                                            te::TbeOpInfo &tbe_op_info) const {
  std::map<std::string, ge::GeAttrValue::ValueType> op_attr_map;
  if (parse_attr_list_from_op(op_desc_ptr, op_attr_map) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][FdAttrsToOPInfo] parse _unregist_attrlist from Op %s failed.",
                    op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  for (auto op_attr : op_attr_map) {
    te::TbeAttrValue attr_value;
    FE_LOGD("Start to parse attr %d from node[Name:%s Type:%s]", op_attr.second, op_desc_ptr->GetName().c_str(),
            op_desc_ptr->GetType().c_str());
    auto func = k_attr_get_funcs.find(op_attr.second);
    if (func == k_attr_get_funcs.end()) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][FdAttrsToOPInfo] Unknown attr %s value type in op %s.",
                      op_attr.first.c_str(), op_desc_ptr->GetName().c_str());
      return FAILED;
    } else {
      if (func->second((*op_desc_ptr.get()), op_attr.first, attr_value, nullptr) != SUCCESS) {
        return FAILED;
      }
    }
    tbe_op_info.AddAttrValue(attr_value);
  }
  return SUCCESS;
}

Status TbeSingleOpInfoAssembler::FeedFlagInt64ToTbeOpInfo(const ge::Node *node, te::TbeOpInfo &op_info) const {
  auto op_desc = node->GetOpDesc();
  bool flag = false;
  if (UnknownShapeUtils::IsUnknownShapeOp(*op_desc)) {
    FE_LOGD("[node: %s] is UnKnownShape.", node->GetName().c_str());
    op_info.SetFlagUseInt64(true);
  } else {
    (void)tbe_info_assembler_ptr_->JudgeShapeToSetFlag(op_desc, true, op_info, flag);
    (void)tbe_info_assembler_ptr_->JudgeShapeToSetFlag(op_desc, false, op_info, flag);
  }
  FE_LOGD("[node: %s] TbeSingleOpInfoAssembler FlagUseInt64: %d.", node->GetName().c_str(), op_info.GetFlagUseInt64());
  return SUCCESS;
}

Status TbeSingleOpInfoAssembler::AssembleSingleTbeInfo(ge::Node *node, te::TbeOpInfo &tbe_op_info,
                                                       const string &engine_name) {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  std::string opFileName;
  if (!ge::AttrUtils::GetStr(op_desc_ptr, ge::ATTR_NAME_UNREGST_OPPATH, opFileName)) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][AssmSingleTbeInfo] Failed to get _unregst_oppath of node[%s]",
                    op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  FE_LOGD("Op[%s, %s]: tbe_op_info.opFileName=[%s].", node->GetNamePtr(), node->GetTypePtr(), opFileName.c_str());
  vector<OpTensorStruct> op_input_tensor_struct;
  if (ParseInputInfoFromOp(node, op_desc_ptr, op_input_tensor_struct) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][AssmSingleTbeInfo] Failed to ParseInputInfoFromOp of node[%s].",
                    op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  if (FeedInputInfoToSingleTbeInfo(op_desc_ptr, op_input_tensor_struct, tbe_op_info) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][AssmSingleTbeInfo] Failed to feed input info to single TBE info for node [%s].",
                    op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  vector<OpTensorStruct> op_output_tensor_struct;
  if (ParseOutputInfoFromOp(node, op_desc_ptr, op_output_tensor_struct) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][AssmSingleTbeInfo] Failed to parse output info from op for node [%s].",
                    op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  if (FeedOutputInfoToSingleTbeInfo(op_desc_ptr, op_output_tensor_struct, tbe_op_info) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][AssmSingleTbeInfo] Failed to execute FeedOutputInfoToSingleTbeInfo for node [%s].",
                    op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  if (op_desc_ptr->HasAttr(ge::ATTR_NAME_UNREGST_ATTRLIST)) {
    if (FeedAttrsToSingleTbeOpInfo(op_desc_ptr, tbe_op_info) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][AssmSingleTbeInfo] Failed to FeedAttrsToSingleTbeOpInfo of node[%s].",
                      op_desc_ptr->GetName().c_str());
      return FAILED;
    }
    if (FeedFlagInt64ToTbeOpInfo(node, tbe_op_info) != SUCCESS) {
      REPORT_FE_ERROR(
          "[SubGraphOpt][Compile][AssmSingleTbeInfo] AssembleSingleTbeInfo FeedFlagInt64ToTbeOpInfo failed.");
      return FAILED;
    }
  }

  // feed all options to TbeOpInfo
  map<std::string, std::string> options =
      tbe_info_assembler_ptr_->GetAllOptionsForTBE(*op_desc_ptr, engine_name, nullptr);
  tbe_op_info.SetOptions(options);
  tbe_info_assembler_ptr_->SetOpImplMode(engine_name, op_desc_ptr, tbe_op_info);
  tbe_info_assembler_ptr_->SetExtraParams(*op_desc_ptr, tbe_op_info);
  tbe_info_assembler_ptr_->SetSingleOpScene(node, tbe_op_info);
  tbe_info_assembler_ptr_->SetOpImplSwitch(op_desc_ptr, tbe_op_info);
  return SUCCESS;
}
}  // namespace fe
