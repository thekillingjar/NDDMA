/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "heavy_format_propagation.h"
#include "common/configuration.h"
#include "common/format/axis_name_util.h"
#include "common/fe_type_utils.h"
#include "common/graph/fe_graph_utils.h"
#include "common/fe_op_info_common.h"
#include "common/unknown_shape_util.h"
#include "common/platform_utils.h"
#include "graph/utils/node_utils.h"
#include "expand_dimension.h"
#include "ops_store/ops_kernel_manager.h"
#include "graph_optimizer/dynamic_shape_optimizer/fuzzy_compiler/input_node_generalize.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/refresh_cube_c0_fusion_pass.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"

namespace fe {
namespace {
const std::string kStageChkSpfSubgphData = "[GraphOptJdgInst][FmtPropagate][ChkSpfSubgphData]";
const std::string kStageGetNextNdInfoFwd = "[GraphOptJdgInst][FmtPropagate][GetNextNdInfoFwd]";
inline bool IsWeight(const ge::OpDescPtr &op_desc_ptr) {
  if (op_desc_ptr == nullptr) {
    return false;
  }
  return kWeightTypes.count(op_desc_ptr->GetType());
}

inline bool HasHeavyOpAttr(const ge::OpDescPtr &op_desc_ptr) {
  bool is_heavy_op = false;
  return ge::AttrUtils::GetBool(op_desc_ptr, kAttrNameIsHeavyOp, is_heavy_op) && is_heavy_op;
}

inline bool IsHeavyFormat(const ge::Format &format) {
  return std::find(FE_HEAVY_FORMAT_VECTOR.begin(), FE_HEAVY_FORMAT_VECTOR.end(), format) !=
         FE_HEAVY_FORMAT_VECTOR.end();
}

inline bool IsOpKernelHeavyOp(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr) {
  if (op_kernel_info_ptr == nullptr) {
    return false;
  }
  return IsAllowNzMatmul(node) || op_kernel_info_ptr->IsHeavyOp();
}

inline void SetFormatC0ValOnTensorDesc(ge::GeTensorDescPtr &tensor_desc) {
  if (tensor_desc == nullptr) {
    return;
  }
  if (!ge::HasC0Format(static_cast<int32_t>(tensor_desc->GetFormat())) && IsHeavyFormat(tensor_desc->GetFormat())) {
    int32_t c0_bit = GetC0BitByDataType(tensor_desc->GetDataType());
    ge::Format new_format = static_cast<ge::Format>(ge::GetFormatFromC0(tensor_desc->GetFormat(), c0_bit));
    tensor_desc->SetFormat(new_format);
  }
}

bool IsWeightOrData(const ge::OpDescPtr &op_desc_ptr) {
  // sub graph DATA is not DATA op
  bool is_subgraph_data = FeGraphUtils::IsSubGraphData(op_desc_ptr);
  if (!is_subgraph_data) {
    return IsWeight(op_desc_ptr) || IsRootGraphData(op_desc_ptr->GetType());
  }

  return false;
}

Status CheckScalarOrResult(ge::ConstGeTensorDescPtr current_tensor, const TensorInfoPtr &tensor_info_ptr) {
  auto &shape = current_tensor->GetShape();
  if (shape.IsScalar() && tensor_info_ptr->format_selection_type == FormatSelectionType::FORMAT_PENETRATION) {
    FE_LOGW("Stopped traversing from this scalar weight %s!", tensor_info_ptr->op_desc_ptr->GetName().c_str());
    return GRAPH_OPTIMIZER_STOP_TRAVERSING_SCALAR_WEIGHT;
  }
  if (IsScalarInputOrOutput(shape, current_tensor->GetFormat())) {
    FE_LOGW("Stop traversing this Scalar tensor %s!", tensor_info_ptr->op_desc_ptr->GetName().c_str());
    return GRAPH_OPTIMIZER_STOP_TRAVERSING_SCALAR_TENSOR;
  }
  return SUCCESS;
}

Status GetCurrentTensor(const ge::NodePtr &current_node, const TensorInfoPtr &tensor_info_ptr,
                        ge::ConstGeTensorDescPtr &current_tensor) {
  FE_CHECK_NOTNULL(current_node);
  FE_CHECK_NOTNULL(tensor_info_ptr);
  if (tensor_info_ptr->is_input) {
    if (static_cast<size_t>(tensor_info_ptr->anchor_index) >= current_node->GetOpDesc()->GetAllInputsDescPtr().size()) {
      FE_LOGW("AnchorIndex %u is greater than %zu in op desc %s | %s", tensor_info_ptr->anchor_index,
              current_node->GetOpDesc()->GetAllInputsDescPtr().size(), current_node->GetName().c_str(),
              current_node->GetType().c_str());
      return FAILED;
    }
    current_tensor = current_node->GetOpDesc()->GetInputDescPtr(tensor_info_ptr->anchor_index);
  } else {
    if (static_cast<size_t>(tensor_info_ptr->anchor_index) >=
        current_node->GetOpDesc()->GetAllOutputsDescPtr().size()) {
      FE_LOGW("AnchorIndex %u is greater than %zu in op desc %s | %s", tensor_info_ptr->anchor_index,
              current_node->GetOpDesc()->GetAllOutputsDescPtr().size(), current_node->GetName().c_str(),
              current_node->GetType().c_str());
      return FAILED;
    }
    current_tensor = current_node->GetOpDesc()->GetOutputDescPtr(tensor_info_ptr->anchor_index);
  }

  if (current_tensor == nullptr) {
    return FAILED;
  }
  return SUCCESS;
}

ge::Format GetNewHeavyFormat(const TensorInfoPtr &tensor_info_ptr, int64_t &group) {
  ge::Format new_heavy_format = tensor_info_ptr->heavy_format;
  // if the heavy_format is fraz_g, try to set _fe_group attribute on tensor
  if (std::find(FE_GROUP_RELA_FORMAT_VECTOR.begin(), FE_GROUP_RELA_FORMAT_VECTOR.end(),
                tensor_info_ptr->heavy_format) != FE_GROUP_RELA_FORMAT_VECTOR.end()) {
    new_heavy_format =
        static_cast<ge::Format>(ge::GetFormatFromSub(tensor_info_ptr->heavy_format, tensor_info_ptr->sub_format));
    group = tensor_info_ptr->sub_format;
    if (group < GROUPS_DEFAULT_VALUE) {
      group = GROUPS_DEFAULT_VALUE;
    }
  }
  new_heavy_format =
      static_cast<ge::Format>(ge::GetFormatFromC0(new_heavy_format, GetC0BitValFromC0(tensor_info_ptr->c0_format)));

  return new_heavy_format;
}

Status GetCurrentTensor(const TensorInfoPtr &tensor_info_ptr, ge::GeTensorDescPtr &current_tensor) {
  if (tensor_info_ptr->is_input) {
    if (static_cast<size_t>(tensor_info_ptr->anchor_index) >=
        tensor_info_ptr->op_desc_ptr->GetAllInputsDescPtr().size()) {
      FE_LOGW("AnchorIndex %u is larger than %zu of op description %s.", tensor_info_ptr->anchor_index,
              tensor_info_ptr->op_desc_ptr->GetAllInputsDescPtr().size(),
              tensor_info_ptr->op_desc_ptr->GetName().c_str());
      return FAILED;
    }
    current_tensor = tensor_info_ptr->op_desc_ptr->MutableInputDesc(tensor_info_ptr->anchor_index);
  } else {
    if (static_cast<size_t>(tensor_info_ptr->anchor_index) >=
        tensor_info_ptr->op_desc_ptr->GetAllOutputsDescPtr().size()) {
      FE_LOGW("AnchorIndex %u is larger than %zu of op description %s.", tensor_info_ptr->anchor_index,
              tensor_info_ptr->op_desc_ptr->GetAllOutputsDescPtr().size(),
              tensor_info_ptr->op_desc_ptr->GetName().c_str());
      return FAILED;
    }
    current_tensor = tensor_info_ptr->op_desc_ptr->MutableOutputDesc(tensor_info_ptr->anchor_index);
  }
  FE_CHECK_NOTNULL(current_tensor);
  return SUCCESS;
}

bool IsNextNodeVisitedOrNull(const ge::NodePtr &last_node, const ge::NodePtr &next_node) {
  if (next_node == nullptr) {
    return true;
  }
  if (last_node != nullptr && next_node != nullptr &&
      (next_node->GetOwnerComputeGraph()->GetName() == last_node->GetOwnerComputeGraph()->GetName()) &&
      (next_node->GetOpDesc()->GetId() == last_node->GetOpDesc()->GetId())) {
    /* This means we are travering from this last_node, and we won't
     * traverse forwards to this last_node again. */
    FE_LOGD("This node %s has been visited; the last node was %s.", next_node->GetName().c_str(),
            last_node->GetName().c_str());
    return true;
  }
  return false;
}

/* Only the reshape inserted by FE will be penetrable. */
bool IsFEInsertedReshape(const ge::OpDescPtr &op_desc_ptr) {
  bool is_inserted_by_ge = false;
  if (ge::AttrUtils::GetBool(op_desc_ptr, ge::ATTR_INSERTED_BY_GE, is_inserted_by_ge)) {
    /* If the dst trans node is inserted by ge, we won't insert any
     * other trans nodes before it. */
    if (is_inserted_by_ge) {
      return (op_desc_ptr->GetType() == RESHAPE || op_desc_ptr->GetType() == UNSQUEEZE_V2 ||
              op_desc_ptr->GetType() == SQUEEZE_V2) &&
             is_inserted_by_ge;
    }
  }
  return false;
}

/* Our system do not support the TransData of data type int64.
 * If the cast is set as format agnostic, in some case, we must
 * use the int64 TransData which is not supported. */
void CorrectFmtAgnosticType(const ge::OpDescPtr &op_desc_ptr) {
  if (IsDtypeSensitiveOp(op_desc_ptr->GetType()) &&
      ((op_desc_ptr->GetInputDescPtr(0) != nullptr &&
        op_desc_ptr->GetInputDescPtr(0)->GetDataType() == ge::DT_INT64) ||
       (op_desc_ptr->GetOutputDescPtr(0) != nullptr &&
        op_desc_ptr->GetOutputDescPtr(0)->GetDataType() == ge::DT_INT64))) {
    FE_LOGI("Cast operation with %s contains int64 input or output is not allowed as a format-agnostic operator.", op_desc_ptr->GetName().c_str());
    (void)ge::AttrUtils::SetInt(op_desc_ptr, FORMAT_AGNOSTIC, static_cast<int64_t>(
      FormatSelectionType::FORMAT_DEPENDS_ON_OP_KERNEL_INFO));
  }
}
}  // namespace

HeavyFormatPropagation::HeavyFormatPropagation(const std::string &engine_name, RefRelationsPtr reflection_builder_ptr)
    : engine_name_(engine_name), reflection_builder_ptr_(reflection_builder_ptr), format_dtype_querier_ptr_(nullptr) {}

HeavyFormatPropagation::~HeavyFormatPropagation() {}

Status HeavyFormatPropagation::Initialize() {
  FE_MAKE_SHARED(format_dtype_querier_ptr_ = std::make_shared<FormatDtypeQuerier>(engine_name_), return FAILED);
  FE_MAKE_SHARED(format_dtype_setter_ptr_ = std::make_shared<FormatDtypeSetter>(engine_name_), return FAILED);
  FE_MAKE_SHARED(supportformats_updater_ptr_ = std::make_shared<HeavyFormatSupportFormatsUpdater>(
                     format_dtype_querier_ptr_, format_dtype_setter_ptr_),
                 return FAILED);

  FE_MAKE_SHARED(heavy_format_selector_ptr_ = std::make_shared<HeavyFormatSelector>(format_dtype_querier_ptr_),
                 return FAILED);
  if (heavy_format_selector_ptr_ == nullptr) {
    return FAILED;
  }
  return heavy_format_selector_ptr_->Initalize();
}

bool HeavyFormatPropagation::IsPropagateFromNode(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr) {
  bool is_heavy_op = IsOpKernelHeavyOp(node, op_kernel_info_ptr);
  if (!is_heavy_op) {
    // mind spore may set is_heavy_op attr
    is_heavy_op = HasHeavyOpAttr(node->GetOpDesc());
  }

  bool input_is_fp32 = false;
  bool output_is_fp32 = false;
  auto input_desc = RefreshCubeC0FusionPass::GetCubeInputDesc(node->GetOpDesc());
  if (input_desc != nullptr) {
    input_is_fp32 = input_desc->GetDataType() == ge::DT_FLOAT;
  }
  auto output_desc = node->GetOpDesc()->MutableOutputDesc(0);
  if (output_desc != nullptr) {
    output_is_fp32 = output_desc->GetDataType() == ge::DT_FLOAT;
  }
  // if cube support high precision, cube_op's c0 is 8 when input&output are fp32
  const std::unordered_set<string> &cube_op_list = Configuration::Instance(AI_CORE_NAME).GetFp16OpTypeList();
  bool is_cube_input_and_output_fp32 = cube_op_list.count(node->GetOpDesc()->GetType()) > 0 &&
                                       (input_is_fp32 && output_is_fp32) &&
                                       PlatformUtils::Instance().IsEnableCubeHighPrecision();
  return is_heavy_op && !is_cube_input_and_output_fp32;
}

bool HeavyFormatPropagation::IsConstNodeInSubGraph(const ge::NodePtr &node_ptr) const {
  if (node_ptr == nullptr) {
    return false;
  }
  FE_LOGD("IsConstNodeInSubGraph begin: [%s, %s].", node_ptr->GetName().c_str(), node_ptr->GetType().c_str());
  if (node_ptr->GetType() != DATA) {
    return false;
  }

  ge::NodePtr parent_node_ptr = ge::NodeUtils::GetParentInput(node_ptr);
  if (parent_node_ptr == nullptr) {
    FE_LOGD("Parent node is nullptr.");
    return false;
  }

  FE_LOGD("IsConstNodeInSubGraph: Parent node is [%s, %s].", parent_node_ptr->GetName().c_str(),
          parent_node_ptr->GetType().c_str());
  return parent_node_ptr->GetType() == CONSTANTOP || parent_node_ptr->GetType() == CONSTANT;
}

Status HeavyFormatPropagation::SetInferFormat(const TensorInfoPtr &tensor_info_ptr) {
  ge::GeTensorDescPtr current_tensor;
  if (tensor_info_ptr->is_input) {
    current_tensor = tensor_info_ptr->op_desc_ptr->MutableInputDesc(tensor_info_ptr->anchor_index);
  } else {
    current_tensor = tensor_info_ptr->op_desc_ptr->MutableOutputDesc(tensor_info_ptr->anchor_index);
  }
  FE_CHECK_NOTNULL(current_tensor);
  int64_t infer_format = -1;
  /* If the output tensor has been visited, we will still return success.
   * Because one output could give to multiple input. */
  if (ge::AttrUtils::GetInt(current_tensor, INFER_FORMAT, infer_format)) {
    FE_LOGD("%s %d of Op %s has been propagated! Its inference format is %ld.",
            IS_INPUT_TO_STRING(tensor_info_ptr->is_input), tensor_info_ptr->anchor_index,
            tensor_info_ptr->op_desc_ptr->GetName().c_str(), infer_format);
    FE_LOGD("Current heavy_format is %s, sub_format is %d.", FormatToStr(tensor_info_ptr->heavy_format).c_str(),
            tensor_info_ptr->sub_format);
    return GRAPH_OPTIMIZER_STOP_TRAVERSING_OTHER_ANCHORS;
  }
  /* For normal nodes we will always set the INFER_FORMAT attribute, because each node can only be visited once.
   * For sub-graph net-output, we will not set this attribute, because we allow this node being propagated
   * repeatedly. And sub-graph data will not go into this function. */
  if (!tensor_info_ptr->is_sub_graph_data_or_nt_opt || !tensor_info_ptr->is_forward) {
    (void)ge::AttrUtils::SetInt(current_tensor, INFER_FORMAT, tensor_info_ptr->heavy_format);
  }

  if (!IsHeavyFormat(tensor_info_ptr->heavy_format)) {
    FE_LOGD("Format %s is not considered a heavy format; stopping at %s[%d] on node[%s].",
            FormatToStr(tensor_info_ptr->heavy_format).c_str(), IS_INPUT_TO_STRING(tensor_info_ptr->is_input),
            tensor_info_ptr->anchor_index, tensor_info_ptr->op_desc_ptr->GetName().c_str());
    return GRAPH_OPTIMIZER_NOT_HEAVY_FORMAT;
  }
  FE_LOGD("format %s is a heavy format, with sub_format %d. We set the infer format for %s %d of the node: %s.",
          FormatToStr(tensor_info_ptr->heavy_format).c_str(), tensor_info_ptr->sub_format,
          IS_INPUT_TO_STRING(tensor_info_ptr->is_input), tensor_info_ptr->anchor_index,
          tensor_info_ptr->op_desc_ptr->GetName().c_str());
  Status ret = CheckScalarOrResult(current_tensor, tensor_info_ptr);
  return ret;
}

Status HeavyFormatPropagation::GetFormatAndDtypeFromOpKernel(const ge::NodePtr &current_node,
                                                             const TensorInfoPtr &tensor_info_ptr,
                                                             const OpKernelInfoPtr &op_kernel_info_ptr) const {
  if (op_kernel_info_ptr == nullptr) {
    FE_LOGW("opKernelInfoPtr of node %s is nullptr!", current_node->GetName().c_str());
    return FAILED;
  }

  InputOrOutputInfoPtr input_or_output_info;
  if (GetTensorKernelInfo(current_node, tensor_info_ptr, op_kernel_info_ptr, input_or_output_info) != SUCCESS) {
    FE_LOGW("Failed to get tensor kernel info for node %s.", current_node->GetName().c_str());
    return FAILED;
  }

  ge::ConstGeTensorDescPtr current_tensor;
  if (GetCurrentTensor(current_node, tensor_info_ptr, current_tensor) != SUCCESS) {
    return FAILED;
  }
  vector<ge::Format> format_vec;
  if (format_dtype_querier_ptr_->GetSupportFormats(op_kernel_info_ptr, input_or_output_info,
                                                   current_node, format_vec) != SUCCESS) {
    FE_LOGW("Failed to retrieve supported formats, node: [%s].", current_node->GetName().c_str());
    return FAILED;
  }
  if (static_cast<size_t>(tensor_info_ptr->format_index) >= format_vec.size()) {
    FE_LOGW("formatIndex %d exceeds the size %zu for format kernel %s | %s", tensor_info_ptr->anchor_index,
            current_node->GetOpDesc()->GetAllOutputsDescPtr().size(), current_node->GetName().c_str(),
            current_node->GetType().c_str());
    return FAILED;
  }
  tensor_info_ptr->heavy_format = format_vec.at(tensor_info_ptr->format_index);
  if (tensor_info_ptr->heavy_format == ge::FORMAT_ND) {
    tensor_info_ptr->heavy_format = static_cast<ge::Format>(ge::GetPrimaryFormat(current_tensor->GetFormat()));
  }
  vector<ge::DataType> data_type_vec;
  if (format_dtype_querier_ptr_->GetSupportDataTypes(op_kernel_info_ptr, input_or_output_info,
                                                     current_node, data_type_vec) != SUCCESS) {
    FE_LOGW("Failed to retrieve supported data types, returning FAILED.");
    return FAILED;
  }
  if (static_cast<size_t>(tensor_info_ptr->format_index) >= data_type_vec.size()) {
    FE_LOGW("FormatIndex %d exceeds data type size %zu: %s | %s", tensor_info_ptr->format_index,
            data_type_vec.size(), current_node->GetName().c_str(),
            current_node->GetType().c_str());
    return FAILED;
  }
  tensor_info_ptr->data_type = data_type_vec.at(tensor_info_ptr->format_index);
  tensor_info_ptr->op_kernel_tensor_info = input_or_output_info;
  FE_LOGD("Heavy format of %s[%d] on node[%s] is %s. The sub_format is %d and the data_type is %s.",
          IS_INPUT_TO_STRING(tensor_info_ptr->is_input), tensor_info_ptr->anchor_index, current_node->GetName().c_str(),
          ge::TypeUtils::FormatToSerialString(tensor_info_ptr->heavy_format).c_str(), tensor_info_ptr->sub_format,
          ge::TypeUtils::DataTypeToSerialString(tensor_info_ptr->data_type).c_str());
  return SUCCESS;
}

Status HeavyFormatPropagation::CheckSpecificSubGraphDataOrNetoutput(
    const TensorInfoPtr &tensor_info_ptr, bool &is_sub_graph_data_or_net_output,
    std::unordered_set<ge::RefCell, ge::RefCellHash> &reflections) {
  ge::RefCell key(tensor_info_ptr->node_ptr->GetName(), tensor_info_ptr->node_ptr,
                  tensor_info_ptr->is_input ? ge::NODE_IN : ge::NODE_OUT, tensor_info_ptr->anchor_index);
  is_sub_graph_data_or_net_output = FeGraphUtils::IsSubGraphDataOrNetOutput(tensor_info_ptr->op_desc_ptr);
  if (is_sub_graph_data_or_net_output) {
    FE_LOGD("Begin to get relations, node: %s, is_input: %d, anchor_index: %d.",
            tensor_info_ptr->node_ptr->GetName().c_str(),
            tensor_info_ptr->is_input, tensor_info_ptr->anchor_index);

    auto status = reflection_builder_ptr_->LookUpRefRelations(key, reflections);
    if (status != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("%s Node[%s]: failed to look up relations of input %d edge", kStageChkSpfSubgphData.c_str(),
                      tensor_info_ptr->node_ptr->GetName().c_str(), key.in_out_idx);
      return FAILED;
    }

    FE_LOGD("Get relations result, node:%s, relations size:%zu.", tensor_info_ptr->node_ptr->GetName().c_str(),
            reflections.size());

    if (!FeGraphUtils::CheckRelatedEdgesOriginShape(reflections)) {
      return GRAPH_OPTIMIZER_STOP_TRAVERSING_SUBGRAPH;
    }
  }
  return CONTINUE_TO_SET_FORMAT;
}

void HeavyFormatPropagation::FindSameNameVariableNodes(const ge::NodePtr &node_ptr,
                                                       std::vector<ge::NodePtr> &var_nodes) {
  if (node_ptr == nullptr) {
    return;
  }

  if (node_ptr->GetType() != VARIABLE) {
    return;
  }
  ge::RefCell key(node_ptr->GetName(), node_ptr, ge::NODE_OUT, 0);
  std::unordered_set<ge::RefCell, ge::RefCellHash> reflections;
  if (reflection_builder_ptr_->LookUpRefRelations(key, reflections) != ge::GRAPH_SUCCESS || reflections.empty()) {
    FE_LOGD("[GraphOptJdgInst][FmtPropagate][FindSameNmVarNd] failed to look up the relationship of the same name node [%s].",
           node_ptr->GetName().c_str());
    return;
  }

  for (auto &cell : reflections) {
    if (node_ptr != cell.node) {
      var_nodes.push_back(cell.node);
    }
  }
}

Status HeavyFormatPropagation::SetNewShape(const TensorInfoPtr &tensor_info_ptr,
                                           const OpKernelInfoPtr &op_kernel_info_ptr,
                                           const ge::GeTensorDescPtr &current_tensor, ge::GeShape &original_shape,
                                           ge::GeShape &new_shape) {
  auto op_desc = tensor_info_ptr->op_desc_ptr;
  auto op_name = op_desc->GetName();
  auto op_type = op_desc->GetType();
  int64_t group = GROUPS_DEFAULT_VALUE;
  ge::Format new_heavy_format = GetNewHeavyFormat(tensor_info_ptr, group);
  FE_LOGD("Op[%s, %s]: new_heavy_format is %s, the group is %ld for the %s with ID [%d].",
          op_name.c_str(), op_type.c_str(), FormatToStr(new_heavy_format).c_str(), group,
          IS_INPUT_TO_STRING(tensor_info_ptr->is_input), tensor_info_ptr->anchor_index);
  int32_t c0_bit_val = GetC0BitByDataType(current_tensor->GetDataType());
  new_heavy_format = static_cast<ge::Format>(ge::GetFormatFromC0(new_heavy_format, c0_bit_val));
  if (original_shape.GetDimNum() == 0) {
    new_shape = original_shape;
    current_tensor->SetFormat(new_heavy_format);
    FE_LOGD("Op[%s, %s]: -2 dims, only set new_heavy_format to %s for the %s [%d].",
            op_name.c_str(), op_type.c_str(), FormatToStr(new_heavy_format).c_str(),
            IS_INPUT_TO_STRING(tensor_info_ptr->is_input), tensor_info_ptr->anchor_index);
    return SUCCESS;
  }

  std::string reshape_type = GetPropagationReshapeType(tensor_info_ptr, op_kernel_info_ptr, current_tensor);
  FE_LOGD("Op[name:%s] reshape type is %s.", op_name.c_str(), reshape_type.c_str());
  ExpandDimension(current_tensor->GetOriginFormat(), tensor_info_ptr->heavy_format, reshape_type, original_shape,
                  new_shape, current_tensor);

  /* Update the propagated reshape type. Because when reshape type pass through a node, it may change:
   * 1. If the node itself has a reshape type, we should use that reshape type instead of the propagated reshape
   * type from the last node. But when the node does not need dimension padding, we just clear the reshape type.
   *
   * 2. If the node itself does not have a reshape type and it is not a format agnostic node, we
   * should clear the reshape type.
   */
  ge::GeShape origin_shape_before_reshape = current_tensor->GetOriginShape();
  if (original_shape.GetDims() == new_shape.GetDims()) {
    tensor_info_ptr->propagation_info.reshape_type = "";
  } else {
    tensor_info_ptr->propagation_info.reshape_type = reshape_type;
    origin_shape_before_reshape = new_shape;
  }
  original_shape = new_shape;
  ge::Format original_format = current_tensor->GetOriginFormat();
  ShapeAndFormat shape_and_format_info = {original_shape, new_shape, original_format, new_heavy_format,
                                          tensor_info_ptr->data_type, group};
  (void)GetShapeAccordingToFormat(shape_and_format_info);
  current_tensor->SetShape(new_shape);
  current_tensor->SetFormat(new_heavy_format);

  /* update shape range for unknown op */
  if (CalcShapeRange(op_desc, tensor_info_ptr->heavy_format, original_shape, *current_tensor) != SUCCESS) {
    FE_LOGI("Set shape range of op[name:%s,type:%s] not successfully.", op_name.c_str(),
            op_type.c_str());
    return FAILED;
  }
  GetReshapeAxisValueByName(current_tensor->GetOriginFormat(), origin_shape_before_reshape, 'C', *current_tensor);
  FE_LOGW("Set format %s with datatype %u and original shape %s to new shape %s for the %u %s of node %s.",
          FormatToStr(new_heavy_format).c_str(),
          tensor_info_ptr->data_type,
          StringUtils::IntegerVecToString(origin_shape_before_reshape.GetDims()).c_str(),
          StringUtils::IntegerVecToString(new_shape.GetDims()).c_str(),
          tensor_info_ptr->anchor_index, IS_INPUT_TO_STRING(tensor_info_ptr->is_input), op_name.c_str());
  return SUCCESS;
}

Status HeavyFormatPropagation::SetFormatAndDTypeToOpdesc(const TensorInfoPtr &tensor_info_ptr,
                                                         const OpKernelInfoPtr &op_kernel_info_ptr,
                                                         Status set_format_result) {
  /* We will update the shape and format and datatype according
   * to the AnchorIndex */
  FE_LOGD("Begin to set format and dtype for node: %s, anchor index: %d.",
          tensor_info_ptr->node_ptr->GetName().c_str(), tensor_info_ptr->anchor_index);
  ge::GeTensorDescPtr current_tensor;
  if (GetCurrentTensor(tensor_info_ptr, current_tensor) != SUCCESS) {
    return FAILED;
  }
  std::unordered_set<ge::RefCell, ge::RefCellHash> reflections;
  bool is_sub_graph_data_or_net_output = false;
  Status ret = CheckSpecificSubGraphDataOrNetoutput(tensor_info_ptr, is_sub_graph_data_or_net_output, reflections);
  if (ret != CONTINUE_TO_SET_FORMAT) {
    return ret;
  }

  ge::Format old_format = static_cast<ge::Format>(ge::GetPrimaryFormat(current_tensor->GetFormat()));
  FE_LOGD("Old format of %s %u for node %s is %u, heavy_format=%s, sub_format=%d.",
          IS_INPUT_TO_STRING(tensor_info_ptr->is_input), tensor_info_ptr->anchor_index,
          tensor_info_ptr->op_desc_ptr->GetName().c_str(), old_format,
          FormatToStr(tensor_info_ptr->heavy_format).c_str(), tensor_info_ptr->sub_format);
  if (old_format != tensor_info_ptr->heavy_format) {
    ge::GeShape original_shape = current_tensor->GetOriginShape();
    /* We won't change shape of scalar */
    if (set_format_result != GRAPH_OPTIMIZER_STOP_TRAVERSING_SCALAR_TENSOR) {
      ge::GeShape new_shape;
      ret = SetNewShape(tensor_info_ptr, op_kernel_info_ptr, current_tensor, original_shape, new_shape);
      if (ret != SUCCESS) {
        return ret;
      }

      // just update even relative op case, eg. while, optimize later if needed
      // set relative GeTensorDesc, including Function op
      if (is_sub_graph_data_or_net_output) {
        FE_LOGD("Begin updating format: node %s, is input %d, anchor index %d.",
                tensor_info_ptr->node_ptr->GetName().c_str(), tensor_info_ptr->is_input, tensor_info_ptr->anchor_index);

        RelationUpdateInfo relation_update_info = {tensor_info_ptr->heavy_format, tensor_info_ptr->sub_format,
                                                   new_shape, INFER_FORMAT, 1};

        (void)FeGraphUtils::UpdateFormatOfRelatedEdges(reflections, relation_update_info);
      }
    } else {
      FE_LOGW("Dimension of %s %u for node %s is 0, stopping propagation.", IS_INPUT_TO_STRING(tensor_info_ptr->is_input),
              tensor_info_ptr->anchor_index, tensor_info_ptr->op_desc_ptr->GetName().c_str());
      return FAILED;
    }
  } else {
    std::string reshape_type = GetPropagationReshapeType(tensor_info_ptr, op_kernel_info_ptr, current_tensor);
    const ge::GeShape &original_shape = current_tensor->GetOriginShape();
    if (original_shape.GetDimNum() > reshape_type.size()) {
      tensor_info_ptr->propagation_info.reshape_type = "";
    } else {
      tensor_info_ptr->propagation_info.reshape_type = reshape_type;
    }
    FE_LOGD("Node [%s] reshaping type has been set to %s.", tensor_info_ptr->op_desc_ptr->GetName().c_str(),
            tensor_info_ptr->propagation_info.reshape_type.c_str());
  }
  ge::AttrUtils::SetInt(tensor_info_ptr->op_desc_ptr, "judge_match_idx", tensor_info_ptr->format_index);
  FE_LOGD("Set node [%s, %s]'s tensor format and shape successfully, judge_match_idx[%d].",
          tensor_info_ptr->op_desc_ptr->GetNamePtr(), tensor_info_ptr->op_desc_ptr->GetTypePtr(),
          tensor_info_ptr->format_index);
  return SUCCESS;
}

void HeavyFormatPropagation::SetFormatAgnosticType(const ge::OpDescPtr &op_desc_ptr, const NodeInfoPtr &node_info) {
  bool weight_or_data = IsWeightOrData(op_desc_ptr);
  /* Weight, Data, TransData, FE-inserted Reshape and other virtual op
   * will be penetrable. */
  bool penetrable = weight_or_data || op_desc_ptr->GetType() == TRANSDATA || IsFEInsertedReshape(op_desc_ptr) ||
                    CheckVirtualOp(op_desc_ptr);

  CorrectFmtAgnosticType(op_desc_ptr);

  node_info->is_sub_graph_data_or_nt_opt = false;
  if (penetrable) {
    node_info->format_selection = FormatSelectionType::FORMAT_PENETRATION;
  } else {
    int64_t format_agnostic = 0;
    int64_t impl_type = 0;
    /* If current op is tvm op, use ops kernel info to select format. */
    if (ge::AttrUtils::GetInt(op_desc_ptr, ge::ATTR_NAME_IMPLY_TYPE, impl_type) &&
        impl_type == static_cast<int64_t>(domi::ImplyType::TVM)) {
      node_info->format_selection = FormatSelectionType::FORMAT_DEPENDS_ON_OP_KERNEL_INFO;
      return;
    }
    if (ge::AttrUtils::GetInt(op_desc_ptr, FORMAT_AGNOSTIC, format_agnostic)) {
      if (format_agnostic >= static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_BOTTOM) ||
          format_agnostic < 0) {
        node_info->format_selection = FormatSelectionType::FORMAT_DEPENDS_ON_OP_KERNEL_INFO;
      } else {
        FE_LOGD("Put this %s Op's format_agnostic(%ld) into vector.",
                node_info->current_node->GetOpDesc()->GetName().c_str(), format_agnostic);
        format_agnostic_nodes_info_.push_back(node_info);
        node_info->format_selection = static_cast<FormatSelectionType>(format_agnostic);
      }
    } else {
      if (FeGraphUtils::IsSubGraphDataOrNetOutput(op_desc_ptr)) {
        node_info->is_sub_graph_data_or_nt_opt = true;
        node_info->format_selection = FormatSelectionType::FORMAT_AGNOSTIC_FUNCTION_OP;
        return;
      }
      node_info->format_selection = FormatSelectionType::FORMAT_DEPENDS_ON_OP_KERNEL_INFO;
    }
  }
}

Status HeavyFormatPropagation::SetOpKernelAndTensorMap(const NodeInfoPtr &node_info) {
  node_info->current_node_op_kernel_ptr = OpsKernelManager::Instance(engine_name_).GetOpKernelInfoByOpDesc(
      node_info->current_node->GetOpDesc());
  if (node_info->current_node_op_kernel_ptr == nullptr) {
    FE_LOGD("Can not find op kernel for current node %s.", node_info->current_node->GetName().c_str());
    FE_LOGD("Heavy format is %s for %s[%d] of node %s, sub format is %d, reshape type is %s.",
            ge::TypeUtils::FormatToSerialString(node_info->propagation_info.heavy_format).c_str(),
            IS_INPUT_TO_STRING(node_info->is_input_of_curr_node), node_info->anchor_index_of_curr_node,
            node_info->current_node->GetName().c_str(), node_info->propagation_info.sub_format,
            node_info->propagation_info.reshape_type.c_str());
    return HasHeavyOpAttr(node_info->current_node->GetOpDesc()) ? SUCCESS : FAILED;
  }

  IndexNameMap input_tensor_map, output_tensor_map;
  node_info->tensor_map = {input_tensor_map, output_tensor_map};
  if (GetInputOutputNameMap(*(node_info->current_node->GetOpDesc().get()), node_info->current_node_op_kernel_ptr,
                            node_info->tensor_map[INPUT_INDEX], node_info->tensor_map[OUTPUT_INDEX]) != SUCCESS) {
    return FAILED;
  }

  return SUCCESS;
}

void HeavyFormatPropagation::AddNodeInfoToQueue(NodeInfoPtr &node_info, std::deque<NodeInfoPtr> &next_node_queue) {
  if (node_info == nullptr) {
    return;
  }
  std::string next_node_reshape_type;
  Status ret;
  SetFormatAgnosticType(node_info->current_node->GetOpDesc(), node_info);
  if (node_info->format_selection == FormatSelectionType::FORMAT_DEPENDS_ON_OP_KERNEL_INFO) {
    ret = SetOpKernelAndTensorMap(node_info);
    if (ret != SUCCESS) {
      FE_LOGW("Cannot propagate through next node %s. It index is %d, heavy_format is %s, sub_format is %d.",
              node_info->current_node->GetName().c_str(), node_info->anchor_index_of_curr_node,
              FormatToStr(node_info->propagation_info.heavy_format).c_str(), node_info->propagation_info.sub_format);
    } else {
      if (JudgeIsFollowReshapeType(node_info->current_node->GetOpDesc(), node_info->current_node_op_kernel_ptr)) {
        FE_LOGD("node[%s] no need to get reshape type, follow last node.", node_info->current_node->GetName().c_str());
        ret = NEED_IGNORE;
      } else {
        ret = GetNodeReshapeType(node_info, next_node_reshape_type, false);
      }
    }
  } else {
    ret = GetNodeReshapeType(node_info, next_node_reshape_type, true);
  }
  if ((ret == SUCCESS) && !CheckIsNeedHeavy(node_info->last_node, node_info->last_node_info, node_info,
      next_node_reshape_type)) {
    return;
  }

  if (node_info->format_selection == FormatSelectionType::FORMAT_DEPENDS_ON_OP_KERNEL_INFO) {
    if ((ret == SUCCESS) && !IsOpKernelHeavyOp(node_info->current_node,
        node_info->current_node_op_kernel_ptr)) {
      FE_LOGD("Next node %s: %d, heavy_format is %s, sub_format is %d, c0_format is %d.",
              node_info->current_node->GetName().c_str(), node_info->anchor_index_of_curr_node,
              FormatToStr(node_info->propagation_info.heavy_format).c_str(),
              node_info->propagation_info.sub_format, node_info->propagation_info.c0_format);

      HeavyFormatInfo heavy_format_info(node_info->propagation_info.heavy_format,
                                        node_info->propagation_info.sub_format, node_info->propagation_info.c0_format,
                                        node_info->anchor_index_of_curr_node, node_info->is_input_of_curr_node);
      if (supportformats_updater_ptr_->UpdateSupportFormats(node_info->current_node,
                                                            node_info->current_node_op_kernel_ptr,
                                                            node_info->tensor_map, heavy_format_info) != SUCCESS) {
        FE_LOGW("UpdateSupportFormats failed: next node %s. It index is %d, heavy_format is %s, sub_format is %d.",
                node_info->current_node->GetName().c_str(), node_info->anchor_index_of_curr_node,
                FormatToStr(node_info->propagation_info.heavy_format).c_str(),
                node_info->propagation_info.sub_format);
      }
    }
  }
  next_node_queue.emplace_back(node_info);
}

Status HeavyFormatPropagation::GetNodeReshapeType(const NodeInfoPtr &node_info, string &reshape_type,
                                                  bool only_get_default_type) const {
  InputOrOutputInfoPtr input_or_output_info_ptr = nullptr;
  ge::GeTensorDescPtr tensor_desc = nullptr;
  IndexNameMap index_name_map;
  int32_t anchor_index = node_info->anchor_index_of_curr_node;
  ge::NodePtr node = node_info->current_node;
  if (node_info->propagation_info.heavy_format != ge::FORMAT_NC1HWC0) {
    return NEED_IGNORE;
  }
  if (anchor_index < 0) {
    FE_LOGD("[GraphOptJdgInst][FmtPropagate][GetRespTyp] Anchor index[%d] < 0, skip get reshape type.", anchor_index);
    return NEED_IGNORE;
  }
  FE_LOGD("[GraphOptJdgInst][FmtPropagate][GetRespTyp] Node[%s] tensor index [%d] is input index %d.",
          node->GetName().c_str(), anchor_index, node_info->is_input_of_curr_node);
  if (node_info->is_input_of_curr_node) {
    tensor_desc = node->GetOpDesc()->MutableInputDesc(anchor_index);
  } else {
    tensor_desc = node->GetOpDesc()->MutableOutputDesc(anchor_index);
  }
  FE_CHECK_NOTNULL(tensor_desc);
  size_t dim_num = tensor_desc->GetOriginShape().GetDimNum();
  if (dim_num >= NCHW_DIMENSION_NUM) {
    FE_LOGD("[GraphOptJdgInst][FmtPropagate][GetRespTyp] Node[%s] dim_num is %zu, need ignore.",
            node->GetName().c_str(), dim_num);
    return NEED_IGNORE;
  }
  if (!only_get_default_type && !node_info->tensor_map.empty()) {
    index_name_map = node_info->tensor_map[node_info->is_input_of_curr_node ? INPUT_INDEX : OUTPUT_INDEX];
    auto iter = index_name_map.find(anchor_index);
    if (iter == index_name_map.end()) {
      REPORT_FE_ERROR("[GraphOptJdgInst][FmtPropagate][GetRespTyp] Node[%s]: input %d is not found in tensor_map.",
                      node->GetName().c_str(), anchor_index);
      return FAILED;
    }
    if (node_info->current_node_op_kernel_ptr->GetTensorInfoByName(node_info->is_input_of_curr_node, iter->second,
        input_or_output_info_ptr) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOptJdgInst][FmtPropagate][GetRespTyp] Node[%s]: Input value %d was not found in the operation store.",
                      node->GetName().c_str(), anchor_index);
      return FAILED;
    }
    if (!input_or_output_info_ptr->GetReshapeType().empty()) {
      reshape_type = input_or_output_info_ptr->GetReshapeType();
      FE_LOGD("[GraphOptJdgInst][FmtPropagate][GetRespTyp] Node[%s] has reshape_type %s from opStore.",
              node->GetName().c_str(), reshape_type.c_str());
      return SUCCESS;
    }
  }
  if (GetDefaultReshapeType(tensor_desc->GetOriginFormat(), dim_num, reshape_type) != SUCCESS) {
    FE_LOGW("[GraphOptJdgInst][FmtPropagate][GetRespTyp] Node[%s] GetDefaultReshapeType failed.",
            node->GetName().c_str());
    return FAILED;
  }
  FE_LOGD("[GraphOptJdgInst][FmtPropagate][GetRespTyp] Node [%s] has the default reshape_type of %s.", node->GetName().c_str(),
          reshape_type.c_str());
  return SUCCESS;
}

bool HeavyFormatPropagation::CheckIsNeedHeavy(const ge::NodePtr &last_node, const NodeInfoPtr &last_node_info,
                                              const NodeInfoPtr &next_node_info,
                                              const std::string &next_node_reshape_type) const {
  // using next_node_info propagation_info as last node's because that next_node_info has been initialized as
  // last node propagation_info.
  std::string last_node_reshape_type = next_node_info->propagation_info.reshape_type;
  FE_LOGD("[GraphOptJdgInst][FmtPropagate][ChkIsNdHeavy] Node[%s] reshape type is %s.", last_node->GetName().c_str(),
          last_node_reshape_type.c_str());
  if (last_node_reshape_type.empty()) {
    Status ret = GetNodeReshapeType(last_node_info, last_node_reshape_type, true);
    if (ret == NEED_IGNORE) {
      FE_LOGD("[GraphOptJdgInst][FmtPropagate][ChkIsNdHeavy] Node[%s] reshape type need ignore.",
              last_node_info->current_node->GetName().c_str());
      return true;
    }
  }
  if ((next_node_info->propagation_info.heavy_format == ge::FORMAT_NC1HWC0) &&
      (last_node_reshape_type != next_node_reshape_type)) {
    FE_LOGD("[GraphOptJdgInst][FmtPropagate][ChkIsNdHeavy] Skip add node[%s] into queue.",
            next_node_info->current_node->GetName().c_str());
    return false;
  }
  return true;
}

void HeavyFormatPropagation::CreateNextNodeInfo(const ge::NodePtr &next_node, const NodeInfoPtr &last_node_info,
                                                ge::Format heavy_format, int32_t sub_format, int32_t c0_format,
                                                PropagationInfo &propagation_info, int32_t anchor_index, bool is_input,
                                                NodeInfoPtr &next_node_info,
                                                std::deque<NodeInfoPtr> &next_node_queue) {
  if (IsStaticZeroShapeOp(next_node->GetOpDesc())) {
    FE_LOGD("Skip %s because %s is an zero shape op.", next_node->GetName().c_str(), next_node->GetName().c_str());
    return;
  }

  FE_MAKE_SHARED(next_node_info = std::make_shared<NodeInfo>(next_node, last_node_info, heavy_format, sub_format,
                                                             c0_format, propagation_info, anchor_index, is_input),
                 return);
  AddNodeInfoToQueue(next_node_info, next_node_queue);

  std::vector<ge::NodePtr> var_nodes;
  FindSameNameVariableNodes(next_node, var_nodes);
  for (ge::NodePtr &var_node : var_nodes) {
    NodeInfoPtr var_node_info = nullptr;
    FE_MAKE_SHARED(var_node_info = std::make_shared<NodeInfo>(var_node, last_node_info, heavy_format, sub_format,
                                                              c0_format, propagation_info, anchor_index, is_input),
                   return);
    if (var_node_info == nullptr) {
      continue;
    }
    AddNodeInfoToQueue(var_node_info, next_node_queue);
  }
}

Status HeavyFormatPropagation::SetFormats(const TensorInfoPtr &tensor_info_ptr,
                                          const OpKernelInfoPtr &op_kernel_info_ptr) {
  /* Set infer format to indicate that we have already traversed this
   * node */
  FE_LOGD("SetFormats, format_index %d, heavy_format %s, sub_format %d.", tensor_info_ptr->format_index,
          FormatToStr(tensor_info_ptr->heavy_format).c_str(), tensor_info_ptr->sub_format);
  Status ret = SetInferFormat(tensor_info_ptr);
  if (ret != SUCCESS && ret != GRAPH_OPTIMIZER_NOT_HEAVY_FORMAT &&
      ret != GRAPH_OPTIMIZER_STOP_TRAVERSING_SCALAR_TENSOR) {
    return ret;
  }
  /* 1. format_index == -1 means it's the heavy op itself and we don't need to
   * set the format and shape for heavy op.
   * 2. If the next node is weight, we don't need to set the format and
   * shape format it. */
  if (tensor_info_ptr->format_index != -1 &&
      tensor_info_ptr->format_selection_type != FormatSelectionType::FORMAT_PENETRATION) {
    Status status = SetFormatAndDTypeToOpdesc(tensor_info_ptr, op_kernel_info_ptr, ret);
    if (status != SUCCESS) {
      return status;
    }
  }
  return ret;
}

Status HeavyFormatPropagation::PropagateForwards(const NodeInfoPtr &node_info, int32_t format_index,
                                                 FormatSelectionType format_selection_type,
                                                 std::deque<NodeInfoPtr> &next_node_queue) {
  auto current_node_name = node_info->current_node->GetName();
  FE_LOGD("Propagate forwards from %s by format index %d, reshape type %s, sub_format %d.", current_node_name.c_str(),
          format_index, node_info->propagation_info.reshape_type.c_str(), node_info->propagation_info.sub_format);
  ge::OpDescPtr op_desc_ptr = node_info->current_node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);

  TensorInfoPtr tensor_info_ptr;
  /* Three bool: is_input, is_forward */
  FE_MAKE_SHARED(tensor_info_ptr = std::make_shared<FormatAndDataTypeInfo>(
                     node_info->current_node, op_desc_ptr, 0, format_index, false, true, format_selection_type,
                     node_info->propagation_info, node_info->tensor_map[OUTPUT_INDEX]),
                 return FAILED);

  // if current_node is sub graph NETOUTPUT, get next node by relative function node
  bool is_subgraph_netoutput = false;

  if (FeGraphUtils::IsSubGraphNetOutput(op_desc_ptr)) {
    is_subgraph_netoutput = true;
    tensor_info_ptr->is_input = true;
  }

  tensor_info_ptr->is_sub_graph_data_or_nt_opt = node_info->is_sub_graph_data_or_nt_opt;
  // sub graph NETOUTPUT, get Function successor op as NextNode
  if (is_subgraph_netoutput) {
    return PropagateSubNetoutputForwards(node_info, format_index, tensor_info_ptr, next_node_queue);
  } else {
    return PropagateNormalNodeForwards(node_info, format_index, tensor_info_ptr, next_node_queue);
  }
}

void HeavyFormatPropagation::PropagateFarther(const NodeInfoPtr &curr_node_info, const NodeInfoPtr &next_node_info,
                                              std::deque<NodeInfoPtr> &next_node_queue) {
  if (curr_node_info == nullptr) {
    (void)PropagateForwards(next_node_info, -1, FormatSelectionType::FORMAT_DEPENDS_ON_OP_KERNEL_INFO, next_node_queue);
    (void)PropagateBackwards(next_node_info, -1, FormatSelectionType::FORMAT_DEPENDS_ON_OP_KERNEL_INFO,
                             next_node_queue);
  } else {
    auto next_node = next_node_info->current_node;
    FE_LOGD("Propagating further to node: %s using heavy format: %s and sub format: %d.", next_node->GetName().c_str(),
            FormatToStr(next_node_info->propagation_info.heavy_format).c_str(),
            next_node_info->propagation_info.sub_format);
    if (IsNextNodeVisitedOrNull(curr_node_info->last_node, next_node)) {
      return;
    }

    (void)RunPropagation(next_node_info, next_node_queue);
  }
}

Status HeavyFormatPropagation::PropagateBackwards(const NodeInfoPtr &node_info, int32_t format_index,
                                                  FormatSelectionType format_selection_type,
                                                  std::deque<NodeInfoPtr> &next_node_queue) {
  FE_CHECK_NOTNULL(node_info->current_node);
  ge::OpDescPtr op_desc_ptr = node_info->current_node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  if (IsWeightOrData(op_desc_ptr) && !HasHeavyOpAttr(op_desc_ptr)) {
    FE_LOGD("Weight %s does not need to traverse backwards.", op_desc_ptr->GetName().c_str());
    return SUCCESS;
  }
  FE_LOGD("Propagate backwards from %s by format index %d.", node_info->current_node->GetName().c_str(), format_index);
  TensorInfoPtr tensor_info_ptr;
  FE_MAKE_SHARED(tensor_info_ptr = std::make_shared<FormatAndDataTypeInfo>(
                     node_info->current_node, op_desc_ptr, 0, format_index, true, false, format_selection_type,
                     node_info->propagation_info, node_info->tensor_map[INPUT_INDEX]),
                 return FAILED);

  // if current_node is sub graph DATA, get next node by relative function node
  bool is_subgraph_data = false;
  if (FeGraphUtils::IsSubGraphData(op_desc_ptr)) {
    is_subgraph_data = true;
    // if sub graph DATA, should set is_input = false
    tensor_info_ptr->is_input = false;
  }
  tensor_info_ptr->is_sub_graph_data_or_nt_opt = node_info->is_sub_graph_data_or_nt_opt;
  // sub graph NETOUTPUT, get Function successor op as NextNode
  if (is_subgraph_data) {
    return PropagateSubDataBackwards(node_info, tensor_info_ptr, next_node_queue);
  } else {
    return PropagateNormalNodeBackwards(node_info, format_index, tensor_info_ptr, next_node_queue);
  }
}

Status HeavyFormatPropagation::RunPropagation(const NodeInfoPtr &node_info, std::deque<NodeInfoPtr> &next_node_queue) {
  FE_CHECK_NOTNULL(node_info->current_node);
  auto op_desc_ptr = node_info->current_node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  auto curr_node_name = node_info->current_node->GetName();
  FE_LOGD("Propagation run from op %s to %s[%d] of op %s using heavy format %s, sub format %d and reshape type %s.",
          node_info->last_node->GetName().c_str(), IS_INPUT_TO_STRING(node_info->is_input_of_curr_node),
          node_info->anchor_index_of_curr_node, curr_node_name.c_str(),
          FormatToStr(node_info->propagation_info.heavy_format).c_str(), node_info->propagation_info.sub_format,
          node_info->propagation_info.reshape_type.c_str());

  FE_LOGD("Format selection type of %s is %u.", op_desc_ptr->GetName().c_str(),
          static_cast<uint32_t>(node_info->format_selection));
  if (node_info->format_selection != FormatSelectionType::FORMAT_DEPENDS_ON_OP_KERNEL_INFO) {
    Status ret = PropagateForwards(node_info, INVALID_FORMAT_INDEX, node_info->format_selection, next_node_queue);
    if (ret == STOP_PROPAGATION_FROM_WEIGHT || ret == GRAPH_OPTIMIZER_STOP_TRAVERSING_SUBGRAPH) {
      return SUCCESS;
    }

    PropagateBackwards(node_info, INVALID_FORMAT_INDEX, node_info->format_selection, next_node_queue);
  } else {
    FE_CHECK(node_info->current_node_op_kernel_ptr == nullptr,
             FE_LOGW("Can not find op kernel for current node %s.", curr_node_name.c_str()), return SUCCESS);

    /* Stop propagating at the next heavy op. */
    if (IsOpKernelHeavyOp(node_info->current_node, node_info->current_node_op_kernel_ptr) ||
        IsHeavyFormatMatmul(node_info->current_node)) {
      FE_LOGD("Stop propagating from heavy operation [%s].", curr_node_name.c_str());
      return FAILED;
    }

    int32_t heavy_format_index = -1;
    HeavyFormatInfo heavy_format_info(node_info->propagation_info.heavy_format, node_info->propagation_info.sub_format,
                                      node_info->propagation_info.c0_format, node_info->anchor_index_of_curr_node,
                                      node_info->is_input_of_curr_node);
    /* Get the most suitable heavy format index in current node's ops
     * kernel. */
    Status ret = GetHeavyFormatIndex(node_info, heavy_format_info, heavy_format_index);

    if (heavy_format_index == -1 || ret != SUCCESS) {
      return SUCCESS;
    } else {
      PropagateForwards(node_info, heavy_format_index, node_info->format_selection, next_node_queue);
      PropagateBackwards(node_info, heavy_format_index, node_info->format_selection, next_node_queue);
    }
  }
  return SUCCESS;
}

Status HeavyFormatPropagation::UpdateOutputFormatAndShape(const ge::NodePtr &current_node,
    const vector<int64_t> &input_non_format_agnostic_index, const vector<int64_t> &output_non_format_agnostic_index
    ) const {
  PeerFormatAndShapeInfo peer_in_format_and_shape(current_node, ge::FORMAT_RESERVED, 0, 0);
  Status ret = GetPeerInFormat(output_non_format_agnostic_index, peer_in_format_and_shape);
  if (ret != SUCCESS) {
    return ret;
  }
  ge::OpDescPtr op_desc_ptr = current_node->GetOpDesc();
  std::string reshape_type;
  // update this node's all output format
  for (auto &out_data_anchor : current_node->GetAllOutDataAnchors()) {
    int32_t out_anchor_index = out_data_anchor->GetIdx();
    if (std::find(output_non_format_agnostic_index.begin(), output_non_format_agnostic_index.end(), out_anchor_index) !=
        output_non_format_agnostic_index.end()) {
      FE_LOGD("When updating the output anchor, this anchor %d of %s is not format-agnostic!", out_anchor_index,
              current_node->GetName().c_str());
      continue;
    }
    if (peer_in_format_and_shape.peer_primary_format == ge::FORMAT_RESERVED) {
      continue;
    }
    FE_LOGD("Node %s output index %d has format %s.", op_desc_ptr->GetName().c_str(), out_anchor_index,
            ge::TypeUtils::FormatToSerialString(peer_in_format_and_shape.peer_primary_format).c_str());
    auto output_desc_ptr = op_desc_ptr->MutableOutputDesc(out_anchor_index);
    if (output_desc_ptr == nullptr) {
      continue;
    }
    ge::Format new_format = static_cast<ge::Format>(ge::GetFormatFromSub(peer_in_format_and_shape.peer_primary_format,
        peer_in_format_and_shape.peer_sub_format));
    if (IsHeavyFormat(peer_in_format_and_shape.peer_primary_format)) {
      int32_t c0_bit_val = GetC0BitByDataType(output_desc_ptr->GetDataType());
      new_format = static_cast<ge::Format>(ge::GetFormatFromC0(new_format, c0_bit_val));
    }
    for (auto peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      if (peer_in_anchor == nullptr) {
        continue;
      }
      auto peer_in_node = peer_in_anchor->GetOwnerNode();
      int32_t peer_in_anchor_index = peer_in_anchor->GetIdx();
      if (reshape_type.empty()) {
        (void)ge::AttrUtils::GetStr(peer_in_node->GetOpDesc()->MutableOutputDesc(peer_in_anchor_index),
                                    ge::ATTR_NAME_RESHAPE_INFER_TYPE, reshape_type);
      }
    }
    output_desc_ptr->SetFormat(new_format);
    output_desc_ptr->SetShape(peer_in_format_and_shape.peer_shape);
  }
  // update this node's all input format
  for (auto &in_data_anchor : current_node->GetAllInDataAnchors()) {
    int32_t in_anchor_index = in_data_anchor->GetIdx();
    if (std::find(input_non_format_agnostic_index.begin(), input_non_format_agnostic_index.end(), in_anchor_index) !=
        input_non_format_agnostic_index.end()) {
      FE_LOGD("When updating the input anchor, this anchor %d of %s is not format-agnostic!", in_anchor_index,
              current_node->GetName().c_str());
      continue;
    }

    auto input_format = op_desc_ptr->GetInputDesc(in_anchor_index).GetFormat();
    auto input_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(input_format));
    if (peer_in_format_and_shape.peer_primary_format == ge::FORMAT_RESERVED ||
        input_primary_format == peer_in_format_and_shape.peer_primary_format) {
      continue;
    }
    FE_LOGD("Node %s input index %d with format %s.", op_desc_ptr->GetName().c_str(), in_anchor_index,
            ge::TypeUtils::FormatToSerialString(peer_in_format_and_shape.peer_primary_format).c_str());
    ge::GeShape new_shape;
    int64_t hidden_size = 1;
    int64_t input_size = 1;
    int64_t state_size = -1;
    (void)ge::AttrUtils::GetInt(op_desc_ptr, "hidden_size", hidden_size);
    (void)ge::AttrUtils::GetInt(op_desc_ptr, "input_size", input_size);
    (void)ge::AttrUtils::GetInt(op_desc_ptr, "state_size", state_size);
    CalcShapeExtraAttr extra_attr = {hidden_size, input_size, state_size};
    ShapeAndFormat input_and_output_info = {op_desc_ptr->GetInputDesc(in_anchor_index).GetOriginShape(),
                                            new_shape,
                                            op_desc_ptr->GetInputDesc(in_anchor_index).GetOriginFormat(),
                                            peer_in_format_and_shape.peer_primary_format,
                                            op_desc_ptr->GetInputDesc(in_anchor_index).GetDataType(),
                                            peer_in_format_and_shape.peer_sub_format,
                                            extra_attr};
    (void)GetShapeAccordingToFormat(input_and_output_info);
    auto input_desc_ptr = op_desc_ptr->MutableInputDesc(in_anchor_index);
    if (input_desc_ptr == nullptr) {
      continue;
    }
    ge::Format new_format = static_cast<ge::Format>(ge::GetFormatFromSub(peer_in_format_and_shape.peer_primary_format,
        peer_in_format_and_shape.peer_sub_format));
    if (IsHeavyFormat(peer_in_format_and_shape.peer_primary_format)) {
      int32_t c0_bit_val = GetC0BitByDataType(input_desc_ptr->GetDataType());
      new_format = static_cast<ge::Format>(ge::GetFormatFromC0(new_format, c0_bit_val));
      if (!reshape_type.empty()) {
        ExpandDimension(input_and_output_info.old_format, new_format, reshape_type, input_and_output_info.old_shape,
                        new_shape, input_desc_ptr);
        input_and_output_info.old_shape = new_shape;
        input_and_output_info.new_format = new_format;
        (void)GetShapeAccordingToFormat(input_and_output_info);
      }
    }
    input_desc_ptr->SetFormat(new_format);
    input_desc_ptr->SetShape(new_shape);
  }
  return SUCCESS;
}

Status HeavyFormatPropagation::UpdateInputFormatAndShape(ge::NodePtr &current_node,
    const vector<int64_t> &input_non_format_agnostic_index, const vector<int64_t> &output_non_format_agnostic_index
    ) const {
  PeerFormatAndShapeInfo peer_out_format_and_shape(current_node, ge::FORMAT_RESERVED, 0, 0);
  Status ret = GetPeerOutFormat(input_non_format_agnostic_index, peer_out_format_and_shape);
  if (ret != SUCCESS) {
    return ret;
  }
  ge::OpDescPtr op_desc_ptr = current_node->GetOpDesc();
  std::string reshape_type;
  // update this node's all input format
  for (auto &in_data_anchor : current_node->GetAllInDataAnchors()) {
    int32_t in_anchor_index = in_data_anchor->GetIdx();
    if (std::find(input_non_format_agnostic_index.begin(), input_non_format_agnostic_index.end(), in_anchor_index) !=
        input_non_format_agnostic_index.end()) {
      FE_LOGD("When updating the input anchor, this anchor %d of %s is not format-agnostic!", in_anchor_index,
              current_node->GetName().c_str());
      continue;
    }
    if (peer_out_format_and_shape.peer_primary_format == ge::FORMAT_RESERVED) {
      continue;
    }
    FE_LOGD("node %s: input index %d, format %s, sub_format %d.", op_desc_ptr->GetName().c_str(), in_anchor_index,
            ge::TypeUtils::FormatToSerialString(peer_out_format_and_shape.peer_primary_format).c_str(),
            peer_out_format_and_shape.peer_sub_format);
    auto input_desc_ptr = op_desc_ptr->MutableInputDesc(in_anchor_index);
    if (input_desc_ptr == nullptr) {
      continue;
    }
    auto new_format = static_cast<ge::Format>(ge::GetFormatFromSub(peer_out_format_and_shape.peer_primary_format,
                                                                   peer_out_format_and_shape.peer_sub_format));
    if (IsHeavyFormat(peer_out_format_and_shape.peer_primary_format)) {
      int32_t c0_bit_val = GetC0BitByDataType(input_desc_ptr->GetDataType());
      new_format = static_cast<ge::Format>(ge::GetFormatFromC0(new_format, c0_bit_val));
    }
    auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_out_anchor != nullptr) {
      auto peer_out_node = peer_out_anchor->GetOwnerNode();
      int32_t peer_out_anchor_index = peer_out_anchor->GetIdx();
      if (reshape_type.empty()) {
        (void)ge::AttrUtils::GetStr(peer_out_node->GetOpDesc()->MutableOutputDesc(peer_out_anchor_index),
                                    ge::ATTR_NAME_RESHAPE_INFER_TYPE, reshape_type);
      }
    }
    input_desc_ptr->SetFormat(new_format);
    input_desc_ptr->SetShape(peer_out_format_and_shape.peer_shape);
  }
  // update this node's all output format
  for (auto &out_data_anchor : current_node->GetAllOutDataAnchors()) {
    int32_t out_anchor_index = out_data_anchor->GetIdx();
    if (std::find(output_non_format_agnostic_index.begin(), output_non_format_agnostic_index.end(), out_anchor_index) !=
        output_non_format_agnostic_index.end()) {
      FE_LOGD("When updating the output anchor, this anchor %d of %s is not format-agnostic!", out_anchor_index,
              current_node->GetName().c_str());
      continue;
    }
    auto output_format = op_desc_ptr->GetOutputDesc(out_anchor_index).GetFormat();
    auto output_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(output_format));
    if (peer_out_format_and_shape.peer_primary_format == ge::FORMAT_RESERVED ||
        output_primary_format == peer_out_format_and_shape.peer_primary_format) {
      continue;
    }
    FE_LOGD("Node %s output index %d has format %s.", op_desc_ptr->GetName().c_str(), out_anchor_index,
            ge::TypeUtils::FormatToSerialString(peer_out_format_and_shape.peer_primary_format).c_str());
    ge::GeShape new_shape;
    int64_t hidden_size = 1;
    int64_t input_size = 1;
    int64_t state_size = -1;
    (void)ge::AttrUtils::GetInt(op_desc_ptr, "hidden_size", hidden_size);
    (void)ge::AttrUtils::GetInt(op_desc_ptr, "input_size", input_size);
    (void)ge::AttrUtils::GetInt(op_desc_ptr, "state_size", state_size);
    CalcShapeExtraAttr extra_attr = {hidden_size, input_size, state_size};
    ShapeAndFormat input_and_output_info = {op_desc_ptr->GetOutputDesc(out_anchor_index).GetOriginShape(),
                                            new_shape,
                                            op_desc_ptr->GetOutputDesc(out_anchor_index).GetOriginFormat(),
                                            peer_out_format_and_shape.peer_primary_format,
                                            op_desc_ptr->GetOutputDesc(out_anchor_index).GetDataType(),
                                            peer_out_format_and_shape.peer_sub_format,
                                            extra_attr};
    (void)GetShapeAccordingToFormat(input_and_output_info);
    auto output_desc_ptr = op_desc_ptr->MutableOutputDesc(out_anchor_index);
    if (output_desc_ptr == nullptr) {
      continue;
    }
    auto new_format = static_cast<ge::Format>(ge::GetFormatFromSub(peer_out_format_and_shape.peer_primary_format,
                                                                   peer_out_format_and_shape.peer_sub_format));
    if (IsHeavyFormat(peer_out_format_and_shape.peer_primary_format)) {
      int32_t c0_bit_val = GetC0BitByDataType(output_desc_ptr->GetDataType());
      new_format = static_cast<ge::Format>(ge::GetFormatFromC0(new_format, c0_bit_val));
      if (!reshape_type.empty()) {
        ExpandDimension(input_and_output_info.old_format, new_format, reshape_type, input_and_output_info.old_shape,
                        new_shape, output_desc_ptr);
        input_and_output_info.old_shape = new_shape;
        input_and_output_info.new_format = new_format;
        (void)GetShapeAccordingToFormat(input_and_output_info);
      }
    }
    output_desc_ptr->SetFormat(new_format);
    output_desc_ptr->SetShape(new_shape);
  }
  return SUCCESS;
}

bool HeavyFormatPropagation::CheckFormatCompatible(const ge::Format &primary_format,
                                                   const ge::Format &origin_format) const {
  if (primary_format == ge::FORMAT_NC1HWC0 && origin_format == ge::FORMAT_ND) {
    FE_LOGD("Primary format [%s] and origin format [%s] are not compatible.",
            FormatToStr(primary_format).c_str(), FormatToStr(origin_format).c_str());
    return false;
  }
  return true;
}

Status HeavyFormatPropagation::GetPeerInFormat(const vector<int64_t> &output_non_format_agnostic_index,
                                               PeerFormatAndShapeInfo &peer_in_format_and_shape) const {
  auto node_name = peer_in_format_and_shape.current_node->GetName();
  ge::OpDescPtr op_desc_ptr = peer_in_format_and_shape.current_node->GetOpDesc();
  for (ge::OutDataAnchorPtr &out_data_anchor : peer_in_format_and_shape.current_node->GetAllOutDataAnchors()) {
    int64_t output_index = static_cast<int64_t>(out_data_anchor->GetIdx());
    if (std::find(output_non_format_agnostic_index.begin(), output_non_format_agnostic_index.end(), output_index)
        != output_non_format_agnostic_index.end()) {
      FE_LOGD("Output anchor %ld of %s is not format-agnostic!", output_index, node_name.c_str());
      continue;
    }
    uint32_t out_anchor_index = static_cast<uint32_t>(output_index);
    ge::ConstGeTensorDescPtr output_desc_ptr = op_desc_ptr->GetOutputDescPtr(out_anchor_index);
    if (output_desc_ptr == nullptr) {
      continue;
    }
    int64_t tensor_format_continuous = 0;
    if (!ge::AttrUtils::GetInt(output_desc_ptr, FORMAT_CONTINUOUS, tensor_format_continuous) ||
        (tensor_format_continuous == 0)) {
      FE_LOGD("Output tensor %u of %s does not have _format_continuous or has a value of 0!", out_anchor_index,
              node_name.c_str());
      continue;
    }
    for (auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      int32_t in_anchor_index = peer_in_data_anchor->GetIdx();
      auto input_desc = peer_in_data_anchor->GetOwnerNode()->GetOpDesc()->MutableInputDesc(in_anchor_index);
      if (input_desc == nullptr) {
        FE_LOGD("The input_desc is null.");
        continue;
      }
      ge::Format curr_peer_in_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(input_desc->GetFormat()));
      int32_t curr_peer_in_sub_format = static_cast<ge::Format>(ge::GetSubFormat(input_desc->GetFormat()));
      int32_t curr_peer_in_c0_format = static_cast<int32_t>(ge::GetC0Value(input_desc->GetFormat()));

      peer_in_format_and_shape.peer_shape = input_desc->MutableShape();
      if (peer_in_format_and_shape.peer_primary_format == ge::FORMAT_RESERVED) {
        peer_in_format_and_shape.peer_primary_format = curr_peer_in_primary_format;
        peer_in_format_and_shape.peer_sub_format = curr_peer_in_sub_format;
        peer_in_format_and_shape.peer_c0_format = curr_peer_in_c0_format;
      } else if (peer_in_format_and_shape.peer_primary_format != curr_peer_in_primary_format) {
        FE_LOGW("Peer in anchor's format(%s != %s) of %s is not same!",
                ge::TypeUtils::FormatToSerialString(peer_in_format_and_shape.peer_primary_format).c_str(),
                ge::TypeUtils::FormatToSerialString(curr_peer_in_primary_format).c_str(), node_name.c_str());
        return NOT_CHANGED;
      }
    }
    if (!CheckFormatCompatible(peer_in_format_and_shape.peer_primary_format, output_desc_ptr->GetOriginFormat())) {
      FE_LOGD("Output[%u] of node[%s, %s] has an origin format[%s] that is not compatible with peer in format[%s]",
              out_anchor_index, op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), 
              FormatToStr(output_desc_ptr->GetOriginFormat()).c_str(), 
              FormatToStr(peer_in_format_and_shape.peer_primary_format).c_str());
      return NOT_CHANGED;
    }
  }
  return SUCCESS;
}

Status HeavyFormatPropagation::GetPeerOutFormat(const vector<int64_t> &input_non_format_agnostic_index,
                                                PeerFormatAndShapeInfo &peer_out_format_and_shape) const {
  auto node_name = peer_out_format_and_shape.current_node->GetName();
  ge::OpDescPtr op_desc_ptr = peer_out_format_and_shape.current_node->GetOpDesc();
  for (ge::InDataAnchorPtr &in_data_anchor : peer_out_format_and_shape.current_node->GetAllInDataAnchors()) {
    int64_t input_index = static_cast<int64_t>(in_data_anchor->GetIdx());
    if (std::find(input_non_format_agnostic_index.begin(), input_non_format_agnostic_index.end(), input_index)
        != input_non_format_agnostic_index.end()) {
      FE_LOGD("Input anchor %ld of %s is not format-agnostic!", input_index, node_name.c_str());
      continue;
    }
    uint32_t in_anchor_index = static_cast<uint32_t>(input_index);
    ge::ConstGeTensorDescPtr input_desc_ptr = op_desc_ptr->GetInputDescPtr(in_anchor_index);
    if (input_desc_ptr == nullptr) {
      continue;
    }
    int64_t tensor_format_continuous = 0;
    if (!ge::AttrUtils::GetInt(input_desc_ptr, FORMAT_CONTINUOUS, tensor_format_continuous) ||
        (tensor_format_continuous == 0)) {
      FE_LOGD("Input tensor %u of %s does not have _format_continuous or value equal to 0!", in_anchor_index,
              node_name.c_str());
      continue;
    }

    if (in_data_anchor->GetPeerOutAnchor() == nullptr) {
      REPORT_FE_ERROR("[GraphOptJdgInst][FmtPropagate][UpdAllTypeIn] Input anchor %u of %s's output anchor is nullptr!",
                      in_anchor_index, node_name.c_str());
      return FAILED;
    }

    auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    /* If the peer out node is variable or constant, we do not need to use this attribute.
     * Becuase weight will always be the original format and if we use continuous attribute, switch
     * will never be inferred as heavy format(5HD or Fz). */
    auto owner_node = peer_out_anchor->GetOwnerNode();
    if (owner_node == nullptr) {
      REPORT_FE_ERROR("[GraphOptJdgInst][FmtPropagate][UpdAllTypeIn] Input anchor %u of %s's owner_node is nullptr.",
                      in_anchor_index, node_name.c_str());
      return FAILED;
    }
    if (IsWeight(owner_node->GetOpDesc())) {
      FE_LOGD("node %s is connected to weight node, so we do not use its _format_continuous attr.",
              peer_out_format_and_shape.current_node->GetName().c_str());
      continue;
    }

    int peer_out_idx = peer_out_anchor->GetIdx();
    // get peer out anchor format & shape of other way op
    auto ouput_desc = owner_node->GetOpDesc()->MutableOutputDesc(peer_out_idx);
    if (ouput_desc == nullptr) {
      REPORT_FE_ERROR("[GraphOptJdgInst][FmtPropagate][UpdAllTypeIn] Ouput_desc %u of %s's out anchor is nullptr!",
                      in_anchor_index, node_name.c_str());
      continue;
    }
    ge::Format curr_peer_out_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(ouput_desc->GetFormat()));
    ge::Format curr_peer_out_sub_format = static_cast<ge::Format>(ge::GetSubFormat(ouput_desc->GetFormat()));
    int32_t curr_peer_out_c0_format = static_cast<int32_t>(ge::GetC0Value(ouput_desc->GetFormat()));
    if (!CheckFormatCompatible(curr_peer_out_primary_format, input_desc_ptr->GetOriginFormat())) {
      FE_LOGD("The original format [%s] of input[%u] node[%s, %s] is not compatible with the peer output format [%s].",
              FormatToStr(input_desc_ptr->GetOriginFormat()).c_str(), in_anchor_index, op_desc_ptr->GetName().c_str(),
              op_desc_ptr->GetType().c_str(), FormatToStr(curr_peer_out_primary_format).c_str());
      return NOT_CHANGED;
    }

    peer_out_format_and_shape.peer_shape = ouput_desc->MutableShape();
    if (peer_out_format_and_shape.peer_primary_format == ge::FORMAT_RESERVED) {
      peer_out_format_and_shape.peer_primary_format = curr_peer_out_primary_format;
      peer_out_format_and_shape.peer_sub_format = curr_peer_out_sub_format;
      peer_out_format_and_shape.peer_c0_format = curr_peer_out_c0_format;
    } else if (peer_out_format_and_shape.peer_primary_format != curr_peer_out_primary_format) {
      FE_LOGW("Peer out anchor's format(%s != %s) of %s is not same!",
              ge::TypeUtils::FormatToSerialString(peer_out_format_and_shape.peer_primary_format).c_str(),
              ge::TypeUtils::FormatToSerialString(curr_peer_out_primary_format).c_str(), node_name.c_str());
      return NOT_CHANGED;
    }
  }
  return SUCCESS;
}

Status HeavyFormatPropagation::UpdateInputForPairType(const ge::NodePtr &current_node,
                                                      const vector<int64_t> &input_non_format_agnostic_index,
                                                      const vector<int64_t> &output_non_format_agnostic_index) const {
  // for this node 's all input data
  auto node_name = current_node->GetName();
  ge::OpDescPtr op_desc_ptr = current_node->GetOpDesc();
  for (ge::InDataAnchorPtr &in_data_anchor : current_node->GetAllInDataAnchors()) {
    int64_t input_index = static_cast<int64_t>(in_data_anchor->GetIdx());
    if (std::find(input_non_format_agnostic_index.begin(), input_non_format_agnostic_index.end(), input_index)
        != input_non_format_agnostic_index.end()) {
      FE_LOGD("Input anchor %ld of %s is not format-agnostic!", input_index, node_name.c_str());
      continue;
    }
    uint32_t in_anchor_index = static_cast<uint32_t>(input_index);
    ge::GeTensorDescPtr input_desc_ptr = op_desc_ptr->MutableInputDesc(in_anchor_index);
    if (input_desc_ptr == nullptr) {
      continue;
    }
    int64_t tensor_format_continuous = 0;
    if (!ge::AttrUtils::GetInt(input_desc_ptr, FORMAT_CONTINUOUS, tensor_format_continuous) ||
        (tensor_format_continuous == 0)) {
      FE_LOGD("Input tensor %u of %s does not have _format_continuous or value equal to 0!", in_anchor_index,
              node_name.c_str());
      continue;
    }
    if (in_data_anchor->GetPeerOutAnchor() == nullptr) {
      REPORT_FE_ERROR("[GraphOptJdgInst][FmtPropagate][UpdPairTypeIn] Input anchor %u of %s's output anchor is nullptr!",
                      in_anchor_index, node_name.c_str());
      return FAILED;
    }

    int peer_out_idx = in_data_anchor->GetPeerOutAnchor()->GetIdx();
    // get peer out anchor format & shape of other way op
    auto peer_out_output_desc =
        in_data_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(peer_out_idx);
    if (peer_out_output_desc == nullptr) {
      continue;
    }
    auto curr_peer_out_format = peer_out_output_desc->GetFormat();
    input_desc_ptr->SetFormat(curr_peer_out_format);
    ge::GeShape curr_peer_out_shape = peer_out_output_desc->GetShape();
    input_desc_ptr->SetShape(curr_peer_out_shape);

    ge::Format curr_peer_out_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(curr_peer_out_format));
    ge::Format curr_peer_out_sub_format = static_cast<ge::Format>(ge::GetSubFormat(curr_peer_out_format));
    if (!CheckFormatCompatible(curr_peer_out_primary_format, input_desc_ptr->GetOriginFormat())) {
      FE_LOGD("The original format [%s] of input[%u] node[%s, %s] is not compatible with the peer output format [%s].",
              FormatToStr(input_desc_ptr->GetOriginFormat()).c_str(), in_anchor_index, op_desc_ptr->GetName().c_str(),
              op_desc_ptr->GetType().c_str(), FormatToStr(curr_peer_out_primary_format).c_str());
      continue;
    }

    auto output_desc = op_desc_ptr->GetOutputDesc(in_anchor_index);
    auto output_format = static_cast<ge::Format>(ge::GetPrimaryFormat(output_desc.GetFormat()));
    if (std::find(output_non_format_agnostic_index.begin(), output_non_format_agnostic_index.end(), in_anchor_index) ==
            output_non_format_agnostic_index.end() &&
        output_format != curr_peer_out_primary_format) {
      FE_LOGD("Output anchor %u of %s is format-agnostic, so updating this format to %s.", in_anchor_index,
              node_name.c_str(), ge::TypeUtils::FormatToSerialString(curr_peer_out_format).c_str());
      ge::GeShape new_shape;
      int64_t hidden_size = 1;
      int64_t input_size = 1;
      int64_t state_size = -1;
      (void)ge::AttrUtils::GetInt(op_desc_ptr, "hidden_size", hidden_size);
      (void)ge::AttrUtils::GetInt(op_desc_ptr, "input_size", input_size);
      (void)ge::AttrUtils::GetInt(op_desc_ptr, "state_size", state_size);
      CalcShapeExtraAttr extra_attr = {hidden_size, input_size, state_size};
      ShapeAndFormat input_and_output_info = {output_desc.GetOriginShape(),  new_shape,
                                              output_desc.GetOriginFormat(), curr_peer_out_primary_format,
                                              output_desc.GetDataType(),
                                              curr_peer_out_sub_format,      extra_attr};
      (void)GetShapeAccordingToFormat(input_and_output_info);
      auto output_desc_ptr = op_desc_ptr->MutableOutputDesc(in_anchor_index);
      if (output_desc_ptr == nullptr) {
        continue;
      }
      output_desc_ptr->SetFormat(curr_peer_out_format);
      output_desc_ptr->SetShape(new_shape);
    }
  }
  return SUCCESS;
}

Status HeavyFormatPropagation::UpdateOutputForPairType(const ge::NodePtr &current_node,
                                                       const vector<int64_t> &input_non_format_agnostic_index,
                                                       const vector<int64_t> &output_non_format_agnostic_index) const {
  // for this node 's all input data
  auto node_name = current_node->GetName();
  ge::OpDescPtr op_desc_ptr = current_node->GetOpDesc();
  for (ge::OutDataAnchorPtr &out_data_anchor : current_node->GetAllOutDataAnchors()) {
    int64_t output_index = static_cast<int64_t>(out_data_anchor->GetIdx());
    if (std::find(output_non_format_agnostic_index.begin(), output_non_format_agnostic_index.end(), output_index)
        != output_non_format_agnostic_index.end()) {
      FE_LOGD("Output anchor %ld of %s is not format-agnostic!", output_index, node_name.c_str());
      continue;
    }
    uint32_t out_anchor_index = static_cast<uint32_t>(output_index);
    ge::GeTensorDescPtr output_desc_ptr = op_desc_ptr->MutableOutputDesc(out_anchor_index);
    if (output_desc_ptr == nullptr) {
      continue;
    }
    int64_t tensor_format_continuous = 0;
    if (!ge::AttrUtils::GetInt(output_desc_ptr, FORMAT_CONTINUOUS, tensor_format_continuous) ||
        (tensor_format_continuous == 0)) {
      FE_LOGD("Output tensor %u of %s does not have _format_continuous or has a value of 0!", out_anchor_index,
              node_name.c_str());
      continue;
    }
    ge::Format peer_in_primary_format = ge::FORMAT_RESERVED;
    int32_t peer_in_sub_format = 0;
    ge::GeShape peer_in_shape;
    for (auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      int32_t in_anchor_index = peer_in_data_anchor->GetIdx();
      auto peer_in_input_desc = peer_in_data_anchor->GetOwnerNode()->GetOpDesc()->GetInputDescPtr(in_anchor_index);
      if (peer_in_input_desc == nullptr) {
        FE_LOGD("The value of peer_in_input_desc is null.");
        continue;
      }
      ge::Format curr_peer_in_primary_format =
          static_cast<ge::Format>(ge::GetPrimaryFormat(peer_in_input_desc->GetFormat()));
      ge::Format curr_peer_in_sub_format = static_cast<ge::Format>(ge::GetSubFormat(peer_in_input_desc->GetFormat()));

      ge::GeShape curr_peer_in_shape = peer_in_input_desc->GetShape();
      peer_in_shape = curr_peer_in_shape;
      if (peer_in_primary_format == ge::FORMAT_RESERVED) {
        peer_in_primary_format = curr_peer_in_primary_format;
        peer_in_sub_format = curr_peer_in_sub_format;
      } else if (peer_in_primary_format != curr_peer_in_primary_format) {
        FE_LOGW("Peer in anchor's format(%s != %s) (one input-mul output) of %s is not same!",
                ge::TypeUtils::FormatToSerialString(peer_in_primary_format).c_str(),
                ge::TypeUtils::FormatToSerialString(curr_peer_in_primary_format).c_str(), node_name.c_str());
        return NOT_CHANGED;
      }
    }
    if (!CheckFormatCompatible(peer_in_primary_format, output_desc_ptr->GetOriginFormat())) {
      FE_LOGD("Origin format [%s] of output[%u] node [%s, %s] is not compatible with the peers format [%s].",
              FormatToStr(output_desc_ptr->GetOriginFormat()).c_str(), out_anchor_index, op_desc_ptr->GetName().c_str(),
              op_desc_ptr->GetType().c_str(), FormatToStr(peer_in_primary_format).c_str());
      continue;
    }
    auto new_format = static_cast<ge::Format>(ge::GetFormatFromSub(peer_in_primary_format, peer_in_sub_format));
    if (IsHeavyFormat(peer_in_primary_format)) {
      int32_t c0_bit_val = GetC0BitByDataType(output_desc_ptr->GetDataType());
      new_format = static_cast<ge::Format>(ge::GetFormatFromC0(new_format, c0_bit_val));
    }
    output_desc_ptr->SetFormat(new_format);
    output_desc_ptr->SetShape(peer_in_shape);
    auto input_desc = op_desc_ptr->GetInputDesc(out_anchor_index);
    auto input_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(input_desc.GetFormat()));
    if (std::find(input_non_format_agnostic_index.begin(), input_non_format_agnostic_index.end(), out_anchor_index) ==
            input_non_format_agnostic_index.end() &&
        input_primary_format != peer_in_primary_format) {
      FE_LOGD("Input anchor %u of %s is format-agnostic, so we should update its format to %s.", out_anchor_index,
              node_name.c_str(), ge::TypeUtils::FormatToSerialString(new_format).c_str());
      ge::GeShape new_shape;
      int64_t hidden_size = 1;
      int64_t input_size = 1;
      int64_t state_size = -1;
      (void)ge::AttrUtils::GetInt(op_desc_ptr, "hidden_size", hidden_size);
      (void)ge::AttrUtils::GetInt(op_desc_ptr, "input_size", input_size);
      (void)ge::AttrUtils::GetInt(op_desc_ptr, "state_size", state_size);
      CalcShapeExtraAttr extra_attr = {hidden_size, input_size, state_size};
      ShapeAndFormat input_and_output_info = {input_desc.GetOriginShape(),
                                              new_shape,
                                              input_desc.GetOriginFormat(),
                                              peer_in_primary_format,
                                              input_desc.GetDataType(),
                                              peer_in_sub_format,
                                              extra_attr};
      (void)GetShapeAccordingToFormat(input_and_output_info);
      auto input_desc_ptr = op_desc_ptr->MutableInputDesc(out_anchor_index);
      if (input_desc_ptr == nullptr) {
        continue;
      }
      input_desc_ptr->SetFormat(new_format);
      input_desc_ptr->SetShape(new_shape);
    }
  }  // end for
  return SUCCESS;
}

Status HeavyFormatPropagation::UpdateForOneNode(const NodeInfoPtr &node_info) const {
  auto current_node = node_info->current_node;
  auto node_name = current_node->GetName();
  ge::OpDescPtr op_desc_ptr = current_node->GetOpDesc();
  // get format_agnostic
  int64_t format_agnostic = 0;
  (void)ge::AttrUtils::GetInt(op_desc_ptr, FORMAT_AGNOSTIC, format_agnostic);
  // except input non-format agnostic op
  vector<int64_t> input_non_format_agnostic_index;
  (void)ge::AttrUtils::GetListInt(op_desc_ptr, INPUT_FORMAT_AGNOSTIC_EXCEPTION, input_non_format_agnostic_index);
  // except output non-format agnostic op
  vector<int64_t> output_non_format_agnostic_index;
  (void)ge::AttrUtils::GetListInt(op_desc_ptr, OUTPUT_FORMAT_AGNOSTIC_EXCEPTION, output_non_format_agnostic_index);
  if (format_agnostic == static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS)) {
    FE_LOGD("%s Op's format_agnostic is %ld.", node_name.c_str(), format_agnostic);
    // for this node 's all input data
    Status res_input_all_type =
        UpdateInputFormatAndShape(current_node, input_non_format_agnostic_index, output_non_format_agnostic_index);
    if (res_input_all_type == NOT_CHANGED) {
      return SUCCESS;
    } else if (res_input_all_type != SUCCESS) {
      return FAILED;
    }
    // for this node 's all output data
    Status res_output_all_type =
        UpdateOutputFormatAndShape(current_node, input_non_format_agnostic_index, output_non_format_agnostic_index);
    if (res_output_all_type == NOT_CHANGED) {
      return SUCCESS;
    }
  } else if (format_agnostic == static_cast<int64_t>(
             FormatSelectionType::FORMAT_AGNOSTIC_FOR_PAIRED_INPUT_AND_OUTPUT)) {
    FE_LOGD("%s Op's format_agnostic is %ld.", node_name.c_str(), format_agnostic);
    // for this node 's all input data
    if (UpdateInputForPairType(current_node, input_non_format_agnostic_index, output_non_format_agnostic_index) !=
        SUCCESS) {
      return FAILED;
    }
    // for this node 's all output data
    Status res_output_pair_yype =
        UpdateOutputForPairType(current_node, input_non_format_agnostic_index, output_non_format_agnostic_index);
    if (res_output_pair_yype == NOT_CHANGED) {
      return SUCCESS;
    }
  }  // end if
  return SUCCESS;
}

/* We need to take the following topological order into consideration.
 *
 *            Enter          NextIteration
 *               \             /
 *                \           /
 *                 \         /
 *                  \       /
 *                    Merge
 *                      |
 *                      |
 *                     op_a
 *     Topo order is Enter->Merge->NextIteration and they are all set with attr
 *     _format_continuous and _format_agnostic.
 *     So when the format of op_a and Merge is different. If Enter, Merge and
 *     NextIteration are both 5HD or Nz:
 *     1. Enter will be set before Merge, so it will keep 5HD or Nz.
 *     2. And then Merge will be set as op_a's format, we say ND. Merge will become
 *     ND and NextIteration will become ND as well.
 *     So Enter is different from Merge in format which is not allowed.
 *
 *     To solove this case, we need to do the traverse from both topological order and
 *     reversed topological order in which Enter will be set again.
 *
 *   */
Status HeavyFormatPropagation::UpdateInputAndOutputForFormatContinuous() {
  for (const auto &node_info : format_agnostic_nodes_info_) {
    if (UpdateForOneNode(node_info) != SUCCESS) {
      return FAILED;
    }
  }  // end for each node in normal order

  auto size_of_nodes = format_agnostic_nodes_info_.size();
  for (size_t i = size_of_nodes; i > 0; i--) {
    if (UpdateForOneNode(format_agnostic_nodes_info_[i - 1]) != SUCCESS) {
      return FAILED;
    }
  }  // end for each node in normal order
  return SUCCESS;
}

Status HeavyFormatPropagation::PropagateHeavyFormat(const ge::ComputeGraph &graph) {
  FE_TIMECOST_START(PropagateHeavyFormat);
  // use graph.GetAllNodes including sub graph nodes
  for (const ge::NodePtr &current_node : graph.GetAllNodes()) {
    if (current_node == nullptr || current_node->GetOpDesc() == nullptr) {
      continue;
    }

    PropagationInfo propagation_info;
    NodeInfoPtr node_info;
    FE_MAKE_SHARED(node_info = std::make_shared<NodeInfo>(current_node, nullptr, ge::FORMAT_RESERVED, 0, 0,
                                                          propagation_info, INVALID_LAST_NODE_ANCHOR_INDEX, true),
                   return FAILED);

    if (SetOpKernelAndTensorMap(node_info) != SUCCESS) {
      continue;
    }

    if (IsPropagateFromNode(current_node, node_info->current_node_op_kernel_ptr)) {
      RefreshHeavyOpC0Val(current_node->GetOpDesc());
      std::deque<NodeInfoPtr> next_node_queue;
      FE_LOGD("PropagateForwards from heavy node [%s, %s].", current_node->GetName().c_str(),
              current_node->GetType().c_str());

      next_node_queue.push_back(node_info);
      while (!next_node_queue.empty()) {
        NodeInfoPtr node_info_local = next_node_queue.back();
        next_node_queue.pop_back();

        PropagateFarther(node_info_local->last_node_info, node_info_local, next_node_queue);
      }
    }
  }
  if (UpdateInputAndOutputForFormatContinuous() != SUCCESS) {
    format_agnostic_nodes_info_.clear();
    REPORT_FE_ERROR("[GraphOptJdgInst][FmtPropagate] Failed to update input and output for format continuous.");
    return FAILED;
  }
  format_agnostic_nodes_info_.clear();
  FE_TIMECOST_END(PropagateHeavyFormat, "PropagateHeavyFormat during FEGraphOptimizer::OptimizeOriginalGraph");
  return SUCCESS;
}

void HeavyFormatPropagation::RefreshHeavyOpC0Val(const ge::OpDescPtr &op_desc) {
  if (!HasHeavyOpAttr(op_desc)) {
    return;
  }

  for (size_t i = 0; i < op_desc->GetAllInputsSize(); ++i) {
    ge::GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(i);
    SetFormatC0ValOnTensorDesc(tensor_desc);
  }
  for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
    ge::GeTensorDescPtr tensor_desc = op_desc->MutableOutputDesc(i);
    SetFormatC0ValOnTensorDesc(tensor_desc);
  }
}

/* Input Version of GetHeavyFormatIndex */
Status HeavyFormatPropagation::GetHeavyFormatIndex(const NodeInfoPtr &node_info,
                                                   const HeavyFormatInfo &heavy_format_info,
                                                   int32_t &heavy_format_index) const {
  Status ret = heavy_format_selector_ptr_->GetMostSuitableFormatIndex(node_info->current_node_op_kernel_ptr,
                                                                      node_info->current_node, heavy_format_info,
                                                                      node_info->tensor_map, heavy_format_index);

  return ret;
}

bool HeavyFormatPropagation::CheckOriFormatEqual(const ge::NodePtr &next_node,
                                                 const ge::DataAnchorPtr &next_data_anchor,
                                                 const ge::NodePtr &current_node, const TensorInfoPtr &tensor_info_ptr,
                                                 const bool &is_forwards) const {
  if (tensor_info_ptr->heavy_format != ge::FORMAT_FRACTAL_Z && tensor_info_ptr->heavy_format != ge::FORMAT_NC1HWC0 &&
      tensor_info_ptr->heavy_format != ge::FORMAT_FRACTAL_Z_C04) {
    return false;
  }
  if (is_forwards) {
    ge::Format input_format = next_node->GetOpDesc()->GetInputDesc(next_data_anchor->GetIdx()).GetOriginFormat();
    ge::Format output_format = current_node->GetOpDesc()->
        GetOutputDesc(tensor_info_ptr->anchor_index).GetOriginFormat();
    if (current_node->GetOpDesc()->GetType() == NETOUTPUT) {
      output_format = current_node->GetOpDesc()->GetInputDesc(tensor_info_ptr->anchor_index).GetOriginFormat();
    }
    if (input_format != output_format) {
      FE_LOGI("Next node name:[%s], index %d, origin format %s; last node name:[%s], index %d, origin format %s.",
              next_node->GetName().c_str(), next_data_anchor->GetIdx(), FormatToStr(input_format).c_str(),
              current_node->GetName().c_str(), tensor_info_ptr->anchor_index, FormatToStr(output_format).c_str());
      return true;
    }
  } else {
    ge::Format output_format = next_node->GetOpDesc()->GetOutputDesc(next_data_anchor->GetIdx()).GetOriginFormat();
    ge::Format input_format = current_node->GetOpDesc()->
        GetInputDesc(tensor_info_ptr->anchor_index).GetOriginFormat();
    if (output_format != input_format) {
      FE_LOGI("Next node name:[%s], index %d, origin format %s; last node name:[%s], index %d, origin format %s.",
              next_node->GetName().c_str(), next_data_anchor->GetIdx(), FormatToStr(output_format).c_str(),
              current_node->GetName().c_str(), tensor_info_ptr->anchor_index, FormatToStr(input_format).c_str());
      return true;
    }
  }
  return false;
}

Status HeavyFormatPropagation::GetNextNodesInfoForwards(std::deque<NodeInfoPtr> &next_node_queue,
                                                        const ge::InDataAnchorPtr &peer_in_data_anchor,
                                                        const NodeInfoPtr &last_node_info,
                                                        const TensorInfoPtr &tensor_info_ptr) {
  ge::NodePtr next_node = (peer_in_data_anchor == nullptr) ? nullptr : peer_in_data_anchor->GetOwnerNode();
  FE_CHECK_NOTNULL(next_node);
  if (CheckFallbackAclnn(next_node->GetOpDesc()) || IsDumpableOp(next_node->GetOpDesc())) {
    return SUCCESS;
  }
  if (CheckOriFormatEqual(next_node, peer_in_data_anchor, last_node_info->current_node, tensor_info_ptr, true)) {
    return SUCCESS;
  }
  bool is_function_op = !next_node->GetOpDesc()->GetSubgraphInstanceNames().empty();

  if (is_function_op) {
    ge::RefCell key(next_node->GetName(), next_node, ge::NODE_IN, peer_in_data_anchor->GetIdx());

    std::unordered_set<ge::RefCell, ge::RefCellHash> reflections;
    if (reflection_builder_ptr_->LookUpRefRelations(key, reflections) != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("%s Node[%s]: failed to look up relations of input %d edge", kStageGetNextNdInfoFwd.c_str(),
                      next_node->GetName().c_str(), key.in_out_idx);
      return FAILED;
    }
    std::vector<ge::OutDataAnchorPtr> vec_out_data_anchors;
    (void)FeGraphUtils::GetNextSubDatasOutAnchors(reflections, vec_out_data_anchors);

    FE_LOGD("Propagate forward to Function op, next_node: %s, inanchorIdx: %d, data out anchors: %zu.",
            next_node->GetName().c_str(), peer_in_data_anchor->GetIdx(), vec_out_data_anchors.size());
    for (auto &peer_out_anchor : vec_out_data_anchors) {
      ge::NodePtr data_next_node = peer_out_anchor->GetOwnerNode();
      NodeInfoPtr next_node_info;
      CreateNextNodeInfo(data_next_node, last_node_info, tensor_info_ptr->heavy_format, tensor_info_ptr->sub_format,
                         tensor_info_ptr->c0_format,
                         tensor_info_ptr->propagation_info, peer_out_anchor->GetIdx(), false, next_node_info,
                         next_node_queue);
    }
  } else {
    FE_LOGD("Add node %s into queue with reshape type %s.", next_node->GetName().c_str(),
            last_node_info->propagation_info.reshape_type.c_str());
    NodeInfoPtr next_node_info;
    CreateNextNodeInfo(next_node, last_node_info, tensor_info_ptr->heavy_format, tensor_info_ptr->sub_format,
                       tensor_info_ptr->c0_format,
                       tensor_info_ptr->propagation_info, peer_in_data_anchor->GetIdx(), true, next_node_info,
                       next_node_queue);
  }
  return SUCCESS;
}

bool HeavyFormatPropagation::IsAnchorFormatAgnostic(const TensorInfoPtr &tensor_info_ptr, bool is_input,
                                                    int64_t anchor_idex) const {
  if (tensor_info_ptr->format_selection_type != FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS) {
    return true;
  }
  vector<int64_t> non_format_agnostic_index;
  if (is_input) {
    (void)ge::AttrUtils::GetListInt(tensor_info_ptr->op_desc_ptr, INPUT_FORMAT_AGNOSTIC_EXCEPTION,
                                    non_format_agnostic_index);
  } else {
    (void)ge::AttrUtils::GetListInt(tensor_info_ptr->op_desc_ptr, OUTPUT_FORMAT_AGNOSTIC_EXCEPTION,
                                    non_format_agnostic_index);
  }
  if (std::find(non_format_agnostic_index.begin(), non_format_agnostic_index.end(), anchor_idex) !=
      non_format_agnostic_index.end()) {
    return false;
  } else {
    return true;
  }
}

Status HeavyFormatPropagation::GetTensorKernelInfo(const ge::NodePtr &current_node,
                                                   const TensorInfoPtr &tensor_info_ptr,
                                                   const OpKernelInfoPtr &op_kernel_info_ptr,
                                                   InputOrOutputInfoPtr &input_or_output_info) const {
  if (op_kernel_info_ptr == nullptr) {
    FE_LOGD("The op kernel of node %s is null.", current_node->GetName().c_str());
    return FAILED;
  }
  std::map<uint32_t, std::string>::const_iterator iter = tensor_info_ptr->tensor_map.find(
      tensor_info_ptr->anchor_index);
  if (iter == tensor_info_ptr->tensor_map.end()) {
    FE_LOGW("Anchor index %u is not found in the output index map of size %zu.", tensor_info_ptr->anchor_index,
            tensor_info_ptr->tensor_map.size());
    return FAILED;
  }

  if (tensor_info_ptr->is_input) {
    if (op_kernel_info_ptr->GetInputInfoByName(iter->second, input_or_output_info) != SUCCESS) {
      FE_LOGW("Tensor %u named %s is not in kernel %s, %s, which has a size of %zu.", tensor_info_ptr->anchor_index,
              iter->second.c_str(), current_node->GetName().c_str(), current_node->GetType().c_str(),
              op_kernel_info_ptr->GetAllInputInfo().size());
      return FAILED;
    }
  } else {
    if (op_kernel_info_ptr->GetOutputInfoByName(iter->second, input_or_output_info) != SUCCESS) {
      FE_LOGW("Tensor %u name %s isn't in out kernel %s,%s, which size is %zu", tensor_info_ptr->anchor_index,
              iter->second.c_str(), current_node->GetName().c_str(), current_node->GetType().c_str(),
              op_kernel_info_ptr->GetAllOutputInfo().size());
      return FAILED;
    }
  }
  return SUCCESS;
}

bool HeavyFormatPropagation::CheckForwardPropagtionNecessary(const NodeInfoPtr &curr_node_info,
                                                             const ge::OpDescPtr &op_desc_ptr,
                                                             const ge::OutDataAnchorPtr &out_data_anchor,
                                                             const TensorInfoPtr &tensor_info_ptr) {
  if (!IsOutputAvailable(op_desc_ptr, out_data_anchor)) {
    return false;
  }
  int32_t out_anchor_index = out_data_anchor->GetIdx();
  if (!IsAnchorFormatAgnostic(tensor_info_ptr, false, out_anchor_index)) {
    FE_LOGD("Out anchor %d of %s is not format-agnostic!", out_anchor_index, op_desc_ptr->GetName().c_str());
    return false;
  }

  if (tensor_info_ptr->only_to_paired_input_or_output &&
      out_anchor_index != curr_node_info->anchor_index_of_curr_node) {
    FE_LOGD("%s needs paired input and output! Out anchor index %d != %d.", op_desc_ptr->GetName().c_str(),
            out_anchor_index, curr_node_info->anchor_index_of_curr_node);
    return false;
  }

  return true;
}

Status HeavyFormatPropagation::PropagateNormalNodeForwards(const NodeInfoPtr &curr_node_info, int32_t format_index,
                                                           const TensorInfoPtr &tensor_info_ptr,
                                                           std::deque<NodeInfoPtr> &next_node_queue) {
  auto current_node = curr_node_info->current_node;
  auto node_name = current_node->GetName();
  ge::OpDescPtr op_desc_ptr = current_node->GetOpDesc();
  std::deque<NodeInfoPtr> pending_next_nodes;
  for (auto &out_data_anchor : current_node->GetAllOutDataAnchors()) {
    if (!CheckForwardPropagtionNecessary(curr_node_info, op_desc_ptr, out_data_anchor, tensor_info_ptr)) {
      continue;
    }

    tensor_info_ptr->anchor_index = out_data_anchor->GetIdx();
    auto output_desc_ptr = op_desc_ptr->GetOutputDescPtr(tensor_info_ptr->anchor_index);
    if (output_desc_ptr == nullptr) {
      continue;
    }
    if (tensor_info_ptr->format_selection_type != FormatSelectionType::FORMAT_DEPENDS_ON_OP_KERNEL_INFO) {
      tensor_info_ptr->heavy_format = curr_node_info->propagation_info.heavy_format;
      tensor_info_ptr->sub_format = curr_node_info->propagation_info.sub_format;
      tensor_info_ptr->c0_format = curr_node_info->propagation_info.c0_format;
      FE_LOGD("op[name:%s,type:%s]: use heavy_format: %s, sub_format: %d.", node_name.c_str(),
              op_desc_ptr->GetType().c_str(), FormatToStr(tensor_info_ptr->heavy_format).c_str(),
              tensor_info_ptr->sub_format);
    } else if (format_index == -1) {
      /* At the beginning, we don't have the heavy format. */
      auto output_format = output_desc_ptr->GetFormat();
      tensor_info_ptr->heavy_format = static_cast<ge::Format>(ge::GetPrimaryFormat(output_format));
      tensor_info_ptr->sub_format = static_cast<ge::Format>(ge::GetSubFormat(output_format));
      tensor_info_ptr->c0_format = static_cast<int32_t>(ge::GetC0Value(output_format));
      (void)GetTensorKernelInfo(current_node, tensor_info_ptr, curr_node_info->current_node_op_kernel_ptr,
                                tensor_info_ptr->op_kernel_tensor_info);
    } else {
      tensor_info_ptr->sub_format = tensor_info_ptr->propagation_info.sub_format;
      tensor_info_ptr->c0_format = tensor_info_ptr->propagation_info.c0_format;
      if (GetFormatAndDtypeFromOpKernel(current_node, tensor_info_ptr, curr_node_info->current_node_op_kernel_ptr) !=
          SUCCESS) {
        continue;
      }
    }

    auto ori_format = output_desc_ptr->GetOriginFormat();
    if (SetReshapeType(op_desc_ptr, curr_node_info->current_node_op_kernel_ptr, ori_format, tensor_info_ptr) !=
            SUCCESS &&
        tensor_info_ptr->heavy_format != ge::FORMAT_FRACTAL_NZ) {
      FE_LOGD("Failed to get reshape type of op [name: %s, type: %s].", node_name.c_str(), op_desc_ptr->GetType().c_str());
      return SUCCESS;
    }

    Status ret = SetFormats(tensor_info_ptr, curr_node_info->current_node_op_kernel_ptr);
    if (ret != SUCCESS) {
      if (ret == GRAPH_OPTIMIZER_STOP_TRAVERSING_SUBGRAPH) {
        return ret;
      }
      bool ret_invalid_flag = (ret == GRAPH_OPTIMIZER_STOP_TRAVERSING_SCALAR_WEIGHT ||
                               ret == GRAPH_OPTIMIZER_STOP_TRAVERSING_OTHER_ANCHORS);
      if (ret_invalid_flag) {
        FE_LOGD("We stopped traversing from output %d of %s %s due to %u.", tensor_info_ptr->anchor_index,
                op_desc_ptr->GetType().c_str(), node_name.c_str(), ret);
        return SUCCESS;
      }
      continue;
    }

    for (auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      /* Current node will become last node. */
      GetNextNodesInfoForwards(pending_next_nodes, peer_in_data_anchor, curr_node_info, tensor_info_ptr);
    }
  }

  if (JudgePropagationWorthy(current_node, pending_next_nodes, next_node_queue)) {
    return SUCCESS;
  }
  return STOP_PROPAGATION_FROM_WEIGHT;
}

Status HeavyFormatPropagation::SetHeavyFormat(const ge::InDataAnchorPtr &in_data_anchor,
                                              const NodeInfoPtr &curr_node_info, const ge::OpDescPtr &op_desc_ptr,
                                              const int32_t &format_index, const TensorInfoPtr &tensor_info_ptr) const {
  int32_t in_anchor_index = in_data_anchor->GetIdx();
  tensor_info_ptr->anchor_index = in_anchor_index;
  auto input_desc_ptr = op_desc_ptr->GetInputDescPtr(in_anchor_index);
  if (input_desc_ptr == nullptr) {
    FE_LOGD("The input_desc_ptr is null.");
    return FAILED;
  }
  if (tensor_info_ptr->format_selection_type != FormatSelectionType::FORMAT_DEPENDS_ON_OP_KERNEL_INFO) {
    tensor_info_ptr->heavy_format = curr_node_info->propagation_info.heavy_format;
    tensor_info_ptr->sub_format = curr_node_info->propagation_info.sub_format;
    tensor_info_ptr->c0_format = curr_node_info->propagation_info.c0_format;
    tensor_info_ptr->data_type = input_desc_ptr->GetDataType();
  } else if (format_index == -1) {
    /* At the beginning, we don't have the heavy format. */
    auto input_format = input_desc_ptr->GetFormat();
    tensor_info_ptr->heavy_format = static_cast<ge::Format>(ge::GetPrimaryFormat(input_format));
    tensor_info_ptr->sub_format = static_cast<ge::Format>(ge::GetSubFormat(input_format));
    tensor_info_ptr->c0_format = static_cast<int32_t>(ge::GetC0Value(input_format));
  } else {
    FE_LOGD("Invalid parameter, node: %s, format_index: %d", op_desc_ptr->GetName().c_str(), format_index);
    return FAILED;
  }
  return SUCCESS;
}

Status HeavyFormatPropagation::PropagateSubNetoutputForwards(const NodeInfoPtr &curr_node_info, int32_t format_index,
                                                             const TensorInfoPtr &tensor_info_ptr,
                                                             std::deque<NodeInfoPtr> &next_node_queue) {
  auto current_node = curr_node_info->current_node;
  auto node_name = current_node->GetName();

  FE_LOGD("Propagate forward from function net_output: %s.", node_name.c_str());

  // set netoutput in_data_anchor
  ge::OpDescPtr op_desc_ptr = current_node->GetOpDesc();
  for (auto &in_data_anchor : current_node->GetAllInDataAnchors()) {
    if (in_data_anchor != nullptr && in_data_anchor->GetIdx() == curr_node_info->anchor_index_of_curr_node) {
      FE_LOGD("subnet output peer inanchor, node: %s, anchor_idx: %d.", node_name.c_str(), in_data_anchor->GetIdx());
      if (SetHeavyFormat(in_data_anchor, curr_node_info, op_desc_ptr, format_index, tensor_info_ptr) != SUCCESS) {
        continue;
      }

      auto ori_format = op_desc_ptr->GetInputDescPtr(tensor_info_ptr->anchor_index)->GetOriginFormat();
      if (SetReshapeType(op_desc_ptr, curr_node_info->current_node_op_kernel_ptr, ori_format, tensor_info_ptr) !=
              SUCCESS &&
          tensor_info_ptr->heavy_format != ge::FORMAT_FRACTAL_NZ) {
        FE_LOGD("Failed to get reshape type for op[name:%s, type:%s]", op_desc_ptr->GetName().c_str(),
                op_desc_ptr->GetType().c_str());
        return SUCCESS;
      }
      Status ret = SetFormats(tensor_info_ptr, curr_node_info->current_node_op_kernel_ptr);
      if (ret != SUCCESS) {
        if (ret == GRAPH_OPTIMIZER_STOP_TRAVERSING_SUBGRAPH) {
          return ret;
        }
        bool ret_invalid_flag = (ret == GRAPH_OPTIMIZER_STOP_TRAVERSING_SCALAR_WEIGHT ||
                                 ret == GRAPH_OPTIMIZER_STOP_TRAVERSING_OTHER_ANCHORS);
        if (ret_invalid_flag) {
          FE_LOGD("We stopped traversing from output %d of %s %s due to %u.", in_data_anchor->GetIdx(),
                  op_desc_ptr->GetType().c_str(), node_name.c_str(), ret);
          return SUCCESS;
        }
        continue;
      }
      // get function op's all successor in_data_anchor
      // ge::OutDataAnchor::Vistor<ge::InDataAnchorPtr> vec_in_data_anchors;
      std::vector<ge::InDataAnchorPtr> vec_in_data_anchors;
      if (FeGraphUtils::GetNextInAnchorsOfSubNetOutput(current_node, tensor_info_ptr->anchor_index,
                                                       vec_in_data_anchors) != SUCCESS) {
        FE_LOGW("Failed to get subnet output peer inanchor, node: %s, anchorIdx: %d", node_name.c_str(),
                tensor_info_ptr->anchor_index);
        return FAILED;
      }

      FE_LOGD("Get netoutput peer inanchor, node: %s, anchor index: %d, inanchor size: %zu.", node_name.c_str(),
              tensor_info_ptr->anchor_index, vec_in_data_anchors.size());

      for (const auto &peer_in_data_anchor : vec_in_data_anchors) {
        /* Current node -> last node. */
        GetNextNodesInfoForwards(next_node_queue, peer_in_data_anchor, curr_node_info, tensor_info_ptr);
      }
    }
  }
  return SUCCESS;
}

void HeavyFormatPropagation::GetNextNodesInfoBackWards(std::deque<NodeInfoPtr> &next_node_queue,
                                                       const ge::OutDataAnchorPtr &peer_out_anchor,
                                                       const NodeInfoPtr &last_node_info,
                                                       const TensorInfoPtr &tensor_info_ptr) {
  if (peer_out_anchor == nullptr) {
    return;
  }
  auto next_node = peer_out_anchor->GetOwnerNode();
  if (next_node == nullptr) {
    return;
  }
  if (CheckFallbackAclnn(next_node->GetOpDesc()) || IsDumpableOp(next_node->GetOpDesc())) {
    return;
  }
  if (CheckOriFormatEqual(next_node, peer_out_anchor, last_node_info->current_node, tensor_info_ptr, false)) {
    return;
  }
  auto next_node_name = next_node->GetName();
  bool is_function_op = !next_node->GetOpDesc()->GetSubgraphInstanceNames().empty();
  if (is_function_op) {
    FE_LOGD("Propagate backward to Function op, next_node:%s.", next_node_name.c_str());

    ge::RefCell key(next_node_name, next_node, ge::NODE_OUT, peer_out_anchor->GetIdx());

    std::unordered_set<ge::RefCell, ge::RefCellHash> reflections;

    if (reflection_builder_ptr_->LookUpRefRelations(key, reflections) != ge::GRAPH_SUCCESS || reflections.empty()) {
      FE_LOGD("[GraphOptJdgInst][FmtPropagate][GetNextNdInfoBkwd] Node[%s]: failed to look up relation of input edge %d.",
              next_node_name.c_str(), key.in_out_idx);
      return;
    }

    std::vector<ge::InDataAnchorPtr> vec_netoutput_in_anchor;
    if (FeGraphUtils::GetPreSubNetoutputInAnchor(reflections, vec_netoutput_in_anchor) != SUCCESS) {
      FE_LOGD("[GraphOptJdgInst][FmtPropagate][GetNextNdInfoBkwd] Node %s's input data anchor is empty",
              next_node_name.c_str());
      return;
    }

    for (auto &peer_in_anchor : vec_netoutput_in_anchor) {
      ge::NodePtr data_next_node = peer_in_anchor->GetOwnerNode();
      NodeInfoPtr next_node_info;
      CreateNextNodeInfo(data_next_node, last_node_info, tensor_info_ptr->heavy_format, tensor_info_ptr->sub_format,
                         tensor_info_ptr->c0_format,
                         tensor_info_ptr->propagation_info, peer_in_anchor->GetIdx(), true, next_node_info,
                         next_node_queue);
    }
  } else {
    NodeInfoPtr next_node_info;
    CreateNextNodeInfo(next_node, last_node_info, tensor_info_ptr->heavy_format, tensor_info_ptr->sub_format,
                       tensor_info_ptr->c0_format,
                       tensor_info_ptr->propagation_info, peer_out_anchor->GetIdx(), false, next_node_info,
                       next_node_queue);
  }
  return;
}

bool HeavyFormatPropagation::CheckBackwardPropagtionNecessary(const NodeInfoPtr &curr_node_info,
                                                              const ge::InDataAnchorPtr &in_data_anchor,
                                                              const TensorInfoPtr &tensor_info_ptr,
                                                              uint32_t &count_anchor) {
  auto current_node = curr_node_info->current_node;
  auto node_name = current_node->GetName();
  auto op_desc_ptr = current_node->GetOpDesc();
  FE_LOGD("inDataAnchor size is %zu, count is %u, node is %s.", current_node->GetAllInDataAnchors().size(),
          count_anchor++, node_name.c_str());
  if (!IsInputAvailable(op_desc_ptr, in_data_anchor)) {
    return false;
  }
  int32_t in_anchor_index = in_data_anchor->GetIdx();
  if (!IsAnchorFormatAgnostic(tensor_info_ptr, true, in_anchor_index)) {
    FE_LOGD("Out anchor %d of %s is not format-agnostic!", in_anchor_index, node_name.c_str());
    return false;
  }

  bool sub_not = FeGraphUtils::IsSubGraphNetOutput(op_desc_ptr);
  if ((tensor_info_ptr->only_to_paired_input_or_output || sub_not) &&
      in_anchor_index != curr_node_info->anchor_index_of_curr_node) {
    FE_LOGD("%s need paired input and output! In anchor index %d != %d.", node_name.c_str(), in_anchor_index,
            curr_node_info->anchor_index_of_curr_node);
    return false;
  }
  return true;
}

Status HeavyFormatPropagation::PropagateNormalNodeBackwards(const NodeInfoPtr &curr_node_info, int32_t format_index,
                                                            TensorInfoPtr &tensor_info_ptr,
                                                            std::deque<NodeInfoPtr> &next_node_queue) {
  uint32_t count_anchor = 0;
  auto current_node = curr_node_info->current_node;
  ge::OpDescPtr op_desc_ptr = current_node->GetOpDesc();
  auto node_name = current_node->GetName();

  for (auto &in_data_anchor : current_node->GetAllInDataAnchors()) {
    if (!CheckBackwardPropagtionNecessary(curr_node_info, in_data_anchor, tensor_info_ptr, count_anchor)) {
      continue;
    }
    tensor_info_ptr->anchor_index = in_data_anchor->GetIdx();
    auto input_desc_ptr = op_desc_ptr->GetInputDescPtr(tensor_info_ptr->anchor_index);
    if (input_desc_ptr == nullptr) {
      continue;
    }
    if (tensor_info_ptr->format_selection_type != FormatSelectionType::FORMAT_DEPENDS_ON_OP_KERNEL_INFO) {
      tensor_info_ptr->heavy_format = curr_node_info->propagation_info.heavy_format;
      tensor_info_ptr->sub_format = curr_node_info->propagation_info.sub_format;
      tensor_info_ptr->c0_format = curr_node_info->propagation_info.c0_format;
    } else if (format_index == -1) {
      /* At the beginning, we don't have the heavy format. */
      auto input_format = input_desc_ptr->GetFormat();
      tensor_info_ptr->heavy_format = static_cast<ge::Format>(ge::GetPrimaryFormat(input_format));
      tensor_info_ptr->sub_format = static_cast<ge::Format>(ge::GetSubFormat(input_format));
      tensor_info_ptr->c0_format = static_cast<int32_t>(ge::GetC0Value(input_format));
      (void)GetTensorKernelInfo(current_node, tensor_info_ptr, curr_node_info->current_node_op_kernel_ptr,
                                tensor_info_ptr->op_kernel_tensor_info);
    } else {
      tensor_info_ptr->sub_format = tensor_info_ptr->propagation_info.sub_format;
      tensor_info_ptr->c0_format = tensor_info_ptr->propagation_info.c0_format;
      if (GetFormatAndDtypeFromOpKernel(current_node, tensor_info_ptr, curr_node_info->current_node_op_kernel_ptr) !=
          SUCCESS) {
        continue;
      }
    }

    auto ori_format = input_desc_ptr->GetOriginFormat();
    if (SetReshapeType(op_desc_ptr, curr_node_info->current_node_op_kernel_ptr, ori_format, tensor_info_ptr) !=
            SUCCESS &&
        tensor_info_ptr->heavy_format != ge::FORMAT_FRACTAL_NZ) {
      FE_LOGD("Failed to get reshape type for op[name:%s, type:%s]", op_desc_ptr->GetName().c_str(),
              op_desc_ptr->GetType().c_str());
      return SUCCESS;
    }
    Status ret = SetFormats(tensor_info_ptr, curr_node_info->current_node_op_kernel_ptr);
    if (ret != SUCCESS) {
      if (ret == GRAPH_OPTIMIZER_STOP_TRAVERSING_OTHER_ANCHORS) {
        FE_LOGD("We stop traversing from input %d of %s %s.", tensor_info_ptr->anchor_index,
                op_desc_ptr->GetType().c_str(), node_name.c_str());
        return SUCCESS;
      }
      continue;
    }

    ge::OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GetNextNodesInfoBackWards(next_node_queue, peer_out_anchor, curr_node_info, tensor_info_ptr);
  }
  return SUCCESS;
}

Status HeavyFormatPropagation::PropagateSubDataBackwards(const NodeInfoPtr &curr_node_info,
                                                         const TensorInfoPtr &tensor_info_ptr,
                                                         std::deque<NodeInfoPtr> &next_node_queue) {
  auto current_node = curr_node_info->current_node;
  auto node_name = current_node->GetName();
  FE_LOGD("Backward propagation from function data: %s.", node_name.c_str());

  /* 1. Check whether this sub data has been back-propagated */
  int64_t infer_format = -1;
  /* Sub-graph data has only one output desc. */
  auto current_tensor = current_node->GetOpDesc()->MutableOutputDesc(0);
  (void)ge::AttrUtils::GetInt(current_tensor, INFER_FORMAT, infer_format);

  if (infer_format != -1) {
    FE_LOGD("This sub data %s has been back-propagated by format %ld.", node_name.c_str(), infer_format);
    return SUCCESS;
  }
  tensor_info_ptr->heavy_format = curr_node_info->propagation_info.heavy_format;
  tensor_info_ptr->sub_format = curr_node_info->propagation_info.sub_format;
  FE_CHECK_NOTNULL(current_tensor);
  if (!HeavyFormatSelector::IsHeavyFormatConsistentWithOriFormat(current_tensor, tensor_info_ptr->heavy_format,
      current_tensor->GetDataType(), current_node->GetOpDesc())) {
    FE_LOGD("Original format %u is inconsistent with heavy format %u.", current_tensor->GetOriginFormat(),
            tensor_info_ptr->heavy_format);
    return FAILED;
  }

  /* 2. Get the node in front of the function op. */
  ge::OutDataAnchorPtr peer_out_anchor;
  Status ret = FeGraphUtils::GetPreOutAnchorOfSubData(current_node, peer_out_anchor);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][FmtPropagate][PropagateSubDataBkwd] Failed to get pre out anchor for node %s",
                    current_node->GetName().c_str());
    return ret;
  }

  /* 3. Set attr INFER_FORMAT */
  (void)ge::AttrUtils::SetInt(current_tensor, INFER_FORMAT, tensor_info_ptr->heavy_format);

  /* 4. Propate backwards this sub data. */
  GetNextNodesInfoBackWards(next_node_queue, peer_out_anchor, curr_node_info, tensor_info_ptr);

  return SUCCESS;
}

bool HeavyFormatPropagation::JudgeIsFollowReshapeType(const ge::OpDescPtr &op_desc_ptr,
                                                      const OpKernelInfoPtr &op_kernel_info_ptr) const {
  return op_desc_ptr->GetAllInputsSize() == 1 && op_desc_ptr->GetAllOutputsDescSize() == 1 &&
         op_kernel_info_ptr != nullptr && op_kernel_info_ptr->GetOpPattern() == OP_PATTERN_FORMAT_AGNOSTIC;
}

std::string HeavyFormatPropagation::GetPropagationReshapeType(const TensorInfoPtr &tensor_info_ptr,
                                                              const OpKernelInfoPtr &op_kernel_info_ptr,
                                                              const ge::GeTensorDescPtr &current_tensor) const {
  std::string reshape_type;
  if (tensor_info_ptr->op_kernel_tensor_info == nullptr || op_kernel_info_ptr == nullptr) {
    return reshape_type;
  }
  if (!tensor_info_ptr->op_kernel_tensor_info->GetReshapeType().empty()) {
    reshape_type = tensor_info_ptr->op_kernel_tensor_info->GetReshapeType();
  } else if (JudgeIsFollowReshapeType(tensor_info_ptr->op_desc_ptr, op_kernel_info_ptr)) {
    reshape_type = tensor_info_ptr->propagation_info.reshape_type;
  }

  if (op_kernel_info_ptr->GetOpPattern() == OP_PATTERN_REDUCE) {
    vector<int64_t> axis_values;
    (void)ge::AttrUtils::GetListInt(tensor_info_ptr->op_desc_ptr, AXES_ATTR_NAME, axis_values);
    auto ori_format = current_tensor->GetOriginFormat();
    reshape_type = AxisNameUtil::GetReshapeType(ori_format, axis_values);
    if (AxisNameUtil::CheckFormatConveByExpect(ori_format, reshape_type, axis_values) != SUCCESS) {
      REPORT_FE_ERROR("Op[name:%s,type:%s]'s format [%d] is not expected to be converted by default.",
                      tensor_info_ptr->op_desc_ptr->GetName().c_str(), tensor_info_ptr->op_desc_ptr->GetType().c_str(),
                      ori_format);
    }
    FE_LOGD("Op[name:%s,type:%s] reshape type is %s, original format is %d.",
            tensor_info_ptr->op_desc_ptr->GetName().c_str(), tensor_info_ptr->op_desc_ptr->GetType().c_str(),
            reshape_type.c_str(), ori_format);
  }
  (void)ge::AttrUtils::SetStr(current_tensor, ge::ATTR_NAME_RESHAPE_INFER_TYPE, reshape_type);
  return reshape_type;
}

Status HeavyFormatPropagation::SetReshapeType(const ge::OpDescPtr &op_desc_ptr,
                                              const OpKernelInfoPtr &op_kernel_info_ptr,
                                              const ge::Format &ori_format,
                                              const TensorInfoPtr &tensor_info_ptr) const {
  ge::GeTensorDescPtr tensor_desc_ptr = nullptr;
  if (tensor_info_ptr->is_input) {
    tensor_desc_ptr = op_desc_ptr->MutableInputDesc(tensor_info_ptr->anchor_index);
  } else {
    tensor_desc_ptr = op_desc_ptr->MutableOutputDesc(tensor_info_ptr->anchor_index);
  }
  if (op_kernel_info_ptr == nullptr || tensor_info_ptr->op_kernel_tensor_info == nullptr) {
    FE_LOGD("The op_kernel_info_ptr for op[name:%s, type:%s] is null!", op_desc_ptr->GetName().c_str(),
            op_desc_ptr->GetType().c_str());
    if (HasHeavyOpAttr(op_desc_ptr)) {
      std::string reshape_infer_type;
      (void) ge::AttrUtils::GetStr(tensor_desc_ptr, ge::ATTR_NAME_RESHAPE_INFER_TYPE, reshape_infer_type);
      if (!reshape_infer_type.empty()) {
        tensor_info_ptr->propagation_info.reshape_type = reshape_infer_type;
      }
    }
    return SUCCESS;
  }

  if (op_kernel_info_ptr->GetOpPattern() == OP_PATTERN_REDUCE) {
    vector<int64_t> axis_values;
    (void)ge::AttrUtils::GetListInt(op_desc_ptr, AXES_ATTR_NAME, axis_values);
    std::string reshape_type = AxisNameUtil::GetReshapeType(ori_format, axis_values);
    tensor_info_ptr->propagation_info.reshape_type = reshape_type;
    FE_LOGD("Operator [name: %s, type: %s] has a new reshape type of %s, with the original format being %d.", op_desc_ptr->GetName().c_str(),
            op_desc_ptr->GetType().c_str(), reshape_type.c_str(), ori_format);
  } else {
    bool kernel_reshape_empty = tensor_info_ptr->op_kernel_tensor_info->GetReshapeType().empty();
    bool prop_reshape_empty = tensor_info_ptr->propagation_info.reshape_type.empty();

    size_t dim_size = tensor_desc_ptr == nullptr ? 0 : tensor_desc_ptr->GetOriginShape().GetDimNum();
    if (dim_size >= NCHW_DIMENSION_NUM) {
      /* if the dim_size is 4, that means we will not use reshape type, we just clean the
       * propagation reshape type. */
      tensor_info_ptr->propagation_info.reshape_type = "";
      FE_LOGD("Reshape is required for Op[name:%s,type:%s].", tensor_info_ptr->op_desc_ptr->GetName().c_str(),
              tensor_info_ptr->op_desc_ptr->GetType().c_str());
      return SUCCESS;
    }

    if (!kernel_reshape_empty && !prop_reshape_empty &&
        tensor_info_ptr->op_kernel_tensor_info->GetReshapeType() != tensor_info_ptr->propagation_info.reshape_type) {
      FE_LOGD("Op[name:%s,type:%s] reshape type [%s] is not same with propagate reshape type [%s].",
              tensor_info_ptr->op_desc_ptr->GetName().c_str(), tensor_info_ptr->op_desc_ptr->GetType().c_str(),
              tensor_info_ptr->op_kernel_tensor_info->GetReshapeType().c_str(),
              tensor_info_ptr->propagation_info.reshape_type.c_str());
      return FAILED;
    }

    if (!kernel_reshape_empty && prop_reshape_empty) {
      tensor_info_ptr->propagation_info.reshape_type = tensor_info_ptr->op_kernel_tensor_info->GetReshapeType();
      FE_LOGD("Use %s as the propagation reshape type for node %s.",
              tensor_info_ptr->propagation_info.reshape_type.c_str(), op_desc_ptr->GetName().c_str());
      return SUCCESS;
    }

    if (!prop_reshape_empty) {
      FE_LOGD("Op[name:%s, type:%s] has reshape type %s. Op pattern is %d.", op_desc_ptr->GetName().c_str(),
              op_desc_ptr->GetType().c_str(), tensor_info_ptr->propagation_info.reshape_type.c_str(),
              op_kernel_info_ptr->GetOpPattern());
      return SUCCESS;
    }
  }
  FE_LOGD("Use default reshape type for node %s.", op_desc_ptr->GetName().c_str());
  return SUCCESS;
}

bool HeavyFormatPropagation::IsInputAvailable(const ge::OpDescPtr &op_desc_ptr,
                                              const ge::InDataAnchorPtr &in_data_anchor_ptr) const {
  if (in_data_anchor_ptr == nullptr || in_data_anchor_ptr->GetPeerOutAnchor() == nullptr) {
    return false;
  }

  int32_t in_anchor_index = in_data_anchor_ptr->GetIdx();
  if (op_desc_ptr->MutableInputDesc(in_anchor_index) == nullptr) {
    return false;
  }
  return true;
}

bool HeavyFormatPropagation::IsOutputAvailable(const ge::OpDescPtr &op_desc_ptr,
                                               const ge::OutDataAnchorPtr &out_data_anchor_ptr) const {
  if (out_data_anchor_ptr == nullptr) {
    return false;
  }

  int32_t out_anchor_index = out_data_anchor_ptr->GetIdx();
  if (op_desc_ptr->MutableOutputDesc(out_anchor_index) == nullptr) {
    return false;
  }
  return true;
}

bool HeavyFormatPropagation::IsFormatAgnosticAnchor(const ge::NodePtr &current_node,
                                                    const NodeInfoPtr &next_node) const {
  auto next_op_desc = next_node->current_node->GetOpDesc();
  auto heavy_format = next_node->propagation_info.heavy_format;
  /* Check whether the node is format agnostic. If it is, we can keep doing the propagation. */
  if (next_node->format_selection != FormatSelectionType::FORMAT_DEPENDS_ON_OP_KERNEL_INFO) {
    if (next_node->format_selection == FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS) {
      std::vector<int64_t> invalid_anchors;
      auto index = next_node->anchor_index_of_curr_node;
      if (ge::AttrUtils::GetListInt(next_op_desc, INPUT_FORMAT_AGNOSTIC_EXCEPTION, invalid_anchors) &&
          std::find(invalid_anchors.begin(), invalid_anchors.end(), index) != invalid_anchors.end()) {
        FE_LOGD("Stop propagation by heavy format %u from weight %s because its user %s's input %d is in exception.",
                heavy_format, current_node->GetName().c_str(), next_op_desc->GetName().c_str(), index);
        return false;
      }
    }
    return true;
  } else {
    FE_LOGD("Stop propagation of heavy format %d from weight %s because its user %s does not support this format.",
            heavy_format, current_node->GetName().c_str(), next_op_desc->GetName().c_str());
    return false;
  }
}

bool HeavyFormatPropagation::IsHeavyFormatSupported(const ge::NodePtr &current_node,
                                                    const NodeInfoPtr &next_node_info) const {
  auto next_op_desc = next_node_info->current_node->GetOpDesc();
  if (next_node_info->tensor_map.empty()) {
    FE_LOGW("Tensor map for node %s is empty.", next_op_desc->GetName().c_str());
    return false;
  }

  InputOrOutputInfoPtr input_info;
  auto index = next_node_info->anchor_index_of_curr_node;
  std::map<uint32_t, std::string>::const_iterator iter = next_node_info->tensor_map[INPUT_INDEX].find(index);
  if (iter == next_node_info->tensor_map[INPUT_INDEX].end()) {
    FE_LOGW("Cannot find input %u of node %s in tensor map!", index, next_op_desc->GetName().c_str());
    return false;
  }

  Status ret = next_node_info->current_node_op_kernel_ptr->GetInputInfoByName(iter->second, input_info);
  if (ret != SUCCESS) {
    FE_LOGW("Cannot obtain information for input %u of node %s in the ops kernel store.", index, next_op_desc->GetName().c_str());
    return false;
  }

  vector<ge::Format> input_formats;
  vector<uint32_t> input_sub_formats;
  auto heavy_format = next_node_info->propagation_info.heavy_format;
  uint32_t sub_format = static_cast<uint32_t>(next_node_info->propagation_info.sub_format);
  if (sub_format < DEFAULT_SUB_FORMAT) {
    sub_format = DEFAULT_SUB_FORMAT;
  }
  if (format_dtype_querier_ptr_->GetSupportFormatSubFormat(next_node_info->current_node_op_kernel_ptr, input_info,
                                                           next_node_info->current_node, input_formats,
                                                           input_sub_formats, sub_format) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][FmtPropagate][IsHeavyFmtSupted] Failed to obtain format and sub_format for [%s].",
                    next_op_desc->GetName().c_str());
    return false;
  }

  if (IsSuppoertedFormat(heavy_format, sub_format, input_formats, input_sub_formats)) {
    FE_LOGD("Propagation continues, heavy format %s, sub-format %u, weight %s.", FormatToStr(heavy_format).c_str(),
            sub_format, current_node->GetName().c_str());
    return true;
  }
  FE_LOGD(
    "Stop propagation by heavy format %s, sub_format %u from weight %s because its user %s cannot support this "
    "format.",
    FormatToStr(heavy_format).c_str(), sub_format, current_node->GetName().c_str(), next_op_desc->GetName().c_str());
  return false;
}

bool HeavyFormatPropagation::IsNextNodeQualified(const ge::NodePtr &current_node, const NodeInfoPtr &next_node) {
  if (next_node->current_node_op_kernel_ptr == nullptr) {
    FE_LOGD("Check format agnostic attr for node %s.", next_node->current_node->GetName().c_str());
    if (IsFormatAgnosticAnchor(current_node, next_node)) {
      return true;
    }
  } else {
    FE_LOGD("Checking format in op kernel for node %s.", next_node->current_node->GetName().c_str());
    if (IsHeavyFormatSupported(current_node, next_node)) {
      return true;
    }
  }
  return false;
}

/* If one of the peer out node of switch cannot support the heavy format,
 * we stop propagation. */
bool HeavyFormatPropagation::CheckSwitchQualified(const std::vector<NodeInfoPtr> &node_list) {
  for (auto &node_info : node_list) {
    auto node = node_info->current_node;
    for (auto &out_anchor : node->GetAllOutDataAnchors()) {
      if (out_anchor->GetPeerInDataAnchors().empty()) {
        continue;
      }
      for (auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
        if (peer_in_anchor == nullptr) {
          continue;
        }
        auto successor = peer_in_anchor->GetOwnerNode();
        NodeInfoPtr successor_info;
        auto heavy_format = node_info->propagation_info.heavy_format;
        auto sub_format = node_info->propagation_info.sub_format;
        auto c0_format = node_info->propagation_info.c0_format;
        std::deque<NodeInfoPtr> successor_queue;
        CreateNextNodeInfo(successor, node_info, heavy_format, sub_format, c0_format, node_info->propagation_info,
                           peer_in_anchor->GetIdx(), true, successor_info, successor_queue);
        /* Check whether there is a unqualified successor. */
        if (IsNextNodeQualified(node, successor_info)) {
          continue;
        }
        FE_LOGD("The only successor(%s) of format agnostic node %s cannot support heavy format %u.",
                successor->GetName().c_str(), node->GetName().c_str(), heavy_format);
        return false;
      }
    }
  }
  return true;
}

void HeavyFormatPropagation::RollBackFormatAndShape(const ge::NodePtr &node, bool is_sub_graph_weight) const {
  if (!is_sub_graph_weight) {
    return;
  }

  auto output_desc = node->GetOpDesc()->MutableOutputDesc(0);
  if (output_desc == nullptr) {
    return;
  }

  auto sub_format = ge::GetSubFormat(output_desc->GetFormat());
  auto ori_format = output_desc->GetOriginFormat();
  auto ori_shape = output_desc->GetOriginShape();
  output_desc->SetFormat(ori_format);
  output_desc->SetShape(ori_shape);

  auto input_desc = node->GetOpDesc()->MutableInputDesc(0);
  if (input_desc != nullptr) {
    auto new_format = static_cast<ge::Format>(ge::GetFormatFromSub(ori_format, sub_format));
    if (IsHeavyFormat(ori_format)) {
      int32_t c0_bit_val = GetC0BitByDataType(input_desc->GetDataType());
      new_format = static_cast<ge::Format>(ge::GetFormatFromC0(new_format, c0_bit_val));
    }
    input_desc->SetFormat(new_format);
    input_desc->SetShape(ori_shape);
  }

  std::unordered_set<ge::RefCell, ge::RefCellHash> reflections;
  ge::RefCell key(node->GetName(), node, ge::NODE_IN, 0);

  (void)reflection_builder_ptr_->LookUpRefRelations(key, reflections);

  RelationUpdateInfo relation_update_info = {ori_format, sub_format, ori_shape, INFER_FORMAT, 1};

  (void)FeGraphUtils::UpdateFormatOfRelatedEdges(reflections, relation_update_info);
}

bool HeavyFormatPropagation::JudgePropagationWorthy(const ge::NodePtr &node,
                                                    const std::deque<NodeInfoPtr> &pending_next_nodes,
                                                    std::deque<NodeInfoPtr> &next_nodes) {
  vector<NodeInfoPtr> switch_list;
  /* In current stage, we only handle variable/const. And Other nodes we
   * will always do the propagation. */
  bool is_sub_graph_weight = IsConstNodeInSubGraph(node);
  if (IsWeight(node->GetOpDesc()) || is_sub_graph_weight) {
    FE_LOGD("Evaluate if propagation is warranted for weight node %s.", node->GetName().c_str());
    for (auto &next_node : pending_next_nodes) {
      FE_CHECK(next_node == nullptr, FE_LOGW("The next_node is null."), return false);
      auto op_desc = next_node->current_node->GetOpDesc();
      FE_CHECK(op_desc == nullptr, FE_LOGW("op_desc is null."), return false);
      if (op_desc->GetType() == SWITCH) {
        switch_list.emplace_back(next_node);
      }
      if (IsNextNodeQualified(node, next_node)) {
        continue;
      }

      RollBackFormatAndShape(node, is_sub_graph_weight);
      return false;
    }
    FE_LOGD("Propagating through this weight is worthwhile.");
  }

  if (!CheckSwitchQualified(switch_list)) {
    RollBackFormatAndShape(node, is_sub_graph_weight);
    return false;
  }

  for (auto &next_node : pending_next_nodes) {
    next_nodes.emplace_back(next_node);
  }
  return true;
}
}  // namespace fe
