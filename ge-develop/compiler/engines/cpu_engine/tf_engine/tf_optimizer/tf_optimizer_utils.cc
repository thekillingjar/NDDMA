/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tf_optimizer_utils.h"

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/ge_tensor.h"
#include "util/util.h"
#include "aicpu_graph_optimizer/graph_optimizer_utils.h"
#include "util/tf_util.h"
#include "util/constant.h"
#include "common/sgt_slice_type.h"
#include "ir2tf/ir2tf_parser_factory.h"

namespace {
const std::string kValidateShapeAttr = "validate_shape";
const std::string kPlaceholderOp = "PlaceHolder";
const std::string kEndOp = "End";
const std::string kUseLocking = "use_locking";
const std::string kDecodeJpegOp = "DecodeJpeg";
const std::string kDecodeAndCropJpegOp = "DecodeAndCropJpeg";
const char *const kAICPUEngineName = "DNN_VM_AICPU";
const char *const kAICPUKernelLibName = "aicpu_tf_kernel";
constexpr int64_t kConstdim = 3;
const std::string kImgFormat = "dst_img_format";
const std::string kBatchMatMulV2Op = "BatchMatMulV2";
}

namespace aicpu {
ge::Status TfVariableGraph::CreateAssign(ge::GeTensorDesc &first_src_tensor_desc,
                                         ge::GeTensorDesc &second_src_tensor_desc,
                                         ge::GeTensorDesc &dst_tensor_desc,
                                         ge::ComputeGraph &graph,
                                         ge::NodePtr &assign_node) {
  static std::atomic<uint32_t> index{0};
  std::string assign_op_name = "Assign_Insert" + std::to_string(index++);
  const auto assign_op_owner = ge::OperatorFactoryImpl::CreateOperator(assign_op_name.c_str(), "Assign");
  CHECK_RES_BOOL(
      !(assign_op_owner.IsEmpty()), ge::FAILED,
      AICPU_REPORT_INNER_ERR_MSG("[Verify][CheckParam] Get op from OperatorFactory failed, type: Assign"));
  AICPUE_LOGI("Get op from OperatorFactory succeeded. opType: Assign");
  auto assign_op = ge::OpDescUtils::GetOpDescFromOperator(assign_op_owner);
  AICPU_CHECK_NOTNULL(assign_op)
  ge::OpDescUtilsEx::SetType(assign_op, "AssignExt");

  AICPU_CHECK_RES_WITH_LOG(assign_op->UpdateInputDesc(0, first_src_tensor_desc),
      "Call ge::op::Assign::UpdateInputDesc function failed, input[0].")
  AICPU_CHECK_RES_WITH_LOG(assign_op->UpdateInputDesc(1, second_src_tensor_desc),
      "Call ge::op::Assign::UpdateInputDesc function failed, input[1].")
  AICPU_CHECK_RES_WITH_LOG(assign_op->UpdateOutputDesc(0, dst_tensor_desc),
      "Call ge::op::Assign::UpdateOutputDesc function failed, output[0].")

  CHECK_RES_BOOL(ge::AttrUtils::SetBool(assign_op, kValidateShapeAttr, false),
      ge::FAILED,
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::SetBool failed to set attr[%s], op[%s]",
          kValidateShapeAttr.c_str(), assign_op->GetName().c_str()))
  CHECK_RES_BOOL(ge::AttrUtils::SetBool(assign_op, kUseLocking, false),
      ge::FAILED,
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::SetBool failed to set attr[%s], op[%s]",
          kUseLocking.c_str(), assign_op->GetName().c_str()))
  assign_node = graph.AddNode(assign_op);
  AICPU_CHECK_NOTNULL(assign_node)
  return ge::SUCCESS;
}

ge::Status TfVariableGraph::CreateIdentity(ge::GeTensorDesc &src_tensor_desc,
                                           ge::GeTensorDesc &dst_tensor_desc,
                                           ge::ComputeGraph &graph,
                                           ge::NodePtr &identity_node) {
  static std::atomic<uint32_t> index{0};
  std::string identity_op_name = "Identity_Insert" + std::to_string(index++);
  const auto identity_op_owner = ge::OperatorFactoryImpl::CreateOperator(identity_op_name.c_str(), "Identity");
  CHECK_RES_BOOL(
      !(identity_op_owner.IsEmpty()), ge::FAILED,
      AICPU_REPORT_INNER_ERR_MSG("[Verify][CheckParam] Get op from OperatorFactory failed, type: Identity"));
  AICPUE_LOGI("Get op from OperatorFactory succeeded. opType: Identity");
  auto identity_op = ge::OpDescUtils::GetOpDescFromOperator(identity_op_owner);
  AICPU_CHECK_NOTNULL(identity_op)
  ge::OpDescUtilsEx::SetType(identity_op, "IdentityExt");
  AICPU_CHECK_RES_WITH_LOG(identity_op->UpdateInputDesc(0, src_tensor_desc),
      "Call ge::op::Identity::UpdateInputDesc function failed, input[0].")
  AICPU_CHECK_RES_WITH_LOG(identity_op->UpdateOutputDesc(0, dst_tensor_desc),
      "Call ge::op::Identity::UpdateOutputDesc function failed, output[0].")
  identity_node = graph.AddNode(identity_op);
  AICPU_CHECK_NOTNULL(identity_node)
  return ge::SUCCESS;
}

ge::Status TfVariableGraph::CreateVariable(ge::GeTensorDesc &dst_tensor_desc,
                                           ge::ComputeGraph &graph,
                                           ge::NodePtr &variable_node) {
  static std::atomic<uint32_t> op_name_index{0};
  std::string variable_op_name = "TemporaryVariable_Insert" + std::to_string(op_name_index++);
  const auto variable_op_owner = ge::OperatorFactoryImpl::CreateOperator(variable_op_name.c_str(), "TemporaryVariable");
  CHECK_RES_BOOL(
      !(variable_op_owner.IsEmpty()), ge::FAILED,
      AICPU_REPORT_INNER_ERR_MSG("[Verify][CheckParam] Get op from OperatorFactory failed, type: TemporaryVariable"));
  AICPUE_LOGI("Get op from OperatorFactory succeeded. opType: TemporaryVariable");
  auto variable_op = ge::OpDescUtils::GetOpDescFromOperator(variable_op_owner);
  AICPU_CHECK_NOTNULL(variable_op)
  ge::OpDescUtilsEx::SetType(variable_op, "VariableExt");
  std::vector<int64_t> values;
  for (const auto value : dst_tensor_desc.GetShape().GetDims()) {
    AICPUE_LOGD("Dim value is [%ld].", value);
    values.push_back(value);
  }
  variable_op->SetAttr("shape", ge::GeAttrValue::CreateFrom<std::vector<int64_t>>(values));
  std::string current_time = CurrentTimeInStr();
  static int index = 0;
  std::string shared_name;
  shared_name.append("Variable").append("_").append(Stringcat(index));
  index++;
  variable_op->SetAttr("container", ge::GeAttrValue::CreateFrom<std::string>("Variable"));
  variable_op->SetAttr("shared_name", ge::GeAttrValue::CreateFrom<std::string>(shared_name.c_str()));
  AICPU_CHECK_NOTNULL(variable_op)

  AICPU_CHECK_RES_WITH_LOG(variable_op->UpdateOutputDesc(0, dst_tensor_desc),
      "Call ge::op::TemporaryVariable::UpdateOutputDesc function failed, output[0].")

  variable_node = graph.AddNode(variable_op);
  AICPU_CHECK_NOTNULL(variable_node)
  return ge::SUCCESS;
}

ge::Status TfVariableGraph::InsertVariable(ge::OutDataAnchorPtr &src_anchor,
                                           const ge::InDataAnchorPtr &dst_anchor,
                                           ge::NodePtr &variable_node) {
  AICPUE_LOGD("Enter InsertVariable Func.");
  AICPU_CHECK_NOTNULL(variable_node)
  AICPU_CHECK_NOTNULL(src_anchor)
  AICPU_CHECK_NOTNULL(dst_anchor)
  AICPU_CHECK_RES_WITH_LOG(ge::GraphUtils::RemoveEdge(src_anchor, dst_anchor),
      "Call ge::GraphUtils::RemoveEdge function failed to remove edge between"
      " op[%s] and op[%s]", src_anchor->GetOwnerNode()->GetName().c_str(),
      dst_anchor->GetOwnerNode()->GetName().c_str())
  ge::OutDataAnchorPtr y_anchor_from_variable = variable_node->GetOutDataAnchor(0);
  AICPUE_LOGD("Get Variable outDataAnchor success.");
  AICPU_CHECK_NOTNULL(y_anchor_from_variable)
  AICPU_CHECK_RES_WITH_LOG(ge::GraphUtils::AddEdge(y_anchor_from_variable, dst_anchor),
      "Call ge::GraphUtils::AddEdge function failed to add edge between"
      " op[%s] and [%s]", variable_node->GetName().c_str(),
      dst_anchor->GetOwnerNode()->GetName().c_str())
  return ge::SUCCESS;
}

ge::Status TfVariableGraph::InsertAssign(ge::OutDataAnchorPtr &src_anchor,
                                         const ge::InDataAnchorPtr &dst_anchor,
                                         const ge::NodePtr &node_ptr,
                                         ge::NodePtr &assign_node) {
  AICPUE_LOGI("Enter InsertAssign Func.");
  AICPU_CHECK_NOTNULL(src_anchor)
  AICPU_CHECK_NOTNULL(dst_anchor)
  AICPU_CHECK_NOTNULL(assign_node)
  ge::OutDataAnchorPtr y_anchor_from_variable = node_ptr->GetOutDataAnchor(0);
  AICPUE_LOGD("Get Variable outDataAnchor success.");
  AICPU_CHECK_RES_WITH_LOG(ge::GraphUtils::RemoveEdge(y_anchor_from_variable, dst_anchor),
      "Call ge::GraphUtils::RemoveEdge function failed to remove edge between op[%s] and op[%s]",
      node_ptr->GetName().c_str(), dst_anchor->GetOwnerNode()->GetName().c_str())
  ge::InDataAnchorPtr input1_anchor_from_assign = assign_node->GetInDataAnchor(0);
  AICPUE_LOGD("Get Assign inDataAnchor success.");
  ge::InDataAnchorPtr input2_anchor_from_assign = assign_node->GetInDataAnchor(1);
  AICPUE_LOGD("Get Assign inDataAnchor success.");
  ge::OutDataAnchorPtr output_anchor_from_assign = assign_node->GetOutDataAnchor(0);
  AICPUE_LOGD("Get Assign outDataAnchor success.");
  AICPU_CHECK_RES_WITH_LOG(ge::GraphUtils::AddEdge(src_anchor, input2_anchor_from_assign),
      "Call ge::GraphUtils::AddEdge function failed to add edge between op[%s] and op[%s]",
      src_anchor->GetOwnerNode()->GetName().c_str(), assign_node->GetName().c_str())
  AICPU_CHECK_RES_WITH_LOG(ge::GraphUtils::AddEdge(y_anchor_from_variable, input1_anchor_from_assign),
      "Call ge::GraphUtils::AddEdge function failed to add edge between op[%s] and op[%s]",
      node_ptr->GetName().c_str(), assign_node->GetName().c_str())
  AICPU_CHECK_RES_WITH_LOG(ge::GraphUtils::AddEdge(output_anchor_from_assign, dst_anchor),
      "Call ge::GraphUtils::AddEdge function failed to add edge between op[%s] and op[%s]",
      assign_node->GetName().c_str(), dst_anchor->GetOwnerNode()->GetName().c_str())
  return ge::SUCCESS;
}

ge::Status TfVariableGraph::InsertIdentity(ge::OutDataAnchorPtr &src_anchor,
                                           const ge::InDataAnchorPtr &dst_anchor,
                                           ge::NodePtr &identity_node) {
  AICPUE_LOGI("Enter InsertIdentity Func.");
  AICPU_CHECK_RES_WITH_LOG(ge::GraphUtils::RemoveEdge(src_anchor, dst_anchor),
      "Call ge::GraphUtils::RemoveEdge function failed to remove edge between op[%s] and op[%s]",
      src_anchor->GetOwnerNode()->GetName().c_str(),
      dst_anchor->GetOwnerNode()->GetName().c_str())
  AICPU_CHECK_NOTNULL(identity_node)
  ge::InDataAnchorPtr x_anchor_from_identity = identity_node->GetInDataAnchor(0);
  AICPUE_LOGD("Get Identity inDataAnchor success.");
  ge::OutDataAnchorPtr y_anchor_from_identity = identity_node->GetOutDataAnchor(0);
  AICPUE_LOGD("Get Identity outDataAnchor success.");
  AICPU_CHECK_RES_WITH_LOG(ge::GraphUtils::AddEdge(src_anchor, x_anchor_from_identity),
      "Call ge::GraphUtils::AddEdge function failed to add edge between op[%s] and op[%s]",
      src_anchor->GetOwnerNode()->GetName().c_str(),
      identity_node->GetName().c_str())
  AICPU_CHECK_RES_WITH_LOG(ge::GraphUtils::AddEdge(y_anchor_from_identity, dst_anchor),
      "Call ge::GraphUtils::AddEdge function failed to add edge between op[%s] and op[%s]",
      identity_node->GetName().c_str(),
      dst_anchor->GetOwnerNode()->GetName().c_str())
  return ge::SUCCESS;
}

ge::Status TfVariableGraph::CreateAndInsertVariable(ge::OutDataAnchorPtr &src_anchor,
                                                    const ge::InDataAnchorPtr &dst_anchor,
                                                    ge::NodePtr &variable_node,
                                                    ge::ComputeGraph &graph) {
  AICPUE_LOGI("Enter CreateAndInsertVariable Func.");
  const auto src_node = src_anchor->GetOwnerNodeBarePtr();
  AICPU_CHECK_NOTNULL(src_node)
  const ge::OpDescPtr src_op = src_node->GetOpDesc();
  AICPU_CHECK_NOTNULL(src_op)
  int idx = ge::AnchorUtils::GetIdx(src_anchor);
  // -1 is invalid index
  CHECK_RES_BOOL((idx != -1), ge::FAILED,
      AICPU_REPORT_INNER_ERR_MSG("invalid output anchor index[%d], op[%s].",
          idx, src_op->GetType().c_str()))
  uint32_t src_anchor_idx = static_cast<uint32_t>(idx);

  // dst_anchor already check not null and valid
  const ge::OpDescPtr dst_op = dst_anchor->GetOwnerNode()->GetOpDesc();
  AICPU_CHECK_NOTNULL(dst_op);
  uint32_t dst_anchor_idx = static_cast<uint32_t>(ge::AnchorUtils::GetIdx(dst_anchor));

  ge::GeTensorDesc src_tensor_desc = src_op->GetOutputDesc(src_anchor_idx);
  ge::GeTensorDesc dst_tensor_desc = dst_op->GetInputDesc(dst_anchor_idx);
  AICPUE_LOGD("Enter CreateVariable Func.");
  AICPU_CHECK_RES_WITH_LOG(CreateVariable(dst_tensor_desc, graph, variable_node),
      "Call TfVariableGraph::CreateVariable function failed, op[%s].",
      src_op->GetName().c_str())
  AICPUE_LOGD("Enter InsertVariable Func.");
  AICPU_CHECK_RES_WITH_LOG(InsertVariable(src_anchor, dst_anchor, variable_node),
      "Call TfVariableGraph::InsertVariable function failed, op[%s].",
      src_op->GetName().c_str())
  return ge::SUCCESS;
}

ge::Status TfVariableGraph::CreateAndInsertIdentity(ge::OutDataAnchorPtr &src_anchor,
                                                    const ge::InDataAnchorPtr &dst_anchor,
                                                    ge::ComputeGraph &graph) {
  AICPUE_LOGI("Enter CreateAndInsertIdentity Func.");
  const auto src_node = src_anchor->GetOwnerNodeBarePtr();
  AICPU_CHECK_NOTNULL(src_node)
  const ge::OpDescPtr src_op = src_node->GetOpDesc();
  AICPU_CHECK_NOTNULL(src_op)

  int idx = ge::AnchorUtils::GetIdx(src_anchor);
  CHECK_RES_BOOL((idx != -1), ge::FAILED,
      AICPU_REPORT_INNER_ERR_MSG("invalid output anchor index[%d], op[%s].",
                               idx, src_op->GetType().c_str()))
  uint32_t src_anchor_idx = static_cast<uint32_t>(idx);
  ge::GeTensorDesc src_tensor_desc = src_op->GetOutputDesc(src_anchor_idx);

  // dst_anchor already check not null and valid
  const ge::OpDescPtr dst_op = dst_anchor->GetOwnerNode()->GetOpDesc();
  uint32_t dst_anchor_idx = static_cast<uint32_t>(ge::AnchorUtils::GetIdx(dst_anchor));
  ge::GeTensorDesc dst_tensor_desc = dst_op->GetInputDesc(dst_anchor_idx);

  ge::NodePtr identity_node;
  AICPUE_LOGD("Enter CreateIdentity Func.");
  AICPU_CHECK_RES_WITH_LOG(CreateIdentity(src_tensor_desc, dst_tensor_desc, graph, identity_node),
      "Call TfVariableGraph::CreateIdentity function failed, op[%s]",
      src_op->GetName().c_str())
  AICPUE_LOGD("Enter CreateIdentity Func.");
  AICPU_CHECK_RES_WITH_LOG(InsertIdentity(src_anchor, dst_anchor, identity_node),
      "Call TfVariableGraph::InsertIdentity function failed, op[%s]",
      src_op->GetName().c_str())
  return ge::SUCCESS;
}

ge::Status TfVariableGraph::CreateAndInsertAssign(ge::OutDataAnchorPtr &src_anchor,
                                                  const ge::InDataAnchorPtr &dst_anchor,
                                                  const ge::NodePtr &node_ptr,
                                                  ge::ComputeGraph &graph) {
  AICPUE_LOGI("Enter CreateAndInsertAssign Func.");
  const auto src_node = src_anchor->GetOwnerNodeBarePtr();
  AICPU_CHECK_NOTNULL(src_node)
  const ge::OpDescPtr src_op = src_node->GetOpDesc();
  AICPU_CHECK_NOTNULL(src_op)
  const ge::OpDescPtr variable_op = node_ptr->GetOpDesc();
  AICPU_CHECK_NOTNULL(variable_op)

  int idx = ge::AnchorUtils::GetIdx(src_anchor);
  CHECK_RES_BOOL((idx != -1), ge::FAILED,
      AICPU_REPORT_INNER_ERR_MSG("invalid output anchor index[%d], op[%s].",
                               idx, src_op->GetType().c_str()))
  uint32_t src_anchor_idx = static_cast<uint32_t>(idx);

  ge::GeTensorDesc tensor_desc1 = variable_op->GetOutputDesc(0);
  ge::GeTensorDesc tensor_desc2 = src_op->GetOutputDesc(src_anchor_idx);
  ge::NodePtr assign_node;
  AICPUE_LOGD("Enter CreateAssign Func.");
  AICPU_CHECK_RES_WITH_LOG(CreateAssign(tensor_desc1, tensor_desc2, tensor_desc1, graph, assign_node),
      "Call TfVariableGraph::CreateAssign function failed, op[%s]",
      src_op->GetName().c_str())
  AICPUE_LOGD("Enter InsertAssign Func.");
  AICPU_CHECK_RES_WITH_LOG(InsertAssign(src_anchor, dst_anchor, node_ptr, assign_node),
      "Call TfVariableGraph::InsertAssign function failed, op[%s]",
      src_op->GetName().c_str())
  return ge::SUCCESS;
}

// generate tf graph
ge::Status TfVariableGraph::GenerateTfVariableGraph(ge::ComputeGraph &graph, bool &have_insert_variable) {
  for (const ge::NodePtr &cur_node : graph.GetDirectNode()) {
    AICPU_CHECK_NOTNULL(cur_node)
    ge::OpDescPtr cur_op_desc_ptr = cur_node->GetOpDesc();
    AICPU_CHECK_NOTNULL(cur_op_desc_ptr)
    std::string op_type = cur_op_desc_ptr->GetType();
    // if op type is variable, replace it with assign and variable
    if (op_type == kPlaceholderOp) {
      // get all output node and anchors
      for (auto &anchor : cur_node->GetOutDataNodesAndAnchors()) {
        auto peer_op_type = anchor.first->GetOpDesc()->GetType();
        std::shared_ptr<Ir2tfBaseParser> parser = Ir2tfParserFactory::Instance().CreateIRParser(peer_op_type);
        AICPU_CHECK_NOTNULL_ERRCODE(parser, ErrorCode::INPUT_PARAM_NULL);
        std::set<std::string> refinput_set;
        (void)parser->GetRefInputSet(peer_op_type, refinput_set);
        uint32_t anchor_index = static_cast<uint32_t>(anchor.second->GetIdx());
        std::string input_name =
            anchor.first->GetOpDesc()->GetInputNameByIndex(anchor_index);
        if (IsRefTensorDesc(anchor.first->GetOpDesc()->GetInputDesc(anchor_index)) ||
            (refinput_set.find(input_name) != refinput_set.end())) {
          AICPUE_LOGD("Current op type is [%s]. Insert Variable Op and replace it with TF Variable and Assign.",
                      op_type.c_str());
          AICPUE_LOGD("Before get src_anchor from PlaceHolder.");
          // placeholder op just have one output edge
          ge::OutDataAnchorPtr src_anchor = cur_node->GetOutDataAnchor(0);
          AICPU_CHECK_NOTNULL(src_anchor)

          AICPUE_LOGD("After get src_anchor from PlaceHolder.");
          // get idx
          int idx = ge::AnchorUtils::GetIdx(src_anchor);
          AICPUE_LOGD("InDataAnchor Idx value is [%d].", idx);
          // -1 is invalid index
          if (idx == -1) {
            AICPUE_LOGD("Current op's inDataAnchor Idx value -1 is invalid.");
            continue;
          }
          AICPUE_LOGD("Before get dst_anchor from PlaceHolder.");
          bool sgt_flag = false;
          ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
          (void)GraphOptimizerUtils::CheckIsFftsPlus(cur_op_desc_ptr, slice_info_ptr, sgt_flag);
          // see all others anchor
          for (const auto &dst_anchor : src_anchor->GetPeerInDataAnchors()) {
            ge::NodePtr variable_node;
            AICPU_CHECK_RES_WITH_LOG(CreateAndInsertVariable(src_anchor, dst_anchor, variable_node, graph),
                "Call TfVariableGraph::CreateAndInsertVariable function failed, op[%s].",
                cur_node->GetName().c_str())
            // Variable op just have one output edge
            ge::OutDataAnchorPtr out_data_anchor = variable_node->GetOutDataAnchor(0);
            AICPU_CHECK_NOTNULL(out_data_anchor)
            ge::InDataAnchorPtr dst_variable_anchor = (*out_data_anchor->GetPeerInDataAnchors().begin());
            AICPU_CHECK_RES_WITH_LOG(CreateAndInsertAssign(src_anchor, dst_variable_anchor, variable_node, graph),
                "Call TfVariableGraph::CreateAndInsertAssign function failed, op[%s].",
                cur_node->GetName().c_str())
            if ((sgt_flag) && (slice_info_ptr != nullptr)) {
              (void)variable_node->GetOpDesc()->SetExtAttr(kAttrNameSgtStruct, slice_info_ptr);
              (void)ge::AttrUtils::SetInt(variable_node->GetOpDesc(), kAttrNameThreadScopeId,
                                          slice_info_ptr->thread_scope_id);
            }
            have_insert_variable = true;
          }
        }
      }
    }
    if (op_type == kEndOp) {
      AICPUE_LOGD("Current op type is [%s]. Insert Variable Op and replace it with TF Variable and Assign.",
                  op_type.c_str());
      for (auto &node_and_anchor : cur_node->GetInDataNodesAndAnchors()) {
        if (IsRefTensorDesc(node_and_anchor.first->GetOpDesc()->GetOutputDesc(
            static_cast<uint32_t>(node_and_anchor.second->GetIdx())))) {
          ge::NodePtr node = (*cur_node->GetInNodes().begin());
          AICPU_CHECK_NOTNULL(node)
          ge::OutDataAnchorPtr src_anchor = node->GetOutDataAnchor(0);
          AICPU_CHECK_NOTNULL(src_anchor)
          ge::InDataAnchorPtr dst_anchor = cur_node->GetInDataAnchor(0);
          AICPU_CHECK_NOTNULL(dst_anchor)
          CreateAndInsertIdentity(src_anchor, dst_anchor, graph);
        }
      }
    }
  }
  return ge::SUCCESS;
}

bool TfVariableGraph::IsRefTensorDesc(const ge::GeTensorDesc &tensor_desc) {
  return !tensor_desc.GetRefPortIndex().empty();
}

ge::OpDescPtr TfTransposeGraph::CreateConstNode() {
  ge::GeTensorPtr tensor = MakeShared<ge::GeTensor>();
  auto shape = ge::GeShape(std::vector<int64_t>({kConstdim}));
  tensor->MutableTensorDesc().SetDataType(ge::DT_INT32);
  tensor->MutableTensorDesc().SetFormat(ge::FORMAT_ND);
  tensor->MutableTensorDesc().SetShape(shape);
  tensor->MutableTensorDesc().SetOriginDataType(ge::DT_INT32);
  tensor->MutableTensorDesc().SetOriginFormat(ge::FORMAT_ND);
  tensor->MutableTensorDesc().SetOriginShape(shape);
  // 2, 0, 1 for HWC->CHW
  std::vector<int32_t> tensor_value = {2, 0, 1};
  tensor->SetData(reinterpret_cast<uint8_t *>(tensor_value.data()),
                  kConstdim * sizeof(int32_t));
  ge::OpDescPtr const_desc = ge::OpDescUtils::CreateConstOpZeroCopy(tensor);
  if (const_desc == nullptr) {
    return nullptr;
  }

  auto const_out = const_desc->MutableOutputDesc(0);
  const_out->SetOriginShape(shape);
  const_out->SetOriginDataType(ge::DT_INT32);
  const_out->SetOriginFormat(ge::FORMAT_ND);
  ge::TensorUtils::SetRealDimCnt(*const_out, 1U);
  return const_desc;
}

ge::NodePtr TfTransposeGraph::CreateTransposeNode(
    ge::ComputeGraph &graph, const std::string &name,
    const ge::GeTensorDesc tensor_desc) {
  ge::OpDescPtr perm_const_desc = CreateConstNode();
  if (perm_const_desc == nullptr) {
    AICPU_REPORT_INNER_ERR_MSG("CreateConstNode failed");
    return nullptr;
  }

  ge::OpDescPtr op_desc = MakeShared<ge::OpDesc>(name.c_str(), "Transpose");
  if (op_desc == nullptr) {
    AICPU_REPORT_INNER_ERR_MSG("[New][OpDesc Transpose] failed");
    return nullptr;
  }
  op_desc->SetOpEngineName(kAICPUEngineName);
  op_desc->SetOpKernelLibName(kAICPUKernelLibName);

  ge::Status ret = op_desc->AddInputDesc("x", tensor_desc);
  if (ret != ge::GRAPH_SUCCESS) {
    AICPU_REPORT_INNER_ERR_MSG("Call ge::AddInputDesc  x failed ret[%d]", ret);
    return nullptr;
  }
  
  ret = op_desc->AddInputDesc("perm", perm_const_desc->GetOutputDesc(0));
  if (ret != ge::GRAPH_SUCCESS) {
    AICPU_REPORT_INNER_ERR_MSG("Call ge::AddInputDesc  x failed ret[%d]", ret);
    return nullptr;
  }
  ret = op_desc->AddOutputDesc(tensor_desc);
  if (ret != ge::GRAPH_SUCCESS) {
    AICPU_REPORT_INNER_ERR_MSG("Call ge::AddOutputDesc failed ret[%d]", ret);
    return nullptr;
  }
  /* The output shape of reshape depends on the weight value of its
   * second input which name is "shape". */
  op_desc->SetOpInferDepends({"perm"});
  ge::NodePtr transpose = graph.AddNode(op_desc);
  if (transpose == nullptr) {
    return nullptr;
  }
  ge::NodePtr perm_node = graph.AddNode(perm_const_desc);
  if (perm_node == nullptr) {
    AICPU_REPORT_INNER_ERR_MSG("Call ge::AddNode const failed");
    return nullptr;
  }
  ret = ge::GraphUtils::AddEdge(perm_node->GetOutDataAnchor(0),
                                transpose->GetInDataAnchor(1));
  if (ret != ge::GRAPH_SUCCESS) {
    AICPU_REPORT_INNER_ERR_MSG("Call ge::AddOutputDesc failed ret[%d]", ret);
    return nullptr;
  }
  return transpose;
}

ge::Status TfTransposeGraph::CreateAndInsertTransposeNode(
    ge::ComputeGraph &graph, const ge::NodePtr &node) {
  ge::OutDataAnchorPtr src_anchor = node->GetOutDataAnchor(0);
  if (src_anchor == nullptr) {
    return ge::GRAPH_FAILED;
  }
  std::string name = node->GetName() + "/transpose";
  ge::GeTensorDesc desc = node->GetOpDesc()->GetOutputDesc(0);
  ge::NodePtr transpose = CreateTransposeNode(graph, name, desc);
  if (transpose == nullptr) {
    return ge::GRAPH_FAILED;
  }
  for (auto peer_in_anchor : src_anchor->GetPeerInDataAnchors()) {
    if (peer_in_anchor == nullptr) {
      continue;
    }
    AICPU_CHECK_RES_WITH_LOG(
        ge::GraphUtils::RemoveEdge(src_anchor, peer_in_anchor),
        "call RemoveEdge between %s and %s", node->GetName().c_str(),
        peer_in_anchor->GetOwnerNode()->GetName().c_str());
    AICPU_CHECK_RES_WITH_LOG(
        ge::GraphUtils::AddEdge(transpose->GetOutDataAnchor(0), peer_in_anchor),
        "call AddEdge between transpose and %s",
        peer_in_anchor->GetOwnerNode()->GetName().c_str());
  }
  AICPU_CHECK_RES_WITH_LOG(
      ge::GraphUtils::AddEdge(src_anchor, transpose->GetInDataAnchor(0)),
      "call AddEdge between transpose %s and transpose",
      node->GetName().c_str());
  return ge::GRAPH_SUCCESS;
}

ge::Status TfTransposeGraph::GenerateTfTransposeGraph(ge::ComputeGraph &graph) {
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    std::string type = node->GetType();
    if ((type != kDecodeJpegOp) && (type != kDecodeAndCropJpegOp)) {
      continue;
    }
    const std::string *format = ge::AttrUtils::GetStr(node->GetOpDesc(), kImgFormat);
    if ((format == nullptr) || (*format != "CHW")) {
      continue;
    }
    AICPUE_LOGD("insert transpose for [%s]", type.c_str());
    AICPU_CHECK_RES_WITH_LOG(CreateAndInsertTransposeNode(graph, node),
                             "op[%s] insert transpose fail", type.c_str());
    // for infershape,
    (void)ge::AttrUtils::SetStr(node->GetOpDesc(), kImgFormat, "HWC");
  }
  return ge::SUCCESS;
}

ge::Status TfBatchMatMulV2Graph::CreateAndInsertBiasConstNode(const ge::NodePtr &dst_node, ge::ComputeGraph &graph) {
  const ge::OpDescPtr dst_op = dst_node->GetOpDesc();
  AICPU_CHECK_NOTNULL(dst_op);
  ge::GeTensorDesc dst_tensor_desc = dst_op->GetInputDesc("bias");
  if (dst_tensor_desc.IsValid() == ge::GRAPH_SUCCESS) {
    return ge::SUCCESS;
  }
  ge::GeTensorDesc y_desc = dst_op->GetOutputDesc(0);
  dst_tensor_desc.SetDataType(y_desc.GetDataType());
  dst_tensor_desc.SetFormat(ge::FORMAT_ND);
  dst_tensor_desc.SetShape(ge::GeShape({0}));

  dst_tensor_desc.SetOriginDataType(y_desc.GetDataType());
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  dst_tensor_desc.SetOriginShape(ge::GeShape({0}));
  AICPU_CHECK_RES_WITH_LOG(dst_op->UpdateInputDesc("bias", dst_tensor_desc),
                           "Call bias UpdateInputDesc failed, op name[%s].", dst_node->GetName().c_str())

  // Create const node with an empty tensor
  ge::GeTensorPtr tensor = MakeShared<ge::GeTensor>();
  AICPU_CHECK_NOTNULL(tensor);
  tensor->MutableTensorDesc().SetDataType(y_desc.GetDataType());
  tensor->MutableTensorDesc().SetFormat(ge::FORMAT_ND);
  tensor->MutableTensorDesc().SetShape(ge::GeShape({0}));
  tensor->MutableTensorDesc().SetOriginDataType(y_desc.GetDataType());
  tensor->MutableTensorDesc().SetOriginFormat(ge::FORMAT_ND);
  tensor->MutableTensorDesc().SetOriginShape(ge::GeShape({0}));
  ge::OpDescPtr const_op_desc = ge::OpDescUtils::CreateConstOpZeroCopy(tensor);
  AICPU_CHECK_NOTNULL(const_op_desc);
  AICPU_CHECK_RES_WITH_LOG(const_op_desc->UpdateOutputDesc(0, dst_tensor_desc),
                           "Call const UpdateOutputDesc failed, output[0].")
  // Insert const node to the current graph
  const auto const_node = graph.AddNode(const_op_desc);
  AICPU_CHECK_NOTNULL(const_node);
  const auto in_anchor = dst_node->GetInDataAnchor(dst_op->GetInputIndexByName("bias"));
  AICPU_CHECK_NOTNULL(in_anchor);
  AICPU_CHECK_RES_WITH_LOG(ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), in_anchor),
                           "Call AddEdge function failed, op name[%s].", dst_node->GetName().c_str())
  return ge::SUCCESS;
}

ge::Status TfBatchMatMulV2Graph::CreateAndInsertOffsetWConstNode(const ge::NodePtr &dst_node, ge::ComputeGraph &graph) {
  const ge::OpDescPtr dst_op = dst_node->GetOpDesc();
  AICPU_CHECK_NOTNULL(dst_op);
  ge::GeTensorDesc dst_tensor_desc = dst_op->GetInputDesc("offset_w");
  if (dst_tensor_desc.IsValid() == ge::GRAPH_SUCCESS) {
    return ge::SUCCESS;
  }

  dst_tensor_desc.SetDataType(ge::DT_INT8);
  dst_tensor_desc.SetFormat(ge::FORMAT_ND);
  dst_tensor_desc.SetShape(ge::GeShape({0}));

  dst_tensor_desc.SetOriginDataType(ge::DT_INT8);
  dst_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  dst_tensor_desc.SetOriginShape(ge::GeShape({0}));
  AICPU_CHECK_RES_WITH_LOG(dst_op->UpdateInputDesc("offset_w", dst_tensor_desc),
                           "Call bias UpdateInputDesc failed, op name[%s].", dst_node->GetName().c_str())

  // Create const node with an empty tensor
  ge::GeTensorPtr tensor = MakeShared<ge::GeTensor>();
  AICPU_CHECK_NOTNULL(tensor);
  tensor->MutableTensorDesc().SetDataType(ge::DT_INT8);
  tensor->MutableTensorDesc().SetFormat(ge::FORMAT_ND);
  tensor->MutableTensorDesc().SetShape(ge::GeShape({0}));
  tensor->MutableTensorDesc().SetOriginDataType(ge::DT_INT8);
  tensor->MutableTensorDesc().SetOriginFormat(ge::FORMAT_ND);
  tensor->MutableTensorDesc().SetOriginShape(ge::GeShape({0}));
  ge::OpDescPtr const_op_desc = ge::OpDescUtils::CreateConstOpZeroCopy(tensor);
  AICPU_CHECK_NOTNULL(const_op_desc);
  AICPU_CHECK_RES_WITH_LOG(const_op_desc->UpdateOutputDesc(0, dst_tensor_desc),
                           "Call const UpdateOutputDesc failed, output[0].")
  // Insert const node to the current graph
  const auto const_node = graph.AddNode(const_op_desc);
  AICPU_CHECK_NOTNULL(const_node);
  const auto in_anchor = dst_node->GetInDataAnchor(dst_op->GetInputIndexByName("offset_w"));
  AICPU_CHECK_NOTNULL(in_anchor);
  AICPU_CHECK_RES_WITH_LOG(ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), in_anchor),
                           "Call AddEdge function failed, op name[%s].", dst_node->GetName().c_str())
  return ge::SUCCESS;
}

ge::Status TfBatchMatMulV2Graph::GenerateBatchMatMulV2Graph(ge::ComputeGraph &graph) {
  for (const ge::NodePtr &cur_node : graph.GetDirectNode()) {
    if (cur_node == nullptr) {
      continue;
    }
    std::string type = cur_node->GetType();
    if (type != kBatchMatMulV2Op) {
      continue;
    }
    AICPUE_LOGI("insert const node for node[%s]", cur_node->GetName().c_str());
    AICPU_CHECK_RES_WITH_LOG(CreateAndInsertBiasConstNode(cur_node, graph),
                             "node[%s] insert bias constnode fail", cur_node->GetName().c_str());
    AICPU_CHECK_RES_WITH_LOG(CreateAndInsertOffsetWConstNode(cur_node, graph),
                             "node[%s] insert bias constnode fail", cur_node->GetName().c_str());
  }
  return ge::SUCCESS;
}

ge::Status TfFixInvalidNodeName::TfGraphFixInvalidNodeName(ge::ComputeGraph &graph) {
  for (const ge::NodePtr &cur_node : graph.GetDirectNode()) {
    if (cur_node != nullptr) {
      const ge::OpDescPtr op_desc = cur_node->GetOpDesc();
      AICPU_CHECK_NOTNULL(op_desc);
      std::string nodeName = op_desc->GetName();
      if ((nodeName.size() >= 1) && (nodeName[0] == '/')) {
        nodeName.erase(0, 1);
        op_desc->SetName(nodeName);
      }
    }
  }
  return ge::SUCCESS;
}
}
