/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "format_selector/manager/format_dtype_setter.h"
#include "common/configuration.h"
#include "common/fe_thread_pool.h"
#include "ops_store/ops_kernel_manager.h"
#include "register/graph_optimizer/fusion_common/fusion_turbo.h"
namespace fe {
namespace {
constexpr size_t kConvMinInputSize = 2;
constexpr uint32_t kMaxDepth = 5;
const std::string kSetFormatDtypeThreadPrefix = "judge_1_";
}

FormatDtypeSetter::FormatDtypeSetter(const std::string& engine_name) : FormatDtypeManagerBase(engine_name) {}

FormatDtypeSetter::~FormatDtypeSetter() {}

Status FormatDtypeSetter::SetSupportFormatDtype(const ge::ComputeGraph& graph) const {
  FE_TIMECOST_START(SetSupportFormatDtype);
  for (const ge::NodePtr &node : graph.GetAllNodes()) {
    Status result = SetSupportFormatDtypeByNode(node);
    if (result != SUCCESS) {
      return result;
    }
    SetNodeDtypeByCustomDtypes(node);
  }
  FE_TIMECOST_END(SetSupportFormatDtype, "SetSupportFormatDtype during FEGraphOptimizer::OptimizeOriginalGraph");
  return SUCCESS;
}

Status FormatDtypeSetter::MultiThreadSetSupportFormatDtype(const ge::ComputeGraph &graph) {
  // set the highest prior imply type of op
  const auto &nodes = graph.GetAllNodes();
  if (nodes.size() < DEFAULT_THREAD_NUM) {
    return SetSupportFormatDtype(graph);
  }
  FE_TIMECOST_START(MultiThreadSetSupportFormatDtype);
  uint32_t thread_num = 8;
  fe::ThreadPool executor(kSetFormatDtypeThreadPrefix + fe::GetCurThreadIdStr(), thread_num);
  std::vector<std::future<Status>> vector_future;
  for (auto &node : graph.GetAllNodes()) {
    std::future<Status> f = executor.commit(FormatDtypeSetter::MultiThreadSetSupportFormatDtypeOneNode, node, this,
                                            ge::GetThreadLocalContext());
    if (!f.valid()) {
      FE_LOGE("[Call][Commit] failed, Future is invalid, node name:%s", node->GetName().c_str());
      return FAILED;
    }
    vector_future.emplace_back(std::move(f));
  }
  for (size_t i = 0; i < vector_future.size(); ++i) {
    Status ret_status = FAILED;
    try {
      ret_status = vector_future[i].get();
    } catch (const std::exception &exp) {
      FE_LOGE("Exception happened, error message is [%s].", exp.what());
      ret_status = FAILED;
    }
    if (ret_status != SUCCESS) {
      REPORT_FE_ERROR("Multi-thread set op format dtype failed, graph %s", graph.GetName().c_str());
      return ret_status;
    }
  }
  FE_TIMECOST_END(MultiThreadSetSupportFormatDtype,
      "MultiThreadSetSupportFormatDtype during FEGraphOptimizer::OptimizeOriginalGraph");
  return SUCCESS;
}

Status FormatDtypeSetter::MultiThreadSetSupportFormatDtypeOneNode(ge::NodePtr &node_ptr,
    FormatDtypeSetter *format_dtype_setter_ptr, const ge::GEThreadLocalContext &ge_context) {
  FE_CHECK_NOTNULL(format_dtype_setter_ptr);
  ge::GetThreadLocalContext() = ge_context;
  Status result = format_dtype_setter_ptr->SetSupportFormatDtypeByNode(node_ptr);
  if (result != SUCCESS) {
    return result;
  }
  format_dtype_setter_ptr->SetNodeDtypeByCustomDtypes(node_ptr);
  return SUCCESS;
}

void FormatDtypeSetter::JudgeFirstLayerConv(const ge::NodePtr& node, const ge::OpDescPtr& op_desc) const {
  bool enable_small_channel = Configuration::Instance(GetEngineName()).IsEnableSmallChannel();
  bool first_lay_conv2d = ((op_desc->GetType() == CONV2D || op_desc->GetType() == DEPTHWISECONV2D) &&
                          node->GetAllInDataAnchors().size() >= kConvMinInputSize &&
                          enable_small_channel);
  if (!first_lay_conv2d) {
    FE_LOGD("This op %s is not first layer conv2d.", op_desc->GetName().c_str());
    return;
  }
  JudgeFirstLayerConvForInfer(node, op_desc);
  JudgeFirstLayerConvForTrain(node, op_desc);
}

void FormatDtypeSetter::JudgeFirstLayerConvForInfer(const ge::NodePtr& node, const ge::OpDescPtr& op_desc) const {
  auto in_data_anchor = node->GetInDataAnchor(0);
  auto data_node = FusionTurbo::GetPeerOutNode(node, 0);
  auto weight_node = FusionTurbo::GetPeerOutNode(node, 1);
  if (data_node != nullptr && weight_node != nullptr) {
    auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    /* Peer in data anchors is 1 is the necessary condition. Because
      * we do not have transdata supporting NC1HWC0_C04 to 4D. */
    if (peer_out_anchor == nullptr) { FE_LOGE("peer_out_anchor is nullptr."); return; }
    if (peer_out_anchor->GetPeerInDataAnchors().size() != 1) {
      return;
    }
    ge::OpDescPtr father_op_desc = data_node->GetOpDesc();
    ge::OpDescPtr weight_op_desc = weight_node->GetOpDesc();
    std::string father_op_type = father_op_desc->GetType();

    bool weight_qualified = CheckWeightTypeQualified(weight_node, CONSTANT);
    FE_LOGD("This op %s predecessor on the first edge is %s.", op_desc->GetNamePtr(),
            father_op_desc->GetNamePtr());
    /* First layer means the op in front of  */
    FE_LOGD("Weight qualification result is %u, father_op_type is %s.",
            weight_qualified, father_op_type.c_str());
    if (weight_qualified && father_op_type == AIPP) {
      FE_LOGD("This op %s is the first layer conv.", op_desc->GetName().c_str());
      (void)ge::AttrUtils::SetBool(op_desc, IS_FIRST_LAYER_CONV_FOR_OP, true);
    }
    if (father_op_type != "BNInferenceD") {
      return;
    }
    auto input_anchor = data_node->GetInDataAnchor(0);
    auto grandfather_node = FusionTurbo::GetPeerOutNode(data_node, 0);
    if (grandfather_node != nullptr && input_anchor != nullptr) {
      auto peer_anchor = input_anchor->GetPeerOutAnchor();
      if (peer_anchor != nullptr && peer_anchor->GetPeerInDataAnchors().size() != 1) {
        return;
      }
      ge::OpDescPtr input_op_desc = grandfather_node->GetOpDesc();
      if (input_op_desc->GetType() == AIPP) {
        FE_LOGD("This node %s is the first layer conv.", op_desc->GetName().c_str());
        (void)ge::AttrUtils::SetBool(op_desc, IS_FIRST_LAYER_CONV_FOR_OP, true);
      }
    }
  }
}

void FormatDtypeSetter::JudgeFirstLayerConvForTrain(const ge::NodePtr& node, const ge::OpDescPtr& op_desc) const {
  FE_LOGD("JudgeFirstLayerConvForTrain enter.");
  uint32_t depth = 0;
  ge::NodePtr data = nullptr;
  if (!GetFirstLayerConv2DInputData(node, data, depth)) {
    FE_LOGD("GetFirstLayerConv2DInputData false.");
    return;
  }
  if (data == nullptr) {
    FE_LOGD("GetFirstLayerConv2DInputData data == nullptr");
    return;
  }
  FE_LOGD("JudgeFirstLayerConvForTrain data:%s.", data->GetName().c_str());
  ge::NodePtr weight = nullptr;
  if (!GetFirstLayerConv2DWeight(node, weight)) {
    FE_LOGD("GetFirstLayerConv2DWeight false.");
    return;
  }
  FE_LOGD("JudgeFirstLayerConvForTrain weight:%s.", weight->GetName().c_str());
  ge::NodePtr conv2d_dw = nullptr;
  depth = 0;
  bool flag = false;
  GetFirstLayerConv2DDW(data, conv2d_dw, node, depth, flag);
  if (!flag) {
    FE_LOGD("GetFirstLayerConv2DDW flag = %d.", flag);
    return;
  }
  if (conv2d_dw == nullptr) {
    FE_LOGD("GetFirstLayerConv2DDW conv2d_dw = nullptr");
    return;
  }
  FE_LOGD("This node %s and relate dw %s is the first layer conv and dw.",
          op_desc->GetName().c_str(), conv2d_dw->GetName().c_str());
  (void)ge::AttrUtils::SetBool(conv2d_dw->GetOpDesc(), IS_FIRST_LAYER_CONV, true);
  (void)ge::AttrUtils::SetBool(op_desc, IS_FIRST_LAYER_CONV, true);
}

bool FormatDtypeSetter::GetFirstLayerConv2DInputData(const ge::NodePtr &node, ge::NodePtr &data, uint32_t depth) const {
  if (IsRootGraphData(node->GetType())) {
    data = node;
    FE_LOGD("GetFirstLayerConv2DInputData data:%s.", data->GetName().c_str());
    return true;
  }
  ge::InDataAnchorPtr in_data_anchor = nullptr;
  bool has_no_father = false;
  CheckHasNoFather(true, 0, node, in_data_anchor, has_no_father);
  if (has_no_father) {
    data = node;
    FE_LOGD("GetFirstLayerConv2DInputData data:%s.", data->GetName().c_str());
    return true;
  }
  if (depth > kMaxDepth) {
    FE_LOGD("GetFirstLayerConv2DInputData depth:%u overflow.", depth);
    return false;
  }
  depth++;
  auto father = in_data_anchor->GetPeerOutAnchor()->GetOwnerNode();
  if (father->GetType() == CONV2D) {
    FE_LOGD("Father of %s is already Conv2D(%s).", node->GetName().c_str(),
            father->GetName().c_str());
    return false;
  }
  return GetFirstLayerConv2DInputData(father, data, depth);
}

bool FormatDtypeSetter::GetFirstLayerConv2DWeight(const ge::NodePtr &node, ge::NodePtr &weight) const {
  auto weight_node = FusionTurbo::GetPeerOutNode(node, 1);
  if (weight_node == nullptr) {
    FE_LOGD("There is no weight.");
    return false;
  }
  if (weight_node->GetType() == VARIABLE) {
    FE_LOGD("Get weight node:%s.", weight_node->GetName().c_str());
    weight = weight_node;
    return true;
  }
  FE_LOGD("Final don't have variable type weight.");
  return false;
}

void FormatDtypeSetter::GetFirstLayerConv2DDW(const ge::NodePtr &data, ge::NodePtr &conv2d_dw,
                                              const ge::NodePtr &conv2d_node,
                                              uint32_t depth, bool &find_flag) const {
  if (find_flag) {
    FE_LOGD("GetFirstLayerConv2DDW get conv2ddw conv2d_node:%s", conv2d_dw->GetName().c_str());
    return;
  }
  if (data != nullptr) {
    if (data->GetType() == CONV2DDW || data->GetType() == CONV2DDWD) {
      conv2d_dw = data;
      find_flag = true;
      FE_LOGD("Get conv2ddw conv2d_node:%s.", conv2d_dw->GetName().c_str());
      return;
    }
    if (depth > kMaxDepth) {
      FE_LOGD("GetFirstLayerConv2DDW depth:%u overflow", depth);
      return;
    }
    depth++;
    for (auto out_node : data->GetOutDataNodes()) {
      if (out_node == conv2d_node) {
        continue;
      }
      GetFirstLayerConv2DDW(out_node, conv2d_dw, conv2d_node, depth, find_flag);
    }
  }
}

Status FormatDtypeSetter::SetSupportFormatDtypeByNode(ge::NodePtr node_ptr) const {
    HeavyFormatInfo heavy_foramt_info;
    return SetSupportFormatDtypeByNode(node_ptr, heavy_foramt_info);
}

Status FormatDtypeSetter::SetSupportFormatDtypeByNode(ge::NodePtr node_ptr,
                                                      const HeavyFormatInfo& heavy_format_info) const {
  // 1. check the node_ptr and the op_desc_ptr
  FE_CHECK_NOTNULL(node_ptr);
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);

  // 2. check the imply_type
  string op_name = op_desc_ptr->GetName();
  string op_type = op_desc_ptr->GetType();
  int64_t imply_type = -1;
  if (!ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, imply_type)) {
    FE_LOGD("Op[name=%s,type=%s]: get the attribute FE_IMPLY_TYPE unsuccessful.", op_name.c_str(), op_type.c_str());
    return SUCCESS;
  }

  // 3. get op_kernel_info_ptr by op_impl_type and op_type
  OpImplType op_impl_type = static_cast<OpImplType>(imply_type);
  OpKernelInfoPtr op_kernel_info_ptr =
          OpsKernelManager::Instance(GetEngineName()).GetOpKernelInfoByOpType(op_impl_type, op_type);
  if (op_kernel_info_ptr == nullptr) {
    FE_LOGW("Engine[%s] not support op_impl_type[%ld].", GetEngineName().c_str(), op_impl_type);
    return SUCCESS;
  }

  /* 4. Judge whether it's first layer convolution.
   * The first layer convolution is able to use feature C0 = 4 to boost the
   * computation. */
  JudgeFirstLayerConv(node_ptr, op_desc_ptr);

  // 5. save the support format and data_types
  bool is_dynamic_check = IsOpDynamicImpl(op_desc_ptr);
  FormatDtypeSelectorBasePtr selector = GetSelector(op_kernel_info_ptr, is_dynamic_check);
  FE_CHECK_NOTNULL(selector);
  return selector->SetSupportFormatDtype(op_kernel_info_ptr, heavy_format_info, node_ptr, is_dynamic_check);
}

void FormatDtypeSetter::SetNodeDtypeByCustomDtypes(const ge::NodePtr &node_ptr) const {
  FE_CHECK(node_ptr == nullptr, FE_LOGW("Node is null."), return);
  ge::OpDescPtr op_desc = node_ptr->GetOpDesc();
  FE_CHECK(op_desc == nullptr, FE_LOGW("Op desc is null."), return);

  // only deal with the non aicore node
  if (ge::AttrUtils::HasAttr(op_desc, FE_IMPLY_TYPE)) {
    return;
  }

  OpCustomizeDtype op_cust_dtype;
  if (!Configuration::Instance(GetEngineName()).GetCustomizeDtypeByOpName(op_desc->GetName(), op_cust_dtype)) {
    if (!Configuration::Instance(GetEngineName()).GetCustomizeDtypeByOpType(op_desc->GetType(), op_cust_dtype)) {
      return;
    }
  }

  SetTensorDtype(op_desc, true, op_cust_dtype.input_dtypes);
  SetTensorDtype(op_desc, false, op_cust_dtype.output_dtypes);
}

void FormatDtypeSetter::SetTensorDtype(const ge::OpDescPtr &op_desc, const bool &is_input,
                                       const std::vector<ge::DataType> &tensor_dtypes) const {
  for (size_t i = 0; i < tensor_dtypes.size(); i++) {
    ge::DataType data_type = tensor_dtypes[i];
    if (data_type == ge::DT_UNDEFINED) {
      continue;
    }
    ge::GeTensorDescPtr tensor_desc_ptr = is_input ? op_desc->MutableInputDesc(i) : op_desc->MutableOutputDesc(i);
    if (tensor_desc_ptr == nullptr) {
      continue;
    }
    tensor_desc_ptr->SetDataType(data_type);
    FE_LOGI("%s[%zu]'s data type of op[%s, %s] has been set to [%s].", is_input ? "Input" : "Output",
            i, op_desc->GetName().c_str(), op_desc->GetType().c_str(),
            ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
  }
}
}  // namespace fe
