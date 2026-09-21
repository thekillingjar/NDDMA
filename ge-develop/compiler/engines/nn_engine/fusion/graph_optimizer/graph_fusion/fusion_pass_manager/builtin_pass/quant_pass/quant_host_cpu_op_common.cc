/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/quant_pass/quant_host_cpu_op_common.h"
#include <atomic>
#include <sstream>
#include <vector>
#include "common/fe_inner_error_codes.h"
#include "common/math_util.h"
#include "common/util/op_info_util.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
namespace fe {
const float kFloatEsp = 10e-6f;

uint64_t GetHostCpuAtomicId() {
  static std::atomic<uint64_t> global_trans_atomic_id(0);
  return global_trans_atomic_id.fetch_add(1, std::memory_order_relaxed);
}

Status CreateNewRequantHostCpuOp(const string &op_type, const ge::NodePtr &dequant_node, const float &scale_quant,
                                 ge::ComputeGraph &graph, vector<ge::NodePtr> &new_nodes) {
  FE_LOGI("Create new requant host op for dequant node [%s].", dequant_node->GetName().c_str());
  std::stringstream op_name_temp;
  // The atomic id of trans nodes must be unique.(start from 0)
  op_name_temp << op_type << "_" << GetHostCpuAtomicId();
  ge::OpDescPtr requant_host_cpu_op;
  FE_MAKE_SHARED(requant_host_cpu_op = std::make_shared<ge::OpDesc>(op_name_temp.str(), op_type), return FAILED);
  // 1. Get the const of dequant
  vector<ge::GeTensorPtr> weights = ge::OpDescUtils::MutableWeights(dequant_node);
  if (weights.size() < 1) {
    FE_LOGI("Get weights not successfully. Node name: %s", dequant_node->GetName().c_str());
    return NOT_CHANGED;
  }
  ge::GeTensorPtr scale_output = weights[0];
  FE_CHECK(scale_output == nullptr, REPORT_FE_ERROR("[GraphOpt][Quant][CrtNewRqtHsCpuOp] scaleInput is nullptr."),
           return PARAM_INVALID);

  // 2. Add input and output desc of new host op
  auto &scale_input_of_new_host_cpu_op = scale_output->GetTensorDesc();
  requant_host_cpu_op->AddInputDesc(REQUANT_HOST_CPU_OP_INPUT, scale_input_of_new_host_cpu_op);
  FE_CHECK_NOTNULL(requant_host_cpu_op->MutableInputDesc(0));
  requant_host_cpu_op->MutableInputDesc(0)->SetOriginDataType(scale_input_of_new_host_cpu_op.GetDataType());
  requant_host_cpu_op->MutableInputDesc(0)->SetOriginFormat(
      static_cast<ge::Format>(ge::GetPrimaryFormat(scale_input_of_new_host_cpu_op.GetFormat())));
  requant_host_cpu_op->MutableInputDesc(0)->SetOriginShape(scale_input_of_new_host_cpu_op.GetShape());

  /* The output original datatype is fp32, and it actually is fp32, so
   * we don't need to specify this. */
  requant_host_cpu_op->AddOutputDesc(REQUANT_HOST_CPU_OP_OUTPUT, scale_input_of_new_host_cpu_op);
  FE_CHECK_NOTNULL(requant_host_cpu_op->MutableOutputDesc(0));
  requant_host_cpu_op->MutableOutputDesc(0)->SetOriginFormat(
      static_cast<ge::Format>(ge::GetPrimaryFormat(scale_input_of_new_host_cpu_op.GetFormat())));
  requant_host_cpu_op->MutableOutputDesc(0)->SetOriginShape(scale_input_of_new_host_cpu_op.GetShape());

  auto requant_node = graph.AddNode(requant_host_cpu_op);
  FE_CHECK_NOTNULL(requant_node);
  new_nodes.emplace_back(requant_node);
  /* 3. Add edges between dequant_scale_weight <-> new_host_cpu_op:0 */
  ge::InDataAnchorPtr dequant_input_anchor = dequant_node->GetInDataAnchor(DEQUANT_SCALE_INDEX_OF_DEQUANT_OP);
  FE_CHECK_NOTNULL(dequant_input_anchor);
  ge::OutDataAnchorPtr dequant_scale_peer_out_anchor = dequant_input_anchor->GetPeerOutAnchor();
  FE_CHECK_NOTNULL(dequant_scale_peer_out_anchor);
  auto requant_host_cpu_input_anchor = requant_node->GetInDataAnchor(DEQUANT_SCALE_INDEX_OF_REQUANT_OP);
  if (ge::GraphUtils::AddEdge(dequant_scale_peer_out_anchor, requant_host_cpu_input_anchor) != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOpt][Quant][CrtNewRqtHsCpuOp] Failed to add edge between dequant scale %s and new host cpu op %s.",
        dequant_node->GetName().c_str(), requant_node->GetName().c_str());
    return FAILED;
  }

  if (ge::GraphUtils::RemoveEdge(dequant_scale_peer_out_anchor, dequant_input_anchor) != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Quant][CrtNewRqtHsCpuOp] Failed to remove edge between dequant scale %s and its weight.",
                    dequant_node->GetName().c_str());
    return FAILED;
  }

  auto requant_host_cpu_output_anchor = requant_node->GetOutDataAnchor(0);
  if (ge::GraphUtils::AddEdge(requant_host_cpu_output_anchor, dequant_input_anchor) != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOpt][Quant][CrtNewRqtHsCpuOp] Add Edge between new host cpu op %s and dequant scale %s failed.",
        requant_node->GetName().c_str(), dequant_node->GetName().c_str());
    return FAILED;
  }
  // aipp + quant + conv2d: new modify deq_scale as q_scale
  float new_scale = scale_quant;
  float scale_rate = 0;
  if (ge::AttrUtils::GetFloat(dequant_node->GetOpDesc(), kQuantScaleRate, scale_rate) &&
      !FloatEqual(scale_rate, 0)) {
    new_scale = new_scale / scale_rate;
    FE_LOGD("Change dequant scale from:%f to %f.", scale_quant, new_scale);
  }
  /* Set Attr of Scale Quant */
  (void)ge::AttrUtils::SetFloat(requant_node->GetOpDesc(), QUANT_SCALE, new_scale);
  return SUCCESS;
}

Status PadShapeTo4Dim(const ge::Format &filter_format, const std::vector<int64_t> &filter_dims,
                      std::vector<int64_t> &filter_dims4_d) {
  size_t size_of_filter = filter_dims.size();
  FE_LOGD("sizeOfFilter is %zu.", size_of_filter);
  for (size_t i = 0; i <= LAST_AXIS_INDEX; i++) {
    if (i < size_of_filter) {
      FE_LOGD("dim [%zu] is %ld.", i, filter_dims.at(i));
      filter_dims4_d.emplace_back(filter_dims.at(i));
    } else {
      if (filter_format == ge::FORMAT_NCHW) {
        filter_dims4_d.emplace_back(1);
      } else if (filter_format == ge::FORMAT_HWCN) {
        filter_dims4_d.insert(filter_dims4_d.cbegin(), 1);
      } else if (filter_format == ge::FORMAT_NHWC) {
        filter_dims4_d.insert(filter_dims4_d.cbegin() + 1, 1);
      } else if (filter_format == ge::FORMAT_ND) {
        filter_dims4_d.emplace_back(0);
      } else {
        REPORT_FE_ERROR("[GraphOpt][Quant][PadShpTo4Dim] cannot pad shape for format %s.",
                        ge::TypeUtils::FormatToSerialString(filter_format).c_str());
        return FAILED;
      }
    }
  }

  if (!filter_dims4_d.empty() && filter_dims4_d.size() >= LAST_AXIS_INDEX) {
    FE_LOGD("Quant bias optimize, filter_format is %s, weight shape is [%ld, %ld, %ld, %ld].",
            ge::TypeUtils::FormatToSerialString(filter_format).c_str(), filter_dims4_d[NCHW_DIM_N],
            filter_dims4_d[NCHW_DIM_C], filter_dims4_d[NCHW_DIM_H], filter_dims4_d[NCHW_DIM_W]);
  }
  return SUCCESS;
}

bool IsValidConcatNode(const ge::NodePtr &concat_node) {
  std::string node_type = concat_node->GetType();
  return (node_type == CONCAT || node_type == CONCATD || node_type == CONCATV2 || node_type == CONCATV2D);
}

Status TagNodes(vector<ge::NodePtr> &quant_nodes, vector<ge::NodePtr> &dequant_nodes, const int &attr_value) {
  for (size_t i = 0; i < dequant_nodes.size(); ++i) {
    if (!ge::AttrUtils::SetInt(dequant_nodes[i]->GetOpDesc(), ATTR_REQUANTED, attr_value)) {
      REPORT_FE_ERROR("[GraphOpt][Quant][TagNd] Set dequant requanted attr failed!");
      return FAILED;
    }
  }

  for (size_t i = 0; i < quant_nodes.size(); ++i) {
    if (!ge::AttrUtils::SetInt(quant_nodes[i]->GetOpDesc(), ATTR_REQUANTED, attr_value)) {
      REPORT_FE_ERROR("[GraphOpt][Quant][TagNd] Failed to set quant requantized attribute!");
      return FAILED;
    }
  }
  return SUCCESS;
}

bool IsEnableReluOfDequant(const ge::NodePtr &dequant_node) {
  bool del_relu_flag = false;
  if (!ge::AttrUtils::GetBool(dequant_node->GetOpDesc(), ATTR_DEL_RELU, del_relu_flag)) {
    FE_LOGD("Relu del attr is empty for node: %s.", dequant_node->GetName().c_str());
    return false;
  }
  return del_relu_flag;
}

Status SetReluFlag(const ge::NodePtr &dequant_node) {
  ge::NodePtr node_next = dequant_node->GetOutAllNodes().at(0);
  ge::NodePtr node_next_next;

  if (node_next->GetType() == RELU || node_next->GetType() == LEAKY_RELU) {
    if (!ge::AttrUtils::SetBool(dequant_node->GetOpDesc(), ATTR_RELU_FLAG, true)) {
      REPORT_FE_ERROR("[GraphOpt][Quant][SetReluFlag] Failed to set sqrt_mode! node name: %s",
                      dequant_node->GetName().c_str());
      return FAILED;
    }
    return SUCCESS;
  }

  if (IsValidConcatNode(node_next)) {
    if (node_next->GetOutAllNodes().empty()) {
      REPORT_FE_ERROR("[GraphOpt][Quant][SetReluFlag] Node name: %s has no output nodes", node_next->GetName().c_str());
      return FAILED;
    }
    node_next_next = node_next->GetOutAllNodes().at(0);
    if (node_next_next->GetType() == RELU || node_next_next->GetType() == LEAKY_RELU) {
      if (!ge::AttrUtils::SetBool(dequant_node->GetOpDesc(), ATTR_RELU_FLAG, true)) {
        REPORT_FE_ERROR("[GraphOpt][Quant][SetReluFlag] Setting relu flag failed! Node name: %s",
                        dequant_node->GetName().c_str());
        return FAILED;
      }
      return SUCCESS;
    }
  }

  if (!ge::AttrUtils::SetBool(dequant_node->GetOpDesc(), ATTR_RELU_FLAG, false)) {
    REPORT_FE_ERROR("[GraphOpt][Quant][SetReluFlag] Setting relu flag failed! Node name: %s",
                    dequant_node->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status SetReluFlagToDequant(ge::NodePtr &dequant_node) {
  if (IsEnableReluOfDequant(dequant_node)) {
    if (SetReluFlag(dequant_node) != SUCCESS) {
      FE_LOGE("Setting relu flag failed!");
      return FAILED;
    }
    FE_LOGD("set relu flag=true, node name: %s.", dequant_node->GetName().c_str());
  } else {
    if (!ge::AttrUtils::SetBool(dequant_node->GetOpDesc(), ATTR_RELU_FLAG, false)) {
      FE_LOGE("Setting relu flag failed!");
      return FAILED;
    }
  }
  return SUCCESS;
}

bool CheckReluValid(const ge::NodePtr &node) {
  if (node->GetType() != RELU && node->GetType() != LEAKY_RELU && node->GetType() != RELU6) {
    return false;
  }
  FE_LOGD("Relu op [%s], input size: %zu, output size: %zu.", node->GetName().c_str(), node->GetInAllNodes().size(),
          node->GetOutAllNodes().size());
  if ((node->GetInAllNodes().size() != 1) || (node->GetOutAllNodes().size() != 1)) {
    return false;
  }
  return true;
}

/**
 * only relu node can be removed,
 * or leakyrelu node and attr negative_slope is zero
 */
bool CheckNeedRemoveRelu(const ge::NodePtr &node) {
  if (node->GetType() == RELU) {
    return true;
  }
  if (node->GetType() == LEAKY_RELU) {
    float negative_slope;
    if (!ge::AttrUtils::GetFloat(node->GetOpDesc(), ATTR_NEGATIVE_SLOPE, negative_slope)) {
      FE_LOGD("leakyRelu node[%s] lacks negative slope attribute.", node->GetName().c_str());
      return false;
    }
    if (fabs(negative_slope) < kFloatEsp) {
      FE_LOGD("Node [%s]: attribute negative_slope is equal to 0.", node->GetName().c_str());
      return true;
    }
  }
  return false;
}

Status DealRelu(ge::ComputeGraph &graph, vector<ge::NodePtr> &relu_nodes, const float &scale_quant) {
  for (auto &relu_node : relu_nodes) {
    if (relu_node->GetType() == RELU6 && !FloatEqual(scale_quant, 1.0)) {
      auto op_desc = relu_node->GetOpDesc();
      ge::OpDescUtilsEx::SetType(op_desc, "Relu6D");
      (void)ge::AttrUtils::SetFloat(op_desc, ATTR_SCALE, scale_quant);
    }
    if (CheckNeedRemoveRelu(relu_node)) {
      Status ret = graph.RemoveNode(relu_node);
      if (ret != SUCCESS) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}
void AippGetInt64Value(ge::NamedAttrs &aipp_attr, const std::string &key, int64_t &value) {
  if (aipp_attr.GetItem(key).GetValue<int64_t>(value) != SUCCESS) {
     FE_LOGW("Aipp attr does not have the key: %s.", key.c_str());
  }
  return;
}

void AippGetFloatValue(ge::NamedAttrs &aipp_attr, const std::string &key, float &value) {
  if (aipp_attr.GetItem(key).GetValue<float>(value) != SUCCESS) {
    FE_LOGW("Aipp attr does not have the key: %s.", key.c_str());
  }
  return;
}

void AippGetFloatVecFirstValue(ge::NamedAttrs &aipp_attr, const std::string &key, float &value) {
  std::vector<float> tmp_vec;
  if (aipp_attr.GetItem(key).GetValue<std::vector<float>>(tmp_vec) != SUCCESS) {
    FE_LOGW("Aipp attr does not have the key: %s.", key.c_str());
  }
  if (tmp_vec.empty()) {
    FE_LOGW("AIpp key: %s attribute is empty.", key.c_str());
  } else {
    value = tmp_vec[0];
  }
  return;
}

Status DealQauntNotRequant(const ge::NodePtr &quant_node, vector<ge::NodePtr> &fusion_nodes) {
  if (quant_node == nullptr) {
    return NOT_CHANGED;
  }
  int requanted_flag;
  ge::OpDescPtr op_desc = quant_node->GetOpDesc();
  if (ge::AttrUtils::GetInt(op_desc, ATTR_REQUANTED, requanted_flag)) {
    FE_LOGD("this node has been requanted, skip, node name: %s.", op_desc->GetName().c_str());
    return NOT_CHANGED;
  }

  float scale = 1.0;
  if (!ge::AttrUtils::GetFloat(op_desc, ATTR_SCALE, scale)) {
    REPORT_FE_ERROR("[GraphOpt][NotReqntFus][DealQauntNotReqnt] Get quant scale failed!");
    return FAILED;
  }
  if ((scale < MIN_SCALE_VALUE) || (scale > MAX_SCALE_VALUE)) {
    scale = sqrt(scale);
    (void)ge::AttrUtils::SetFloat(op_desc, ATTR_SCALE, scale);
    (void)ge::AttrUtils::SetBool(op_desc, ATTR_SQRT_MODE, true);
  } else {
    (void)ge::AttrUtils::SetBool(op_desc, ATTR_SQRT_MODE, false);
  }

  vector<ge::NodePtr> quants;
  vector<ge::NodePtr> dequants;
  quants.push_back(quant_node);
  Status ret = TagNodes(quants, dequants, 0);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][NotReqntFus][DealQauntNotReqnt] tag nodes failed");
    return FAILED;
  }
  fusion_nodes.emplace_back(quant_node);
  return SUCCESS;
}
}  // namespace fe
