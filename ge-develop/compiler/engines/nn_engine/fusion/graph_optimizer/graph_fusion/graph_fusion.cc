/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/graph_fusion/graph_fusion.h"

#include <string>
#include <utility>
#include <vector>

#include "common/fe_inner_error_codes.h"
#include "common/fe_error_code.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/fe_report_error.h"
#include "common/graph/fe_graph_utils.h"
#include "common/util/constants.h"
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "fusion_rule_manager/fusion_rule_data/fusion_rule_pattern.h"
#include "fusion_rule_manager/fusion_rule_manager.h"
#include "graph_optimizer/fusion_common/fusion_pass_name.h"
#include "graph_optimizer/fusion_common/graph_node_map_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/conv_weight_compress_fusion_pass.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/split_conv_fusion_pass.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/split_conv_concat_fusion_pass.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/quant_pass/quant_host_cpu_op_common.h"
#include "graph_optimizer/graph_fusion/graph_matcher.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/quant_pass/delete_no_const_folding_fusion_pass.h"
#include "param_calculate/tensorsize_calculator.h"
#include "common/util/trace_manager/trace_manager.h"
#include "graph_metadef/graph/utils/graph_utils_ex.h"
#include "pass_registry.h"

using std::vector;

namespace fe {
namespace {

// 从 PassRegistry 获取 kCompatibleInherited 阶段注册的融合 pass，构建 name -> CreateFusionPassFn 映射
std::map<std::string, ge::fusion::CreateFusionPassFn> GetGePassMapFromRegistry() {
  std::map<std::string, ge::fusion::CreateFusionPassFn> ge_pass_map;
  auto reg_data_vec =
      ge::fusion::PassRegistry::GetInstance().GetFusionPassRegDataByStage(ge::CustomPassStage::kCompatibleInherited);
  for (const auto &reg_data : reg_data_vec) {
    auto create_fn = reg_data.GetCreatePassFn();
    if (create_fn != nullptr) {
      ge_pass_map[reg_data.GetPassName().GetString()] = create_fn;
    }
  }
  return ge_pass_map;
}

CastOptimizationType GetCastOptimizationType(const ge::DataType &input_dtype, const ge::DataType &output_dtype) {
  /* input and output dtype should be one of the following:
   * a. input: fp32/fp16, output: arbitrary except fp32/fp16.
   * b. input: arbitrary except fp32/fp16, output: fp32/fp16
   * c. input: fp32, NDC1HWC0, output: fp16, NDC1HWC0. (c is for the case of
   * Transdata from 5D to NDC1HWC0 with dtype fp32 wich is currently not
   * supported by TBE op) */
  bool condition1 = ((input_dtype == ge::DT_FLOAT16 || input_dtype == ge::DT_FLOAT) && output_dtype != ge::DT_FLOAT16 &&
                     output_dtype != ge::DT_FLOAT);

  bool condition2 = ((output_dtype == ge::DT_FLOAT16 || output_dtype == ge::DT_FLOAT) &&
                     input_dtype != ge::DT_FLOAT16 && input_dtype != ge::DT_FLOAT);

  bool condition3 = (input_dtype == ge::DT_FLOAT16 && output_dtype == ge::DT_FLOAT);

  bool condition4 = (input_dtype == ge::DT_FLOAT && output_dtype == ge::DT_FLOAT16);

  if (condition1 || condition3) {
    return CastOptimizationType::OPTIMIZE_WITH_TRANSDATA_AT_TAIL;
  } else if (condition2 || condition4) {
    return CastOptimizationType::OPTIMIZE_WITH_TRANSDATA_IN_FRONT;
  } else {
    return CastOptimizationType::CAST_OPMIZATION_BOTTOM;
  }
}

ge::NodePtr GetFirstSuccessor(const ge::NodePtr &node) {
  if (node->GetOutDataAnchor(0) != nullptr && !node->GetOutDataAnchor(0)->GetPeerInDataAnchors().empty() &&
      node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0) != nullptr &&
      node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode() != nullptr) {
    return node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
  }
  return nullptr;
}

bool IsSuccessorTwoContinuousReformat(const ge::NodePtr &node) {
  ge::NodePtr successor = GetFirstSuccessor(node);
  FE_CHECK(successor == nullptr, FE_LOGW("First successor of node %s is null.", node->GetName().c_str()), return false);
  FE_LOGD("First successor of node %s is %s.", node->GetName().c_str(), successor->GetName().c_str());
  if (successor->GetType() != REFORMAT) {
    return false;
  }
  ge::NodePtr successor_of_successor = GetFirstSuccessor(successor);
  FE_CHECK(successor_of_successor == nullptr, FE_LOGW("First successor of node %s is null.", node->GetName().c_str()),
           return false);
  FE_LOGD("First successor of node %s is %s", successor->GetName().c_str(), successor_of_successor->GetName().c_str());
  return (successor_of_successor->GetType() == REFORMAT);
}

bool JudgeUserQualified(int index, ge::NodePtr &first_trans, const ge::NodePtr &user) {
  if (user->GetType() != TRANSDATA) {
    FE_LOGD("User %d is not TransData.", index);
    return false;
  }
  /* In the follow case we will not optimize:
   * Cast(fp16->fp32, Nz) -> TransData(Nz->ND) -> Reformat(ND->NCHW) -> Reformat(NCHW->ND) -> TransData (ND->NZ)
   * It this case if the first TransData and Cast are switched, Cast will be in the middle of two TransDatas.
   * This impedes the cancel of two TransDatas.
   *
   * There is another case which current code does not give the perfect solution:
   * Cast(fp16->fp32, 5HD) -> TransData(5HD->NCHW) ->TransData(NCHW->5HD).
   * It will becomes :
   * TransData(5HD->NCHW) -> Cast(fp16->fp32, NCHW) -> TransData(NCHW->5HD)
   * currently.
   * And the perfect condition would be:
   * TransData(5HD->NCHW) ->TransData(NCHW->5HD) -> Cast(fp16->fp32, 5HD) */
  if (IsSuccessorTwoContinuousReformat(user)) {
    FE_LOGD("User %d %s encountered the TransData-Reformat-Reformat-TransData case.", index, user->GetName().c_str());
    return false;
  }

  if (index == 0) {
    first_trans = user;
    return true;
  } else {
    FE_CHECK_NOTNULL(first_trans);
    auto first_trans_input = first_trans->GetOpDesc()->MutableInputDesc(0);
    auto user_input = user->GetOpDesc()->MutableInputDesc(0);

    auto first_trans_output = first_trans->GetOpDesc()->MutableOutputDesc(0);
    auto user_output = user->GetOpDesc()->MutableOutputDesc(0);
    FE_CHECK_NOTNULL(first_trans_input);
    auto first_trans_input_format = ge::GetPrimaryFormat(first_trans_input->GetFormat());
    FE_CHECK_NOTNULL(first_trans_output);
    auto first_trans_output_format = ge::GetPrimaryFormat(first_trans_output->GetFormat());
    FE_CHECK_NOTNULL(user_input);
    auto user_input_format = ge::GetPrimaryFormat(user_input->GetFormat());
    FE_CHECK_NOTNULL(user_output);
    auto user_output_format = ge::GetPrimaryFormat(user_output->GetFormat());

    return first_trans_input_format == user_input_format &&
           first_trans_input->GetDataType() == user_input->GetDataType() &&
           first_trans_input->MutableShape().GetDims() == user_input->MutableShape().GetDims() &&
           first_trans_output_format == user_output_format &&
           first_trans_output->GetDataType() == user_output->GetDataType() &&
           first_trans_output->MutableShape().GetDims() == user_output->MutableShape().GetDims();
  }
}

/*  A ----> Cast ----> TransData1 ----> B
 *               \---> TransData2 ----> C
 *               \---> TransData3 ----> D
 *
 *  After merging, it becomes:
 *  A ----> Cast ----> TransData1 ----> B
 *                                \---> C
 *                                \---> D
 */
Status MergeTransData(ge::ComputeGraph &graph, const vector<ge::NodePtr> &trans_list, ge::NodePtr &peer_in_trans) {
  if (trans_list.size() <= 1) {
    return SUCCESS;
  }

  ge::OutDataAnchorPtr first_trans_out_anchor;
  vector<ge::InDataAnchorPtr> all_peer_in_anchors;
  for (size_t i = 0; i < trans_list.size(); i++) {
    auto trans_node = trans_list[i];
    FE_CHECK_NOTNULL(trans_node);
    FE_LOGD("Attempting to merge the first transdata node: %s.", trans_node->GetName().c_str());

    auto trans_out_anchor = trans_list[i]->GetOutDataAnchor(0);
    FE_CHECK_NOTNULL(trans_out_anchor);
    if (trans_out_anchor->GetPeerInDataAnchors().empty()) {
      FE_LOGW("There is no peer in anchor for No.%zu TransData %s.", i, trans_node->GetName().c_str());
      return FAILED;
    }
    /* We won't delete the first TransData and only get its output anchor. */
    if (i == 0) {
      first_trans_out_anchor = trans_out_anchor;
      peer_in_trans = trans_node;
      continue;
    }
    /* 1. Get the peer in data anchors of all TransData's output anchor. We can emit the first TransData. */
    if (trans_out_anchor->GetPeerInDataAnchors().empty()) {
      FE_LOGW("There is no peer in anchor for No.%zu TransData %s.", i, trans_node->GetName().c_str());
      return FAILED;
    }
    auto peer_in_anchor = trans_out_anchor->GetPeerInDataAnchors().at(0);
    FE_CHECK_NOTNULL(peer_in_anchor);
    all_peer_in_anchors.emplace_back(peer_in_anchor);
    /* 2. unlink edge */
    if (ge::GraphUtils::RemoveEdge(trans_out_anchor, peer_in_anchor) != SUCCESS) {
      FE_LOGW("Failed to remove edge from %s to its successor node.", trans_node->GetName().c_str());
      return FAILED;
    }
    /* 3. Remove redundant TransData(only keep the first one). */
    FE_LOGD("Try to remove %zuth transdata node %s.", i, trans_node->GetName().c_str());
    graph.RemoveNode(trans_node);
  }

  /* 4. link all peer in data anchors which is got from step 1. to the output anchor of the first TransData. */
  for (const auto &peer_in_anchor : all_peer_in_anchors) {
    ge::GraphUtils::AddEdge(first_trans_out_anchor, peer_in_anchor);
  }

  return SUCCESS;
}
/* 1. Check whether the output node is TransData. If there are multiple output
 * nodes, we need check whether they are all the same. If all output nodes
 * are the same TransData, we can switch them and current Cast.
 * 2. Merge TransData if necessary(more than one peer input).
 * This merging make convenience for switching Cast and TransData.
 * 3. Set the output_node by the merging node. */
Status GetPeerInTransdata(ge::ComputeGraph &graph, const ge::NodePtr &cast, const string &cast_name,
                          ge::NodePtr &peer_in_trans) {
  bool out_anchor_null =
      cast->GetOutDataAnchor(0) == nullptr || cast->GetOutDataAnchor(0)->GetPeerInDataAnchors().empty();
  if (out_anchor_null) {
    FE_LOGD("Cast %s does not have a user.", cast_name.c_str());
    return FAILED;
  }

  size_t peer_in_size = cast->GetOutDataAnchor(0)->GetPeerInDataAnchors().size();
  if (peer_in_size > 1) {
    FE_LOGD("Cast %s has more than one peer in nodes.", cast_name.c_str());
    int i = 0;
    ge::NodePtr first_trans;
    vector<ge::NodePtr> trans_list;
    for (const auto &peer_in_anchor : cast->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      FE_CHECK(peer_in_anchor == nullptr || peer_in_anchor->GetOwnerNode() == nullptr,
               FE_LOGW("User %d of Cast %s is null.", i, cast_name.c_str()), return FAILED);
      auto user = peer_in_anchor->GetOwnerNode();
      if (!JudgeUserQualified(i, first_trans, user)) {
        FE_LOGD("User %d of Cast %s is not qualified Transdata.", i, cast_name.c_str());
        return FAILED;
      } else {
        trans_list.emplace_back(user);
      }
      i++;
    }
    /* Do transdata merging if arrives here. */
    FE_LOGD("Merge transdatas after cast %s.", cast_name.c_str());
    return MergeTransData(graph, trans_list, peer_in_trans);
  } else if (peer_in_size == 1) {
    peer_in_trans = GetFirstSuccessor(cast);
    FE_CHECK(peer_in_trans == nullptr, FE_LOGW("Cast %s's peer input is null.", cast_name.c_str()), return FAILED);

    if (peer_in_trans->GetType() != TRANSDATA) {
      FE_LOGD("Cast %s's output is not TransData.", cast_name.c_str());
      return FAILED;
    } else {
      if (IsSuccessorTwoContinuousReformat(peer_in_trans)) {
        FE_LOGD("Transdata %s meet the TransData-Reformat-Reformat-TransData case.", peer_in_trans->GetName().c_str());
        return FAILED;
      }
      return SUCCESS;
    }
  }

  FE_LOGW("Cast %s does not have a successor.", cast_name.c_str());
  return FAILED;
}

bool CheckCastSpecicalDtypeIsOptimizable(bool special_cast_dtypem, const ge::Format &transdata_format) {
  if (!special_cast_dtypem) {
    return true;
  }
  return std::find(FE_HEAVY_FORMAT_VECTOR.begin(), FE_HEAVY_FORMAT_VECTOR.end(), transdata_format)
                   == FE_HEAVY_FORMAT_VECTOR.end();
}

bool IsCastOptimizable(ge::ComputeGraph &graph, ge::NodePtr cast, CastOptimizationType &type, ge::NodePtr &transdata) {
  FE_CHECK_NOTNULL(cast);

  string cast_name = cast->GetName();
  /* 1. Check input valid. */
  ge::NodePtr input_node;
  bool in_anchor_condition =
      cast->GetInDataAnchor(0) == nullptr || cast->GetInDataAnchor(0)->GetPeerOutAnchor() == nullptr;
  if (!in_anchor_condition) {
    input_node = cast->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
    FE_CHECK(input_node == nullptr, FE_LOGD("cast %s's producer node is null.", cast_name.c_str()), return false);
  } else {
    FE_LOGD("cast %s does not have producer.", cast_name.c_str());
    return false;
  }

  /* 2. Judge whether the TransData is in front or at tail. */
  auto cast_desc = cast->GetOpDesc();
  auto input_desc = cast_desc->MutableInputDesc(0);
  FE_CHECK(input_desc == nullptr, FE_LOGD("Input_desc is null."), return false);
  auto output_desc = cast_desc->MutableOutputDesc(0);
  FE_CHECK(output_desc == nullptr, FE_LOGD("Output_desc is null."), return false);
  ge::DataType input_dtype = input_desc->GetDataType();
  ge::DataType output_dtype = output_desc->GetDataType();
  type = GetCastOptimizationType(input_dtype, output_dtype);
  bool special_cast_dtype = input_dtype == ge::DT_FLOAT16 && output_dtype == ge::DT_INT8;
  FE_LOGD("%s input dtype is %u and output dtype is %u.", cast_name.c_str(), input_dtype, output_dtype);

  /* 3. Get the TransData which is waiting for switching runing sequence. */
  if (type == CastOptimizationType::OPTIMIZE_WITH_TRANSDATA_AT_TAIL) {
    Status ret = GetPeerInTransdata(graph, cast, cast_name, transdata);
    if (ret != SUCCESS) {
      return false;
    }
    auto trans_data_output = transdata->GetOpDesc()->MutableOutputDesc(0);
    FE_CHECK_NOTNULL(trans_data_output);
    return CheckCastSpecicalDtypeIsOptimizable(special_cast_dtype, trans_data_output->GetFormat());
  } else if (type == CastOptimizationType::OPTIMIZE_WITH_TRANSDATA_IN_FRONT) {
    if (input_node->GetType() != TRANSDATA || input_node->GetOutDataNodesSize() > 1) {
      FE_LOGD("Cast %s's input is not TransData or out_data_node.size > 1.", cast_name.c_str());
      return false;
    }
    transdata = input_node;
    auto trans_data_input = transdata->GetOpDesc()->MutableInputDesc(0);
    FE_CHECK_NOTNULL(trans_data_input);
    return CheckCastSpecicalDtypeIsOptimizable(special_cast_dtype, trans_data_input->GetFormat());
  } else {
    FE_LOGD("Cast %s is not optimizable.", cast_name.c_str());
    return false;
  }

  return true;
}

Status Relink(const ge::OutControlAnchorPtr &ori_out,
              const ge::OutControlAnchorPtr &new_out,
              const ge::InControlAnchorPtr &ori_in,
              const ge::InControlAnchorPtr &new_in) {
  FE_CHECK_NOTNULL(ori_out);
  FE_CHECK_NOTNULL(new_out);
  FE_CHECK_NOTNULL(ori_in);
  FE_CHECK_NOTNULL(new_in);

  const string ori_out_name = ori_out->GetOwnerNode()->GetName();
  const string ori_out_type = ori_out->GetOwnerNode()->GetType();

  const string ori_in_name = ori_in->GetOwnerNode()->GetName();
  const string ori_in_type = ori_in->GetOwnerNode()->GetType();
  FE_LOGD("Remove edge between src:%s and dst:%s.",
          ori_out_name.c_str(), ori_in_name.c_str());

  if (ge::GraphUtils::RemoveEdge(ori_out, ori_in) != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("Remove control edge between op:%s(%s) and op:%s(%s) failed",
                    ori_out_name.c_str(), ori_out_type.c_str(),
                    ori_in_name.c_str(), ori_in_type.c_str());
    return FAILED;
  }

  const string new_out_name = new_out->GetOwnerNode()->GetName();
  const string new_out_type = new_out->GetOwnerNode()->GetType();

  const string new_in_name = new_in->GetOwnerNode()->GetName();
  const string new_in_type = new_in->GetOwnerNode()->GetType();
  FE_LOGD("Add edge between src:%s and dst:%s.", new_out_name.c_str(), new_in_name.c_str());
  auto ret = ge::GraphUtils::AddEdge(new_out, new_in);
  if (ret != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("Add control edge between op:%s(%s) and op:%s(%s) failed",
                    new_out_name.c_str(), new_out_type.c_str(),
                    new_in_name.c_str(), new_in_type.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status RelinkControlEdge(const ge::NodePtr &node_front,
                         const ge::NodePtr &node_tail) {
  const std::string front_name(node_front->GetName());
  const std::string tail_name(node_tail->GetName());
  /* 1. Move in control edge from front node to tail node. */
  auto front_node_in_ctrl = node_front->GetInControlAnchor();
  if (front_node_in_ctrl != nullptr) {
    auto in_control_anchor = node_tail->GetInControlAnchor();
    for (auto &peer_out_control_anchor : front_node_in_ctrl->GetPeerOutControlAnchors()) {
      if (Relink(peer_out_control_anchor, peer_out_control_anchor,
                 front_node_in_ctrl, in_control_anchor) != SUCCESS) {
        return FAILED;
      }
    }
  }

  /* 2. Move out control edge from tail node to front node. */
  auto tail_node_out_ctrl = node_tail->GetOutControlAnchor();
  if (tail_node_out_ctrl != nullptr) {
    auto front_node_out_ctrl = node_front->GetOutControlAnchor();
    auto peer_in_ctrl_anchors = tail_node_out_ctrl->GetPeerInControlAnchors();
    for (const auto &peer_in_ctrl : peer_in_ctrl_anchors) {
      if (Relink(tail_node_out_ctrl, front_node_out_ctrl,
                 peer_in_ctrl, peer_in_ctrl) != SUCCESS) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

/* A----> node1 ----> node2----> B
 *                        \----> C
 *                        \----> D

 * node2 may have multiple peer in data anchors
 *
 * Loop will be generated in the following case if
 * we do not take care of the control edges:
 *                  A
 *                  |
 *                  |
 *                 node1------->X
 *                  |           |
 *                  |           | control edge
 *                 node2<--------
 *                / | \
 *               /  |  \
 *               B  C  D
 *
 *                  A
 *                  |
 *                  |
 *                 node2<-------X
 *                  |           |
 *                  |           | control edge
 *                 node1-------->
 *                / | \
 *               /  |  \
 *               B  C  D
 *
 * If there is a in control edge on node2, after switching,
 * the control in edge should be placed on node1. And
 * If there is a out control edge on node1, after switching,
 * the out control edge should emit from node2. */
Status SwitchTwoNode(ge::NodePtr node1, ge::NodePtr node2) {
  FE_LOGD("Switch %s with %s.", node1->GetName().c_str(), node2->GetName().c_str());
  auto in_anchor_of_node1 = node1->GetInDataAnchor(0);
  FE_CHECK_NOTNULL(in_anchor_of_node1);
  auto peer_out_anchor_of_node1 = in_anchor_of_node1->GetPeerOutAnchor();
  FE_CHECK_NOTNULL(peer_out_anchor_of_node1);
  auto node_front = peer_out_anchor_of_node1->GetOwnerNode();
  auto out_anchor_of_node1 = node1->GetOutDataAnchor(0);

  auto in_anchor_of_node2 = node2->GetInDataAnchor(0);
  auto out_anchor_of_node2 = node2->GetOutDataAnchor(0);
  FE_CHECK_NOTNULL(out_anchor_of_node2);
  auto peer_in_anchors_of_node2 = out_anchor_of_node2->GetPeerInDataAnchors();
  Status ret;
  /* 1. Remove edge between front node and node1 */
  if (ge::GraphUtils::RemoveEdge(peer_out_anchor_of_node1, in_anchor_of_node1) != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][SwitchTransData] [1]:Failed to remove edge from [%s] to [%s].",
                    node_front->GetName().c_str(), node1->GetName().c_str());
    return FAILED;
  }

  /* 2. Remove edge between node1 and node2 */
  if (ge::GraphUtils::RemoveEdge(out_anchor_of_node1, in_anchor_of_node2) != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][SwitchTransData] [2]:Failed to remove edge from [%s] to [%s].",
                    node1->GetName().c_str(), node2->GetName().c_str());
    return FAILED;
  }

  /* 3. Remove edge between node2 and all input of tail nodes.... */
  for (auto in_anchor : peer_in_anchors_of_node2) {
    auto node_tail = in_anchor->GetOwnerNode();
    FE_CHECK_NOTNULL(node_tail);
    if (ge::GraphUtils::RemoveEdge(out_anchor_of_node2, in_anchor) != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("[GraphOptJdgInst][SwitchTransData] [3]:Failed to remove edge from [%s] to [%s].",
                      node2->GetName().c_str(), node_tail->GetName().c_str());
      return FAILED;
    }
  }

  /* 4. Link front node and node2 */
  ret = ge::GraphUtils::AddEdge(peer_out_anchor_of_node1, in_anchor_of_node2);
  if (ret != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][SwitchTransData] [4]Failed to add edge between %s and %s",
                    node_front->GetName().c_str(), node2->GetName().c_str());
    return FAILED;
  }

  /* 5. Link node2 and node1 */
  ret = ge::GraphUtils::AddEdge(out_anchor_of_node2, in_anchor_of_node1);
  if (ret != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][SwitchTransData] [5]Failed to add edge between %s and %s",
                    node2->GetName().c_str(), node1->GetName().c_str());
    return FAILED;
  }

  /* 6. Link node1 and all tail nodes */
  for (auto &in_anchor : peer_in_anchors_of_node2) {
    ret = ge::GraphUtils::AddEdge(out_anchor_of_node1, in_anchor);
    auto node_tail = in_anchor->GetOwnerNode();
    FE_CHECK_NOTNULL(node_tail);
    if (ret != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("[GraphOptJdgInst][SwitchTransData] [6]Failed to add edge between %s and %s",
                      node1->GetName().c_str(), node_tail->GetName().c_str());
      return FAILED;
    }
  }

  /* 7. Relink control edges to avoid cycle. */
  if (RelinkControlEdge(node2, node1) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

/* Fuse two concecutive cast if they meet the following pattern
 * ----> TransData ----> Cast1(x -> fp32) ----> Cast2 (fp32 -> fp16)----
 * This case will become:
 * ----> TransData ----> Cast1(x -> fp16) ---- */
Status FuseCastWhenTransDataInFront(ge::ComputeGraph &graph, ge::NodePtr cast) {
  auto output_desc_ptr = cast->GetOpDesc()->GetOutputDescPtr(0);
  FE_CHECK_NOTNULL_WARNLOG(output_desc_ptr);
  if (output_desc_ptr->GetDataType() != ge::DT_FLOAT) {
    return SUCCESS;
  }
  /* 1. All users of Cast1 must be Cast if we want to fuse them. */
  FE_CHECK_NOTNULL_WARNLOG(cast->GetOutDataAnchor(0));
  for (const auto &peer_out_anchor : cast->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FE_CHECK_NOTNULL_WARNLOG(peer_out_anchor);
    auto user = peer_out_anchor->GetOwnerNode();
    FE_CHECK_NOTNULL_WARNLOG(user);
    if (user->GetType() != CAST) {
      FE_LOGD("%s is not Cast type.", user->GetName().c_str());
      return SUCCESS;
    } else {
      auto user_input = user->GetOpDesc()->MutableInputDesc(0);
      FE_CHECK_NOTNULL_WARNLOG(user_input);
      auto user_output = user->GetOpDesc()->MutableOutputDesc(0);
      FE_CHECK_NOTNULL_WARNLOG(user_output);
      if (user_input->GetDataType() != ge::DT_FLOAT || user_output->GetDataType() != ge::DT_FLOAT16) {
        FE_LOGD("user %s of cast %s is not float type.", user->GetName().c_str(), cast->GetName().c_str());
        return SUCCESS;
      }
    }
  }

  /* 2. Set output data type of Cast1 using Cast2's output datatype which must
   * be fp16. */
  auto cast_output = cast->GetOpDesc()->MutableOutputDesc(0);
  FE_CHECK_NOTNULL_WARNLOG(cast_output);
  cast_output->SetDataType(ge::DT_FLOAT16);
  (void)ge::AttrUtils::SetInt(cast->GetOpDesc(), CAST_ATTR_DST_TYPE, static_cast<int64_t>(ge::DT_FLOAT16));
  /* 3. Remove all Cast2. */
  FE_CHECK_NOTNULL_WARNLOG(cast->GetOutDataAnchor(0));
  for (const auto &peer_out_anchor : cast->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    auto cast2 = peer_out_anchor->GetOwnerNode();
    FE_LOGD("Try to remove node %s.", cast2->GetName().c_str());
    if (graph.RemoveNode(cast2) != SUCCESS) {
      FE_LOGW("[GraphOptJdgInst][SwitchTransData] Failed to remove %s", cast2->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

/* Fuse two concecutive cast if they meet the following pattern
 * ----> Cast2(16 -> fp32) ----> Cast1 (fp32 -> x) ----> TransData ---->
 * This case will become:
 * ----> Cast1(16 -> x) ----> TransData ----> */
Status FuseCastWhenTransDataAtTail(ge::ComputeGraph &graph, ge::NodePtr cast) {
  auto input_desc_ptr = cast->GetOpDesc()->GetInputDescPtr(0);
  FE_CHECK_NOTNULL_WARNLOG(input_desc_ptr);
  if (input_desc_ptr->GetDataType() != ge::DT_FLOAT) {
    return SUCCESS;
  }

  /* 1. Currently, we only allow Cast2(the front one) to be one producer and one
   * user. And Cast1's input must be fp16 and output must be fp32. */
  FE_CHECK_NOTNULL_WARNLOG(cast->GetInDataAnchor(0));
  FE_CHECK_NOTNULL_WARNLOG(cast->GetInDataAnchor(0)->GetPeerOutAnchor());
  auto cast2 = cast->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  FE_CHECK_NOTNULL_WARNLOG(cast2);
  if (cast2->GetType() != CAST) {
    return SUCCESS;
  }

  if (cast2->GetOutAllNodes().size() != 1) {
    FE_LOGD("Front cast %s has more than one user.", cast2->GetName().c_str());
    return SUCCESS;
  }

  auto cast2_input = cast2->GetOpDesc()->MutableInputDesc(0);
  FE_CHECK_NOTNULL_WARNLOG(cast2_input);
  auto cast2_output = cast2->GetOpDesc()->MutableOutputDesc(0);
  FE_CHECK_NOTNULL_WARNLOG(cast2_output);
  if ((cast2_input->GetDataType() != ge::DT_FLOAT16 || cast2_output->GetDataType() != ge::DT_FLOAT)) {
    FE_LOGD("Front cast2's dtype %u and %u is not qualified.", cast2_input->GetDataType(), cast2_output->GetDataType());
    return SUCCESS;
  }

  /* 2. Set input data type of Cast1 using Cast2's input datatype which must
   * be fp16. */
  auto cast_input = cast->GetOpDesc()->MutableInputDesc(0);
  FE_CHECK_NOTNULL_WARNLOG(cast_input);
  cast_input->SetDataType(cast2_input->GetDataType());

  /* 3. Remove all Cast2. */
  if (graph.RemoveNode(cast2) != SUCCESS) {
    FE_LOGW("[GraphOptJdgInst][SwitchTransData] Failed to remove %s", cast2->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

/* ----> TransData ----> Cast ----
 * TransData must be one output and have only one user of this output.
 * Cast must be one output but probably has more than one user. */
Status HandleTransDataInFront(ge::NodePtr transdata, ge::NodePtr cast) {
  FE_CHECK_NOTNULL(transdata);
  FE_CHECK_NOTNULL(cast);
  FE_LOGD("transdata %s is in front of %s.", transdata->GetName().c_str(), cast->GetName().c_str());

  auto transdata_desc = transdata->GetOpDesc();
  auto cast_desc = cast->GetOpDesc();

  auto trans_data_input = transdata_desc->MutableInputDesc(0);
  FE_CHECK_NOTNULL(trans_data_input);
  auto trans_data_output = transdata_desc->MutableOutputDesc(0);
  FE_CHECK_NOTNULL(trans_data_output);

  auto cast_input = cast_desc->MutableInputDesc(0);
  FE_CHECK_NOTNULL(cast_input);
  auto cast_output = cast_desc->MutableOutputDesc(0);
  FE_CHECK_NOTNULL(cast_output);
  ge::Format trans_data_in_format = trans_data_input->GetFormat();
  if (ge::HasC0Format(trans_data_in_format) && ge::GetC0Value(trans_data_in_format) == SHAPE_NUMBER_8) {
    FE_LOGI("Cast not support c08 format.");
    return FAILED;
  }
  if (!VerifyCastC0Format(cast_desc, trans_data_in_format)) {
    FE_LOGI("Cast not support current c0 format:%d.", trans_data_in_format);
    return FAILED;
  }
  const ge::GeShape &trans_data_in_shape = trans_data_input->GetShape();
  ge::DataType cast_out_dtype = cast_output->GetDataType();
  const ge::GeShape &cast_out_shape = cast_output->GetShape();

  /* 2. change dtype of TransData and change format of Cast. */
  trans_data_input->SetDataType(cast_out_dtype);
  trans_data_output->SetDataType(cast_out_dtype);
  ge::Format trans_data_out_format = trans_data_output->GetFormat();
  if (ge::HasC0Format(trans_data_in_format)) {
    trans_data_input->SetFormat(ModifyC0Format(trans_data_in_format, cast_out_dtype));
  }
  if (ge::HasC0Format(trans_data_out_format)) {
    trans_data_output->SetFormat(ModifyC0Format(trans_data_out_format, cast_out_dtype));
  }

  cast_input->SetFormat(trans_data_in_format);
  cast_output->SetFormat(trans_data_in_format);

  /* 3. Set Transdata's new shape by new dtype. */
  trans_data_output->SetShape(cast_out_shape);

  /* 4. Set cast's new shape */
  cast_input->SetShape(trans_data_in_shape);
  cast_output->SetShape(trans_data_in_shape);

  /* 5. Switch two op in graph. */
  return SwitchTwoNode(transdata, cast);
}

/* ----> Cast ----> TransData ----
 * Cast must be one output and have only one user.
 * TransData must be one output and may have more than one user. */
Status HandleTransDataAtTail(ge::NodePtr transdata, ge::NodePtr cast) {
  FE_CHECK_NOTNULL(transdata);
  FE_CHECK_NOTNULL(cast);
  auto transdata_desc = transdata->GetOpDesc();
  auto cast_desc = cast->GetOpDesc();

  auto trans_data_input = transdata_desc->MutableInputDesc(0);
  FE_CHECK_NOTNULL(trans_data_input);
  auto trans_data_output = transdata_desc->MutableOutputDesc(0);
  FE_CHECK_NOTNULL(trans_data_output);

  auto cast_input = cast_desc->MutableInputDesc(0);
  FE_CHECK_NOTNULL(cast_input);
  auto cast_output = cast_desc->MutableOutputDesc(0);
  FE_CHECK_NOTNULL(cast_output);

  ge::Format trans_data_out_format = trans_data_output->GetFormat();
  if (ge::HasC0Format(trans_data_out_format) && ge::GetC0Value(trans_data_out_format) == SHAPE_NUMBER_8) {
    FE_LOGI("Cast not support c08 format.");
    return FAILED;
  }
  ge::DataType cast_in_dtype = cast_input->GetDataType();
  const ge::GeShape &trans_data_out_shape = trans_data_output->GetShape();
  FE_LOGD("trans_data_out_format[%u], cast_in_dtype[%u], trans_data_input_format[%u], cast_output_dtype[%u].",
           trans_data_out_format, cast_in_dtype, trans_data_input->GetFormat(), cast_output->GetDataType());

  /* 1. change data type of TransData and change format of Cast. */
  trans_data_input->SetDataType(cast_in_dtype);
  trans_data_output->SetDataType(cast_in_dtype);
  ge::Format trans_data_in_format = trans_data_input->GetFormat();
  if (ge::HasC0Format(trans_data_in_format)) {
    trans_data_input->SetFormat(ModifyC0Format(trans_data_in_format, cast_in_dtype));
  }
  if (ge::HasC0Format(trans_data_out_format)) {
    trans_data_output->SetFormat(ModifyC0Format(trans_data_out_format, cast_in_dtype));
  }

  cast_input->SetFormat(trans_data_out_format);
  cast_output->SetFormat(trans_data_out_format);

  /* 2. Set shape of cast. */
  cast_input->SetShape(trans_data_out_shape);
  cast_output->SetShape(trans_data_out_shape);

  FE_LOGD("trans_data_out_format[%u], cast_in_dtype[%u], trans_data_input_format[%u], cast_output_dtype[%u].",
           trans_data_out_format, cast_in_dtype, trans_data_input->GetFormat(), cast_output->GetDataType());
  /* 3. Switch two op in graph. */
  return SwitchTwoNode(cast, transdata);
}

void SetAllTesnorWithDtype(const ge::OpDescPtr &op_desc, ge::DataType dtype) {
  std::vector<int64_t> invalid_input;
  (void)ge::AttrUtils::GetListInt(op_desc, INPUT_FORMAT_AGNOSTIC_EXCEPTION, invalid_input);

  for (size_t i = 0; i < op_desc->GetAllInputsSize(); ++i) {
    auto input = op_desc->MutableInputDesc(i);
    if (input == nullptr) {
      continue;
    }
    if (std::find(invalid_input.begin(), invalid_input.end(), static_cast<int64_t>(i)) != invalid_input.end()) {
      FE_LOGD("%s's input %zu is in exception.", op_desc->GetName().c_str(), i);
      continue;
    }
    input->SetDataType(dtype);
  }

  std::vector<int64_t> invalid_output;
  (void)ge::AttrUtils::GetListInt(op_desc, OUTPUT_FORMAT_AGNOSTIC_EXCEPTION, invalid_output);

  for (size_t i = 0; i < op_desc->GetAllOutputsDescSize(); ++i) {
    auto output = op_desc->MutableOutputDesc(i);
    if (output == nullptr) {
      continue;
    }

    if (std::find(invalid_output.begin(), invalid_output.end(), static_cast<int64_t>(i)) != invalid_output.end()) {
      FE_LOGD("%s's output %zu is in exception.", op_desc->GetName().c_str(), i);
      continue;
    }
    output->SetDataType(dtype);
  }
}
}  // namespace

GraphFusion::GraphFusion(FusionRuleManagerPtr fusion_rule_mgr_ptr, OpsKernelInfoStorePtr ops_kernel_info_store_ptr,
                         FusionPriorityMgrPtr fusion_priority_mgr_ptr)
    : fusion_rule_mgr_ptr_(std::move(fusion_rule_mgr_ptr)),
      ops_kernel_info_store_ptr_(std::move(ops_kernel_info_store_ptr)),
      fusion_priority_mgr_ptr_(std::move(fusion_priority_mgr_ptr)) {}

GraphFusion::~GraphFusion() = default;

/*
 *  @ingroup fe
 *  @brief   run graph fusion
 *  @param   [in|out] compute graph
 *  @return  SUCCESS or FAILED or ERRCODE
 */
Status GraphFusion::Fusion(ge::ComputeGraph &graph) {
  if (fusion_rule_mgr_ptr_ == nullptr) {
    REPORT_FE_ERROR("[GraphOpt][FirstRoundFusion] GraphFusion, fusion_rule_mgr_ptr_ is null.");
    return FAILED;
  }

  if (fusion_priority_mgr_ptr_ == nullptr) {
    REPORT_FE_ERROR("[GraphOpt][FirstRoundFusion] GraphFusion, fusion_priority_mgr_ptr_ is null.");
    return FAILED;
  }

  Status ret = FusionEachGraph(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FirstRoundFusion] MainGraph[%s]: RunGraphFusion unsuccessfully.",
                    graph.GetName().c_str());
    return ret;
  }
  FE_LOGI("MainGraph[%s]: RunGraphFusion successfully.", graph.GetName().c_str());

  for (const auto &sub_graph : graph.GetAllSubgraphs()) {
    Status in_ret = FusionEachGraph(*(sub_graph.get()));
    if (in_ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FirstRoundFusion] SubGraph[%s]: RunGraphFusion unsuccessfully.",
                      sub_graph->GetName().c_str());
      return in_ret;
    }
    FE_LOGI("SubGraph[%s]: RunGraphFusion successfully.", sub_graph->GetName().c_str());
  }
  return SUCCESS;
}

Status GraphFusion::FusionPruningPass(ge::ComputeGraph &graph) {
  FE_LOGD("Graph[%s] process fusion pruning pass.", graph.GetName().c_str());
  FE_CHECK(fusion_priority_mgr_ptr_ == nullptr, FE_LOGE("Fusion_priority_mgr_ptr_ is null."), return FAILED);

  Status ret = FusionEachPruningPass(graph);
  if (ret != SUCCESS) {
    FE_LOGE("MainGraph[%s]: FusionPruningPass unsuccessfully.", graph.GetName().c_str());
    return ret;
  }
  FE_LOGI("MainGraph[%s]: FusionPruningPass successfully.", graph.GetName().c_str());

  for (const auto &sub_graph : graph.GetAllSubgraphs()) {
    Status in_ret = FusionEachPruningPass(*(sub_graph.get()));
    if (in_ret != SUCCESS) {
      FE_LOGE("SubGraph[%s]: FusionPruningPass did not succeed.", sub_graph->GetName().c_str());
      return in_ret;
    }
    FE_LOGI("SubGraph[%s]: FusionPruningPass success.", sub_graph->GetName().c_str());
  }
  return SUCCESS;
}

Status GraphFusion::RunGraphFusionPassByType(const string &stage, ge::ComputeGraph &graph,
                                             GraphFusionPassType type) {
  FE_TIMECOST_START(RunGraphFusionPassByType);
  string pass_type_str = GetPassTypeString(type);
  Status ret = RunBuiltInFusionByType(graph, type);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[%s][RunGraphFusion] MainGraph[%s]: Run graph fusion pass by type %s unsuccessfully.",
                    stage.c_str(), graph.GetName().c_str(), pass_type_str.c_str());
    return ret;
  }

  FE_LOGI("MainGraph[%s]: Run graph fusion pass by type %s successfully.", graph.GetName().c_str(),
          pass_type_str.c_str());

  for (const auto &sub_graph : graph.GetAllSubgraphs()) {
    Status in_ret = RunBuiltInFusionByType(*(sub_graph.get()), type);
    if (in_ret != SUCCESS) {
      REPORT_FE_ERROR("[%s][RunGraphFusion] SubGraph[%s]: Running graph fusion pass of type %s was unsuccessful.",
                      stage.c_str(), sub_graph->GetName().c_str(), pass_type_str.c_str());
      return in_ret;
    }
    FE_LOGI("SubGraph[%s]: Run graph fusion pass by type %s successfully.", sub_graph->GetName().c_str(),
            pass_type_str.c_str());
  }
  FE_TIMECOST_END(RunGraphFusionPassByType, "GraphFusion::RunGraphFusionPassByType");
  return SUCCESS;
}

void AdjustRunCountAfterPass(ge::ComputeGraph &graph, const FusionPassOrRule &pass_or_rule, int64_t &run_count) {
  int64_t run_count_attr = 0;
  (void)ge::AttrUtils::GetInt(graph, "run_count", run_count_attr);
  run_count++;
  if (run_count_attr < run_count) {
    FE_LOGI("pass:%s, run_count is not equal. run_count:%ld, cur_count:%ld", pass_or_rule.name.c_str(),
            run_count_attr, run_count);
    (void)GraphNodeMapUtil::ReCreateNodeTypeMapInGraph(graph);
    (void)ge::AttrUtils::SetInt(graph, "run_count", run_count);
  }
  return;
}

Status GraphFusion::FusionEachGraph(ge::ComputeGraph &graph) {
  bool is_single_op_scene = false;
  (void) ge::AttrUtils::GetBool(graph, ge::ATTR_SINGLE_OP_SCENE, is_single_op_scene);
  FE_LOGD("The attr is single op scene of graph[%s] is [%d].", graph.GetName().c_str(), is_single_op_scene);
  const std::vector<FusionPassOrRule> &sorted_graph_fusion_vec =
          fusion_priority_mgr_ptr_->GetSortedGraphFusionList(is_single_op_scene);
  NodeMapInfoPtr node_map_info;
  (void)ge::AttrUtils::SetInt(graph, "run_count", 0);
  if (GraphNodeMapUtil::CreatAndSetOpTypeMap(node_map_info, graph) != SUCCESS) {
    return FAILED;
  }
  int64_t run_count = 0;
  auto ge_pass_map = GetGePassMapFromRegistry();
  for (const FusionPassOrRule &pass_or_rule : sorted_graph_fusion_vec) {
    FE_TIMECOST_START(RunOnePassFusion);
    FE_LOGD("Start Graph Fusion:%s Owner:%s Method:%s Priority:%d.", pass_or_rule.name.c_str(),
            GetPassTypeString(static_cast<GraphFusionPassType>(pass_or_rule.type)).c_str(),
            pass_or_rule.method.c_str(),
            FusionPriorityManager::GetRealPriority(pass_or_rule.priority));
    Status ret = SUCCESS;
    if (pass_or_rule.method == PASS_METHOD) {
      if (find(GRAPH_FUSION_QUANT_PASS_VEC.begin(), GRAPH_FUSION_QUANT_PASS_VEC.end(),
               pass_or_rule.type) == GRAPH_FUSION_QUANT_PASS_VEC.end()) {
        ret = RunOnePassFusion(graph, pass_or_rule, ge_pass_map);
        AdjustRunCountAfterPass(graph, pass_or_rule, run_count);
      }
    } else if (pass_or_rule.method == RULE_METHOD) {
      ret = RunOneRuleFusion(graph, pass_or_rule);
    } else {
      FE_LOGW("Unknown fusion method:%s.", pass_or_rule.method.c_str());
      continue;
    }
    string out_s = "GraphFusion::RunOnePassFusion pass_name: " + pass_or_rule.name;
    FE_TIMECOST_END_LOGI(RunOnePassFusion, out_s.c_str());
    if (ret != SUCCESS) {
      return FAILED;
    }
  }
  if (GraphNodeMapUtil::ClearOpTypeMapToGraph(graph) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FirstRoundFusion] Failed to clear op type map.");
    return FAILED;
  }
  return SUCCESS;
}

Status GraphFusion::FusionEachPruningPass(ge::ComputeGraph &graph) {
  bool is_single_op_scene = false;
  (void)ge::AttrUtils::GetBool(graph, ge::ATTR_SINGLE_OP_SCENE, is_single_op_scene);
  FE_LOGD("The attr is single op scene of graph[%s] is [%d].", graph.GetName().c_str(), is_single_op_scene);
  const std::vector<FusionPassOrRule> &sorted_graph_fusion_vec =
      fusion_priority_mgr_ptr_->GetSortedGraphFusionList(is_single_op_scene);
  FE_CHECK(sorted_graph_fusion_vec.empty(),
           FE_LOGD("There is no registered graph fusion pass or rule."), return SUCCESS);

  NodeMapInfoPtr node_map_info;
  if (GraphNodeMapUtil::CreatAndSetOpTypeMap(node_map_info, graph) != SUCCESS) {
    return FAILED;
  }

  auto ge_pass_map = GetGePassMapFromRegistry();
  for (const FusionPassOrRule &pass_or_rule : sorted_graph_fusion_vec) {
    if (pass_or_rule.method != PASS_METHOD ||
        !IsPassAttrTypeOn(pass_or_rule.pass_desc.attr, PassAttrType::PRUNING_FLAG)) {
      continue;
    }
    FE_TIMECOST_START(RunOnePassFusion);
    FE_LOGD("Start running pass fusion:%s, owner:%s, priority:%d.", pass_or_rule.name.c_str(),
            GetPassTypeString(static_cast<GraphFusionPassType>(pass_or_rule.type)).c_str(),
            FusionPriorityManager::GetRealPriority(pass_or_rule.priority));
    Status ret = RunOnePassFusion(graph, pass_or_rule, ge_pass_map);
    string out_s = "GraphFusion::FusionEachPruningPass pass_name: " + pass_or_rule.name;
    FE_TIMECOST_END_LOGI(RunOnePassFusion, out_s.c_str());
    if (ret != SUCCESS) {
      FE_LOGE("Pruning passes fusion run: %s for graph %s failed.", pass_or_rule.name.c_str(), graph.GetName().c_str());
      return FAILED;
    }
  }

  if (GraphNodeMapUtil::ClearOpTypeMapToGraph(graph) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status GraphFusion::SwitchTransDataAndCast(ge::ComputeGraph &graph,
                                           const vector<ge::NodePtr> &special_cast_list) const {
  CastOptimizationType type;
  ge::NodePtr transdata = nullptr;
  for (auto cast : special_cast_list) {
    /* 1. Check the cast valid or not */
    if (IsCastOptimizable(graph, cast, type, transdata)) {
      FE_CHECK_NOTNULL(transdata);
      if (type == CastOptimizationType::OPTIMIZE_WITH_TRANSDATA_IN_FRONT) {
        /* 2.0 Fuse multiple cast nodes. */
        (void)FuseCastWhenTransDataInFront(graph, cast);
        /* 3.0 Switch cast and transdata pattern 0. */
        if (HandleTransDataInFront(transdata, cast) != SUCCESS) {
          FE_LOGD("Switch %s and %s where transdata is at the front was unsuccessful.", transdata->GetName().c_str(),
                  cast->GetName().c_str());
        } else {
          Status ret = ComputeTensorSize(cast, transdata);
          if (ret != SUCCESS) {
            REPORT_FE_ERROR("[SubGraphOpt][CalcTensorSize][CalcRunPara] Failed to compute tensor size.");
            return ret;
          }
        }
      } else if (type == CastOptimizationType::OPTIMIZE_WITH_TRANSDATA_AT_TAIL) {
        /* 2.1 Fuse multiple cast nodes. */
        (void)FuseCastWhenTransDataAtTail(graph, cast);
        /* 3.1 Switch cast and transdata pattern 0. */
        if (HandleTransDataAtTail(transdata, cast) != SUCCESS) {
          FE_LOGD("Switch %s with %s. where transdata is at tail unsuccessful.", transdata->GetName().c_str(),
                  cast->GetName().c_str());
        } else {
          Status ret = ComputeTensorSize(cast, transdata);
          if (ret != SUCCESS) {
            REPORT_FE_ERROR("[SubGraphOpt][CalcTensorSize][CalcRunPara] Failed to compute tensor size.");
            return ret;
          }
        }
      }
    }
  }
  return SUCCESS;
}

Status GraphFusion::ComputeTensorSize(ge::NodePtr &cast, ge::NodePtr &transdata) const {
  ge::OpDescPtr op_desc_ptr;
  Status status;
  if (cast != nullptr) {
    op_desc_ptr = cast->GetOpDesc();
    FE_CHECK_NOTNULL(op_desc_ptr);
    status = TensorSizeCalculator::CalculateOpTensorSize(cast);
    if (status != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][CalcTensorSize][CalcRunPara] Failed to calculate the tensor size for operation [%s, %s].",
                      op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
      return status;
    }
  }
  if (transdata != nullptr) {
    op_desc_ptr = transdata->GetOpDesc();
    FE_CHECK_NOTNULL(op_desc_ptr);
    status = TensorSizeCalculator::CalculateOpTensorSize(transdata);
    if (status != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][CalcTensorSize][CalcRunPara] Failed to calculate the tensor size for operation [%s, %s].",
                      op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
      return status;
    }
  }
  return SUCCESS;
}

Status GraphFusion::FusionQuantOp(ge::ComputeGraph &graph) {
  Status ret = FusionQuantOpOfEachGraph(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionQuantOp] Quant fusion failed.");
    return ret;
  }
  for (auto sub_graph : graph.GetAllSubgraphs()) {
    ret = FusionQuantOpOfEachGraph(*(sub_graph.get()));
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionQuantOp] Quant fusion failed.");
      return ret;
    }
  }
  return SUCCESS;
}

Status GraphFusion::SetContinuousDtypeForSingleNode(ge::NodePtr &node) const {
  auto op_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  bool flag = false;
  if (!ge::AttrUtils::GetBool(op_desc, REFRESH_CONTINUOUS_FLAG, flag) || !flag) {
    return SUCCESS;
  }

  FE_LOGD("Set continuous dtype for %s.", node->GetName().c_str());
  for (size_t i = 0; i < op_desc->GetAllOutputsDescSize(); ++i) {
    int64_t format_continuous = 0;
    auto tensor = op_desc->MutableOutputDesc(i);
    if (tensor == nullptr) {
      FE_LOGW("Output %zu of %s is null.", i, node->GetName().c_str());
      continue;
    }
    if (!ge::AttrUtils::GetInt(tensor, FORMAT_CONTINUOUS, format_continuous) || format_continuous == 0) {
      continue;
    }
    FE_LOGD("Output %zu needs continuous dtype.", i);
    /* Get the peer in dtype */
    FE_CHECK_NOTNULL(node->GetOutDataAnchor(i));
    auto peer_in_anchors = node->GetOutDataAnchor(i)->GetPeerInDataAnchors();
    if (peer_in_anchors.empty()) {
      FE_LOGW("Peer in data anchor is empty for output %zu of %s.", i, node->GetName().c_str());
      continue;
    }

    /* Pick the first one of peer in anchors */
    auto peer_in_anchor = peer_in_anchors.at(0);
    auto peer_in_node = peer_in_anchor->GetOwnerNode();
    auto peer_in_op_desc = peer_in_node->GetOpDesc();
    auto peer_in_index = peer_in_anchor->GetIdx();
    FE_CHECK_NOTNULL(peer_in_op_desc);
    auto input_desc_ptr = peer_in_op_desc->MutableInputDesc(peer_in_index);
    FE_CHECK_NOTNULL(input_desc_ptr);
    ge::DataType peer_in_dtype = input_desc_ptr->GetDataType();
    ge::DataType output_dtype = tensor->GetDataType();
    if (peer_in_dtype != output_dtype) {
      FE_LOGD("Output %zu and its peer in dtype is not equal. Dtypes are %u and %u.", i, output_dtype, peer_in_dtype);
      SetAllTesnorWithDtype(op_desc, peer_in_dtype);
      /* Only care about the first output with format_continuous. */
      break;
    }
  }
  return SUCCESS;
}

Status GraphFusion::SetContinuousDtypeForOutput(const ge::ComputeGraph &graph) const {
  auto nodes = graph.GetDirectNode();
  for (auto &node : nodes) {
    if (SetContinuousDtypeForSingleNode(node) != SUCCESS) {
      return FAILED;
    }
  }
  /* Loop using reverse post order */
  FE_LOGD("Set format continuous in reversed post order.");
  auto size_of_all_nodes = nodes.size();
  if (size_of_all_nodes == 0) {
    FE_LOGW("Graph %s contains no node.", graph.GetName().c_str());
    return SUCCESS;
  }
  for (int loop_index = static_cast<int>(size_of_all_nodes - 1); loop_index >= 0; loop_index--) {
    if (SetContinuousDtypeForSingleNode(nodes.at(loop_index)) != SUCCESS) {
      return FAILED;
    }
  }

  return SUCCESS;
}

Status GraphFusion::FusionQuantOpOfEachGraph(ge::ComputeGraph &graph) {
  Status ret = GraphFusionQuantByPass(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionQuantOp] Graph[%s]: quant fusion failed, ret %u.", graph.GetName().c_str(), ret);
    return FAILED;
  }
  FE_LOGI("Graph[%s]: quant fusion successfully.", graph.GetName().c_str());
  return SUCCESS;
}

void GraphFusion::ReportAfterCheckGraphCycle(const ge::ComputeGraph &graph,
                                             const FusionPassOrRule &pass_or_rule) const {
  if (pass_or_rule.method == RULE_METHOD) {
    REPORT_FE_ERROR("[GraphOpt] Graph [%s] is cyclic; topological sorting failed after graph fusion with ReLU %s.",
                    graph.GetName().c_str(), pass_or_rule.name.c_str());
    FeGraphUtils::DumpGraphAndOnnx(graph, "RunOneRuleFusion_Graph_Cyclic");
  } else if (pass_or_rule.method == PASS_METHOD) {
    REPORT_FE_ERROR("[GraphOpt] Graph %s is cyclic. Failed to do topological sorting after graph fusion by pass %s.",
                    graph.GetName().c_str(), pass_or_rule.name.c_str());
    FeGraphUtils::DumpGraphAndOnnx(graph, "RunOnePassFusion_Graph_Cyclic");
  } else {
    FE_LOGW("Unknown fusion method:%s.", pass_or_rule.method.c_str());
  }
}

Status GraphFusion::RunOneRuleFusion(ge::ComputeGraph &graph, const FusionPassOrRule &pass_or_rule) {
  RuleType rule_type = static_cast<RuleType>(pass_or_rule.type);

  int32_t priority = FusionPriorityManager::GetRealPriority(pass_or_rule.priority);
  // start to run one single rule
  if (priority < CUSTOM_PASS_PRIORITY_MIN) {
    FE_LOGD("Start to match and run rule fusion, rule name:%s, rule type:%s, configured priority:%d, engine:%s.",
            pass_or_rule.name.c_str(), GetRuleTypeString(rule_type).c_str(), priority, engine_name_.c_str());
  } else {
    FE_LOGD("Start to match and run rule fusion, rule name:%s, rule type:%s, default priority:%d, engine:%s.",
            pass_or_rule.name.c_str(), GetRuleTypeString(rule_type).c_str(), priority, engine_name_.c_str());
  }
  Status ret = fusion_rule_mgr_ptr_->RunGraphFusionRuleByType(graph, rule_type, pass_or_rule.name);
  if (ret != SUCCESS && ret != NOT_CHANGED && ret != ge::NOT_CHANGED) {
    REPORT_FE_ERROR("[GraphOpt][FirstRoundFusion] Failed to run graph fusion rule: %s",
                    pass_or_rule.name.c_str());
    return ret;
  }
  if (Configuration::Instance(AI_CORE_NAME).IsEnableNetworkAnalysis()) {
    if (CheckGraphCycle(graph)) {
      ReportAfterCheckGraphCycle(graph, pass_or_rule);
      return FAILED;
    }
  }
  FE_LOGD("Run graph fusion rule successfully, rule name:%s.", pass_or_rule.name.c_str());
  return SUCCESS;
}

Status GraphFusion::RunOnePassFusion(ge::ComputeGraph &graph, const FusionPassOrRule &pass_or_rule,
                                    const std::map<std::string, ge::fusion::CreateFusionPassFn> &ge_pass_map) {
  auto pass_type = static_cast<GraphFusionPassType>(pass_or_rule.type);

  int32_t priority = FusionPriorityManager::GetRealPriority(pass_or_rule.priority);
  // start to run one single pass
  if (priority < CUSTOM_PASS_PRIORITY_MIN) {
    FE_LOGD("Start to match and run pass fusion, pass name:%s, pass type:%s, configured priority:%d, engine:%s.",
            pass_or_rule.name.c_str(), GetPassTypeString(pass_type).c_str(), priority, engine_name_.c_str());
  } else {
    FE_LOGD("Start to match and run pass fusion, pass name:%s, pass type:%s, default priority:%d, engine:%s.",
            pass_or_rule.name.c_str(), GetPassTypeString(pass_type).c_str(), priority, engine_name_.c_str());
  }
  string graph_name = graph.GetName();
  string pass_type_str = GetPassTypeString(pass_type);
  Status ret = SUCCESS;
  if (pass_type == CUSTOM_AI_CORE_GRAPH_PASS || pass_type == CUSTOM_VECTOR_CORE_GRAPH_PASS ||
      pass_type == BUILT_IN_GRAPH_PASS || pass_type == BUILT_IN_VECTOR_CORE_GRAPH_PASS) {
    auto ge_pass = ge_pass_map.find(pass_or_rule.name);
    if (ge_pass != ge_pass_map.end()) {
      FE_LOGI("Begin running the open-source registration pass, pass name:%s, pass type:%s",
               pass_or_rule.name.c_str(), GetPassTypeString(pass_type).c_str());
      ge::GraphPtr ge_graph = ge::GraphUtilsEx::CreateGraphPtrFromComputeGraph(graph.shared_from_this());
      ge::CustomPassContext context;
      FE_CHECK_NOTNULL(ge_pass->second);
      context.SetPassName(ge_pass->first.c_str());
      ret = ge_pass->second()->Run(ge_graph, context);
      if (ret != SUCCESS) {
        int64_t run_count_attr;
        (void)ge::AttrUtils::GetInt(graph, "run_count", run_count_attr);
        (void)ge::AttrUtils::SetInt(graph, "run_count", ++run_count_attr);
      }
    } else {
      auto pattern_fusion_base_pass_ptr = std::unique_ptr<PatternFusionBasePass>(
          dynamic_cast<PatternFusionBasePass *>(pass_or_rule.pass_desc.create_fn()));
      FE_CHECK(pattern_fusion_base_pass_ptr == nullptr,
              REPORT_FE_ERROR("[GraphOpt][FirstRoundFusion] Graph[%s], Pass[%s, %s]: the pattern fusion is nullptr.",
                              graph_name.c_str(), pass_or_rule.name.c_str(), pass_type_str.c_str()),
              return FAILED);
      pattern_fusion_base_pass_ptr->SetName(pass_or_rule.name);
      ge::TraceOwnerGuard guard(FE_MODULE_NAME, pass_or_rule.name, graph.GetName());
      ret = pattern_fusion_base_pass_ptr->Run(graph, this->ops_kernel_info_store_ptr_);
    }
  } else {
    int64_t run_count_attr;
    (void)ge::AttrUtils::GetInt(graph, "run_count", run_count_attr);
    (void)ge::AttrUtils::SetInt(graph, "run_count", ++run_count_attr);
  }
  if (ret != SUCCESS && ret != NOT_CHANGED && ret != ge::NOT_CHANGED && ret != GRAPH_FUSION_CYCLE) {
    ErrorMessageDetail err_msg(EM_RUN_PASS_FAILED, {pass_or_rule.name, GetPassTypeString(pass_type)});
    ReportErrorMessage(err_msg);
    REPORT_FE_ERROR("[GraphOpt][FirstRoundFusion] Failed to run graph fusion pass [%s, %s]. Return value is %u.",
                    pass_or_rule.name.c_str(), GetPassTypeString(pass_type).c_str(), ret);
    return ret;
  } else if (ret == GRAPH_FUSION_CYCLE) {
    ReportAfterCheckGraphCycle(graph, pass_or_rule);
    return FAILED;
  }
  FE_LOGD("Run graph fusion pass successfully, pass name:%s, pass type:%s.", pass_or_rule.name.c_str(),
          GetPassTypeString(pass_type).c_str());
  return SUCCESS;
}

Status GraphFusion::RunOnePassFusionByType(ge::ComputeGraph &graph, const FusionPassOrRule &pass_or_rule,
                                          const GraphFusionPassType &pass_type,
                                          const std::map<std::string, ge::fusion::CreateFusionPassFn> &ge_pass_map) {
  Status ret = SUCCESS;
  auto ge_pass = ge_pass_map.find(pass_or_rule.name);
  if (ge_pass != ge_pass_map.end()) {
    FE_LOGI("Begin running the open-source registration pass, pass name:%s, pass type:%s",
              pass_or_rule.name.c_str(), GetPassTypeString(pass_type).c_str());
    ge::GraphPtr ge_graph = ge::GraphUtilsEx::CreateGraphPtrFromComputeGraph(graph.shared_from_this());
    ge::CustomPassContext context;
    FE_CHECK_NOTNULL(ge_pass->second);
    context.SetPassName(ge_pass->first.c_str());
    ret = ge_pass->second()->Run(ge_graph, context);
  } else {
    auto pattern_fusion_base_pass_ptr = std::unique_ptr<PatternFusionBasePass>(
        dynamic_cast<PatternFusionBasePass *>(pass_or_rule.pass_desc.create_fn()));
    if (pattern_fusion_base_pass_ptr == nullptr) {
      REPORT_FE_ERROR(
          "[GraphOptJdgInst][RunGraphFusion] Graph[%s], Pass[%s, %s]: the pattern_fusion_ptr is nullptr.",
          graph.GetName().c_str(), GetPassTypeString(pass_type).c_str(), pass_or_rule.name.c_str());
      return FAILED;
    }
    pattern_fusion_base_pass_ptr->SetName(pass_or_rule.name);
    ge::TraceOwnerGuard guard(FE_MODULE_NAME, pass_or_rule.name, graph.GetName());
    ret = pattern_fusion_base_pass_ptr->Run(graph, this->ops_kernel_info_store_ptr_);
  }
  return ret;
}

Status GraphFusion::RunBuiltInFusionByType(ge::ComputeGraph &graph, GraphFusionPassType pass_type) {
  bool is_single_op_scene = false;
  (void) ge::AttrUtils::GetBool(graph, ge::ATTR_SINGLE_OP_SCENE, is_single_op_scene);
  FE_LOGD("The attr is single op scene of graph[%s] is [%d].", graph.GetName().c_str(), is_single_op_scene);
  const std::vector<FusionPassOrRule> &sorted_graph_fusion_vec =
          fusion_priority_mgr_ptr_->GetSortedGraphFusionList(is_single_op_scene);
  if (sorted_graph_fusion_vec.empty()) {
    FE_LOGD("There is no registered graph fusion pass or rule.");
  }

  // 1. get all create_fns
  auto ge_pass_map = GetGePassMapFromRegistry();
  for (const FusionPassOrRule &pass_or_rule : sorted_graph_fusion_vec) {
    auto pass_type_curr = static_cast<GraphFusionPassType>(pass_or_rule.type);
    Status ret = SUCCESS;
    if (pass_or_rule.method == PASS_METHOD && pass_type_curr == pass_type) {
      // 2. run each pass
      int32_t priority = FusionPriorityManager::GetRealPriority(pass_or_rule.priority);
      if (priority < CUSTOM_PASS_PRIORITY_MIN) {
        FE_LOGD("Start to match and run pass fusion, pass name:%s, pass type:%s, configured priority:%d, engine:%s.",
                pass_or_rule.name.c_str(), GetPassTypeString(pass_type).c_str(), priority, engine_name_.c_str());
      } else {
        FE_LOGD("Start to match and run pass fusion, pass name:%s, pass type:%s, default priority:%d, engine:%s.",
                pass_or_rule.name.c_str(), GetPassTypeString(pass_type).c_str(), priority, engine_name_.c_str());
      }
      if (pass_type == SECOND_ROUND_BUILT_IN_GRAPH_PASS &&
          IsPassAttrTypeOn(pass_or_rule.pass_desc.attr, PassAttrType::NEED_TOPO_SORT) &&
          graph.TopologicalSorting() != ge::GRAPH_SUCCESS) {
        REPORT_FE_ERROR("[GraphOptJdgInst][RunGraphFusion][RunBuiltInFusion] TopologicalSorting failed.");
        return FAILED;
      }
      // start to run one single pass
      FE_TIMECOST_START(RunOpsFusionPass);
      ret = RunOnePassFusionByType(graph, pass_or_rule, pass_type, ge_pass_map);
      if (ret != SUCCESS && ret != NOT_CHANGED && ret != ge::NOT_CHANGED) {
        ErrorMessageDetail err_msg(EM_RUN_PASS_FAILED, {pass_or_rule.name, GetPassTypeString(pass_type)});
        ReportErrorMessage(err_msg);
        REPORT_FE_ERROR(
            "[GraphOptJdgInst][RunGraphFusion][RunBuiltInFusion] Failed to run graph fusion pass [%s, %s].",
            pass_or_rule.name.c_str(), GetPassTypeString(pass_type).c_str());
        return ret;
      }
      FE_LOGD("Run graph fusion pass successfully, pass name:%s, pass type:%s.", pass_or_rule.name.c_str(),
              GetPassTypeString(pass_type).c_str());
      string out_s = "GraphFusion::RunOpsFusionPass pass_name: " + pass_or_rule.name;
      FE_TIMECOST_END_LOGI(RunOpsFusionPass, out_s.c_str());
    }
  }
  return SUCCESS;
}

uint32_t GraphFusion::JudgeQuantMode(const ge::ComputeGraph &graph) const {
  ISAArchVersion isa_arch_version = PlatformUtils::Instance().GetIsaArchVersion();
  if (isa_arch_version == ISAArchVersion::EN_ISA_ARCH_V100 || isa_arch_version == ISAArchVersion::EN_ISA_ARCH_V300 ||
      isa_arch_version == ISAArchVersion::EN_ISA_ARCH_V350) {
    FE_LOGD("IsaArchVersion is V100 or V300 or V350.");
    return SUCCESS;
  }
  for (auto &node : graph.GetAllNodes()) {
    FE_CHECK(node == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionQuantOp] dequant node is null."), return FAILED);
    if (node->GetType() != ASCEND_DEQUANT) {
      continue;
    }

    // get Dequant deq_scale_tensor(64)
    ge::InDataAnchorPtr weight_in_anchor = node->GetInDataAnchor(1);
    if (weight_in_anchor == nullptr) {
      REPORT_FE_ERROR("[GraphOpt][FusionQuantOp] Node[%s]: get weight_in_anchor failed.", node->GetName().c_str());
      return PARAM_INVALID;
    }
    ge::OutDataAnchorPtr weight_peer_out_anchor = weight_in_anchor->GetPeerOutAnchor();
    if (weight_peer_out_anchor == nullptr) {
      REPORT_FE_ERROR("[GraphOpt][FusionQuantOp] Node[%s]: get weight_peer_out_anchor failed.",
                      node->GetName().c_str());
      return PARAM_INVALID;
    }
    ge::NodePtr weight_node = weight_peer_out_anchor->GetOwnerNode();
    if (weight_node == nullptr) {
      REPORT_FE_ERROR("[GraphOpt][FusionQuantOp] Node[%s]: get weight_node failed.", node->GetName().c_str());
      return PARAM_INVALID;
    }
    vector<ge::GeTensorPtr> weights_dequant = ge::OpDescUtils::MutableWeights(weight_node);
    if (weights_dequant.size() != 1) {
      REPORT_FE_ERROR("[GraphOpt][FusionQuantOp] weights_dequant is invalid, size %zu.", weights_dequant.size());
      return PARAM_INVALID;
    }
    ge::GeTensorPtr deq_scale_tensor = weights_dequant[0];
    FE_CHECK(deq_scale_tensor == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionQuantOp] deqScale is nullptr."),
             return PARAM_INVALID);

    // translate deq_scale_tensor to scale_deq[32:63], N[24:31], offset_w[16:23]
    std::uint8_t *data = const_cast<uint8_t *>(deq_scale_tensor->GetData().data());
    uint64_t *deq_scale_data = reinterpret_cast<uint64_t *>(data);
    FE_CHECK(deq_scale_data == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionQuantOp] deqScaleData is nullptr"),
             return PARAM_INVALID);
    const ge::GeShape &deq_scale_shape = deq_scale_tensor->GetTensorDesc().GetShape();
    if (deq_scale_shape.GetDimNum() != 1) {
      REPORT_FE_ERROR("[GraphOpt][FusionQuantOp] deqScale shape error, shape is %zu.", deq_scale_shape.GetDimNum());
      return PARAM_INVALID;
    }
    // set default value
    (void)ge::AttrUtils::SetStr(node->GetOpDesc(), STR_QUANT_MODE, STR_QUANT_HIGH_PRECISION);
    int64_t deq_co = deq_scale_shape.GetDim(0);
    for (int64_t i = 0; i < deq_co; i++) {
      int8_t deq_n = GET_DEQUANT_N(deq_scale_data[i]);
      FE_LOGD("deq_scale N value[%ld] is %d.", i, deq_n);
      if (deq_n != 0) {
        (void)ge::AttrUtils::SetStr(node->GetOpDesc(), STR_QUANT_MODE, STR_QUANT_HIGH_PERFORMANCE);
        break;
      }
    }
    FE_LOGD("Set quant mode to node [%s] successfully.", node->GetName().c_str());
  }
  return SUCCESS;
}

void GraphFusion::MapPassByType(std::map<GraphFusionPassType, std::vector<FusionPassOrRule>> &quant_pass_map,
                                ge::ComputeGraph &graph) {
  bool is_single_op_scene = false;
  (void) ge::AttrUtils::GetBool(graph, ge::ATTR_SINGLE_OP_SCENE, is_single_op_scene);
  FE_LOGD("The attr is single op scene of graph[%s] is [%d].", graph.GetName().c_str(), is_single_op_scene);
  const std::vector<FusionPassOrRule> &sorted_graph_fusion_vec =
      fusion_priority_mgr_ptr_->GetSortedGraphFusionList(is_single_op_scene);
  if (sorted_graph_fusion_vec.empty()) {
    FE_LOGI("There is no registered graph fusion pass or rule.");
  }
  // map quant pass by type
  for (const FusionPassOrRule &pass_or_rule : sorted_graph_fusion_vec) {
    auto pass_type_curr = static_cast<GraphFusionPassType>(pass_or_rule.type);
    auto it = quant_pass_map.find(pass_type_curr);
    if (it != quant_pass_map.end()) {
      it->second.emplace_back(pass_or_rule);
    }
  }
  return;
}

uint32_t GraphFusion::GraphFusionQuantByPass(ge::ComputeGraph &graph) {
  // init quant pass map
  std::vector<FusionPassOrRule> quant_pass_empty_vec;
  std::map<GraphFusionPassType, std::vector<FusionPassOrRule>> quant_pass_map;
  for (auto quant_pass_type : GRAPH_FUSION_QUANT_PASS_VEC) {
    quant_pass_map[quant_pass_type] = quant_pass_empty_vec;
  }
  MapPassByType(quant_pass_map, graph);
  // run tf merge sub graph pass
  Status ret = SUCCESS;
  auto ge_pass_map = GetGePassMapFromRegistry();
  for (const FusionPassOrRule &quant_pass_curr : quant_pass_map[BUILT_IN_TF_MERGE_SUB_GRAPH_PASS]) {
    auto pass_type_curr = static_cast<GraphFusionPassType>(quant_pass_curr.type);
    FE_LOGD("Start Graph Fusion:%s Owner:%s Method:%s Priority:%d", quant_pass_curr.name.c_str(),
            GetPassTypeString(pass_type_curr).c_str(), quant_pass_curr.method.c_str(),
            FusionPriorityManager::GetRealPriority(quant_pass_curr.priority));
    ret = RunOnePassFusionByType(graph, quant_pass_curr, pass_type_curr, ge_pass_map);
    if (ret != SUCCESS && ret != NOT_CHANGED && ret != ge::NOT_CHANGED) {
      REPORT_FE_ERROR("[GraphOpt][FusionQuantOp] Graph[%s]: Run tf_merge fusion pass failed.", graph.GetName().c_str());
      return ret;
    }
  }
  // judge high precision or high performance
  if (JudgeQuantMode(graph) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionQuantOp] Graph[%s]:Set quant mode failed.", graph.GetName().c_str());
    return FAILED;
  }
  // add quant pass by version
  const ISAArchVersion isa_version = PlatformUtils::Instance().GetIsaArchVersion();
  std::vector<FusionPassOrRule> quant_pass_vec = quant_pass_map[BUILT_IN_QUANT_OPTIMIZATION_GRAPH_PASS];
  std::vector<FusionPassOrRule> tmp_vec;
  if ((isa_version != ISAArchVersion::EN_ISA_ARCH_V300 && isa_version != ISAArchVersion::EN_ISA_ARCH_V350 &&
       isa_version != ISAArchVersion::EN_ISA_ARCH_V220)) {
    tmp_vec = quant_pass_map[BUILT_IN_EN_ISA_ARCH_EXC_V300_AND_V220_GRAPH_PASS];
    quant_pass_vec.insert(quant_pass_vec.end(), tmp_vec.begin(), tmp_vec.end());
  }
  if (isa_version == ISAArchVersion::EN_ISA_ARCH_V100) {
    tmp_vec = quant_pass_map[BUILT_IN_EN_ISA_ARCH_V100_GRAPH_PASS];
    quant_pass_vec.insert(quant_pass_vec.end(), tmp_vec.begin(), tmp_vec.end());
  } else if (isa_version == ISAArchVersion::EN_ISA_ARCH_V200) {
    tmp_vec = quant_pass_map[BUILT_IN_EN_ISA_ARCH_V200_GRAPH_PASS];
    quant_pass_vec.insert(quant_pass_vec.end(), tmp_vec.begin(), tmp_vec.end());
  }
  // add delete no const folding pass
  tmp_vec = quant_pass_map[BUILT_IN_DELETE_NO_CONST_FOLDING_GRAPH_PASS];
  quant_pass_vec.insert(quant_pass_vec.end(), tmp_vec.begin(), tmp_vec.end());

  return RunQuantPass(graph, quant_pass_vec);
}

uint32_t GraphFusion::RunQuantPass(ge::ComputeGraph &graph, std::vector<FusionPassOrRule> &quant_pass_vec) {
  // run quant pass
  auto ge_pass_map = GetGePassMapFromRegistry();
  for (const FusionPassOrRule &quant_pass_curr : quant_pass_vec) {
    auto pass_type_curr = static_cast<GraphFusionPassType>(quant_pass_curr.type);
    FE_LOGD("Start Graph Fusion:%s Owner:%s Method:%s Priority:%d.", quant_pass_curr.name.c_str(),
            GetPassTypeString(pass_type_curr).c_str(), quant_pass_curr.method.c_str(),
            FusionPriorityManager::GetRealPriority(quant_pass_curr.priority));
    auto ret = RunOnePassFusionByType(graph, quant_pass_curr, pass_type_curr, ge_pass_map);
    if (ret != SUCCESS && ret != NOT_CHANGED && ret != ge::NOT_CHANGED) {
      REPORT_FE_ERROR("[GraphOpt][FusionQuantOp] Graph[%s]: Run fusion pass failed.", graph.GetName().c_str());
      return ret;
    }
  }
  return SUCCESS;
}

bool GraphFusion::CheckGraphCycle(ge::ComputeGraph &graph) const {
  Status ret = graph.TopologicalSorting();
  if (ret != ge::GRAPH_SUCCESS) {
    return true;
  }
  return false;
}
}  // namespace fe
