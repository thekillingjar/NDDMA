/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flatten_concat_pass.h"
#include "common/checker.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/type/tensor_type_impl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "register/shape_inference.h"
#include "utils/cg_utils.h"
#include "utils/auto_fuse_config.h"
#include "lowering/asc_lowerer/loop_common.h"
#include "backend/backend_spec.h"
#include "base/err_msg.h"

namespace ge {
const string CONCAT = "Concat";
const string CONCATD = "ConcatD";
const string CONCATV2D = "ConcatV2D";
const string CONCATV2 = "ConcatV2";

namespace {
ge::NodePtr CreateNewConcatD(const ComputeGraphPtr &graph, uint32_t inputcnt_per_concat, NodePtr &concat_end_node,
                               int64_t original_concatdim) {
  std::string new_concatnode_name = "Fusion_" + concat_end_node->GetName();
  auto op = ge::OperatorFactory::CreateOperator(new_concatnode_name.c_str(), "ConcatD");
  op.BreakConnect();
  op.DynamicInputRegister("x", inputcnt_per_concat);
  op.SetAttr("concat_dim", original_concatdim);
  op.SetAttr("N", inputcnt_per_concat);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto new_node = graph->AddNode(op_desc);
  GELOGD("create new concat node:input count num %u; concat node name is %s end",
         inputcnt_per_concat, new_concatnode_name.c_str());
  return new_node;
}

bool GetConcatNodeConcatDim(const NodePtr &concat_node, int64_t &node_concat_dimvalue) {
  int64_t concat_dim = 0;
  auto op_desc = concat_node->GetOpDesc();
  auto output_desc_ptr = op_desc->GetOutputDescPtr(0);
  if (output_desc_ptr->GetShape().IsUnknownDimNum()) {
    return false;
  }
  auto out_dim_num = static_cast<int64_t>(output_desc_ptr->GetShape().GetDimNum());
  GELOGD("concat node shape dim num is [%ld]", out_dim_num);
  if ((CONCAT == concat_node->GetType()) || (CONCATV2 == concat_node->GetType())) {
    int32_t index = op_desc->GetInputIndexByName("concat_dim");
    auto op = ge::OpDescUtils::CreateOperatorFromNode(concat_node);
    const GeTensor *perm_tensor = ge::OpDescUtils::GetInputConstData(op, index);
    GE_WARN_ASSERT(perm_tensor != nullptr, "concat dim input is not const data");

    const auto &tensor_desc = perm_tensor->GetTensorDesc();
    GE_ASSERT_TRUE((tensor_desc.GetShape().GetShapeSize() == 1) || (tensor_desc.GetShape().IsScalar()));

    if (tensor_desc.GetDataType() == DT_INT32) {
      const auto *perm_data = reinterpret_cast<const int32_t *>(perm_tensor->GetData().data());
      concat_dim = static_cast<int64_t>(perm_data[0]);
    } else if (tensor_desc.GetDataType() == DT_INT64) {
      const auto *perm_data = reinterpret_cast<const int64_t *>(perm_tensor->GetData().data());
      concat_dim = static_cast<int64_t>(perm_data[0]);
    } else {
      REPORT_INNER_ERR_MSG("E19999", "Remove node:%s(%s)", concat_node->GetName().c_str(), concat_node->GetType().c_str());
      return false;
    }
  } else {
    (void)ge::AttrUtils::GetInt(concat_node->GetOpDesc(), "concat_dim", concat_dim);
  }

  if (concat_dim < 0) {
    concat_dim += (out_dim_num);  // 如果拼接轴是负数，表示是倒数第N个轴，将其转变为正数的轴。
  }
  node_concat_dimvalue = concat_dim;
  return true;
}

uint32_t PeerOutNodeFuseParaCal(const NodePtr &concat_node, const NodePtr &peer_out_node, int64_t original_concatdim,
                                uint32_t &concat_node_flag, std::vector<uint32_t> &concat_node_flagvec) {
  (void)concat_node;
  int64_t concat_dimvalue = 0;
  uint32_t peerout_node_concatnum = 1;
  if ((CONCATV2 == peer_out_node->GetType()) || (CONCATV2D == peer_out_node->GetType()) ||
      (peer_out_node->GetType() == CONCAT) || (peer_out_node->GetType() == CONCATD)) {
    auto nodeout_anchorsize = peer_out_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size();
    auto concat_dim_can_get_flag = GetConcatNodeConcatDim(peer_out_node, concat_dimvalue);
    GELOGD("fuse concat node info:original concat dim is %d, fuse concat dim is %d", original_concatdim,
           concat_dimvalue);
    if ((original_concatdim != concat_dimvalue) || (nodeout_anchorsize != 1) || (concat_dim_can_get_flag != true)) {
      concat_node_flag = static_cast<uint32_t>(0);
      peerout_node_concatnum = static_cast<uint32_t>(1);
    } else {
      concat_node_flag = static_cast<uint32_t>(1);
      (void)ge::AttrUtils::GetInt(peer_out_node->GetOpDesc(), "N", peerout_node_concatnum);
    }
    GELOGD("fuse concat node info:nodeout anchorsize[%d], concat dim can get flag[%d], final concat flag[%d]",
           nodeout_anchorsize, concat_dim_can_get_flag, concat_node_flag);     
  }
  concat_node_flagvec.push_back(concat_node_flag);
  return static_cast<uint32_t>(peerout_node_concatnum);
}

Status ConcatNodeNeedCombineProcess(const NodePtr &in_ower_node, uint32_t &fusion_newnode_anchorIdx,
                                    std::vector<int32_t> &inputs_map, uint32_t inanchor_index_inputs_map,
                                    std::vector<InDataAnchorPtr> &new_concat_in_data_anchors) {
  int32_t input_index_temp;
  uint32_t indata_anchor_size = static_cast<uint32_t>(in_ower_node->GetAllInDataAnchorsSize());

  for (uint32_t index = 0; index < indata_anchor_size; index++) {
    input_index_temp = in_ower_node->GetOpDesc()->GetInputIndexByName("x" + std::to_string(index));
    if (input_index_temp == (-1)) {
      break;
    }
    inputs_map.push_back(static_cast<int32_t>(inanchor_index_inputs_map) + input_index_temp);
    new_concat_in_data_anchors.push_back(in_ower_node->GetInDataAnchor(input_index_temp));
    fusion_newnode_anchorIdx++;
  }
  return GRAPH_SUCCESS;
}

void NewConcatNodeUpdateDesc(NodePtr &concat_node, NodePtr &new_concat_node,
                             std::vector<InDataAnchorPtr> &new_concat_in_data_anchors) {
  uint32_t new_node_input_index = 0;
  for (const auto &in_data_anchor : new_concat_in_data_anchors) {
    auto input_anchor_idx = in_data_anchor->GetIdx();
    auto input_desc_x = in_data_anchor->GetOwnerNode()->GetOpDesc()->GetInputDesc(input_anchor_idx);
    new_concat_node->GetOpDesc()->UpdateInputDesc(new_node_input_index, input_desc_x);
    new_node_input_index++;
  }

  const auto output_desc_y = concat_node->GetOpDesc()->GetOutputDesc(0);
  new_concat_node->GetOpDesc()->UpdateOutputDesc(0, output_desc_y);
  return;
}

Status ConcatNodeRealizeCombine(const ComputeGraphPtr &graph, NodePtr &concat_node, NodePtr &new_concat_node,
                                const std::vector<uint32_t> &concat_node_flagVec,
                                const std::vector<int> &input_indexs) {
  uint32_t flag_index = 0;
  uint32_t fusion_newnode_anchorIdx = 0;
  std::vector<NodePtr> new_nodes_lists;
  std::vector<NodePtr> old_nodes_lists;
  std::vector<int32_t> inputs_map;
  std::vector<int32_t> outputs_map;
  int32_t inanchor_index_inputs_map = concat_node->GetAllInDataAnchorsSize();
  std::vector<InDataAnchorPtr> new_concat_in_data_anchors;

  old_nodes_lists.push_back(concat_node);
  new_nodes_lists.push_back(new_concat_node);
  for (auto indata_anchor_index : input_indexs) {
    auto in_ower_node = NodeUtils::GetInDataNodeByIndex(*concat_node, indata_anchor_index);
    GE_ASSERT_NOTNULL(in_ower_node);

    GELOGD("concat node input anchor info:peerout node name is %s, fusion new node anchorIdx is %d",
           in_ower_node->GetName().c_str(), fusion_newnode_anchorIdx);

    if (concat_node_flagVec[flag_index] == 0) {
      inputs_map.push_back(indata_anchor_index);
      new_concat_in_data_anchors.push_back(concat_node->GetInDataAnchor(indata_anchor_index));
      fusion_newnode_anchorIdx++;
    } else {
      GE_ASSERT_SUCCESS(ConcatNodeNeedCombineProcess(in_ower_node, fusion_newnode_anchorIdx, inputs_map,
                                                     inanchor_index_inputs_map, new_concat_in_data_anchors));
      old_nodes_lists.push_back(in_ower_node);
      inanchor_index_inputs_map += in_ower_node->GetAllInDataAnchorsSize();
    }
    flag_index++;
  }
  outputs_map.push_back(0);
  GE_CHK_STATUS(GraphUtils::ReplaceNodesDataAnchors(new_nodes_lists, old_nodes_lists, inputs_map, outputs_map));
  GE_CHK_STATUS(GraphUtils::InheritExecutionOrder(new_nodes_lists, old_nodes_lists, graph));

  NewConcatNodeUpdateDesc(concat_node, new_concat_node, new_concat_in_data_anchors);
  for (auto node : old_nodes_lists) {
    if (graph->RemoveNode(node) != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Remove node:%s(%s) from graph:%s failed", new_concat_node->GetName().c_str(),
                        node->GetType().c_str(), graph->GetName().c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

Status ConcatNodeCombine(const ComputeGraphPtr &graph, NodePtr &concat_node) {
  uint32_t fusion_incount_total = 0;
  uint32_t currfuse_concatCnt = 0;
  NodePtr in_owner_node = nullptr;
  NodePtr new_concat_node;
  uint32_t concat_node_Flag = 0;
  uint32_t fuse_flag = 0;
  int64_t original_concatdim = 0;

  int32_t input_index_temp;
  std::vector<int> input_indexs;
  std::vector<uint32_t> concat_node_flag_vec;
  uint32_t indata_anchor_size = concat_node->GetAllInDataAnchorsSize();
  std::vector<NodePtr> cycle_nodes_lists;

  if (!GetConcatNodeConcatDim(concat_node, original_concatdim)) {
    return GRAPH_SUCCESS;
  }

  for (uint32_t index = 0; index < indata_anchor_size; index++) {
    input_index_temp = concat_node->GetOpDesc()->GetInputIndexByName("x" + std::to_string(index));
    if (input_index_temp == (-1)) {
      break;
    }
    input_indexs.push_back(input_index_temp);
  }

  for (auto indata_anchor_index : input_indexs) {
    concat_node_Flag = 0;
    auto in_data_anchor = concat_node->GetInDataAnchor(indata_anchor_index);
    in_owner_node = in_data_anchor->GetPeerOutAnchor()->GetOwnerNode();
    currfuse_concatCnt =
        PeerOutNodeFuseParaCal(concat_node, in_owner_node, original_concatdim, concat_node_Flag, concat_node_flag_vec);
    fusion_incount_total += currfuse_concatCnt;
    fuse_flag += concat_node_Flag;
    if (concat_node_Flag != 0) {
      cycle_nodes_lists.push_back(in_owner_node);
    }
  }

  if (static_cast<int>(fuse_flag) == 0 ||
      FlattenConcatPass::CanFlatten(concat_node, original_concatdim, fusion_incount_total) != ge::GRAPH_SUCCESS) {
    return SUCCESS;
  }

    /** 判断融合节点是否成环**/
  cycle_nodes_lists.push_back(concat_node);
  const CycleDetectorSharedPtr cycle_detector = GraphUtils::CreateSharedCycleDetector(graph);
  bool has_cycle_flag = cycle_detector->HasDetectedCycle({cycle_nodes_lists});
  GELOGD("Cycle flag is : %d;", has_cycle_flag);
  if (has_cycle_flag == true) {
    return SUCCESS;
  }

  GELOGD("can fuse concat node info:concat_node_name is %s;", concat_node->GetName().c_str());
  new_concat_node = CreateNewConcatD(graph, fusion_incount_total, concat_node, original_concatdim);
  auto result = ConcatNodeRealizeCombine(graph, concat_node, new_concat_node, concat_node_flag_vec, input_indexs);
  return result;
}

bool CheckApproxFieldNum(const std::vector<Expression> &output_shape, size_t concat_dim, size_t num_inputs) {
  // 过多的symbol会导致TilingData大小膨胀，导致编译失败。在有需求削减前，先限制
  constexpr size_t kMaxFieldNum = 24UL * 1024UL / sizeof(uint32_t);
  bool can_group_concat = true;
  for (size_t i = concat_dim; i < output_shape.size(); ++i) {
    if (!output_shape[i].IsConstExpr()) {
      can_group_concat = false;
      GELOGD("dim[%zu] is dynamic, can not group concat to sub concats", i);
      break;
    }
  }
  bool check_ret = true;
  if (!can_group_concat) {
    constexpr size_t kSharedFieldNum = 10;
    std::set<std::string> free_symbols;
    for (const auto &output_dim : output_shape) {
      auto symbols = output_dim.FreeSymbols();
      for (const auto &symbol : symbols) {
        free_symbols.insert(SymbolicUtils::ToString(symbol));
      }
    }
    const auto num_free_symbols = free_symbols.size();
    const auto approx_filed_num = (num_free_symbols + kSharedFieldNum) * num_inputs;
    check_ret = approx_filed_num <= kMaxFieldNum;
    GELOGD("num_free_symbols = %zu, concat_output_shape = %s, num_inputs = %zu, check_ret = %d", num_free_symbols,
           ToString(output_shape).c_str(), num_inputs, static_cast<int32_t>(check_ret));
  }
  return check_ret;
}
}  // namespace

graphStatus FlattenConcatPass::Run(const ComputeGraphPtr &graph) const {
  if (!autofuse::AutoFuseConfig::LoweringConfig().experimental_lowering_concat) {
    GELOGI("You can enable concat by setting AUTOFUSE_FLAGS=\"--autofuse_enable_pass=concat\" and unsetting AUTOFUS_FLAGS=\"--autofuse_disable_pass=concat\"");
    return ge::GRAPH_SUCCESS;
  }
  GE_CHECK_NOTNULL(graph);
  GELOGD("ConcatProBeforeAutoFuse:main func begin");

  for (auto &node : graph->GetDirectNode()) {
    if ((node->GetType() == CONCATV2) || (node->GetType() == CONCATV2D) || (node->GetType() == CONCAT) ||
        (node->GetType() == CONCATD)) {
      GELOGD("concat node info:node name is %s, indata anchors size is %d", node->GetName().c_str(),
             node->GetAllInDataAnchorsSize());

      auto ret = ConcatNodeCombine(graph, node);
      if (ret != SUCCESS) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

graphStatus FlattenConcatPass::CanFlatten(const NodePtr &node, size_t concat_dim, size_t num_inputs) {
  constexpr size_t kMaxFusedInputNum = 800;
  constexpr size_t kMaxFreeSymbols = 16; // 过多的symbol会导致TilingData大小膨胀，导致编译失败。在有需求削减前，先限制
  constexpr int32_t kConcatAlgTranspose = 0;
  GE_CHK_BOOL_RET_SPECIAL_STATUS(num_inputs > kMaxFusedInputNum,
                                 ge::GRAPH_FAILED,
                                 "number of inputs(%zu) exceeds max input num(%zu), do not flatten concat",
                                 num_inputs, kMaxFusedInputNum);
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  GE_CHECK_NOTNULL(backend_spec);
  const auto max_single_op_input_num = backend_spec->concat_max_input_num;
  const auto may_deteriorate =
      (backend_spec->concat_alg != kConcatAlgTranspose) && (num_inputs > max_single_op_input_num);
  GE_WARN_ASSERT((!may_deteriorate),
      "concat_alg = %d, num_inputs(%zu) > max_single_op_input_num(%u), may cause deterioration, do not flatten",
      backend_spec->concat_alg, num_inputs, max_single_op_input_num);
  std::vector<Expression> output_shape;
  GE_WARN_ASSERT(ge::loop::GetBufferShape(node->GetOutDataAnchor(0), output_shape) == ge::GRAPH_SUCCESS);
  GE_WARN_ASSERT(concat_dim < output_shape.size());
  GE_WARN_ASSERT(CheckApproxFieldNum(output_shape, concat_dim, num_inputs),
                 "flatten concat will cause too many free symbols, do not flatten concat");
  const auto &concat_dim_size = output_shape[concat_dim];
  // 如果FreeSymbols多，但融合完不超过单算子能处理的上限，也可以融，只是后续不Lowering
  GE_WARN_ASSERT((num_inputs <= max_single_op_input_num) ||
      (concat_dim_size.FreeSymbols().size() <= kMaxFreeSymbols),
                 "flatten concat will cause too many free symbols, do not flatten concat, output concat dim size = %s",
                 concat_dim_size.Str().get());
  GELOGD("concat_alg = %d, num_inputs = %zu, max_single_op_input_num = %u, can_flatten = true",
         backend_spec->concat_alg, num_inputs, max_single_op_input_num);
  return ge::GRAPH_SUCCESS;
}

graphStatus FlattenConcatPass::ResolveConcatDim(const NodePtr &concat_node, size_t &concat_dim) {
  int64_t dim;
  GE_WARN_ASSERT(GetConcatNodeConcatDim(concat_node, dim));
  concat_dim = static_cast<size_t>(dim);
  return GRAPH_SUCCESS;
}
}  // namespace ge