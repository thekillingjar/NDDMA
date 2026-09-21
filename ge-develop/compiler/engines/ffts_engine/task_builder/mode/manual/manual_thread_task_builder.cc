/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "manual_thread_task_builder.h"
#include <vector>
#include <stack>
#include <unordered_set>
#include <map>
#include "inc/ffts_utils.h"
#include "common/aicore_util_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/range_vistor.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
namespace ffts {
static const uint32_t MAX_DEPTH = 7;
static const uint8_t THENBRANCHINDEX = 0U;
static const uint8_t ELSEBRANCHINDEX = 1U;
static const uint8_t CONDEXBRANCHINDEX = 0U;
static const uint8_t BODYBRANCHINDEX = 1U;
static const size_t CONTROLNODEBRANCHNUM = 2;

ManualTheadTaskBuilder::ManualTheadTaskBuilder() {}
ManualTheadTaskBuilder::~ManualTheadTaskBuilder() {}
Status ManualTheadTaskBuilder::Initialize() {
  mode_type_ = ModeType::MANUAL_MODE_TYPE;
  FFTS_MAKE_SHARED(aic_aiv_task_builder_ptr_ =
          std::make_shared<AICAIVTaskBuilder>(), return FAILED);
  FFTS_MAKE_SHARED(mix_aic_aiv_task_builder_ptr_ =
          std::make_shared<MixAICAIVTaskBuilder>(), return FAILED);
  FFTS_MAKE_SHARED(collection_ops_task_builder_ptr_ =
          std::make_shared<CollectionOpsTaskBuilder>(), return FAILED);
  FFTS_MAKE_SHARED(aicpu_task_builder_ptr_ =
          std::make_shared<AicpuTaskBuilder>(), return FAILED);
  FFTS_MAKE_SHARED(runtime_ops_task_builder_ptr_ =
          std::make_shared<RuntimeOpsTaskBuilder>(), return FAILED);
  FFTS_MAKE_SHARED(dsa_ops_task_builder_ptr_ =
          std::make_shared<DSAManualTaskBuilder>(), return FAILED);
  return SUCCESS;
}

void ManualTheadTaskBuilder::GenFftsPlusHcclId(const ge::NodePtr &node, uint32_t &contextId) const {
  if (kHCCLOpType.count(node->GetType()) == 0) {
    return;
  }
  ge::OpDescPtr op_desc = node->GetOpDesc();
  std::vector<domi::FftsPlusCtxDef> hccl_sub_tasks;
  std::vector<uint32_t> ctx_id_list;
  hccl_sub_tasks = op_desc->TryGetExtAttr(kHcclSubTasks, hccl_sub_tasks);
  for (size_t i = 0; i < hccl_sub_tasks.size(); i++) {
    ctx_id_list.push_back(contextId++);
  }
  if (!ctx_id_list.empty()) {
    (void)ge::AttrUtils::SetListInt(op_desc, kCtxIdList, ctx_id_list);
  }
  FFTS_LOGD("GenFftsPlusHcclId nodetype: %s, name: %s, ctx_id_list: %s.", op_desc->GetType().c_str(),
            op_desc->GetName().c_str(), fe::StringUtils::IntegerVecToString(ctx_id_list).c_str());
  return;
}

bool ManualTheadTaskBuilder::IsOrdinaryNode(ge::NodePtr node) const {
  auto node_type = node->GetType();
  if (node_type == "StreamActive" || node_type == "NetOutput" ||
      node_type == "LabelSet" || node_type == "LabelGoto" ||
      node_type == "LabelGotoEx" || node_type == "LabelSwitchByIndex") {
      return true;
  }
  return false;
}

void ManualTheadTaskBuilder::GenFftsPlusContextIdCommon(ge::NodePtr node,
                                                        std::vector<ge::NodePtr> &sub_graph_nodes,
                                                        uint32_t &contextId) const {
  if (IsNoCtx(node)) {
    return;
  }
  GenFftsPlusHcclId(node, contextId);
  if (kHCCLOpType.count(node->GetType()) == 0) {
    (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kContextId, contextId++);
  }
  bool is_only_add_contextid = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), kOnlyAddContext, is_only_add_contextid);
  if (!is_only_add_contextid) {
    sub_graph_nodes.push_back(node);
  }
}


Status ManualTheadTaskBuilder::GenFftsPlusContextIdWithoutControlType(std::vector<ge::NodePtr> &pre_sub_graph_nodes,
                                                                      std::vector<ge::NodePtr> &no_pre_sub_graph_nodes,
                                                                      ge::ComputeGraph &sgt_graph) const {
  // firstly, find node which precnt is 0
  for (auto node : sgt_graph.GetDirectNode()) {
    if (!node) {
      continue;
    }
    if (IsOrdinaryNode(node)) {
      continue;
    }
    ge::OpDescPtr op_desc = node->GetOpDesc();
    FFTS_LOGD("GenFftsPlusContextIdWithoutControlType: NodeType %s, Name %s.", op_desc->GetType().c_str(),
              op_desc->GetName().c_str());
    if (IsNoCtx(node)) {
      continue;
    }

    // judge node's pre_cnt is 0
    bool pre_node = true;
    for (const auto &up_node : node->GetInAllNodes()) {
      FFTS_CHECK_NOTNULL(up_node);
      if (!IsSubGraphDataWithControlEdge(up_node, kRecuriseCntMax)) {
        FFTS_LOGD("Up node is not part of the subgraph data, nodetype: %s, name: %s, up_op_desc type: %s, name: %s.",
                  op_desc->GetType().c_str(), op_desc->GetName().c_str(), up_node->GetType().c_str(),
                  up_node->GetName().c_str());
        pre_node = false;
        break;
      }
    }
    ge::NodePtr inner_graph_outputs_node = nullptr;
    inner_graph_outputs_node = op_desc->TryGetExtAttr(ATTR_NAME_PARENT_OUTPUTS_INPUT_NODES, inner_graph_outputs_node);
    if (inner_graph_outputs_node != nullptr) {
      FFTS_LOGD("Node has inner_graph_outputs_node, nodetype: %s, name: %s.", op_desc->GetType().c_str(),
                op_desc->GetName().c_str());
      pre_node = false;
    }
    /*
     *  If the node has same_memset_nodes attr, it will be inserted by atomic op, can't be as a pre node
     */
    ThreadSliceMapPtr slice_info_ptr = nullptr;
    slice_info_ptr = op_desc->TryGetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
    ge::NodePtr memset_node = nullptr;
    memset_node = op_desc->TryGetExtAttr(fe::ATTR_NAME_MEMSET_NODE, memset_node);
    if (memset_node != nullptr ||
        (slice_info_ptr != nullptr && !slice_info_ptr->same_atomic_clean_nodes.empty())) {
      FFTS_LOGD("Inserted by memset, node type: %s, name: %s.", op_desc->GetType().c_str(),
                op_desc->GetName().c_str());
      pre_node = false; // It's input will be inserted by memset
    }

    if (!pre_node) {
      continue;
    }
    pre_sub_graph_nodes.push_back(node);
  }
  // secondly, find node which precnt is non 0
  for (auto &node : sgt_graph.GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    ge::OpDescPtr op_desc = node->GetOpDesc();
    if (IsNoCtx(node)) {
      continue;
    }
    uint32_t has_set_contextId;
    if (ge::AttrUtils::GetInt(op_desc, kContextId, has_set_contextId)) {
      continue;
    }
    no_pre_sub_graph_nodes.push_back(node);
  }
  return SUCCESS;
}

Status ManualTheadTaskBuilder::GenFftsPlusContextId(ge::ComputeGraph &sgt_graph,
                                                    std::vector<ge::NodePtr> &sub_graph_nodes,
                                                    uint64_t &ready_context_num,
                                                    uint64_t &total_context_number,
                                                    std::vector<ge::NodePtr> &memset_nodes) {
  ControlGraphMap controlnode_graphmap;
  ge::ComputeGraphPtr sgt_graph_ptr = nullptr;
  std::vector<ge::NodePtr> pre_sub_graph_nodes;
  std::vector<ge::NodePtr> no_pre_sub_graph_nodes;
  std::stack<ge::NodePtr> control_node_stacks;
  FFTS_MAKE_SHARED(sgt_graph_ptr = std::make_shared<ge::ComputeGraph>(sgt_graph), return FAILED);
  FFTS_CHECK_NOTNULL(sgt_graph_ptr);
  DoWithControlOpImpl(sgt_graph_ptr, controlnode_graphmap, control_node_stacks);
  if (controlnode_graphmap.empty()) {
    FFTS_LOGD("controlnode_graphmap is empty.");
    if (GenFftsPlusContextIdWithMemSet(pre_sub_graph_nodes, memset_nodes, sgt_graph) != SUCCESS) {
      return FAILED;
    }
    if (GenFftsPlusContextIdWithoutControlType(pre_sub_graph_nodes, no_pre_sub_graph_nodes, sgt_graph) != SUCCESS) {
      return FAILED;
    }
  } else {
    FFTS_LOGD("controlnode_graphmap isn't empty.");
    RunContextPtr contextptr = nullptr;
    contextptr = sgt_graph.TryGetExtAttr(kRuntimeContentx, contextptr);
    if (contextptr == nullptr) {
      FFTS_LOGD("contextptr is null for sgt_graphname = '%s'", sgt_graph.GetName().c_str());
    } else {
      FFTS_LOGD("contextptr isn't null; sgt_graphname = '%s'.", sgt_graph.GetName().c_str());
    }
    if (DoWithFftsPlusCtxIdForControlNode(controlnode_graphmap, contextptr, control_node_stacks) != SUCCESS) {
      return FAILED;
    }
    if (GenFftsPlusContextIdWithMemSet(pre_sub_graph_nodes, memset_nodes, sgt_graph) != SUCCESS) {
      return FAILED;
    }
    if (GenFftsPlusContextIdWithoutControlType(pre_sub_graph_nodes, no_pre_sub_graph_nodes, sgt_graph) != SUCCESS) {
      return FAILED;
    }
    if (GenFftsPlusContextIdForControlNode(pre_sub_graph_nodes, no_pre_sub_graph_nodes, controlnode_graphmap)
        != SUCCESS) {
      return FAILED;
    }
  }
  if (GenFftsPlusContextIdAll(pre_sub_graph_nodes, no_pre_sub_graph_nodes,
      sub_graph_nodes, ready_context_num, total_context_number) != SUCCESS) {
    return FAILED;
  }
  FFTS_LOGD("after GenFftsPlusContextId, ready_context_num: %lu, total_context_number: %lu.", ready_context_num,
            total_context_number);
  return SUCCESS;
}

void ManualTheadTaskBuilder::GenerateAtomicCtx(std::vector<ge::NodePtr> &memset_nodes,
                                               domi::FftsPlusTaskDef *ffts_plus_task_def) const {
  if (memset_nodes.empty()) {
    return;
  }

  for (size_t i = 0; i < memset_nodes.size(); i++) {
    auto memset_node_desc = memset_nodes[i]->GetOpDesc();
    int64_t block_dim = 1;
    (void)ge::AttrUtils::GetInt(memset_node_desc, ge::TVM_ATTR_NAME_BLOCKDIM, block_dim);

    uint32_t context_id = 0;
    (void)ge::AttrUtils::GetInt(memset_node_desc, kContextId, context_id);

    std::string kernel_name;
    (void)ge::AttrUtils::GetStr(memset_node_desc, "_kernelname", kernel_name);

    domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
    ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AIV);
    ffts_plus_ctx_def->set_op_index(memset_node_desc->GetId());
    ffts_plus_ctx_def->set_op_type(domi::FftsPlusCtxDef::ATOMIC);
    ffts_plus_ctx_def->set_context_id(context_id);

    // This uniq_ctx_name for profiling parser
    ffts_plus_ctx_def->set_uniq_ctx_name(memset_node_desc->GetName() + "_" + fe::MEMSET_OP_TYPE + "_" + to_string(i));

    domi::FftsPlusAicAivCtxDef *aic_aiv_ctx_def = ffts_plus_ctx_def->mutable_aic_aiv_ctx();
    FFTS_CHECK(aic_aiv_ctx_def == nullptr, FFTS_LOGW("aic_aiv_ctx_def is nullptr."), return);
    std::vector<int64_t> work_spaces = memset_node_desc->GetWorkspace();
    for (auto &work_space : work_spaces) {
      if (work_space != 0) {
        aic_aiv_ctx_def->add_task_addr(work_space);
      }
    }
    aic_aiv_ctx_def->add_kernel_name(kernel_name);
    aic_aiv_ctx_def->set_aten(0);
    aic_aiv_ctx_def->set_successor_num(0);
    aic_aiv_ctx_def->set_thread_dim(1);
    aic_aiv_ctx_def->set_non_tail_block_dim(block_dim);
    aic_aiv_ctx_def->set_tail_block_dim(block_dim);
    uint32_t addr_size = aic_aiv_ctx_def->task_addr_size();
    uint32_t cur_addr_size = ffts_plus_task_def->addr_size();
    ffts_plus_task_def->set_addr_size(cur_addr_size + addr_size);
  }
}

void ManualTheadTaskBuilder::UpdateAtomicForOneNode(const ge::NodePtr &atomic_node, const ge::NodePtr &updated_node,
                                                    domi::FftsPlusTaskDef *ffts_plus_task_def) {
  uint32_t atomic_ctx_id = 0;
  (void)ge::AttrUtils::GetInt(atomic_node->GetOpDesc(), kContextId, atomic_ctx_id);

  uint32_t updated_ctx_id = 0;
  (void)ge::AttrUtils::GetInt(updated_node->GetOpDesc(), kContextId, updated_ctx_id);

  aic_aiv_task_builder_ptr_->UpdatePreCnt(updated_ctx_id, ffts_plus_task_def, 1);
  aic_aiv_task_builder_ptr_->UpdateSuccList(updated_ctx_id, atomic_ctx_id, ffts_plus_task_def);
}

void ManualTheadTaskBuilder::UpdateAtomicSuccList(const SameAtomicNodeMap &same_memset_nodes_map,
                                                  const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) {
  ge::NodePtr memset_node = nullptr;
  memset_node = node->GetOpDesc()->TryGetExtAttr(fe::ATTR_NAME_MEMSET_NODE, memset_node);
  if (memset_node == nullptr) {
    return;
  }

  ThreadSliceMapPtr slice_info_ptr = nullptr;
  slice_info_ptr = node->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  if (slice_info_ptr == nullptr || slice_info_ptr->same_atomic_clean_nodes.empty()) {
    return UpdateAtomicForOneNode(memset_node, node, ffts_plus_task_def);
  }
  auto iter = same_memset_nodes_map.find(slice_info_ptr->same_atomic_clean_nodes);
  if (iter == same_memset_nodes_map.end()) {
    return;
  }

  for (size_t index = 0; index < iter->second.size(); ++index) {
    UpdateAtomicForOneNode(memset_node, iter->second[index], ffts_plus_task_def);
  }
  return;
}

Status ManualTheadTaskBuilder::GenSubGraphTaskDef(std::vector<ge::NodePtr> &memset_nodes,
                                                  std::vector<ge::NodePtr> &sub_graph_nodes,
                                                  domi::TaskDef &task_def) {
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  FFTS_CHECK_NOTNULL(ffts_plus_task_def);

  GenerateAtomicCtx(memset_nodes, ffts_plus_task_def);
  uint32_t ctx_size = ffts_plus_task_def->ffts_plus_ctx_size();
  FFTS_LOGD("[GenSubGraphTaskDef] Context size: %u, memset_nodes size: %zu.", ctx_size, memset_nodes.size());
  for (auto &sub_node : sub_graph_nodes) {
    TaskBuilderType task_builder_type;
    if (GetNodeContextTypeByNode(sub_node, task_builder_type) != SUCCESS) {
      FFTS_LOGE("Node [%s, %s] failed to get task builder type.", sub_node->GetNamePtr(), sub_node->GetTypePtr());
      return FAILED;
    }
    FFTS_LOGD("Node[%s] has a task builder type of %u.", sub_node->GetNamePtr(), static_cast<uint32_t>(task_builder_type));

    FFTSPlusTaskBuilderPtr task_builder = GetTaskBuilder(task_builder_type);
    FFTS_CHECK_NOTNULL(task_builder);
    if (task_builder->GenerateTaskDef(sub_node, ffts_plus_task_def) != SUCCESS) {
      return FAILED;
    }
  }
  ctx_size = ffts_plus_task_def->ffts_plus_ctx_size();
  FFTS_LOGD("[GenSubGraphTaskDef] Context size: %u, sub_graph_nodes size: %zu.", ctx_size, sub_graph_nodes.size());
  SameAtomicNodeMap same_memset_nodes_map;
  GenerateSameAtomicNodesMap(sub_graph_nodes, same_memset_nodes_map);
  for (auto &sub_node : sub_graph_nodes) {
    FFTS_CHECK_NOTNULL(sub_node);
    auto sub_op_desc = sub_node->GetOpDesc();
    if (kHCCLOpType.count(sub_op_desc->GetType()) > 0) {
      vector<vector<int64_t>> succ_list_list;
      (void)ge::AttrUtils::GetListListInt(sub_op_desc, kSuccListList, succ_list_list);
      std::vector<uint32_t> ctx_id_list;
      (void)ge::AttrUtils::GetListInt(sub_op_desc, kCtxIdList, ctx_id_list);
      for (size_t i = 0; i < succ_list_list.size(); i++) {
        for (const auto &succ_id : succ_list_list[i]) {
          aic_aiv_task_builder_ptr_->UpdateSuccList(succ_id, ctx_id_list[i], ffts_plus_task_def);
        }
      }
    } else {
      UpdateAtomicSuccList(same_memset_nodes_map, sub_node, ffts_plus_task_def);
      GenSuccList(sub_node, ffts_plus_task_def);
    }
  }
  for (auto &sub_node : sub_graph_nodes) {
    if (GenerateDataTaskDef(sub_node, ffts_plus_task_def, mode_type_) != SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ManualTheadTaskBuilder::GenSuccList(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) {
  vector <uint32_t> succ_lists;
  uint32_t ctx_id = 0;
  auto op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::GetInt(op_desc, kContextId, ctx_id);
  (void)ge::AttrUtils::GetListInt(op_desc, kSuccList, succ_lists);
  FFTS_LOGD("GenContextDef nodetype: %s, name: %s, succ_lists: %s.", node->GetType().c_str(), node->GetName().c_str(),
            fe::StringUtils::IntegerVecToString(succ_lists).c_str());
  for (auto succ_id : succ_lists) {
    aic_aiv_task_builder_ptr_->UpdateSuccList(succ_id, ctx_id, ffts_plus_task_def);
  }
  return SUCCESS;
}

ge::NodePtr ManualTheadTaskBuilder::HasControlNode(ge::ComputeGraphPtr &cur_graph) const {
  FFTS_LOGD("enter HasControlNode graph with name = %s.", cur_graph->GetName().c_str());
  for (auto &node : cur_graph->GetDirectNode()) {
    if (CONTROL_OP_V2_TYPE.count(node->GetType()) != 0) {
      FFTS_LOGD("Enter HasControlNode with control node %s.", node->GetName().c_str());
      return node;
    }
  }
  return nullptr;
}

Status ManualTheadTaskBuilder::DoWithControlOpImpl(ge::ComputeGraphPtr &cur_graph,
                                                   ControlGraphMap &controlnode_graphmap,
                                                   std::stack<ge::NodePtr> &control_node_stacks) const {
  FFTS_LOGD("DoWithControlOpImpl control_graph_names = '%s'.", cur_graph->GetName().c_str());
  auto control_node = HasControlNode(cur_graph);
  if (control_node == nullptr) {
    return SUCCESS;
  }
  if (UnfoldPartionCallOnlyOneDepth(*(cur_graph.get()), PARTITIONEDCALL) != SUCCESS) {
    return FAILED;
  }
  control_node_stacks.push(control_node);
  auto node_type = control_node->GetType();
  if (node_type == "If") {
    PreInsertIfNode(control_node, controlnode_graphmap, control_node_stacks);
  } else if (node_type == "While") {
    PreInsertWhileNode(control_node, controlnode_graphmap, control_node_stacks);
  } else if (node_type == "Case") {
    PreInsertCaseNode(control_node, controlnode_graphmap, control_node_stacks);
  }
  return SUCCESS;
}

Status ManualTheadTaskBuilder::PreInsertIfNode(ge::NodePtr &control_node,
                                               ControlGraphMap &controlnode_graphmap,
                                               std::stack<ge::NodePtr> &control_node_stacks) const {
  auto if_desc = control_node->GetOpDesc();
  ge::ComputeGraphPtr then_sub_graph = ge::NodeUtils::GetSubgraph(*control_node, THENBRANCHINDEX);
  ge::ComputeGraphPtr else_sub_graph = ge::NodeUtils::GetSubgraph(*control_node, ELSEBRANCHINDEX);
  if (then_sub_graph == nullptr || else_sub_graph == nullptr) {
    FFTS_LOGD("control_op_subgraph = nullptr");
    return FAILED;
  }
  if (UnfoldPartionCallOnlyOneDepth(*(then_sub_graph.get()), PARTITIONEDCALL) != SUCCESS) {
    return FAILED;
  }
  if (UnfoldPartionCallOnlyOneDepth(*(else_sub_graph.get()), PARTITIONEDCALL) != SUCCESS) {
    return FAILED;
  }
  std::vector<ge::ComputeGraphPtr> subgraphs;
  subgraphs.emplace_back(else_sub_graph);
  subgraphs.emplace_back(then_sub_graph);
  bool insert_flag = true;
  for (auto subgraph : subgraphs) {
    if (HasControlNode(subgraph) != nullptr) {
      insert_flag = false;
      (void)DoWithControlOpImpl(subgraph, controlnode_graphmap, control_node_stacks);
    }
  }
  if (insert_flag) {
    controlnode_graphmap.emplace(std::make_pair(control_node, subgraphs));
  }
  return SUCCESS;
}

Status ManualTheadTaskBuilder::PreInsertWhileNode(ge::NodePtr &control_node,
                                                  ControlGraphMap &controlnode_graphmap,
                                                  std::stack<ge::NodePtr> &control_node_stacks) const {
  auto while_desc = control_node->GetOpDesc();
  ge::ComputeGraphPtr cond_sub_graph = ge::NodeUtils::GetSubgraph(*control_node, CONDEXBRANCHINDEX);
  ge::ComputeGraphPtr body_sub_graph = ge::NodeUtils::GetSubgraph(*control_node, BODYBRANCHINDEX);
  if (cond_sub_graph == nullptr || body_sub_graph == nullptr) {
    FFTS_LOGD("control_op_subgraph = nullptr");
    return FAILED;
  }
  if (UnfoldPartionCallOnlyOneDepth(*(cond_sub_graph.get()), PARTITIONEDCALL) != SUCCESS) {
    return FAILED;
  }
  if (UnfoldPartionCallOnlyOneDepth(*(body_sub_graph.get()), PARTITIONEDCALL) != SUCCESS) {
    return FAILED;
  }
  std::vector<ge::ComputeGraphPtr> subgraphs;
  subgraphs.emplace_back(cond_sub_graph);
  subgraphs.emplace_back(body_sub_graph);
  bool insert_flag = true;
  for (auto subgraph : subgraphs) {
    if (HasControlNode(subgraph) != nullptr) {
      insert_flag = false;
      (void)DoWithControlOpImpl(subgraph, controlnode_graphmap, control_node_stacks);
    }
  }
  if (insert_flag) {
    controlnode_graphmap.emplace(std::make_pair(control_node, subgraphs));
  }
  return SUCCESS;
}

Status ManualTheadTaskBuilder::PreInsertCaseNode(ge::NodePtr &control_node,
                                                 ControlGraphMap &controlnode_graphmap,
                                                 std::stack<ge::NodePtr> &control_node_stacks) const {
  auto case_desc = control_node->GetOpDesc();
  const auto graph_names = case_desc->GetSubgraphInstanceNames();
  const uint32_t graph_num = static_cast<uint32_t>(graph_names.size());
  std::vector<ge::ComputeGraphPtr> subgraphs;
  bool insert_flag = true;
  for (uint32_t index = 0; index < graph_num; ++index) {
    ge::ComputeGraphPtr subgraph = ge::NodeUtils::GetSubgraph(*control_node, index);
    if (subgraph == nullptr) {
      return FAILED;
    }
    if (UnfoldPartionCallOnlyOneDepth(*(subgraph.get()), PARTITIONEDCALL) != SUCCESS) {
      return FAILED;
    }
    if (HasControlNode(subgraph) != nullptr) {
      insert_flag = false;
      (void)DoWithControlOpImpl(subgraph, controlnode_graphmap, control_node_stacks);
    }
    subgraphs.emplace_back(subgraph);
  }
  if (insert_flag) {
    controlnode_graphmap.emplace(std::make_pair(control_node, subgraphs));
  }
  return SUCCESS;
}

Status ManualTheadTaskBuilder::DoWithFftsPlusCtxIdForControlNode(ControlGraphMap controlnode_graphmap,
                                                                 const RunContextPtr &contextptr,
                                                                 std::stack<ge::NodePtr> &control_node_stacks) const {
  Status status = SUCCESS;
  while (control_node_stacks.size() != 0) {
    auto control_node = control_node_stacks.top();
    FFTS_CHECK_NOTNULL(control_node);
    PrintNode(control_node);
    auto iter = controlnode_graphmap.find(control_node);
    if (iter != controlnode_graphmap.end()) {
      auto node_type = control_node->GetType();
      if (node_type == "If") {
        status = DoWithFftsPlusContextIdForIfNode(control_node, iter->second, contextptr);
      } else if (node_type == "While") {
        status = DoWithFftsPlusContextIdForWhileNode(control_node, iter->second, contextptr);
      } else if (node_type == "Case") {
        status = DoWithFftsPlusContextIdForCaseNode(control_node, iter->second, contextptr);
      }
    }
    control_node_stacks.pop();
  }
  return status;
}

void ManualTheadTaskBuilder::DoWithIfNodeStreamActiveNetOutPut(std::vector<ge::ComputeGraphPtr> &sub_graphs) const {
  for (auto cur_graph : sub_graphs) {
    FFTS_CHECK(cur_graph == nullptr, FFTS_LOGW("cur_graph is a nullptr."), return);
    for (auto node : cur_graph->GetDirectNode()) {
      if (node == nullptr) {
        continue;
      }
      if (node->GetType() == "StreamActive") {
        (void)DoWithIfSwitchActive(node);
      }
      if (node->GetType() == "NetOutput") {
        (void)DoWithIfNetOutput(node);
      }
   }
  }
}

Status ManualTheadTaskBuilder::DoWithIfNodeOhterNode(ge::NodePtr &control_node,
                                                     std::vector<ge::ComputeGraphPtr> &sub_graphs) const {
  Status status = SUCCESS;
  for (auto cur_graph : sub_graphs) {
    for (auto node : cur_graph->GetDirectNode()) {
      if (IsOrdinaryNode(node)) {
        continue;
      }
      status = DoWithIfOther(control_node, node);
      if (status != SUCCESS) {
        return status;
      }
    }
  }
  return SUCCESS;
}

Status ManualTheadTaskBuilder::DoWithFftsPlusContextIdForIfNode(ge::NodePtr &control_node,
                                                                std::vector<ge::ComputeGraphPtr> &sub_graphs,
                                                                const RunContextPtr &contextptr) const {
  if (sub_graphs.size() != CONTROLNODEBRANCHNUM || control_node == nullptr) {
    return FAILED;
  }
  DoWithIfNodeStreamActiveNetOutPut(sub_graphs);
  auto else_branch = sub_graphs.at(0);
  auto then_branch = sub_graphs.at(1);
  ge::NodePtr label_node = nullptr;
  label_node = FindNodeByType(else_branch, "labelsetenter");
  FFTS_CHECK_NOTNULL(label_node);
  if (SUCCESS != DoWithIfElseLabeSetEnter(control_node, label_node)) {
    return FAILED;
  }
  label_node = FindNodeByType(else_branch, "labelsetleave");
  FFTS_CHECK_NOTNULL(label_node);

  if (SUCCESS != DoWithIfElseLabeSetLeave(control_node, label_node)) {
    return FAILED;
  }
  label_node = FindNodeByType(then_branch, "labelswitch");
  FFTS_CHECK_NOTNULL(label_node);
  if (SUCCESS != SetLabelSwitchByIndexAddr(label_node, contextptr)) {
    return FAILED;
  }
  if (SUCCESS != DoWithIfThenLabeSwitch(control_node, label_node)) {
    return FAILED;
  }
  label_node = FindNodeByType(then_branch, "labelsetenter");
  FFTS_CHECK_NOTNULL(label_node);
  if (SUCCESS != DoWithIfThenLabelSet(control_node, label_node)) {
    return FAILED;
  }
  FFTS_LOGD("enter DoWithFftsPlusContextIdForIfNode labelgoto.");
  label_node = FindNodeByType(then_branch, "labelgoto");
  FFTS_CHECK_NOTNULL(label_node);
  if (SUCCESS != DoWithIfThenLabelGoto(control_node, label_node, then_branch, sub_graphs)) {
    return FAILED;
  }
  if (SUCCESS != DoWithIfNodeOhterNode(control_node, sub_graphs)) {
    return FAILED;
  }
  return SUCCESS;
}

void ManualTheadTaskBuilder::DoWithWhileNodeStreamActiveNetOutPut(ge::NodePtr &control_node,
                                                                  std::vector<ge::ComputeGraphPtr> &sub_graphs) const {
  for (auto cur_graph : sub_graphs) {
    FFTS_CHECK(cur_graph == nullptr, FFTS_LOGW("cur_graph is a nullptr."), return);
    for (auto node : cur_graph->GetDirectNode()) {
      if (node == nullptr) {
        continue;
      }
      if (node->GetType() == "StreamActive") {
        DoWithWhileSwitchActive(control_node, node, cur_graph, sub_graphs);
      }
      if (node->GetType() == "NetOutput") {
        DoWithWhileNetOutput(node);
      }
   }
  }
}

Status ManualTheadTaskBuilder::DoWithWhileNodeOhterNode(ge::NodePtr &control_node,
                                                        std::vector<ge::ComputeGraphPtr> &sub_graphs) const {
  Status status = SUCCESS;
  for (auto cur_graph : sub_graphs) {
    FFTS_CHECK_NOTNULL(cur_graph);
    for (auto node : cur_graph->GetDirectNode()) {
      if (IsOrdinaryNode(node)) {
        continue;
      }
      status = DoWithWhileOther(control_node, node);
      if (status != SUCCESS) {
        return status;
      }
    }
  }
  return SUCCESS;
}

Status ManualTheadTaskBuilder::DoWithFftsPlusContextIdForWhileNode(ge::NodePtr &control_node,
                                                                   std::vector<ge::ComputeGraphPtr> &sub_graphs,
                                                                   const RunContextPtr &contextptr) const {
  if (sub_graphs.size() != CONTROLNODEBRANCHNUM || control_node == nullptr) {
    return FAILED;
  }
  DoWithWhileNodeStreamActiveNetOutPut(control_node, sub_graphs);
  auto cond_sub_graph = sub_graphs.at(0);
  auto body_sub_graph = sub_graphs.at(1);
  ge::NodePtr label_node = nullptr;
  label_node = FindNodeByType(body_sub_graph, "labelsetenter");
  if (label_node == nullptr) {
    return FAILED;
  }
  if (SUCCESS != DoWithWhileBodyLabeSetEnter(control_node, label_node)) {
    return FAILED;
  }
  label_node = FindNodeByType(body_sub_graph, "labelgoto");
  if (label_node == nullptr) {
    return FAILED;
  }
  if (SUCCESS != DoWithWhileBodyLabelGoto(control_node, label_node, body_sub_graph, sub_graphs)) {
    return FAILED;
  }
  label_node = FindNodeByType(body_sub_graph, "labelsetleave");
  if (label_node == nullptr) {
    return FAILED;
  }
  if (SUCCESS != DoWithWhileBodyLabeSetLeave(control_node, label_node)) {
    return FAILED;
  }
  label_node = FindNodeByType(cond_sub_graph, "labelsetenter");
  if (label_node == nullptr) {
    return FAILED;
  }
  if (SUCCESS != DoWithWhileCondLabelSet(control_node, label_node)) {
    return FAILED;
  }
  label_node = FindNodeByType(cond_sub_graph, "labelswitch");
  if (label_node == nullptr) {
    return FAILED;
  }
  if (SUCCESS != SetLabelSwitchByIndexAddr(label_node, contextptr)) {
    return FAILED;
  }
  if (SUCCESS != DoWithWhileCondLabeSwitch(control_node, label_node, cond_sub_graph, sub_graphs)) {
    return FAILED;
  }
  if (SUCCESS != DoWithWhileNodeOhterNode(control_node, sub_graphs)) {
    return FAILED;
  }
  return SUCCESS;
}

void ManualTheadTaskBuilder::DoWithCaseNodeStreamActiveNetOutPut(std::vector<ge::ComputeGraphPtr> &sub_graphs) const {
  for (auto cur_graph : sub_graphs) {
    FFTS_CHECK(cur_graph == nullptr, FFTS_LOGW("cur_graph is a nullptr."), return);
    for (auto node : cur_graph->GetDirectNode()) {
      if (!node) {
        continue;
      }
      if (node->GetType() == "StreamActive") {
        DoWithCaseSwitchActive(node);
      }
      if (node->GetType() == "NetOutput") {
        DoWithCaseNetOutput(node);
      }
    }
  }
}

Status ManualTheadTaskBuilder::DoWithCaseNodeOhterNode(ge::NodePtr &control_node,
                                                       std::vector<ge::ComputeGraphPtr> &sub_graphs) const {
  Status status = SUCCESS;
  for (auto cur_graph : sub_graphs) {
    for (auto node : cur_graph->GetDirectNode()) {
      if (IsOrdinaryNode(node)) {
        continue;
      }
      status = DoWithCaseOther(control_node, node);
      if (status != SUCCESS) {
        return status;
      }
    }
  }
  return SUCCESS;
}

Status ManualTheadTaskBuilder::DoWithFftsPlusContextIdForCaseNode(ge::NodePtr &control_node,
                                                                  std::vector<ge::ComputeGraphPtr> &sub_graphs,
                                                                  const RunContextPtr &contextptr) const {
  if (control_node == nullptr) {
    return FAILED;
  }
  DoWithCaseNodeStreamActiveNetOutPut(sub_graphs);
  ge::NodePtr label_node = nullptr;
  size_t index = 0;
  for (auto cur_graph : sub_graphs) {
    label_node = FindNodeByType(cur_graph, "labelsetenter");
    if (label_node == nullptr) {
      return FAILED;
    }
    if (SUCCESS != DoWithCaseLabeSetEnter(control_node, label_node)) {
      return FAILED;
    }
    if (index != sub_graphs.size() -1) {
      label_node = FindNodeByType(cur_graph, "labelgoto");
      if (label_node == nullptr) {
        return FAILED;
      }
      if (SUCCESS != DoWithCaseLabelGoto(control_node, label_node, cur_graph, sub_graphs)) {
        return FAILED;
      }
    }
    index++;
  }
  auto last_graph = sub_graphs.at(sub_graphs.size() -1);
  label_node = FindNodeByType(last_graph, "labelsetleave");
  if (label_node == nullptr) {
    return FAILED;
  }
  if (SUCCESS != DoWithCaseLabeSetLeave(control_node, label_node)) {
    return FAILED;
  }
  auto first_graph = sub_graphs.at(0);
  label_node = FindNodeByType(first_graph, "labelswitch");
  if (label_node == nullptr) {
    return FAILED;
  }
  if (SUCCESS != SetLabelSwitchByIndexAddr(label_node, contextptr)) {
    return FAILED;
  }
  if (SUCCESS != DoWithCaseLabeSwitch(control_node, label_node, first_graph, sub_graphs)) {
    return FAILED;
  }
  if (SUCCESS != DoWithCaseNodeOhterNode(control_node, sub_graphs)) {
    return FAILED;
  }
  return SUCCESS;
}

Status ManualTheadTaskBuilder::DoWithIfElseLabeSetEnter(ge::NodePtr &control_node,
                                                        ge::NodePtr &node) const {
  FFTS_LOGD("enter DoWithIfElseLabeSetEnter node: %s.", node->GetName().c_str());
  if (SUCCESS != RepalceDataWithReal(control_node, node)) {
    FFTS_LOGD("ReplaceDataWithReal unsuccessful");
    return FAILED;
  }
  return SUCCESS;
}

Status ManualTheadTaskBuilder::DoWithIfElseLabeSetLeave(ge::NodePtr &control_node, ge::NodePtr &node) const {
  FFTS_LOGD("Entering DoWithIfElseLabeSetLeave node: %s.", node->GetName().c_str());
  if (SUCCESS != RepalceDataWithReal(control_node, node)) {
    FFTS_LOGD("ReplaceDataWithReal unsuccessful");
    return FAILED;
  }
  return DoWithLabeSetLeave(control_node, node);
}

Status ManualTheadTaskBuilder::DoWithIfThenLabeSwitch(ge::NodePtr &control_node, ge::NodePtr &node) const {
  FFTS_LOGD("Entering DoWithIfThenLabelSwitch node: %s.", node->GetName().c_str());
  if (SUCCESS != RepalceDataWithReal(control_node, node)) {
    FFTS_LOGD("ReplaceDataWithReal unsuccessful");
    return FAILED;
  }
  return SUCCESS;
}

Status ManualTheadTaskBuilder::DoWithIfThenLabelSet(ge::NodePtr &control_node, ge::NodePtr &node) const {
  FFTS_LOGD("Enter DoWithIfThenLabelSet node: %s.", node->GetName().c_str());
  if (SUCCESS != RepalceDataWithReal(control_node, node)) {
    FFTS_LOGD("ReplaceDataWithReal unsuccessful");
    return FAILED;
  }
  return SUCCESS;
}


Status ManualTheadTaskBuilder::DoWithIfThenLabelGoto(ge::NodePtr &control_node,
                                                     ge::NodePtr &node,
                                                     ge::ComputeGraphPtr cur_graph,
                                                     std::vector<ge::ComputeGraphPtr> &sub_graphs) const {
  FFTS_LOGD("Entering DoWithIfThenLabelGoto node: %s.", node->GetName().c_str());
  if (SUCCESS != RepalceDataWithReal(control_node, node)) {
    FFTS_LOGD("ReplaceDataWithReal unsuccessful");
    return FAILED;
  }
  return SetLabelGoto(cur_graph, node, sub_graphs);
}

Status ManualTheadTaskBuilder::DoWithIfNetOutput(ge::NodePtr &node) const {
  return DoWithNetOutPut(node);
}

Status ManualTheadTaskBuilder::DoWithIfSwitchActive(ge::NodePtr &node) const {
  FFTS_LOGD("DoWithIfSwitchActive node: %s.", node->GetName().c_str());
  DeleteNode(node);
  return SUCCESS;
}

Status ManualTheadTaskBuilder::DoWithIfOther(ge::NodePtr &control_node, ge::NodePtr &node) const {
  FFTS_LOGD("Entered DoWithIfOther node: %s.", node->GetName().c_str());
  if (SUCCESS != RepalceDataWithReal(control_node, node)) {
    FFTS_LOGD("ReplaceDataWithReal unsuccessful");
    return FAILED;
  }
  return SUCCESS;
}


Status ManualTheadTaskBuilder::DoWithWhileBodyLabeSetEnter(ge::NodePtr &control_node, ge::NodePtr &node) const {
  FFTS_LOGD("enter DoWithWhileBodyLabeSetEnter node:%s.", node->GetName().c_str());
  if (SUCCESS != RepalceDataWithReal(control_node, node)) {
    FFTS_LOGD("ReplaceDataWithReal unsuccessful");
    return FAILED;
  }
  return SUCCESS;
}

Status ManualTheadTaskBuilder::DoWithWhileBodyLabelGoto(ge::NodePtr &control_node,
                                                        ge::NodePtr &node,
                                                        ge::ComputeGraphPtr cur_graph,
                                                        std::vector<ge::ComputeGraphPtr> &sub_graphs) const {
  FFTS_LOGD("enter DoWithWhileBodyLabelGoto node:%s.", node->GetName().c_str());
  if (SUCCESS != RepalceDataWithReal(control_node, node)) {
    FFTS_LOGD("ReplaceDataWithReal unsuccessful");
    return FAILED;
  }
  return SetLabelGoto(cur_graph, node, sub_graphs);
}

Status ManualTheadTaskBuilder::DoWithWhileBodyLabeSetLeave(ge::NodePtr &control_node, ge::NodePtr &node) const {
  FFTS_LOGD("enter DoWithWhileBodyLabeSetLeave node: %s.", node->GetName().c_str());
  if (SUCCESS != RepalceDataWithReal(control_node, node)) {
    FFTS_LOGD("ReplaceDataWithReal unsuccessful");
    return FAILED;
  }
  return DoWithLabeSetLeave(control_node, node);
}

Status ManualTheadTaskBuilder::DoWithWhileCondLabelSet(ge::NodePtr &control_node, ge::NodePtr &node) const {
  FFTS_LOGD("Enter DoWithWhileCondLabelSet node: %s.", node->GetName().c_str());
  if (SUCCESS != RepalceDataWithReal(control_node, node)) {
    FFTS_LOGD("ReplaceDataWithReal unsuccessful");
    return FAILED;
  }
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), "_label_while_cond_labelsetenter", true);
  return SUCCESS;
}

Status ManualTheadTaskBuilder::DoWithWhileCondLabeSwitch(ge::NodePtr &control_node,
                                                         ge::NodePtr &node,
                                                         ge::ComputeGraphPtr cur_graph,
                                                         std::vector<ge::ComputeGraphPtr> &sub_graphs) const {
  FFTS_LOGD("enter DoWithWhileCondLabeSwitch node:%s.", node->GetName().c_str());
  if (SUCCESS != RepalceDataWithReal(control_node, node)) {
    FFTS_LOGD("ReplaceDataWithReal unsuccessful");
    return FAILED;
  }
  Status status = SUCCESS;
  status = SetLabelSwitchByIndexJumpNode(control_node, cur_graph, node, sub_graphs);
  if (status != SUCCESS) {
    return status;
  }
  return SUCCESS;
}

Status ManualTheadTaskBuilder::DoWithWhileNetOutput(ge::NodePtr &node) const {
  return DoWithNetOutPut(node);
}

Status ManualTheadTaskBuilder::DoWithWhileSwitchActive(ge::NodePtr &control_node,
                                                       ge::NodePtr &node,
                                                       ge::ComputeGraphPtr cur_graph,
                                                       std::vector<ge::ComputeGraphPtr> &sub_graphs) const {
  (void)control_node;
  (void)cur_graph;
  (void)sub_graphs;
  FFTS_LOGD("enter DoWithWhileSwitchActive node:%s.", node->GetName().c_str());
  DeleteNode(node);
  return SUCCESS;
}

Status ManualTheadTaskBuilder::DoWithWhileOther(ge::NodePtr &control_node, ge::NodePtr &node) const {
  FFTS_LOGD("enter DoWithWhileOther node:%s.", node->GetName().c_str());
  if (SUCCESS != RepalceDataWithReal(control_node, node)) {
    FFTS_LOGD("ReplaceDataWithReal unsuccessful");
    return FAILED;
  }
  return SUCCESS;
}

Status ManualTheadTaskBuilder::DoWithCaseLabeSetEnter(ge::NodePtr &control_node, ge::NodePtr &node) const {
  FFTS_LOGD("Enter DoWithCaseLabelSetEnter node: %s.", node->GetName().c_str());
  if (SUCCESS != RepalceDataWithReal(control_node, node)) {
    FFTS_LOGD("ReplaceDataWithReal unsuccessful");
    return FAILED;
  }
  return SUCCESS;
}

Status ManualTheadTaskBuilder::DoWithCaseLabelGoto(ge::NodePtr &control_node,
                                                   ge::NodePtr &node,
                                                   ge::ComputeGraphPtr cur_graph,
                                                   std::vector<ge::ComputeGraphPtr> &sub_graphs) const {
  FFTS_LOGD("enter DoWithCaseLabelGoto node:%s.", node->GetName().c_str());
  if (SUCCESS != RepalceDataWithReal(control_node, node)) {
    FFTS_LOGD("ReplaceDataWithReal unsuccessful");
    return FAILED;
  }
  Status status = SUCCESS;
  status = SetLabelGoto(cur_graph, node, sub_graphs);
  return status;
}

Status ManualTheadTaskBuilder::DoWithCaseLabeSetLeave(ge::NodePtr &control_node, ge::NodePtr &node) const {
  FFTS_LOGD("enter DoWithCaseLabeSetLeave node:%s.", node->GetName().c_str());
  if (SUCCESS != RepalceDataWithReal(control_node, node)) {
    FFTS_LOGD("ReplaceDataWithReal unsuccessful");
    return FAILED;
  }
  return DoWithLabeSetLeave(control_node, node);
}

Status ManualTheadTaskBuilder::DoWithCaseLabeSwitch(ge::NodePtr &control_node,
                                                    ge::NodePtr &node,
                                                    ge::ComputeGraphPtr cur_graph,
                                                    std::vector<ge::ComputeGraphPtr> &sub_graphs) const {
  FFTS_LOGD("enter DoWithCaseLabeSwitch node:%s.", node->GetName().c_str());
  if (SUCCESS != RepalceDataWithReal(control_node, node)) {
    FFTS_LOGD("ReplaceDataWithReal unsuccessful");
    return FAILED;
  }
  Status status = SUCCESS;
  status = SetLabelSwitchByIndexJumpNode(control_node, cur_graph, node, sub_graphs);
  if (status != SUCCESS) {
    return status;
  }
  return SUCCESS;
}

Status ManualTheadTaskBuilder::DoWithCaseNetOutput(ge::NodePtr &node) const {
  return DoWithNetOutPut(node);
}

Status ManualTheadTaskBuilder::DoWithCaseSwitchActive(ge::NodePtr &node) const {
  FFTS_LOGD("Entering DoWithCaseSwitchActive node: %s.", node->GetName().c_str());
  DeleteNode(node);
  return SUCCESS;
}

Status ManualTheadTaskBuilder::DoWithCaseOther(ge::NodePtr &control_node, ge::NodePtr &node) const {
  FFTS_LOGD("enter DoWithCaseOther node:%s.", node->GetName().c_str());
  if (SUCCESS != RepalceDataWithReal(control_node, node)) {
    FFTS_LOGD("ReplaceDataWithReal unsuccessful");
    return FAILED;
  }
  return SUCCESS;
}

ge::NodePtr ManualTheadTaskBuilder::FindNodeByType(ge::ComputeGraphPtr cur_graph, std::string nodetypename) const {
  if (nodetypename == "labelsetenter") {
    return FindLabelSetEenter(cur_graph);
  } else if (nodetypename == "labelsetleave") {
    return FindLabelSetLeave(cur_graph);
  } else if (nodetypename == "labelswitch") {
    return FindLabelSwitch(cur_graph);
  } else if (nodetypename == "labelgoto") {
    return FindLabelGoto(cur_graph);
  }
  return nullptr;
}

ge::NodePtr ManualTheadTaskBuilder::FindLabelSetEenter(ge::ComputeGraphPtr cur_graph) const {
  FFTS_LOGD("FindLabelSetEenter graph: %s.", cur_graph->GetName().c_str());
  for (auto node : cur_graph->GetDirectNode()) {
    if (!node) {
      continue;
    }
    if (node->GetType() == "LabelSet") {
      if (node->GetInAllNodes().size() == 0) {
        return node;
      } else if (node->GetInControlNodes().size() == 1) {
        auto inputnode = node->GetInControlNodes().at(0);
        if (inputnode != nullptr && inputnode->GetType() == "LabelSwitchByIndex") {
          return node;
        }
      }
    }
  }
  return nullptr;
}

ge::NodePtr ManualTheadTaskBuilder::FindLabelSetLeave(ge::ComputeGraphPtr cur_graph) const {
  FFTS_LOGD("FindLabelSetLeave graph: %s.", cur_graph->GetName().c_str());
  for (auto node : cur_graph->GetDirectNode()) {
    if (!node) {
      continue;
    }
    if (node->GetType() == "LabelSet") {
      PrintNode(node);
      if (node->GetOutAllNodes().size() == 0) {
        return node;
      }
    }
  }
  return nullptr;
}

ge::NodePtr ManualTheadTaskBuilder::FindLabelSwitch(ge::ComputeGraphPtr cur_graph) const {
  FFTS_LOGD("FindLabelSwitch graph: %s.", cur_graph->GetName().c_str());
  for (auto node : cur_graph->GetDirectNode()) {
    if (!node) {
      continue;
    }
    if (node->GetType() == "LabelSwitchByIndex") {
      return node;
    }
  }
  return nullptr;
}

ge::NodePtr ManualTheadTaskBuilder::FindLabelGoto(ge::ComputeGraphPtr cur_graph) const {
  FFTS_LOGD("FindLabelGoto graphname:%s.", cur_graph->GetName().c_str());
  for (auto node : cur_graph->GetDirectNode()) {
    if (!node) {
      continue;
    }
    PrintNode(node);
    if (node->GetType() == "LabelGoto" || node->GetType() == "LabelGotoEx") {
       FFTS_LOGD("Entering FindLabelGoto find.");
      PrintNode(node);
      return node;
    }
  }
  return nullptr;
}

Status ManualTheadTaskBuilder::SetIfWhileLastLabelNext(ge::NodePtr &control_node, ge::NodePtr &node) const {
  FFTS_CHECK_NOTNULL(node);
  FFTS_LOGD("Enter SetIfWhileLastLabelNext node: %s.", node->GetName().c_str());
  auto node_type = node->GetType();
  std::vector<ge::NodePtr> parent_outputs_nodes;
  auto outputs_node = control_node->GetOutNodes();
  for (auto parent_outnode : outputs_node) {
    FFTS_CHECK_NOTNULL(parent_outnode);
    if (!parent_outnode->GetOpDesc()->SetExtAttr(ATTR_NAME_PARENT_OUTPUTS_INPUT_NODES, node)) {
      FFTS_LOGD("Setting label x control-node parent graph outgraphout nodes to prenode was unsuccessful.");
      PrintNode(parent_outnode);
      return FAILED;
    }
    PrintNodeAttrExtNode(parent_outnode, ATTR_NAME_PARENT_OUTPUTS_INPUT_NODES);
    parent_outputs_nodes.push_back(parent_outnode);
  }
  std::shared_ptr<std::vector<ge::NodePtr>> shared_idx_nodes_ptr = nullptr;
  FFTS_MAKE_SHARED(shared_idx_nodes_ptr = std::make_shared<std::vector<ge::NodePtr>>(parent_outputs_nodes),
                   return FAILED);
  if (shared_idx_nodes_ptr == nullptr) {
    return FAILED;
  }
  if (!node->GetOpDesc()->SetExtAttr(ATTR_NAME_LASTLABELSET_OUT_NODES, shared_idx_nodes_ptr)) {
    FFTS_LOGD("Set control-node parent graph outgraphout nodes to labelx: successlist, unsuccessful.");
    PrintNode(node);
    return FAILED;
  }
  PrintNodeAttrExtNodes(node, ATTR_NAME_LASTLABELSET_OUT_NODES);
  return SUCCESS;
}

Status ManualTheadTaskBuilder::SetLabelGoto(ge::ComputeGraphPtr &cur_graph,
                                            ge::NodePtr &node,
                                            std::vector<ge::ComputeGraphPtr> &sub_graphs) const {
  FFTS_LOGD("enter SetLabelGoto node:%s.", node->GetName().c_str());
  PrintNode(node);
  ge::NodePtr dst_label_node = nullptr;
  uint32_t jump_label_idx = 0;
  ge::OpDescPtr op_desc = node->GetOpDesc();
  if (!ge::AttrUtils::GetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, jump_label_idx)) {
    FFTS_LOGD("Get labelgoto attr _label_switch_index unsuccessful.");
    return FAILED;
  }
  FFTS_LOGD("labelgoto has attribute _label_switch_index with value = %u", jump_label_idx);
  dst_label_node = FindJumpIfWhileLableX(jump_label_idx, node, cur_graph, sub_graphs);
  if (dst_label_node == nullptr) {
    FFTS_LOGD("Get label and goto dst label, set node to unsuccess.");
    return FAILED;
  }
  PrintNode(dst_label_node);
  if (!op_desc->SetExtAttr(ATTR_NAME_LABEL_JUMP_NODE, dst_label_node)) {
    FFTS_LOGD("Can't set attr _label_start_node to dst_label_node.");
    return FAILED;
  }
  PrintNodeAttrExtNode(node, ATTR_NAME_LABEL_JUMP_NODE);
  return SUCCESS;
}


ge::NodePtr ManualTheadTaskBuilder::FindJumpIfWhileLableX(uint32_t jump_id,
                                                          ge::NodePtr &node,
                                                          ge::ComputeGraphPtr &cur_graph,
                                                          std::vector<ge::ComputeGraphPtr> &sub_graphs) const {
  FFTS_LOGD("enter FindJumpIfWhileLableX node:%s.", node->GetName().c_str());
  auto cur_graph_name = cur_graph->GetName();
  for (auto cur_graph_iter : sub_graphs) {
    for (auto labelsetnode : cur_graph_iter->GetDirectNode()) {
      if (!labelsetnode) {
        continue;
      }
      if (labelsetnode->GetType() != "LabelSet") {
        continue;
      }
      uint32_t success_idx_false = 0;
      if (!ge::AttrUtils::GetInt(labelsetnode->GetOpDesc(), ATTR_NAME_LABEL_SWITCH_INDEX, success_idx_false)) {
        FFTS_LOGD("Cannot get labelset _label_switch_index attr");
        continue;
      }
      if (jump_id != success_idx_false) {
        continue;
      }
      PrintNode(labelsetnode);
      bool control_node_flag = false;
      if (labelsetnode->GetInControlNodes().size() != 0 &&
          labelsetnode->GetInControlNodes().at(0) != nullptr &&
          node->GetName() == labelsetnode->GetInControlNodes().at(0)->GetName()) {
        control_node_flag = true;
        FFTS_LOGD("labelset has inputnode = %s; since inputnode == control_node, skip setting ATTR_NAME_LABELSET_PRE_LABEL.",
                  labelsetnode->GetInControlNodes().at(0)->GetName().c_str());
      }
      if (!control_node_flag) {
        (void)labelsetnode->GetOpDesc()->SetExtAttr(ATTR_NAME_LABELSET_PRE_LABEL, node);
        PrintNodeAttrExtNode(labelsetnode, ATTR_NAME_LABELSET_PRE_LABEL);
      }
      return labelsetnode;
    }
  }
  FFTS_LOGD("Cannot find destination labelx.");
  return nullptr;
}

ge::NodePtr ManualTheadTaskBuilder::FindFinalParentNode(ge::NodePtr &node, uint32_t &max_depth) const {
  if (max_depth >= MAX_DEPTH) {
    return nullptr;
  }
  if (node->GetType() != "Data") {
    return node;
  }
  FFTS_LOGD("Enter find data: %s in final parent node.", node->GetName().c_str());
  PrintNode(node);
  uint32_t parent_index = 0;
  if (!ge::AttrUtils::GetInt(node->GetOpDesc(), "_parent_node_index", parent_index)) {
    return node;
  }
  if (node->GetOwnerComputeGraph() == nullptr || node->GetOwnerComputeGraph()->GetParentNode() == nullptr) {
    return nullptr;
  }
  auto parent_node = node->GetOwnerComputeGraph()->GetParentNode();
  PrintNode(parent_node);
  auto parent_node_in_anchor = parent_node->GetInDataAnchor(parent_index);
  if (parent_node_in_anchor == nullptr) {
    FFTS_LOGD("FindFinalParentNode: parent_node_in_anchor is null");
    return nullptr;
  }
  auto src_out_anchor = parent_node_in_anchor->GetPeerOutAnchor();
  if (src_out_anchor == nullptr || src_out_anchor->GetOwnerNode() == nullptr) {
    FFTS_LOGD("FindFinalParentNode: src_out_anchor is null; GetOwnerNode returns null");
    return nullptr;
  } else {
    auto src_node = src_out_anchor->GetOwnerNode();
    max_depth++;
    return FindFinalParentNode(src_node, max_depth);
  }
}

Status ManualTheadTaskBuilder::RepalceDataWithReal(ge::NodePtr &control_node, ge::NodePtr &node) const {
  FFTS_LOGD("Entering RepalceDataWithReal node: %s.", node->GetName().c_str());
  std::vector<ge::NodePtr> parent_nodes;
  for (auto &data : node->GetInNodes()) {
    if (data == nullptr || data->GetType() != "Data") {
      continue;
    }
    FFTS_LOGD("ReplaceDataWithReal: inputdata[%s], datasize = %zu.", data->GetNamePtr(), data->GetInNodes().size());
    uint32_t parent_index = 0;
    if (!ge::AttrUtils::GetInt(data->GetOpDesc(), "_parent_node_index", parent_index)) {
      continue;
    }
    FFTS_LOGD("ReplaceDataWithReal parent controlnode[%s] with parent_index = %u.", control_node->GetNamePtr(), parent_index);
    auto parent_node_in_anchor = control_node->GetInDataAnchor(parent_index);
    if (parent_node_in_anchor == nullptr) {
      FFTS_LOGD("ReplaceDataWithReal parent_node_in_anchor == null");
      continue;
    }
    auto src_out_anchor = parent_node_in_anchor->GetPeerOutAnchor();
    if (src_out_anchor == nullptr || src_out_anchor->GetOwnerNode() == nullptr) {
      FFTS_LOGD("RepalceDataWithReal src_out_anchor == null GetOwnerNode = null");
      continue;
    }
    FFTS_LOGD("ReplaceDataWithReal parent controlnode inputnode.");
    auto src_node = src_out_anchor->GetOwnerNode();
    uint32_t max_depth = 0;
    auto parent_node = FindFinalParentNode(src_node, max_depth);
    if (parent_node != nullptr) {
      parent_nodes.push_back(src_out_anchor->GetOwnerNode());
    } else {
      FFTS_LOGD("ReplaceDataWithReal parent_node = nullptr srcnode = %s", src_node->GetName().c_str());
    }
  }
  for (auto parent_node : parent_nodes) {
    if (!parent_node->GetOpDesc()->SetExtAttr(ATTR_NAME_PARENT_OUTPUTS_OUTPUT_NODE, node)) {
      FFTS_LOGD("Setting preout parent attribute unsuccessful.");
      return FAILED;
    }
    PrintNodeAttrExtNode(parent_node, ATTR_NAME_PARENT_OUTPUTS_OUTPUT_NODE);
  }
  std::shared_ptr<std::vector<ge::NodePtr>> shared_idx_nodes = nullptr;
  FFTS_MAKE_SHARED(shared_idx_nodes = std::make_shared<std::vector<ge::NodePtr>>(parent_nodes), return FAILED);
  FFTS_CHECK_NOTNULL(shared_idx_nodes);
  if (!node->GetOpDesc()->SetExtAttr(ATTR_NAME_PARENT_PRE_NODES, shared_idx_nodes)) {
    return FAILED;
  }
  PrintNodeAttrExtNode(node, ATTR_NAME_LABEL_JUMP_NODES);
  return SUCCESS;
}

Status ManualTheadTaskBuilder::SetLabelSwitchByIndexAddr(ge::NodePtr &node, const RunContextPtr &contextptr) const {
  if (contextptr == nullptr) {
    FFTS_LOGD("SetLabelSwitchByIndexAddr contextptr == null");
    return FAILED;
  }
  bool ret = node->GetOpDesc()->SetExtAttr(kRuntimeContentx, contextptr);
  if (!ret) {
    FFTS_LOGD("Setting _ffts_runtime_context attribute was unsuccessful.");
    return FAILED;
  }
  return SUCCESS;
}

Status ManualTheadTaskBuilder::SetLabelSwitchByIndexJumpNode(ge::NodePtr &control_node,
                                                             ge::ComputeGraphPtr &cur_graph,
                                                             ge::NodePtr &node,
                                                             std::vector<ge::ComputeGraphPtr> &sub_graphs) const {
  FFTS_LOGD("enter SetLabelSwitchByIndexJumpNode node: %s.", node->GetName().c_str());
  ge::OpDescPtr op_desc = node->GetOpDesc();
  std::vector<uint32_t> label_idx_list;
  std::vector<ge::NodePtr> v_jump_nodes;
  if (!ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_LABEL_SWITCH_LIST, label_idx_list)) {
    FFTS_LOGD("LabelSwitchByIndex does not have _label_switch_list.");
    return FAILED;
  }
  if (control_node->GetType() == "If" || control_node->GetType() == "While") { // if while false 0 true 1 need swap
    if (label_idx_list.size() != CONTROLNODEBRANCHNUM) {
      FFTS_LOGD("LabelSwitchByIndex: _label_switch_list.size != 2.");
      return FAILED;
    }
    auto temp = label_idx_list.at(0);
    label_idx_list.at(0) = label_idx_list.at(1);
    label_idx_list.at(1) = temp;
  }
  for (auto label_idx : label_idx_list) {
    FFTS_LOGD("LabelSwitchByIndex list index = %u.", label_idx);
    auto to_add_node = FindJumpIfWhileLableX(label_idx, node, cur_graph, sub_graphs);
    FFTS_LOGD("Got LabelSwitchByIndex destination label set node.");
    if (to_add_node != nullptr) {
      FFTS_LOGD("get to_add_node");
      PrintNode(to_add_node);
      v_jump_nodes.push_back(to_add_node);
    }
  }
  if (v_jump_nodes.empty()) {
    FFTS_LOGD("LabelSwitchByIndexv_jump_nodes is empty");
    return FAILED;
  }
  std::shared_ptr<std::vector<ge::NodePtr>> shared_idx_nodes_ptr_v_jump_nodes = nullptr;
  FFTS_MAKE_SHARED(shared_idx_nodes_ptr_v_jump_nodes = std::make_shared<std::vector<ge::NodePtr>>(v_jump_nodes),
                   return FAILED);
  if (shared_idx_nodes_ptr_v_jump_nodes == nullptr) {
    return FAILED;
  }
  if (!op_desc->SetExtAttr(ATTR_NAME_LABEL_JUMP_NODES, shared_idx_nodes_ptr_v_jump_nodes)) {
    FFTS_LOGD("can't set jumpnodes to labelswitch.");
    return FAILED;
  }
  FFTS_LOGD("get all label set jump nodes");
  PrintNodeAttrExtNode(node, ATTR_NAME_LABEL_JUMP_NODES);
  return SUCCESS;
}

Status ManualTheadTaskBuilder::DoWithLabeSetLeave(ge::NodePtr &control_node, ge::NodePtr &node) const {
  Status status;
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), "_labelsetleave", true);
  FFTS_LOGD("DoWithLabeSetLeavethis label x is last labelnext.");
  status = SetIfWhileLastLabelNext(control_node, node);
  return status;
}

Status ManualTheadTaskBuilder::DoWithNetOutPut(ge::NodePtr &node) const {
  FFTS_LOGD("DoWithNetOutPut node:%s.", node->GetName().c_str());
  PrintNodeAttrExtNode(node, ATTR_NAME_PARENT_OUTPUTS_INPUT_NODES);
  ge::NodePtr inner_graph_outputs_node = nullptr;
  inner_graph_outputs_node = node->GetOpDesc()->TryGetExtAttr(ATTR_NAME_PARENT_OUTPUTS_INPUT_NODES,
                                                              inner_graph_outputs_node);
  if (inner_graph_outputs_node != nullptr) {
    FFTS_LOGD("NetOutputNode has an internal label set from the previous node.");
    std::vector<ge::NodePtr> parent_outputs_nodes;
    for (const auto &out_label_node : node->GetOutControlNodes()) {
      FFTS_CHECK_NOTNULL(out_label_node);
      if (!out_label_node->GetOpDesc()->SetExtAttr(ATTR_NAME_PARENT_OUTPUTS_INPUT_NODES, out_label_node)) {
        FFTS_LOGD("Setting the out label of control-node parent graph outgraphout nodes to prenode was unsuccessful.");
        PrintNode(out_label_node);
        return FAILED;
      }
      parent_outputs_nodes.push_back(out_label_node);
    }
    std::shared_ptr<std::vector<ge::NodePtr>> shared_idx_nodes_ptr = nullptr;
    FFTS_MAKE_SHARED(shared_idx_nodes_ptr = std::make_shared<std::vector<ge::NodePtr>>(parent_outputs_nodes),
                     return FAILED);
    if (shared_idx_nodes_ptr == nullptr) {
      return FAILED;
    }
    (void)inner_graph_outputs_node->GetOpDesc()->DelAttr(ATTR_NAME_LASTLABELSET_OUT_NODES);
    if (!inner_graph_outputs_node->GetOpDesc()->SetExtAttr(ATTR_NAME_LASTLABELSET_OUT_NODES, shared_idx_nodes_ptr)) {
      FFTS_LOGD("Set control-node parent graph outgraphout nodes to labelx: successlist, unsuccessful.");
      PrintNode(node);
      return FAILED;
    }
    PrintNodeAttrExtNodes(inner_graph_outputs_node, ATTR_NAME_LASTLABELSET_OUT_NODES);
  }
  DeleteNode(node);
  return SUCCESS;
}

void ManualTheadTaskBuilder::GenContextIdWithLabelSet(ge::NodePtr &node,
                                                      std::vector<ge::NodePtr> &no_pre_sub_graph_nodes) const {
  FFTS_LOGD("enter GenContextIdWithLabelSet node:%s.", node->GetName().c_str());
  if (JudgeNodeToNeedDoWith(node)) {
    return;
  }
  PrintNode(node);
  bool pre_node_flag = true;
  ge::OpDescPtr op_desc = node->GetOpDesc();
  ge::NodePtr attrnode = nullptr;
  PrintNodeAttrExtNode(node, ATTR_NAME_LABELSET_PRE_LABEL);
  attrnode = op_desc->TryGetExtAttr(ATTR_NAME_LABELSET_PRE_LABEL, attrnode);
  if (attrnode == nullptr) {
    FFTS_LOGD("Label set does not have a prenode.");
    pre_node_flag = false;
  }
  if (pre_node_flag) {
    no_pre_sub_graph_nodes.push_back(node);
  }
}

void ManualTheadTaskBuilder::GenContextIdWithLabelSwitch(ge::NodePtr &node,
                                                         std::vector<ge::NodePtr> &pre_sub_graph_nodes,
                                                         std::vector<ge::NodePtr> &no_pre_sub_graph_nodes) const {
  FFTS_LOGD("enter GenContextIdWithLabelSwitch node:%s", node->GetName().c_str());
  if (JudgeNodeToNeedDoWith(node)) {
    return;
  }
  PrintNode(node);
  bool pre_node_flag = true;
  ge::OpDescPtr op_desc = node->GetOpDesc();
  ge::NodePtr attrnode = nullptr;
  attrnode = op_desc->TryGetExtAttr(ATTR_NAME_PARENT_PRE_NODES, attrnode);
  if (attrnode == nullptr) {
    FFTS_LOGD("Label set does not have a prenode.");
    pre_node_flag = false;
  }
  if (pre_node_flag) {
    no_pre_sub_graph_nodes.push_back(node);
  } else {
    pre_sub_graph_nodes.push_back(node);
  }
}


void ManualTheadTaskBuilder::GenContextIdWithLabelGoto(ge::NodePtr &node,
                                                       std::vector<ge::NodePtr> &no_pre_sub_graph_nodes) const {
  FFTS_LOGD("enter GenContextIdWithLabelGoto node:%s", node->GetName().c_str());
  no_pre_sub_graph_nodes.push_back(node);
}

void ManualTheadTaskBuilder::GenContextIdOhter(ge::NodePtr &node,
                                               std::vector<ge::NodePtr> &no_pre_sub_graph_nodes) const {
  if (node == nullptr) {
    return;
  }
  FFTS_LOGD("Enter GenContextIdOther node: %s", node->GetName().c_str());
  PrintNode(node);
  auto node_type = node->GetType();
  if (node_type == "Data" ||
      node_type == "NetOutput" ||
      node_type == "MemcpyAsync" ||
      node_type == "PartitionedCall" ||
      node_type == "StreamActive" ||
      NO_NEED_GEN_TASK_OP_TYPE.count(node_type) != 0 ||
      CONTROL_OP_V2_TYPE.count(node_type) != 0) {
    FFTS_LOGD("Node did not generate context ID.");
    return;
  }
  no_pre_sub_graph_nodes.push_back(node);
  return;
}

bool ManualTheadTaskBuilder::JudgeNodeToNeedDoWith(ge::NodePtr node) const {
  FFTS_LOGD("enter JudgeNodeToNeedDoWith node:%s", node->GetName().c_str());
  bool has_partition = false;
  if (!ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_HAS_PARTITION, has_partition)) {
    return false;
  }
  PrintNode(node);
  FFTS_LOGD("has_partition = %d", has_partition);
  return has_partition;
}

Status ManualTheadTaskBuilder::GenFftsPlusContextIdForControlNode(std::vector<ge::NodePtr> &pre_sub_graph_nodes,
                                                                  std::vector<ge::NodePtr> &no_pre_sub_graph_nodes,
                                                                  ControlGraphMap controlnode_graphmap) const {
  for (auto &iter : controlnode_graphmap) {
    for (auto iiter: iter.second) {
      for (auto node : iiter->GetDirectNode()) {
        if (!node) {
          continue;
        }
        if (node->GetType() == "LabelSet") {
          GenContextIdWithLabelSet(node, no_pre_sub_graph_nodes);
        } else if (node->GetType() == "LabelGoto" || node->GetType() == "LabelGotoEx") {
          GenContextIdWithLabelGoto(node, no_pre_sub_graph_nodes);
        } else if (node->GetType() == "LabelSwitchByIndex") {
          GenContextIdWithLabelSwitch(node, pre_sub_graph_nodes, no_pre_sub_graph_nodes);
        } else {
          GenContextIdOhter(node, no_pre_sub_graph_nodes);
        }
      }
    }
  }
  return SUCCESS;
}


Status ManualTheadTaskBuilder::GenFftsPlusContextIdAll(std::vector<ge::NodePtr> &pre_sub_graph_nodes,
                                                       std::vector<ge::NodePtr> &no_pre_sub_graph_nodes,
                                                       std::vector<ge::NodePtr> &sub_graph_nodes,
                                                       uint64_t &ready_context_num,
                                                       uint64_t &total_context_number) const {
  uint32_t contextId = total_context_number;
  uint32_t has_set_contextId = 0;
  for (auto &node : pre_sub_graph_nodes) {
    has_set_contextId = 0;
    if (!ge::AttrUtils::GetInt(node->GetOpDesc(), kContextId, has_set_contextId)) {
      GenFftsPlusContextIdCommon(node, sub_graph_nodes, contextId);
    }
  }
  ready_context_num = contextId;
  for (auto &node : no_pre_sub_graph_nodes) {
    has_set_contextId = 0;
    if (!ge::AttrUtils::GetInt(node->GetOpDesc(), kContextId, has_set_contextId)) {
      GenFftsPlusContextIdCommon(node, sub_graph_nodes, contextId);
    }
  }
  total_context_number = contextId;
  return SUCCESS;
}
}  // namespace ffts
