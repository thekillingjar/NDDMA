/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "automatic_buffer_fusion.h"
#include <utility>
#include "common/fe_log.h"
#include "common/op_slice_util.h"
#include "common/platform_utils.h"
#include "common/scope_allocator.h"
#include "common/aicore_util_constants.h"
#include "common/aicore_util_attr_define.h"
#include "common/fe_inner_error_codes.h"
#include "common/calc_slice_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_constant.h"

namespace fe {
namespace {
const size_t MAX_OUT_BRANCH_NUMBER = 6;
const size_t MAX_NODE_NUMBER_IN_ONE_SCOPE = 28;
const std::string kAttrDumpAble = "_dump_able";
const std::string CONSUMER_FUSIBLE_LIST = "_consumers_fusible_list";
size_t GetMaxUsersCountForSingleOutput(const ge::NodePtr &producer) {
  size_t max_users_count = 0;
  for (auto &output_anchor : producer->GetAllOutDataAnchors()) {
    auto current_output_users_size = output_anchor->GetPeerInDataAnchors().size();
    if (current_output_users_size > max_users_count) {
      max_users_count = current_output_users_size;
    }
  }
  FE_LOGD("The maximum branch number for %s is %zu.", producer->GetName().c_str(), max_users_count);
  return max_users_count;
}

bool IsOpDynamicImpl(const ge::OpDescPtr &op_desc_ptr) {
  if (op_desc_ptr == nullptr) {
    return false;
  }
  bool is_dynamic_impl = false;
  return ge::AttrUtils::GetBool(op_desc_ptr, ATTR_NAME_IS_OP_DYNAMIC_IMPL, is_dynamic_impl) && is_dynamic_impl;
}

bool IsDumpableOp(const ge::OpDescPtr &op_desc) {
  bool isDumpAble = false;
  return ge::AttrUtils::GetBool(op_desc, kAttrDumpAble, isDumpAble) && isDumpAble;
}

/* If the producer is fusible, but it may be unfusible on some
 * output edges. */
bool CheckConsumerFusibleWithProducer(const ge::NodePtr &producer, const ge::NodePtr &consumer) {
  vector<bool> all_consumers_fusible_default;
  vector<bool> all_consumers_fusible =
      producer->GetOpDesc()->TryGetExtAttr(CONSUMER_FUSIBLE_LIST, all_consumers_fusible_default);
  uint32_t loop_count = 0;

  for (auto &consumer_temp : producer->GetOutAllNodes()) {
    if (consumer == consumer_temp) {
      if (loop_count >= all_consumers_fusible.size()) {
        FE_LOGW("The consumer index %u >= the size of attr %s of %s.", loop_count, CONSUMER_FUSIBLE_LIST.c_str(),
                consumer->GetName().c_str());
        return false;
      }
      return all_consumers_fusible[loop_count];
    }
    loop_count++;
  }

  FE_LOGW("Cannot find consumer %s in producer %s's peer output nodes.", consumer->GetName().c_str(),
          producer->GetName().c_str());
  return false;
}
} // namespace

AutomaticBufferFusion::AutomaticBufferFusion(std::unique_ptr<ConnectionMatrix> connection_matrix)
    : max_out_branch_num_(MAX_OUT_BRANCH_NUMBER),
      may_duplicate_(false),
      scope_id_lower_bound_(0),
      connection_matrix_(std::move(connection_matrix)) {
  may_duplicate_ = PlatformUtils::Instance().GetIsaArchVersion() == ISAArchVersion::EN_ISA_ARCH_V220;
}

void AutomaticBufferFusion::SetScopeIdLowerBound() {
  scope_id_lower_bound_ = ScopeAllocator::Instance().GetCurrentScopeId();
  FE_LOGD("Lower bound of Scope Id is %ld.", scope_id_lower_bound_);
}

bool AutomaticBufferFusion::AbleToFuseOnAllPaths(const ge::NodePtr &producer, const ge::NodePtr &consumer,
                                                 const NodeSet &unable_to_fuse,
                                                 std::map<std::pair<ge::NodePtr, ge::NodePtr>, bool> &result) {
  if (producer == consumer) {
    return true;
  }

  if (IsOpDynamicImpl(producer->GetOpDesc()) != IsOpDynamicImpl(consumer->GetOpDesc())) {
    FE_LOGD("%s and %s have different implied types dynamically.", producer->GetOpDesc()->GetName().c_str(),
            consumer->GetOpDesc()->GetName().c_str());
    return false;
  }
  string op_pattern;
  if (!IsFusible(consumer, op_pattern)) {
    FE_LOGD("Consumer %s is not fusable, pattern is %s.", consumer->GetName().c_str(), op_pattern.c_str());
    return false;
  }

  auto iter = result.find(std::make_pair(producer, consumer));
  if (iter != result.end()) {
    FE_LOGD("The result of %s and %s is %u.", producer->GetName().c_str(), consumer->GetName().c_str(), iter->second);
    return iter->second;
  }

  bool curr_result = true;
  auto all_producers = consumer->GetInAllNodes();

  size_t input_size_of_consumer = all_producers.size();
  for (size_t i = 0; i < input_size_of_consumer; ++i) {
    ge::NodePtr consumer_operand = all_producers.at(i);
    // If the operand is not on a path to the producer, it doesn't matter
    // whether it's fusible.
    if (!connection_matrix_->IsConnected(producer, consumer_operand)) {
      FE_LOGD("Producer %s and consumer operand %s are not reachable.", producer->GetName().c_str(),
              consumer_operand->GetName().c_str());
      continue;
    }

    FE_LOGD("producer %s and consumer operand %s are reachable.", producer->GetName().c_str(),
            consumer_operand->GetName().c_str());
    if (unable_to_fuse.count(consumer_operand) > 0) {
      FE_LOGD("Consumer operand %s is not fusible.", consumer_operand->GetName().c_str());
      curr_result = false;
      break;
    }
    // The producer is reachable from consumer_operand which means we need
    // to be able to fuse consumer_operand into consumer in order for
    // producer to be fusible into consumer on all paths.
    // Perform the recursive step: make sure producer can be fused into
    // consumer_operand on all paths.
    if (!AbleToFuseOnAllPaths(producer, consumer_operand, unable_to_fuse, result)) {
      FE_LOGD("Producer %s and consumer operand %s cannot be fused across all paths.", producer->GetName().c_str(),
              consumer_operand->GetName().c_str());
      curr_result = false;
      break;
    }
  }
  result.emplace(std::make_pair(producer, consumer), curr_result);
  return curr_result;
}

AutomaticBufferFusion::NodeSet AutomaticBufferFusion::ComputeAllUnFusibleNodes(ge::ComputeGraph &graph) {
  /* All nodes in unable_to_fuse is unfusible as producer.
   * So they may be fusible as consumer. */
  NodeSet unable_to_fuse;
  std::map<std::pair<ge::NodePtr, ge::NodePtr>, bool> result;
  auto nodes = graph.GetDirectNode();
  if (nodes.empty()) {
    return unable_to_fuse;
  }
  int last_index = static_cast<int>(nodes.size() - 1);
  for (int loop_index = last_index; loop_index >= 0; loop_index--) {
    ge::NodePtr producer = nodes.at(loop_index);
    if (producer == nullptr) {
      continue;
    }
    string producer_name = producer->GetName();
    string op_pattern;

    if (IsFusible(producer, op_pattern)) {
      if (GetMaxUsersCountForSingleOutput(producer) > max_out_branch_num_) {
        FE_LOGI("One input of producer %s contains more than %zu users.", producer_name.c_str(), max_out_branch_num_);
        unable_to_fuse.insert(producer);
        continue;
      }
      /* For each producer and its consumer, if they can fused together,
       * count the total fusible nodes, it's depends on the consumers's fusible
       * node number. If the consumer can not be fused with the consumer's
       * consumer, than the total fusible nodes by fuse producer and consumer
       * is 1. For example:
       * A  -> B -> C
       *        \-> D
       * If B cannot be fused with C and D, than fuse A and B will make B
       * disappear and it becomes:
       * AB -> C
       *   \-> D,
       * So size of fusible nodes is 2 and if C and D is also fusible with B,
       * all these four nodes will be fused to one. The size of fusible nodes
       * is four.
       *
       * The total of duplicated nodes is 1 (only need to duplicate the prodcuer
       * once no matter how many consumers are unfusible).
       * Duplication is because there are some consumers which can
       * not be fused and they need the original output of the producer. We must
       * duplicate that producer to make the calculation result correct. The
       * duplication will introducer more nodes in the graph. So we need to
       * balance the duplication.
       * For example:
       * Data- >A -> B -> C -> E
       *             \-> D -> F
       * given A,B,C is fusible and B and D is not fusible, if we want to fuse
       * A,B,C the graph will be look like:
       * Data -> ABC -> E
       *      \->  A -> D -> F
       * A is duplicated for D and D's consumers.
       * In this case, when we fuse C and B, the income is 2 nodes (actually
       * the time income will be less than 2 * coefficient = 1 and the
       * coefficient is
       * because B and C still will be computed in AI-Core,
       * so the computation time can not be omitted and we assume the time
       * of moving data from memory to UB is as same as computational time.)
       * And if total size of fusible nodes is larger than 2 or duplication is
       * not necessary. We will consider the producer is fusible. */
      auto all_consumers = producer->GetOutAllNodes();
      vector<bool> fusible_with_all_consumers(all_consumers.size(), false);
      for (size_t i = 0; i < all_consumers.size(); i++) {
        auto &consumer = all_consumers.at(i);
        if (consumer == nullptr) {
          continue;
        }
        bool ret = AbleToFuseOnAllPaths(producer, consumer, unable_to_fuse, result);
        if (!ret) {
          FE_LOGI("Cannot fuse producer %s and consumer %s.", producer_name.c_str(), consumer->GetName().c_str());
          /* producer and consumer cannot be fused, */
          if (!may_duplicate_ && producer->GetType() != "BNTrainingUpdateV2") {
            FE_LOGD("Do not allow duplicates; producer %s is unfusible.", producer_name.c_str());
            unable_to_fuse.insert(producer);
            break;
          }
        } else {
          fusible_with_all_consumers[i] = true;
        }
      }
      /* Set extra attributes to record which consumer is fusible. */
      producer->GetOpDesc()->SetExtAttr(CONSUMER_FUSIBLE_LIST, fusible_with_all_consumers);
      continue;
    }
    unable_to_fuse.insert(producer);
  }
  return unable_to_fuse;
}

bool AutomaticBufferFusion::IsScopeIdValid(const ge::NodePtr &node, int64_t &scope_id) const {
  if (ScopeAllocator::GetScopeAttr(node->GetOpDesc(), scope_id)) {
    FE_LOGD("Node %s is a fusion node with a scope_id of %ld.", node->GetName().c_str(), scope_id);
    if (scope_id <= scope_id_lower_bound_) {
      FE_LOGD("Node %s is fused; skipping this node. The lower bound is %ld.", node->GetName().c_str(), scope_id_lower_bound_);
      return false;
    }
  } else {
    scope_id = -1;
  }
  return true;
}

bool AutomaticBufferFusion::CheckPathExists(const ge::NodePtr &node1, int64_t consumer_scope_id,
                                            const ge::NodePtr &consumer) const {
  auto old_scope_id_iter = scope_id_nodes_map_.find(consumer_scope_id);
  if (old_scope_id_iter == scope_id_nodes_map_.end()) {
    FE_LOGD("Could not find any nodes in old_scope_id %ld.", consumer_scope_id);
    return connection_matrix_->IsConnected(node1, consumer);
  }

  for (auto &node_iter : old_scope_id_iter->second.nodes) {
    if (connection_matrix_->IsConnected(node1, node_iter.second)) {
      FE_LOGD("Node1 %s and node2 %s in scope id %ld is reachable.", node1->GetName().c_str(),
              node_iter.second->GetName().c_str(), consumer_scope_id);
      return true;
    }
  }
  return false;
}

void AutomaticBufferFusion::GetAllNodesByScopeId(int64_t ScopeId, vector<ge::NodePtr> &all_nodes,
                                                 const ge::NodePtr &producer) {
  auto iter = scope_id_nodes_map_.find(ScopeId);
  if (iter == scope_id_nodes_map_.end()) {
    all_nodes.emplace_back(producer);
  } else {
    for (auto &pair : iter->second.nodes) {
      all_nodes.emplace_back(pair.second);
    }
  }
}

bool AutomaticBufferFusion::CheckLoopExistAfterFusion(const ge::NodePtr &producer, const ge::NodePtr &consumer,
                                                      int64_t producer_scope_id, int64_t consumer_scope_id,
                                                      const NodeSet &unable_to_fuse) {
  if (consumer_scope_id == -1) {
    /* That means consumer is the first node trying to fuse into producer.
     * Before this function we have checked whether there is a loop if we fuse
     * producer and consumer. */
    auto count_unable_to_fuse = unable_to_fuse.count(producer);
    if (count_unable_to_fuse != 0) {
      return true;
    }
  }

  /* For all nodes in producer's scope_id, we need to do loop check */
  vector<ge::NodePtr> all_producers;
  GetAllNodesByScopeId(producer_scope_id, all_producers, producer);
  auto iter_consumer_scope = scope_id_nodes_map_.find(consumer_scope_id);

  for (auto &node : all_producers) {
    for (auto &output : node->GetOutAllNodes()) {
      if (output == consumer) {
        continue;
      }

      /* Because there contains control edges, so we
          always check the loop no matter whether the output node is
          fused by automatic ub fusion or not.
          Only when the output node is from the consumer's scope,
          we do not need to check cycle. */
      if (iter_consumer_scope != scope_id_nodes_map_.end()) {
        const std::unordered_map<int64_t, ge::NodePtr> &all_consumer_ids = iter_consumer_scope->second.nodes;
        if (all_consumer_ids.count(output->GetOpDesc()->GetId()) != 0) {
          continue;
        }
      }

      /* Check whether there is a path from output to any node in consumer's scope */
      int64_t output_scope_id = -2;
      (void)ScopeAllocator::GetScopeAttr(output->GetOpDesc(), output_scope_id);
      if (output_scope_id == producer_scope_id) { // Inner output from the same scope_id with producer_scope_id, Ignore
        continue;
      }
      std::vector<ge::NodePtr> all_outputs;
      GetAllNodesByScopeId(output_scope_id, all_outputs, output);
      for (auto &scope_node : all_outputs) {
        if (scope_node == producer) {
          continue;
        }
        if (CheckPathExists(scope_node, consumer_scope_id, consumer)) {
          FE_LOGD("Loop exists when fusing %s and %s", node->GetName().c_str(), consumer->GetName().c_str());
          return true;
        }
      }
    }
  }

  FE_LOGD("No loop exists when fusing %s and %s.", producer->GetName().c_str(), consumer->GetName().c_str());
  return false;
}

void AutomaticBufferFusion::FuseOneProducer(ge::ComputeGraph &graph, const ge::NodePtr &consumer,
                                            int64_t consumer_scope_id, const string &node_name,
                                            const NodeSet &unable_to_fuse) {
  for (auto &producer : consumer->GetInDataNodes()) {
    std::vector<ge::NodePtr> fusion_nodes = {consumer, producer};
    std::string reason_not_support;
    if (!graph.IsSupportFuse(fusion_nodes, reason_not_support)) {
      FE_LOGD("IsSupportFuse unsuccessful, reason is [%s].", reason_not_support.c_str());
      continue;
    }

    string op_pattern;
    if (!IsFusible(producer, op_pattern)) {
      FE_LOGD("Operand %s of the consumer %s cannot be fused due to pattern %s.", producer->GetName().c_str(),
              node_name.c_str(), op_pattern.c_str());
      continue;
    }
    if (!OpSliceUtil::IsSgtInfoConsistant(consumer, producer)) {
      FE_LOGD("Consumer %s and producer %s do not have the same sgt info.",
              consumer->GetName().c_str(), producer->GetName().c_str());
      continue;
    }
    FE_LOGD("Fusing consumer %s with producer %s.", consumer->GetName().c_str(), producer->GetName().c_str());

    int64_t producer_scope_id = -1;
    bool is_producer_scope_id_valid = IsScopeIdValid(producer, producer_scope_id);
    auto CountUnableToFuse = unable_to_fuse.count(producer);
    if (!is_producer_scope_id_valid) {
      continue;
    }

    if (CountUnableToFuse != 0) {
      FE_LOGD("Producer %s is on the unfusible list. ScopeIdValid: %u", producer->GetName().c_str(),
              is_producer_scope_id_valid);
      continue;
    }
    /* Producer must be one of:
     * 1. unfused node.
     * 2. fused in automatic ub fusion instead of built-in ub fusion. */
    bool loop_exist_after_fusion =
        CheckLoopExistAfterFusion(producer, consumer, producer_scope_id, consumer_scope_id, unable_to_fuse);
    if (!loop_exist_after_fusion && CheckConsumerFusibleWithProducer(producer, consumer)) {
      Status ret = FuseTwoNodes(producer, consumer, producer_scope_id, consumer_scope_id);
      if (ret == FAILED) {
        FE_LOGD("Fuse producer %s and consumer %s unsuccessful.", producer->GetName().c_str(),
                consumer->GetName().c_str());

        break;
      }
      auto scope_info = scope_id_nodes_map_.find(consumer_scope_id);
      if (scope_info != scope_id_nodes_map_.end()) {
        vector<ge::NodePtr> local_fusion_nodes;
        for (const auto &ele : scope_info->second.nodes) {
          local_fusion_nodes.emplace_back(ele.second);
        }
        connection_matrix_->Update(graph, local_fusion_nodes);
      }
    }
  }
}

void AutomaticBufferFusion::CalcSliceInfo(const ge::ComputeGraph &graph,
                                          const ge::ComputeGraph::Vistor<ge::NodePtr> &nodes) {
  std::unordered_map<int64_t, vector<ge::NodePtr>> scope_nodes;
  for (auto &node : nodes) {
    int64_t scope_id = 0;
    (void)ScopeAllocator::GetScopeAttr(node->GetOpDesc(), scope_id);
    if (scope_id > scope_id_lower_bound_) {
      FE_LOGD("After auto UB fusion, node %s's scope_id is %ld.", node->GetName().c_str(), scope_id);
      if (ge::AttrUtils::SetStr(node->GetOpDesc(), kPassNameUbAttr, "AutomaticUbFusion")) {
        FE_LOGD("Node[%s]: set pass_name[AutomaticUbFusion] successfully.", node->GetName().c_str());
      }
      if (ge::AttrUtils::SetStr(node->GetOpDesc(), UB_FUSION_OP_TYPE, "AutomaticBufferFusionOp")) {
        FE_LOGD("Node[%s]: set ub fusion op_type [AutomaticBufferFusionOp] successfully.", node->GetName().c_str());
      }
      auto iter = scope_nodes.find(scope_id);
      if (iter == scope_nodes.end()) {
        vector<ge::NodePtr> fusion_nodes = {node};
        scope_nodes.emplace(std::make_pair(scope_id, fusion_nodes));
      } else {
        iter->second.emplace_back(node);
      }
    }
  }

  FE_LOGD("Finish auto-ub-fusion for graph %s. Total fused times is %zu.", graph.GetName().c_str(), scope_nodes.size());
  bool set_slice_info = true;
  (void)ge::AttrUtils::GetBool(graph, kNeedSetSliceInfo, set_slice_info);
  if (set_slice_info) {
    BufferFusionPassBasePtr buffer_fusion_pass_base_ptr = nullptr;
    for (auto &one_scope : scope_nodes) {
      CalcSliceUtils::CalcSliceInfo(buffer_fusion_pass_base_ptr, one_scope.second);
    }
  }
}

Status AutomaticBufferFusion::Run(ge::ComputeGraph &graph) {
  FE_LOGD("Starting automatic buffer fusion for graph %s.", graph.GetName().c_str());
  /* Get the reverse post order traverse result by topo logical sorting. */
  Status ret = graph.TopologicalSorting();
  if (ret != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][AutoUb][TopoSort] Topological sorting for graph [%s] before auto fusion failed.",
                    graph.GetName().c_str());
    return FAILED;
  }
  SetScopeIdLowerBound();
  if (connection_matrix_ == nullptr) {
    connection_matrix_ = std::unique_ptr<ConnectionMatrix>(new (std::nothrow) ConnectionMatrix());
    FE_CHECK_NOTNULL(connection_matrix_);
    connection_matrix_->Generate(graph);
  }

  NodeSet unable_to_fuse;

  unable_to_fuse = ComputeAllUnFusibleNodes(graph);
  auto nodes = graph.GetDirectNode();
  /* Loop using reverse post order */
  auto size_of_all_nodes = nodes.size();
  for (int loop_index = static_cast<int>(size_of_all_nodes - 1); loop_index >= 0; loop_index--) {
    const auto consumer = nodes.at(loop_index);
    FE_CHECK_NOTNULL(consumer);
    auto node_name = consumer->GetName();
    string op_pattern;
    if (!IsFusible(consumer, op_pattern)) {
      FE_LOGD("Consumer %s is not fusionable, pattern is %s.", node_name.c_str(), op_pattern.c_str());
      continue;
    }

    int64_t consumer_scope_id = -1;
    if (!IsScopeIdValid(consumer, consumer_scope_id)) {
      continue;
    }
    FuseOneProducer(graph, consumer, consumer_scope_id, node_name, unable_to_fuse);
  }
  CalcSliceInfo(graph, nodes);
  return SUCCESS;
}

Status AutomaticBufferFusion::FuseTwoNodes(const ge::NodePtr &producer, const ge::NodePtr &consumer,
                                           int64_t producer_scope_id, int64_t &consumer_scope_id) {
  FE_LOGD("Fusing producer %s and consumer %s with IDs %ld and %ld.", producer->GetName().c_str(),
          consumer->GetName().c_str(), producer_scope_id, consumer_scope_id);

  int64_t final_scope_id;
  if (producer_scope_id == -1) {
    /* The producer is not fused. Set scope id according to the consumer. */
    if (consumer_scope_id == -1) {
      /* The consumer is also not fused. Use a completely new scope id */
      int64_t new_scope_id = ScopeAllocator::Instance().AllocateScopeId();
      /* node number in side new scope id will not exceed
       * the limit of MAX_NODE_NUMBER_IN_ONE_SCOPE nodes. */
      if (IsOpDynamicImpl(producer->GetOpDesc()) != IsOpDynamicImpl(consumer->GetOpDesc())) {
        FE_LOGD("%s and %s have different implied types dynamically.", producer->GetOpDesc()->GetName().c_str(),
                consumer->GetOpDesc()->GetName().c_str());
        return SUCCESS;
      }
      (void)SetAndRecordScopeId(producer, new_scope_id);

      (void)SetAndRecordScopeId(consumer, new_scope_id);
      final_scope_id = new_scope_id;
      consumer_scope_id = new_scope_id;
    } else {
      /* Use consumer's scope Id. */
      if (SetAndRecordScopeId(producer, consumer_scope_id) != SUCCESS) {
        FE_LOGI("Set scope %ld, unable to set Id for producer %s", consumer_scope_id, producer->GetName().c_str());
        return FAILED;
      }
      final_scope_id = consumer_scope_id;
    }
  } else {
    /* The producer is fused by automatic ub fusion. */
    if (consumer_scope_id == -1) {
      /* Follow the producer's scope id */
      if (SetAndRecordScopeId(consumer, producer_scope_id) != SUCCESS) {
        FE_LOGI("Set scope %ld, unable to set Id for producer %s.", producer_scope_id, consumer->GetName().c_str());
        return FAILED;
      }
      consumer_scope_id = producer_scope_id;
    } else {
      /* In this case, both producer and consumer has its own scope id, we
       * change the consumer's id using the producer's because topologically,
       * the producer is set with a scope id, it must be fused from another path
       * to the leaf node. And the nodes count in that path should be equal or
       * larger than the path we are currently on.
       *
       * Change all nodes with consumer's original scope id to the producer's
       * scope id. */
      Status ret = ChangeScopeId(consumer_scope_id, producer_scope_id);
      if (ret != SUCCESS && ret != GRAPH_OPTIMIZER_NOT_FUSE_TWO_SCOPE) {
        FE_LOGI("Change scope for producer %s and consumer %s was not successfully.", producer->GetName().c_str(),
                consumer->GetName().c_str());
        return FAILED;
      }
      if (ret != GRAPH_OPTIMIZER_NOT_FUSE_TWO_SCOPE) {
        consumer_scope_id = producer_scope_id;
      }
    }
    final_scope_id = producer_scope_id;
  }
  FE_LOGD("Final scope Id is %ld.", final_scope_id);
  return SUCCESS;
}

Status AutomaticBufferFusion::SetAndRecordScopeId(const ge::NodePtr &node, int64_t scope_id) {
  const auto iter = scope_id_nodes_map_.find(scope_id);
  bool is_dynamic_impl = IsOpDynamicImpl(node->GetOpDesc());
  if (iter == scope_id_nodes_map_.end()) {
    auto node_id = node->GetOpDesc()->GetId();
    ScopeInfo scope_info;
    scope_info.nodes[node_id] = node;
    scope_info.is_dynamic_impl = is_dynamic_impl;
    scope_id_nodes_map_.emplace(std::make_pair(scope_id, scope_info));
  } else {
    if (iter->second.is_dynamic_impl != is_dynamic_impl) {
      FE_LOGD("Cannot fuse %s with scope %ld. Dynamic implementation types are %d and %d.",
              node->GetName().c_str(), scope_id, iter->second.is_dynamic_impl, is_dynamic_impl);
      return SUCCESS;
    }

    if (iter->second.nodes.size() > MAX_NODE_NUMBER_IN_ONE_SCOPE) {
      FE_LOGW("Total element wise fusion op number is greater than %zu!", MAX_NODE_NUMBER_IN_ONE_SCOPE+1);
      return FAILED;
    }

    auto node_id = node->GetOpDesc()->GetId();
    iter->second.nodes.emplace(std::make_pair(node_id, node));
  }

  ScopeAllocator::SetScopeAttr(node->GetOpDesc(), scope_id);
  return SUCCESS;
}

Status AutomaticBufferFusion::ChangeScopeId(int64_t old_scope_id, int64_t new_scope_id) {
  if (old_scope_id == -1) {
    return SUCCESS;
  }
  if (old_scope_id == new_scope_id) {
    FE_LOGD("The old scope ID is the same as the new scope ID, which is %ld.", new_scope_id);
    return SUCCESS;
  } else {
    auto old_scope_id_iter = scope_id_nodes_map_.find(old_scope_id);
    if (old_scope_id_iter == scope_id_nodes_map_.end()) {
      FE_LOGW("Could not find any nodes in old_scope_id %ld.", old_scope_id);
      return SUCCESS;
    } else {
      auto fused_node_size_of_old_scope_id = old_scope_id_iter->second.nodes.size();
      auto new_scope_id_iter = scope_id_nodes_map_.find(new_scope_id);
      if (new_scope_id_iter == scope_id_nodes_map_.end()) {
        FE_LOGW("Cannot find any nodes in new_scope_id %ld.", new_scope_id);
        return SUCCESS;
      }
      auto fused_node_size_of_new_scope_id = new_scope_id_iter->second.nodes.size();
      if (fused_node_size_of_old_scope_id + fused_node_size_of_new_scope_id > MAX_NODE_NUMBER_IN_ONE_SCOPE) {
        FE_LOGW("The difference between two scopes exceeds 28. Sizes are %zu and %zu.", fused_node_size_of_old_scope_id,
                fused_node_size_of_new_scope_id);
        return GRAPH_OPTIMIZER_NOT_FUSE_TWO_SCOPE;
      }
      for (auto &node_id_map : old_scope_id_iter->second.nodes) {
        FE_LOGD("Changed the scope_id of %s from %ld to %ld.", node_id_map.second->GetName().c_str(), old_scope_id,
                new_scope_id);
        if (SetAndRecordScopeId(node_id_map.second, new_scope_id) != SUCCESS) {
          FE_LOGW("Failed to set scope %ld for producer %s", new_scope_id, node_id_map.second->GetName().c_str());
          return FAILED;
        }
      }
      scope_id_nodes_map_.erase(old_scope_id);
    }
  }
  return SUCCESS;
}

bool AutomaticBufferFusion::IsTbeOp(const ge::NodePtr &node) const {
  FE_CHECK((node == nullptr), FE_LOGD("Null node encountered when judging TVMOp."), return false);
  int64_t type = 0;
  (void)ge::AttrUtils::GetInt(node->GetOpDesc(), ge::ATTR_NAME_IMPLY_TYPE, type);
  const bool res = (type == static_cast<int64_t>(domi::ImplyType::TVM));
  return res;
}

bool AutomaticBufferFusion::GetOpAttrPattern(const ge::NodePtr &node, string &op_pattern) const {
  FE_CHECK((node == nullptr), FE_LOGD("Node is null."), return false);
  string name = node->GetName();
  if (!ge::AttrUtils::GetStr(node->GetOpDesc(), kOpPattern, op_pattern)) {
    FE_LOGD("Node [%s] unable to get pattern.", name.c_str());
    return false;
  }

  if (op_pattern == "") {
    FE_LOGD("optype is empty for node name [%s].", name.c_str());
    return false;
  }

  return true;
}

bool AutomaticBufferFusion::IsFusible(const ge::NodePtr &node, std::string &op_pattern) const {
  bool result = IsTbeOp(node) && !IsDumpableOp(node->GetOpDesc()) && GetOpAttrPattern(node, op_pattern) &&
                (op_pattern == OP_PATTERN_ELEMWISE || op_pattern == OP_PATTERN_BROAD_CAST);
  auto opdesc = node->GetOpDesc();
  
  string matched_pass_name;
  if (ge::AttrUtils::GetStr(opdesc, kPassNameUbAttr, matched_pass_name) && !matched_pass_name.empty()) {
    return false;
  }
  int64_t scope_id = -1;
  if (ScopeAllocator::GetScopeAttr(node->GetOpDesc(), scope_id) && scope_id <= scope_id_lower_bound_) {
    /* If a node has been fused by built-in pass, it's not fusible. */
    return false;
  }
  bool add_n_condition = opdesc->GetType() == "AddN" && node->GetAllInDataAnchorsSize() > MAX_OUT_BRANCH_NUMBER;
  /* Restrict that one AddN can only contain 6 input tensors. */
  FE_CHECK(add_n_condition, FE_LOGD("AddN %s has more than 6 inputs, skipping it.", node->GetName().c_str()), return false);
  return result;
}
}
