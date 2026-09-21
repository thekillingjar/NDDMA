/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/graph_builder.h"
#include "graph/build/memory/graph_mem_assigner.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/helper/model_helper.h"
#include "graph/build/run_context_util.h"
#include "graph/build/stream_graph_optimizer.h"
#include "graph/build/memory/var_mem_assign_util.h"
#include "graph/build/stream/stream_utils.h"
#include "graph/build/stream/dynamic_stream_allocator.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "graph/ge_context.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/passes/memory_conflict/mark_same_addr_pass.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/type_utils_inner.h"
#include "api/gelib/gelib.h"
#include "common/model/ge_model.h"
#include "common/math/math_util.h"
#include "engines/manager/opskernel_manager/ops_kernel_builder_manager.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/def_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_type_utils.h"
#include "task_generator_utils.h"
#include "graph/unfold/graph_unfolder.h"
#include "graph/passes/feature/super_kernel_pass.h"
#include "common/compile_profiling/ge_trace_wrapper.h"

using domi::BuildMode;

namespace {
const int32_t kDecimalBase = 10;
const int32_t kInvalidPerfLevel = -1;
const int64_t kProfilingArStep = 2;
const int64_t kProfilingArStartLogid = 10000;
const int64_t kProfilingGetNextStartLogid = 20000;
const uint32_t kSubgraphIndexOfPartitionedCall = 0U;
constexpr const ge::char_t *kConstLifecycleGraph = "graph";
constexpr const ge::char_t *kConstLifecycleSession = "session";
enum class NodeType : uint32_t {
  kSubgraphData = 0U,
  kSubgraphNode = 1U,
  kOthers = 2U
};

bool IsInRange(const int64_t val, const int64_t left, const int64_t right) {
  if ((val >= left) && (val < right)) {
    return true;
  }
  return false;
}
}  // namespace
namespace ge {
static NodeType TransferNodeType(const NodePtr &node) {
  const std::string type = node->GetType();
  if (type == ge::DATA) {
    if (node->GetOwnerComputeGraph()->GetParentNode() == nullptr) {
      GELOGD("access src data node:%s", node->GetName().c_str());
      return NodeType::kOthers;
    }
    GELOGD("access subgraph input node:%s", node->GetName().c_str());
    return NodeType::kSubgraphData;
  } else if (type == PARTITIONEDCALL) {
    GELOGD("access subgraph node:%s", node->GetName().c_str());
    return NodeType::kSubgraphNode;
  }
  GELOGD("access other node:%s", node->GetName().c_str());
  return NodeType::kOthers;
}

static Status HandleSubgraphNode(NodePtr &src_node, OutDataAnchorPtr &src_out_anchor) {
  auto subgraph = NodeUtils::GetSubgraph(*src_node, 0);
  GE_CHECK_NOTNULL(subgraph);
  const NodePtr &net_output_node = subgraph->FindFirstNodeMatchType(NETOUTPUT);
  GE_CHECK_NOTNULL(net_output_node);
  const InDataAnchorPtr &in_data_anchor = net_output_node->GetInDataAnchor(src_out_anchor->GetIdx());
  GE_CHECK_NOTNULL(in_data_anchor);
  const OutDataAnchorPtr &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(peer_out_anchor);

  src_node = peer_out_anchor->GetOwnerNode();
  src_out_anchor = peer_out_anchor;
  return SUCCESS;
}

static Status HandleSubgraphDataNode(NodePtr &src_node, OutDataAnchorPtr &src_out_anchor) {
  uint32_t index = 0;
  if (!AttrUtils::GetInt(src_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, index)) {
    REPORT_INNER_ERR_MSG("E19999", "get attr:%s failed from node:%s",
                       ATTR_NAME_PARENT_NODE_INDEX.c_str(), src_node->GetName().c_str());
    GELOGE(FAILED, "[Get][Attr] %s failed, node:%s.", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
           src_node->GetName().c_str());
    return FAILED;
  }
  const NodePtr &parent_node = src_node->GetOwnerComputeGraph()->GetParentNode();
  GE_CHECK_NOTNULL(parent_node);
  const InDataAnchorPtr &in_data_anchor = parent_node->GetInDataAnchor(index);
  GE_CHECK_NOTNULL(in_data_anchor);
  const OutDataAnchorPtr &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(peer_out_anchor);

  src_node = peer_out_anchor->GetOwnerNode();
  src_out_anchor = peer_out_anchor;
  return SUCCESS;
}

static void ConvertOpType(const ge::ComputeGraphPtr &compute_graph, const std::string &old_type, const std::string &new_type) {
  for (const auto &node : compute_graph->GetAllNodes()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc->GetType() == old_type) {
      ge::OpDescUtilsEx::SetType(op_desc, new_type);
    }
  }
}

static void ConvertConstOrConstantByLifecycle(const ge::ComputeGraphPtr &compute_graph) {
  bool train_flag = false;
  std::string run_mode;
  if ((GetContext().GetOption(ge::OPTION_GRAPH_RUN_MODE, run_mode) == SUCCESS) && (!run_mode.empty())) {
    if (GraphRunMode(std::strtol(run_mode.c_str(), nullptr, kDecimalBase)) >= TRAIN) {
      train_flag = true;
    }
  }

  std::string memory_optimization_policy;
  ge::GetContext().GetOption(ge::MEMORY_OPTIMIZATION_POLICY, memory_optimization_policy);
  std::string const_lifecycle;
  (void)ge::GetContext().GetOption(ge::OPTION_CONST_LIFECYCLE, const_lifecycle);
  const bool enable_lifecycle_graph =
      ((memory_optimization_policy == ge::kMemoryPriority) || (const_lifecycle == kConstLifecycleGraph));

  /* 使用 Session 级 Const 生命周期: 1)在线推理场景权重复用；2）训练场景且不使能内存优先和图级 Const 生命周期;
   * 使用 Graph 级 Const 生命周期: 使能内存优先，或者通过 OPTION_CONST_LIFECYCLE 指定图级 Const 生命周期
   */
  if ((const_lifecycle == kConstLifecycleSession) || (train_flag && (!enable_lifecycle_graph))) {
    GELOGI(
        "Convert Const to Constant, OPTION_GRAPH_RUN_MODE=%s, MEMORY_OPTIMIZATION_POLICY=%s, OPTION_CONST_LIFECYCLE=%s",
        run_mode.c_str(), memory_optimization_policy.c_str(), const_lifecycle.c_str());
    ConvertOpType(compute_graph, ge::CONSTANT, ge::CONSTANTOP);
  } else if (enable_lifecycle_graph) {
    GELOGI(
        "Convert Constant to Const in memory priority, OPTION_GRAPH_RUN_MODE=%s, MEMORY_OPTIMIZATION_POLICY=%s, "
        "OPTION_CONST_LIFECYCLE=%s",
        run_mode.c_str(), memory_optimization_policy.c_str(), const_lifecycle.c_str());
    ConvertOpType(compute_graph, ge::CONSTANTOP, ge::CONSTANT);
  }
}

static void RefreshStreamAttrs(AttrUtils::AttrHolderAdapter &&attr_holder, const int64_t total_stream_num,
                               const int64_t main_stream_num, const int64_t event_num, const int64_t notify_num,
                               const std::vector<uint32_t> &notify_types) {
  (void)AttrUtils::SetInt(attr_holder.get(), ATTR_MODEL_STREAM_NUM, total_stream_num);
  (void)AttrUtils::SetInt(attr_holder.get(), ATTR_MODEL_EVENT_NUM, event_num);
  (void)AttrUtils::SetInt(attr_holder.get(), ATTR_MODEL_NOTIFY_NUM, notify_num);
  (void)AttrUtils::SetListInt(attr_holder.get(), ATTR_MODEL_NOTIFY_TYPES, notify_types);
  (void)AttrUtils::SetInt(attr_holder.get(), "_attached_stream_num", total_stream_num - main_stream_num);
}

GraphBuilder::GraphBuilder() : build_mode_(BuildMode::GEN_TASK_WITH_FUSION), hcom_parallel_(false) {}

void GraphBuilder::SetOptions(const ge::GraphManagerOptions &options) {
  stream_max_parallel_num_ = options.stream_max_parallel_num;
  hcom_parallel_ = options.hcom_parallel;

  if (options.perf_level == kInvalidPerfLevel) {
    build_mode_ = static_cast<int32_t>(BuildMode::GEN_TASK_WITH_FUSION);
  } else {
    build_mode_ = options.perf_level;
  }
}

Status GraphBuilder::CalcOpParam(const ge::ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  auto instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "check gelib instance null, graph:%s",
                       graph->GetName().c_str());
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Check][GELib] GraphBuilder: GE has not initialized, graph:%s",
           graph->GetName().c_str());
    return GE_CLI_GE_NOT_INITIALIZED;
  }

  for (const auto &node_ptr : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    GE_CHECK_NOTNULL(node_ptr->GetOpDesc());
    std::string kernel_lib_name = node_ptr->GetOpDesc()->GetOpKernelLibName();
    if (kernel_lib_name.empty()) {
      // reset op kernel lib
      (void)instance_ptr->DNNEngineManagerObj().GetDNNEngineName(node_ptr);
      kernel_lib_name = node_ptr->GetOpDesc()->GetOpKernelLibName();
      if (kernel_lib_name.empty()) {
        REPORT_INNER_ERR_MSG("E19999", "op kernel lib is empty in node:%s(%s)",
                           node_ptr->GetName().c_str(), node_ptr->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Get][KernelLibName] of node:%s(%s) failed.", node_ptr->GetName().c_str(),
               node_ptr->GetType().c_str());
        return INTERNAL_ERROR;
      }
    }

    auto ret = SetInputSize(node_ptr);
    if (ret != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Set node:%s(%s) inputDesc size failed",
                        node_ptr->GetName().c_str(), node_ptr->GetType().c_str());
      GELOGE(ret, "[Set][InputSize] to node:%s failed.", node_ptr->GetName().c_str());
      return ret;
    }

    ret = OpsKernelBuilderManager::Instance().CalcOpRunningParam(*node_ptr);
    if (ret != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Call Calculate op:%s(%s) running param failed",
                        node_ptr->GetName().c_str(), node_ptr->GetType().c_str());
      GELOGE(ret, "[Call][Calculate] op running param failed, node name is %s", node_ptr->GetName().c_str());
      return ret;
    }
    GE_CHK_STATUS_RET(AddOutputMemTypeForNode(node_ptr));
  }

  auto parent_node = graph->GetParentNode();
  if (parent_node == nullptr) {
    return SUCCESS;
  }

  GE_CHK_STATUS_RET(UpdateParentNodeOutputSize(graph, parent_node));
  GELOGI("Success to calculate op running param.");
  return SUCCESS;
}

Status GraphBuilder::UpdateParentNodeOutputSize(const ge::ComputeGraphPtr &graph,
                                                const ge::NodePtr &parent_node_ptr) const {
  GELOGI("Begin to update parent node[%s] of graph[%s] output size.", parent_node_ptr->GetName().c_str(),
         graph->GetName().c_str());
  auto parent_op_desc = parent_node_ptr->GetOpDesc();
  GE_CHECK_NOTNULL(parent_op_desc);
  bool is_unknown_shape = graph->GetGraphUnknownFlag();
  if (is_unknown_shape) {
    GELOGI("Current graph[%s] is unknown, no need to update parent node[%s], output size.", graph->GetName().c_str(),
           parent_node_ptr->GetName().c_str());
    return SUCCESS;
  }
  for (const auto &node_ptr : graph->GetDirectNode()) {
    if (node_ptr->GetType() != NETOUTPUT) {
      continue;
    }
    auto op_desc = node_ptr->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    for (const auto &in_data_anchor : node_ptr->GetAllInDataAnchors()) {
      auto index = in_data_anchor->GetIdx();
      ge::GeTensorDesc desc_temp = op_desc->GetInputDesc(index);
      uint32_t parent_index = 0;
      if (!AttrUtils::GetInt(desc_temp, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
        GELOGI("NetOutput input tensor %d, attr %s not found.", index, ATTR_NAME_PARENT_NODE_INDEX.c_str());
        continue;
      }

      int64_t size = 0;
      GE_IF_BOOL_EXEC(ge::TensorUtils::GetSize(desc_temp, size) != SUCCESS, GELOGI("Tensor has no size!"));
      ge::GeTensorDesc parent_desc_temp = parent_op_desc->GetOutputDesc(parent_index);
      ge::TensorUtils::SetSize(parent_desc_temp, size);
      GE_CHK_STATUS_RET(parent_op_desc->UpdateOutputDesc(parent_index, parent_desc_temp));
      GELOGI("Update parent node[%s] output index[%u] to size[%ld].", parent_node_ptr->GetName().c_str(), parent_index,
             size);
    }
  }
  return SUCCESS;
}

Status GraphBuilder::Build(ComputeGraphPtr &comp_graph, GeRootModelPtr &ge_root_model_ptr, uint64_t session_id) {
  FuncPerfScope func_perf_scope("GraphBuilder", __FUNCTION__);
  if (comp_graph == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "check compute_graph nullptr, session_id:%lu", session_id);
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[Check][Param] comp_graph is null, session_id:%lu", session_id);
    return GE_GRAPH_PARAM_NULLPTR;
  }
  ge_root_model_ptr = MakeShared<ge::GeRootModel>();
  if (ge_root_model_ptr == nullptr) {
    return MEMALLOC_FAILED;
  }

  // when Constant's memory is large, can be converted to Const,
  // because Const's memory can be released when model is unload
  ConvertConstOrConstantByLifecycle(comp_graph);
  GE_ASSERT_SUCCESS(GraphMemoryAssigner::AssignVarMemory(comp_graph));

  GeModelPtr ge_model_ptr = nullptr;
  bool is_dynamic_shape = false;
  // To be compatible with the old process, do not verify the return value temporarily.
  (void)AttrUtils::GetBool(comp_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape);
  if (is_dynamic_shape || comp_graph->GetGraphUnknownFlag()) {
    GE_CHK_STATUS_RET(
        BuildForDynamicShapeGraph(comp_graph, ge_root_model_ptr, ge_model_ptr, session_id, true),
        "[Build][DynamicShapeGraph] failed, graph:%s, session id:%lu.", comp_graph->GetName().c_str(), session_id);
    GE_ASSERT_SUCCESS(ge_root_model_ptr->Initialize(comp_graph));
  } else {
    GE_CHK_STATUS_RET(BuildForKnownShapeGraph(comp_graph, ge_model_ptr, session_id, true),
                      "[Build][KnownShapeGraph] failed, graph:%s, session id:%lu.",
                      comp_graph->GetName().c_str(), session_id);
    GE_ASSERT_SUCCESS(ge_root_model_ptr->Initialize(comp_graph));
    ge_root_model_ptr->SetModelName(ge_model_ptr->GetName());
    ge_root_model_ptr->SetSubgraphInstanceNameToModel(comp_graph->GetName(), ge_model_ptr);
  }
  GE_DUMP(comp_graph, "AfterAssignResource");
  return SUCCESS;
}

Status GraphBuilder::BuildForKnownShapeGraph(ComputeGraphPtr &comp_graph,
                                             GeModelPtr &ge_model_ptr, const uint64_t session_id,
                                             const bool has_assigned_var_mem) {
  GELOGI("Begin to build known shape graph[%s].", comp_graph->GetName().c_str());
  Status ret = SecondPartition(comp_graph);
  GE_CHK_STATUS_RET(ret, "[Call][SecondPartition] for Graph[%s] failed.", comp_graph->GetName().c_str());
  auto subgraph_map = graph_partitioner_.GetSubGraphMap();

  GE_TRACE_START(BuildSubgraph);
  ge::ModelBuilder builder(session_id, comp_graph, subgraph_map, stream_max_parallel_num_, hcom_parallel_, build_mode_);
  GE_TRACE_START(PreBuildModel);
  GE_CHK_STATUS_RET(builder.PreBuildModel(), "[PreBuild][Model] failed, Graph[%s].",
                    comp_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(PreBuildModel, "GraphBuilder::PreBuildModel");

  GE_TRACE_START(CalcOpParam);
  GE_CHK_STATUS_RET(CalcOpParam(comp_graph), "[Calc][OpParam] fail, Graph[%s].", comp_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(CalcOpParam, "GraphBuilder::CalcOpParam");

  ModelPtr model_ptr = MakeShared<ge::Model>();
  if (model_ptr == nullptr) {
    return MEMALLOC_FAILED;
  }
  builder.SetHasAssignedVarMemFlag(has_assigned_var_mem);
  GE_TRACE_START(BuildModelForGetTask);
  GE_CHK_STATUS_RET(builder.BuildModelForGetTask(*model_ptr), "[Build][Model] ForGetTask fail, Graph[%s].",
                    comp_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(BuildModelForGetTask, "GraphBuilder::BuildModelForGetTask");

  GE_TRACE_START(GetTaskInfo);
  GE_ASSERT_SUCCESS(GetTaskInfo(builder, model_ptr, comp_graph, subgraph_map, session_id),
                    "[Get][FirstTaskInfo] fail, Graph[%s].", comp_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(GetTaskInfo, "GraphBuilder::GetTaskInfo");

  GE_TRACE_START(RefreshRealStream);
  GE_ASSERT_NOTNULL(graph_2_task_generator_[comp_graph]);
  GE_ASSERT_SUCCESS(builder.RefreshRealStream(graph_2_task_generator_[comp_graph]->MutableNodeId2TaskDefs()),
                    "RefreshRealStream fail, Graph[%s].", comp_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(RefreshRealStream, "GraphBuilder::RefreshRealStream");

  GE_TRACE_START(BuildModelDefForStream);
  GE_ASSERT_SUCCESS(builder.BuildModelDefForStream(*model_ptr), "[Build][Model] Part two fail, Graph[%s].");
  GE_COMPILE_TRACE_TIMESTAMP_END(BuildModelDefForStream, "GraphBuilder::BuildModelDefForStream");

  GE_TRACE_START(ReGetTaskInfo);
  GE_ASSERT_SUCCESS(ReGetTaskInfo(comp_graph, session_id, *model_ptr),
                    "ReGetTaskInfo fail, Graph[%s].", comp_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(ReGetTaskInfo, "GraphBuilder::ReGetTaskInfo");

  ge_model_ptr = MakeShared<ge::GeModel>();
  if (ge_model_ptr == nullptr) {
    return MEMALLOC_FAILED;
  }
  GE_CHK_STATUS_RET(builder.SaveDataToModel(*model_ptr, *ge_model_ptr),
                    "[Save][Data] ToModel fail, Graph[%s].", comp_graph->GetName().c_str());
  GELOGD("Success to build graph[%s] model.", comp_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(BuildSubgraph, "GraphBuilder::Build");
  return SUCCESS;
}

Status GraphBuilder::ReGetTaskInfo(const ComputeGraphPtr &comp_graph, uint64_t session_id, Model &model) {
  auto task_generator = graph_2_task_generator_[comp_graph].get();
  GE_ASSERT_NOTNULL(task_generator);
  GE_ASSERT_SUCCESS(task_generator->ReGetTaskInfo(comp_graph), "Graph %s second gen task failed",
                    comp_graph->GetName().c_str());
  GE_ASSERT_SUCCESS(task_generator->GenModelTaskDef(comp_graph, session_id, model),
                    "Graph %s gen ModelDef failed", comp_graph->GetName().c_str());
  return SUCCESS;
}

Status GraphBuilder::SetConstantInputOffset(const ComputeGraphPtr &comp_graph) const {
  const auto var_mng = VarManager::Instance(comp_graph->GetSessionID());
  for (auto &node : comp_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    auto num_inputs = op_desc->GetInputsSize();
    std::vector<int64_t> input_offsets(num_inputs, 0);
    int32_t valid_input_index = -1;
    for (uint32_t i = 0; i < node->GetAllInDataAnchorsSize(); ++i) {
      auto in_anchor = node->GetInDataAnchor(i);
      auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
      if (peer_out_anchor == nullptr) {
        continue;
      }

      ++valid_input_index;
      auto peer_node = peer_out_anchor->GetOwnerNode();
      if (peer_node == nullptr) {
        continue;
      }

      if (peer_node->GetType() != CONSTANT) {
        continue;
      }

      std::vector<GeTensorPtr> weights = OpDescUtils::MutableWeights(peer_node);
      if (weights.empty()) {
        REPORT_INNER_ERR_MSG("E19999", "check weights size of node %s(%s) is empty",
                           node->GetName().c_str(), node->GetType().c_str());
        GELOGE(FAILED, "[Check][Param] weights size of node %s is empty", node->GetName().c_str());
        return FAILED;
      }
      GeTensorPtr weight = weights[0];
      GE_CHECK_NOTNULL(weight);
      int64_t input_offset = 0;
      (void) TensorUtils::GetDataOffset(weight->MutableTensorDesc(), input_offset);
      // valid_input_index must smaller than num_inputs
      input_offsets[valid_input_index] = input_offset;
      GELOGD("[%s] input[%u] is const, offset = %ld", node->GetName().c_str(), valid_input_index, input_offset);
    }

    op_desc->SetInputOffset(input_offsets);
    if (!var_mng->IsVarExist(VarMemAssignUtil::GetNameForVarManager(op_desc))) {
      std::vector<int64_t> output_offsets(op_desc->GetOutputsSize(), 0);
      op_desc->SetOutputOffset(output_offsets);
    }
  }
  return SUCCESS;
}

Status GraphBuilder::BuildForUnknownShapeGraph(ComputeGraphPtr &comp_graph, GeModelPtr &ge_model_ptr,
                                               uint64_t session_id) {
  GELOGI("Begin to build unknown shape graph[%s].", comp_graph->GetName().c_str());
  Graph2SubGraphInfoList subgraph_map;
  if (comp_graph->GetParentGraph() == nullptr) {
    GE_ASSERT_SUCCESS(SecondPartition(comp_graph));
    subgraph_map = graph_partitioner_.GetSubGraphMap();
  }
  ge::ModelBuilder builder(session_id, comp_graph, subgraph_map, stream_max_parallel_num_, hcom_parallel_, build_mode_);
  GE_ASSERT_SUCCESS(builder.AssignStreamForDynamicShapeGraph(comp_graph), "Allocate stream for graph: %s failed.",
                    comp_graph->GetName().c_str());
  subgraph_map.clear();

  GE_TRACE_START(PreBuildModel);
  GE_CHK_STATUS_RET(builder.PreBuildModel(), "[PreBuild][Model] fail, Graph[%s].", comp_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(PreBuildModel, "GraphBuilder::PreBuildModel");

  GE_TRACE_START(CalcOpParam);
  GE_CHK_STATUS_RET(CalcOpParam(comp_graph), "[Calc][OpParam] fail, Graph[%s].", comp_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(CalcOpParam, "GraphBuilder::CalcOpParam");

  GE_TRACE_START(SetConstantInputOffset);
  GE_CHK_STATUS_RET(SetConstantInputOffset(comp_graph),
                    "[Set][Offset] Graph[%s] failed to set constant input offset.", comp_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(SetConstantInputOffset, "GraphBuilder::SetConstantInputOffset");
  GE_TRACE_START(MergeWeights);
  GE_CHK_STATUS_RET(builder.MergeWeights(), "[Merge][Weights] failed for Graph[%s].", comp_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(MergeWeights, "GraphBuilder::MergeWeights");

  ModelPtr model_ptr = MakeShared<ge::Model>();
  GE_CHK_BOOL_RET_STATUS(model_ptr != nullptr, MEMALLOC_FAILED, "[Make][Model] nullptr failed.");

  GE_TRACE_START(BuildModelForGetDynShapeTask);
  GE_CHK_STATUS_RET(builder.BuildModelForGetDynShapeTask(*model_ptr),
                    "[Build][Model] ForGetDynShapeTask fail, Graph[%s].", comp_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(BuildModelForGetDynShapeTask, "GraphBuilder::BuildModelForGetDynShapeTask");

  GE_TRACE_START(GetTaskInfo);
  GE_ASSERT_SUCCESS(GetTaskInfo(builder, model_ptr, comp_graph, subgraph_map, session_id),
                    "[Get][GetTaskInfo] fail, Graph[%s].", comp_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(GetTaskInfo, "GraphBuilder::GetTaskInfo");

  GE_TRACE_START(ReGetTaskInfo);
  GE_ASSERT_SUCCESS(ReGetTaskInfo(comp_graph, session_id, *model_ptr),
                    "[Get][ReGetTaskInfo] fail, Graph[%s].", comp_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(ReGetTaskInfo, "GraphBuilder::ReGetTaskInfo");

  ge_model_ptr = MakeShared<ge::GeModel>();
  GE_CHK_BOOL_RET_STATUS(ge_model_ptr != nullptr, MEMALLOC_FAILED, "[Make][GeModel] nullptr failed");
  GE_CHK_STATUS_RET(builder.SaveDataToModel(*model_ptr, *ge_model_ptr),
                    "[Save][Data] ToModel fail, Graph[%s].", comp_graph->GetName().c_str());
  GELOGD("Success to build graph[%s] model.", comp_graph->GetName().c_str());
  return SUCCESS;
}

Status GraphBuilder::CalcLogIdAndSetAttr(const OpDescPtr &op_desc, const std::vector<int64_t> &trace_nodes,
                                         const int64_t node_index, const int64_t start_id) const {
  for (size_t i = 0U; i < trace_nodes.size(); i++) {
    if (trace_nodes[i] == node_index) {
      GELOGD("Node of dynamic graph is %s, idx %lld", op_desc->GetName().c_str(), node_index);
      int64_t log_id = static_cast<int64_t>(i) * kProfilingArStep + start_id;
      (void)ge::AttrUtils::SetInt(op_desc, ATTR_NAME_INSERT_PROFILILNG_TASK_LOG_ID, log_id);
      break;
    }
  }

  return SUCCESS;
}

Status GraphBuilder::MarkFpBpProfilingTaskAttr(ComputeGraphPtr &com_graph) const {
  bool original_unknown_shape_flag = com_graph->GetGraphUnknownFlag();
  com_graph->SetGraphUnknownFlag(false);

  GELOGD("Start to mark profiling task attr for fp and bp.");
  TaskGenerator task_generator;
  ProfilingPoint profiling_point;
  Status ret = task_generator.FindProfilingNodeIndex(com_graph, profiling_point);
  if (ret != SUCCESS) {
    GELOGW("Find profiling node index failed.");
  }
  com_graph->SetGraphUnknownFlag(original_unknown_shape_flag);
  // mark profiling task attr for node
  for (const auto &node : com_graph->GetAllNodes()) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const auto node_index = op_desc->GetId();
    if (profiling_point.fp_index == node_index) {
      GELOGI("The first fp node of dynamic graph is %s, idx %ld", op_desc->GetName().c_str(), node_index);
      (void)ge::AttrUtils::SetBool(op_desc, ATTR_NAME_INSERT_FP_PROFILILNG_TASK, true);
    }
    if (profiling_point.bp_index == node_index) {
      GELOGI("The bp node of dynamic graph is %s, idx %ld", op_desc->GetName().c_str(), node_index);
      (void)ge::AttrUtils::SetBool(op_desc, ATTR_NAME_INSERT_BP_PROFILILNG_TASK, true);
    }
    for (size_t i = 0; i < profiling_point.all_reduce_node_index.size(); i++) {
      if (profiling_point.all_reduce_node_index[i] == node_index) {
        GELOGI("The all reduce node of dynamic graph is %s, idx %ld", op_desc->GetName().c_str(), node_index);
        (void)ge::AttrUtils::SetBool(op_desc, ATTR_NAME_INSERT_BP_PROFILILNG_TASK, true);
        GE_IF_BOOL_EXEC(TypeUtilsInner::CheckUint64MulOverflow(i, kProfilingArStep),
                        REPORT_INNER_ERR_MSG("E19999", "Multiply result is out of range when calc profiling ar log id "
                                           "for node:%s(%s)", op_desc->GetName().c_str(), op_desc->GetType().c_str());
                        GELOGE(FAILED, "[Check][Param] Multiply result is out of range, node:%s(%s)",
                               op_desc->GetName().c_str(), op_desc->GetType().c_str());
                        return FAILED);
        FMK_UINT64_ADDCHECK((i * kProfilingArStep), kProfilingArStartLogid);
        int64_t log_id = i * kProfilingArStep + kProfilingArStartLogid;
        (void)ge::AttrUtils::SetInt(op_desc, ATTR_NAME_INSERT_PROFILILNG_TASK_LOG_ID, log_id);
        break;
      }
    }
    GE_ASSERT_SUCCESS(CalcLogIdAndSetAttr(op_desc, profiling_point.get_next_node_index,
        node_index, kProfilingGetNextStartLogid));
    if (profiling_point.end_index.find(node_index) != profiling_point.end_index.end()) {
      GELOGI("The end node of dynamic graph is %s, idx %ld", op_desc->GetName().c_str(), node_index);
      (void)ge::AttrUtils::SetBool(op_desc, ATTR_NAME_INSERT_END_PROFILILNG_TASK, true);
    }
  }
  return SUCCESS;
}

/*
  只有data->op->netoutput三个算子动态shape的图，原图优化过程中op被优化调，产生了data直连netoutput的动态图
  图上打上了ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED属性。
  此类动态图也需要插入all_graph。
  该函数可以识别出此场景
  ┌───────┐          ┌───────────┐
  │ Data  │ ───────> │ NetOutput │
  └───────┘          └───────────┘
*/
bool GraphBuilder::IsDataDirectConnNetoutput(ComputeGraphPtr &comp_graph) {
  const auto nodes = comp_graph->GetDirectNode();
  return ((nodes.size() == 2U) && (nodes.at(0)->GetType() == "Data") && (nodes.at(1)->GetType() == "NetOutput"));
}

Status GraphBuilder::BuildForDynamicShapeGraph(ComputeGraphPtr &comp_graph,
                                               GeRootModelPtr &ge_root_model_ptr, GeModelPtr &ge_model_ptr,
                                               const uint64_t session_id, const bool has_assigned_var_mem) {
  GELOGI("Start to build BuildForDynamicShape for dynamic shape.");
  // Update Root Graph Data size

  for (auto &node : comp_graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (OpTypeUtils::IsDataNode(node->GetType())) {
      GE_CHK_STATUS_RET(CalcDynShapeRootGraphDataSize(op_desc), "[Calc][DynShapeRootGraphDataSize] failed, op:%s.",
                        op_desc->GetName().c_str());
    }
  }

  // Set fp bp profiling task attr for graph
  GE_ASSERT_SUCCESS(MarkFpBpProfilingTaskAttr(comp_graph));

  GE_ASSERT_SUCCESS(
      BuildForUnknownShapeAllGraphs(comp_graph, ge_root_model_ptr, ge_model_ptr, session_id, has_assigned_var_mem));

  GE_ASSERT_SUCCESS(AssignAttachedResourceForAllDynamicGraph(comp_graph, ge_root_model_ptr));

  if (!gert::GraphUnfolder::IsGraphNeedUnfold(comp_graph)) {
    GE_ASSERT_SUCCESS(RefreshInfoOfDynamicShapeGraph(comp_graph, ge_root_model_ptr));
  }

  return SUCCESS;
}

Status GraphBuilder::AssignAttachedResourceForAllDynamicGraph(const ComputeGraphPtr &comp_graph,
                                                              GeRootModelPtr &ge_root_model) const {
  const auto &ge_models = ge_root_model->GetSubgraphInstanceNameToModel();
  const auto iter_root = ge_models.find(comp_graph->GetName());
  int64_t stream_num = 1;
  int64_t event_num = 0;
  int64_t notify_num = 0;
  std::vector<uint32_t> notify_types;
  if (iter_root != ge_models.end()) {
    const auto &root_model = iter_root->second;
    GE_CHECK_NOTNULL(root_model);
    (void)AttrUtils::GetInt(root_model, ATTR_MODEL_STREAM_NUM, stream_num);
    (void)AttrUtils::GetInt(root_model, ATTR_MODEL_EVENT_NUM, event_num);
    (void)AttrUtils::GetInt(root_model, ATTR_MODEL_NOTIFY_NUM, notify_num);
    (void)AttrUtils::GetListInt(root_model, ATTR_MODEL_NOTIFY_TYPES, notify_types);
    GELOGI("Root model: %s, stream num: %ld, event num: %ld, notify_num: %ld.", iter_root->first.c_str(), stream_num,
           event_num, notify_num);

    if (stream_num == 0) {
      stream_num = 1;
    }
    const auto dynamic_stream_allocator = MakeShared<DynamicStreamAllocator>();
    GE_CHECK_NOTNULL(dynamic_stream_allocator);
    int64_t stream_num_before = stream_num;
    GE_ASSERT_SUCCESS(
        dynamic_stream_allocator->AssignAttachedResource(comp_graph, stream_num, event_num, notify_num, notify_types));
    RefreshStreamAttrs(root_model, stream_num, stream_num_before, event_num, notify_num, notify_types);
    GELOGI(
        "After assign attached resources, main stream num:[%ld], total stream_num:[%ld], event_num:[%ld], "
        "notify_num:[%ld]",
        stream_num_before, stream_num, event_num, notify_num);
  } else {
    // 当前动态shape图展开没有默认开启，为了兼容attach流的多流场景引入的临时修改，待动态shape图展开改为默认开启后删除该代码
    const auto dynamic_stream_allocator = MakeShared<DynamicStreamAllocator>();
    GE_CHECK_NOTNULL(dynamic_stream_allocator);
    int64_t stream_num_before = stream_num;
    GE_ASSERT_SUCCESS(
        dynamic_stream_allocator->AssignAttachedResource(comp_graph, stream_num, event_num, notify_num, notify_types));
    RefreshStreamAttrs(comp_graph, stream_num, stream_num_before, event_num, notify_num, notify_types);
    GELOGI(
        "After assign attached resources, main stream num:[%ld], total stream_num:[%ld], event_num:[%ld], "
        "notify_num:[%ld]",
        stream_num_before, stream_num, event_num, notify_num);
  }

  return SUCCESS;
}

Status GraphBuilder::BuildForUnknownShapeAllGraphs(ComputeGraphPtr &comp_graph,
                                                   GeRootModelPtr &ge_root_model_ptr, GeModelPtr &ge_model_ptr,
                                                   const uint64_t session_id, const bool has_assigned_var_mem) {
  auto all_graphs = comp_graph->GetAllSubgraphs();
  bool is_partitioned = false;
  // is_partitioned in singleop and helper hostcpu scenes is false
  (void) AttrUtils::GetBool(comp_graph,  ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_partitioned);
  if ((!gert::GraphUnfolder::IsGraphNeedUnfold(comp_graph)) || (!is_partitioned) ||
      IsDataDirectConnNetoutput(comp_graph)) {
    all_graphs.insert(all_graphs.cbegin(), comp_graph);
  }
  for (auto &sub_graph : all_graphs) {
    // exclude functional subgraph in known subgraph
    const auto &parent_graph = sub_graph->GetParentGraph();
    if ((parent_graph != nullptr) && (parent_graph != comp_graph) && (!parent_graph->GetGraphUnknownFlag())) {
      continue;
    }

    // exclude ffts plus subgraph in known subgraph
    const auto &functional_node = sub_graph->GetParentNode();
    if ((functional_node != nullptr) && functional_node->GetOpDesc()->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH)) {
      continue;
    }

    if (sub_graph->GetGraphUnknownFlag()) {
      // unknown shape build flow
      GE_CHK_STATUS_RET(BuildForUnknownShapeGraph(sub_graph, ge_model_ptr, session_id),
                        "[Build][Graph] as unknown shape failed, session id:%lu.", session_id);
    } else {
      // reset functional subgraph parent graph as known subgraph
      for (const auto &node : sub_graph->GetAllNodes()) {
        for (const auto &sub_graph_name : node->GetOpDesc()->GetSubgraphInstanceNames()) {
          auto sub_sub_graph = comp_graph->GetSubgraph(sub_graph_name);
          GE_CHK_STATUS_RET(sub_graph->AddSubgraph(sub_sub_graph),
                            "[Add][SubGraph] %s to known graph:%s failed.", sub_sub_graph->GetName().c_str(),
                            sub_graph->GetName().c_str());
        }
      }
      // known shape build flow
      GE_CHK_STATUS_RET(BuildForKnownShapeGraph(sub_graph, ge_model_ptr, session_id, has_assigned_var_mem),
                        "[Build][Graph] for known shape failed, session id:%lu.", session_id);
    }
    const auto &model_info = ge_root_model_ptr->GetSubgraphInstanceNameToModel();
    GE_ASSERT_TRUE(model_info.find(sub_graph->GetName()) == model_info.end());
    ge_root_model_ptr->SetSubgraphInstanceNameToModel(sub_graph->GetName(), ge_model_ptr);
  }
  ge_root_model_ptr->SetModelName(comp_graph->GetName());

  return SUCCESS;
}

Status GraphBuilder::RefreshInfoOfDynamicShapeGraph(ComputeGraphPtr &comp_graph, GeRootModelPtr &ge_root_model) const {
  const auto &ge_models = ge_root_model->GetSubgraphInstanceNameToModel();
  const auto iter_root = ge_models.find(comp_graph->GetName());
  GE_ASSERT_TRUE(iter_root != ge_models.end());
  const auto &root_model = iter_root->second;
  GE_CHECK_NOTNULL(root_model);

  uint32_t stream_num = 0U;
  uint32_t event_num = 0U;
  uint32_t notify_num = 0U;
  std::vector<uint32_t> notify_types;
  (void)AttrUtils::GetInt(root_model, ATTR_MODEL_STREAM_NUM, stream_num);
  (void)AttrUtils::GetInt(root_model, ATTR_MODEL_EVENT_NUM, event_num);
  (void)AttrUtils::GetInt(root_model, ATTR_MODEL_NOTIFY_NUM, notify_num);
  GELOGI("Root model: %s, stream num: %u, event num: %u, notify num: %u.", iter_root->first.c_str(), stream_num,
         event_num, notify_num);

  for (const auto &ge_model : ge_models) {
    if (ge_model.first == comp_graph->GetName()) {
      continue;
    }
    uint32_t tmp_stream = 0U;
    uint32_t tmp_event = 0U;
    uint32_t tmp_notify = 0U;
    (void)AttrUtils::GetInt(ge_model.second, ATTR_MODEL_STREAM_NUM, tmp_stream);
    (void)AttrUtils::GetInt(ge_model.second, ATTR_MODEL_EVENT_NUM, tmp_event);
    (void)AttrUtils::GetInt(ge_model.second, ATTR_MODEL_NOTIFY_NUM, tmp_notify);
    GELOGI("Sub model: %s, stream num: %u, event num: %u, notify num: %u.", ge_model.first.c_str(), tmp_stream,
           tmp_event, tmp_notify);

    GE_ASSERT_SUCCESS(CheckUint32AddOverflow(stream_num, tmp_stream));
    GE_ASSERT_SUCCESS(CheckUint32AddOverflow(event_num, tmp_event));
    stream_num += tmp_stream;
    event_num += tmp_event;
    notify_num += tmp_notify;
  }

  GELOGI("Total stream num: %u, event num: %u, notify num: %u.", stream_num, event_num, notify_num);
  GE_ASSERT_TRUE(AttrUtils::SetInt(root_model, ATTR_MODEL_STREAM_NUM, stream_num));
  GE_ASSERT_TRUE(AttrUtils::SetInt(root_model, ATTR_MODEL_EVENT_NUM, event_num));
  GE_ASSERT_TRUE(AttrUtils::SetInt(root_model, ATTR_MODEL_NOTIFY_NUM, notify_num));

  return SUCCESS;
}

Status GraphBuilder::GetCurrentRunContext(const ge::ModelBuilder &builder, const ModelPtr &model_ptr,
                                          ComputeGraphPtr &comp_graph, uint64_t session_id) {
  int64_t memory_size = 0;
  GE_ASSERT_TRUE(AttrUtils::GetInt(model_ptr, ATTR_MODEL_MEMORY_SIZE, memory_size),
                 "[Get][Attr] memory size fail, graph:%s, session id:%lu.", comp_graph->GetName().c_str(), session_id);

  int64_t p2p_memory_size = 0;
  GE_ASSERT_TRUE(AttrUtils::GetInt(model_ptr, ATTR_MODEL_P2P_MEMORY_SIZE, p2p_memory_size),
                 "[Get][Attr] %s fail in model", ATTR_MODEL_P2P_MEMORY_SIZE.c_str());

  int64_t weight_size = 0;
  GE_ASSERT_TRUE(AttrUtils::GetInt(model_ptr, ATTR_MODEL_WEIGHT_SIZE, weight_size),
                 "[Get][Attr] %s fail in model", ATTR_MODEL_WEIGHT_SIZE.c_str());
  int64_t host_memory_size = 0;
  (void)AttrUtils::GetInt(model_ptr, MODEL_ATTR_HOST_MEMORY_SIZE, host_memory_size);
  int64_t host_svm_size = 0;
  (void)AttrUtils::GetInt(model_ptr, MODEL_ATTR_HOST_SVM_SIZE, host_svm_size);
  // since var_mem_logic_base_ = graph_mem_max_size_ + kGraphMemoryBuffer in graph_var_manager.cc,
  const int32_t bit_move = 4;
  auto *get_mem_base = PtrToPtr<void, uint8_t>(ValueToPtr(kGraphMemoryBuffer >> bit_move));
  uint8_t *get_weight_mem_base = get_mem_base;
  if (weight_size > 0) {
    get_weight_mem_base = get_mem_base + memory_size + p2p_memory_size;
  }
  std::map<int64_t, uint8_t *> mem_type_to_data_mem_base;
  mem_type_to_data_mem_base[RT_MEMORY_HBM] = get_mem_base;
  if (p2p_memory_size == 0) {
    mem_type_to_data_mem_base[RT_MEMORY_P2P_DDR] = nullptr;
  } else {
    mem_type_to_data_mem_base[RT_MEMORY_P2P_DDR] = get_mem_base + memory_size;
  }
  mem_type_to_data_mem_base[RT_MEMORY_HOST] = PtrToPtr<void, uint8_t>(ValueToPtr(kMemoryHostFeatureMapLogicBase));
  mem_type_to_data_mem_base[RT_MEMORY_HOST_SVM] =
      PtrToPtr<void, uint8_t>(ValueToPtr(kMemoryHostSVMFeatureMapLogicBase));
  std::map<int64_t, uint64_t> mem_type_to_data_mem_size;
  mem_type_to_data_mem_size[RT_MEMORY_HBM] = memory_size;
  mem_type_to_data_mem_size[RT_MEMORY_P2P_DDR] = p2p_memory_size;
  mem_type_to_data_mem_size[RT_MEMORY_HOST] = host_memory_size;
  mem_type_to_data_mem_size[RT_MEMORY_HOST_SVM] = host_svm_size;
  RunContextUtil run_context_util;
  GE_ASSERT_SUCCESS(run_context_util.InitMemInfo(get_mem_base, memory_size, mem_type_to_data_mem_base,
                                                 mem_type_to_data_mem_size, get_weight_mem_base, weight_size));
  auto weight_buffer = builder.GetWeightBuffer();
  GE_ASSERT_SUCCESS(run_context_util.CreateRunContext(*model_ptr, comp_graph, weight_buffer, session_id),
                    "[Create][RunContext] fail, graph:%s.", comp_graph->GetName().c_str());

  graph_2_run_context_[comp_graph] = std::move(run_context_util.GetRunContext());
  return SUCCESS;
}

Status GraphBuilder::GetTaskInfo(const ge::ModelBuilder &builder, const ModelPtr &model_ptr,
                                      ComputeGraphPtr &comp_graph, Graph2SubGraphInfoList &subgraph_map,
                                      uint64_t session_id) {
  GE_CHECK_NOTNULL(model_ptr);
  GE_CHECK_NOTNULL(comp_graph);
  GE_ASSERT_SUCCESS(GetCurrentRunContext(builder, model_ptr, comp_graph, session_id));
  StreamGraphOptimizer stream_optimizer;
  auto ret = stream_optimizer.OptimizeStreamedSubGraph(comp_graph, subgraph_map, graph_2_run_context_[comp_graph]);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Optimize][StreamedSubGraph] fail, graph:%s.", comp_graph->GetName().c_str());
    return ret;
  }
  auto var_manager = VarManager::Instance(session_id);
  GE_ASSERT_NOTNULL(var_manager);
  auto *get_var_mem_base = reinterpret_cast<uint8_t *>(reinterpret_cast<uintptr_t>(var_manager->GetVarMemLogicBase()));
  uint64_t var_size = (var_manager->GetVarMemSize(RT_MEMORY_HBM) > 0) ? var_manager->GetVarMemMaxSize() : 0;
  SuperKernelPass sk_pass;
  GE_ASSERT_SUCCESS(sk_pass.Run(comp_graph));
  graph_2_task_generator_[comp_graph] =
      ge::MakeUnique<TaskGenerator>(get_var_mem_base, var_size, &graph_2_run_context_[comp_graph]);
  GE_ASSERT_NOTNULL(graph_2_task_generator_[comp_graph]);
  GE_ASSERT_SUCCESS(graph_2_task_generator_[comp_graph]->GetTaskInfo(comp_graph, session_id, *model_ptr));
  std::vector<Node *> refresh_nodes;
  GE_ASSERT_SUCCESS(ProcessAppendWs(model_ptr, comp_graph, refresh_nodes));
  if (!refresh_nodes.empty()) {
    GELOGI("start to re generate task %zu", refresh_nodes.size());
    GE_ASSERT_SUCCESS(GetCurrentRunContext(builder, model_ptr, comp_graph, session_id));
    graph_2_task_generator_[comp_graph]->GenerateTaskForNodes(refresh_nodes);
  }
  return SUCCESS;
}

Status GraphBuilder::UpdateMemAfterAppendWs(const ModelPtr &model_ptr,
                                            const std::map<int64_t, int64_t> &append_ws_stm_max,
                                            const std::map<int64_t, std::vector<NodePtr>> &append_ws_stm_nodes,
                                            int64_t &last_append_ws_size) const {
  int64_t origin_memory_size = 0;
  GE_ASSERT_TRUE(AttrUtils::GetInt(model_ptr, ATTR_MODEL_MEMORY_SIZE, origin_memory_size));
  int64_t zero_copy_size = 0;
  GE_ASSERT_TRUE(AttrUtils::GetInt(model_ptr, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, zero_copy_size));
  GE_ASSERT_TRUE(origin_memory_size >= zero_copy_size, "%ld vs %ld", origin_memory_size, zero_copy_size);
  int64_t cur_memory_size = origin_memory_size - zero_copy_size;
  for (const auto &stm_max : append_ws_stm_max) {
    const int64_t cur_stm = stm_max.first;
    const int64_t cur_max = stm_max.second;
    const auto it = append_ws_stm_nodes.find(cur_stm);
    GE_ASSERT_TRUE(it != append_ws_stm_nodes.end());
    for (auto &node : it->second) {
      auto cur_ws_vec = node->GetOpDesc()->GetWorkspace();
      auto cur_ws_size_vec = node->GetOpDesc()->GetWorkspaceBytes();
      GELOGI("current %s has ws %s, ws_size %s", node->GetNamePtr(),
             ToString(cur_ws_vec).c_str(), ToString(cur_ws_size_vec).c_str());
      std::vector<int64_t> cur_append_ws_vec;
      GE_ASSERT_TRUE(AttrUtils::GetListInt(node->GetOpDesc(), "_append_ws", cur_append_ws_vec));
      const size_t append_ws_start_id = cur_ws_vec.size();
      int64_t cur_offset = cur_memory_size;
      for (auto &append_ws : cur_append_ws_vec) {
        cur_ws_size_vec.emplace_back(append_ws);
        cur_ws_vec.emplace_back(cur_offset);
        cur_offset += MemSizeAlign(append_ws, 512U);
      }
      node->GetOpDesc()->SetWorkspace(cur_ws_vec);
      node->GetOpDesc()->SetWorkspaceBytes(cur_ws_size_vec);
      GELOGI("refresh %s ws %s, ws_size %s, append_ws_start_id %zu", node->GetNamePtr(),
             ToString(cur_ws_vec).c_str(), ToString(cur_ws_size_vec).c_str(), append_ws_start_id);
    }
    cur_memory_size += cur_max;
    last_append_ws_size += cur_max;
  }
  int64_t last_mem_size = origin_memory_size + last_append_ws_size;
  GELOGI("after append ws, origin memory size %ld, last memory size %ld", origin_memory_size, last_mem_size);
  GE_ASSERT_TRUE(AttrUtils::SetInt(model_ptr, ATTR_MODEL_MEMORY_SIZE, last_mem_size));
  std::vector<std::vector<int64_t>> sub_memory_infos;
  GE_ASSERT_TRUE(AttrUtils::GetListListInt(model_ptr, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos));
  GE_ASSERT_TRUE(!sub_memory_infos.empty());
  size_t info_num = sub_memory_infos.size();
  // last is zero copy mem info
  sub_memory_infos[info_num - 1][1] = origin_memory_size - zero_copy_size + last_append_ws_size;
  std::vector<int64_t> tmp_mem_info = sub_memory_infos.back();
  // SubMemoryInfo { mem_type, mem_offset_base, mem_size, is_fixed_addr_prior}
  GE_ASSERT_TRUE(tmp_mem_info.size() > 3U);
  tmp_mem_info[1] = origin_memory_size - zero_copy_size;
  tmp_mem_info[2] = last_append_ws_size;
  sub_memory_infos.insert(sub_memory_infos.begin(), tmp_mem_info);
  GE_ASSERT_TRUE(AttrUtils::SetListListInt(model_ptr, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos));
  return SUCCESS;
}

Status GraphBuilder::RefreshNodeOffsetAfterAppendWs(const Node *node, const int64_t origin_memory_size,
    const int64_t zero_copy_size, const int64_t last_append_ws_size, bool &need_refresh) const {
  const int64_t left = origin_memory_size - zero_copy_size;
  const int64_t right = origin_memory_size;
  bool need_ignore_type = OpTypeUtils::IsVariableNode(node->GetType()) || OpTypeUtils::IsConstNode(node->GetType());
  if (!need_ignore_type) {
    auto origin_ws_vec = node->GetOpDesc()->GetWorkspace();
    auto cur_ws_vec = node->GetOpDesc()->GetWorkspace();
    std::vector<int64_t> cur_append_ws_vec;
    (void)AttrUtils::GetListInt(node->GetOpDesc(), "_append_ws", cur_append_ws_vec);
    GE_ASSERT_TRUE(cur_ws_vec.size() >= cur_append_ws_vec.size(),
                   "%zu vs %zu", cur_ws_vec.size(), cur_append_ws_vec.size());
    // need to ignore append ws
    const size_t up_limit = cur_ws_vec.size() - cur_append_ws_vec.size();
    bool need_update_ws = false;
    for (size_t i = 0; i < up_limit; ++i) {
      if (IsInRange(cur_ws_vec[i], left, right)) {
        need_update_ws = true;
        cur_ws_vec[i] += last_append_ws_size;
      }
    }
    if (need_update_ws) {
      node->GetOpDesc()->SetWorkspace(cur_ws_vec);
      GELOGI("[IMAS]after append ws, node %s refresh ws from %s to %s", node->GetNamePtr(),
             ToString(origin_ws_vec).c_str(), ToString(cur_ws_vec).c_str());
    }
    auto origin_input_offsets = node->GetOpDesc()->GetInputOffset();
    auto cur_input_offsets = node->GetOpDesc()->GetInputOffset();
    bool need_update_input = false;
    for (size_t i = 0; i < cur_input_offsets.size(); ++i) {
      if (IsInRange(cur_input_offsets[i], left, right)) {
        need_update_input = true;
        cur_input_offsets[i] += last_append_ws_size;
      }
    }
    if (need_update_input) {
      node->GetOpDesc()->SetInputOffset(cur_input_offsets);
      GELOGI("[IMAS]after append ws, node %s refresh input offset from %s to %s", node->GetNamePtr(),
             ToString(origin_input_offsets).c_str(), ToString(cur_input_offsets).c_str());
    }

    auto origin_output_offsets = node->GetOpDesc()->GetOutputOffset();
    auto cur_output_offsets = node->GetOpDesc()->GetOutputOffset();
    bool need_update_output = false;
    for (size_t i = 0; i < cur_output_offsets.size(); ++i) {
      if (IsInRange(cur_output_offsets[i], left, right)) {
        need_update_output = true;
        cur_output_offsets[i] += last_append_ws_size;
      }
    }
    if (need_update_output) {
      node->GetOpDesc()->SetOutputOffset(cur_output_offsets);
      GELOGI("[IMAS]after append ws, node %s refresh output offset from %s to %s", node->GetNamePtr(),
             ToString(origin_output_offsets).c_str(), ToString(cur_output_offsets).c_str());
    }
    need_refresh = (need_update_ws || need_update_input || need_update_output);
  }
  return SUCCESS;
}

Status GraphBuilder::RefreshOffsetAfterAppendWs(const ComputeGraphPtr &comp_graph,
                                                const int64_t origin_memory_size, const int64_t zero_copy_size,
                                                const int64_t last_append_ws_size,
                                                std::vector<Node *> &refresh_nodes) const {
  for (auto &node : comp_graph->GetAllNodesPtr()) {
    bool need_refresh = false;
    if (node->GetType() != "SuperKernel") {
      GE_ASSERT_SUCCESS(RefreshNodeOffsetAfterAppendWs(node, origin_memory_size, zero_copy_size,
                                                       last_append_ws_size, need_refresh));
      if (need_refresh) {
        refresh_nodes.emplace_back(node);
      }
    } else {
      auto op_desc = node->GetOpDesc();
      GE_ASSERT_NOTNULL(op_desc);
      ComputeGraphPtr sub_graph = nullptr;
      sub_graph = op_desc->TryGetExtAttr("_sk_sub_graph", sub_graph);
      if (sub_graph != nullptr) {
        for (auto &sub_node : sub_graph->GetAllNodesPtr()) {
          GE_ASSERT_NOTNULL(sub_node);
          GE_ASSERT_SUCCESS(RefreshNodeOffsetAfterAppendWs(sub_node, origin_memory_size,
              zero_copy_size, last_append_ws_size, need_refresh));
        }
      }
    }
  }
  return SUCCESS;
}

Status GraphBuilder::ProcessAppendWs(const ModelPtr &model_ptr, const ComputeGraphPtr &comp_graph,
                                     std::vector<Node *> &refresh_nodes) const {
  std::map<int64_t, std::vector<NodePtr>> append_ws_stm_nodes;
  std::map<int64_t, int64_t> append_ws_stm_max;
  for (auto &node : comp_graph->GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    const auto op_desc = node->GetOpDesc();
    std::vector<int64_t> append_ws_vec;
    if (AttrUtils::GetListInt(op_desc, "_append_ws", append_ws_vec)) {
      GE_ASSERT_TRUE(!append_ws_vec.empty());
      append_ws_stm_nodes[op_desc->GetStreamId()].emplace_back(node);
      int64_t append_ws_size = 0;
      for (auto &ele : append_ws_vec) {
        GE_ASSERT_TRUE((ele > 0), "find %s invalid val %ld in _append_ws", node->GetNamePtr(), ele);
        const size_t align_ele = MemSizeAlign(ele, 512);
        append_ws_size += align_ele;
      }
      GELOGI("find %s %ld has _append_ws %s, size is %ld", node->GetNamePtr(), op_desc->GetStreamId(),
             ToString(append_ws_vec).c_str(), append_ws_size);
      auto it = append_ws_stm_max.find(op_desc->GetStreamId());
      if ((it == append_ws_stm_max.end()) || (it->second < append_ws_size)) {
        append_ws_stm_max[op_desc->GetStreamId()] = append_ws_size;
        GELOGI("refresh %ld max size %ld", op_desc->GetStreamId(), append_ws_size);
      }
    }
  }
  if (append_ws_stm_nodes.empty()) {
    return SUCCESS;
  }
  int64_t origin_memory_size = 0;
  GE_ASSERT_TRUE(AttrUtils::GetInt(model_ptr, ATTR_MODEL_MEMORY_SIZE, origin_memory_size));
  int64_t zero_copy_size = 0;
  GE_ASSERT_TRUE(AttrUtils::GetInt(model_ptr, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, zero_copy_size));
  GE_ASSERT_TRUE(origin_memory_size >= zero_copy_size, "%ld vs %ld", origin_memory_size, zero_copy_size);
  int64_t last_append_ws_size = 0;
  GE_ASSERT_SUCCESS(UpdateMemAfterAppendWs(model_ptr, append_ws_stm_max, append_ws_stm_nodes, last_append_ws_size));
  GE_ASSERT_SUCCESS(RefreshOffsetAfterAppendWs(comp_graph, origin_memory_size, zero_copy_size,
                                               last_append_ws_size, refresh_nodes));
  return SUCCESS;
}

Status GraphBuilder::SetInputSize(const ge::NodePtr &node_ptr) {
  // Set the size of input_desc to 'src_node.output_desc.size'
  if (OpTypeUtils::IsDataNode(node_ptr->GetType())) {
    bool is_unknown_shape = false;
    GE_CHK_STATUS_RET(ge::NodeUtils::GetNodeUnknownShapeStatus(*node_ptr, is_unknown_shape),
                      "[Get][Status] of data node[%s] shape failed!", node_ptr->GetName().c_str());
    if (is_unknown_shape) {
      GELOGD("data node: %s is unknown shape, do not set input size!", node_ptr->GetName().c_str());
      return SUCCESS;
    }
    if (UpdateDataInputSize(node_ptr) != SUCCESS) {
      GELOGE(FAILED, "[Update][Data] input size failed, node:%s.", node_ptr->GetName().c_str());
      return FAILED;
    }
  }

  for (const auto &in_data_anchor : node_ptr->GetAllInDataAnchors()) {
    const auto &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    const auto &src_node = peer_out_anchor->GetOwnerNodeBarePtr();
    const auto &src_op = src_node->GetOpDesc();
    GE_IF_BOOL_EXEC(src_op == nullptr, continue);
    auto node_op_desc = node_ptr->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);
    // Set the input_desc of dst_node to 'src_node.output_desc'
    auto output_desc = src_op->MutableOutputDesc(peer_out_anchor->GetIdx());
    GE_ASSERT_NOTNULL(output_desc);
    int64_t size = 0;
    GE_IF_BOOL_EXEC(ge::TensorUtils::GetSize(*output_desc, size) != SUCCESS, GELOGI("Tensor has no size!"));
    GELOGD("src node %s output desc, dim_size: %zu, mem_size: %ld, format: %s, type: %s.", src_node->GetName().c_str(),
           output_desc->GetShape().GetDimNum(), size, TypeUtils::FormatToSerialString(output_desc->GetFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(output_desc->GetDataType()).c_str());
    for (size_t i = 0; i < output_desc->GetShape().GetDimNum(); ++i) {
      GELOGD("dims[%zu]: %ld", i, output_desc->GetShape().GetDim(i));
    }

    auto input_desc = node_op_desc->MutableInputDesc(in_data_anchor->GetIdx());
    GE_CHECK_NOTNULL(input_desc);
    (void) ge::TensorUtils::SetSize(*input_desc, size);
    GELOGD("%s input desc, dim_size: %zu, mem_size: %ld, format: %s, type: %s.", node_ptr->GetName().c_str(),
           input_desc->GetShape().GetDimNum(), size, TypeUtils::FormatToSerialString(input_desc->GetFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(input_desc->GetDataType()).c_str());
    // inherit some attr
    // node->netoutput; update netoutput's input
    int64_t tensor_size_attr;
    if (AttrUtils::GetInt(output_desc, ATTR_NAME_SPECIAL_OUTPUT_SIZE, tensor_size_attr) && (tensor_size_attr > 0)) {
      GE_IF_BOOL_EXEC(!AttrUtils::SetInt(*input_desc, ATTR_NAME_SPECIAL_OUTPUT_SIZE, tensor_size_attr),
                      GELOGW("Set size attr failed!"); continue);
      GELOGD("node[%s] [%d]th output has special size[%ld], and update to node[%s] [%d]th input",
             src_op->GetName().c_str(), peer_out_anchor->GetIdx(), tensor_size_attr,
             node_op_desc->GetName().c_str(), in_data_anchor->GetIdx());
    }
    // data->node; update data's output
    if (AttrUtils::GetInt(input_desc, ATTR_NAME_SPECIAL_INPUT_SIZE, tensor_size_attr) && (tensor_size_attr > 0)) {
      GE_IF_BOOL_EXEC(!AttrUtils::SetInt(*output_desc, ATTR_NAME_SPECIAL_INPUT_SIZE, tensor_size_attr),
                      GELOGW("Set size attr failed!"); continue);
      GELOGD("node[%s] [%d]th output update special size[%ld] according to node[%s] [%d]th input",
             src_op->GetName().c_str(), peer_out_anchor->GetIdx(), tensor_size_attr,
             node_op_desc->GetName().c_str(), in_data_anchor->GetIdx());
    }
  }

  return SUCCESS;
}

Status GraphBuilder::UpdateDataInputSize(const ge::NodePtr &node_ptr) const {
  const auto &op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "check op_desc is nullptr");
    GELOGE(FAILED, "[Check][Param] Op desc is nullptr.");
    return FAILED;
  }
  // data op only has one output anchor
  const auto &output_desc = op_desc->GetOutputDesc(0U);
  int64_t output_size = 0;
  if (ge::TensorUtils::GetSize(output_desc, output_size) != SUCCESS) {
    GELOGW("Get size failed!");
  }

  if (output_size > 0) {
    GELOGI("No need to update data input size.");
    return SUCCESS;
  } else {
    int64_t real_dim_size = 0;
    ge::graphStatus graph_status = TensorUtils::GetTensorSizeInBytes(output_desc, real_dim_size);
    if (graph_status != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Get tensor size in bytes failed for op:%s(%s) index:0",
                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Get][TensorSize] in bytes failed, op:%s.", op_desc->GetName().c_str());
      return FAILED;
    }
    // data op only has one input anchor
    const auto &input_desc = op_desc->MutableInputDesc(0U);
    GE_CHECK_NOTNULL(input_desc);
    ge::TensorUtils::SetSize(*input_desc, real_dim_size);
  }
  return SUCCESS;
}

Status GraphBuilder::CalcDynShapeRootGraphDataSize(const ge::OpDescPtr &op_desc) const {
  GELOGI("Begin to calc dynamic shape graph data[%s] size.", op_desc->GetName().c_str());
  // data op only has one output anchor
  ge::GeTensorDesc output_desc = op_desc->GetOutputDesc(0);
  if (output_desc.MutableShape().IsUnknownShape()) {
    GELOGI("No need to update dynamic shape graph data output size for unknown shape data.");
    return SUCCESS;
  }

  int64_t output_size = 0;
  if (ge::TensorUtils::GetSize(output_desc, output_size) != SUCCESS) {
    GELOGW("Get size failed!");
  }

  if (output_size > 0) {
    GELOGI("No need to update dynamic shape graph data output size[%ld].", output_size);
    return SUCCESS;
  } else {
    int64_t real_dim_size = 0;
    ge::graphStatus graph_status = TensorUtils::GetTensorSizeInBytes(output_desc, real_dim_size);
    if (graph_status != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Get tensor size in bytes failed for op:%s(%s) index:0 ",
                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Get][TensorSize] in bytes failed, op:%s.", op_desc->GetName().c_str());
      return FAILED;
    }

    ge::TensorUtils::SetSize(output_desc, real_dim_size);
    GELOGI("Update dynamic shape graph data output size to [%ld].", real_dim_size);
    if (op_desc->UpdateOutputDesc(0, output_desc) != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Update output desc size failed for op:%s(%s) index:0 ",
                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Update][OutputDesc] for dynamic shape graph data failed, op:%s.", op_desc->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status GraphBuilder::SecondPartition(const ge::ComputeGraphPtr &comp_graph) {
  GE_TRACE_START(GraphPartition2);
  // 二拆是流分配的入口，需要在此之前，将user stream label转储
  GE_ASSERT_SUCCESS(StreamUtils::TransUserStreamLabel(comp_graph));
  auto ret = graph_partitioner_.Partition(comp_graph, EnginePartitioner::Mode::kSecondPartitioning);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][Partition] for Graph Failed");
    return ret;
  }
  const auto &graph_2_subgraphlist = graph_partitioner_.GetSubGraphMap();
  if (graph_2_subgraphlist.find(comp_graph) == graph_2_subgraphlist.end()) {
    REPORT_INNER_ERR_MSG("E19999", "find subgraphlis in graph:%s failed", comp_graph->GetName().c_str());
    GELOGE(FAILED, "[Check][Param] Find subgraph graph:%s failed.", comp_graph->GetName().c_str());
    return FAILED;
  }
  GE_COMPILE_TRACE_TIMESTAMP_END(GraphPartition2, "EnginePartitioner::Partition2");
  return ret;
}

Status GraphBuilder::AddOutputMemTypeForNode(const NodePtr &node) const {
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  uint32_t mem_type;
  if (!AttrUtils::GetInt(op_desc, ATTR_INPUT_MEMORY_TYPE, mem_type)) {
    return SUCCESS;
  }
  GELOGD("[%s] has attr input_memory_type %u", op_desc->GetName().c_str(), mem_type);
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    const auto &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    bool valid_flag = false;
    auto src_node = peer_out_anchor->GetOwnerNode();
    auto src_out_anchor = peer_out_anchor;
    while (true) {
      const auto &src_desc = src_node->GetOpDesc();
      GE_IF_BOOL_EXEC(src_desc == nullptr, continue);
      GELOGD("[%s:%u] set attr output_memory_type %d", src_desc->GetName().c_str(), src_out_anchor->GetIdx(),
             mem_type);
      const bool ret =
          AttrUtils::SetInt(src_desc->MutableOutputDesc(src_out_anchor->GetIdx()), ATTR_OUTPUT_MEMORY_TYPE, mem_type);
      GE_ASSERT_TRUE(ret, "Set Attr:%s for node:%s(%s) out_index:%d failed",
                     ATTR_OUTPUT_MEMORY_TYPE.c_str(), src_desc->GetName().c_str(), src_desc->GetType().c_str(),
                     src_out_anchor->GetIdx());
      switch (TransferNodeType(src_node)) {
        case NodeType::kSubgraphNode:
          GE_CHK_STATUS_RET(HandleSubgraphNode(src_node, src_out_anchor),
                            "[Handle][Node] %s in subgraph failed", src_node->GetName().c_str());
          break;
        case NodeType::kSubgraphData:
          GE_CHK_STATUS_RET(HandleSubgraphDataNode(src_node, src_out_anchor),
                            "[Handle][DataNode] %s in subgraph failed", src_node->GetName().c_str());
          break;
        case NodeType::kOthers:
        default:
          valid_flag = true;
          break;
      }
      if (valid_flag) {
        break;
      }
    }
  }

  return SUCCESS;
}

Status GraphBuilder::BuildForEvaluate(ComputeGraphPtr &compute_graph, ModelDataInfo &model) {
  GELOGI("Begin to build for evaluate graph:%s.", compute_graph->GetName().c_str());
  GE_TRACE_START(GraphPartition);
  Status ret = graph_partitioner_.Partition(compute_graph, EnginePartitioner::Mode::kAtomicEnginePartitioning);
  GE_CHK_STATUS_RET(ret, "[Call][Partition] for Graph[%s] Failed", compute_graph->GetName().c_str());

  ge::ComputeGraphPtr merged_compute_graph;
  ret = graph_partitioner_.MergeAfterSubGraphOptimization(merged_compute_graph, compute_graph,
                                                          EnginePartitioner::Mode::kAtomicEnginePartitioning);
  GE_CHK_STATUS_RET(ret, "[Call][MergeAfterSubGraphOptimization] for Graph[%s] Failed",
                    merged_compute_graph->GetName().c_str());

  merged_compute_graph->SetSessionID(compute_graph->GetSessionID());
  merged_compute_graph->SetGraphID(compute_graph->GetGraphID());
  GE_COMPILE_TRACE_TIMESTAMP_END(GraphPartition, "EnginePartitioner::Partition");

  ret = SecondPartition(merged_compute_graph);
  GE_CHK_STATUS_RET(ret, "[Call][SecondPartition] for Graph[%s] failed.", merged_compute_graph->GetName().c_str());
  auto subgraph_map = graph_partitioner_.GetSubGraphMap();

  // when Constant's memory is large, can be converted to Const,
  // because Const's memory can be released when model is unload
  ConvertConstOrConstantByLifecycle(merged_compute_graph);

  GE_TRACE_START(BuildSubgraph);
  ge::ModelBuilder builder(merged_compute_graph->GetSessionID(), merged_compute_graph, subgraph_map,
                           stream_max_parallel_num_, hcom_parallel_);
  GE_DUMP(merged_compute_graph, "BeforePreBuildModel");
  GE_TRACE_START(PreBuildModel);
  GE_CHK_STATUS_RET(builder.PreBuildModel(), "[PreBuild][Model] failed, Graph[%s].",
                    merged_compute_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(PreBuildModel, "GraphBuilder::PreBuildModel");

  GE_DUMP(merged_compute_graph, "AfterPreBuildModel");
  GE_TRACE_START(CalcOpParam);
  GE_CHK_STATUS_RET(CalcOpParam(merged_compute_graph), "[Calc][OpParam] fail, Graph[%s].",
                    merged_compute_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(CalcOpParam, "GraphBuilder::CalcOpParam");
  GE_DUMP(merged_compute_graph, "AfterCalcOpParam");

  ret = builder.BuildModelForEvaluate(model);
  GE_COMPILE_TRACE_TIMESTAMP_END(BuildSubgraph, "GraphBuilder::BuildForEvaluate");
  return ret;
}
}  // namespace ge
