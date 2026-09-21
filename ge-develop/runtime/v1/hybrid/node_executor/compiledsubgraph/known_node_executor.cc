/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/node_executor/compiledsubgraph/known_node_executor.h"
#include "aicpu_engine_struct.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/fmk_error_codes.h"
#include "common/dump/dump_manager.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/load/model_manager/model_manager.h"
#include "hybrid/executor/hybrid_execution_context.h"

namespace ge {
namespace hybrid {
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::COMPILED_SUBGRAPH, KnownNodeExecutor);

Status KnownNodeTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(),
                         "[KnownNodeTaskExecuteAsync] Start");
  GELOGD("[%s] KnownNodeTask::ExecuteAsync in, model id: %u.", context.GetNodeName(), davinci_model_->Id());
  if (davinci_model_->GetTaskList().empty()) {
    GELOGW("KnownNodeExecutor::ExecuteAsync davinci model has no taskinfo.");

    // todo if data is connected to netoutput, forward address ? copy data?
    if (context.NumInputs() == context.NumOutputs()) {
      for (int32_t i = 0; i < context.NumInputs(); ++i) {
        const auto tensor = context.MutableInput(i);
        GE_CHECK_NOTNULL(tensor);
        GE_CHK_STATUS_RET(context.SetOutput(i, *tensor),
                          "[%s] Failed to set output[%d]",
                          context.GetNodeName(),
                          i);
      }
    }

    GE_CHK_STATUS_RET_NOLOG(context.RegisterCallback(done_callback));
    return SUCCESS;
  }

  rtError_t rt_ret;
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodertModelExecute] Start");
  rt_ret = rtModelExecute(davinci_model_->GetRtModelHandle(), context.GetStream(), 0U);
  GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                  REPORT_INNER_ERR_MSG("E19999", "rtModelExecute error, ret:%d", rt_ret);
                  GELOGE(FAILED, "[Invoke][rtModelExecute] error, ret:%d", rt_ret);
                  return FAILED;
                 );
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodertModelExecute] End");

  GE_CHK_STATUS_RET_NOLOG(context.RegisterCallback(done_callback));
  GELOGD("[%s] KnownNodeTask::ExecuteAsync success, model id: %u.", context.GetNodeName(), davinci_model_->Id());
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeTaskExecuteAsync] End");
  return SUCCESS;
}

Status KnownNodeTask::UpdateArgs(TaskContext &context) {
  GELOGD("[%s] KnownNodeExecutor::UpdateArgs in.", context.GetNodeName());
  if (davinci_model_->GetTaskList().empty()) {
    GELOGW("KnownNodeExecutor::UpdateArgs davinci model has no taskinfo.");
    return SUCCESS;
  }

  std::vector<void *> inputs;
  for (int32_t i = 0; i < context.NumInputs(); ++i) {
    TensorValue *const tv = context.MutableInput(i);
    GE_CHECK_NOTNULL(tv);
    inputs.emplace_back(tv->MutableData());
  }

  std::vector<void *> outputs;
  for (int32_t i = 0; i < context.NumOutputs(); ++i) {
    TensorValue *const tv = context.MutableOutput(i);
    GE_CHECK_NOTNULL(tv);
    outputs.emplace_back(tv->MutableData());
  }

  auto v_inputs = VPtrToValue(inputs);
  auto v_outputs = VPtrToValue(outputs);
  GE_CHK_STATUS_RET(davinci_model_->InitModelStream(context.GetStream()), "[Init][ModelStream] failed");
  GE_CHK_STATUS_RET(davinci_model_->UpdateKnownNodeArgs(v_inputs, v_outputs),
                    "[Update][KnownNodeArgs] failed for %s(%s).",
                    context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
  GELOGD("[%s] KnownNodeExecutor::UpdateArgs success, task_size = %zu", context.GetNodeName(),
         davinci_model_->GetTaskList().size());
  return SUCCESS;
}

Status KnownNodeTask::Init(TaskContext &context) {
  // allocate output mem
  GE_CHK_STATUS_RET(context.AllocateOutputs(), "[Allocate][Outputs] failed for %s(%s).",
                    context.GetNodeName(), context.GetNodeItem().NodeType().c_str());

  int64_t total_useful_size = 0;
  GE_CHK_STATUS_RET(davinci_model_->GetTotalMemSizeExcludeZeroCopy(total_useful_size),
                    "[Get][TotalMemSizeExcludeZeroCopy] failed, node:%s(%s).",
                    context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
  if (total_useful_size != 0) {
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(),
                           "[KnownNodeTask_AllocateWorkspace] Start");
    // allocate mem base
    std::vector<uint8_t *> hbm_fm_mem_bases;
    for (const auto &mem_info : davinci_model_->GetFmMemoryInfos()) {
      void *buffer = nullptr;
      void *const ori_mem_base = PtrToPtr<uint8_t, void>(mem_info.memory_base);
      GE_CHK_STATUS_RET(context.AllocateWorkspace(static_cast<size_t>(mem_info.memory_size), buffer, ori_mem_base),
                        "[Allocate][Workspace] failed for %s(%s).", context.GetNodeName(),
                        context.GetNodeItem().NodeType().c_str());
      RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(),
                             "[KnownNodeTask_AllocateWorkspace] End, size %ld", mem_info.memory_size);
      hbm_fm_mem_bases.push_back(PtrToPtr<void, uint8_t>(buffer));
    }
    // update mem base
    davinci_model_->UpdateHbmFmMemBases(hbm_fm_mem_bases);
    GELOGI("KnownNodeTask::Init mem base is %lx, size %ld.",
           davinci_model_->GetRuntimeParam().mem_base, total_useful_size);
  }
  if (!davinci_model_->GetRuntimeParam().memory_infos.empty()) {
    GE_CHK_STATUS_RET(davinci_model_->MallocExMem(), "[Malloc][ExMem] failed, node:%s(%s).", context.GetNodeName(),
                      context.GetNodeItem().NodeType().c_str());
  }
  if (!load_flag_) {
    const auto execution_context = context.GetExecutionContext();
    GE_CHECK_NOTNULL(execution_context);
    auto &davinci_model = execution_context->davinci_model;
    davinci_model.emplace_back(davinci_model_);
    load_flag_ = true;
  }

  GELOGI("[%s] KnownNodeExecutor::Init success.", context.GetNodeName());
  return SUCCESS;
}

Status KnownNodeTask::InitDavinciModel(const HybridModel &model, const TensorBuffer *const weight_buffer) {
  GELOGD("[Init][DavinciModel] start");
  int32_t device_id = 0;
  GE_CHK_RT_RET(aclrtGetDevice(&device_id));
  davinci_model_->SetDeviceId(static_cast<uint32_t>(device_id));
  davinci_model_->InitRuntimeParams();
  GE_CHK_STATUS_RET(davinci_model_->InitVariableMem(), "[Init][VariableMem] failed");
  GELOGD("session id from model is %lu while %lu is in use", model.GetSessionId(), kInferSessionId);
  const auto dump_properties = DumpManager::GetInstance().GetDumpProperties(kInferSessionId);
  if (dump_properties.IsDumpOpen() || dump_properties.IsOpDebugOpen()) {
    davinci_model_->SetDumpProperties(dump_properties);
  }

  void *weight = nullptr;
  size_t weight_size = 0U;
  if (weight_buffer != nullptr) {
    weight = weight_buffer->GetData();
    weight_size = weight_buffer->GetSize();
  }
  GELOGD("Start to init davinci model, weight size = %zu", weight_size);
  GE_CHK_STATUS_RET(DoInitDavinciModel(PtrToValue(weight), weight_size),
                    "[Init][DavinciModel] Failed to init davinci model.");
  GELOGD("[Init][DavinciModel] success");
  return SUCCESS;
}

Status KnownNodeTask::DoInitDavinciModel(const uintptr_t weight, const size_t weight_size) {
  const ModelParam param {0, 0U, 0U, weight, weight_size};
  return davinci_model_->Init(param);
}

Status KnownNodeTask::ReportProfilingData() {
  return davinci_model_->ReportProfilingData();
}

Status KnownNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  GELOGD("[%s] KnownNodeExecutor::PrepareTask in.", context.GetNodeName());
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorPrepareTask] Start");
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorTaskInit] Start");
  GE_CHK_STATUS_RET(task.Init(context), "[Invoke][Init] %s(%s) known node init davinci model failed.",
                    context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorTaskInit] End");

  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorUpdateArgs] Start");
  GE_CHK_STATUS_RET(task.UpdateArgs(context), "[Invoke][UpdateArgs] %s(%s) known node task update args failed.",
                    context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorUpdateArgs] End");
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorPrepareTask] End");
  GELOGD("[%s] KnownNodeExecutor::PrepareTask success.", context.GetNodeName());
  return SUCCESS;
}

Status KnownNodeExecutor::SetDaviciModel(const HybridModel &model, const NodePtr &node,
                                         std::shared_ptr<DavinciModel> &davinci_model) const {
  // set known node flag as true
  davinci_model->SetKnownNode(true);
  davinci_model->SetId(model.GetModelId());
  davinci_model->SetDumpModelName(model.GetModelName());
  davinci_model->SetOmName(model.GetOmName());
  davinci_model->SetRootGraphId(model.GetRootGraph()->GetGraphID());
  void *const global_step = model.GetGlobalStep();
  GE_CHECK_NOTNULL(global_step);
  davinci_model->SetGlobalStep(PtrToValue(global_step), sizeof(int64_t));
  // set model id as root node's node id
  davinci_model->SetSubModelId(static_cast<uint32_t>(node->GetOpDesc()->GetId()));
  return SUCCESS;
}

Status KnownNodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node,
                                   shared_ptr<NodeTask> &task) const {
  GELOGI("[%s] KnownNodeExecutor::LoadTask in.", node->GetName().c_str());
  GE_CHECK_NOTNULL(node);

  GeModelPtr ge_model;
  ComputeGraphPtr compute_graph;
  GE_CHK_STATUS_RET(GetModelAndGraph(model, node, ge_model, compute_graph),
                    "[Get][ModelAndGraph][%s(%s)] Failed to get model and graph",
                    node->GetName().c_str(), node->GetType().c_str());
  const auto node_item = model.MutableNodeItem(node);
  GE_CHECK_NOTNULL(node_item);
  GE_CHK_STATUS_RET_NOLOG(ParseAttrForAllocatingOutputs(*node_item, *compute_graph));
  const auto weight_buffer = model.GetModelWeight(compute_graph->GetName());
  std::shared_ptr<DavinciModel> davinci_model = MakeShared<DavinciModel>(0, nullptr);
  GE_CHECK_NOTNULL(davinci_model);

  if (model.GetOverflowAddr() != nullptr) {
    davinci_model->SetOverflowAddr(model.GetOverflowAddr());
  }

  GE_CHK_STATUS_RET_NOLOG(SetDaviciModel(model, node, davinci_model));
  GELOGD("KnownNodeExecutor::LoadTask node id %ld.", node->GetOpDesc()->GetId());

  davinci_model->Assign(ge_model);
  auto known_node_task = MakeShared<KnownNodeTask>(davinci_model);
  GE_CHECK_NOTNULL(known_node_task);
  GE_CHK_STATUS_RET_NOLOG(known_node_task->InitDavinciModel(model, weight_buffer));
  GELOGI("[%s] KnownNodeExecutor::LoadTask success.", node->GetName().c_str());
  task = std::move(known_node_task);
  return SUCCESS;
}

Status KnownNodeExecutor::ExecuteTask(NodeTask &task, TaskContext &context,
                                      const std::function<void()> &callback) const {
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorExecuteTask] Start");
  GE_CHK_STATUS_RET(task.ExecuteAsync(context, callback), "[Invoke][ExecuteAsync]Failed to execute task. node:%s(%s)",
                    context.GetNodeItem().NodeName().c_str(), context.GetNodeItem().NodeType().c_str());
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorExecuteTask] End");
  return SUCCESS;
}

Status KnownNodeExecutor::ParseAttrForAllocatingOutputs(NodeItem &node_item, const ComputeGraph &graph) {
  GELOGD("[%s] Start to parse attributes for outputs", node_item.NodeName().c_str());
  const auto &net_output_node = graph.FindFirstNodeMatchType(NETOUTPUT);
  if (net_output_node == nullptr) {
    GELOGD("[%s] Subgraph do not got net output", graph.GetName().c_str());
    return SUCCESS;
  }

  const auto &net_output_desc = net_output_node->GetOpDesc();
  GE_CHECK_NOTNULL(net_output_desc);
  std::map<std::string, int32_t> connected_inputs;
  std::map<NodePtr, int32_t> data_indices;
  GE_CHK_STATUS_RET(GetDataNodes(graph, data_indices), "[Get][DataNodes] failed, node:%s(%s)",
                    node_item.NodeName().c_str(), node_item.NodeType().c_str());
  for (const auto &in_data_anchor : net_output_node->GetAllInDataAnchors()) {
    const auto &out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    if (out_data_anchor == nullptr) {
      continue;
    }
    auto src_node = out_data_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);
    const auto &op_desc = src_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const auto src_op_type = src_node->GetType();
    auto output_index = in_data_anchor->GetIdx();
    GELOGD("Node %s, output %d, src node = %s, src node type = %s",
           node_item.NodeName().c_str(),
           output_index,
           src_node->GetName().c_str(),
           src_op_type.c_str());
    // parse reuse outputs
    std::string input_key = std::to_string(op_desc->GetId()) + "_" + std::to_string(out_data_anchor->GetIdx());
    const std::map<std::string, int32_t>::const_iterator it = connected_inputs.find(input_key);
    if (it == connected_inputs.cend()) {
      (void)connected_inputs.emplace(input_key, output_index);
    } else {
      GELOGD("[%s] output [%d] reuse output [%d] input node = %s, idx = %d.", node_item.NodeName().c_str(),
             output_index,
             it->second,
             src_node->GetName().c_str(),
             out_data_anchor->GetIdx());
      (void)node_item.reuse_outputs.emplace(output_index, it->second);
    }

    const std::set<std::string> constant_like_task_ops = {CONSTANT, CONSTANTOP, VARIABLE, FILECONSTANT};
    if (src_op_type == DATA) {
      int32_t data_index = data_indices[src_node];
      (void)node_item.reuse_inputs.emplace(output_index, data_index);
      GELOGD("[%s] output[%u] reuses input[%d]", node_item.NodeName().c_str(), output_index, data_index);
    }
    if (constant_like_task_ops.count(src_op_type) > 0U) {
      (void)node_item.ref_outputs.emplace(output_index, src_node);
      GELOGD("[%s] output[%d] ref to node [%s]",
             node_item.NodeName().c_str(),
             output_index,
             src_node->GetName().c_str());
    }
  }

  GELOGD("[%s] Done parsing attributes for outputs successfully", node_item.NodeName().c_str());
  return SUCCESS;
}

Status KnownNodeExecutor::GetDataNodes(const ComputeGraph &graph, std::map<NodePtr, int32_t> &data_indices) {
  std::map<int32_t, NodePtr> ordered_data_nodes;
  for (const auto &node : graph.GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetType() == DATA) {
      int32_t index = -1;
      (void)AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_INDEX, index);
      (void)ordered_data_nodes.emplace(index, node);
    }
  }

  // reindex
  int32_t data_index = 0;
  for (const auto &it : ordered_data_nodes) {
    (void)data_indices.emplace(it.second, data_index++);
  }

  return SUCCESS;
}

Status KnownNodeExecutor::GetModelAndGraph(const HybridModel &model, const NodePtr &node,
                                           GeModelPtr &ge_model,
                                           ComputeGraphPtr &graph) {
  ge_model = model.GetGeModel(node);
  GE_CHECK_NOTNULL(ge_model);
  const auto &root_graph = model.GetRootGraph();
  GE_CHECK_NOTNULL(root_graph);
  const auto compute_graph = ge_model->GetGraph();
  GE_CHECK_NOTNULL(compute_graph, ", [Get][Name] of subgraph failed, node:%s(%s)",
                   node->GetName().c_str(), node->GetType().c_str());
  graph = root_graph->GetSubgraph(compute_graph->GetName());
  GE_CHECK_NOTNULL(graph);
  return SUCCESS;
}

Status KnownNodeExecutor::ReportProfilingData(const NodeItem &node_item) const {
  return node_item.kernel_task->ReportProfilingData();
}
}  // namespace hybrid
}  // namespace ge
