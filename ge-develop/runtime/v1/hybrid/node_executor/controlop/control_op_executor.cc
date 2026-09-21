/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/node_executor/controlop/control_op_executor.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "common/plugin/ge_make_unique_util.h"
#include "hybrid/executor/hybrid_execution_context.h"
namespace {
template <class T>
ge::Status CopyFloatValueToHost(const ge::hybrid::TensorValue &tensor, bool &value) {
  T scalar{};
  GE_CHK_STATUS_RET(tensor.CopyScalarValueToHost(scalar));
  value = (fabs(static_cast<double_t>(scalar)) > std::numeric_limits<T>::epsilon());
  return ge::SUCCESS;
}

template <class T>
ge::Status CopyIntegerValueToHost(const ge::hybrid::TensorValue &tensor, bool &value) {
  T scalar{};
  GE_CHK_STATUS_RET(tensor.CopyScalarValueToHost(scalar));
  value = (static_cast<int64_t>(scalar) != 0);
  return ge::SUCCESS;
}

ge::Status CopyBoolValueToHost(const ge::hybrid::TensorValue &tensor, bool &value) {
  GE_CHK_STATUS_RET(tensor.CopyScalarValueToHost(value));
  return ge::SUCCESS;
}
}

namespace ge {
namespace hybrid {
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::CONTROL_OP, ControlOpNodeExecutor);

Status ExecutorCache::ExecuteSubgraph(const TaskContext &task_context, const std::function<void()> &done_callback) {
  GE_CHECK_NOTNULL(graph_item_);
  GELOGD("[%s] Start to execute subgraph.", graph_item_->GetName().c_str());
  if (graph_executor_ == nullptr) {
    auto execution_context = task_context.GetExecutionContext();
    graph_executor_ = MakeUnique<SubgraphExecutor>(graph_item_, execution_context);
    GE_CHECK_NOTNULL(graph_executor_);
    GE_CHK_STATUS_RET(graph_executor_->Init(), "[Init][SubgraphExecutor]Failed.");
  } else {
     graph_executor_->Reset();
  }

  GE_CHK_STATUS_RET(graph_executor_->ExecuteAsync(task_context),
                    "[Invoke][ExecuteAsync] Failed to execute subgraph.");

  GE_CHK_STATUS_RET_NOLOG(task_context.RegisterCallback(done_callback));
  GELOGD("[%s] Done executing subgraph successfully.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status ExecutorCache::ExecuteSubgraph(const TaskContext &task_context, std::vector<TensorValue> &outputs,
                                      std::vector<ConstGeTensorDescPtr> &output_desc_list) {
  std::vector<TensorValue> inputs;
  std::vector<ConstGeTensorDescPtr> input_desc;
  for (int32_t i = 0; i < task_context.NumInputs(); ++i) {
    const auto input_tensor = task_context.GetInput(i);
    GE_CHECK_NOTNULL(input_tensor);
    inputs.emplace_back(*input_tensor);
    input_desc.emplace_back(task_context.GetInputDesc(i));
  }

  if (graph_executor_ == nullptr) {
    auto execution_context = task_context.GetExecutionContext();
    graph_executor_ = MakeUnique<SubgraphExecutor>(graph_item_, execution_context);
    GE_CHECK_NOTNULL(graph_executor_);
    GE_CHK_STATUS_RET(graph_executor_->Init(), "[Init][SubgraphExecutor]Failed.");
  } else {
    graph_executor_->Reset();
  }

  GELOGD("[%s] Start to execute cond-subgraph.", task_context.GetNodeName());
  GE_CHK_STATUS_RET(graph_executor_->ExecuteAsync(inputs, input_desc),
                    "[Invoke][ExecuteAsync] %s(%s) Failed to execute partitioned call.",
                    task_context.GetNodeName(), task_context.GetNodeItem().NodeType().c_str());
  GELOGD("[%s] Done executing cond-subgraph successfully.", graph_item_->GetName().c_str());

  // get cond output
  GE_CHK_STATUS_RET(graph_executor_->Synchronize(),
                    "[Invoke][Synchronize][%s] Failed to sync cond-subgraph result.", graph_item_->GetName().c_str());
  GE_CHK_STATUS_RET(graph_executor_->GetOutputs(outputs, output_desc_list),
                    "[Invoke][GetOutputs][%s] Failed to get cond-output.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status ControlOpNodeTask::ToBool(const TensorValue &tensor, const DataType data_type, bool &value) {
  std::map<DataType, std::function<Status(const TensorValue &, bool &)>> data_type_to_copy_func = {
      {DT_FLOAT, &CopyFloatValueToHost<float32_t>},
      {DT_DOUBLE, &CopyFloatValueToHost<float64_t>},
      {DT_INT32, &CopyIntegerValueToHost<int32_t>},
      {DT_UINT8, &CopyIntegerValueToHost<uint8_t>},
      {DT_INT16, &CopyIntegerValueToHost<int16_t>},
      {DT_INT8, &CopyIntegerValueToHost<int8_t>},
      {DT_INT64, &CopyIntegerValueToHost<int64_t>},
      {DT_BOOL, &CopyBoolValueToHost}
  };

  const std::map<ge::DataType, std::function<ge::Status(const ge::hybrid::TensorValue &, bool &)>>::const_iterator
      iter = data_type_to_copy_func.find(data_type);
  if (iter == data_type_to_copy_func.cend()) {
    GELOGE(UNSUPPORTED, "Data type %s is not supported by cond.", TypeUtils::DataTypeToSerialString(data_type).c_str());
    return UNSUPPORTED;
  }
  return iter->second(tensor, value);
}

Status ControlOpNodeTask::UpdateArgs(TaskContext &context) {
  // do nothing
  (void)context;
  return SUCCESS;
}

Status ControlOpNodeTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  const auto ret = DoExecuteAsync(context, done_callback);
  context.SetStatus(ret);
  return ret;
}

Status IfOpNodeTask::Init(const NodePtr &node, const HybridModel &model) {
  GELOGD("[%s] Start to init IfOpNodeTask.", node->GetName().c_str());
  const auto &then_subgraph = NodeUtils::GetSubgraph(*node, kThenBranchIndex);
  GE_CHECK_NOTNULL(then_subgraph);
  GELOGD("[%s] Adding subgraph [%s] to then-subgraph.", node->GetName().c_str(), then_subgraph->GetName().c_str());
  auto graph_item = model.GetSubgraphItem(then_subgraph);
  GE_CHECK_NOTNULL(graph_item);
  then_ = ExecutorCache(graph_item);

  const auto &else_subgraph = NodeUtils::GetSubgraph(*node, kElseBranchIndex);
  GE_CHECK_NOTNULL(else_subgraph);
  GELOGD("[%s] Adding subgraph [%s] to else-subgraph.", node->GetName().c_str(), else_subgraph->GetName().c_str());
  graph_item = model.GetSubgraphItem(else_subgraph);
  GE_CHECK_NOTNULL(graph_item);
  else_ = ExecutorCache(graph_item);

  GELOGD("[%s] Done initialization successfully.", node->GetName().c_str());
  return SUCCESS;
}

Status IfOpNodeTask::DoExecuteAsync(TaskContext &task_context, const std::function<void()> &done_callback) {
  const auto &cond_tensor_desc = task_context.MutableInputDesc(kIfCondIndex);
  GE_CHECK_NOTNULL(cond_tensor_desc);
  const auto data_type = cond_tensor_desc->GetDataType();
  const auto &shape = cond_tensor_desc->MutableShape();
  bool cond_val = false;
  if (shape.IsScalar()) {
    const auto cond_tensor = task_context.GetInput(kIfCondIndex);
    GE_CHECK_NOTNULL(cond_tensor);
    GE_CHK_STATUS_RET(ToBool(*cond_tensor, data_type, cond_val),
                      "[Invoke][ToBool][%s(%s)] Failed to get cond value.",
                      task_context.GetNodeName(), task_context.GetNodeItem().NodeType().c_str());
  } else {
    // true if num elements is non-zero
    cond_val = shape.GetShapeSize() != 0;
    GELOGD("[%s] Cond tensor shape = [%s], cond value = %d",
           task_context.GetNodeName(),
           shape.ToString().c_str(),
           static_cast<int32_t>(cond_val));
  }

  auto &executor_cache = cond_val ? then_ : else_;
  GELOGD("[%s] Taking subgraph by cond = [%d]", task_context.GetNodeName(), static_cast<int32_t>(cond_val));
  GE_CHK_STATUS_RET(executor_cache.ExecuteSubgraph(task_context, done_callback),
                    "[Execute][Subgraph] failed for [%s(%s)]. cond = %d",
                    task_context.GetNodeName(), task_context.GetNodeItem().NodeType().c_str(),
                    static_cast<int32_t>(cond_val));

  GELOGD("[%s] Done executing with cond = %d successfully.", task_context.GetNodeName(),
         static_cast<int32_t>(cond_val));
  return SUCCESS;
}

Status CaseOpNodeTask::Init(const NodePtr &node, const HybridModel &model) {
  const size_t num_subgraphs = node->GetOpDesc()->GetSubgraphInstanceNames().size();
  GE_CHECK_LE(num_subgraphs, kMaxBranchNum);
  GE_CHECK_GE(num_subgraphs, kMinBranchNum);
  const auto num_branches = static_cast<uint32_t>(num_subgraphs);
  GELOGD("[%s] Start to init CaseOpNodeTask with %u branches.", node->GetName().c_str(), num_branches);

  for (uint32_t i = 0U; i < num_branches; ++i) {
    const auto sub_graph = NodeUtils::GetSubgraph(*node, i);
    GE_CHECK_NOTNULL(sub_graph);
    const auto graph_item = model.GetSubgraphItem(sub_graph);
    executors_.emplace_back(graph_item);
    GELOGD("[%s] Adding subgraph [%s] to branch %u.", node->GetName().c_str(), sub_graph->GetName().c_str(), i);
  }

  GELOGD("[%s] Done initialization successfully.", node->GetName().c_str());
  return SUCCESS;
}

ExecutorCache &CaseOpNodeTask::SelectBranch(size_t branch_index) {
  // subgraphs_ is non-empty. checked int32_t Init
  if (branch_index >= executors_.size()) {
    GELOGI("Branch index out of range. index = %zu, num_subgraphs = %zu, will taking last branch.",
           branch_index,
           executors_.size());
    branch_index = executors_.size() - 1U;
  }

  return executors_[branch_index];
}

Status CaseOpNodeTask::DoExecuteAsync(TaskContext &task_context, const std::function<void()> &done_callback) {
  const auto branch_tensor = task_context.GetInput(kCaseBranchIndex);
  GE_CHECK_NOTNULL(branch_tensor);
  int32_t branch_index = 0;
  GE_CHK_STATUS_RET(branch_tensor->CopyScalarValueToHost(branch_index));
  auto &subgraph_executor = SelectBranch(static_cast<size_t>(branch_index));
  GELOGI("[%s] Taking subgraph by branch = [%d]", task_context.GetNodeName(), branch_index);

  std::vector<TensorValue> inputs;
  for (int32_t i = 0; i < task_context.NumInputs(); ++i) {
    const auto input_tensor = task_context.GetInput(i);
    GE_CHECK_NOTNULL(input_tensor);
    inputs.emplace_back(*input_tensor);
  }

  GE_CHK_STATUS_RET(subgraph_executor.ExecuteSubgraph(task_context, done_callback),
                    "[Execute][Subgraph] failed for [%s(%s)].",
                    task_context.GetNodeName(), task_context.GetNodeItem().NodeType().c_str());

  GELOGD("[%s] Done executing subgraph[%d] successfully.", task_context.GetNodeName(), branch_index);
  return SUCCESS;
}

Status WhileOpNodeTask::Init(const NodePtr &node, const HybridModel &model) {
  GELOGD("[%s] Start to init WhileOpNodeTask.", node->GetName().c_str());
  const auto cond_subgraph = NodeUtils::GetSubgraph(*node, kCondBranchIndex);
  GE_CHECK_NOTNULL(cond_subgraph);
  GELOGD("[%s] Adding subgraph [%s] to cond-subgraph.", node->GetName().c_str(), cond_subgraph->GetName().c_str());
  auto graph_item = model.GetSubgraphItem(cond_subgraph);
  GE_CHECK_NOTNULL(graph_item);
  cond_ = ExecutorCache(graph_item);

  const auto body_subgraph = NodeUtils::GetSubgraph(*node, kBodyBranchIndex);
  GE_CHECK_NOTNULL(body_subgraph);
  GELOGD("[%s] Adding subgraph [%s] to body-subgraph.", node->GetName().c_str(), body_subgraph->GetName().c_str());
  graph_item = model.GetSubgraphItem(body_subgraph);
  GE_CHECK_NOTNULL(graph_item);
  body_ = ExecutorCache(graph_item);

  GELOGD("[%s] Done initialization successfully.", node->GetName().c_str());
  return SUCCESS;
}

Status WhileOpNodeTask::DoExecuteAsync(TaskContext &task_context, const std::function<void()> &done_callback) {
  if (task_context.NumInputs() != task_context.NumOutputs()) {
    REPORT_INNER_ERR_MSG("E19999",
                       "[%s(%s)] Invalid while args. num_inputs = %d not equal num_outputs = %d",
                       task_context.GetNodeName(), task_context.GetNodeItem().NodeType().c_str(),
                       task_context.NumInputs(), task_context.NumOutputs());
    GELOGE(INTERNAL_ERROR,
           "[Check][Param:task_context][%s(%s)] Invalid while args. num_inputs = %d, num_outputs = %d",
           task_context.GetNodeName(), task_context.GetNodeItem().NodeType().c_str(),
           task_context.NumInputs(), task_context.NumOutputs());
    return INTERNAL_ERROR;
  }

  bool is_continue = false;
  GE_CHK_STATUS_RET(ExecuteCond(task_context, is_continue),
                    "[Execute][Cond] failed for [%s(%s)]",
                    task_context.GetNodeName(), task_context.GetNodeItem().NodeType().c_str());
  if (!is_continue) {
    for (int32_t i = 0; i < task_context.NumInputs(); ++i) {
      const auto input_tensor = task_context.GetInput(i);
      const auto input_tensor_desc = task_context.GetInputDesc(i);
      const auto output_tensor_desc = task_context.MutableOutputDesc(i);
      GE_CHECK_NOTNULL(input_tensor);
      GE_CHECK_NOTNULL(input_tensor_desc);
      GE_CHECK_NOTNULL(output_tensor_desc);
      GE_CHK_STATUS_RET_NOLOG(task_context.SetOutput(i, *input_tensor));
      *output_tensor_desc = *input_tensor_desc;
    }

    if (done_callback) {
      done_callback();
    }
    return SUCCESS;
  }

  // backup original input tensor desc
  std::vector<GeTensorDesc> ori_input_desc(static_cast<size_t>(task_context.NumInputs()));
  for (int32_t i = 0; i < task_context.NumInputs(); ++i) {
    GE_CHK_STATUS_RET_NOLOG(task_context.GetInputDesc(i, ori_input_desc[static_cast<size_t>(i)]));
  }

  int32_t iteration = 0;
  while (is_continue) {
    ++iteration;
    GELOGD("[%s] Start to execute, iteration = %d", task_context.GetNodeName(), iteration);
    GE_CHK_STATUS_RET(ExecuteOneLoop(task_context, is_continue),
                      "[Invoke][ExecuteOneLoop][%s(%s)] Failed to execute iteration %d.",
                      task_context.GetNodeName(), task_context.GetNodeItem().NodeType().c_str(), iteration);
  }
  GELOGD("[%s] Quit from loop. current iteration = %d", task_context.GetNodeName(), iteration);
  if (done_callback) {
    done_callback();
  }

  const int32_t input_num = task_context.NumInputs();
  for (int32_t i = 0; i < input_num; ++i) {
    GE_CHK_STATUS_RET_NOLOG(task_context.UpdateInputDesc(i, ori_input_desc[static_cast<size_t>(i)]));
  }
  return SUCCESS;
}

Status WhileOpNodeTask::ExecuteCond(const TaskContext &task_context, bool &is_continue) {
  std::vector<TensorValue> cond_outputs;
  std::vector<ConstGeTensorDescPtr> cond_output_desc_list;
  GE_CHK_STATUS_RET(cond_.ExecuteSubgraph(task_context, cond_outputs, cond_output_desc_list),
                    "[Execute][Cond_Subgraph] failed for [%s(%s)]", task_context.GetNodeName(),
                    task_context.GetNodeItem().NodeType().c_str());

  if ((cond_outputs.size() != kCondOutputSize) || (cond_output_desc_list.size() != kCondOutputSize)) {
    REPORT_INNER_ERR_MSG("E19999", "[%s(%s)] Number of cond outputs(%zu) or size of cond output desc(%zu)"
                       "not equal %zu, check invalid", task_context.GetNodeName(),
                       task_context.GetNodeItem().NodeType().c_str(), cond_outputs.size(),
                       cond_output_desc_list.size(), kCondOutputSize);
    GELOGE(INTERNAL_ERROR,
           "[Check][Size][%s(%s)] Number of cond outputs(%zu) or Number of cond output desc(%zu) not equal %zu",
           task_context.GetNodeName(), task_context.GetNodeItem().NodeType().c_str(),
           cond_outputs.size(), cond_output_desc_list.size(), kCondOutputSize);
    return INTERNAL_ERROR;
  }

  auto &cond_tensor_desc = cond_output_desc_list[0U];
  const auto &shape = cond_tensor_desc->GetShape();
  if (shape.IsScalar()) {
    const auto data_type = cond_tensor_desc->GetDataType();
    GE_CHK_STATUS_RET(ToBool(cond_outputs[0U], data_type, is_continue),
                      "[Invoke][ToBool][%s(%s)] Failed to get cond value.",
                      task_context.GetNodeName(), task_context.GetNodeItem().NodeType().c_str());
  } else {
    // true if num elements is non-zero
    is_continue = shape.GetShapeSize() > 0;
    GELOGD("[%s] Cond tensor shape = [%s], is_continue = %d",
           task_context.GetNodeName(),
           shape.ToString().c_str(),
           static_cast<int32_t>(is_continue));
  }

  return SUCCESS;
}

Status WhileOpNodeTask::MoveOutputs2Inputs(const TaskContext &task_context) {
  // set outputs to inputs for next iteration
  for (int32_t i = 0; i < task_context.NumInputs(); ++i) {
    TensorValue *const input_tensor = task_context.MutableInput(i);
    TensorValue *const output_tensor = task_context.MutableOutput(i);
    GE_CHECK_NOTNULL(input_tensor);
    GE_CHECK_NOTNULL(output_tensor);
    *input_tensor = *output_tensor;
    output_tensor->Destroy();

    const auto input_tensor_desc = task_context.MutableInputDesc(i);
    GE_CHECK_NOTNULL(input_tensor_desc);
    const auto output_tensor_desc = task_context.MutableOutputDesc(i);
    GE_CHECK_NOTNULL(output_tensor_desc);
    GELOGD("[%s] To update input shape[%d] by output shape. from [%s] to [%s]",
           task_context.GetNodeName(),
           i,
           input_tensor_desc->GetShape().ToString().c_str(),
           output_tensor_desc->GetShape().ToString().c_str());
    *input_tensor_desc = *output_tensor_desc;
  }

  return SUCCESS;
}

Status WhileOpNodeTask::ExecuteOneLoop(const TaskContext &task_context, bool &is_continue) {
  GELOGD("[%s] Start to execute body-subgraph.", task_context.GetNodeName());
  GE_CHK_STATUS_RET(body_.ExecuteSubgraph(task_context, nullptr),
                    "[Execute][Subgraph] failed for [%s(%s)]",
                    task_context.GetNodeName(), task_context.GetNodeItem().NodeType().c_str());
  GELOGD("[%s] Done executing body-subgraph successfully.", task_context.GetNodeName());

  // set outputs to inputs for next iteration
  GE_CHK_STATUS_RET(MoveOutputs2Inputs(task_context),
                    "[Move][Outputs2Inputs] failed for [%s(%s)]",
                    task_context.GetNodeName(), task_context.GetNodeItem().NodeType().c_str());

  GE_CHK_STATUS_RET(ExecuteCond(task_context, is_continue),
                    "[Invoke][ExecuteCond][%s(%s)] Failed to execute cond-subgraph",
                    task_context.GetNodeName(), task_context.GetNodeItem().NodeType().c_str());

  if (!is_continue) {
    for (int32_t i = 0; i < task_context.NumInputs(); ++i) {
      const auto input_desc = task_context.GetInput(i);
      GE_CHECK_NOTNULL(input_desc);
      GE_CHK_STATUS_RET_NOLOG(task_context.SetOutput(i, *input_desc));
    }
  }
  return SUCCESS;
}

Status FusedOpNodeTask::Init(const NodePtr &node, const HybridModel &model) {
  GELOGD("[%s] Start to init FusedOpNodeTask.", node->GetName().c_str());
  const auto node_item = model.GetNodeItem(node);
  GE_CHECK_NOTNULL(node_item);
  GE_CHECK_NOTNULL(node_item->fused_subgraph);
  const auto &origin_graph = node_item->fused_subgraph->graph;
  GE_CHECK_NOTNULL(origin_graph);
  GELOGD("[%s] Adding fused_graph [%s] to origin-graph executor.", node->GetName().c_str(),
         origin_graph->GetName().c_str());
  const auto origin_subgraph_item = model.GetSubgraphItem(origin_graph);
  GE_CHECK_NOTNULL(origin_subgraph_item);
  origin_ = ExecutorCache(origin_subgraph_item);
  GELOGD("[%s] Done origin graph initialization successfully.", node->GetName().c_str());
  return SUCCESS;
}

Status FusedOpNodeTask::DoExecuteAsync(TaskContext &task_context, const std::function<void()> &done_callback) {
  GELOGD("[%s] Taking fused subgraph.", task_context.GetNodeName());
  GE_CHK_STATUS_RET(origin_.ExecuteSubgraph(task_context, done_callback), "[Execute][Subgraph] failed for [%s(%s)]",
                    task_context.GetNodeName(), task_context.GetNodeItem().NodeType().c_str());
  GELOGD("[%s] Done executing origin graph successfully.", task_context.GetNodeName());
  return SUCCESS;
}

Status ControlOpNodeExecutor::LoadTask(const HybridModel &model,
                                       const NodePtr &node,
                                       shared_ptr<NodeTask> &task) const {
  const auto node_item = model.GetNodeItem(node);
  GE_CHECK_NOTNULL(node_item);

  std::unique_ptr<ControlOpNodeTask> node_task;
  const auto node_type = node->GetType();
  if ((node_type == IF) || (node_type == STATELESSIF)) {
    node_task = MakeUnique<IfOpNodeTask>();
  } else if ((node_type == CASE) || (node_type == STATELESSCASE)) {
    node_task = MakeUnique<CaseOpNodeTask>();
  } else if ((node_type == WHILE) || (node_type == STATELESSWHILE)) {
    node_task = MakeUnique<WhileOpNodeTask>();
  } else {
    REPORT_INNER_ERR_MSG("E19999", "[%s] Unsupported type: %s", node->GetName().c_str(), node_type.c_str());
    GELOGE(PARAM_INVALID, "[Check][NodeType][%s] Unsupported type: %s", node->GetName().c_str(), node_type.c_str());
    return PARAM_INVALID;
  }

  GE_CHECK_NOTNULL(node_task);
  GE_CHK_STATUS_RET(node_task->Init(node, model), "[Invoke][Init][%s(%s)] Failed to init ControlOpNodeTask.",
                    node->GetName().c_str(), node_type.c_str());
  task = std::move(node_task);
  return SUCCESS;
}

Status ControlOpNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  (void)task;
  (void)context;
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
