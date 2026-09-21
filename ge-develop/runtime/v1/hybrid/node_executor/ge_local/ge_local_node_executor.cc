/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/node_executor/ge_local/ge_local_node_executor.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_type_utils.h"
#include "framework/common/util.h"
#include "hybrid/model/hybrid_model.h"
#include "host_kernels/kernel.h"
#include "host_kernels/kernel_factory.h"
#include "common/plugin/ge_make_unique_util.h"
#include "hybrid/executor/hybrid_execution_context.h"

namespace ge {
namespace hybrid {
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::GE_LOCAL, GeLocalNodeExecutor);

const std::map<std::string, std::vector<uint32_t>>
    RefInputTask::out_ref_input_index_ = {{DATA, {}},
                                          {REFDATA, {}},
                                          {AIPPDATA, {}},
                                          {RESHAPE, {}},
                                          {EXPANDDIMS, {}},
                                          {SQUEEZE, {}},
                                          {UNSQUEEZE, {}},
                                          {SQUEEZEV2, {}},
                                          {UNSQUEEZEV2, {}},
                                          {FLATTENV2, {}},
                                          {BROADCASTGRADIENTARGS, {}},
                                          {SQUEEZEV3, {}},
                                          {UNSQUEEZEV3, {}}
                                         };

const std::set<std::string> DependInputShapeTask::depend_input_shape_ops_ = {SHAPE, SHAPEN, RANK, SIZE, GATHERSHAPES};

const std::set<std::string> ConstantNodeTask::constant_like_task_ops_ = {CONSTANT, CONSTANTOP, FILECONSTANT};

const std::set<std::string> NoOpNodeTask::control_only_task_ops_ = {NOOP, CONTROLTRIGGER};

const std::set<std::string> DataFlowNodeTask::data_flow_ops_ = {STACK, STACKPUSH, STACKPOP, STACKCLOSE};

Status RefInputTask::UpdateArgs(TaskContext &context) {
  // no need update args
  (void)context;
  return SUCCESS;
}

Status RefInputTask::Execute(const TaskContext &context) const {
  const auto iter = out_ref_input_index_.find(node_type_);
  if (iter == out_ref_input_index_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "node %s type %s can not use RefInputTask.",
                       node_name_.c_str(), node_type_.c_str());
    GELOGE(UNSUPPORTED, "[Find][Node]node %s type %s can not use RefInputTask.",
           node_name_.c_str(), node_type_.c_str());
    return UNSUPPORTED;
  }

  auto &ref_index = iter->second;
  if (ref_index.empty()) {
    return RefOneByOne(context);
  } else {
    return RefByOrder(ref_index, context);
  }
}

Status RefInputTask::RefOneByOne(const TaskContext &context) const {
  GELOGI("node %s type %s ref input one by one begin.", node_name_.c_str(), node_type_.c_str());
  const int32_t input_num = context.NumInputs();
  const int32_t output_num = context.NumOutputs();
  if (output_num > input_num) {
    REPORT_INNER_ERR_MSG("E19999", "node %s type %s has %d outputs but only %d inputs, can't ref one by one.",
                       node_name_.c_str(), node_type_.c_str(), output_num, input_num);
    GELOGE(INTERNAL_ERROR, "[Check][Size]node %s type %s has %d outputs but only %d inputs, can't ref one by one.",
           node_name_.c_str(), node_type_.c_str(), output_num, input_num);
    return INTERNAL_ERROR;
  }
  for (int32_t out_index = 0; out_index < output_num; ++out_index) {
    const auto input = context.GetInput(out_index);
    GE_CHECK_NOTNULL(input);
    const auto node_stat = context.GetNodeState();
    GE_CHECK_NOTNULL(node_stat);
    if (node_stat->IsUserAllocated()) {
      const auto output = context.MutableOutput(out_index);
      GE_CHECK_NOTNULL(output);
      const GeTensorDescPtr output_desc = context.MutableOutputDesc(out_index);
      GE_CHECK_NOTNULL(output_desc);
      int64_t expected_size = 0;
      if (output_desc->GetDataType() == DT_STRING) {
        expected_size = static_cast<int64_t>(input->GetSize());
      } else {
        GE_CHK_STATUS_RET(TensorUtils::GetTensorSizeInBytes(*output_desc, expected_size));
      }
      if (static_cast<size_t>(expected_size) > output->GetSize()) {
        GELOGE(GRAPH_PARAM_INVALID,
               "[Check][Size] %s(%s) index[%d] mem size out of range! Expected size: %ld, but given input size: %ld.",
               node_name_.c_str(), node_type_.c_str(), out_index, expected_size, output->GetSize());

        std::string reason = "The memory " + std::to_string(expected_size) + " required by the output " + std::to_string(out_index) +
                             " of the node " + node_name_ + "(" + node_type_ + ") is greater than the allocated memory " +
                             std::to_string(output->GetSize());
        REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const char_t *>({"reason"}),
                                  std::vector<const char_t *>({reason.c_str()}));
        return GRAPH_PARAM_INVALID;
      }
      GE_CHK_RT_RET(rtMemcpyAsync(output->MutableData(), output->GetSize(), input->GetData(),
          static_cast<uint64_t>(expected_size), RT_MEMCPY_DEVICE_TO_DEVICE, context.GetStream()));
    } else {
      GE_CHK_STATUS_RET(context.SetOutput(out_index, *input));
    }
    GELOGD("node %s type %s output[%d] ref input[%d] addr=%p.",
           node_name_.c_str(), node_type_.c_str(), out_index, out_index, input->GetData());
  }
  GELOGI("node %s type %s ref input one by one end.", node_name_.c_str(), node_type_.c_str());
  return SUCCESS;
}

Status RefInputTask::RefByOrder(const std::vector<uint32_t> &ref_order, const TaskContext &context) const {
  GELOGI("node %s type %s ref input by order begin.", node_name_.c_str(), node_type_.c_str());
  const int32_t output_num = context.NumOutputs();
  if (ref_order.size() != static_cast<size_t>(output_num)) {
    REPORT_INNER_ERR_MSG("E19999", "node %s type %s has %d outputs but only has %zu out ref index.",
                       node_name_.c_str(), node_type_.c_str(), output_num, ref_order.size());
    GELOGE(INTERNAL_ERROR, "[Check][Size]node %s type %s has %d outputs but only has %zu out ref index.",
           node_name_.c_str(), node_type_.c_str(), output_num, ref_order.size());
    return INTERNAL_ERROR;
  }
  for (int32_t out_index = 0; out_index < output_num; ++out_index) {
    const auto ref_input_index = ref_order[static_cast<size_t>(out_index)];
    const auto input = context.GetInput(static_cast<int32_t>(ref_input_index));
    GE_CHECK_NOTNULL(input);
    GE_CHK_STATUS_RET(context.SetOutput(out_index, *input));
    GELOGD("node %s type %s output[%d] ref input[%u] addr=%p.",
           node_name_.c_str(), node_type_.c_str(), out_index, ref_input_index, input->GetData());
  }
  GELOGI("node %s type %s ref input by order end.", node_name_.c_str(), node_type_.c_str());
  return SUCCESS;
}

Status RefInputTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[RefInputTaskExecuteAsync] Start");
  GE_CHK_STATUS_RET(Execute(context), "[Invoke][Execute]node:%s type:%s ref input task execute failed",
                    node_name_.c_str(), node_type_.c_str());
  if (done_callback != nullptr) {
    // host cpu no need register callback, call it directly.
    GE_CHK_STATUS_RET(context.TryExecuteCallback(done_callback));
  }
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[RefInputTaskExecuteAsync] End");
  return SUCCESS;
}

bool RefInputTask::IsBelong(const std::string &op_type) {
  return out_ref_input_index_.count(op_type) > 0U;
}

Status DependInputShapeTask::UpdateArgs(TaskContext &context) {
  // no need update args
  (void)context;
  return SUCCESS;
}

Status DependInputShapeTask::CopyDataToOutput(const size_t output_num,
                                              std::vector<GeTensorPtr> &outputs,
                                              const std::string &node_type,
                                              const TaskContext &context) const {
  // copy data to output
  for (size_t i = 0U; i < output_num; ++i) {
    GeTensorPtr &tensor_out = outputs[i];
    GE_CHECK_NOTNULL(tensor_out);
    const auto tensor_data_out = tensor_out->GetData();
    const auto tensor_value_out = context.MutableOutput(static_cast<int32_t>(i));
    GE_CHECK_NOTNULL(tensor_value_out);
    if (tensor_data_out.GetSize() > tensor_value_out->GetSize()) {
      REPORT_INNER_ERR_MSG("E19999", "node:%s type:%s [%" PRIu64 "]th compute data size=%zu, but context data size=%zu."
                                   "check invalid",
                         node_->GetName().c_str(), node_type.c_str(), static_cast<uint64_t>(i),
                         tensor_data_out.GetSize(), tensor_value_out->GetSize());
      GELOGE(INTERNAL_ERROR, "[Check][Size]node:%s type:%s [%lu]th compute data size=%zu, but context data size=%zu.",
             node_->GetName().c_str(), node_type.c_str(), i, tensor_data_out.GetSize(), tensor_value_out->GetSize());
      return INTERNAL_ERROR;
    }

    GELOGI("node:%s type:%s [%lu]th output data=%p, out size=%zu, data size=%zu.",
           node_->GetName().c_str(), node_type.c_str(), i,
           tensor_value_out->GetData(), tensor_value_out->GetSize(), tensor_data_out.GetSize());

    if (tensor_data_out.GetSize() > 0UL) {
      GE_CHK_RT_RET(rtMemcpyAsync(tensor_value_out->MutableData(),
                                  tensor_value_out->GetSize(),
                                  tensor_data_out.GetData(),
                                  tensor_data_out.GetSize(),
                                  RT_MEMCPY_HOST_TO_DEVICE_EX,
                                  context.GetStream()));
    }
    GELOGI("node:%s type:%s [%lu]th set data success, data size=%zu.",
           node_->GetName().c_str(), node_type.c_str(), i, tensor_data_out.GetSize());
  }
  return SUCCESS;
}

Status DependInputShapeTask::Execute(const TaskContext &context) const {
  KernelFactory &factory = KernelFactory::Instance();
  const std::string node_type = node_->GetType();
  const auto compute_kernel = factory.Create(node_type);
  if (compute_kernel == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "create failed for node %s type %s is not supported by host kernel.",
                      node_->GetName().c_str(), node_type.c_str());
    GELOGE(UNSUPPORTED, "[Invoke][Create]node %s type %s is not supported by host kernel.",
           node_->GetName().c_str(), node_type.c_str());
    return UNSUPPORTED;
  }
  std::vector<GeTensorPtr> outputs;
  const Status compute_ret = compute_kernel->Compute(node_, outputs);
  if (compute_ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "node %s type %s compute failed.", node_->GetName().c_str(), node_type.c_str());
    GELOGE(compute_ret, "[Invoke][Compute]node %s type %s compute failed or not imply.",
           node_->GetName().c_str(), node_type.c_str());
    return compute_ret;
  }
  const auto output_num = static_cast<uint32_t>(context.NumOutputs());
  if (static_cast<size_t>(output_num) != outputs.size()) {
    REPORT_INNER_ERR_MSG("E19999", "node %s type %s has %u output,"
                       "but kernel compute only has %zu output. check invalid",
                       node_->GetName().c_str(), node_type.c_str(), output_num, outputs.size());
    GELOGE(INTERNAL_ERROR, "[Check][Size]node %s type %s has %u output, but kernel compute only has %zu output.",
           node_->GetName().c_str(), node_type.c_str(), output_num, outputs.size());
    return INTERNAL_ERROR;
  }

  // alloc output
  GE_CHK_STATUS_RET_NOLOG(context.AllocateOutputs(NpuMemoryAllocator::AttrWithDefaultPadding()));

  // copy data to output
  return CopyDataToOutput(static_cast<size_t>(output_num), outputs, node_type, context);
}

Status DependInputShapeTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(),
                         "[DependInputShapeTaskExecuteAsync] Start");
  GE_CHK_STATUS_RET(Execute(context), "[Invoke][Execute]node:%s type:%s depend input shape task execute failed",
                    node_->GetName().c_str(), node_->GetType().c_str());
  if (done_callback != nullptr) {
    GE_CHK_STATUS_RET_NOLOG(context.RegisterCallback(done_callback));
  }
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(),
                         "[DependInputShapeTaskExecuteAsync] End");
  return SUCCESS;
}

bool DependInputShapeTask::IsBelong(const std::string &op_type) {
  return depend_input_shape_ops_.count(op_type) > 0U;
}

Status GeLocalNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(),
                         "[GeLocalNodeExecutorPrepareTask] Start");
  const Status status1 = task.UpdateArgs(context);
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[GeLocalNodeExecutorPrepareTask] End");
  return status1;
}

Status GeLocalNodeExecutor::LoadTask(const HybridModel &model,
                                     const NodePtr &node,
                                     std::shared_ptr<NodeTask> &task) const {
  GE_CHECK_NOTNULL(node);
  const std::string node_type = node->GetType();
  if (RefInputTask::IsBelong(node_type)) {
    GELOGI("node %s type %s is ref input task, use RefInputTask.",
           node->GetName().c_str(), node_type.c_str());
    task = MakeShared<RefInputTask>(node);
    if (task == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Create RefInputTask failed for node %s(%s).",
                        node->GetName().c_str(), node_type.c_str());
      GELOGE(MEMALLOC_FAILED, "[Create][RefInputTask] failed for node %s(%s).",
             node->GetName().c_str(), node_type.c_str());
      return MEMALLOC_FAILED;
    }
  } else if (DependInputShapeTask::IsBelong(node_type)) {
    GELOGI("node %s type %s is depend input shape task, use DependInputShapeTask.",
           node->GetName().c_str(), node_type.c_str());
    task = MakeShared<DependInputShapeTask>(node);
    if (task == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Create DependInputShapeTask failed for node %s type %s.",
                        node->GetName().c_str(), node_type.c_str());
      GELOGE(MEMALLOC_FAILED, "[Create][DependInputShapeTask]failed for node %s type %s.",
             node->GetName().c_str(), node_type.c_str());
      return MEMALLOC_FAILED;
    }
  } else if (ConstantNodeTask::IsBelong(node_type)) {
    GELOGI("node %s type %s, use ConstantNodeTask.", node->GetName().c_str(), node_type.c_str());
    auto tensor = model.GetTensor(node);
    if (tensor == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "GetTensor failed for node:%s(%s)", node->GetName().c_str(), node_type.c_str());
      GELOGE(INTERNAL_ERROR, "[Get][Tensor] failed for node:%s(%s)", node->GetName().c_str(), node_type.c_str());
      return INTERNAL_ERROR;
    }
    task = MakeShared<ConstantNodeTask>(tensor);
    GE_CHECK_NOTNULL(task);
  } else if (NoOpNodeTask::IsBelong(node_type)) {
    GELOGI("node %s type %s , use NoOpNodeTask.", node->GetName().c_str(), node_type.c_str());
    task = MakeShared<NoOpNodeTask>();
    if (task == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Create NoOpNodeTask failed for NoOp node %s(%s).",
                        node->GetName().c_str(), node_type.c_str());
      GELOGE(MEMALLOC_FAILED, "[Create][NoOpNodeTask]failed for NoOp node %s(%s).",
             node->GetName().c_str(), node_type.c_str());
      return MEMALLOC_FAILED;
    }
  } else if (DataFlowNodeTask::IsBelong(node_type)) {
    GELOGI("node %s type %s is data flow task, use DataFlowNodeTask.",
           node->GetName().c_str(), node_type.c_str());
    task = MakeShared<DataFlowNodeTask>(node_type);
    if (task == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Create DataFlowNodeTask failed for node %s type %s.",
                        node->GetName().c_str(), node_type.c_str());
      GELOGE(MEMALLOC_FAILED, "[Create][DataFlowNodeTask]failed for node %s type %s.",
             node->GetName().c_str(), node_type.c_str());
      return MEMALLOC_FAILED;
    }
    GE_CHK_STATUS_RET(task->InitTaskBasicInfo(node), "[Init][TaskBasicInfo]node:%s type:%s",
                      node->GetName().c_str(), node_type.c_str());
  } else {
    GELOGE(UNSUPPORTED, "[Check][Param] node %s type %s is not supported in GeLocalNodeExecutor now.",
        node->GetName().c_str(), node_type.c_str());
    return UNSUPPORTED;
  }
  return SUCCESS;
}

ConstantNodeTask::ConstantNodeTask(const TensorValue * const tensor_in) : NodeTask(), tensor_(tensor_in) {}

Status ConstantNodeTask::UpdateArgs(TaskContext &context) {
  (void)context;
  return SUCCESS;
}

Status ConstantNodeTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  GELOGD("[%s] Start execute.", context.GetNodeName());
  GE_CHK_STATUS_RET(context.SetOutput(0, *tensor_), "[Set][Output] failed for [%s(%s)].",
                    context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
  if (done_callback) {
    GELOGD("[%s] Start invoke callback.", context.GetNodeName());
    done_callback();
  }

  GELOGD("[%s] Done execute successfully.", context.GetNodeName());
  return SUCCESS;
}

bool ConstantNodeTask::IsBelong(const std::string &op_type) {
  return (constant_like_task_ops_.count(op_type) > 0U) || OpTypeUtils::IsVariableNode(op_type);
}

Status NoOpNodeTask::UpdateArgs(TaskContext &context) {
  // no need to update args
  (void)context;
  return SUCCESS;
}

Status NoOpNodeTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  GELOGD("[%s] Start execute.", context.GetNodeName());
  GE_CHK_STATUS_RET(context.TryExecuteCallback(done_callback));
  GELOGD("[%s] Done execute successfully.", context.GetNodeName());
  return SUCCESS;
}

bool NoOpNodeTask::IsBelong(const std::string &op_type) {
  return control_only_task_ops_.count(op_type) > 0U;
}

Status DataFlowNodeTask::InitTaskBasicInfo(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  if (!(AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_DATA_FLOW_HANDLE, handle_))) {
    GELOGE(INTERNAL_ERROR, "[Init][TaskBasicInfo]Failed to get handle, node:%s", node->GetName().c_str());
    return INTERNAL_ERROR;
  }
  GELOGD("Init data flow handle[%ld], node[%s]", handle_, node->GetName().c_str());
  return SUCCESS;
}

Status DataFlowNodeTask::UpdateArgs(TaskContext &context) {
  if (context.GetNodeItem().shape_inference_type != DEPEND_COMPUTE) {
    GE_CHK_STATUS_RET_NOLOG(context.AllocateOutputs());
  }
  return SUCCESS;
}

Status DataFlowNodeTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[DataFlowNodeTask] Start");
  GE_CHK_STATUS_RET(Execute(context), "[Invoke][Execute]node:%s type:%s execute failed",
                    context.GetNodeName(), node_type_.c_str());
  if (done_callback) {
    GELOGD("[%s] Start invoke callback.", context.GetNodeName());
    done_callback();
  }
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[DataFlowNodeTask] End");
  return SUCCESS;
}

bool DataFlowNodeTask::IsBelong(const std::string &op_type) {
  return data_flow_ops_.count(op_type) != 0UL;
}

Status DataFlowNodeTask::Execute(TaskContext &context) const {
  GE_CHECK_NOTNULL(context.GetExecutionContext());
  const auto kernel = context.GetExecutionContext()->res_manager.GetDataFlowKernel(node_type_);
  if (kernel == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Get kernel failed for node %s type %s.", context.GetNodeName(), node_type_.c_str());
    GELOGE(INTERNAL_ERROR, "[Invoke][Get]node %s type %s kernel is nullptr.",
           context.GetNodeName(), node_type_.c_str());
    return INTERNAL_ERROR;
  }
  const Status ret = kernel->Compute(context, handle_);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "node %s type %s compute failed.", context.GetNodeName(), node_type_.c_str());
    GELOGE(ret, "[Invoke][Compute]node %s type %s compute failed.", context.GetNodeName(), node_type_.c_str());
    return ret;
  }
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
