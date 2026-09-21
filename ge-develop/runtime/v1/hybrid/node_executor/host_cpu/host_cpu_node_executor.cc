/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/node_executor/host_cpu/host_cpu_node_executor.h"

#include "hybrid/model/hybrid_model.h"
#include "graph/def_types.h"
#include "graph/manager/mem_manager.h"
#include "host_cpu_engine/host_cpu_engine.h"
#include "aicpu_task_struct.h"
#include "mmpa/mmpa_api.h"

namespace ge {
namespace hybrid {
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::HOST_CPU, HostCpuNodeExecutor);

Status HostAicpuNodeTask::UpdateArgs(TaskContext &context) {
  if ((context.NumInputs() == 0) && (context.NumOutputs() == 0)) {
    GELOGD("Node[%s] has no input and output, no need to update args.", node_name_.c_str());
    return SUCCESS;
  }

  std::vector<uint64_t> io_addrs;
  io_addrs.reserve(static_cast<uint64_t>(context.NumInputs() + context.NumOutputs()));
  for (int32_t i = 0; i < context.NumInputs(); ++i) {
    const auto tensor = context.GetInput(i);
    GE_CHECK_NOTNULL(tensor);
    io_addrs.emplace_back(PtrToValue(tensor->GetData()));
  }

  for (int32_t i = 0; i < context.NumOutputs(); ++i) {
    const auto &output_desc = context.GetOutputDesc(i);
    GE_CHECK_NOTNULL(output_desc);
    AllocationAttr attr;
    attr.SetMemType(MemStorageType::HOST_DDR);
    if (context.AllocateOutput(i, *output_desc, nullptr, &attr) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "node:%s(%s) Failed to allocate output %d",
                        context.GetNodeName(), context.GetNodeItem().NodeType().c_str(), i);
      GELOGE(FAILED, "[Allocate][Output] for node:%s(%s) failed, output idx:%d",
             context.GetNodeName(), context.GetNodeItem().NodeType().c_str(), i);
      return FAILED;
    }
    const auto tensor = context.GetOutput(i);
    GE_CHECK_NOTNULL(tensor);
    io_addrs.emplace_back(PtrToValue(tensor->GetData()));
  }
  const auto io_addr = PtrAdd<uint8_t>(args_.get(), static_cast<size_t>(args_size_), sizeof(aicpu::AicpuParamHead));

  // if has input and output, need copy to ioaddr
  const int32_t cpy_ret = memcpy_s(io_addr, static_cast<size_t>(args_size_ - sizeof(aicpu::AicpuParamHead)),
      &io_addrs[0UL], sizeof(uint64_t) * io_addrs.size());
  if (cpy_ret != EOK) {
    REPORT_INNER_ERR_MSG("E19999", "Node[%s(%s)] memcpy io addr to AicpuParamHead failed,"
                       "ret=%d, args_size=%u, io nums=%zu.",
                       node_name_.c_str(), node_type_.c_str(), cpy_ret, args_size_, io_addrs.size());
    GELOGE(INTERNAL_ERROR, "[Update][IoAddr]Node[%s(%s)] memcpy io addr to AicpuParamHead failed,"
           "ret=%d, args_size=%u, io nums=%zu.",
           node_name_.c_str(), node_type_.c_str(), cpy_ret, args_size_, io_addrs.size());
    return INTERNAL_ERROR;
  }
  if (node_item_->is_dynamic) {
    // dynamic node and all_shape kernel need update ext info.
    GE_CHK_STATUS_RET(UpdateExtInfo(), "[Update][ExtInfo] failed for Node[%s(%s)].",
                      node_name_.c_str(), node_type_.c_str());
  }
  return SUCCESS;
}

Status HostAicpuNodeTask::UpdateExtInfo() {
  GELOGI("Node[%s] update ext info begin, unknown_type=%d.", node_name_.c_str(), unknown_type_);
  if ((node_item_->num_inputs == 0) && (node_item_->num_outputs == 0)) {
    GELOGD("Node[%s] has no input and output, no need update ext info.", node_name_.c_str());
    return SUCCESS;
  }

  for (auto i = 0; i < node_item_->num_inputs; ++i) {
    const auto input_desc = node_item_->MutableInputDesc(i);
    GE_CHECK_NOTNULL(input_desc);
    GE_CHK_STATUS_RET(aicpu_ext_handle_.UpdateInputShapeAndType(static_cast<uint32_t>(i), *input_desc),
                      "[Update][InputShapeAndType] failed for Node[%s(%s)] input[%d].",
                      node_name_.c_str(), node_type_.c_str(), i);
  }

  if ((unknown_type_ != DEPEND_COMPUTE) || (!node_item_->is_dynamic)) {
    for (auto j = 0; j < node_item_->num_outputs; ++j) {
      const auto output_desc = node_item_->MutableOutputDesc(j);
      GE_CHECK_NOTNULL(output_desc);

      GE_CHK_STATUS_RET(aicpu_ext_handle_.UpdateOutputShapeAndType(static_cast<uint32_t>(j), *output_desc),
                        "[Update][OutputShapeAndType] failed for Node[%s(%s)] output[%d].",
                        node_name_.c_str(), node_type_.c_str(), j);
    }
  }

  GELOGD("Node[%s] update ext info end.", node_name_.c_str());
  return SUCCESS;
}

Status HostAicpuNodeTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  GELOGD("[%s] Start execute.", context.GetNodeName());
  GE_CHK_STATUS_RET(Execute(), "[Invoke][Execute] failed for node:%s(%s).",
                    node_name_.c_str(), node_type_.c_str());
  if (done_callback) {
    GELOGD("[%s] Start invoke callback.", context.GetNodeName());
    done_callback();
  }
  GELOGD("[%s] Done execute successfully.", context.GetNodeName());
  return SUCCESS;
}

Status HostAicpuNodeTask::Execute(void) const {
  GELOGD("Node[%s] launch task start.", node_name_.c_str());
  if (run_cpu_kernel_) {
    GE_CHK_STATUS_RET(run_cpu_kernel_(args_.get()), "[Run][CpuKernel] failed for node:%s(%s).",
                      node_name_.c_str(), node_type_.c_str());
  } else {
    REPORT_INNER_ERR_MSG("E19999", "Run cpu kernel failed node:%s(%s), cpu kernel is not initialized.",
                      node_name_.c_str(), node_type_.c_str());
    GELOGE(INTERNAL_ERROR,
           "[Run][Kernel]Run cpu kernel failed node:%s(%s), cpu kernel is not initialized.",
           node_name_.c_str(), node_type_.c_str());
    return INTERNAL_ERROR;
  }

  GELOGD("Node[%s] launch task successfully.", node_name_.c_str());
  return SUCCESS;
}

Status HostAicpuNodeTask::SetHostExtInfo() const {
  if (aicpu_ext_handle_.GetExtInfoLen() == 0UL) {
    GELOGD("Node[%s] don't have ext info, no need update.", node_name_.c_str());
    return SUCCESS;
  }

  const auto aicpu_param_head = PtrToPtr<uint8_t, aicpu::AicpuParamHead>(args_.get());
  GE_CHECK_NOTNULL(aicpu_param_head);
  aicpu_param_head->extInfoLength = aicpu_ext_handle_.GetExtInfoLen();
  aicpu_param_head->extInfoAddr = PtrToValue(static_cast<void *>(aicpu_ext_handle_.GetExtInfo()));
  return SUCCESS;
}

Status HostCpuNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  return task.UpdateArgs(context);
}

Status HostCpuNodeExecutor::ValidateTaskDef(const domi::TaskDef &task_def) {
  const auto task_type = static_cast<ModelTaskType>(task_def.type());
  if (task_type != ModelTaskType::MODEL_TASK_KERNEL) {
    REPORT_INNER_ERR_MSG("E19999", "[Check][TaskType]Invalid task type (%d) in host cpu excutor.",
                      static_cast<int32_t>(task_type));
    GELOGE(INTERNAL_ERROR,
           "[Check][TaskType]Invalid task type (%d) in host cpu excutor.", static_cast<int32_t>(task_type));
    return INTERNAL_ERROR;
  }
  const auto kernel_type = static_cast<ccKernelType>(task_def.kernel().context().kernel_type());
  if (kernel_type != ccKernelType::HOST_CPU) {
    REPORT_INNER_ERR_MSG("E19999", "Invalid kernel type(%d) in host cpu excutor.",
                       static_cast<int32_t>(kernel_type));
    GELOGE(INTERNAL_ERROR,
           "[Check][TaskType]Invalid kernel type(%d) in host cpu excutor.", static_cast<int32_t>(kernel_type));
    return INTERNAL_ERROR;
  }

  return SUCCESS;
}

Status HostCpuNodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node,
                                     std::shared_ptr<NodeTask> &task) const {
  GE_CHECK_NOTNULL(node);
  auto node_item = model.GetNodeItem(node);
  GE_CHECK_NOTNULL(node_item);
  const auto task_defs = model.GetTaskDefs(node);
  GE_CHECK_NOTNULL(task_defs);

  if ((*task_defs).size() != 1UL) {
    REPORT_INNER_ERR_MSG("E19999", "Node[%s(%s)] task_def num[%zu] != 1",
                      node->GetName().c_str(), node->GetType().c_str(), (*task_defs).size());
    GELOGE(PARAM_INVALID, "[Check][Size] Node[%s(%s)] task_def num[%zu] != 1",
           node->GetName().c_str(), node->GetType().c_str(), (*task_defs).size());
    return PARAM_INVALID;
  }
  const auto &task_def = (*task_defs)[0UL];
  GE_CHK_STATUS_RET(ValidateTaskDef(task_def),
                    "[Validate][TaskDef] failed for Node[%s(%s)].",
                    node->GetName().c_str(), node->GetType().c_str());
  auto host_aicpu_task = MakeShared<HostAicpuNodeTask>(node_item, task_def);
  GE_CHK_BOOL_RET_STATUS(host_aicpu_task != nullptr, MEMALLOC_FAILED,
                         "[Create][HostAicpuNodeTask] Load task for node %s(%s) failed.",
                         node->GetName().c_str(), node->GetType().c_str());
  GE_CHK_STATUS_RET(host_aicpu_task->Init(model),
                    "[Init][HostAicpuNodeTask] failed for Node[%s(%s)].",
                    node->GetName().c_str(), node->GetType().c_str());
  GE_CHK_STATUS_RET(host_aicpu_task->SetHostExtInfo(),
                    "[Set][HostExtInfo] failed for Node[%s(%s)].", node->GetName().c_str(), node->GetType().c_str());

  const auto handle = HostCpuEngine::GetInstance().GetConstantFoldingHandle();
  if (handle == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Get constant folding handle failed.");
    GELOGE(INTERNAL_ERROR, "[Get][Handle]Get constant folding handle failed.");
    return INTERNAL_ERROR;
  }
  const auto run_cpu_kernel = reinterpret_cast<uint32_t (*)(void *)>(mmDlsym(handle, "RunHostCpuKernel"));
  if (run_cpu_kernel != nullptr) {
    host_aicpu_task->SetRunKernel(run_cpu_kernel);
  } else {
    REPORT_INNER_ERR_MSG("E19999", "Get run cpu kernel failed.");
    GELOGE(INTERNAL_ERROR, "[Get][Kernel]Get run cpu kernel failed.");
    return INTERNAL_ERROR;
  }

  task = std::move(host_aicpu_task);
  GELOGD("Node[%s] load task end.", node->GetName().c_str());

  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
