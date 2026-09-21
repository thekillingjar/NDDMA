/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/node_executor/rts/rts_node_executor.h"
#include "hybrid/node_executor/rts/rts_task_factory.h"

#include "common/plugin/ge_make_unique_util.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "hybrid/model/hybrid_model.h"
#include "framework/omg/omg_inner_types.h"
#include "framework/common/types.h"
#include "common/runtime_api_wrapper.h"

namespace {
constexpr uint64_t kProfilingIterStartLogid = 5UL;
constexpr uint64_t kProfilingIterEndLogid = 4UL;
constexpr uint64_t kProfilingArStartLogid = 10000UL;
constexpr uint64_t kProfilingArMaxLogid = 19999UL;
} // namespace
namespace ge {
namespace hybrid {
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::RTS, RtsNodeExecutor);

REGISTER_RTS_TASK_CREATOR(IDENTITY, IdentityNodeTask);
REGISTER_RTS_TASK_CREATOR(IDENTITYN, IdentityNNodeTask);
REGISTER_RTS_TASK_CREATOR(READVARIABLEOP, IdentityNNodeTask);
REGISTER_RTS_TASK_CREATOR(PROFILINGTRAININGTRACE, ProfilingTraceNodeTask);
REGISTER_RTS_TASK_CREATOR(STARTOFSEQUENCE, StartOfSequenceTask);
REGISTER_RTS_TASK_CREATOR(MEMCPYASYNC, IdentityNodeTask);
REGISTER_RTS_TASK_CREATOR(NPUGETFLOATSTATUS, NpuGetFloatStatusTask);
REGISTER_RTS_TASK_CREATOR(NPUCLEARFLOATSTATUS, NpuClearFloatStatusTask);
REGISTER_RTS_TASK_CREATOR(NPUGETFLOATSTATUSV2, NpuGetFloatStatusTask);
REGISTER_RTS_TASK_CREATOR(NPUCLEARFLOATSTATUSV2, NpuClearFloatStatusTask);

Status IdentityNodeTask::DoCopyTensor(const TaskContext &context, const int32_t index) const {
  GE_CHK_STATUS_RET_NOLOG(context.AllocateOutputs());
  const auto input_desc = context.MutableInputDesc(index);
  GE_CHECK_NOTNULL(input_desc);
  int64_t copy_size = 0;
  GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetTensorSizeInBytes(*input_desc, copy_size));
  // copy_size would not be negative since GetTensorSizeInBytes returned successfully.
  if (copy_size != 0) {
    GELOGD("[%s] index = %d, copy size = %ld", context.GetNodeName(), index, copy_size);
    const auto input = context.MutableInput(index);
    const auto output = context.MutableOutput(index);
    GE_CHECK_NOTNULL(input);
    GE_CHECK_NOTNULL(output);
    GE_CHK_RT_RET(rtMemcpyAsync(output->MutableData(),
                                output->GetSize(),
                                input->GetData(),
                                static_cast<uint64_t>(copy_size),
                                kind_,
                                context.GetStream()));
  } else {
    GELOGW("[%s] index = %d, copy size = 0", context.GetNodeName(), index);
  }

  return SUCCESS;
}

Status NpuGetFloatStatusTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  (void)done_callback;
  GELOGD("NpuGetFloatStatusTask execute async start.");

  GE_CHK_STATUS_RET_NOLOG(context.AllocateOutputs());
  const auto output = context.MutableOutput(0);
  GE_CHECK_NOTNULL(output);
  const auto output_addr = output->MutableData();
  const size_t args_size = sizeof(uint8_t *);
  if (args_ == nullptr) {
    GE_CHK_RT_RET(rtMalloc(&args_, args_size, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  }
  GE_CHK_RT_RET(
      rtMemcpyAsync(args_, args_size, &output_addr, args_size, RT_MEMCPY_HOST_TO_DEVICE_EX, context.GetStream()));

  const uint32_t mode = 0U;
  GE_CHK_RT_RET(ge::rtNpuGetFloatStatus(args_, output->GetSize(), mode, context.GetStream()));
  GELOGD("[%s] Done executing successfully.", context.GetNodeName());
  return SUCCESS;
}

Status NpuClearFloatStatusTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  (void)done_callback;
  GELOGD("NpuClearFloatStatusTask execute async start.");

  const uint32_t mode = 0U;
  GE_CHK_RT_RET(ge::rtNpuClearFloatStatus(mode, context.GetStream()));
  GELOGD("[%s] Done executing successfully.", context.GetNodeName());
  return SUCCESS;
}

Status IdentityNodeTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  GELOGD("[%s] ExecuteAsync Start.", context.GetNodeName());
  GE_CHK_STATUS_RET(DoCopyTensor(context, 0));

  if (done_callback != nullptr) {
    GE_CHK_STATUS_RET(context.RegisterCallback(done_callback));
  }

  GELOGD("[%s] Done executing successfully.", context.GetNodeName());
  return SUCCESS;
}

Status IdentityNodeTask::Init(const HybridModel &model, const NodePtr &node) {
  (void)model;
  const auto &op_desc = node->GetOpDesc();
  std::vector<int64_t> v_output_memory_type;
  std::vector<int64_t> v_input_memory_type;
  const bool has_output_mem_type_attr =
      AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_output_memory_type);
  const bool has_input_mem_type_attr =
      AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, v_input_memory_type);
  const bool is_output_svm =
      has_output_mem_type_attr && v_output_memory_type.size() == 1U && v_output_memory_type[0U] == RT_MEMORY_HOST_SVM;
  const bool is_input_svm =
      has_input_mem_type_attr && v_input_memory_type.size() == 1U && v_input_memory_type[0U] == RT_MEMORY_HOST_SVM;

  if (is_output_svm && !has_input_mem_type_attr) {
    kind_ = RT_MEMCPY_DEVICE_TO_HOST;
  } else if (is_input_svm && !has_output_mem_type_attr) {
    kind_ = RT_MEMCPY_HOST_TO_DEVICE;
  } else {
  }
  return SUCCESS;
}

Status IdentityNNodeTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  GELOGD("[%s] ExecuteAsync Start.", context.GetNodeName());
  for (int32_t i = 0; i < context.NumInputs(); ++i) {
    GE_CHK_STATUS_RET(DoCopyTensor(context, i));
  }

  if (done_callback != nullptr) {
    GE_CHK_STATUS_RET(context.RegisterCallback(done_callback));
  }

  GELOGD("[%s] Done executing successfully.", context.GetNodeName());
  return SUCCESS;
}

Status ProfilingTraceNodeTask::Init(const HybridModel &model, const NodePtr &node) {
  const auto *const task_defs = model.GetTaskDefs(node);
  if ((task_defs == nullptr) || task_defs->empty()) {
    GELOGE(INTERNAL_ERROR, "[Check][Param] Profiling node has no task to execute.");
    return INTERNAL_ERROR;
  }
  GE_CHECK_NOTNULL(model.GetRootGraph());
  model_id_ = model.GetModelId();
  GELOGD("The model is online:%d, model id is %u",
         static_cast<int32_t>(domi::GetContext().is_online_model), model_id_);
  task_defs_ = *task_defs;
  GELOGD("[%s] Done initialization successfully.", node->GetName().c_str());
  return SUCCESS;
}

Status ProfilingTraceNodeTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  (void)done_callback;
  uint64_t index_id = 1UL;
  for (const auto &task_def : task_defs_) {
    const auto log_time_stamp_def = task_def.log_timestamp();
    const uint64_t log_time_stamp_id = log_time_stamp_def.logid();

    GELOGD("ProfilingTraceTask execute async start. logid = %lu", log_time_stamp_id);

    if (((log_time_stamp_id > kProfilingIterEndLogid) && (log_time_stamp_id < kProfilingArStartLogid)) ||
        (log_time_stamp_id > kProfilingArMaxLogid)) {
      GELOGD("ProfilerTraceNodeTask log id:%lu out of range.", log_time_stamp_id);
      continue;
    }
    const rtError_t rt_ret = rtProfilerTraceEx(index_id, static_cast<uint64_t>(model_id_),
                                               static_cast<uint16_t>(log_time_stamp_id), context.GetStream());
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_INNER_ERR_MSG("E19999", "Call rtProfilerTraceEx failed, ret:%d", rt_ret);
      GELOGE(RT_FAILED, "[Call][RtProfilerTraceEx] failed, ret:%d", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
    GELOGD("[%s] ProfilingTraceTask[%lu] execute success.", context.GetNodeName(), log_time_stamp_id);
    index_id += 1UL;
  }
  return SUCCESS;
}

Status StartOfSequenceTask::Init(const HybridModel &model, const NodePtr &node) {
  GE_CHECK_NOTNULL(model.GetRootGraph());
  model_id_ = model.GetModelId();
  GELOGD("The model is online:%d, model id is %u",
         static_cast<int32_t>(domi::GetContext().is_online_model), model_id_);
  GELOGD("[%s] Done initialization successfully.", node->GetName().c_str());
  return SUCCESS;
}

Status StartOfSequenceTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  (void)done_callback;
  const uint64_t index_id = 1UL;
  GELOGD("StartOfSequenceTask execute async start. logid = %lu", kProfilingIterStartLogid);

  const rtError_t rt_ret = rtProfilerTraceEx(index_id, static_cast<uint64_t>(model_id_), kProfilingIterStartLogid,
                                             context.GetStream());
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtProfilerTraceEx failed, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtProfilerTraceEx] failed, ret:%d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  GELOGD("[%s] StartOfSequenceTask[%lu] execute success.", context.GetNodeName(), kProfilingIterStartLogid);
  return SUCCESS;
}

Status RtsNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  return task.UpdateArgs(context);
}

Status RtsNodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const {
  GE_CHECK_NOTNULL(node);
  GELOGD("[%s] Load for local task.", node->GetName().c_str());
  const std::string node_type = NodeUtils::GetNodeType(node);
  const RtsNodeTaskPtr rts_task = RtsTaskFactory::GetInstance().Create(node_type);
  if (rts_task == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "[%s] Unsupported RTS op type:%s", node->GetName().c_str(), node_type.c_str());
    GELOGE(UNSUPPORTED, "[Create][Task] failed, as [%s] Unsupported RTS op type:%s",
           node->GetName().c_str(), node_type.c_str());
    return UNSUPPORTED;
  }

  task = rts_task;
  return rts_task->Init(model, node);
}
}  // namespace hybrid
}  // namespace ge

