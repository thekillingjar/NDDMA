/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "user_graph_ctrl.h"
#include "common/memory/tensor_trans_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "ge_context.h"
#include "formats/utils/formats_trans_utils.h"

#define JIT_CTRL_ASSERT(exp, ...)               \
  do {                                          \
    bool tmp_ret = (exp);                       \
    if (!tmp_ret) {                             \
      (void)Finalize();                         \
      GE_ASSERT_TRUE(tmp_ret, __VA_ARGS__);     \
      return ::ErrorResult();                   \
    }                                           \
  } while (false)

#define JIT_CTRL_ASSERT_NOTNULL(v, ...) JIT_CTRL_ASSERT(((v) != nullptr), __VA_ARGS__)
#define JIT_CTRL_ASSERT_TRUE(v, ...) JIT_CTRL_ASSERT(((v)), __VA_ARGS__)
#define JIT_CTRL_ASSERT_SUCCESS(v, ...) JIT_CTRL_ASSERT(((v) == ge::SUCCESS), __VA_ARGS__)
#define JIT_CTRL_ASSERT_RT_OK(v, ...) JIT_CTRL_ASSERT(((v) == 0), __VA_ARGS__)
namespace ge {
namespace {
std::string UserGraphExecutionToString(const std::unique_ptr<UserGraphExecution> &task) {
  std::stringstream ss;
  if ((task != nullptr) && (task->external_rt_inputs != nullptr)) {
    ss << "ExeTask inputs_size: [" << task->external_rt_inputs->size() << "],";
    for (size_t i = 0U; i < task->external_rt_inputs->size(); ++i) {
      const auto &gert_tensor = task->external_rt_inputs->at(i);
      ss << i << ":[";
      ss << "shape:[" << formats::GertShapeToString(gert_tensor.GetStorageShape()) << "],";
      ss << "origin_shape:[" << formats::GertShapeToString(gert_tensor.GetOriginShape()) << "],";
      ss << "format:[" << TypeUtils::FormatToSerialString(gert_tensor.GetStorageFormat()) << "],";
      ss << "origin_format:[" << TypeUtils::FormatToSerialString(gert_tensor.GetOriginFormat()) << "],";
      ss << "dtype:[" << TypeUtils::DataTypeToSerialString(gert_tensor.GetDataType()) << "]";
      ss << "]";
    }
  }
  return ss.str();
}
} // namespace
Status JitExecutorPool::AddJitExecutor(unique_ptr<JitExecutor> &jit_executor) {
  auto idle_executor = jit_executor.get();
  std::lock_guard<std::mutex> locker(executors_mutex_);
  executors.emplace_back(std::move(jit_executor));

  std::lock_guard<std::mutex> idle_locker(idle_mutex_);
  idle_executors.push(idle_executor);
  GELOGD("Add new jit executor, total num is %zu.", executors.size());
  return SUCCESS;
}

JitExecutor *JitExecutorPool::GetIdleExecutor() {
  std::lock_guard<std::mutex> locker(idle_mutex_);
  if (idle_executors.empty()) {
    return nullptr;
  }
  auto idle_executor = idle_executors.top();
  idle_executors.pop();
  return idle_executor;
}

void JitExecutorPool::BackToIdle(JitExecutor *jit_executor) {
  std::lock_guard<std::mutex> locker(idle_mutex_);
  idle_executors.push(jit_executor);
}

Status JitExecutorPool::Finalize() {
  std::lock_guard<std::mutex> locker(executors_mutex_);
  for (const auto &executor : executors) {
    GE_ASSERT_SUCCESS(executor->Finalize());
  }
  executors.clear();
  return SUCCESS;
}
size_t JitExecutorPool::Size() {
  std::lock_guard<std::mutex> locker(executors_mutex_);
  return executors.size();
}
bool JitExecutorPool::IsGraphNeedRebuild() {
  std::lock_guard<std::mutex> locker(executors_mutex_);
  for (const auto &executor : executors) {
    if (executor->IsUserGraphNeedRebuild()) {
      return true;
    }
  }
  return false;
}

Status UserGraphControl::Finalize() {
  GELOGD("Start to finalize user graph ctrl of %d", user_graph_id_);
  {
    std::unique_lock<std::mutex> lock(jit_futures_mutex_);
    while (!jit_futures_.empty()) {
      GELOGI("Wait jit thread to finish.");
      const Status ret_status = jit_futures_.front().get();
      if (ret_status != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "UserGraph %u run failed", user_graph_id_);
        GELOGE(ret_status, "[UserGraphCtrl][RunGraphAsync] UserGraph %u run failed", user_graph_id_);
      }
      jit_futures_.pop();
    }
  }
  StopQueue();
  GE_ASSERT_SUCCESS(jit_executor_pool_.Finalize());
  GE_ASSERT_SUCCESS(cmc_.SaveCache(order_));
  return SUCCESS;
}

Status UserGraphControl::AddGraphInstance() {
  auto jit_executor = JitExecutor::Create(graph_manager_, executions_, order_, compile_context_, cmc_, compile_mutex_);
  JIT_CTRL_ASSERT_NOTNULL(jit_executor, "[UserGraph:%u]Failed to create jit executor instance.", user_graph_id_);
  JIT_CTRL_ASSERT_SUCCESS(jit_executor_pool_.AddJitExecutor(jit_executor));
  const auto size = jit_executor_pool_.Size();
  GELOGD("[AddGraphInstance]Add new jit executor for graph[%u], total instance is %zu.", user_graph_id_,
         size);
  return SUCCESS;
}

void UserGraphControl::RunGraphAsync(std::unique_ptr<UserGraphExecution> &task) {
  GELOGI("RunGraphAsync USER_GRAPH[%u] with %s", user_graph_id_, UserGraphExecutionToString(task).c_str());
  std::lock_guard<std::mutex> lock(add_execution_mutex_);
  executions_.push(std::move(task));
}

void UserGraphControl::SetLoadOptions(const std::map<AscendString, AscendString> &load_options) {
  std::unique_lock<std::mutex> lock(options_mutex_);
  load_options_ = load_options;
}

std::map<AscendString, AscendString> UserGraphControl::GetLoadOptions() const {
  std::unique_lock<std::mutex> lock(options_mutex_);
  return load_options_;
}

Status UserGraphControl::CompileCompleteGraph(uint64_t session_id) {
  auto iter = user_graph_id_to_ins_id.find(user_graph_id_);
  uint32_t instance_id = 0;
  if (iter == user_graph_id_to_ins_id.end()) {
    instance_id = compile_context_.GenNewGraphId();
    ComputeGraphPtr new_graph = MakeShared<ComputeGraph>(order_.GetUserGraph().compute_graph->GetName());
    GE_ASSERT_NOTNULL(new_graph);
    GE_ASSERT_SUCCESS(GraphUtils::CopyComputeGraph(order_.GetUserGraph().compute_graph, new_graph));
    Graph graph_to_add = GraphUtilsEx::CreateGraphFromComputeGraph(new_graph);
    GE_ASSERT_SUCCESS(graph_manager_.AddGraph(instance_id, graph_to_add, {}, domi::GetContext()));
    user_graph_id_to_ins_id.emplace(user_graph_id_, instance_id);
  } else {
    instance_id = iter->second;
  }
  GELOGD("CompileGraph by inner session user_graph_id:%u instance_id:%u.", user_graph_id_, instance_id);
  return graph_manager_.CompileGraph(instance_id, session_id, {});
}

/*
 * CompileGraph接口必须要对整图进行编译，因为GetCompiledGraphSummary需要获取整图编译的结果，所以如果有切图的情况都不需要进入jit的流程，因此该接口的行为如下：
 * 1、对于图的data节点为静态tensor：a.不会切图的情况下(没有二三四类算子)在jit中正常生成EP、GEP进行编译；
 *                                b.会切图的情况下，生成对于的ins id进行一次整图编译，在jit流程中只编译第一张slice graph
 * 2、对于图的data节点为动态tensor：a.不会切图的情况下，生成对于的ins id直接进行整图编译，不进入jit流程(因为通过动态tensor构造的hint产生的符号以及guard没有意义，后面执行时还是要根据input重编译，因此也不需要进入jit流程)
 *                                b.会切图的情况下，生成对于的ins id直接进行整图编译，不进入jit流程
 */
Status UserGraphControl::CompileGraph(uint64_t session_id) {
  bool is_unknown_input_shape{false};
  const auto &inputs = order_.GetInputTensors(is_unknown_input_shape);
  if (is_unknown_input_shape) {
    GELOGI("CompileGraph dynamic graph %u need compile whole graph not by JIT.", user_graph_id_);
    GE_ASSERT_SUCCESS(CompileCompleteGraph(session_id));
    return SUCCESS;
  }
  auto compile_task = MakeUnique<UserGraphExecution>(user_graph_id_, inputs, nullptr, session_id);
  GE_ASSERT_NOTNULL(compile_task);

  GELOGI("CompileGraph USER_GRAPH[%u] with %s", user_graph_id_, UserGraphExecutionToString(compile_task).c_str());
  auto jit_executor = jit_executor_pool_.GetIdleExecutor();
  JIT_CTRL_ASSERT_NOTNULL(jit_executor);
  auto ret = jit_executor->CompileGraph(*compile_task, session_id);
  jit_executor_pool_.BackToIdle(jit_executor);
  if (ret == ge::GE_GRAPH_NOT_BUILT) {
    GELOGI("CompileGraph graph %u need compile whole graph not by JIT.", user_graph_id_);
    GE_ASSERT_SUCCESS(CompileCompleteGraph(session_id));
  }
  return SUCCESS;
}

CompiledGraphSummaryPtr UserGraphControl::GetCompiledGraphSummary() {
  CompiledGraphSummaryPtr ret{nullptr};
  auto iter = user_graph_id_to_ins_id.find(user_graph_id_);
  if (iter != user_graph_id_to_ins_id.end()) {
    GE_ASSERT_SUCCESS(graph_manager_.GetCompiledGraphSummary(iter->second, ret));
    return ret;
  }

  bool is_unknown_input_shape{false};
  const auto &inputs = order_.GetInputTensors(is_unknown_input_shape);
  if (is_unknown_input_shape) {
    GELOGI("dynamic graph %u skip.", user_graph_id_);
    return nullptr;
  }

  GELOGI("GetCompiledGraphSummary USER_GRAPH[%u]", user_graph_id_);
  ExecutionPoint *ep = order_.GetFirstPoint();
  if (ep == nullptr) {
    GELOGI("CompiledGraph is not exist. USER_GRAPH[%u]", user_graph_id_);
    return nullptr;
  }
  GELOGD("Get EP[%ld] of USER_GRAPH[%u] for GetCompiledGraphSummary", ep->GetId(), user_graph_id_);
  if (!ep->IsLast()) {
    GELOGD("CompiledGraph is not last");
    return nullptr;
  }

  auto gep = ep->FindGuarded(inputs);
  if (gep == nullptr || !gep->Compiled()) {
    GELOGD("Guarde is not exist or Compiled");
    return nullptr;
  }
  GELOGD("Get GEP[compiled_graph_id:%u] [compiled? %d] of EP[%ld] USER_GRAPH[%u].", gep->GetCompiledGraphId(),
         gep->Compiled(), ep->GetId(), user_graph_id_);
  GE_ASSERT_SUCCESS(compile_context_.GetCompiledGraphSummary(gep->GetCompiledGraphId(), ret));
  return ret;
}

Status UserGraphControl::LoadGraph(const std::map<AscendString, AscendString> &options, void *stream) {
  bool is_unknown_input_shape{false};
  const auto &inputs = order_.GetInputTensors(is_unknown_input_shape);
  if (is_unknown_input_shape) {
    GELOGI("CompileGraph dynamic graph %u skip.", user_graph_id_);
    SetLoadOptions(options);
    return SUCCESS;
  }
  auto load_task = MakeUnique<UserGraphExecution>(user_graph_id_, inputs, nullptr, INVALID_SESSION_ID);
  GE_ASSERT_NOTNULL(load_task);
  load_task->stream = stream;
  load_task->load_options = options;

  GELOGI("LoadGraph USER_GRAPH[%u] with %s", user_graph_id_, UserGraphExecutionToString(load_task).c_str());
  auto jit_executor = jit_executor_pool_.GetIdleExecutor();
  JIT_CTRL_ASSERT_NOTNULL(jit_executor);
  GELOGD("Start to LoadGraph");
  const auto ret = jit_executor->LoadGraph(*load_task);
  jit_executor_pool_.BackToIdle(jit_executor);
  return ret;
}

Status UserGraphControl::ExecuteGraphWithStreamAsync(std::unique_ptr<UserGraphExecution> task) {
  GELOGI("ExecuteGraphWithStreamAsync USER_GRAPH[%u] with %s", user_graph_id_, UserGraphExecutionToString(task).c_str());
  auto jit_executor = jit_executor_pool_.GetIdleExecutor();
  JIT_CTRL_ASSERT_NOTNULL(jit_executor);
  GELOGD("Start to ExecuteGraphWithStreamAsync");
  auto ret = jit_executor->Execute(std::move(*task));
  jit_executor_pool_.BackToIdle(jit_executor);
  return ret;
}

void UserGraphControl::StopQueue() {
  thread_is_stopped_.store(true);
  if (user_graph_exe_thread_.joinable()) {
    user_graph_exe_thread_.join();
  }
  jit_exe_thread_pool_.Destroy();
}

void UserGraphControl::ExecuteUserGraph() {
  while (!thread_is_stopped_) {
    if (executions_.empty()) {
      continue;
    }
    auto jit_executor = jit_executor_pool_.GetIdleExecutor();
    if (jit_executor == nullptr) {
      // 当前没有可用执行器，task要等等
      continue;
    }
    std::unique_lock<std::mutex> lock(add_execution_mutex_);
    std::shared_ptr<UserGraphExecution> exe_task = std::move(executions_.front());
    executions_.pop();
    GELOGD("Start to commit user graph execution task");
    lock.unlock();
    auto ge_context = ge::GetThreadLocalContext();
    auto fut = jit_exe_thread_pool_.commit(
        [pool = &jit_executor_pool_, executor = jit_executor, task = std::move(exe_task), ge_context]() -> Status {
          GELOGI("Start to commit user graph execution task in thread");
          GE_MAKE_GUARD(jit_executor_to_idle, ([&pool, &executor]() { pool->BackToIdle(executor); }));
          ge::GetThreadLocalContext() = ge_context;
          GE_ASSERT_SUCCESS(executor->RunWithCallback(std::move(*task)));
          return SUCCESS;
        });
    std::lock_guard<std::mutex> futs_lock(jit_futures_mutex_);
    if (!fut.valid()) {
      GELOGE(GRAPH_FAILED, "Failed to commit new task.");
      StopQueue();
      return;
    }
    jit_futures_.emplace(std::move(fut));
  }
}

bool UserGraphControl::GetCompiledFlag() const {
  return compiled_flag_;
}

void UserGraphControl::SetCompiledFlag(bool flag) {
  compiled_flag_ = flag;
}

bool UserGraphControl::IsUserGraphNeedRebuild() {
  const auto is_need_rebuild = jit_executor_pool_.IsGraphNeedRebuild();
  GELOGI("Graph instance id %u need rebuild : %d", user_graph_id_, is_need_rebuild);
  return is_need_rebuild;
}
}  // namespace ge