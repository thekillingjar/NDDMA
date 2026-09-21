/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge/ge_error_codes.h"
#include "common/llm_flow_service.h"
#include "acl/acl.h"
#include "swap_impl.h"
#include "common/llm_checker.h"
#include "common/llm_scope_guard.h"

namespace llm {
namespace {
constexpr int64_t kMillsToMicros = 1000;
constexpr int32_t kTimeoutOffset = 500;
constexpr int32_t kInitializeTimeoutInMillis = 5000;
constexpr uint64_t kDefaultCacheSizePerLayer = 2U;
constexpr int64_t kDefaultDstCacheId = -1;

void AddFuncDef(FlowNodeDef &flow_node_def,
                const std::string &func_name,
                std::vector<uint32_t> input_indices,
                std::vector<uint32_t> output_indices) {
  FlowFuncDef flow_func_def;
  flow_func_def.func_name = func_name;
  flow_func_def.input_name_prefix = func_name + "_args_";
  flow_func_def.input_indices = std::move(input_indices);
  flow_func_def.output_indices = std::move(output_indices);
  flow_node_def.flow_func_defs.emplace_back(std::move(flow_func_def));
}

void SetEnvAttrs(ge::dflow::FunctionPp &pp) {
  const std::vector<std::string> kHcclEnvNames = {
      "HCCL_RDMA_TC", "HCCL_RDMA_SL", "HCCL_RDMA_TIMEOUT", "HCCL_RDMA_RETRY_CNT"
  };
  std::vector<ge::AscendString> env_names;
  std::vector<ge::AscendString> env_values;
  for (const auto &env_name : kHcclEnvNames) {
    const auto env_value = std::getenv(env_name.c_str());
    if (env_value != nullptr) {
      LLMLOGI("[SetEnvAttrs] name = %s, value = %s", env_name.c_str(), env_value);
      env_names.emplace_back(env_name.c_str());
      env_values.emplace_back(env_value);
    }
  }
  pp.SetInitParam("ENV_NAMES", env_names);
  pp.SetInitParam("ENV_VALUES", env_values);
}

ge::Status FillBlockIndices(const PullCacheParam &pull_cache_param, PullKvReqInfo *req_info) {
  req_info->block_count = static_cast<uint32_t>(pull_cache_param.decoder_blocks.size());
  req_info->prompt_block_count = static_cast<uint32_t>(pull_cache_param.prompt_blocks.size());
  req_info->dst_tensor_index_count = static_cast<uint32_t>(pull_cache_param.dst_tensor_indices.size());
  req_info->src_tensor_index_count = static_cast<uint32_t>(pull_cache_param.src_tensor_indices.size());
  for (uint32_t i = 0U; i < req_info->block_count; ++i) {
    req_info->block_indices[i] = pull_cache_param.decoder_blocks[i];
  }
  if (!pull_cache_param.prompt_blocks.empty()) {
    LLM_CHK_BOOL_RET_STATUS(pull_cache_param.decoder_blocks.size() == pull_cache_param.prompt_blocks.size(),
                           ge::LLM_PARAM_INVALID, "Param prompt_blocks and decoder_blocks size should be same.");
    for (uint32_t i = 0U; i < req_info->prompt_block_count; ++i) {
      req_info->block_indices[req_info->block_count + i] = pull_cache_param.prompt_blocks[i];
    }
  }
  auto *dst_indices = req_info->block_indices + req_info->block_count + req_info->prompt_block_count;
  std::copy(pull_cache_param.dst_tensor_indices.cbegin(), pull_cache_param.dst_tensor_indices.cend(), dst_indices);
  auto *src_indices = dst_indices + req_info->dst_tensor_index_count;
  std::copy(pull_cache_param.src_tensor_indices.cbegin(), pull_cache_param.src_tensor_indices.cend(), src_indices);
  return ge::SUCCESS;
}

ge::Status FillPullKvReqInfo(const PullCacheParam &pull_cache_param, const CacheKey &cache_key,
                             PullKvReqInfo *req_info) {
  req_info->prompt_cluster_id = cache_key.prompt_cluster_id;
  req_info->prompt_cache_id = cache_key.prompt_cache_id;
  req_info->prompt_batch_index = cache_key.prompt_batch_index;
  req_info->req_id = cache_key.req_id;
  req_info->prefix_id = cache_key.prefix_id;
  req_info->model_id = cache_key.model_id;
  req_info->batch_index = pull_cache_param.batch_index;
  req_info->block_len = pull_cache_param.size > 0 ? static_cast<uint64_t>(pull_cache_param.size) : 0UL;
  LLM_CHK_STATUS_RET(FillBlockIndices(pull_cache_param, req_info));
  req_info->is_pull_with_offset =
      ((pull_cache_param.src_cache_offset >= 0) || (pull_cache_param.dst_cache_offset >= 0));
  req_info->src_cache_offset =
      pull_cache_param.src_cache_offset > 0 ? static_cast<uint64_t>(pull_cache_param.src_cache_offset) : 0UL;
  req_info->dst_cache_offset =
      pull_cache_param.dst_cache_offset > 0 ? static_cast<uint64_t>(pull_cache_param.dst_cache_offset) : 0UL;
  return ge::SUCCESS;
}
}  // namespace

ge::Status LlmWorker::Initialize(const std::map<ge::AscendString, ge::AscendString> &options) {
  ge_api_ = GeApi::GetInstance().NewSession(options);
  LLM_CHECK_NOTNULL(ge_api_);
  return ge::SUCCESS;
}

void LlmWorker::Finalize() {
  ge_api_.reset();
}

ge::Status LlmWorker::LoadFlowFuncs(const FlowNodeDef &flow_node_def,
                                    const std::map<ge::AscendString, ge::AscendString> &options,
                                    const std::vector<int32_t> &device_ids) {
  LLMLOGI("Load flow funcs start, device_index = %zu", device_index_);
  FlowGraphManager flow_graph_manager(cluster_id_);
  flow_graph_manager.SetCacheEngineMode(true);
  LLM_CHK_STATUS_RET(flow_graph_manager.ConstructFlowGraph(flow_node_def, options, device_ids));
  LLM_CHK_STATUS_RET(flow_graph_manager.BuildFlowGraph({}, ge_api_.get()));
  flow_node_def_ = flow_node_def;
  // not loaded as udf func, LlmFlowService::BuildFlowNodeDef中固定在index = 3
  AddFuncDef(flow_node_def_, "GetCacheTensors", {}, {3});
  mutexes_ = std::vector<std::mutex>(flow_node_def_.flow_func_defs.size());
  transaction_ids_ = std::vector<uint64_t>(flow_node_def_.flow_func_defs.size(), 1U);
  return ge::SUCCESS;
}

ge::Status LlmWorker::FeedInputData(const FlowFuncDef &flow_func_def, const std::vector<ge::Tensor> &inputs,
                                    uint64_t transaction_id) const {
  ge::DataFlowInfo data_flow_info;
 data_flow_info.SetTransactionId(transaction_id);
  const auto ret =
      ge_api_->FeedDataFlowGraph(graph_id_, flow_func_def.input_indices, inputs, data_flow_info, timeout_);
  LLM_CHK_BOOL_RET_STATUS(ret != llm::ConvertAclError2Ge(ACL_ERROR_GE_MODEL_EXECUTE_TIMEOUT), ge::LLM_WAIT_PROC_TIMEOUT,
                         "Feed timeout.");
  LLM_CHK_STATUS_RET(ret, "[%zu] [%s] Failed to feed inputs", device_index_, flow_func_def.func_name.c_str());
  LLMLOGD("[%zu] [%s] Feed success", device_index_, flow_func_def.func_name.c_str());
  return ge::SUCCESS;
}

ge::Status LlmWorker::RunFlowFunc(FlowFuncType flow_func_type, size_t flow_func_index,
                                  const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs,
                                  std::vector<std::chrono::steady_clock::time_point> &time_points) const {
  LLM_CHK_BOOL_RET_STATUS(flow_func_index < flow_node_def_.flow_func_defs.size(), ge::FAILED,
                         "flow func index (%zu) out of range [0, %zu)", flow_func_index,
                         flow_node_def_.flow_func_defs.size());
  const auto &flow_func_def = flow_node_def_.flow_func_defs[flow_func_index];
  LLMLOGD("[%zu] [%s] Run flow func start", device_index_, flow_func_def.func_name.c_str());
  auto feed_start = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lk(mutexes_[flow_func_index]);
  auto transaction_id = transaction_ids_[flow_func_index]++;
  bool has_input = !flow_func_def.input_indices.empty();
  if (has_input) {
    LLM_CHK_STATUS_RET(FeedInputData(flow_func_def, inputs, transaction_id), "Feed input data failed.");
  }
  time_points.emplace_back(std::chrono::steady_clock::now());  // feed end
  auto feed_end = time_points.back();
  auto feed_time_cost = std::chrono::duration_cast<std::chrono::microseconds>(feed_end - feed_start).count();
  auto left_timeout = timeout_ * kMillsToMicros;
  // feed cost time already over time limit.
  LLM_CHK_BOOL_RET_STATUS(feed_time_cost < left_timeout, ge::LLM_WAIT_PROC_TIMEOUT, "Feed timeout, cost:%ld, left:%ld.",
                         feed_time_cost, left_timeout);

  auto last_time = feed_end;
  while (true) {
    ge::DataFlowInfo data_flow_info;
    auto fetch_timeout = static_cast<int32_t>(left_timeout / kMillsToMicros);
    LLM_CHK_BOOL_RET_STATUS(fetch_timeout >= 1, ge::LLM_WAIT_PROC_TIMEOUT, "Fetch timeout.");
    const auto ret =
        ge_api_->FetchDataFlowGraph(graph_id_, flow_func_def.output_indices, outputs, data_flow_info, fetch_timeout);
    if (flow_func_type != FlowFuncType::kInitialize) {
      LLM_CHK_BOOL_RET_STATUS(ret != llm::ConvertAclError2Ge(ACL_ERROR_GE_MODEL_EXECUTE_TIMEOUT),
                             ge::LLM_WAIT_PROC_TIMEOUT, "Fetch timeout.");
    }
    LLM_CHK_STATUS_RET(ret, "[%zu] [%s] Failed to fetch outputs", device_index_, flow_func_def.func_name.c_str());
    if (!has_input || (data_flow_info.GetTransactionId() == transaction_id)) {
      break;
    }
    LLMEVENT("Ignore invalid transaction id:%lu for func index:%zu", data_flow_info.GetTransactionId(), flow_func_index);
    auto cur_time = std::chrono::steady_clock::now();
    auto fetch_time_cost = std::chrono::duration_cast<std::chrono::microseconds>(cur_time - last_time).count();
    last_time = cur_time;
    // current fetch cost time already over time limit.
    if (flow_func_type != FlowFuncType::kInitialize) {
      LLM_CHK_BOOL_RET_STATUS(fetch_time_cost < left_timeout, ge::LLM_WAIT_PROC_TIMEOUT, "Fetch timeout.");
    } else {
      LLM_CHK_BOOL_RET_STATUS(fetch_time_cost < left_timeout, ge::FAILED, "Fetch timeout.");
    }
    left_timeout -= static_cast<int32_t>(fetch_time_cost);
  }
  time_points.emplace_back(std::chrono::steady_clock::now());  // fetch end
  LLMLOGD("[%zu] [%s] Fetch output success", device_index_, flow_func_def.func_name.c_str());
  return ge::SUCCESS;
}

size_t LlmWorker::GetDeviceIndex() const {
  return device_index_;
}

const std::string &LlmWorker::GetLogicalDeviceId() const {
  return logical_device_id_;
}

void LlmWorker::SetTimeout(int32_t timeout) {
  timeout_ = timeout;
}

GeApi *LlmWorker::GetGeApi() const {
  return ge_api_.get();
}

LlmFlowService::LlmFlowService(const std::string &role, bool enable_switch_role, uint64_t cluster_id)
    : role_(role), enable_switch_role_(enable_switch_role), cluster_id_(cluster_id) {
}

ge::Status LlmFlowService::Initialize(const std::map<ge::AscendString, ge::AscendString> &options,
                                      const std::vector<int32_t> &device_ids) {
  device_ids_ = device_ids;
  deploy_info_.SetCacheEngineMode(true);
  LLM_CHK_STATUS_RET(deploy_info_.ParserDeployInfo(options), "Failed to parse deploy info");
  DecoderWaitTimeInfo wait_time_info{};
  LLM_CHK_STATUS_RET(LLMUtils::ParserWaitTimeInfo(options, wait_time_info), "parser wait time info failed");
  sync_kv_wait_time_ = wait_time_info.sync_kv_wait_time;
  device_indices_ = deploy_info_.GetDeviceIndices();
  const auto &logical_device_ids = deploy_info_.GetLogicalDeviceIds();
  is_spmd_ = device_indices_.size() == 1U;
  LLMLOGI("device_num in numa config = %zu, deploy to device_indices = %s",
         logical_device_ids.size(),
         llm::ToString(device_indices_).c_str());
  auto ret = main_pool_.commit([&options]() -> ge::Status {
    return GeApi::GetInstance().Initialize(options);
  }).get(); // 异步执行，防止在主线程SetDevice，否则可能会影响torch_npu场景
  LLM_CHK_STATUS_RET(ret, "Failed to initialize GE");

  worker_pool_ = llm::MakeUnique<llm::LLMThreadPool>("ge_llm_wker", device_indices_.size());
  LLM_CHECK_NOTNULL(worker_pool_);
  std::vector<GeApi *> all_ge_apis;
  for (const size_t device_index : device_indices_) {
    LlmWorker worker(cluster_id_, device_index, logical_device_ids[device_index]);
    ret = worker_pool_->commit([device_index, this, &worker, &options]() -> ge::Status {
      if (device_index < device_ids_.size()) {
        LLM_CHK_BOOL_RET_STATUS(aclrtSetDevice(device_ids_[device_index]) == ACL_ERROR_NONE, ge::FAILED,
                               "Failed to set device, device id = %d", device_ids_[device_index]);
      }
      return worker.Initialize(options);
    }).get();
    LLM_CHK_STATUS_RET(ret, "Failed to init worker");
    worker.SetTimeout(kInitializeTimeoutInMillis);
    all_ge_apis.emplace_back(worker.GetGeApi());
    workers_.emplace_back(std::move(worker));
  }

  auto flow_node_def = BuildFlowNodeDef(role_);
  LLM_CHK_STATUS_RET(LoadDataFlow(flow_node_def, options));
  flow_func_defs_ = flow_node_def.flow_func_defs;
  SetLinkManagerIo(role_);
  link_manager_.SetAllGeApis(all_ge_apis);

  LLM_CHK_STATUS_RET(InitializeUdf());
  for (auto &worker : workers_) {
    worker.SetTimeout(wait_time_info.sync_kv_wait_time);
  }
  cache_manager_.InitCopyStreams(device_ids_.size());
  return ge::SUCCESS;
}

void LlmFlowService::SetLinkManagerIo(const std::string &role) {
  if (role != kMix) {
    const auto flow_func_type = (role == kPrompt) ? FlowFuncType::kUnlink : FlowFuncType::kUpdateLink;
    const auto flow_func_index = flow_func_indices_[static_cast<size_t>(flow_func_type)];
    const auto cluster_flow_func_def = flow_func_defs_[flow_func_index];
    link_manager_.Initialize(cluster_flow_func_def.input_indices,
                             cluster_flow_func_def.output_indices,
                             deploy_info_.GetDeployedDeviceNum(), cluster_id_);
    if (role == kDecoder) {
      const auto unlink_flow_func_index = flow_func_indices_[static_cast<size_t>(FlowFuncType::kUnlink)];
      const auto unlink_cluster_flow_func_def = flow_func_defs_[unlink_flow_func_index];
      link_manager_.SetDifferentUnlinkIndices(unlink_cluster_flow_func_def.input_indices,
                                              unlink_cluster_flow_func_def.output_indices);
    }
  }
}

void LlmFlowService::Finalize() {
  // 在线程池中执行，防止在主线程SetDevice，否则可能会影响torch_npu场景
  std::vector<std::future<void>> futures;
  for (size_t i = 0U; i < workers_.size(); ++i) {
    auto &worker = workers_[i];
    auto fut = worker_pool_->commit([this, i, &worker]() -> void {
      cache_manager_.DestroyCopyStream(i);
      worker.Finalize();
      if (i < device_ids_.size()) {
        LLM_CHK_ACL(aclrtResetDevice(device_ids_[i]));
      }
    });
    futures.emplace_back(std::move(fut));
  }
  for (auto &fut : futures) {
    fut.wait();
  }
  auto ret = main_pool_.commit([]() -> ge::Status {
    return GeApi::GetInstance().SafeFinalize();
  }).get();
  LLM_CHK_STATUS(ret, "Failed to invoke GEFinalize");
  worker_pool_.reset();
  main_pool_.Destroy();
  LLMLOGI("LLM flow service finalized");
}

void LlmFlowService::SetFunctionPpInitParam(const FlowNodeDef &flow_node_def, ge::dflow::FunctionPp &pp) const {
  pp.SetInitParam("device_index", static_cast<int64_t>(flow_node_def.logical_device_index));
  pp.SetInitParam("role", flow_node_def.role.c_str());
  SetEnvAttrs(pp);
  std::vector<int64_t> output_indices;
  for (const auto flow_func_index : flow_func_indices_) {
    int64_t out_index = -1;
    if (flow_func_index != UINT32_MAX) {
      const auto &flow_func_def = flow_node_def.flow_func_defs[flow_func_index];
      if (!flow_func_def.output_indices.empty()) {
        out_index = flow_func_def.output_indices.front();
      }
      LLMLOGI("flow_func_id = %zu, name = %s, out_index = %ld",
             flow_func_index,
             flow_func_def.func_name.c_str(),
             out_index);
    }
    output_indices.emplace_back(out_index);
  }
  LLMLOGI("output_indices = %s", llm::ToString(output_indices).c_str());
  pp.SetInitParam("output_indices", output_indices);
  const auto &listen_ips_info = deploy_info_.GetListenIpsInfo();
  if (!listen_ips_info.empty()) {
    const auto &ip_and_port = listen_ips_info[flow_node_def.logical_device_index];
    pp.SetInitParam("ip", ip_and_port.first);
    pp.SetInitParam("port", ip_and_port.second);
    LLMLOGI("device_index = %zu, ip = %ld, port = %ld",
           flow_node_def.logical_device_index,
           ip_and_port.first,
           ip_and_port.second);
  }
}

ge::Status LlmFlowService::LoadDataFlow(const FlowNodeDef &flow_node_def,
                                        const std::map<ge::AscendString, ge::AscendString> &options) {
  std::vector<std::future<ge::Status>> futures;
  llm::LLMThreadPool pool("ge_llm_ldff", 1); // not support parallel compile yet
  for (size_t i = 0U; i < device_indices_.size(); ++i) {
    auto fut = pool.commit([this, i, &flow_node_def, &options]() -> ge::Status {
      if (i < device_ids_.size()) {
        LLM_CHK_BOOL_RET_STATUS(aclrtSetDevice(device_ids_[i]) == ACL_ERROR_NONE, ge::FAILED,
                                "Failed to set device, device id = %d", device_ids_[i]);
      }
      auto node_def = flow_node_def;
      node_def.logical_device_index = device_indices_[i];
      node_def.pp_setter = [this, &node_def](ge::dflow::FunctionPp pp) {
        SetFunctionPpInitParam(node_def, pp);
        return pp;
      };
      LLM_CHK_STATUS_RET(workers_[i].LoadFlowFuncs(node_def, options, device_ids_));
      return ge::SUCCESS;
    });
    futures.emplace_back(std::move(fut));
  }
  LLM_CHK_STATUS_RET(GetAndCheckFutures(futures), "Failed to load data flow in all devices");
  const auto get_cache_tensor_func_id = flow_node_def.flow_func_defs.size();
  flow_func_indices_[static_cast<size_t>(FlowFuncType::kGetTensor)] = get_cache_tensor_func_id;
  LLMLOGI("get_cache_tensor_func_id = %zu", get_cache_tensor_func_id);
  return ge::SUCCESS;
}

ge::Status LlmFlowService::GetAndCheckFutures(std::vector<std::future<ge::Status>> &futures) {
  size_t success_count = 0U;
  ge::Status status = ge::SUCCESS;
  for (size_t i = 0U; i < futures.size(); ++i) {
    auto &fut = futures[i];
    auto ret = fut.get();
    LLM_CHK_STATUS(ret, "execute failed, device index = %zu", i);
    if (ret == ge::SUCCESS) {
      ++success_count;
    } else {
      status = ret;
    }
  }
  LLM_CHK_STATUS_RET(status, "Failed to execute in all devices, success/total = %zu/%zu", success_count, futures.size());
  return ge::SUCCESS;
}

FlowNodeDef LlmFlowService::BuildFlowNodeDef(const std::string &role) {
  std::map<FlowFuncType, std::string> flow_func_type_to_name{
      {FlowFuncType::kUpdateLink, "_BuiltIn_cluster_update_link_func"},
      {FlowFuncType::kUnlink, "_BuiltIn_cluster_unlink_func"},
      {FlowFuncType::kAllocate, "_BuiltIn_cache_allocate_func"},
      {FlowFuncType::kDeallocate, "_BuiltIn_cache_deallocate_func"},
      {FlowFuncType::kGetTensorSummary, "_BuiltIn_cache_get_func"},
      {FlowFuncType::kCopy, "_BuiltIn_cache_copy_func"},
      {FlowFuncType::kRemoveIndex, "_BuiltIn_cache_remove_index_func"},
      {FlowFuncType::kPull, "_BuiltIn_cache_pull_func"},
      {FlowFuncType::kTransfer, "_BuiltIn_cache_transfer_func"},
      {FlowFuncType::kSwitchRole, "_BuiltIn_switch_role_func"},
      {FlowFuncType::kInitialize, "_BuiltIn_initialize_func"},
      {FlowFuncType::kCheckLink, "_BuiltIn_check_link_func"},
  };

  // can not change kGetTensorSummary position which is 3, GetCacheTensors hard code to 3 in this file.
  std::vector<FlowFuncType> enabled_flow_funcs = {FlowFuncType::kAllocate, FlowFuncType::kDeallocate,
                                                  FlowFuncType::kCopy, FlowFuncType::kGetTensorSummary,
                                                  FlowFuncType::kUnlink};
  if (enable_switch_role_) {
    enabled_flow_funcs.emplace_back(FlowFuncType::kUpdateLink);
    enabled_flow_funcs.emplace_back(FlowFuncType::kPull);
    enabled_flow_funcs.emplace_back(FlowFuncType::kRemoveIndex);
    enabled_flow_funcs.emplace_back(FlowFuncType::kSwitchRole);
    enabled_flow_funcs.emplace_back(FlowFuncType::kTransfer);
    enabled_flow_funcs.emplace_back(FlowFuncType::kCheckLink);
  } else if (role == kPrompt) {
    enabled_flow_funcs.emplace_back(FlowFuncType::kRemoveIndex);
    enabled_flow_funcs.emplace_back(FlowFuncType::kTransfer);
  } else if (role == kDecoder) {
    enabled_flow_funcs.emplace_back(FlowFuncType::kPull);
    enabled_flow_funcs.emplace_back(FlowFuncType::kCheckLink);
    enabled_flow_funcs.emplace_back(FlowFuncType::kUpdateLink);
  } else {
    // mix, do nothing
  }
  enabled_flow_funcs.emplace_back(FlowFuncType::kInitialize);
  flow_func_indices_.fill(UINT32_MAX);
  FlowNodeDef flow_node_def;
  flow_node_def.is_prompt = (role == kPrompt);
  flow_node_def.role = role;
  for (size_t flow_func_index = 0U; flow_func_index < enabled_flow_funcs.size(); ++flow_func_index) {
    const auto flow_func_type = enabled_flow_funcs[flow_func_index];
    const auto &flow_func_name = flow_func_type_to_name[flow_func_type];
    const auto queue_index = static_cast<uint32_t>(flow_func_index);
    flow_func_indices_[static_cast<size_t>(flow_func_type)] = flow_func_index;
    AddFuncDef(flow_node_def, flow_func_name, {queue_index}, {queue_index});
    LLMLOGI("flow_func added, name = %s, flow_func_index = %zu", flow_func_name.c_str(), flow_func_index);
  }
  AddFuncDef(flow_node_def, "_BuiltIn_cluster_kv_data_sync_func", {}, {});
  LLMLOGI("flow_func added, name = _BuiltIn_cluster_kv_data_sync_func");
  return flow_node_def;
}

template<typename T>
ge::Status LlmFlowService::RunFlowFunc(
    FlowFuncType flow_func_type,
    const std::vector<ge::Tensor> &inputs,
    const std::function<ge::Status(size_t, const std::vector<ge::Tensor> &, T &)> &output_handler,
    std::vector<TaskContext<T>> &per_device_task_context) const {
  per_device_task_context.resize(device_indices_.size());
  auto flow_func_index = flow_func_indices_[static_cast<size_t>(flow_func_type)];
  if (is_spmd_) {
    return RunSingle(workers_.front(), std::make_pair(flow_func_type, flow_func_index), inputs, output_handler,
                     per_device_task_context.front());
  } else {
    return RunMulti(flow_func_type, flow_func_index, inputs, output_handler, per_device_task_context);
  }
}

template<typename T>
ge::Status LlmFlowService::RunSingle(
    const LlmWorker &worker,
    std::pair<FlowFuncType, size_t> flow_func_type_and_index,
    const std::vector<ge::Tensor> &inputs,
    const std::function<ge::Status(size_t, const std::vector<ge::Tensor> &, T &)> &output_handler,
    TaskContext<T> &task_context) const {
  std::vector<ge::Tensor> output_tensors;
  const auto device_index = worker.GetDeviceIndex();
  LLM_CHK_STATUS_RET(worker.RunFlowFunc(flow_func_type_and_index.first, flow_func_type_and_index.second, inputs,
                                       output_tensors, task_context.time_points),
                    "Failed to FunFlowFunc, device_index = %zu, logical_device_id = %s", device_index,
                    worker.GetLogicalDeviceId().c_str());
  LLM_CHK_STATUS_RET(output_handler(device_index, output_tensors, task_context.output),
                    "Failed to handle task_context tensors, device_index = %zu, logical_device_id = %s",
                    device_index, worker.GetLogicalDeviceId().c_str());
  return ge::SUCCESS;
}

template<typename T>
ge::Status LlmFlowService::RunMulti(
    FlowFuncType flow_func_type,
    size_t flow_func_index,
    const std::vector<ge::Tensor> &inputs,
    const std::function<ge::Status(size_t, const std::vector<ge::Tensor> &, T &)> &output_handler,
    std::vector<TaskContext<T>> &per_device_task_context) const {
  std::vector<std::future<ge::Status>> futures;
  for (size_t i = 0U; i < workers_.size(); ++i) {
    auto fut = worker_pool_->commit(
        [this, i, flow_func_type, flow_func_index, &inputs, &per_device_task_context, &output_handler]() -> ge::Status {
          return RunSingle(workers_[i], std::make_pair(flow_func_type, flow_func_index), inputs, output_handler,
                           per_device_task_context[i]);
        });
    futures.emplace_back(std::move(fut));
  }
  LLM_CHK_STATUS_RET(GetAndCheckFutures(futures), "Failed to run flow func in all devices");
  return ge::SUCCESS;
}

template<typename T>
ge::Tensor LlmFlowService::BuildTensor(const T &req_info, const size_t req_size) {
  ge::TensorDesc req_tensor_desc(
      ge::Shape(std::vector<int64_t>{static_cast<int64_t>(req_size)}), ge::FORMAT_ND, ge::DT_UINT8);
  ge::Tensor input_tensor(req_tensor_desc);
  input_tensor.SetData(llm::PtrToPtr<const T, const uint8_t>(&req_info), req_size);
  return input_tensor;
}

ge::Status LlmFlowService::ConvertToStatus(size_t device_index,
                                           const std::vector<ge::Tensor> &output_tensors,
                                           ge::Status &status) {
  const auto &output_tensor = output_tensors[0];
  LLM_CHK_BOOL_RET_STATUS(output_tensor.GetSize() >= sizeof(uint32_t), ge::FAILED,
                         "expect at least 4 bytes, but got %zu, device_index = %zu",
                         output_tensor.GetSize(), device_index);
  const int32_t *output_data = llm::PtrToPtr<uint8_t, int32_t>(output_tensor.GetData());
  const auto flow_func_ret = *output_data;
  LLMLOGI("FlowFunc returned: %d, device_index = %zu", flow_func_ret, device_index);
  status = ConvertRetCode(flow_func_ret);
  LLM_CHK_STATUS(status, "FlowFunc returned: %d, device_index = %zu", flow_func_ret, device_index);
  return status;
}

ge::Status LlmFlowService::ConvertToTensor(size_t device_index,
                                           const std::vector<ge::Tensor> &output_tensors,
                                           ge::Tensor &tensor) {
  (void) device_index;
  tensor = output_tensors.front();
  return ge::SUCCESS;
}

ge::Status LlmFlowService::ConvertToTensorSummary(size_t device_index,
                                                  const std::vector<ge::Tensor> &output_tensors,
                                                  int32_t &num_tensors) {
  constexpr size_t kExpectedTensorSize = 8U;
  const auto &output_tensor = output_tensors.front();
  const auto tensor_size = output_tensor.GetSize();
  if (tensor_size < kExpectedTensorSize) {
    ge::Status unused;
    return ConvertToStatus(device_index, output_tensors, unused);
  }
  const auto *output_data = llm::PtrToPtr<uint8_t, int32_t>(output_tensor.GetData());
  num_tensors = output_data[1U];
  LLMLOGI("device_index = %zu, num_tensors = %d", device_index, num_tensors);
  LLM_CHK_BOOL_RET_STATUS(num_tensors > 0,
                         ge::LLM_PARAM_INVALID,
                         "num_tensors (%d) < 0, device_index = %zu",
                         num_tensors, device_index);
  return ge::SUCCESS;
}

ge::Status LlmFlowService::InitializeUdf() const {
  std::vector<ge::Tensor> input_tensors{BuildTensor(0)};
  std::vector<TaskContext<ge::Status>> unused(device_indices_.size());
  auto flow_func_index = flow_func_indices_[static_cast<size_t>(FlowFuncType::kInitialize)];
  // call RunMulti to run in sub-thread
  const auto ret = RunMulti<ge::Status>(FlowFuncType::kInitialize, flow_func_index, input_tensors, LlmFlowService::ConvertToStatus, unused);
  LLM_CHK_STATUS(ret, "[Initialize] RunFlowFunc failed");
  return ret;
}

ge::Status LlmFlowService::Allocate(const CacheDesc &cache_desc,
                                    const std::vector<CacheKey> &cache_keys, Cache &cache) const {
  // 1. Build request tensor
  size_t num_requests_to_index = cache_keys.size();
  auto tensor_size = sizeof(AllocateCacheReqInfo) + sizeof(int64_t) * num_requests_to_index;
  std::vector<uint8_t> tensor_data(tensor_size);
  auto &req_info = *llm::PtrToPtr<uint8_t, AllocateCacheReqInfo>(tensor_data.data());
  req_info.cache_id = cache.cache_id;
  req_info.num_tensors = static_cast<uint64_t>(cache_desc.num_tensors);
  req_info.dtype = static_cast<int32_t>(cache_desc.data_type);
  req_info.seq_len_dim_index = cache_desc.seq_len_dim_index;
  const auto &dims = cache_desc.shape;
  req_info.num_dims = static_cast<uint32_t>(dims.size());
  for (size_t i = 0U; i < dims.size(); ++i) {
    req_info.dims[i] = dims[i];
  }
  req_info.num_requests = num_requests_to_index;
  req_info.is_allocate_blocks = false;
  bool is_prefix = false;
  if (!cache_keys.empty()) {
    req_info.model_id = cache_keys.front().model_id;
    for (size_t i = 0U; i < num_requests_to_index; ++i) {
      is_prefix = (cache_keys[i].prefix_id != UINT64_MAX);
      req_info.req_ids[i] = is_prefix ? cache_keys[i].prefix_id : cache_keys[i].req_id;
      req_info.is_allocate_blocks = cache_keys[i].is_allocate_blocks;
    }
  }
  req_info.is_prefix = is_prefix;
  ge::TensorDesc req_tensor_desc(ge::Shape(std::vector<int64_t>{static_cast<int64_t>(tensor_size)}),
                                 ge::FORMAT_ND, ge::DT_UINT8);
  ge::Tensor input_tensor(req_tensor_desc);
  input_tensor.SetData(std::move(tensor_data));
  // 2. invoke udf
  std::vector<ge::Tensor> input_tensors{input_tensor};
  std::vector<TaskContext<CacheAllocateResult>> per_device_task_context;
  const auto tp_start = std::chrono::steady_clock::now();
  const auto ret = RunFlowFunc<CacheAllocateResult>(FlowFuncType::kAllocate, input_tensors,
                                                    ConvertAllocateResult, per_device_task_context);
  LLM_CHK_STATUS_RET(ret, "[cache_id:%ld][Allocate] RunFlowFunc failed", cache.cache_id);
  const auto &feed_end = per_device_task_context.front().time_points.front();
  const auto &tp_end = is_spmd_ ? per_device_task_context.front().time_points.back() :
                       std::chrono::steady_clock::now();
  LLMLOGI("[LlmPerf] [cache_id:%ld] [Allocate] ended, num_tensors = %u, shape = %s, dtype = %d, "
         "feed_elapsed_us = %ld, total_elapsed_us = %ld", cache.cache_id, cache_desc.num_tensors,
         llm::ToString(dims).c_str(), static_cast<int32_t>(cache_desc.data_type),
         std::chrono::duration_cast<std::chrono::microseconds>(feed_end - tp_start).count(),
         std::chrono::duration_cast<std::chrono::microseconds>(tp_end - tp_start).count());
  for (size_t i = 0U; i < per_device_task_context.size(); ++i) {
    auto &result = per_device_task_context[i].output;
    LLM_CHK_BOOL_RET_STATUS(result.tensor_addrs.size() == static_cast<size_t>(cache_desc.num_tensors), ge::FAILED,
                           "[Check][Result] check tensor addresses failed, expect %u, bot got %zu, device_index = %zu",
                           cache_desc.num_tensors, result.tensor_addrs.size(), device_indices_[i]);
    cache.per_device_tensor_addrs.emplace_back(std::move(result.tensor_addrs));
  }
  return ge::SUCCESS;
}

ge::Status LlmFlowService::ConvertAllocateResult(size_t device_index, const std::vector<ge::Tensor> &output_tensors,
                                                 CacheAllocateResult &result) {
  const auto &output_tensor = output_tensors[0];
  const auto output_size = output_tensor.GetSize();
  LLM_CHECK_GE(output_size, sizeof(int32_t));
  const auto flow_func_ret = *llm::PtrToPtr<uint8_t, int32_t>(output_tensor.GetData());
  auto ret = ConvertRetCode(flow_func_ret);
  LLM_CHK_STATUS_RET(ret, "Allocate cache return code = %u, device_index = %zu", ret, device_index);
  const auto num_items = output_size / sizeof(uint64_t);
  const auto *output_data = llm::PtrToPtr<uint8_t, uint64_t>(output_tensor.GetData());
  for (size_t output_index = 1U; output_index < num_items; ++output_index) {
    result.tensor_addrs.emplace_back(static_cast<uintptr_t>(output_data[output_index]));
  }
  return ge::SUCCESS;
}

ge::Status LlmFlowService::Deallocate(int64_t cache_id) const {
  std::vector<ge::Tensor> input_tensors{BuildTensor(cache_id)};
  std::vector<TaskContext<ge::Status>> unused;
  const auto ret = RunFlowFunc<ge::Status>(FlowFuncType::kDeallocate,
                                           input_tensors,
                                           LlmFlowService::ConvertToStatus,
                                           unused);
  LLM_CHK_STATUS_RET(ret, "[cache_id:%ld][Deallocate] RunFlowFunc failed", cache_id);
  return ge::SUCCESS;
}

ge::Status LlmFlowService::RemoveCacheIndex(const CacheKey &cache_key) const {
  RemoveCacheIndexReqInfo req_info{cache_key.req_id, cache_key.prefix_id, cache_key.model_id};
  std::vector<ge::Tensor> input_tensors{BuildTensor(req_info)};
  std::vector<TaskContext<ge::Status>> unused;
  const auto ret = RunFlowFunc<ge::Status>(FlowFuncType::kRemoveIndex,
                                           input_tensors,
                                           LlmFlowService::ConvertToStatus,
                                           unused);
  LLM_CHK_STATUS_RET(ret, "[RemoveCacheIndex] RunFlowFunc failed cache_key = (%lu, %lu)",
                    cache_key.req_id, cache_key.model_id);
  return ge::SUCCESS;
}

ge::Status LlmFlowService::PullCache(int64_t cache_id,
                                     const CacheKey &cache_key,
                                     const PullCacheParam &pull_cache_param,
                                     uint32_t num_tensors) const {
  LLM_CHECK_LE(pull_cache_param.decoder_blocks.size(), UINT32_MAX);
  LLM_CHECK_LE(pull_cache_param.prompt_blocks.size(), UINT32_MAX);
  LLM_CHECK_LE(pull_cache_param.dst_tensor_indices.size(), UINT32_MAX);
  LLM_CHECK_LE(pull_cache_param.src_tensor_indices.size(), UINT32_MAX);
  const auto num_block_indices = pull_cache_param.decoder_blocks.size() + pull_cache_param.prompt_blocks.size() +
      pull_cache_param.dst_tensor_indices.size() + pull_cache_param.src_tensor_indices.size();
  uint64_t tensor_size = sizeof(PullKvReqInfo) + num_block_indices * sizeof(uint64_t);
  std::vector<uint8_t> tensor_data(tensor_size);
  PullKvReqInfo *req_info = llm::PtrToPtr<uint8_t, PullKvReqInfo>(tensor_data.data());
  req_info->cache_id = cache_id;
  req_info->timeout = sync_kv_wait_time_ * kMillsToMicros;
  LLM_CHK_STATUS_RET(FillPullKvReqInfo(pull_cache_param, cache_key, req_info));
  ge::TensorDesc req_tensor_desc(ge::Shape(std::vector<int64_t>{static_cast<int64_t>(tensor_size)}), ge::FORMAT_ND,
                                 ge::DT_UINT8);
  ge::Tensor input_tensor(req_tensor_desc);
  (void) input_tensor.SetData(std::move(tensor_data));
  std::vector<ge::Tensor> input_tensors{input_tensor};
  std::vector<TaskContext<ge::Status>> per_device_task_context;
  const auto tp_start = std::chrono::steady_clock::now();
  const auto ret = RunFlowFunc<ge::Status>(FlowFuncType::kPull,
                                           input_tensors,
                                           LlmFlowService::ConvertToStatus,
                                           per_device_task_context);
  LLM_CHK_STATUS_RET(ret, "[Pull] Failed to run flow func, %s, cache_id = %ld, "
                         "num_tensors = %u, size = %ld, dst_tensor_indices = %s, src_tensor_indices = %s",
                    LLMUtils::DebugString(cache_key).c_str(), cache_id,
                    num_tensors, pull_cache_param.size, llm::ToString(pull_cache_param.dst_tensor_indices).c_str(),
                    llm::ToString(pull_cache_param.src_tensor_indices).c_str());
  const auto &feed_end = per_device_task_context.front().time_points.front();
  const auto &tp_end = is_spmd_ ? per_device_task_context.front().time_points.back() :
                                std::chrono::steady_clock::now();
  LLMLOGI("[LlmPerf] Request[%lu] [Pull] ended, %s, cache_id = %ld, "
      "num_tensors = %u, size = %ld, dst_tensor_indices = %s, src_tensor_indices = %s, tensor_num_per_layer=%lu, "
      "feed_elapsed_us = %ld, total_elapsed_us = %ld",
      cache_key.req_id, LLMUtils::DebugString(cache_key).c_str(), cache_id,
      num_tensors, pull_cache_param.size, llm::ToString(pull_cache_param.dst_tensor_indices).c_str(),
      llm::ToString(pull_cache_param.src_tensor_indices).c_str(), pull_cache_param.tensor_num_per_layer,
      std::chrono::duration_cast<std::chrono::microseconds>(feed_end - tp_start).count(),
      std::chrono::duration_cast<std::chrono::microseconds>(tp_end - tp_start).count());
  return ge::SUCCESS;
}

ge::Status LlmFlowService::TransferCache(const llm::TransferCacheConfig &transfer_cache_config,
                                         const llm::TransferBlockConfig &transfer_block_config) const {
  uint64_t tensor_size = sizeof(TransferKvReqInfo) + transfer_block_config.src_blocks.size() * sizeof(uint64_t) +
                         transfer_block_config.dst_blocks.size() * sizeof(uint64_t);
  std::vector<uint8_t> tensor_data(tensor_size);
  TransferKvReqInfo *req_info = llm::PtrToPtr<uint8_t, TransferKvReqInfo>(tensor_data.data());
  req_info->cache_id = transfer_cache_config.src_cache_id;
  req_info->src_batch_index = transfer_cache_config.batch_index;
  req_info->src_layer_index = transfer_cache_config.layer_index;
  req_info->dst_cluster_id = transfer_cache_config.cluster_id;
  req_info->timeout = sync_kv_wait_time_ * kMillsToMicros;
  req_info->block_len = transfer_block_config.dst_blocks.empty() ? 0U : transfer_block_config.block_mem_size;
  req_info->block_count = static_cast<uint32_t>(transfer_block_config.dst_blocks.size());
  req_info->prompt_block_count = static_cast<uint32_t>(transfer_block_config.src_blocks.size());
  req_info->dst_cache_id = transfer_cache_config.dst_cache_id;
  req_info->dst_batch_index = transfer_cache_config.dst_batch_index;
  req_info->dst_layer_index = transfer_cache_config.dst_layer_index;
  req_info->tensor_num_per_layer = transfer_cache_config.tensor_num_per_layer;
  // already valid dst_addrs size before.
  if (transfer_cache_config.type == 0U) {
    req_info->dst_key_addr = transfer_cache_config.dst_addrs[0U];
    req_info->dst_value_addr = transfer_cache_config.dst_addrs[1U];
    req_info->tensor_num_per_layer = kDefaultCacheSizePerLayer;
    req_info->dst_cache_id = kDefaultDstCacheId;
  }
  LLMLOGI("Transfer cache with tensor num per layer:%lu.", req_info->tensor_num_per_layer);
  for (uint32_t i = 0U; i < req_info->block_count; ++i) {
    req_info->block_indices[i] = transfer_block_config.dst_blocks[i];
  }
  for (uint32_t i = 0U; i < req_info->prompt_block_count; ++i) {
    req_info->block_indices[req_info->block_count + i] = transfer_block_config.src_blocks[i];
  }
  ge::TensorDesc req_tensor_desc(ge::Shape(std::vector<int64_t>{static_cast<int64_t>(tensor_size)}), ge::FORMAT_ND,
                                 ge::DT_UINT8);
  ge::Tensor input_tensor(req_tensor_desc);
  (void) input_tensor.SetData(std::move(tensor_data));
  std::vector<ge::Tensor> input_tensors{input_tensor};
  std::vector<TaskContext<ge::Status>> per_device_task_context;
  const auto tp_start = std::chrono::steady_clock::now();
  const auto ret = RunFlowFunc<ge::Status>(FlowFuncType::kTransfer, input_tensors, LlmFlowService::ConvertToStatus,
                                           per_device_task_context);
  LLM_CHK_STATUS_RET(ret, "[Transfer] Failed to run flow func");
  const auto &feed_end = per_device_task_context.front().time_points.front();
  const auto &tp_end = is_spmd_ ? per_device_task_context.front().time_points.back() : std::chrono::steady_clock::now();
  LLMLOGI(
      "[LlmPerf] [Transfer] ended, cache_id = %ld, layer_index = %lu, cluster_id = %lu,"
      "feed_elapsed_us = %ld, total_elapsed_us = %ld.",
      transfer_cache_config.src_cache_id, transfer_cache_config.layer_index, transfer_cache_config.cluster_id,
      std::chrono::duration_cast<std::chrono::microseconds>(feed_end - tp_start).count(),
      std::chrono::duration_cast<std::chrono::microseconds>(tp_end - tp_start).count());
  return ge::SUCCESS;
}

ge::Status LlmFlowService::SwapBlocks(const Cache &src, const Cache &dst, const SwapBlockParam &swap_param,
                                      const std::vector<int32_t> &device_ids) {
  LLM_CHK_BOOL_RET_STATUS(device_ids.size() == workers_.size(),
                         ge::FAILED, "Device id num is not equal to workers num.");
  std::vector<std::future<ge::Status>> futures;
  for (size_t i = 0U; i < workers_.size(); ++i) {
    auto fut = worker_pool_->commit(
        [i, &src, &dst, &swap_param, &device_ids]() -> ge::Status {
          llm::SwapImpl swap_impl(device_ids[i]);
          return swap_impl.SwapBlocks(src, dst, swap_param.block_size, swap_param.type, swap_param.block_mapping, i);
        });
    futures.emplace_back(std::move(fut));
  }
  LLM_CHK_STATUS_RET(GetAndCheckFutures(futures), "Failed to run flow func in all devices");
  return ge::SUCCESS;
}

ge::Status LlmFlowService::CopyCache(const CacheEntry &src_cache_entry, const CacheEntry &dst_cache_entry,
                                     const CopyCacheParam &copy_cache_param, const std::vector<int32_t> &device_ids) {
  LLM_CHK_BOOL_RET_STATUS(device_ids.size() == workers_.size(),
                         ge::FAILED, "Device id num is not equal to workers num.");
  std::vector<std::future<ge::Status>> futures;
  auto per_device_addr_num = src_cache_entry.cache_addrs.size() / device_ids.size();
  for (size_t i = 0U; i < workers_.size(); ++i) {
    auto fut =
        worker_pool_->commit([this, i, &src_cache_entry, &dst_cache_entry, &per_device_addr_num, &copy_cache_param,
                              &device_ids]() -> ge::Status {
          if (copy_cache_param.copy_block_infos.empty()) {
            LLM_CHK_STATUS_RET(cache_manager_.CopyCacheForContinuous(src_cache_entry, dst_cache_entry, copy_cache_param,
                                                                    per_device_addr_num, i),
                              "Copy cache in device:%d failed.", device_ids[i]);
          } else {
            LLM_CHK_STATUS_RET(cache_manager_.CopyCacheForBlocks(src_cache_entry, dst_cache_entry, copy_cache_param,
                                                                per_device_addr_num, i),
                              "Copy cache in device:%d failed.", device_ids[i]);
          }
          return ge::SUCCESS;
        });
    futures.emplace_back(std::move(fut));
  }
  LLM_CHK_STATUS_RET(GetAndCheckFutures(futures), "Failed to run flow func in all devices");
  return ge::SUCCESS;
}

ge::Status LlmFlowService::CopyCache(const CopyCacheParam &copy_cache_param) const {
  uint64_t tensor_size = sizeof(CopyReqInfo) + copy_cache_param.copy_block_infos.size() * sizeof(CopyBlockInfo);
  std::vector<uint8_t> tensor_data(tensor_size);
  CopyReqInfo *req_info = llm::PtrToPtr<uint8_t, CopyReqInfo>(tensor_data.data());
  req_info->size = copy_cache_param.size;
  req_info->offset = copy_cache_param.offset;
  req_info->src_cache_id = copy_cache_param.src_cache_id;
  req_info->dst_cache_id = copy_cache_param.dst_cache_id;
  req_info->src_batch_index = copy_cache_param.src_batch_index;
  req_info->dst_batch_index = copy_cache_param.dst_batch_index;
  req_info->copy_block_count = static_cast<uint32_t>(copy_cache_param.copy_block_infos.size());
  for (size_t i = 0U; i < copy_cache_param.copy_block_infos.size(); ++i) {
    req_info->copy_block_infos[i].src_block_index = copy_cache_param.copy_block_infos[i].first;
    req_info->copy_block_infos[i].dst_block_index = copy_cache_param.copy_block_infos[i].second;
  }
  // exclude req_id
  ge::TensorDesc req_tensor_desc(ge::Shape(std::vector<int64_t>{static_cast<int64_t>(tensor_size)}), ge::FORMAT_ND,
                                 ge::DT_UINT8);
  ge::Tensor input_tensor(req_tensor_desc);
  (void) input_tensor.SetData(std::move(tensor_data));
  std::vector<ge::Tensor> input_tensors{input_tensor};
  std::vector<TaskContext<ge::Status>> per_device_task_context;
  const auto tp_start = std::chrono::steady_clock::now();
  const auto ret = RunFlowFunc<ge::Status>(FlowFuncType::kCopy,
                                           input_tensors,
                                           LlmFlowService::ConvertToStatus,
                                           per_device_task_context);
  LLM_CHK_STATUS_RET(ret, "[Copy] Failed to run flow func");
  const auto feed_end = per_device_task_context.front().time_points.front();
  const auto fetch_end = per_device_task_context.front().time_points.back();
  LLMLOGI("[LlmPerf] Request[%s] [Copy][%ld->%ld] ended, size = %ld, feed_elapsed_us = %ld, total_elapsed_us = %ld",
         copy_cache_param.req_id == UINT64_MAX ? "-1" : std::to_string(copy_cache_param.req_id).c_str(),
         copy_cache_param.src_cache_id,
         copy_cache_param.dst_cache_id,
         copy_cache_param.size,
         std::chrono::duration_cast<std::chrono::microseconds>(feed_end - tp_start).count(),
         std::chrono::duration_cast<std::chrono::microseconds>(fetch_end - tp_start).count());
  return ge::SUCCESS;
}

ge::Status LlmFlowService::GetCacheTensors(int64_t cache_id,
                                           std::vector<ge::Tensor> &outputs,
                                           int32_t tensor_index) const {
  GetCacheTensorsReqInfo req_info;
  req_info.cache_id = cache_id;
  req_info.tensor_index = tensor_index;
  std::vector<ge::Tensor> input_tensors{BuildTensor(req_info)};
  std::vector<TaskContext<int32_t>> per_device_task_context;
  // 包含多次操作，防止错乱
  std::lock_guard<std::mutex> lk(get_cache_mu_);
  auto ret = RunFlowFunc<int32_t>(FlowFuncType::kGetTensorSummary,
                                  input_tensors,
                                  LlmFlowService::ConvertToTensorSummary,
                                  per_device_task_context);
  LLM_CHK_STATUS_RET(ret, "Failed to get cache tensor summary");
  const auto num_tensors = per_device_task_context.front().output;
  if (!is_spmd_) {
    std::vector<int32_t> num_tensors_each_device;
    for (const auto &task_context : per_device_task_context) {
      num_tensors_each_device.emplace_back(task_context.output);
    }
    std::set<int32_t> distinct_num_tensors(num_tensors_each_device.cbegin(), num_tensors_each_device.cend());
    LLM_CHK_BOOL_RET_STATUS(distinct_num_tensors.size() == 1U,
                           ge::FAILED,
                           "num_tensors differs between devices: %s",
                           llm::ToString(num_tensors_each_device).c_str());
  }
  LLM_CHK_BOOL_RET_STATUS(num_tensors == 1,
                         ge::FAILED,
                         "multiple tensor is not supported yet, num_tensors = %d",
                         num_tensors);
  std::vector<TaskContext<ge::Tensor>> per_device_get_tensor_task_context;
  ret = RunFlowFunc<ge::Tensor>(FlowFuncType::kGetTensor,
                                {},
                                LlmFlowService::ConvertToTensor,
                                per_device_get_tensor_task_context);
  LLM_CHK_STATUS_RET(ret, "Failed to get cache tensor");
  for (auto &task_context : per_device_get_tensor_task_context) {
    outputs.emplace_back(std::move(task_context.output));
  }
  return ge::SUCCESS;
}

ge::Status LlmFlowService::CheckLinkStatus(uint64_t remote_cluster_id) const {
  uint64_t tensor_size = sizeof(CheckLinkReqInfo);
  std::vector<uint8_t> tensor_data(tensor_size);
  CheckLinkReqInfo *req_info = llm::PtrToPtr<uint8_t, CheckLinkReqInfo>(tensor_data.data());
  req_info->dst_cluster_id = remote_cluster_id;
  req_info->timeout = sync_kv_wait_time_ * kMillsToMicros;
  ge::TensorDesc req_tensor_desc(ge::Shape(std::vector<int64_t>{static_cast<int64_t>(tensor_size)}), ge::FORMAT_ND,
                                 ge::DT_UINT8);
  ge::Tensor input_tensor(req_tensor_desc);
  (void) input_tensor.SetData(std::move(tensor_data));
  std::vector<ge::Tensor> input_tensors{input_tensor};
  std::vector<TaskContext<ge::Status>> unused;
  const auto ret =
      RunFlowFunc<ge::Status>(FlowFuncType::kCheckLink, input_tensors, LlmFlowService::ConvertToStatus, unused);
  LLM_CHK_STATUS_RET(ret, "[remote_cluster_id:%lu][CheckLinkStatus] failed", remote_cluster_id);
  return ge::SUCCESS;
}

ge::Status LlmFlowService::LinkClusters(const std::vector<ClusterInfo> &clusters, std::vector<ge::Status> &rets,
                                        const int32_t timeout) {
  const auto ret = main_pool_.commit([this, &clusters, &rets, timeout]() -> ge::Status {
    return link_manager_.LinkClusters(clusters, rets, timeout);
  }).get();
  LLM_CHK_STATUS_RET(ret, "link clusters failed");
  return ret;
}

ge::Status LlmFlowService::UnlinkClusters(const std::vector<ClusterInfo> &clusters, std::vector<ge::Status> &rets,
                                          const int32_t timeout, bool force_flag) {
  const auto ret = main_pool_.commit([this, &clusters, &rets, timeout, force_flag]() -> ge::Status {
    const auto unlink_type = (role_ == kPrompt) ? LinkOperator::ServerUnlink : LinkOperator::ClientUnlink;
    return link_manager_.UnlinkClusters(clusters, unlink_type, rets, timeout, force_flag);
  }).get();
  LLM_CHK_STATUS_RET(ret, "unlink clusters failed");
  return ret;
}

ge::Status LlmFlowService::SwitchRole(const std::string &role, const std::map<std::string, std::string> &options) {
  std::map<std::string, int32_t> role_to_role_id {
      {kPrompt, 0}, {kDecoder, 1}, {kMix, 2},
  };
  LLM_CHK_BOOL_RET_STATUS(role_to_role_id.count(role) > 0U, ge::LLM_PARAM_INVALID, "Invalid role: %s", role.c_str());
  SwitchRoleReqInfo req_info{};
  req_info.role_type = role_to_role_id[role];
  if (role == kPrompt) {
    const auto it_ip = options.find(kLlmOptionListenIp);
    const auto it_port = options.find(kLlmOptionListenPort);
    LLM_CHK_BOOL_RET_STATUS(it_ip != options.cend(),
                           ge::LLM_PARAM_INVALID,
                           "required option (%s) not found while switching to Prompt ", kLlmOptionListenIp);
    LLM_CHK_BOOL_RET_STATUS(it_port != options.cend(),
                           ge::LLM_PARAM_INVALID,
                           "required option (%s) not found while switching to Prompt ", kLlmOptionListenPort);
    LLM_CHK_STATUS_RET(LLMUtils::ToNumber(it_ip->second, req_info.prompt_listen_ip),
                      "Option %s is invalid: [%s]",
                      kLlmOptionListenIp,
                      it_ip->second.c_str());
    LLM_CHK_STATUS_RET(LLMUtils::ToNumber(it_port->second, req_info.prompt_listen_port),
                      "Option %s is invalid: [%s]",
                      kLlmOptionListenPort,
                      it_port->second.c_str());
  }
  std::vector<ge::Tensor> input_tensors{BuildTensor(req_info)};
  std::vector<TaskContext<ge::Status>> unused;
  const auto ret = RunFlowFunc<ge::Status>(FlowFuncType::kSwitchRole,
                                           input_tensors,
                                           LlmFlowService::ConvertToStatus,
                                           unused);
  LLM_CHK_STATUS_RET(ret, "[SwitchRole] RunFlowFunc failed, role = %s, listen_ip = %u, listen_port = %u",
                    role.c_str(), req_info.prompt_listen_ip, req_info.prompt_listen_port);
  LLMLOGI("[SwitchRole] RunFlowFunc success, role = %s, listen_ip = %u, listen_port = %u",
         role.c_str(), req_info.prompt_listen_ip, req_info.prompt_listen_port);
  role_ = role;
  SetLinkManagerIo(role);
  return ge::SUCCESS;
}

ge::Status SpmdLinkManager::FeedInputs(const uint32_t graph_id,
                                       const std::vector<uint32_t> &indices,
                                       const std::vector<ge::Tensor> &inputs,
                                       const int32_t timeout,
                                       bool is_link) {
  auto &trans_id = is_link ? link_transaction_id_ : unlink_transaction_id_;
  for (size_t i = 0U; i < all_ge_apis_.size(); ++i) {
    ge::DataFlowInfo data_flow_info;
    LLMLOGI("Feed transaction id:%lu.", trans_id);
    data_flow_info.SetTransactionId(trans_id);
    const auto ret = all_ge_apis_[i]->FeedDataFlowGraph(graph_id, indices, inputs, data_flow_info, timeout);
    LLM_CHK_BOOL_RET_STATUS(ret != llm::ConvertAclError2Ge(ACL_ERROR_GE_MODEL_EXECUTE_TIMEOUT), ge::LLM_WAIT_PROC_TIMEOUT,
                           "Feed timeout.");
    LLM_CHK_STATUS_RET(ret, "Failed to feed input, device_index = %zu", i);
  }
  trans_id++;
  return ge::SUCCESS;
}

ge::Status SpmdLinkManager::FetchOutputs(const uint32_t graph_id,
                                         const std::vector<uint32_t> &indices,
                                         std::vector<ge::Tensor> &outputs,
                                         int64_t timeout,
                                         uint64_t transaction_id) {
  auto last_time = std::chrono::steady_clock::now();
  std::vector<ge::Tensor> output_tensors;
  for (size_t i = 0U; i < all_ge_apis_.size(); ++i) {
    LLMLOGI("Fetch outputs of device index:%zu", i);
    auto fetch_timeout = timeout;
    while (true) {
      ge::DataFlowInfo flow_info;
      std::vector<ge::Tensor> one_outputs;
      auto timeout_in_millis = static_cast<int32_t>(fetch_timeout / kMillsToMicros);
      LLM_CHK_BOOL_RET_STATUS(timeout_in_millis >= 1, ge::LLM_WAIT_PROC_TIMEOUT, "Fetch timeout.");
      auto ret = all_ge_apis_[i]->FetchDataFlowGraph(graph_id, indices, one_outputs, flow_info, timeout_in_millis);
      LLM_CHK_BOOL_RET_STATUS(ret != llm::ConvertAclError2Ge(ACL_ERROR_GE_MODEL_EXECUTE_TIMEOUT),
                             ge::LLM_WAIT_PROC_TIMEOUT, "Fetch timeout.");
      LLM_CHK_STATUS_RET(ret, "Failed to fetch input, device_index = %zu", i);
      LLM_CHK_BOOL_RET_STATUS(!one_outputs.empty(), ge::FAILED, "Fetch output is empty.");
      if (flow_info.GetTransactionId() == transaction_id) {
        output_tensors.emplace_back(std::move(one_outputs.front()));
        break;
      }
      LLMEVENT("Ignore invalid transaction id:%lu for index:%u", flow_info.GetTransactionId(), indices.front());
      auto cur_time = std::chrono::steady_clock::now();
      auto fetch_time_cost = std::chrono::duration_cast<std::chrono::microseconds>(cur_time - last_time).count();
      last_time = cur_time;
      // current fetch cost time already over time limit.
      LLM_CHK_BOOL_RET_STATUS(fetch_time_cost < fetch_timeout, ge::LLM_WAIT_PROC_TIMEOUT,
                             "Fetch timeout, cost:%ld us, left:%ld us.", fetch_time_cost, fetch_timeout);
      fetch_timeout -= fetch_time_cost;
    }
  }
  outputs = std::move(output_tensors);
  return ge::SUCCESS;
}

void SpmdLinkManager::SetAllGeApis(const std::vector<GeApi *> &all_ge_apis) {
  all_ge_apis_ = all_ge_apis;
}
}  // namespace llm
