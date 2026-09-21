/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_LLM_ENGINE_COMMON_LLM_FLOW_SERVICE_H
#define AIR_RUNTIME_LLM_ENGINE_COMMON_LLM_FLOW_SERVICE_H

#include <memory>
#include "common/llm_thread_pool.h"
#include "common/llm_ge_api.h"
#include "common/flow_graph_manager.h"
#include "common/link_manager.h"
#include "common/llm_utils.h"
#include "common/cache_manager.h"
#include "acl/acl.h"

namespace llm {
template<typename T>
struct TaskContext {
  T output;
  std::vector<std::chrono::steady_clock::time_point> time_points;
};

enum class FlowFuncType : size_t {
  kUpdateLink = 0,
  kUnlink = 1,
  kAllocate = 2,
  kDeallocate = 3,
  kGetTensorSummary = 4,
  kGetTensor = 5,
  kCopy = 6,
  kRemoveIndex = 7,
  kPull = 8,
  kSwitchRole = 9,
  kTransfer = 10,
  kInitialize = 11,
  kCheckLink = 12,
  kMax = 13,
};

class LlmWorker {
 public:
  explicit LlmWorker(uint64_t cluster_id, size_t device_index, std::string logical_device_id)
      : cluster_id_(cluster_id), device_index_(device_index), logical_device_id_(std::move(logical_device_id)) {}

  ge::Status Initialize(const std::map<ge::AscendString, ge::AscendString> &options);

  void Finalize();

  ge::Status LoadFlowFuncs(const FlowNodeDef &flow_node_def,
                           const std::map<ge::AscendString, ge::AscendString> &options,
                           const std::vector<int32_t> &device_ids);

  ge::Status RunFlowFunc(FlowFuncType flow_func_type,
                         size_t flow_func_index,
                         const std::vector<ge::Tensor> &inputs,
                         std::vector<ge::Tensor> &outputs,
                         std::vector<std::chrono::steady_clock::time_point> &time_points) const;

  size_t GetDeviceIndex() const;
  const std::string &GetLogicalDeviceId() const;
  GeApi *GetGeApi() const;
  void SetTimeout(int32_t timeout);

 private:
  ge::Status FeedInputData(const FlowFuncDef &flow_func_def, const std::vector<ge::Tensor> &inputs,
                           uint64_t transaction_id) const;
  mutable std::vector<std::mutex> mutexes_;
  mutable std::vector<uint64_t> transaction_ids_;
  uint64_t cluster_id_;
  size_t device_index_;
  std::string logical_device_id_;
  uint32_t graph_id_ = 0U;
  std::unique_ptr<GeApi> ge_api_{};
  FlowNodeDef flow_node_def_{};
  int32_t timeout_ = kDefaultSyncKvWaitTime;
};

class SpmdLinkManager : public LinkManager {
 public:
  void SetAllGeApis(const std::vector<GeApi *> &all_ge_apis);

 protected:
  ge::Status FeedInputs(const uint32_t graph_id,
                        const std::vector<uint32_t> &indices,
                        const std::vector<ge::Tensor> &inputs,
                        const int32_t timeout,
                        bool is_link) override;
  ge::Status FetchOutputs(const uint32_t graph_id,
                          const std::vector<uint32_t> &indices,
                          std::vector<ge::Tensor> &outputs,
                          int64_t timeout,
                          uint64_t transaction_id) override;
 private:
  std::vector<GeApi *> all_ge_apis_;
};

struct CacheAllocateResult {
  std::vector<uintptr_t> tensor_addrs;
};

class LlmFlowService {
 public:
  LlmFlowService(const std::string &role, bool enable_switch_role, uint64_t cluster_id);
  ~LlmFlowService() = default;

  ge::Status Initialize(const std::map<ge::AscendString, ge::AscendString> &options,
                        const std::vector<int32_t> &device_ids);
  void Finalize();

  ge::Status InitializeUdf() const;
  ge::Status Allocate(const CacheDesc &cache_desc,
                      const std::vector<CacheKey> &cache_keys,
                      Cache &cache) const;
  ge::Status Deallocate(int64_t cache_id) const;
  ge::Status RemoveCacheIndex(const CacheKey &cache_key) const;
  ge::Status PullCache(int64_t cache_id,
                       const CacheKey &cache_key,
                       const PullCacheParam &pull_cache_param,
                       uint32_t num_tensors) const;
  ge::Status TransferCache(const TransferCacheConfig &transfer_cache_config,
                           const TransferBlockConfig &transfer_block_config) const;
  ge::Status SwapBlocks(const Cache &src, const Cache &dst, const SwapBlockParam &swap_param,
                        const std::vector<int32_t> &device_ids);
  ge::Status CopyCache(const CacheEntry &src_cache_entry,
                       const CacheEntry &dst_cache_entry,
                       const CopyCacheParam &copy_cache_param,
                       const std::vector<int32_t> &device_ids);
  ge::Status CopyCache(const CopyCacheParam &copy_cache_param) const;
  ge::Status GetCacheTensors(int64_t cache_id,
                             std::vector<ge::Tensor> &outputs,
                             int32_t tensor_index) const;
  ge::Status CheckLinkStatus(uint64_t remote_cluster_id) const;
  ge::Status LinkClusters(const std::vector<ClusterInfo> &clusters, std::vector<ge::Status> &rets,
                          const int32_t timeout);

  ge::Status UnlinkClusters(const std::vector<ClusterInfo> &clusters, std::vector<ge::Status> &rets,
                            const int32_t timeout, bool force_flag);

  ge::Status SwitchRole(const std::string &role, const std::map<std::string, std::string> &options);

 private:
  FlowNodeDef BuildFlowNodeDef(const std::string &role);
  ge::Status LoadDataFlow(const FlowNodeDef &flow_node_def,
                          const std::map<ge::AscendString, ge::AscendString> &options);
  void SetFunctionPpInitParam(const FlowNodeDef &flow_node_def, ge::dflow::FunctionPp &pp) const;
  static ge::Status GetAndCheckFutures(std::vector<std::future<ge::Status>> &futures);
  void SetLinkManagerIo(const std::string &role);

  template<typename T>
  ge::Status RunSingle(const LlmWorker &worker,
                       std::pair<FlowFuncType, size_t> flow_func_type_and_index,
                       const std::vector<ge::Tensor> &inputs,
                       const std::function<ge::Status(size_t, const std::vector<ge::Tensor> &, T &)> &output_handler,
                       TaskContext<T> &task_context) const;

  template<typename T>
  ge::Status RunMulti(FlowFuncType flow_func_type,
                      size_t flow_func_index,
                      const std::vector<ge::Tensor> &inputs,
                      const std::function<ge::Status(size_t, const std::vector<ge::Tensor> &, T &)> &output_handler,
                      std::vector<TaskContext<T>> &per_device_task_context) const;

  template<typename T>
  ge::Status RunFlowFunc(FlowFuncType flow_func_type,
                         const std::vector<ge::Tensor> &inputs,
                         const std::function<ge::Status(size_t, const std::vector<ge::Tensor> &, T &)> &output_handler,
                         std::vector<TaskContext<T>> &per_device_task_context) const;

  template<typename T>
  static ge::Tensor BuildTensor(const T &req_info, const size_t req_size = sizeof(T));
  static ge::Status ConvertToStatus(size_t device_index,
                                    const std::vector<ge::Tensor> &output_tensors,
                                    ge::Status &status);
  static ge::Status ConvertAllocateResult(size_t device_index,
                                          const std::vector<ge::Tensor> &output_tensors,
                                          CacheAllocateResult &result);
  static ge::Status ConvertToTensor(size_t device_index,
                                    const std::vector<ge::Tensor> &output_tensors,
                                    ge::Tensor &tensor);
  static ge::Status ConvertToTensorSummary(size_t device_index,
                                           const std::vector<ge::Tensor> &output_tensors,
                                           int32_t &num_tensors);

  std::string role_;
  bool enable_switch_role_ = false;
  bool is_spmd_ = false;
  uint64_t cluster_id_;
  DeployInfo deploy_info_;
  std::vector<size_t> device_indices_;
  std::map<int32_t, int32_t> device_id_to_device_index_;
  std::unique_ptr<llm::LLMThreadPool> worker_pool_;
  llm::LLMThreadPool main_pool_{"ge_llm_main", 1U};
  std::vector<LlmWorker> workers_;
  std::vector<GeApi *> all_ge_apis_;
  std::vector<FlowFuncDef> flow_func_defs_;
  std::array<size_t, static_cast<size_t>(FlowFuncType::kMax)> flow_func_indices_;
  SpmdLinkManager link_manager_;
  int32_t sync_kv_wait_time_ = 1000;
  mutable std::mutex get_cache_mu_;
  llm::CacheManager cache_manager_;
  std::vector<int32_t> device_ids_;
};
}  // namespace llm
#endif  // AIR_RUNTIME_LLM_ENGINE_COMMON_LLM_FLOW_SERVICE_H
