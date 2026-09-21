/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cache_engine_stubs.h"
#include "llm_utils.h"
#include "common/llm_log.h"
#include "common/llm_checker.h"
#include "llm_test_helper.h"

namespace ge {
Status GEInitialize(const std::map<std::string, std::string> &options) {
  LLMLOGI("Stub GEInitialize");
  return SUCCESS;
}

Status GEInitialize(const std::map<AscendString, AscendString> &options) {
  LLMLOGI("Stub GEInitialize");
  return SUCCESS;
}

Status GEFinalize() {
  LLMLOGI("Stub GEFinalize");
  return SUCCESS;
}
} // namespace ge

namespace llm {
namespace {
struct FlowMsg {
  int32_t* data;
  FlowMsg(int64_t size) : data(new int32_t[size]) {}
  ~FlowMsg() { delete[] data; }
};

using CacheIndex = std::pair<int64_t, uint32_t>;

struct StubCacheEntry {
  size_t batch_num = 1;
  size_t ref_count = 1;
  std::vector<std::shared_ptr<FlowMsg>> tensors;
  // Deallocate时保底释放CacheIndex
  std::vector<CacheIndex> cache_indices;
  std::map<int64_t, std::pair<int64_t, size_t>> id_to_offset_and_size;
};

class MyCacheManager {
 public:
  ge::Status AllocateCache(const AllocateCacheReqInfo &req_info,
                           std::back_insert_iterator<std::vector<uint64_t>> address_inserter);

  ge::Status DeallocateCache(int64_t cache_id);

  ge::Status ReleaseCache(const CacheIndex &cache_index, bool is_prefix);

  const StubCacheEntry *GetStubCacheEntry(int64_t cache_id) {
    const auto it = cache_id_to_entry_.find(cache_id);
    if (it != cache_id_to_entry_.cend()) {
      return &it->second;
    }
    return nullptr;
  }

  const StubCacheEntry *GetStubCacheEntry(CacheIndex cache_index, bool is_prefix);

 private:
  static std::shared_ptr<FlowMsg> DoAllocate(const std::vector<int64_t> &shape, ge::DataType data_type);
  std::mutex mu_;
  std::map<int64_t, StubCacheEntry> cache_id_to_entry_;
  std::map<CacheIndex, int64_t> cache_index_to_id_;
  std::map<CacheIndex, int64_t> prefix_index_to_id_;
};
}  // namespace

class CacheEngineGeApi::MockFlowNode {
 public:
  ge::Status AllocateCache(const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs);

  ge::Status DeallocateCache(const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs);

  ge::Status GetCache(const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs);

  ge::Status PullCache(const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs);

  ge::Status CopyCache(const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs);

 private:
  MyCacheManager cache_manager_;
};

ge::Status MyCacheManager::AllocateCache(const AllocateCacheReqInfo &req_info,
                                       std::back_insert_iterator<std::vector<uint64_t>> address_inserter) {
  // do allocate
  std::vector<int64_t> shape(req_info.dims, req_info.dims + req_info.num_dims);
  auto dtype = static_cast<ge::DataType>(req_info.dtype);
  StubCacheEntry cache_entry;
  cache_entry.tensors.resize(req_info.num_tensors);
  for (size_t i = 0U; i < req_info.num_tensors; ++i) {
    auto cache_tensor = DoAllocate(shape, dtype);
    address_inserter = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(cache_tensor->data));
    cache_entry.tensors[i] = std::move(cache_tensor);
  }
  shape[0U] = 1;  // single batch
  int64_t tensor_size_per_batch = -1;
  llm::LLMUtils::CalcTensorMemSize(shape, dtype, tensor_size_per_batch);
  bool valid = (req_info.num_requests == 1LU);
  auto &indexToId = req_info.is_prefix ? prefix_index_to_id_ : cache_index_to_id_;
  for (size_t i = 0; i < req_info.num_requests; ++i) {
    if (!valid && (req_info.req_ids[i] == UINT64_MAX)) {
      LLMLOGI("AllocateCache ignore invalid req id");
      continue;
    }
    LLMLOGI("AllocateCache is prefix:%d, req id:%lu.", req_info.is_prefix, req_info.req_ids[i]);
    auto cache_index = std::make_pair(req_info.req_ids[i], req_info.model_id);
    LLM_ASSERT_TRUE(indexToId.emplace(cache_index, req_info.cache_id).second);
    cache_entry.id_to_offset_and_size[req_info.req_ids[i]] =
        std::make_pair(tensor_size_per_batch * i, tensor_size_per_batch);  // TODO
    cache_entry.cache_indices.emplace_back(std::move(cache_index));
  }
  cache_id_to_entry_[req_info.cache_id] = std::move(cache_entry);
  return ge::SUCCESS;
}

ge::Status MyCacheManager::DeallocateCache(int64_t cache_id) {
  const auto it = cache_id_to_entry_.find(cache_id);
  if (it == cache_id_to_entry_.cend()) {
    LLMLOGD("already deallocated");
    return ge::SUCCESS;
  }
  auto &cache_entry = it->second;
  cache_entry.ref_count -= 1;
  if ((cache_entry.ref_count == 0) && cache_entry.id_to_offset_and_size.empty()) {
    (void) cache_id_to_entry_.erase(it);
  }
  return ge::SUCCESS;
}

const StubCacheEntry *MyCacheManager::GetStubCacheEntry(CacheIndex cache_index, bool is_prefix) {
  auto &indexToId = is_prefix ? prefix_index_to_id_ : cache_index_to_id_;
  const auto it = indexToId.find(cache_index);
  if (it == indexToId.cend()) {
    return nullptr;
  }
  return GetStubCacheEntry(it->second);
}

std::shared_ptr<FlowMsg> MyCacheManager::DoAllocate(const std::vector<int64_t> &shape, ge::DataType data_type) {
  int64_t tensor_size_per_batch = -1;
  llm::LLMUtils::CalcTensorMemSize(shape, data_type, tensor_size_per_batch);
  auto flow_msg = std::make_shared<FlowMsg>(tensor_size_per_batch / sizeof(int32_t));
  return flow_msg;
}

ge::Status MyCacheManager::ReleaseCache(const CacheIndex &cache_index, bool is_prefix) {
  auto &indexToId = is_prefix ? prefix_index_to_id_ : cache_index_to_id_;
  const auto cache_id_it = indexToId.find(cache_index);
  LLM_CHK_BOOL_RET_STATUS(cache_id_it != indexToId.cend(), ge::LLM_PARAM_INVALID, "cache index not exist");
  const auto cache_id = cache_id_it->second;
  // 保证cache index指向的cache entry必然存在
  auto &cache_entry = cache_id_to_entry_.at(cache_id);
  (void) cache_entry.id_to_offset_and_size.erase(cache_index.first);
  if (cache_entry.id_to_offset_and_size.empty() && (cache_entry.ref_count == 0)) {
    DeallocateCache(cache_id);
  } else {
    LLMLOGI("ReleaseCache is req id:%lu.", cache_id_it->first.first);
    indexToId.erase(cache_id_it);
  }
  return ge::SUCCESS;
}

ge::Status CacheEngineGeApi::MockFlowNode::AllocateCache(const std::vector<ge::Tensor> &inputs,
                                                         std::vector<ge::Tensor> &outputs) {
  // response: Status(u64) | tensor_addr * N
  std::vector<uint64_t> output_data(1);  // status code
  const auto &tensor = inputs[0];
  LLM_ASSERT_TRUE(tensor.GetSize() >= sizeof(AllocateCacheReqInfo));
  const auto *req_info = reinterpret_cast<const AllocateCacheReqInfo *>(tensor.GetData());
  LLM_ASSERT_TRUE(tensor.GetSize() == (sizeof(AllocateCacheReqInfo) + (sizeof(int64_t) * req_info->num_requests)),
                 "Expect tensor size = %zu, but only %zu",
                 sizeof(AllocateCacheReqInfo) + (sizeof(int64_t) * req_info->num_requests),
                 tensor.GetSize());
  // do allocate
  output_data.reserve(output_data.size() + req_info->num_tensors);
  auto ret = cache_manager_.AllocateCache(*req_info, std::back_inserter(output_data));
  output_data[0] = static_cast<uint64_t>(ret);

  ge::TensorDesc output_desc(ge::Shape({static_cast<int64_t>(output_data.size())}), ge::FORMAT_ND, ge::DT_UINT64);
  ge::Tensor output_tensor(output_desc);
  output_tensor.SetData(reinterpret_cast<uint8_t *>(output_data.data()), output_data.size() * sizeof(uint64_t));
  outputs.emplace_back(output_tensor);
  return ge::SUCCESS;
}

ge::Status CacheEngineGeApi::MockFlowNode::DeallocateCache(const std::vector<ge::Tensor> &inputs,
                                                           std::vector<ge::Tensor> &outputs) {
  const auto &tensor = inputs[0];
  LLM_ASSERT_TRUE(tensor.GetSize() == sizeof(int64_t));
  const auto cache_id = *reinterpret_cast<const int64_t *>(tensor.GetData());
  auto ret = cache_manager_.DeallocateCache(cache_id);
  ge::TensorDesc output_desc(ge::Shape({1L}), ge::FORMAT_ND, ge::DT_UINT32);
  ge::Tensor output_tensor(output_desc);
  output_tensor.SetData(reinterpret_cast<uint8_t *>(&ret), sizeof(ret));
  outputs.emplace_back(output_tensor);
  return ge::SUCCESS;
}

ge::Status CacheEngineGeApi::MockFlowNode::GetCache(const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs) {
  {
    ge::TensorDesc output_desc(ge::Shape({2L}), ge::FORMAT_ND, ge::DT_UINT32);
    ge::Tensor output_tensor(output_desc);
    std::vector<uint8_t> output_data(8);
    reinterpret_cast<int32_t *>(output_data.data())[0] = ge::SUCCESS;
    reinterpret_cast<int32_t *>(output_data.data())[1] = 1;
    output_tensor.SetData(output_data);
    outputs.emplace_back(output_tensor);
    outputs.insert(outputs.cbegin(), output_tensor);
  }
  return ge::SUCCESS;
}

ge::Status CacheEngineGeApi::MockFlowNode::PullCache(const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs) {
  const auto &tensor = inputs[0];
  LLM_ASSERT_TRUE(tensor.GetSize() == sizeof(PullKvReqInfo));
  const auto *req_info = reinterpret_cast<const PullKvReqInfo *>(tensor.GetData());
  const StubCacheEntry *src_cache_entry = nullptr;
  bool is_prefix = (req_info->prefix_id != UINT64_MAX);
  auto real_req_id = (is_prefix ? req_info->prefix_id : req_info->req_id);
  CacheIndex cache_index;
  if (req_info->prompt_cache_id != -1) {
    src_cache_entry = cache_manager_.GetStubCacheEntry(req_info->cache_id);
  } else {
    cache_index = std::make_pair(real_req_id, req_info->model_id);
    LLMLOGI("PullCache is prefix:%d, req id:%lu, prefix id:%lu.", is_prefix, req_info->req_id, req_info->prefix_id);
    src_cache_entry = cache_manager_.GetStubCacheEntry(cache_index, is_prefix);
  }

  ge::TensorDesc output_desc(ge::Shape({1L}), ge::FORMAT_ND, ge::DT_UINT32);
  ge::Tensor output_tensor(output_desc);
  if (src_cache_entry == nullptr) {
    LLMLOGE(ge::LLM_KV_CACHE_NOT_EXIST, "src cache entry not found");
    auto ret = 2;
    output_tensor.SetData(reinterpret_cast<uint8_t *>(&ret), sizeof(ret));
    outputs.emplace_back(output_tensor);
    return ge::LLM_KV_CACHE_NOT_EXIST;
  }

  auto dst_cache_entry = cache_manager_.GetStubCacheEntry(req_info->cache_id);
  if (dst_cache_entry == nullptr) {
    LLMLOGE(ge::LLM_PARAM_INVALID, "dst cache entry not found");
    auto ret = 2;
    output_tensor.SetData(reinterpret_cast<uint8_t *>(&ret), sizeof(ret));
    outputs.emplace_back(output_tensor);
    return ge::LLM_KV_CACHE_NOT_EXIST;
  }

  if (dst_cache_entry->tensors.size() != src_cache_entry->tensors.size()) {
    LLMLOGE(ge::LLM_PARAM_INVALID, "tensor number mismatches");
    auto ret = ge::LLM_PARAM_INVALID;
    output_tensor.SetData(reinterpret_cast<uint8_t *>(&ret), sizeof(ret));
    outputs.emplace_back(output_tensor);
    return ge::LLM_PARAM_INVALID;
  }

  std::pair<int64_t, size_t> src_offset_and_size;
  if (req_info->prompt_cache_id == -1) {
    src_offset_and_size = src_cache_entry->id_to_offset_and_size.at(real_req_id);
  } else {
    src_offset_and_size = std::make_pair(0, 1);
  }
  const auto copy_size = src_offset_and_size.second;
  for (size_t i = 0U; i < src_cache_entry->tensors.size(); ++i) {
    const auto &src_tensor = src_cache_entry->tensors[i];
    const auto cpy_ret = memcpy_s(reinterpret_cast<uint8_t *>(src_tensor->data) + src_offset_and_size.first,
                                  src_offset_and_size.second,
                                  dst_cache_entry->tensors[i]->data,
                                  copy_size);
    LLM_ASSERT_TRUE(cpy_ret == EOK);
  }

  // prompt发送成功后删除
  if (req_info->prompt_cache_id == -1) {
    (void) cache_manager_.ReleaseCache(cache_index, is_prefix);
  }

  auto ret = ge::SUCCESS;
  output_tensor.SetData(reinterpret_cast<uint8_t *>(&ret), sizeof(ret));
  outputs.emplace_back(output_tensor);
  return ge::SUCCESS;
}

ge::Status CacheEngineGeApi::MockFlowNode::CopyCache(const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs) {
  ge::TensorDesc output_desc(ge::Shape({1L}), ge::FORMAT_ND, ge::DT_UINT32);
  ge::Tensor output_tensor(output_desc);
  const auto &tensor = inputs[0];
  const auto *req_info = reinterpret_cast<const CopyCacheParam *>(tensor.GetData());
  auto src_cache_entry = cache_manager_.GetStubCacheEntry(req_info->src_cache_id);
  auto dst_cache_entry = cache_manager_.GetStubCacheEntry(req_info->dst_cache_id);
  if (dst_cache_entry == nullptr || src_cache_entry == nullptr) {
    LLMLOGE(ge::LLM_PARAM_INVALID, "src or dst cache entry not found");
    auto ret = ge::LLM_PARAM_INVALID;
    output_tensor.SetData(reinterpret_cast<uint8_t *>(&ret), sizeof(ret));
    outputs.emplace_back(output_tensor);
    return ge::LLM_PARAM_INVALID;
  }

  if (dst_cache_entry->tensors.size() != src_cache_entry->tensors.size()) {
    LLMLOGE(ge::LLM_PARAM_INVALID, "tensor number mismatches");
    auto ret = ge::LLM_PARAM_INVALID;
    output_tensor.SetData(reinterpret_cast<uint8_t *>(&ret), sizeof(ret));
    outputs.emplace_back(output_tensor);
    return ge::LLM_PARAM_INVALID;
  }
  auto ret = ge::SUCCESS;
  output_tensor.SetData(reinterpret_cast<uint8_t *>(&ret), sizeof(ret));
  outputs.emplace_back(output_tensor);
  return ge::SUCCESS;
}

ge::Status CacheEngineGeApi::FeedDataFlowGraph(uint32_t graph_id,
                                               const std::vector<uint32_t> &indices,
                                               const std::vector<ge::Tensor> &inputs,
                                               const ge::DataFlowInfo &info,
                                               int32_t timeout) {
  auto index = indices[0];
  std::vector<ge::Tensor> outputs;
  if (index == 4U || index == 8U || index == 6U || index == 10U || index == 7U || index == 9 || index == 11 ||
      index == 12) {
    ge::Tensor output_tensor;
    int32_t input_tensor_data = 0;
    ge::TensorDesc desc(ge::Shape(std::vector<int64_t>{1}), ge::FORMAT_ND, ge::DT_INT32);
    output_tensor.SetTensorDesc(desc);
    output_tensor.SetData(reinterpret_cast<uint8_t *>(&input_tensor_data), sizeof(int32_t));
    outputs.push_back(output_tensor);
  } else if (index == 0U) {  // allocate
    udf_node_->AllocateCache(inputs, outputs);
  } else if (index == 1U) {  // deallocate
    udf_node_->DeallocateCache(inputs, outputs);
  } else if (index == 3U) {  // get
    cache_get_tensors_.clear();
    udf_node_->GetCache(inputs, cache_get_tensors_);
  } else if (index == 2U) {
    udf_node_->CopyCache(inputs, outputs);
  } else if (index == 5U) {
    if (inputs[0].GetSize() == sizeof(PullKvReqInfo)) {
      udf_node_->PullCache(inputs, outputs);
    } else {
      ge::Tensor output_tensor;
      int32_t input_tensor_data = 0;
      ge::TensorDesc desc(ge::Shape(std::vector<int64_t>{1}), ge::FORMAT_ND, ge::DT_INT32);
      output_tensor.SetTensorDesc(desc);
      output_tensor.SetData(reinterpret_cast<uint8_t *>(&input_tensor_data), sizeof(int32_t));
      outputs.push_back(output_tensor);
    }
  } else {
    LLMLOGE(ge::FAILED, "invalid indices: %s", llm::ToString(indices).c_str());
    return ge::FAILED;
  }

  index_to_output_tensors_[index] = std::move(outputs);
  return ge::SUCCESS;
}

ge::Status CacheEngineGeApi::FetchDataFlowGraph(uint32_t graph_id,
                                                const std::vector<uint32_t> &indexes,
                                                std::vector<ge::Tensor> &outputs,
                                                ge::DataFlowInfo &info,
                                                int32_t timeout) {
  auto index = indexes[0];
  if (index == 3) {
    outputs = {cache_get_tensors_.back()};
    cache_get_tensors_.pop_back();
  } else {
    outputs = std::move(index_to_output_tensors_[index]);
  }
  info.SetTransactionId(transaction_ids_[index]);
  transaction_ids_[index]++;
  return ge::SUCCESS;
}

ge::Status CacheEngineGeApi::Initialize(const std::map<ge::AscendString, ge::AscendString> &options) {
  LLMLOGI("Stub Initialize");
  return ge::SUCCESS;
}

ge::Status CacheEngineGeApi::Finalize() {
  LLMLOGI("Stub Finalize");
  return ge::SUCCESS;
}

ge::Status CacheEngineGeApi::AddGraph(uint32_t graph_id,
                                      const ge::Graph &graph,
                                      const std::map<ge::AscendString, ge::AscendString> &options) {
  LLMLOGI("Stub AddGraph");
  return ge::SUCCESS;
}

ge::Status CacheEngineGeApi::BuildGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs) {
  LLMLOGI("Stub BuildGraph");
  return ge::SUCCESS;
}

CacheEngineGeApi::CacheEngineGeApi() {
  this->udf_node_.reset(new MockFlowNode());
}

CacheEngineGeApi::~CacheEngineGeApi() = default;
}  // namespace llm
