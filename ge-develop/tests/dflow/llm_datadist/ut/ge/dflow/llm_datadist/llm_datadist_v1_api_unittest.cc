/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <vector>
#include <cstdlib>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "llm_engine_test_helper.h"
#include "llm_datadist/llm_datadist.h"
#include "common/llm_flow_service.h"
#include "common/llm_ge_api.h"
#include "cache_engine_stubs.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/op_desc_utils.h"
#include "dlog_pub.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "acl/acl.h"
#include "mem_utils.h"
#include "llm_test_helper.h"

using namespace std;
using namespace ge;
using namespace ::testing;
using ::testing::Invoke;
using ::testing::Mock;
namespace llm_datadist {

class GeApiIgnoreTransIdStub : public llm::CacheEngineGeApi {
 public:
  std::unique_ptr<GeApi> NewSession(const map<ge::AscendString, ge::AscendString> &options) override {
    return llm::MakeUnique<GeApiIgnoreTransIdStub>();
  }
  ge::Status FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                std::vector<ge::Tensor> &outputs, ge::DataFlowInfo &info, int32_t timeout) override {
    auto index = indexes[0];
    if (first_times[index]) {
      info.SetTransactionId(1000000);
      first_times[index] = false;
      outputs.emplace_back();
    } else {
      CacheEngineGeApi::FetchDataFlowGraph(graph_id, indexes, outputs, info, timeout);
    }
    return ge::SUCCESS;
  }
 private:
  std::vector<bool> first_times = std::vector<bool>(static_cast<int>(llm::FlowFuncType::kMax), true);
};
class GeApiLinkTimeoutStub : public llm::CacheEngineGeApi {
 public:
  std::unique_ptr<GeApi> NewSession(const map<ge::AscendString, ge::AscendString> &options) override {
    return llm::MakeUnique<GeApiLinkTimeoutStub>();
  }
  ge::Status FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                std::vector<ge::Tensor> &outputs, ge::DataFlowInfo &info, int32_t timeout) override {
    if (indexes[0] == 7) {
      count++;
      // two devices, second time fetch.
      if (count == 3) {
        count = 0;
        transaction_id_++;
      }
      info.SetTransactionId(transaction_id_);
      return llm::ConvertAclError2Ge(ACL_ERROR_GE_MODEL_EXECUTE_TIMEOUT);
    } else {
      CacheEngineGeApi::FetchDataFlowGraph(graph_id, indexes, outputs, info, timeout);
    }
    return ge::SUCCESS;
  }
 private:
  int count = 0;
  uint64_t transaction_id_ = 1U;
};
class GeApiLinkOneClusterTimeoutStub : public llm::CacheEngineGeApi {
 public:
  std::unique_ptr<GeApi> NewSession(const map<ge::AscendString, ge::AscendString> &options) override {
    return llm::MakeUnique<GeApiLinkOneClusterTimeoutStub>();
  }
  ge::Status FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                std::vector<ge::Tensor> &outputs, ge::DataFlowInfo &info, int32_t timeout) override {
    if (indexes[0] == 7) {
      count++;
      // two devices, second time fetch.
      if (count == 3) {
        count = 0;
        transaction_id_++;
      }
      info.SetTransactionId(transaction_id_);
      ge::Tensor output_tensor;
      std::vector<int32_t> vec(2, 0);
      vec[0] = 1;
      ge::TensorDesc desc(ge::Shape(std::vector<int64_t>{2}), ge::FORMAT_ND, ge::DT_INT32);
      output_tensor.SetTensorDesc(desc);
      output_tensor.SetData(reinterpret_cast<uint8_t *>(vec.data()), 2 * sizeof(int32_t));
      outputs.push_back(output_tensor);
    } else {
      CacheEngineGeApi::FetchDataFlowGraph(graph_id, indexes, outputs, info, timeout);
    }
    return ge::SUCCESS;
  }
 private:
  int count = 0;
  uint64_t transaction_id_ = 1U;
};
class LlmDataDistUTest : public ::testing::Test {
 protected:
  // 在测试类中设置一些准备工作，如果需要的话
  void SetUp() override {
    llm::GeApi::instance_ = llm::MakeUnique<llm::CacheEngineGeApi>();
    ASSERT_TRUE(llm::GeApi::instance_ != nullptr);
  }
  // 在测试类中进行清理工作，如果需要的话
  void TearDown() override {
  }

  static std::map<ge::AscendString, ge::AscendString> GetOptions(const std::string &role,
                                                                 const std::string &deploy_cluster_info) {
    char_t numa_config_path[4096];
    if (role == "Prompt") {
      (void) mmRealPath("../tests/dflow/llm_datadist/st/testcase/llm_datadist/json_file/numa_config_prompt.json", numa_config_path, 4096);
    } else {
      (void) mmRealPath("../tests/dflow/llm_datadist/st/testcase/llm_datadist/json_file/numa_config_decoder.json", numa_config_path, 4096);
    }
    LLMLOGI("numa_config_path:%s", numa_config_path);
    std::map<ge::AscendString, ge::AscendString> options = {
        {"ge.socVersion", "TestSocType1"},
        {"ge.graphRunMode", "0"},
        {llm::LLM_OPTION_ROLE, role.c_str()},
        {llm::LLM_OPTION_CLUSTER_INFO, deploy_cluster_info.c_str()},
        {"RESOURCE_CONFIG_PATH", numa_config_path},
    };
    return options;
  }
};

TEST_F(LlmDataDistUTest, TestIgnoreTransId) {
  llm::GeApi::instance_ = std::make_unique<GeApiIgnoreTransIdStub>();
  ASSERT_TRUE(llm::GeApi::instance_ != nullptr);
  uint64_t prompt_cluster_id = 0U;
  LlmDataDist llm_datadist(prompt_cluster_id, LlmRole::kDecoder);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1";
  options["llm.BufPoolCfg"] = R"({
"buf_pool_size": 1024
})";
  EXPECT_EQ(llm_datadist.Initialize(options), SUCCESS);

  std::vector<ge::Status> rets;
  std::vector<ClusterInfo> clusters;
  ClusterInfo cluster_info;
  IpInfo ip_info;
  ip_info.ip = "127.0.0.1";
  cluster_info.local_ip_infos = {ip_info};
  cluster_info.remote_ip_infos = {ip_info};
  clusters.emplace_back(cluster_info);
  EXPECT_EQ(llm_datadist.LinkLlmClusters(clusters, rets), ge::SUCCESS);

  CacheDesc tmp_kv_desc{};
  tmp_kv_desc.num_tensors = 80;
  tmp_kv_desc.data_type = DT_FLOAT16;
  tmp_kv_desc.shape = {1, 256};
  Cache tmp_cached_tensors;
  CacheIndex cache_key{prompt_cluster_id, 1, 0};

  CacheDesc kv_desc{};
  kv_desc.num_tensors = 80;
  kv_desc.data_type = DT_FLOAT16;
  kv_desc.shape = {4, 256};
  Cache cached_tensors;

  EXPECT_EQ(llm_datadist.AllocateCache(tmp_kv_desc, tmp_cached_tensors), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.AllocateCache(kv_desc, cached_tensors), ge::SUCCESS);
  llm::PullCacheParam pull_cache_param{};
  pull_cache_param.prompt_blocks = {0LU, 1LU};
  pull_cache_param.decoder_blocks = {2LU, 3LU};
  EXPECT_EQ(llm_datadist.PullKvBlocks(cache_key,
                                      tmp_cached_tensors,
                                      pull_cache_param.prompt_blocks,
                                      pull_cache_param.decoder_blocks), ge::SUCCESS);
  llm_datadist.Finalize();
}

TEST_F(LlmDataDistUTest, InitializeSuccess) {
  LlmDataDist llm_datadist(0, LlmRole::kMix);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1";
  EXPECT_EQ(llm_datadist.Initialize(options), SUCCESS);
  llm_datadist.Finalize();

  options[OPTION_BUF_POOL_CFG] = R"({
"buf_cfg":[{"total_size":2097152,"blk_size":256,"max_buf_size":8192}],
"buf_pool_size": 1024
})";
  EXPECT_EQ(llm_datadist.Initialize(options), SUCCESS);
  llm_datadist.Finalize();

  options[OPTION_BUF_POOL_CFG] = R"({
"buf_pool_size": 1024
})";
  EXPECT_EQ(llm_datadist.Initialize(options), SUCCESS);
  llm_datadist.Finalize();

  options[OPTION_BUF_POOL_CFG] = R"({
"buf_cfg":[{"total_size":2097152,"blk_size":256,"max_buf_size":8192}]
})";
  EXPECT_EQ(llm_datadist.Initialize(options), SUCCESS);
  llm_datadist.Finalize();
}

TEST_F(LlmDataDistUTest, InitializeFailed_ErrorDeviceId) {
  LlmDataDist llm_datadist(0, LlmRole::kMix);
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);

  options[llm_datadist::OPTION_DEVICE_ID] = "";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);

  options[llm_datadist::OPTION_DEVICE_ID] = "a";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);

  options[llm_datadist::OPTION_DEVICE_ID] = "-1";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);

  options[llm_datadist::OPTION_DEVICE_ID] = std::to_string(UINT32_MAX).c_str();
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);
}

TEST_F(LlmDataDistUTest, InitializeFailed_ErrorBufCfg) {
  LlmDataDist llm_datadist(0, LlmRole::kMix);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1";
  options[OPTION_BUF_POOL_CFG] = "";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);
  options[OPTION_BUF_POOL_CFG] = R"({})";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);
  options[OPTION_BUF_POOL_CFG] = R"({invalid json})";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);
  options[OPTION_BUF_POOL_CFG] = R"({
"buf_pool_size": 111111111111111111111111111111111111
})";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);
  options[OPTION_BUF_POOL_CFG] = R"({
"buf_pool_size": -1
})";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);
  options[OPTION_BUF_POOL_CFG] = R"({
"buf_pool_size": 0
})";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);
  options[OPTION_BUF_POOL_CFG] = R"({
"buf_pool_size": ""
})";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);
  options[OPTION_BUF_POOL_CFG] = R"({
"buf_pool_size": "1"
})";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);
}

TEST_F(LlmDataDistUTest, InitializeFailed_ErrorListenIpInfo) {
  LlmDataDist llm_datadist(0, LlmRole::kPrompt);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);
  options[llm_datadist::OPTION_LISTEN_IP_INFO] = "";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);
  options[llm_datadist::OPTION_LISTEN_IP_INFO] = "127.0.0.1";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);
  options[llm_datadist::OPTION_LISTEN_IP_INFO] = "127.0.1:26000";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);
  options[llm_datadist::OPTION_LISTEN_IP_INFO] = "256.0.0.1:26000";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);
  options[llm_datadist::OPTION_LISTEN_IP_INFO] = "a.a.a.a:26000";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);
  options[llm_datadist::OPTION_LISTEN_IP_INFO] = "127.0.0.1:-1";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);
  options[llm_datadist::OPTION_LISTEN_IP_INFO] = "127.0.0.1:";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);
  options[llm_datadist::OPTION_LISTEN_IP_INFO] = "127.0.0.1:a";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);
}

TEST_F(LlmDataDistUTest, Prompt) {
  uint64_t prompt_cluster_id = 0U;
  LlmDataDist llm_datadist(prompt_cluster_id, LlmRole::kPrompt);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1";
  options[llm_datadist::OPTION_LISTEN_IP_INFO] = "127.0.0.1:26000";
  options[OPTION_BUF_POOL_CFG] = R"({
"buf_cfg":[{"total_size":2097152,"blk_size":256,"max_buf_size":8192}],
"buf_pool_size": 1024
})";
  EXPECT_EQ(llm_datadist.Initialize(options), SUCCESS);

  std::vector<ge::Status> rets;
  std::vector<llm_datadist::ClusterInfo> clusters;
  llm_datadist::ClusterInfo cluster_info;
  llm_datadist::IpInfo ip_info;
  ip_info.ip = "127.0.0.1";
  cluster_info.local_ip_infos = {ip_info};
  cluster_info.remote_ip_infos = {ip_info};
  clusters.emplace_back(cluster_info);

  CacheDesc kv_desc{};
  kv_desc.num_tensors = 80;
  kv_desc.data_type = DT_FLOAT16;
  kv_desc.shape = {1, 256};
  Cache cached_tensors;
  std::vector<llm::CacheKey> cache_keys;
  cache_keys.emplace_back(llm::CacheKey{0, 1, 0});
  EXPECT_EQ(llm_datadist.AllocateCache(kv_desc, cached_tensors), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.DeallocateCache(cached_tensors.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.UnlinkLlmClusters(clusters, rets), ge::SUCCESS);
  llm_datadist.Finalize();
}

TEST_F(LlmDataDistUTest, Decoder) {
  setenv("HCCL_RDMA_TC", "236", 1);
  setenv("HCCL_RDMA_SL", "5", 1);
  uint64_t prompt_cluster_id = 0U;
  LlmDataDist llm_datadist(prompt_cluster_id, LlmRole::kDecoder);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1";
  options["llm.BufPoolCfg"] = R"({
"buf_pool_size": 1024
})";
  EXPECT_EQ(llm_datadist.Initialize(options), SUCCESS);

  std::vector<ge::Status> rets;
  std::vector<ClusterInfo> clusters;
  ClusterInfo cluster_info;
  IpInfo ip_info;
  ip_info.ip = "127.0.0.1";
  cluster_info.local_ip_infos = {ip_info};
  cluster_info.remote_ip_infos = {ip_info};
  clusters.emplace_back(cluster_info);
  EXPECT_EQ(llm_datadist.LinkLlmClusters(clusters, rets), ge::SUCCESS);

  CacheDesc tmp_kv_desc{};
  tmp_kv_desc.num_tensors = 80;
  tmp_kv_desc.data_type = DT_FLOAT16;
  tmp_kv_desc.shape = {1, 256};
  Cache tmp_cached_tensors;
  CacheIndex cache_key{prompt_cluster_id, 1, 0};

  CacheDesc kv_desc{};
  kv_desc.num_tensors = 80;
  kv_desc.data_type = DT_FLOAT16;
  kv_desc.shape = {4, 256};
  Cache cached_tensors;

  EXPECT_EQ(llm_datadist.AllocateCache(tmp_kv_desc, tmp_cached_tensors), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.AllocateCache(kv_desc, cached_tensors), ge::SUCCESS);
  llm::PullCacheParam pull_cache_param{};
  pull_cache_param.prompt_blocks = {0LU, 1LU};
  pull_cache_param.decoder_blocks = {2LU, 3LU};
  EXPECT_EQ(llm_datadist.PullKvBlocks(cache_key,
                                      tmp_cached_tensors,
                                      pull_cache_param.prompt_blocks,
                                      pull_cache_param.decoder_blocks), ge::SUCCESS);

  // test pull by cache id
  CacheIndex cache_key_by_id{};
  cache_key_by_id.cluster_id = 0;
  cache_key_by_id.cache_id = cached_tensors.cache_id;
  cache_key_by_id.batch_index = 0;
  EXPECT_EQ(llm_datadist.PullKvCache(cache_key_by_id, cached_tensors), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.DeallocateCache(tmp_cached_tensors.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.DeallocateCache(cached_tensors.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.UnlinkLlmClusters(clusters, rets), ge::SUCCESS);
  unsetenv("HCCL_RDMA_TC");
  unsetenv("HCCL_RDMA_SL");
  llm_datadist.Finalize();
}

TEST_F(LlmDataDistUTest, CopyKvCaches) {
  uint64_t prompt_cluster_id = 0U;
  LlmDataDist llm_datadist(prompt_cluster_id, LlmRole::kDecoder);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1";
  options["llm.BufPoolCfg"] = R"({
"buf_pool_size": 1024
})";
  EXPECT_EQ(llm_datadist.Initialize(options), SUCCESS);

  CacheDesc kv_desc{};
  kv_desc.num_tensors = 80;
  kv_desc.data_type = DT_INT32;
  kv_desc.shape = {4, 16};
  Cache src_cache;
  src_cache.cache_desc = kv_desc;
  src_cache.cache_desc.placement = CachePlacement::kHost;
  Cache dst_cache;
  dst_cache.cache_desc = kv_desc;
  auto src_buffers = std::vector<std::vector<int32_t>>(80, std::vector<int32_t>(4 * 16));
  auto dst_buffers = std::vector<std::vector<int32_t>>(80, std::vector<int32_t>(4 * 16));
  for (uint32_t i = 0; i < src_cache.cache_desc.num_tensors; ++i) {
    src_cache.tensor_addrs.emplace_back(reinterpret_cast<uintptr_t>(src_buffers[i].data()));
    dst_cache.tensor_addrs.emplace_back(reinterpret_cast<uintptr_t>(dst_buffers[i].data()));
  }

  std::iota(src_buffers[0].begin(), src_buffers[0].end(), 0);
  EXPECT_EQ(llm_datadist.CopyKvCache(src_cache, dst_cache, 0, 1), ge::LLM_PARAM_INVALID);
  src_cache.cache_desc.placement = CachePlacement::kDevice;
  EXPECT_EQ(llm_datadist.CopyKvCache(src_cache, dst_cache, 0, 1), ge::SUCCESS);
  dst_cache.cache_desc.placement = CachePlacement::kHost;
  EXPECT_EQ(llm_datadist.CopyKvCache(src_cache, dst_cache, 2, 3), ge::SUCCESS);
  std::vector<int32_t> expected =
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
  EXPECT_EQ(dst_buffers[0], expected);
  llm_datadist.Finalize();
}

TEST_F(LlmDataDistUTest, CopyKvBlocks) {
  uint64_t prompt_cluster_id = 0U;
  LlmDataDist llm_datadist(prompt_cluster_id, LlmRole::kDecoder);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1";
  options["llm.BufPoolCfg"] = R"({
"buf_pool_size": 1024
})";
  EXPECT_EQ(llm_datadist.Initialize(options), SUCCESS);

  CacheDesc kv_desc{};
  kv_desc.num_tensors = 80;
  kv_desc.data_type = DT_INT32;
  kv_desc.shape = {4, 16};
  Cache src_cache;
  src_cache.cache_desc = kv_desc;
  Cache dst_cache;
  dst_cache.cache_desc = kv_desc;
  auto src_buffers = std::vector<std::vector<int32_t>>(80, std::vector<int32_t>(4 * 16));
  auto dst_buffers = std::vector<std::vector<int32_t>>(80, std::vector<int32_t>(4 * 16));
  for (uint32_t i = 0; i < src_cache.cache_desc.num_tensors; ++i) {
    src_cache.tensor_addrs.emplace_back(reinterpret_cast<uintptr_t>(src_buffers[i].data()));
    dst_cache.tensor_addrs.emplace_back(reinterpret_cast<uintptr_t>(dst_buffers[i].data()));
  }

  std::iota(src_buffers[0].begin(), src_buffers[0].end(), 0);
  EXPECT_EQ(llm_datadist.CopyKvBlocks(src_cache,
                                      dst_cache,
                                      {0, 2},
                                      {{1, 3}}), ge::SUCCESS);
  std::vector<int32_t> expected =
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
  EXPECT_EQ(dst_buffers[0], expected);

  src_cache.cache_desc.placement = CachePlacement::kHost;  // SwapIn
  std::iota(src_buffers[1].begin(), src_buffers[1].end(), 0);
  EXPECT_EQ(llm_datadist.CopyKvBlocks(src_cache,
                                      dst_cache,
                                      {0, 2},
                                      {{1, 3}}), ge::SUCCESS);
  EXPECT_EQ(dst_buffers[1], expected);

  src_cache.cache_desc.placement = CachePlacement::kDevice;  // SwapIn
  dst_cache.cache_desc.placement = CachePlacement::kDevice;  // SwapOut
  std::iota(src_buffers[2].begin(), src_buffers[2].end(), 0);
  EXPECT_EQ(llm_datadist.CopyKvBlocks(src_cache,
                                      dst_cache,
                                      {0, 2},
                                      {{1, 3}}), ge::SUCCESS);
  EXPECT_EQ(dst_buffers[2], expected);
  llm_datadist.Finalize();
}

TEST_F(LlmDataDistUTest, SwitchRoleFailed_OptionNotEnabled) {
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1";
  LlmDataDist llm_datadist(0, LlmRole::kMix);
  EXPECT_EQ(llm_datadist.Initialize(options), ge::SUCCESS);
  EXPECT_NE(llm_datadist.SetRole(LlmRole::kDecoder, {}), ge::SUCCESS);
}

TEST_F(LlmDataDistUTest, SwitchRoleFailed_LackRequiredOption) {
  LlmDataDist llm_datadist(0, LlmRole::kMix);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1";
  options[llm_datadist::OPTION_ENABLE_SET_ROLE] = "1";
  EXPECT_EQ(llm_datadist.Initialize(options), ge::SUCCESS);
  EXPECT_NE(llm_datadist.SetRole(LlmRole::kPrompt, {}), ge::SUCCESS);
}

TEST_F(LlmDataDistUTest, SwitchRoleSuccess) {
  LlmDataDist llm_datadist(0, LlmRole::kMix);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1";
  options[llm::LLM_OPTION_ENABLE_SWITCH_ROLE] = "1";
  EXPECT_EQ(llm_datadist.Initialize(options), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.SetRole(LlmRole::kDecoder, {}), ge::SUCCESS);
  std::map<AscendString, AscendString> switch_options;
  switch_options[llm_datadist::OPTION_LISTEN_IP_INFO] = "127.0.0.1:26001";
  EXPECT_EQ(llm_datadist.SetRole(LlmRole::kPrompt, switch_options), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.SetRole(LlmRole::kMix, {}), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.SetRole(LlmRole::kPrompt, switch_options), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.SetRole(LlmRole::kDecoder, {}), ge::SUCCESS);
}

TEST_F(LlmDataDistUTest, NotInitialized) {
  CacheDesc kv_desc{};
  kv_desc.num_tensors = 80;
  kv_desc.data_type = DT_INT32;
  kv_desc.shape = {4, 16};
  Cache cache{};
  LlmDataDist llm_datadist(0, LlmRole::kMix);
  EXPECT_EQ(llm_datadist.SetRole(LlmRole::kDecoder, {}), FAILED);
  EXPECT_EQ(llm_datadist.AllocateCache(kv_desc, cache), FAILED);
}

TEST_F(LlmDataDistUTest, RepeatInitialized) {
  LlmDataDist llm_datadist(0, LlmRole::kMix);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1";
  options[llm::LLM_OPTION_ENABLE_SWITCH_ROLE] = "1";
  EXPECT_EQ(llm_datadist.Initialize(options), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.Initialize(options), ge::SUCCESS);
  llm_datadist.Finalize();
  EXPECT_EQ(llm_datadist.Initialize(options), ge::SUCCESS);
  llm_datadist.Finalize();
}

TEST_F(LlmDataDistUTest, StatusCodes) {
  EXPECT_EQ(LLM_SUCCESS, ge::SUCCESS);
  EXPECT_EQ(LLM_FAILED, ge::FAILED);
  EXPECT_EQ(LLM_KV_CACHE_NOT_EXIST, ge::LLM_KV_CACHE_NOT_EXIST);
  EXPECT_EQ(LLM_PARAM_INVALID, ge::LLM_PARAM_INVALID);
  EXPECT_EQ(LLM_NOT_YET_LINK, ge::LLM_NOT_YET_LINK);
  EXPECT_EQ(LLM_ALREADY_LINK, ge::LLM_ALREADY_LINK);
  EXPECT_EQ(LLM_LINK_FAILED, ge::LLM_LINK_FAILED);
  EXPECT_EQ(LLM_UNLINK_FAILED, ge::LLM_UNLINK_FAILED);
  EXPECT_EQ(LLM_NOTIFY_PROMPT_UNLINK_FAILED, ge::LLM_NOTIFY_PROMPT_UNLINK_FAILED);
  EXPECT_EQ(LLM_CLUSTER_NUM_EXCEED_LIMIT, ge::LLM_CLUSTER_NUM_EXCEED_LIMIT);
  EXPECT_EQ(LLM_PROCESSING_LINK, ge::LLM_PROCESSING_LINK);
  EXPECT_EQ(LLM_DEVICE_OUT_OF_MEMORY, ge::LLM_DEVICE_OUT_OF_MEMORY);
  EXPECT_EQ(LLM_EXIST_LINK, ge::LLM_EXIST_LINK);
  EXPECT_EQ(LLM_FEATURE_NOT_ENABLED, ge::LLM_FEATURE_NOT_ENABLED);
  EXPECT_EQ(LLM_TIMEOUT, ge::LLM_TIMEOUT);
  EXPECT_EQ(LLM_LINK_BUSY, ge::LLM_LINK_BUSY);
  EXPECT_EQ(LLM_OUT_OF_MEMORY, ge::LLM_OUT_OF_MEMORY);
}

TEST_F(LlmDataDistUTest, MultipleDevicesCopyKvCaches) {
  uint64_t prompt_cluster_id = 0U;
  LlmDataDist llm_datadist(prompt_cluster_id, LlmRole::kDecoder);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1;2";
  options["llm.BufPoolCfg"] = R"({
"buf_pool_size": 1024
})";
  EXPECT_EQ(llm_datadist.Initialize(options), SUCCESS);

  CacheDesc kv_desc{};
  kv_desc.num_tensors = 40;
  kv_desc.data_type = DT_INT32;
  kv_desc.shape = {4, 16};
  Cache src_cache;
  src_cache.cache_desc = kv_desc;
  src_cache.cache_desc.placement = CachePlacement::kDevice;
  Cache dst_cache;
  dst_cache.cache_desc = kv_desc;
  auto src_buffers = std::vector<std::vector<int32_t>>(80, std::vector<int32_t>(4 * 16, 0));
  auto dst_buffers = std::vector<std::vector<int32_t>>(80, std::vector<int32_t>(4 * 16, 0));
  for (uint32_t i = 0; i < src_buffers.size(); ++i) {
    src_cache.tensor_addrs.emplace_back(reinterpret_cast<uintptr_t>(src_buffers[i].data()));
    dst_cache.tensor_addrs.emplace_back(reinterpret_cast<uintptr_t>(dst_buffers[i].data()));
  }

  for (size_t i = 0; i < src_buffers.size(); ++i) {
    std::iota(src_buffers[i].begin(), src_buffers[i].end(), 0);
  }
  EXPECT_EQ(llm_datadist.CopyKvCache(src_cache, dst_cache, 0, 1), ge::SUCCESS);
  for (size_t i = 0; i < src_buffers.size(); ++i) {
    for (size_t j = 0; j < 16; ++j) {
      EXPECT_EQ(dst_buffers[i][16 + j], j);
    }
  }
  llm_datadist.Finalize();
}

TEST_F(LlmDataDistUTest, MultipleDevicesSwapKvBlocks) {
  uint64_t prompt_cluster_id = 0U;
  LlmDataDist llm_datadist(prompt_cluster_id, LlmRole::kDecoder);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1;2";
  options["llm.BufPoolCfg"] = R"({
"buf_pool_size": 1024
})";
  EXPECT_EQ(llm_datadist.Initialize(options), SUCCESS);

  CacheDesc kv_desc{};
  kv_desc.num_tensors = 40;
  kv_desc.data_type = DT_INT32;
  kv_desc.shape = {4, 16};
  Cache src_cache;
  src_cache.cache_desc = kv_desc;
  src_cache.cache_desc.placement = CachePlacement::kDevice;
  Cache dst_cache;
  dst_cache.cache_desc = kv_desc;
  auto src_buffers = std::vector<std::vector<int32_t>>(80, std::vector<int32_t>(4 * 16, 0));
  auto dst_buffers = std::vector<std::vector<int32_t>>(80, std::vector<int32_t>(4 * 16, 0));
  for (uint32_t i = 0; i < src_buffers.size(); ++i) {
    src_cache.tensor_addrs.emplace_back(reinterpret_cast<uintptr_t>(src_buffers[i].data()));
    dst_cache.tensor_addrs.emplace_back(reinterpret_cast<uintptr_t>(dst_buffers[i].data()));
  }
  for (size_t i = 0; i < src_buffers.size(); ++i) {
    std::iota(src_buffers[i].begin(), src_buffers[i].end(), 0);
  }
  EXPECT_EQ(llm_datadist.CopyKvBlocks(src_cache, dst_cache, {0, 1}, {{0, 1}}), ge::SUCCESS);
  for (size_t i = 0; i < src_buffers.size(); ++i) {
    for (size_t j = 0; j < 32; ++j) {
      EXPECT_EQ(dst_buffers[i][j], j);
    }
  }
  llm_datadist.Finalize();
}

TEST_F(LlmDataDistUTest, MultipleDeviceAllocate) {
  uint64_t prompt_cluster_id = 0U;
  LlmDataDist llm_datadist(prompt_cluster_id, LlmRole::kPrompt);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1;2";
  options[llm_datadist::OPTION_LISTEN_IP_INFO] = "127.0.0.1:26000;127.0.0.1:26000";
  options[OPTION_BUF_POOL_CFG] = R"({
"buf_cfg":[{"total_size":2097152,"blk_size":256,"max_buf_size":8192}],
"buf_pool_size": 1024
})";
  EXPECT_EQ(llm_datadist.Initialize(options), SUCCESS);

  CacheDesc kv_desc{};
  kv_desc.num_tensors = 80;
  kv_desc.data_type = DT_FLOAT16;
  kv_desc.shape = {1, 256};
  Cache cached_tensors;
  std::vector<llm::CacheKey> cache_keys;
  cache_keys.emplace_back(llm::CacheKey{0, 1, 0});
  EXPECT_EQ(llm_datadist.AllocateCache(kv_desc, cached_tensors), ge::SUCCESS);
  EXPECT_EQ(cached_tensors.tensor_addrs.size(), 2 * kv_desc.num_tensors);
  EXPECT_EQ(llm_datadist.DeallocateCache(cached_tensors.cache_id), ge::SUCCESS);
  llm_datadist.Finalize();
}

TEST_F(LlmDataDistUTest, OneDeviceHostTimeoutOfMultipleDevicesLink) {
  llm::GeApi::instance_ = std::make_unique<GeApiLinkTimeoutStub>();
  uint64_t prompt_cluster_id = 0U;
  LlmDataDist llm_datadist(prompt_cluster_id, LlmRole::kDecoder);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1;2";
  EXPECT_EQ(llm_datadist.Initialize(options), SUCCESS);

  std::vector<ge::Status> rets;
  std::vector<ClusterInfo> clusters;
  ClusterInfo cluster_info;
  IpInfo ip_info;
  ip_info.ip = "127.0.0.1";
  cluster_info.local_ip_infos = {ip_info, ip_info};
  cluster_info.remote_ip_infos = {ip_info, ip_info};
  clusters.emplace_back(cluster_info);
  clusters.emplace_back(cluster_info);
  EXPECT_EQ(llm_datadist.LinkLlmClusters(clusters, rets), ge::LLM_WAIT_PROC_TIMEOUT);
  EXPECT_EQ(rets.size(), 2);
  EXPECT_EQ(rets[0], ge::LLM_WAIT_PROC_TIMEOUT);
  EXPECT_EQ(rets[1], ge::LLM_WAIT_PROC_TIMEOUT);
  llm_datadist.Finalize();
}

TEST_F(LlmDataDistUTest, OneClusterTimeoutOfMultipleDevicesLink) {
  llm::GeApi::instance_ = std::make_unique<GeApiLinkOneClusterTimeoutStub>();
  uint64_t prompt_cluster_id = 0U;
  LlmDataDist llm_datadist(prompt_cluster_id, LlmRole::kDecoder);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1;2";
  EXPECT_EQ(llm_datadist.Initialize(options), SUCCESS);

  std::vector<ge::Status> rets;
  std::vector<ClusterInfo> clusters;
  ClusterInfo cluster_info;
  IpInfo ip_info;
  ip_info.ip = "127.0.0.1";
  cluster_info.local_ip_infos = {ip_info, ip_info};
  cluster_info.remote_ip_infos = {ip_info, ip_info};
  clusters.emplace_back(cluster_info);
  clusters.emplace_back(cluster_info);
  EXPECT_EQ(llm_datadist.LinkLlmClusters(clusters, rets), ge::LLM_LINK_FAILED);
  EXPECT_EQ(rets.size(), 2);
  EXPECT_EQ(rets[0], ge::LLM_WAIT_PROC_TIMEOUT);
  EXPECT_EQ(rets[1], ge::SUCCESS);
  llm_datadist.Finalize();
}

TEST_F(LlmDataDistUTest, TestPush) {
  uint64_t prompt_cluster_id = 0U;
  LlmDataDist llm_datadist(prompt_cluster_id, LlmRole::kPrompt);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1";
  options[llm_datadist::OPTION_LISTEN_IP_INFO] = "127.0.0.1:26000";
  EXPECT_EQ(llm_datadist.Initialize(options), SUCCESS);

  CacheDesc kv_desc{};
  kv_desc.num_tensors = 80;
  kv_desc.data_type = DT_FLOAT16;
  kv_desc.shape = {1, 256};
  Cache cached_tensors;
  EXPECT_EQ(llm_datadist.AllocateCache(kv_desc, cached_tensors), ge::SUCCESS);
  Cache dst_cached_tensors;
  EXPECT_EQ(llm_datadist.AllocateCache(kv_desc, dst_cached_tensors), ge::SUCCESS);

  CacheIndex dst_cache_index{prompt_cluster_id, dst_cached_tensors.cache_id};
  KvCacheExtParam ext_param{};
  ext_param.src_layer_range = std::make_pair(0, 0);
  ext_param.dst_layer_range = std::make_pair(0, 0);
  EXPECT_EQ(llm_datadist.PushKvBlocks(cached_tensors,
                                      dst_cache_index,
                                      {0U, 1U},
                                      {0U, 1U},
                                      ext_param), ge::SUCCESS);

  EXPECT_EQ(llm_datadist.PushKvCache(cached_tensors, dst_cache_index, 0, -1, ext_param), ge::SUCCESS);

  EXPECT_EQ(llm_datadist.DeallocateCache(dst_cached_tensors.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.DeallocateCache(cached_tensors.cache_id), ge::SUCCESS);
  llm_datadist.Finalize();
}

ge::Status LlmDataDistUTestV1TestChkAclRt(int32_t ret) {
  LLM_CHK_ACL_RET(ret);
  return ge::SUCCESS;
}

TEST_F(LlmDataDistUTest, TestChkAclRt) {
  int32_t test_ret = static_cast<int32_t>(ACL_ERROR_RT_STREAM_SYNC_TIMEOUT);
  EXPECT_EQ(LlmDataDistUTestV1TestChkAclRt(test_ret), static_cast<int32_t>(ACL_ERROR_RT_STREAM_SYNC_TIMEOUT));
  test_ret = 0;
  EXPECT_EQ(LlmDataDistUTestV1TestChkAclRt(test_ret), ge::SUCCESS);
}

TEST_F(LlmDataDistUTest, TestUnsupported) {
  uint64_t prompt_cluster_id = 0U;
  LlmDataDist llm_datadist(prompt_cluster_id, LlmRole::kPrompt);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1";
  options[llm_datadist::OPTION_LISTEN_IP_INFO] = "127.0.0.1:26000";
  EXPECT_EQ(llm_datadist.Initialize(options), SUCCESS);

  CacheDesc desc{};
  int64_t cache_id = -1;
  EXPECT_EQ(llm_datadist.RegisterKvCache(desc, {}, {}, cache_id), LLM_FEATURE_NOT_ENABLED);
  EXPECT_EQ(llm_datadist.UnregisterKvCache(cache_id), LLM_FEATURE_NOT_ENABLED);
  llm_datadist.Finalize();
}

TEST_F(LlmDataDistUTest, TestPushWithRangeAndTensorNumV1) {
  uint64_t prompt_cluster_id = 0U;
  LlmDataDist llm_datadist(prompt_cluster_id, LlmRole::kPrompt);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1";
  options[llm_datadist::OPTION_LISTEN_IP_INFO] = "127.0.0.1:26000";
  EXPECT_EQ(llm_datadist.Initialize(options), SUCCESS);

  CacheDesc kv_desc{};
  kv_desc.num_tensors = 80;
  kv_desc.data_type = DT_FLOAT16;
  kv_desc.shape = {1, 256};
  Cache cached_tensors;
  EXPECT_EQ(llm_datadist.AllocateCache(kv_desc, cached_tensors), ge::SUCCESS);
  Cache dst_cached_tensors;
  EXPECT_EQ(llm_datadist.AllocateCache(kv_desc, dst_cached_tensors), ge::SUCCESS);

  CacheIndex dst_cache_index{prompt_cluster_id, dst_cached_tensors.cache_id};
  KvCacheExtParam ext_param{};

  // test default range and tensor num
  EXPECT_EQ(llm_datadist.PushKvBlocks(cached_tensors,
                                      dst_cache_index,
                                      {0U, 1U},
                                      {0U, 1U},
                                      ext_param), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.PushKvCache(cached_tensors, dst_cache_index, 0, -1, ext_param), ge::SUCCESS);

  // test appointed range and tensor num
  ext_param.src_layer_range = std::make_pair(0, 1);
  ext_param.dst_layer_range = std::make_pair(0, 1);

  ext_param.tensor_num_per_layer = 1;
  EXPECT_EQ(llm_datadist.PushKvBlocks(cached_tensors,
                                      dst_cache_index,
                                      {0U, 1U},
                                      {0U, 1U},
                                      ext_param), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.PushKvCache(cached_tensors, dst_cache_index, 0, -1, ext_param), ge::SUCCESS);

  ext_param.tensor_num_per_layer = 2;
  EXPECT_EQ(llm_datadist.PushKvBlocks(cached_tensors,
                                      dst_cache_index,
                                      {0U, 1U},
                                      {0U, 1U},
                                      ext_param), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.PushKvCache(cached_tensors, dst_cache_index, 0, -1, ext_param), ge::SUCCESS);

  ext_param.tensor_num_per_layer = 7;
  EXPECT_EQ(llm_datadist.PushKvBlocks(cached_tensors,
                                      dst_cache_index,
                                      {0U, 1U},
                                      {0U, 1U},
                                      ext_param), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.PushKvCache(cached_tensors, dst_cache_index, 0, -1, ext_param), ge::SUCCESS);

  // test invalid range and tensor num
  ext_param.tensor_num_per_layer = 2;
  ext_param.dst_layer_range = std::make_pair(0, 1);
  ext_param.src_layer_range = std::make_pair(-2, 0);
  EXPECT_NE(llm_datadist.PushKvBlocks(cached_tensors,
                                      dst_cache_index,
                                      {0U, 1U},
                                      {0U, 1U},
                                      ext_param), ge::SUCCESS);
  EXPECT_NE(llm_datadist.PushKvCache(cached_tensors, dst_cache_index, 0, -1, ext_param), ge::SUCCESS);

  // the range not same
  ext_param.src_layer_range = std::make_pair(0, 1);
  ext_param.dst_layer_range = std::make_pair(0, 3);
  EXPECT_NE(llm_datadist.PushKvBlocks(cached_tensors,
                                      dst_cache_index,
                                      {0U, 1U},
                                      {0U, 1U},
                                      ext_param), ge::SUCCESS);
  EXPECT_NE(llm_datadist.PushKvCache(cached_tensors, dst_cache_index, 0, -1, ext_param), ge::SUCCESS);

  // the src range over range
  ext_param.src_layer_range = std::make_pair(39, 40);
  ext_param.dst_layer_range = std::make_pair(0, 1);
  EXPECT_NE(llm_datadist.PushKvBlocks(cached_tensors,
                                      dst_cache_index,
                                      {0U, 1U},
                                      {0U, 1U},
                                      ext_param), ge::SUCCESS);
  EXPECT_NE(llm_datadist.PushKvCache(cached_tensors, dst_cache_index, 0, -1, ext_param), ge::SUCCESS);

  // the dst range over range
  ext_param.src_layer_range = std::make_pair(0, 1);
  ext_param.dst_layer_range = std::make_pair(39, 40);
  EXPECT_NE(llm_datadist.PushKvBlocks(cached_tensors,
                                      dst_cache_index,
                                      {0U, 1U},
                                      {0U, 1U},
                                      ext_param), ge::SUCCESS);
  EXPECT_NE(llm_datadist.PushKvCache(cached_tensors, dst_cache_index, 0, -1, ext_param), ge::SUCCESS);

  EXPECT_EQ(llm_datadist.DeallocateCache(dst_cached_tensors.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.DeallocateCache(cached_tensors.cache_id), ge::SUCCESS);
  llm_datadist.Finalize();
}

TEST_F(LlmDataDistUTest, TestPullWithRangeAndTensorNumV1) {
  uint64_t prompt_cluster_id = 0U;
  uint64_t decoder_cluster_id = 1U;
  LlmDataDist llm_datadist(decoder_cluster_id, LlmRole::kDecoder);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1";
  options[llm_datadist::OPTION_LISTEN_IP_INFO] = "127.0.0.1:26000";
  EXPECT_EQ(llm_datadist.Initialize(options), SUCCESS);

  CacheDesc kv_desc{};
  kv_desc.num_tensors = 80;
  kv_desc.data_type = DT_FLOAT16;
  kv_desc.shape = {1, 256};
  Cache cached_tensors;
  EXPECT_EQ(llm_datadist.AllocateCache(kv_desc, cached_tensors), ge::SUCCESS);
  Cache dst_cached_tensors;
  EXPECT_EQ(llm_datadist.AllocateCache(kv_desc, dst_cached_tensors), ge::SUCCESS);

  // pull cache
  CacheIndex cache_index{};
  cache_index.cluster_id = prompt_cluster_id;
  cache_index.cache_id = cached_tensors.cache_id;

  Cache dst_cache{};
  dst_cache.cache_id = dst_cached_tensors.cache_id;
  dst_cache.cache_desc = kv_desc;
  KvCacheExtParam ext_param = {};

  EXPECT_EQ(llm_datadist.PullKvCache(cache_index, dst_cache, 0, -1, ext_param), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.PullKvBlocks(cache_index, dst_cache, {0U, 1U}, {0U, 1U}, ext_param), ge::SUCCESS);

  ext_param.src_layer_range = std::make_pair(0, 1);
  ext_param.dst_layer_range = std::make_pair(0, 1);
  EXPECT_EQ(llm_datadist.PullKvCache(cache_index, dst_cache, 0, -1, ext_param), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.PullKvBlocks(cache_index, dst_cache, {0U, 1U}, {0U, 1U}, ext_param), ge::SUCCESS);

  ext_param.tensor_num_per_layer = 1;
  EXPECT_EQ(llm_datadist.PullKvCache(cache_index, dst_cache, 0, -1, ext_param), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.PullKvBlocks(cache_index, dst_cache, {0U, 1U}, {0U, 1U}, ext_param), ge::SUCCESS);

  ext_param.tensor_num_per_layer = 2;
  EXPECT_EQ(llm_datadist.PullKvCache(cache_index, dst_cache, 0, -1, ext_param), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.PullKvBlocks(cache_index, dst_cache, {0U, 1U}, {0U, 1U}, ext_param), ge::SUCCESS);

  ext_param.tensor_num_per_layer = 7;
  EXPECT_EQ(llm_datadist.PullKvCache(cache_index, dst_cache, 0, -1, ext_param), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.PullKvBlocks(cache_index, dst_cache, {0U, 1U}, {0U, 1U}, ext_param), ge::SUCCESS);

  ext_param.tensor_num_per_layer = 2;
  ext_param.src_layer_range = std::make_pair(0, 1);
  ext_param.dst_layer_range = std::make_pair(0, 2);
  EXPECT_EQ(llm_datadist.PullKvCache(cache_index, dst_cache, 0, -1, ext_param), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.PullKvBlocks(cache_index, dst_cache, {0U, 1U}, {0U, 1U}, ext_param), ge::SUCCESS);

  ext_param.src_layer_range = std::make_pair(-2, 1);
  ext_param.dst_layer_range = std::make_pair(0, 1);
  EXPECT_NE(llm_datadist.PullKvCache(cache_index, dst_cache, 0, -1, ext_param), ge::SUCCESS);
  EXPECT_NE(llm_datadist.PullKvBlocks(cache_index, dst_cache, {0U, 1U}, {0U, 1U}, ext_param), ge::SUCCESS);

  EXPECT_EQ(llm_datadist.DeallocateCache(dst_cached_tensors.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.DeallocateCache(cached_tensors.cache_id), ge::SUCCESS);
  llm_datadist.Finalize();
}
}  // namespace llm_datadist
