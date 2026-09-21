/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include <memory>
#include <vector>
#include <cstdlib>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "llm_datadist/llm_datadist.h"
#include "llm_datadist/llm_error_codes.h"
#include "common/llm_ge_api.h"
#include "cache_engine_stubs.h"

#include "dlog_pub.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "llm_engine_test_helper.h"
#include "common/mem_utils.h"
#include "llm_test_helper.h"

using namespace std;
using namespace ge;
using namespace ::testing;
using ::testing::Invoke;
using ::testing::Mock;

namespace llm_datadist {
const std::string PNE_ID_NPU = "NPU";
class GeApiTimeoutStub : public llm::CacheEngineGeApi {
 public:
  std::unique_ptr<GeApi> NewSession(const map<ge::AscendString, ge::AscendString> &options) override {
    return llm::MakeUnique<GeApiTimeoutStub>();
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
class LlmDataDistSTest : public ::testing::Test {
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
        {"ge.socVersion", "TestSocType2"},
        {"ge.graphRunMode", "0"},
        {llm::LLM_OPTION_ROLE, role.c_str()},
        {llm::LLM_OPTION_CLUSTER_INFO, deploy_cluster_info.c_str()},
        {"RESOURCE_CONFIG_PATH", numa_config_path},
    };
    return options;
  }
};

TEST_F(LlmDataDistSTest, TestIgnoreTransId) {
  llm::GeApi::instance_ = std::make_unique<GeApiTimeoutStub>();
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

TEST_F(LlmDataDistSTest, Prompt) {
  uint64_t prompt_cluster_id = 0U;
  LlmDataDist llm_datadist(prompt_cluster_id, LlmRole::kPrompt);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1";
  EXPECT_EQ(llm_datadist.Initialize(options), LLM_PARAM_INVALID);
  options[llm_datadist::OPTION_LISTEN_IP_INFO] = "127.0.0.1:26000";
  options["llm.BufPoolCfg"] = R"({
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

TEST_F(LlmDataDistSTest, Decoder) {
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
  tmp_kv_desc.shape = {4, 256};
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

  auto src_buffers = std::vector<std::vector<int32_t>>(80, std::vector<int32_t>(4 * 256));
  auto dst_buffers = std::vector<std::vector<int32_t>>(80, std::vector<int32_t>(4 * 256));
  tmp_cached_tensors.tensor_addrs.clear();
  cached_tensors.tensor_addrs.clear();
  for (uint32_t i = 0; i < tmp_cached_tensors.cache_desc.num_tensors; ++i) {
    tmp_cached_tensors.tensor_addrs.emplace_back(reinterpret_cast<uintptr_t>(src_buffers[i].data()));
    cached_tensors.tensor_addrs.emplace_back(reinterpret_cast<uintptr_t>(dst_buffers[i].data()));
  }
  EXPECT_EQ(llm_datadist.CopyKvBlocks(tmp_cached_tensors,
                                      cached_tensors,
                                      {0, 2},
                                      {{1, 3}}), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.CopyKvCache(tmp_cached_tensors,
                                     cached_tensors), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.DeallocateCache(tmp_cached_tensors.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.DeallocateCache(cached_tensors.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.UnlinkLlmClusters(clusters, rets), ge::SUCCESS);
  unsetenv("HCCL_RDMA_TC");
  unsetenv("HCCL_RDMA_SL");
  llm_datadist.Finalize();
}

TEST_F(LlmDataDistSTest, SwitchRoleFailed_OptionNotEnabled) {
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1";
  LlmDataDist llm_datadist(0, LlmRole::kMix);
  EXPECT_EQ(llm_datadist.Initialize(options), ge::SUCCESS);
  EXPECT_NE(llm_datadist.SetRole(LlmRole::kDecoder, {}), ge::SUCCESS);
}

TEST_F(LlmDataDistSTest, SwitchRoleFailed_LackRequiredOption) {
  LlmDataDist llm_datadist(0, LlmRole::kMix);
  std::map<AscendString, AscendString> options;
  options[llm_datadist::OPTION_DEVICE_ID] = "1";
  options[llm_datadist::OPTION_ENABLE_SET_ROLE] = "1";
  EXPECT_EQ(llm_datadist.Initialize(options), ge::SUCCESS);
  EXPECT_NE(llm_datadist.SetRole(LlmRole::kPrompt, {}), ge::SUCCESS);
}

TEST_F(LlmDataDistSTest, SwitchRoleSuccess) {
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

TEST_F(LlmDataDistSTest, TestPush) {
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

TEST_F(LlmDataDistSTest, TestUnsupported) {
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

TEST_F(LlmDataDistSTest, TestPushWithRangeAndTensorNumV1) {
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
  EXPECT_EQ(llm_datadist.PushKvBlocks(cached_tensors,
                                      dst_cache_index,
                                      {0U, 1U},
                                      {0U, 1U},
                                      ext_param), ge::SUCCESS);

  EXPECT_EQ(llm_datadist.PushKvCache(cached_tensors, dst_cache_index, 0, -1, ext_param), ge::SUCCESS);

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
  // no real udf, which check the numer, so here is eq
  ext_param.tensor_num_per_layer = 3;
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

TEST_F(LlmDataDistSTest, TestPullWithRangeAndTensorNumV1) {
  setenv("HCCL_RDMA_TC", "236", 1);
  setenv("HCCL_RDMA_SL", "5", 1);
  uint64_t prompt_cluster_id = 0U;
  uint64_t decoder_cluster_id = 1U;
  
  LlmDataDist llm_datadist(decoder_cluster_id, LlmRole::kDecoder);
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

  CacheDesc kv_desc{};
  kv_desc.num_tensors = 80;
  kv_desc.data_type = DT_FLOAT16;
  kv_desc.shape = {4, 256};

  Cache cached_tensors;
  EXPECT_EQ(llm_datadist.AllocateCache(kv_desc, cached_tensors), ge::SUCCESS);
  Cache dst_cached_tensors;
  EXPECT_EQ(llm_datadist.AllocateCache(kv_desc, dst_cached_tensors), ge::SUCCESS);

  CacheIndex src_cache_index{prompt_cluster_id, cached_tensors.cache_id, 0};

  KvCacheExtParam ext_param{};

  // test pull by cache id
  EXPECT_EQ(llm_datadist.PullKvBlocks(src_cache_index, dst_cached_tensors, {0U, 1U}, {0U, 1U}, ext_param), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.PullKvCache(src_cache_index, dst_cached_tensors, 0, -1, ext_param), ge::SUCCESS);

  ext_param.src_layer_range = std::make_pair(0, 1);
  ext_param.dst_layer_range = std::make_pair(0, 1);
  ext_param.tensor_num_per_layer = 1;
  EXPECT_EQ(llm_datadist.PullKvBlocks(src_cache_index, dst_cached_tensors, {0U, 1U}, {0U, 1U}, ext_param), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.PullKvCache(src_cache_index, dst_cached_tensors, 0, -1, ext_param), ge::SUCCESS);

  ext_param.tensor_num_per_layer = 2;
  EXPECT_EQ(llm_datadist.PullKvBlocks(src_cache_index, dst_cached_tensors, {0U, 1U}, {0U, 1U}, ext_param), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.PullKvCache(src_cache_index, dst_cached_tensors, 0, -1, ext_param), ge::SUCCESS);
  // no real udf, which check the numer, so here is eq
  ext_param.tensor_num_per_layer = 3;
  EXPECT_EQ(llm_datadist.PullKvBlocks(src_cache_index, dst_cached_tensors, {0U, 1U}, {0U, 1U}, ext_param), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.PullKvCache(src_cache_index, dst_cached_tensors, 0, -1, ext_param), ge::SUCCESS);

  EXPECT_EQ(llm_datadist.DeallocateCache(cached_tensors.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_datadist.DeallocateCache(dst_cached_tensors.cache_id), ge::SUCCESS);

  EXPECT_EQ(llm_datadist.UnlinkLlmClusters(clusters, rets), ge::SUCCESS);
  unsetenv("HCCL_RDMA_TC");
  unsetenv("HCCL_RDMA_SL");
  llm_datadist.Finalize();
}
}  // namespace llm_datadist
