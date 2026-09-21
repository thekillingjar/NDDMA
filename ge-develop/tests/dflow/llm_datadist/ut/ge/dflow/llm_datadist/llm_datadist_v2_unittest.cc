/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <gtest/gtest.h>

#include "llm_engine_test_helper.h"
#include "cache_engine_stubs.h"
#include "common/cache_engine.h"
#include "llm_datadist.h"
#include "mmpa/mmpa_api.h"
#include "common/llm_inner_types.h"

using namespace std;
using namespace ::testing;
using namespace ge;

namespace llm {
namespace {
class GeApiStub : public ge::DefaultGeApiStub {
 public:
  std::unique_ptr<GeApi> NewSession(const map<ge::AscendString, ge::AscendString> &options) override {
    return MakeUnique<CacheEngineGeApi>();
  }
};
}  // namespace
class LLMDatadistV2UTest : public ::testing::Test {
 protected:
  void SetUp() override {
    api_stub_ = new GeApiStub();
    llm::GeApi::instance_.reset(api_stub_);
  }

  // 在测试类中进行清理工作，如果需要的话
  void TearDown() override {
    llm::GeApi::instance_->Finalize();
  }

  GeApiStub *api_stub_ = nullptr;
};
TEST_F(LLMDatadistV2UTest, TEST_FINALIZE_WHEN_NOT_INITILIZED) {
  LLMDataDist llm_engine(0);
  EXPECT_NO_THROW(llm_engine.LLMDataDistFinalize());
}

TEST_F(LLMDatadistV2UTest, Prompt_With_buf_cfg_success) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::PROMPT).NumStages(1).Build();
  LLMDataDist llm_engine(0);
  options["llm.BufPoolCfg"] = "{\"buf_cfg\":[{\"total_size\":2097152,\"blk_size\":256,\"max_buf_size\":8192}]}";
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);
}

TEST_F(LLMDatadistV2UTest, Prompt_With_buf_cfg_format_invalid) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::PROMPT).NumStages(1).Build();
  LLMDataDist llm_engine(0);
  options["llm.BufPoolCfg"] = "{\"buf_cfg\":[{\"total_size\":2097152,\"blk_size\":256,\"max_buf_size\":8192}}";
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::LLM_PARAM_INVALID);
}

TEST_F(LLMDatadistV2UTest, Prompt_With_buf_cfg_disorder) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::PROMPT).NumStages(1).Build();
  LLMDataDist llm_engine(0);
  options["llm.BufPoolCfg"] =
      "{\"buf_cfg\":[{\"total_size\":33554432,\"blk_size\":256,\"max_buf_size\":8192},"
                    "{\"total_size\":2097152,\"blk_size\":256,\"max_buf_size\":1024}]}";
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::LLM_PARAM_INVALID);
}

TEST_F(LLMDatadistV2UTest, Prompt_With_buf_cfg_blk_invalid) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::PROMPT).NumStages(1).Build();
  LLMDataDist llm_engine(0);
  options["llm.BufPoolCfg"] =
      "{\"buf_cfg\":[{\"total_size\":33554432,\"blk_size\":257,\"max_buf_size\":8192},"
                    "{\"total_size\":2097152,\"blk_size\":256,\"max_buf_size\":8192}]}";
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::LLM_PARAM_INVALID);
}

TEST_F(LLMDatadistV2UTest, Prompt_With_buf_cfg_buf_size_invalid) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::PROMPT).NumStages(1).Build();
  LLMDataDist llm_engine(0);
  options["llm.BufPoolCfg"] =
      "{\"buf_cfg\":[{\"total_size\":33554432,\"blk_size\":8192,\"max_buf_size\":256},"
                    "{\"total_size\":2097152,\"blk_size\":256,\"max_buf_size\":8192}]}";
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::LLM_PARAM_INVALID);
}

TEST_F(LLMDatadistV2UTest, Decoder_LinkUnlinkClusters) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::DECODER).NumStages(1).Build();
  LLMDataDist llm_engine(0);
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);

  std::vector<ge::Status> rets;
  std::vector<llm::ClusterInfo> clusters;
  llm::ClusterInfo cluster_info;
  llm::IpInfo ip_info;
  cluster_info.local_ip_infos = {ip_info, ip_info};
  cluster_info.remote_ip_infos = {ip_info, ip_info};
  clusters.emplace_back(cluster_info);
  EXPECT_EQ(llm_engine.LinkClusters(clusters, rets, -1), ge::SUCCESS);
  EXPECT_EQ(llm_engine.UnlinkClusters(clusters, rets, -1), ge::SUCCESS);
}

TEST_F(LLMDatadistV2UTest, Prompt_UnlinkClusters) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::PROMPT).NumStages(1).Build();
  LLMDataDist llm_engine(0);
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);

  std::vector<ge::Status> rets;
  std::vector<llm::ClusterInfo> clusters;
  llm::ClusterInfo cluster_info;
  llm::IpInfo ip_info;
  cluster_info.local_ip_infos = {ip_info, ip_info};
  cluster_info.remote_ip_infos = {ip_info, ip_info};
  clusters.emplace_back(cluster_info);
  EXPECT_EQ(llm_engine.UnlinkClusters(clusters, rets, -1), ge::SUCCESS);
}

TEST_F(LLMDatadistV2UTest, Prompt_MultiDevices) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::PROMPT).NumStages(1).Build();
  LLMDataDist llm_engine(0);
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);

  CacheDesc kv_desc{};
  kv_desc.num_tensors = 80;
  kv_desc.data_type = ge::DT_FLOAT16;
  kv_desc.shape = {1, 128};

  CacheKey cache_key{};
  cache_key.prompt_cluster_id = 0;
  cache_key.req_id = 1;
  cache_key.model_id = 0;

  Cache kv_cache_1;
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc, kv_cache_1, {cache_key}), ge::SUCCESS);
  EXPECT_EQ(kv_cache_1.cache_id, 1);
  EXPECT_EQ(kv_cache_1.per_device_tensor_addrs.size(), 2U);

  Cache kv_cache_2;
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc, kv_cache_2), ge::SUCCESS);
  EXPECT_EQ(kv_cache_2.cache_id, 2);

  std::vector<ge::Tensor> cached_tensors;
  EXPECT_EQ(llm_engine.GetCacheTensors(kv_cache_1.cache_id, cached_tensors, 0), SUCCESS);

  // test deallocate success
  EXPECT_EQ(llm_engine.DeallocateCache(kv_cache_1.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_engine.DeallocateCache(kv_cache_2.cache_id), ge::SUCCESS);

  EXPECT_EQ(llm_engine.RemoveCacheKey(cache_key), ge::SUCCESS);

  // not exist
  EXPECT_EQ(llm_engine.DeallocateCache(2), ge::LLM_KV_CACHE_NOT_EXIST);
  llm_engine.LLMDataDistFinalize();
}

TEST_F(LLMDatadistV2UTest, Prompt_SingleDevices) {
  ge::LlmEngineOptionBuilder builder;
  setenv("HCCL_RDMA_TC", "236", 1);
  setenv("HCCL_RDMA_SL", "5", 1);
  auto options = builder.Role(llm::RoleType::PROMPT).NumStages(1).Build();

  options.emplace(ge::OPTION_EXEC_RANK_ID, "1");
  LLMDataDist llm_engine(0);
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);

  CacheDesc kv_desc{};
  kv_desc.num_tensors = 80;
  kv_desc.data_type = ge::DT_FLOAT16;
  kv_desc.shape = {1, 128};

  CacheKey cache_key{};
  cache_key.prompt_cluster_id = 0;
  cache_key.req_id = 1;
  cache_key.model_id = 0;

  Cache cached_tensors;
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc, cached_tensors, {cache_key}), ge::SUCCESS);
  EXPECT_EQ(cached_tensors.cache_id, 1);
  EXPECT_EQ(cached_tensors.per_device_tensor_addrs.size(), 1U);

  Cache cached_tensors_2;
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc, cached_tensors_2), ge::SUCCESS);
  EXPECT_EQ(cached_tensors_2.cache_id, 2);

  // test deallocate success
  EXPECT_EQ(llm_engine.DeallocateCache(cached_tensors.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_engine.DeallocateCache(cached_tensors_2.cache_id), ge::SUCCESS);

  // not exist
  EXPECT_EQ(llm_engine.DeallocateCache(2), ge::LLM_KV_CACHE_NOT_EXIST);
  unsetenv("HCCL_RDMA_TC");
  unsetenv("HCCL_RDMA_SL");
}

TEST_F(LLMDatadistV2UTest, Decoder_MultiDevices) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::DECODER).NumStages(1).Build();

  LLMDataDist llm_engine(0);
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);

  CacheDesc kv_desc{};
  kv_desc.data_type = ge::DT_FLOAT16;
  kv_desc.shape = {1, 128};
  kv_desc.num_tensors = 80;

  CacheDesc kv_desc_40{};
  kv_desc_40.data_type = ge::DT_FLOAT16;
  kv_desc_40.shape = {1, 128};
  kv_desc_40.num_tensors = 40;

  Cache cached_tensors;
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc, cached_tensors), ge::SUCCESS);
  Cache cached_tensors_1;
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc, cached_tensors_1), ge::SUCCESS);
  Cache cached_tensors_40;
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc_40, cached_tensors_40), ge::SUCCESS);

  CacheKey cache_key{};
  cache_key.prompt_cluster_id = 0;
  cache_key.req_id = 1;
  cache_key.model_id = 0;
  Cache prompt_cached_tensors;
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc, prompt_cached_tensors, {cache_key}), ge::SUCCESS);
  EXPECT_EQ(llm_engine.PullCache(cached_tensors.cache_id, cache_key, {}), ge::SUCCESS);
  EXPECT_NE(llm_engine.PullCache(cached_tensors_40.cache_id, cache_key, {}), ge::SUCCESS);
  CopyCacheParam copy_cache_param{};
  copy_cache_param.src_cache_id = cached_tensors.cache_id;
  copy_cache_param.dst_cache_id = cached_tensors_1.cache_id;
  EXPECT_EQ(llm_engine.CopyCache(copy_cache_param), ge::SUCCESS);
  copy_cache_param.dst_cache_id = cached_tensors_40.cache_id;
  EXPECT_NE(llm_engine.CopyCache(copy_cache_param), ge::SUCCESS);
  EXPECT_EQ(llm_engine.DeallocateCache(cached_tensors.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_engine.DeallocateCache(cached_tensors_1.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_engine.DeallocateCache(prompt_cached_tensors.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_engine.DeallocateCache(cached_tensors_40.cache_id), ge::SUCCESS);
}

TEST_F(LLMDatadistV2UTest, Decoder_SingleDevices) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::DECODER).NumStages(1).Build();
  LLMDataDist llm_engine(0);
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);

  CacheDesc kv_desc{};
  kv_desc.num_tensors = 80;
  kv_desc.data_type = ge::DT_FLOAT16;
  kv_desc.shape = {1, 128};

  CacheDesc kv_desc_40{};
  kv_desc_40.num_tensors = 40;
  kv_desc_40.data_type = ge::DT_FLOAT16;
  kv_desc_40.shape = {1, 128};

  Cache cached_tensors;
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc, cached_tensors), ge::SUCCESS);
  Cache cached_tensors_1;
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc, cached_tensors_1), ge::SUCCESS);
  Cache cached_tensors_40;
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc_40, cached_tensors_40), ge::SUCCESS);

  CacheKey cache_key{};
  cache_key.prompt_cluster_id = 0;
  cache_key.req_id = 1;
  cache_key.model_id = 0;
  Cache prompt_cached_tensors;
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc, prompt_cached_tensors, {cache_key}), ge::SUCCESS);

  EXPECT_EQ(llm_engine.PullCache(cached_tensors.cache_id, cache_key, {}), ge::SUCCESS);
  EXPECT_NE(llm_engine.PullCache(cached_tensors_40.cache_id, cache_key, {}), ge::SUCCESS);

  // test pull by cache id
  CacheKey cache_key_by_id{};
  cache_key_by_id.prompt_cluster_id = 0;
  cache_key_by_id.prompt_cache_id = cached_tensors.cache_id;
  cache_key_by_id.prompt_batch_index = 0;
  EXPECT_EQ(llm_engine.PullCache(cached_tensors.cache_id, cache_key_by_id, {}), ge::SUCCESS);

  CopyCacheParam copy_cache_param{};
  copy_cache_param.src_cache_id = cached_tensors.cache_id;
  copy_cache_param.dst_cache_id = cached_tensors_1.cache_id;
  EXPECT_EQ(llm_engine.CopyCache(copy_cache_param), ge::SUCCESS);
  copy_cache_param.dst_cache_id = cached_tensors_40.cache_id;
  EXPECT_NE(llm_engine.CopyCache(copy_cache_param), ge::SUCCESS);
  EXPECT_EQ(llm_engine.DeallocateCache(cached_tensors.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_engine.DeallocateCache(cached_tensors_1.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_engine.DeallocateCache(prompt_cached_tensors.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_engine.DeallocateCache(cached_tensors_40.cache_id), ge::SUCCESS);
}

TEST_F(LLMDatadistV2UTest, Decoder_CopyAndPullBlocks) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::DECODER).NumStages(1).Build();
  LLMDataDist llm_engine(0LU);
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);
  CacheDesc kv_desc{};
  kv_desc.num_tensors = 2U;
  kv_desc.data_type = ge::DT_FLOAT16;
  kv_desc.shape = {4, 128};
  Cache block_kv_cache;
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc, block_kv_cache), ge::SUCCESS);
  CacheKey cache_key{};
  cache_key.prompt_cluster_id = 0LU;
  cache_key.req_id = 1LU;
  cache_key.model_id = 0LU;
  Cache prompt_kv_cache;
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc, prompt_kv_cache, {cache_key}), ge::SUCCESS);

  CopyCacheParam copy_cache_param{};
  copy_cache_param.src_cache_id = block_kv_cache.cache_id;
  copy_cache_param.dst_cache_id = block_kv_cache.cache_id;
  copy_cache_param.copy_block_infos = {{0LU, 1LU}, {2LU, 3LU}};
  EXPECT_EQ(llm_engine.CopyCache(copy_cache_param), ge::SUCCESS);
  PullCacheParam pull_cache_param{};
  pull_cache_param.prompt_blocks = {0LU,1LU};
  pull_cache_param.decoder_blocks = {2LU,3LU};
  EXPECT_EQ(llm_engine.PullCache(block_kv_cache.cache_id, cache_key, pull_cache_param), ge::SUCCESS);
  EXPECT_EQ(llm_engine.DeallocateCache(block_kv_cache.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_engine.DeallocateCache(prompt_kv_cache.cache_id), ge::SUCCESS);
}

TEST_F(LLMDatadistV2UTest, SafeFinalize) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::DECODER).NumStages(1).Build();
  llm::LLMDataDist llm_engine(0);
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);
  // Finalize by LLM
  llm_engine.LLMDataDistFinalize();
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);

  // mock 存在其他组件的session
  auto sess = llm::GeApi::GetInstance().NewSession({});
  // 不调用GEFinalize
  llm_engine.LLMDataDistFinalize();
  // 其他组件释放session
  sess.reset();
  // 当前session数为0, 不调用GEFinalize
  llm::GeApi::GetInstance().SafeFinalize();
}

TEST_F(LLMDatadistV2UTest, SwitchRoleFailed_OptionNotEnabled) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::DECODER).NumStages(1).Build();
  llm::LLMDataDist llm_engine(0);
  options[llm::LLM_OPTION_ROLE] = "Mix";
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);
  EXPECT_NE(llm_engine.SwitchRole("Decoder", {}), ge::SUCCESS);
}

TEST_F(LLMDatadistV2UTest, SwitchRoleFailed_LackRequiredOption) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::DECODER).NumStages(1).Build();
  llm::LLMDataDist llm_engine(0);
  options[llm::LLM_OPTION_ROLE] = "Mix";
  options[llm::LLM_OPTION_ENABLE_SWITCH_ROLE] = "1";
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);
  EXPECT_NE(llm_engine.SwitchRole("Prompt", {}), ge::SUCCESS);
}

TEST_F(LLMDatadistV2UTest, SwitchRoleSuccess) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::DECODER).NumStages(1).Build();
  llm::LLMDataDist llm_engine(0);
  options[llm::LLM_OPTION_ROLE] = "Mix";
  options[llm::LLM_OPTION_ENABLE_SWITCH_ROLE] = "1";
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);
  EXPECT_EQ(llm_engine.SwitchRole("Decoder", {}), ge::SUCCESS);
  std::map<std::string, std::string> switch_options;
  switch_options["llm.ListenIp"] = "0";
  switch_options["llm.ListenPort"] = "0";
  EXPECT_EQ(llm_engine.SwitchRole("Prompt", switch_options), ge::SUCCESS);
  EXPECT_EQ(llm_engine.SwitchRole("Mix", {}), ge::SUCCESS);
  EXPECT_EQ(llm_engine.SwitchRole("Prompt", switch_options), ge::SUCCESS);
  EXPECT_EQ(llm_engine.SwitchRole("Decoder", {}), ge::SUCCESS);
}

TEST_F(LLMDatadistV2UTest, Init_WithDeviceIdAndRankId) {
  ge::LlmEngineOptionBuilder builder;
  llm::LLMDataDist llm_engine(0);
  std::map<AscendString, AscendString> options {
      {OPTION_EXEC_DEVICE_ID, "1"},
      {OPTION_EXEC_RANK_ID, "1"},
      {LLM_OPTION_ROLE, "Decoder"},
      {LLM_OPTION_CLUSTER_INFO, R"({"cluster_id": 0, "logic_device_id": ["0:0:0:0"]})"},
  };
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);
}

TEST_F(LLMDatadistV2UTest, Init_WithDeviceId) {
  ge::LlmEngineOptionBuilder builder;
  llm::LLMDataDist llm_engine(0);
  std::map<AscendString, AscendString> options {
      {OPTION_EXEC_DEVICE_ID, "1"},
      {llm::LLM_OPTION_ROLE, "Decoder"},
      {llm::LLM_OPTION_CLUSTER_INFO, R"({"cluster_id": 0, "logic_device_id": ["0:0:0:0"]})"},
  };
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);
}

TEST_F(LLMDatadistV2UTest, SwapBlocks) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::PROMPT).NumStages(1).Build();

  options.emplace(ge::OPTION_EXEC_RANK_ID, "1");
  LLMDataDist llm_engine(0);
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);

  CacheDesc kv_desc{};
  kv_desc.num_tensors = 10;
  kv_desc.data_type = ge::DT_FLOAT16;
  kv_desc.shape = {10, 128};

  CacheKey cache_key{};
  cache_key.prompt_cluster_id = 0;
  cache_key.req_id = 1;
  cache_key.model_id = 0;

  Cache cached_tensors;
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc, cached_tensors, {cache_key}), ge::SUCCESS);
  EXPECT_EQ(cached_tensors.cache_id, 1);
  EXPECT_EQ(cached_tensors.per_device_tensor_addrs.size(), 1U);

  Cache cached_tensors_2;
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc, cached_tensors_2), ge::SUCCESS);
  EXPECT_EQ(cached_tensors_2.cache_id, 2);

  std::vector<std::pair<int64_t, int64_t>> block_mapping{{3,4}, {0,0}, {1,1}, {2,2}, {5,6}, {6,7}, {9,9}};
  // swap in
  EXPECT_EQ(llm_engine.SwapBlocks(cached_tensors, cached_tensors_2, 128, 0, block_mapping), ge::SUCCESS);
  // swap out
  EXPECT_EQ(llm_engine.SwapBlocks(cached_tensors_2, cached_tensors, 128, 1, block_mapping), ge::SUCCESS);

  // test deallocate success
  EXPECT_EQ(llm_engine.DeallocateCache(cached_tensors.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_engine.DeallocateCache(cached_tensors_2.cache_id), ge::SUCCESS);
}

TEST_F(LLMDatadistV2UTest, SwapBlocksFailed) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::PROMPT).NumStages(1).Build();

  options.emplace(ge::OPTION_EXEC_RANK_ID, "1");
  LLMDataDist llm_engine(0);
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);

  Cache cached_tensors;
  cached_tensors.cache_id = -1;
  cached_tensors.per_device_tensor_addrs = {{123}};

  std::vector<std::pair<int64_t, int64_t>> block_mapping{{0, 0}};

  // aclrtMalloc failed
  EXPECT_EQ(llm_engine.SwapBlocks(cached_tensors, cached_tensors, 123, 0, block_mapping), ge::LLM_DEVICE_OUT_OF_MEMORY);
}

TEST_F(LLMDatadistV2UTest, SwapBlocksFailed1) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::PROMPT).NumStages(1).Build();

  options.emplace(ge::OPTION_EXEC_RANK_ID, "1");
  LLMDataDist llm_engine(0);
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);

  CacheDesc kv_desc{};
  kv_desc.num_tensors = 2;
  kv_desc.data_type = ge::DT_FLOAT16;
  kv_desc.shape = {10, 128};

  CacheKey cache_key{};
  cache_key.prompt_cluster_id = 0;
  cache_key.req_id = 1;
  cache_key.model_id = 0;

  Cache cached_tensors;
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc, cached_tensors, {cache_key}), ge::SUCCESS);
  EXPECT_EQ(cached_tensors.cache_id, 1);
  EXPECT_EQ(cached_tensors.per_device_tensor_addrs.size(), 1U);

  std::vector<std::pair<int64_t, int64_t>> block_mapping{{3, 4}, {0, 0}, {1, 1}, {2, 2}, {5, 6}, {6, 7}, {9, 9}};
  const char_t * const kEnvValue = "SET_TRANS_VAR_DATA";
  // 设置环境变量，rtCtxSetCurrent返回失败
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  // swap in
  EXPECT_EQ(llm_engine.SwapBlocks(cached_tensors, cached_tensors, 128, 0, block_mapping), ge::FAILED);

  // 清理环境变量
  mmSetEnv(kEnvValue, "", 1);
}
}  // namespace llm
