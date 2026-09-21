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

#include "llm_datadist.h"
#include "common/llm_ge_api.h"
#include "cache_engine_stubs.h"
#include "dlog_pub.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "llm_engine_test_helper.h"
#include "common/mem_utils.h"
#include "llm_test_helper.h"
#include "common/llm_inner_types.h"

using namespace std;
using namespace ::testing;
using ::testing::Invoke;
using ::testing::Mock;
namespace ge {
namespace {
const std::string PNE_ID_NPU = "NPU";
class GeApiStub1 : public ge::DefaultGeApiStub {
 public:
  std::unique_ptr<GeApi> NewSession(const map<ge::AscendString, ge::AscendString> &options) override {
    return llm::MakeUnique<llm::CacheEngineGeApi>();
  }
};
}  // namespace
class LLMDataDistV2STest : public ::testing::Test {
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
      (void)mmRealPath("../tests/dflow/llm_datadist/st/testcase/llm_datadist/json_file/numa_config_prompt.json", numa_config_path, 4096);
    } else {
      (void)mmRealPath("../tests/dflow/llm_datadist/st/testcase/llm_datadist/json_file/numa_config_decoder.json", numa_config_path, 4096);
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

TEST_F(LLMDataDistV2STest, Prompt) {
  uint64_t prompt_cluster_id = 0U;
  std::string model_name = "model_" + std::to_string(0);
  // 测试 LLMDataDist 类的初始化
  std::vector<int64_t> dim{8192, 8192};
  std::string deploy_cluster_path =
      "{\"cluster_id\":0,\"logic_device_id\":[\"0:0:0:0\", "
      "\"0:0:1:0\"],\"listen_ip_info\":[{\"ip\":176163329,\"port\":6000},{\"ip\":176163329,\"port\":6000}]}";
  auto llm_options = GetOptions("Prompt", deploy_cluster_path);
  // for dynamic shape test

  char_t flow_func_workspace[4096];
  (void)realpath("../dflow/llm_datadist/flow_func/", flow_func_workspace);
  setenv("LLM_ENGINE_WORKSPACE", flow_func_workspace, 1);
  llm::LLMDataDist llm_engine(prompt_cluster_id);
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(llm_options), ge::SUCCESS);

  std::vector<ge::Status> rets;
  std::vector<llm::ClusterInfo> clusters;
  llm::ClusterInfo cluster_info;
  llm::IpInfo ip_info;
  cluster_info.local_ip_infos = {ip_info, ip_info};
  cluster_info.remote_ip_infos = {ip_info, ip_info};
  clusters.emplace_back(cluster_info);

  llm::CacheDesc kv_desc{};
  kv_desc.num_tensors = 80;
  kv_desc.data_type = ge::DT_FLOAT16;
  kv_desc.shape = {1, 256};
  llm::Cache cached_tensors;
  std::vector<llm::CacheKey> cache_keys;
  cache_keys.emplace_back(llm::CacheKey{0, 1, 0});
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc, cached_tensors, cache_keys), ge::SUCCESS);
  EXPECT_EQ(llm_engine.DeallocateCache(cached_tensors.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_engine.UnlinkClusters(clusters, rets, 1), ge::SUCCESS);
}

TEST_F(LLMDataDistV2STest, Decoder) {
  setenv("HCCL_RDMA_TC", "236", 1);
  setenv("HCCL_RDMA_SL", "5", 1);
  uint64_t prompt_cluster_id = 0U;
  std::string model_name = "model_" + std::to_string(0);
  // 测试 LLMDataDist 类的初始化
  std::vector<int64_t> dim{8192, 8192};
  std::string deploy_cluster_path =
      "{\"cluster_id\":0,\"logic_device_id\":[\"0:0:0:0\", "
      "\"0:0:1:0\"],\"listen_ip_info\":[{\"ip\":176163329,\"port\":6000},{\"ip\":176163329,\"port\":6000}]}";
  auto llm_options = GetOptions("Decoder", deploy_cluster_path);
  llm_options["llm.BufPoolCfg"] = "{\"buf_cfg\":[{\"total_size\":2097152,\"blk_size\":256,\"max_buf_size\":8192}]}";
  // for dynamic shape test

  char_t flow_func_workspace[4096];
  (void)realpath("../dflow/llm_datadist/flow_func/", flow_func_workspace);
  setenv("LLM_ENGINE_WORKSPACE", flow_func_workspace, 1);
  llm::LLMDataDist llm_engine(prompt_cluster_id);
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(llm_options), ge::SUCCESS);

  std::vector<ge::Status> rets;
  std::vector<llm::ClusterInfo> clusters;
  llm::ClusterInfo cluster_info;
  llm::IpInfo ip_info;
  cluster_info.local_ip_infos = {ip_info, ip_info};
  cluster_info.remote_ip_infos = {ip_info, ip_info};
  clusters.emplace_back(cluster_info);
  EXPECT_EQ(llm_engine.LinkClusters(clusters, rets, 1), ge::SUCCESS);

  llm::CacheDesc tmp_kv_desc{};
  tmp_kv_desc.num_tensors = 80;
  tmp_kv_desc.data_type = ge::DT_FLOAT16;
  tmp_kv_desc.shape = {1, 256};
  llm::Cache tmp_cached_tensors;
  llm::CacheKey cache_key{prompt_cluster_id, 1, 0};

  llm::CacheDesc kv_desc{};
  kv_desc.num_tensors = 80;
  kv_desc.data_type = ge::DT_FLOAT16;
  kv_desc.shape = {4, 256};
  llm::Cache cached_tensors;

  EXPECT_EQ(llm_engine.AllocateCache(tmp_kv_desc, tmp_cached_tensors, {}), ge::SUCCESS);
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc, cached_tensors, {}), ge::SUCCESS);
  llm::PullCacheParam pull_cache_param{};
  pull_cache_param.prompt_blocks = {0LU, 1LU};
  pull_cache_param.decoder_blocks = {2LU, 3LU};
  EXPECT_EQ(llm_engine.PullCache(tmp_cached_tensors.cache_id, cache_key, pull_cache_param), ge::SUCCESS);

  // test pull by cache id
  llm::CacheKey cache_key_by_id{};
  cache_key_by_id.prompt_cluster_id = 0;
  cache_key_by_id.prompt_cache_id = cached_tensors.cache_id;
  cache_key_by_id.prompt_batch_index = 0;
  EXPECT_EQ(llm_engine.PullCache(cached_tensors.cache_id, cache_key_by_id, {}), ge::SUCCESS);

  llm::CopyCacheParam copy_cache_param{};
  copy_cache_param.src_cache_id = tmp_cached_tensors.cache_id;
  copy_cache_param.dst_cache_id = cached_tensors.cache_id;
  copy_cache_param.dst_batch_index = 1;
  copy_cache_param.copy_block_infos = {{0, 2}, {1, 3}};
  EXPECT_EQ(llm_engine.CopyCache(copy_cache_param), ge::SUCCESS);
  std::vector<ge::Tensor> output_tensors;
  EXPECT_EQ(llm_engine.GetCacheTensors(cached_tensors.cache_id, output_tensors, 0), ge::SUCCESS);
  EXPECT_EQ(output_tensors.size(), 2U);
  EXPECT_EQ(llm_engine.DeallocateCache(tmp_cached_tensors.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_engine.DeallocateCache(cached_tensors.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_engine.UnlinkClusters(clusters, rets, 1), ge::SUCCESS);
  unsetenv("HCCL_RDMA_TC");
  unsetenv("HCCL_RDMA_SL");
}

TEST_F(LLMDataDistV2STest, SafeFinalize) {

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

TEST_F(LLMDataDistV2STest, SwitchRoleFailed_OptionNotEnabled) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::DECODER).NumStages(1).Build();
  llm::LLMDataDist llm_engine(0);
  options[llm::LLM_OPTION_ROLE] = "Mix";
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);
  EXPECT_NE(llm_engine.SwitchRole("Decoder", {}), ge::SUCCESS);
}

TEST_F(LLMDataDistV2STest, SwitchRoleFailed_LackRequiredOption) {
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::DECODER).NumStages(1).Build();
  llm::LLMDataDist llm_engine(0);
  options[llm::LLM_OPTION_ROLE] = "Mix";
  options[llm::LLM_OPTION_ENABLE_SWITCH_ROLE] = "1";
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);
  EXPECT_NE(llm_engine.SwitchRole("Prompt", {}), ge::SUCCESS);
}

TEST_F(LLMDataDistV2STest, SwitchRoleSuccess) {
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

TEST_F(LLMDataDistV2STest, Init_WithDeviceIdAndRankId) {
  ge::LlmEngineOptionBuilder builder;
  llm::LLMDataDist llm_engine(0);
  std::map<AscendString, AscendString> options {
      {OPTION_EXEC_DEVICE_ID, "1"},
      {OPTION_EXEC_RANK_ID, "1"},
      {llm::LLM_OPTION_ROLE, "Decoder"},
      {llm::LLM_OPTION_CLUSTER_INFO, R"({"cluster_id": 0, "logic_device_id": ["0:0:0:0"]})"},
  };
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);
}

TEST_F(LLMDataDistV2STest, Init_WithDeviceId) {
  ge::LlmEngineOptionBuilder builder;
  llm::LLMDataDist llm_engine(0);
  std::map<AscendString, AscendString> options {
      {OPTION_EXEC_DEVICE_ID, "1"},
      {llm::LLM_OPTION_ROLE, "Decoder"},
      {llm::LLM_OPTION_CLUSTER_INFO, R"({"cluster_id": 0, "logic_device_id": ["0:0:0:0"]})"},
  };
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);
}

TEST_F(LLMDataDistV2STest, SwapBlocks) {
  llm::GeApi::instance_.reset(new GeApiStub1());
  ge::LlmEngineOptionBuilder builder;
  auto options = builder.Role(llm::RoleType::PROMPT).NumStages(1).Build();

  options.emplace(ge::OPTION_EXEC_RANK_ID, "1");
  llm::LLMDataDist llm_engine(0);
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::SUCCESS);

  llm::CacheDesc kv_desc{};
  kv_desc.num_tensors = 80;
  kv_desc.data_type = ge::DT_FLOAT16;
  kv_desc.shape = {10, 128};

  llm::CacheKey cache_key{};
  cache_key.prompt_cluster_id = 0;
  cache_key.req_id = 1;
  cache_key.model_id = 0;

  llm::Cache cached_tensors;
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc, cached_tensors, {cache_key}), ge::SUCCESS);
  EXPECT_EQ(cached_tensors.cache_id, 1);
  EXPECT_EQ(cached_tensors.per_device_tensor_addrs.size(), 1U);

  llm::Cache cached_tensors_2;
  EXPECT_EQ(llm_engine.AllocateCache(kv_desc, cached_tensors_2), ge::SUCCESS);
  EXPECT_EQ(cached_tensors_2.cache_id, 2);

  std::vector<std::pair<int64_t, int64_t>> block_mapping{{3, 4}, {0, 0}, {1, 1}, {2, 2}, {5, 6}, {6, 7}, {9, 9}};
  // swap in
  EXPECT_EQ(llm_engine.SwapBlocks(cached_tensors, cached_tensors_2, 128, 0, block_mapping), ge::SUCCESS);
  // swap out
  EXPECT_EQ(llm_engine.SwapBlocks(cached_tensors_2, cached_tensors, 128, 1, block_mapping), ge::SUCCESS);

  // test deallocate success
  EXPECT_EQ(llm_engine.DeallocateCache(cached_tensors.cache_id), ge::SUCCESS);
  EXPECT_EQ(llm_engine.DeallocateCache(cached_tensors_2.cache_id), ge::SUCCESS);
  llm::GeApi::instance_.reset(new llm::CacheEngineGeApi());
}

TEST_F(LLMDataDistV2STest, Init_Failed) {
  ge::LlmEngineOptionBuilder builder;
  llm::LLMDataDist llm_engine(0);
  std::map<AscendString, AscendString> options {
      {OPTION_EXEC_DEVICE_ID, "1"},
      {OPTION_EXEC_RANK_ID, "1"},
      {llm::LLM_OPTION_ROLE, "Decoder"},
  };
  EXPECT_EQ(llm_engine.LLMDataDistInitialize(options), ge::LLM_PARAM_INVALID);
}

}  // namespace ge
