/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <list>
#include "fe_llt_utils.h"
#define protected public
#define private public
#include "fusion_manager/fusion_manager.h"
#include "platform/platform_info.h"
#include "common/platform_utils.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "adapter/common/op_store_adapter_manager.h"
#undef private
#undef protected
using namespace std;
using namespace fe;

bool QueryOpPattern_True(std::vector<std::pair<std::string, std::string>> &ops_patterns) {
  ops_patterns.emplace_back(std::make_pair("Sqrt", "Elewise"));
  ops_patterns.emplace_back(std::make_pair("Square", "Elewise"));
  return true;
}

class fusion_manager_unittest : public testing::Test
{
protected:
  static void TearDownTestCase()
  {
    string path = GetCodeDir() + "/tests/engines/nn_engine/config/data/platform_config";
    string real_path = fe::RealPath(path);
    fe::PlatformInfoManager::Instance().platform_info_map_.clear();
    fe::PlatformInfoManager::Instance().platform_infos_map_.clear();
    uint32_t init_ret = fe::PlatformInfoManager::Instance().LoadConfigFile(real_path);
    EXPECT_EQ(init_ret, 0);
    fe::PlatformInfoManager::Instance().init_flag_ = true;
    fe::PlatformInfoManager::Instance().opti_compilation_info_.soc_version = "Ascend910B1";
    fe::PlatformInfoManager::Instance().opti_compilation_infos_.Init();
    fe::PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B1");

    std::map<string, string> options;
    fe::OpStoreAdapterManager::Instance(fe::AI_CORE_NAME).Initialize(options);
    fe::OpStoreAdapterManager::Instance(fe::kDsaCoreName).Initialize(options);
  }

// AUTO GEN PLEASE DO NOT MODIFY IT
};

TEST_F(fusion_manager_unittest, initialize_and_finalize)
{
  FusionManager fm;

  map<string, string> options;
  std::string engine_name;
  std::string soc_version;
  Status ret = fm.Initialize(engine_name, options);
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = fm.Initialize(engine_name, options);
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = fm.Finalize();
  EXPECT_EQ(ret, fe::SUCCESS);
  ret = fm.Finalize();
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(fusion_manager_unittest, dsa_instance)
{
    FusionManager fm = FusionManager::Instance(kDsaCoreName);
    fm.ops_kernel_info_store_ = make_shared<FEOpsKernelInfoStore>(fe::kDsaCoreName);
    fm.dsa_graph_opt_ = make_shared<DSAGraphOptimizer>(fm.ops_kernel_info_store_, fe::kDsaCoreName);
    map<string, GraphOptimizerPtr> graph_optimizers;
    std::string AIcoreEngine = "DSAEngine";
    fm.GetGraphOptimizerObjs(graph_optimizers, AIcoreEngine);
    EXPECT_EQ(graph_optimizers.size(), 1);
}

TEST_F(fusion_manager_unittest, fm_init)
{
    FusionManager fm = FusionManager::Instance(kDsaCoreName);
    fm.ops_kernel_info_store_ = make_shared<FEOpsKernelInfoStore>(fe::kDsaCoreName);
    fm.dsa_graph_opt_ = make_shared<DSAGraphOptimizer>(fm.ops_kernel_info_store_, fe::kDsaCoreName);
    map<string, GraphOptimizerPtr> graph_optimizers;
    std::string AIcoreEngine = "DSAEngine";
    fm.GetGraphOptimizerObjs(graph_optimizers, AIcoreEngine);
    EXPECT_EQ(graph_optimizers.size(), 1);
}

TEST_F(fusion_manager_unittest, get_ops_kernel_info_stores_success)
{
    FusionManager fm;
    map<string, OpsKernelInfoStorePtr> op_kern_infos;
    fm.ops_kernel_info_store_ = make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
    std::string AIcoreEngine = "AIcoreEngine";
    fm.GetOpsKernelInfoStores(op_kern_infos, AIcoreEngine);
    EXPECT_EQ(op_kern_infos.size(), 1);
}

TEST_F(fusion_manager_unittest, get_graph_optimizer_objs_failed)
{
    FusionManager fm;
    fm.ops_kernel_info_store_ = make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
    map<string, GraphOptimizerPtr> graph_optimizers;
    std::string AIcoreEngine = "AIcoreEngine";
    fm.graph_opt_ = nullptr;
    fm.GetGraphOptimizerObjs(graph_optimizers, AIcoreEngine);
    EXPECT_EQ(graph_optimizers.size(), 0);
}

TEST_F(fusion_manager_unittest, get_graph_optimizer_objs_success)
{
    FusionManager fm;
    fm.ops_kernel_info_store_ = make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
    fm.graph_opt_ = make_shared<FEGraphOptimizer>(fm.ops_kernel_info_store_, fe::AI_CORE_NAME);
    map<string, GraphOptimizerPtr> graph_optimizers;
    std::string AIcoreEngine = "AIcoreEngine";
    fm.GetGraphOptimizerObjs(graph_optimizers, AIcoreEngine);
    EXPECT_EQ(graph_optimizers.size(), 1);
}

TEST_F(fusion_manager_unittest, platform_info_manager_test2)
{
    PlatformInfoManager pm;
    std::string path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_manager/platform_config_new";
    std::string real_path = RealPath(path);
    Status ret = pm.LoadConfigFile(real_path);
    EXPECT_EQ(ret, fe::FAILED);

    path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_manager";
    real_path = RealPath(path);
    ret = pm.LoadConfigFile(real_path);
    EXPECT_EQ(ret, fe::FAILED);

    path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_manager/Ascend000.ini";
    real_path = RealPath(path);
    ret = pm.LoadIniFile(real_path);
    EXPECT_EQ(ret, fe::FAILED);


    map<string, string> new_map;
    map<string, map<string, string>> content_map1_;
    content_map1_.emplace("test", new_map);
    ret = pm.AssemblePlatformInfoVector(content_map1_);
    EXPECT_EQ(ret, fe::SUCCESS);

    pm.init_flag_ = true;
    ret = pm.InitializePlatformInfo();
    EXPECT_EQ(ret, fe::SUCCESS);

    pm.init_flag_ = true;
    ret = pm.Finalize();
    EXPECT_EQ(ret, fe::SUCCESS);
    ret = pm.Finalize();
    EXPECT_EQ(ret, fe::SUCCESS);

    ret = pm.Finalize();
    EXPECT_EQ(ret, fe::SUCCESS);

    string str = "";
    pm.Trim(str);

    str = " ";
    pm.Trim(str);

    path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_manager/Ascend001.ini";
    real_path = RealPath(path);
    ret = pm.LoadIniFile(real_path);
    EXPECT_EQ(ret, fe::FAILED);

    path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_manager/Ascend002.ini";
    real_path = RealPath(path);
    ret = pm.LoadIniFile(real_path);
    EXPECT_EQ(ret, fe::FAILED);

    map<string, string> ai_coreintrinsic_dtype_map;
    PlatformInfo platform_info_temp;
    ai_coreintrinsic_dtype_map.emplace(make_pair("1", "Intrinsic_vcmpv_ne"));
    pm.ParseAICoreintrinsicDtypeMap(ai_coreintrinsic_dtype_map, platform_info_temp);
    pm.ParseVectorCoreintrinsicDtypeMap(ai_coreintrinsic_dtype_map, platform_info_temp);

    map<string, string> content_info_temp;
    content_info_temp.emplace(make_pair("123", "456"));
    map<string, map<string, string>> content_info_map;
    content_info_map.emplace(make_pair("test", content_info_temp));
    string soc_version;
    ret = pm.ParsePlatformInfoFromStrToStruct(content_info_map, soc_version, platform_info_temp);
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(fusion_manager_unittest, platform_info_manager_test3)
{
    PlatformInfoManager pm;
    map<string, string> ai_core_spec_map;
    ai_core_spec_map.emplace(make_pair("unzip_engines", "1"));
    ai_core_spec_map.emplace(make_pair("unzip_max_ratios", "64"));
    ai_core_spec_map.emplace(make_pair("unzip_channels", "4"));
    ai_core_spec_map.emplace(make_pair("unzip_is_tight", "0"));
    // AICoreSpec
    map<string, string> content_info_temp;
    content_info_temp.emplace(make_pair("123", "456"));
    map<string, map<string, string>> content_info_map;
    content_info_map.emplace(make_pair("test", content_info_temp));
    content_info_map.emplace(make_pair("AICoreSpec", ai_core_spec_map));
    string soc_version;
    PlatformInfo platform_info_temp;
    Status ret = pm.ParsePlatformInfoFromStrToStruct(content_info_map, soc_version, platform_info_temp);
    EXPECT_EQ(fe::SUCCESS, ret);
    EXPECT_EQ(platform_info_temp.ai_core_spec.unzip_engines, 1);
    EXPECT_EQ(platform_info_temp.ai_core_spec.unzip_max_ratios, 64);
    EXPECT_EQ(platform_info_temp.ai_core_spec.unzip_channels, 4);
    EXPECT_EQ(platform_info_temp.ai_core_spec.unzip_is_tight, 0);
}

TEST_F(fusion_manager_unittest, platform_info_manager_testfieldvalue)
{
  string path = GetCodeDir() + "/tests/engines/nn_engine/config/data/platform_config";
  string real_path = RealPath(path);
  PlatformInfoManager::Instance().platform_info_map_.clear();
  PlatformInfoManager::Instance().platform_infos_map_.clear();
  uint32_t init_ret = PlatformInfoManager::Instance().LoadConfigFile(real_path);
  EXPECT_EQ(init_ret, ge::GRAPH_SUCCESS);

  PlatformInfo platform_info;
  OptionalInfo opti_compilation_info;

  uint32_t get_ret = PlatformInfoManager::Instance().GetPlatformInfo("Ascend310P3", platform_info, opti_compilation_info);
  EXPECT_EQ(init_ret, ge::GRAPH_SUCCESS);

  EXPECT_EQ(platform_info.str_info.aic_version, "AIC-M-200");
  EXPECT_EQ(platform_info.str_info.ccec_aic_version, "dav-m200");
  EXPECT_EQ(platform_info.str_info.ccec_aiv_version, "dav-m200-vec");
  EXPECT_EQ(platform_info.str_info.is_support_ai_cpu_compiler, "true");

  EXPECT_EQ(platform_info.soc_info.ai_core_cnt, 8);
  EXPECT_EQ(platform_info.soc_info.vector_core_cnt, 7);
  EXPECT_EQ(platform_info.soc_info.ai_cpu_cnt, 16);
  EXPECT_EQ(platform_info.soc_info.l2_size, 16777216);
  EXPECT_EQ(platform_info.soc_info.l2PageNum, 64);

  EXPECT_EQ(platform_info.ai_core_spec.cube_freq, 1060);
  EXPECT_EQ(platform_info.ai_core_spec.cube_m_size, 16);
  EXPECT_EQ(platform_info.ai_core_spec.cube_n_size, 16);
  EXPECT_EQ(platform_info.ai_core_spec.cube_k_size, 16);
  EXPECT_EQ(platform_info.ai_core_spec.vec_calc_size, 128);
  EXPECT_EQ(platform_info.ai_core_spec.l0_a_size, 65536);
  EXPECT_EQ(platform_info.ai_core_spec.l0_b_size, 65536);
  EXPECT_EQ(platform_info.ai_core_spec.l0_c_size, 262144);
  EXPECT_EQ(platform_info.ai_core_spec.l1_size, 1048576);
  EXPECT_EQ(platform_info.ai_core_spec.smask_buffer, 256);
  EXPECT_EQ(platform_info.ai_core_spec.ub_size, 262144);
  EXPECT_EQ(platform_info.ai_core_spec.ubblock_size, 32);
  EXPECT_EQ(platform_info.ai_core_spec.ubbank_size, 4096);
  EXPECT_EQ(platform_info.ai_core_spec.ubbank_num, 64);
  EXPECT_EQ(platform_info.ai_core_spec.ubburst_in_one_block, 32);
  EXPECT_EQ(platform_info.ai_core_spec.ubbank_group_num, 16);
  EXPECT_EQ(platform_info.ai_core_spec.unzip_engines, 4);
  EXPECT_EQ(platform_info.ai_core_spec.unzip_max_ratios, 64);
  EXPECT_EQ(platform_info.ai_core_spec.unzip_channels, 2);
  EXPECT_EQ(platform_info.ai_core_spec.unzip_is_tight, 1);

  EXPECT_EQ(platform_info.ai_core_memory_rates.ddr_rate, 17);
  EXPECT_EQ(platform_info.ai_core_memory_rates.ddr_read_rate, 17);
  EXPECT_EQ(platform_info.ai_core_memory_rates.ddr_write_rate, 17);
  EXPECT_EQ(platform_info.ai_core_memory_rates.l2_rate, 114);
  EXPECT_EQ(platform_info.ai_core_memory_rates.l2_read_rate, 114);
  EXPECT_EQ(platform_info.ai_core_memory_rates.l2_write_rate, 114);
  EXPECT_EQ(platform_info.ai_core_memory_rates.l1_to_l0_a_rate, 512);
  EXPECT_EQ(platform_info.ai_core_memory_rates.l1_to_l0_b_rate, 256);
  EXPECT_EQ(platform_info.ai_core_memory_rates.l1_to_ub_rate, 128);
  EXPECT_EQ(platform_info.ai_core_memory_rates.l0_c_to_ub_rate, 512);
  EXPECT_EQ(platform_info.ai_core_memory_rates.ub_to_l2_rate, 114);
  EXPECT_EQ(platform_info.ai_core_memory_rates.ub_to_ddr_rate, 17);
  EXPECT_EQ(platform_info.ai_core_memory_rates.ub_to_l1_rate, 256);

  EXPECT_EQ(platform_info.cpucache.AICPUSyncBySW, 0);
  EXPECT_EQ(platform_info.cpucache.TSCPUSyncBySW, 0);

  EXPECT_EQ(platform_info.vector_core_spec.vec_freq, 1000);
  EXPECT_EQ(platform_info.vector_core_spec.vec_calc_size, 128);
  EXPECT_EQ(platform_info.vector_core_spec.smask_buffer, 0);
  EXPECT_EQ(platform_info.vector_core_spec.ub_size, 262144);
  EXPECT_EQ(platform_info.vector_core_spec.ubblock_size, 32);
  EXPECT_EQ(platform_info.vector_core_spec.ubbank_size, 4096);
  EXPECT_EQ(platform_info.vector_core_spec.ubbank_num, 64);
  EXPECT_EQ(platform_info.vector_core_spec.ubburst_in_one_block, 32);
  EXPECT_EQ(platform_info.vector_core_spec.ubbank_group_num, 16);
  EXPECT_EQ(platform_info.vector_core_spec.vector_reg_size, 2048);
  EXPECT_EQ(platform_info.vector_core_spec.predicate_reg_size, 256);
//  EXPECT_EQ(platform_info.vector_core_spec.address_reg_size, 32);
//  EXPECT_EQ(platform_info.vector_core_spec.alignment_reg_size, 257);

  EXPECT_EQ(platform_info.vector_core_memory_rates.ddr_rate, 9);
  EXPECT_EQ(platform_info.vector_core_memory_rates.ddr_read_rate, 9);
  EXPECT_EQ(platform_info.vector_core_memory_rates.ddr_write_rate, 9);
  EXPECT_EQ(platform_info.vector_core_memory_rates.l2_rate, 57);
  EXPECT_EQ(platform_info.vector_core_memory_rates.l2_read_rate, 57);
  EXPECT_EQ(platform_info.vector_core_memory_rates.l2_write_rate, 57);
  EXPECT_EQ(platform_info.vector_core_memory_rates.ub_to_l2_rate, 57);
  EXPECT_EQ(platform_info.vector_core_memory_rates.ub_to_ddr_rate, 9);

  PlatFormInfos platform_infos;
  OptionalInfos opti_compilation_infos;
  get_ret = PlatformInfoManager::Instance().GetPlatformInfos("Ascend310P3", platform_infos, opti_compilation_infos);
  EXPECT_EQ(init_ret, ge::GRAPH_SUCCESS);
  string label = "version";
  string key = "AIC_version";
  string val;
  (void)platform_infos.GetPlatformRes(label, key, val);
  EXPECT_EQ(val, "AIC-M-200");

  key = "CCEC_AIC_version";
  val = "";
  (void)platform_infos.GetPlatformRes(label, key, val);
  EXPECT_EQ(val, "dav-m200");

  label = "SoCInfo";
  key = "ai_core_cnt";
  (void)platform_infos.GetPlatformRes(label, key, val);
  EXPECT_EQ(val, "8");

  key = "max_stream_num";
  (void)platform_infos.GetPlatformRes(label, key, val);
  EXPECT_EQ(val, "1024");

  key = "max_hardware_eventid_num";
  (void)platform_infos.GetPlatformRes(label, key, val);
  EXPECT_EQ(val, "1024");

  key = "max_notifyid_num";
  (void)platform_infos.GetPlatformRes(label, key, val);
  EXPECT_EQ(val, "1024");

  key = "max_modelid_num";
  (void)platform_infos.GetPlatformRes(label, key, val);
  EXPECT_EQ(val, "1024");

  label = "AICoreSpec";
  key = "cube_freq";
  (void)platform_infos.GetPlatformRes(label, key, val);
  EXPECT_EQ(val, "1060");

  label = "AICoreMemoryRates";
  key = "ddr_rate";
  (void)platform_infos.GetPlatformRes(label, key, val);
  EXPECT_EQ(val, "17");

  label = "VectorCoreSpec";
  key = "ub_size";
  (void)platform_infos.GetPlatformRes(label, key, val);
  EXPECT_EQ(val, "262144");

  label = "VectorCoreMemoryRates";
  key = "ub_to_ddr_rate";
  (void)platform_infos.GetPlatformRes(label, key, val);
  EXPECT_EQ(val, "9");
}

TEST_F(fusion_manager_unittest, init_platform_info_test)
{
    PlatformInfoManager::Instance().platform_info_map_.clear();
    PlatformInfoManager::Instance().platform_infos_map_.clear();
    PlatformInfoManager::Instance().init_flag_ = true;
    PlatformInfo tmp;
    PlatFormInfos tmp_platform_infos;
    (void)tmp_platform_infos.Init();
    PlatformInfoManager::Instance().platform_info_map_.emplace("Ascend310", tmp);
    PlatformInfoManager::Instance().platform_info_map_.emplace("Ascend910B", tmp);
    PlatformInfoManager::Instance().platform_infos_map_.emplace("Ascend310", tmp_platform_infos);
    PlatformInfoManager::Instance().platform_infos_map_.emplace("Ascend910B", tmp_platform_infos);

    PlatformInfoManager::Instance().platform_info_map_.clear();
    PlatformInfoManager::Instance().platform_infos_map_.clear();
    PlatformInfoManager::Instance().init_flag_ = false;
    EXPECT_EQ(PlatformInfoManager::Instance().init_flag_, false);
}

TEST_F(fusion_manager_unittest, platform_info_manager_test6) {
  string path = GetCodeDir() + "/tests/engines/nn_engine/config/data/platform_config";
  string real_path = RealPath(path);
  PlatformInfoManager::Instance().platform_info_map_.clear();
  PlatformInfoManager::Instance().platform_infos_map_.clear();
  uint32_t init_ret = PlatformInfoManager::Instance().LoadConfigFile(real_path);
  EXPECT_EQ(init_ret, ge::GRAPH_SUCCESS);

  PlatformInfo platform_info;
  OptionalInfo opti_compilation_info;
  uint32_t get_ret =
      PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(
          platform_info, opti_compilation_info);
  PlatformInfoManager::Instance().platform_info_map_.clear();
  PlatformInfoManager::Instance().platform_infos_map_.clear();
  EXPECT_EQ(init_ret, ge::GRAPH_SUCCESS);
}

TEST_F(fusion_manager_unittest, init_bin_kernel_info)
{
  FusionManager fm = FusionManager::Instance(kDsaCoreName);
  std::string AIcoreEngine = "DSAEngine";
  auto res = fm.InitBinKernelInfoStore(AIcoreEngine);
  EXPECT_EQ(res, fe::SUCCESS);
}

TEST_F(fusion_manager_unittest, update_ops_kernel)
{
  FusionManager fm = FusionManager::Instance(AI_CORE_NAME);
  map<string, string> options;
  fm.inited_ = false;

  OpStoreAdapterPtr my_adapter = std::make_shared<TbeOpStoreAdapter>(AI_CORE_NAME);
  (void)OpStoreAdapterManager::Instance(AI_CORE_NAME).map_all_op_store_adapter_.emplace(std::make_pair("tbe_op_adapter", my_adapter));
  auto tbe_adapter_ptr = std::dynamic_pointer_cast<TbeOpStoreAdapter>(my_adapter);
  tbe_adapter_ptr->QueryOpPattern = QueryOpPattern_True;

  auto res = fm.Initialize(AI_CORE_NAME, options);
  EXPECT_EQ(res, fe::SUCCESS);
}
