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
#include <string>
#include <vector>
#include "common/fe_inner_attr_define.h"
#include "common/aicore_util_constants.h"
#include "mmpa/src/mmpa_stub.h"
#define protected public
#define private public
#include "common/fusion_statistic/buffer_fusion_info_collecter.h"
#include "common/fusion_statistic/fusion_statistic_writer.h"
#include "common/configuration.h"
#include "fe_llt_utils.h"
#undef protected
#undef private

using namespace std;
using namespace fe;
using namespace ge;

class BUFFER_FUSION_INFO_COLLECTOR_STEST: public testing::Test {
 protected:
  void SetUp() {
  }

  void TearDown() {
  }
};

TEST_F(BUFFER_FUSION_INFO_COLLECTOR_STEST, get_pass_name_of_scope_id_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddOutputDesc(out_desc);
  ge::AttrUtils::SetInt(op_desc_0, SCOPE_ID_ATTR, 1);
  ge::AttrUtils::SetStr(op_desc_0, "pass_name", "add_pass_name");
  NodePtr node_0 = graph->AddNode(op_desc_0);
  PassNameIdMap pass_name_scope_id_map;

  BufferFusionInfoCollecter buffer_fusion_info_collecter;
  Status ret = buffer_fusion_info_collecter.GetPassNameOfScopeId(*graph, pass_name_scope_id_map);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(BUFFER_FUSION_INFO_COLLECTOR_STEST, get_pass_name_of_failed_id_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddOutputDesc(out_desc);
  ge::AttrUtils::SetInt(op_desc_0, "fusion_failed", 1);
  ge::AttrUtils::SetStr(op_desc_0, "pass_name", "add_pass_name");
  NodePtr node_0 = graph->AddNode(op_desc_0);
  PassNameIdMap pass_name_fusion_failed_id_map;

  BufferFusionInfoCollecter buffer_fusion_info_collecter;
  Status ret = buffer_fusion_info_collecter.GetPassNameOfFailedId(*graph, pass_name_fusion_failed_id_map);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(BUFFER_FUSION_INFO_COLLECTOR_STEST, count_buffer_fusion_times_test) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddOutputDesc(out_desc);
  ge::AttrUtils::SetInt(op_desc_0, SCOPE_ID_ATTR, 1);
  ge::AttrUtils::SetStr(op_desc_0, "pass_name", "add_pass_name");
  NodePtr node_0 = graph->AddNode(op_desc_0);
  std::set<std::string> pass_name_set = {"match", "effect"};
  BufferFusionInfoCollecter buffer_fusion_info_collecter;
  buffer_fusion_info_collecter.SetPassName(*graph, pass_name_set);
  PassNameIdMap pass_name_scope_id_map;
  Status ret = buffer_fusion_info_collecter.GetPassNameOfScopeId(*graph, pass_name_scope_id_map);
  EXPECT_EQ(ret, fe::SUCCESS);
  PassNameIdMap pass_name_fusion_failed_id_map;
  ret = buffer_fusion_info_collecter.GetPassNameOfFailedId(*graph, pass_name_fusion_failed_id_map);
  EXPECT_EQ(ret, fe::SUCCESS);
  std::map<std::string, int32_t> pass_match_map;
  pass_match_map["match"] = 1;
  std::map<std::string, int32_t> pass_effect_map;
  pass_match_map["effect"] = 1;
  ret = buffer_fusion_info_collecter.CountBufferFusionTimes(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(BUFFER_FUSION_INFO_COLLECTOR_STEST, count_buffer_fusion_times_test2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddOutputDesc(out_desc);
  ge::AttrUtils::SetInt(op_desc_0, SCOPE_ID_ATTR, 1);
  ge::AttrUtils::SetStr(op_desc_0, "pass_name", "add_pass_name");

  vector<string> single_op_pass_names = {"A", "B", "C"};
  ge::AttrUtils::SetListStr(op_desc_0, kSingleOpUbPassNameAttr, single_op_pass_names);
  NodePtr node_0 = graph->AddNode(op_desc_0);
  std::set<std::string> pass_name_set = {"match", "effect"};
  BufferFusionInfoCollecter buffer_fusion_info_collecter;
  buffer_fusion_info_collecter.SetPassName(*graph, pass_name_set);
  PassNameIdMap pass_name_scope_id_map;
  Status ret = buffer_fusion_info_collecter.GetPassNameOfScopeId(*graph, pass_name_scope_id_map);
  EXPECT_EQ(ret, fe::SUCCESS);
  PassNameIdMap pass_name_fusion_failed_id_map;
  ret = buffer_fusion_info_collecter.GetPassNameOfFailedId(*graph, pass_name_fusion_failed_id_map);
  EXPECT_EQ(ret, fe::SUCCESS);
  std::map<std::string, int32_t> pass_match_map;
  pass_match_map["match"] = 1;
  std::map<std::string, int32_t> pass_effect_map;
  pass_match_map["effect"] = 1;
  ret = buffer_fusion_info_collecter.CountBufferFusionTimes(*graph);

  EXPECT_EQ(ret, fe::SUCCESS);
  int count = 0;
  for (auto &ele : FusionStatisticRecorder::Instance().buffer_fusion_info_map_) {
    for (auto &ele2 : ele.second) {
      if (ele2.first == "A") {
        EXPECT_EQ(ele2.second.GetMatchTimes(), 1);
        EXPECT_EQ(ele2.second.GetEffectTimes(), 1);
        count++;
      }

      if (ele2.first == "B") {
        EXPECT_EQ(ele2.second.GetMatchTimes(), 1);
        EXPECT_EQ(ele2.second.GetEffectTimes(), 1);
        count++;
      }

      if (ele2.first == "C") {
        EXPECT_EQ(ele2.second.GetMatchTimes(), 1);
        EXPECT_EQ(ele2.second.GetEffectTimes(), 1);
        count++;
      }
    }
  }

  EXPECT_EQ(count, 3);
}

TEST_F(BUFFER_FUSION_INFO_COLLECTOR_STEST, fusion_result_json_with_env) {
  Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
  std::string path = GetCurpath() + "../../../../../../ascend_work_path";
  mmSetEnv("ASCEND_WORK_PATH", path.c_str(), MMPA_MAX_PATH);
  config.InitParamFromEnv();
  FusionStatisticWriter::Instance().fusion_result_path_ = "";
  FusionStatisticWriter::Instance().GetFusionResultFilePath();
  std::string expect_fusion_result_json_path = path + "/FE" + "/" + std::to_string(mmGetPid()) + "/fusion_result.json";
  EXPECT_EQ(FusionStatisticWriter::Instance().fusion_result_path_, expect_fusion_result_json_path);
}

TEST_F(BUFFER_FUSION_INFO_COLLECTOR_STEST, fusion_result_json_without_env) {
  Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
  unsetenv("ASCEND_WORK_PATH");
  std::array<string, static_cast<size_t>(ENV_STR_PARAM::EnvStrParamBottom)> env_str_param_vec_tmp;
  config.env_str_param_vec_.swap(env_str_param_vec_tmp);
  config.InitParamFromEnv();
  FusionStatisticWriter::Instance().fusion_result_path_ = "";
  FusionStatisticWriter::Instance().GetFusionResultFilePath();
  std::string expect_fusion_result_json_path = "fusion_result.json";
  EXPECT_EQ(FusionStatisticWriter::Instance().fusion_result_path_, expect_fusion_result_json_path);
}


TEST_F(BUFFER_FUSION_INFO_COLLECTOR_STEST, clear_history_file_and_dir_with_env) {
  Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
  std::string path = GetCurpath() + "../../../../../../ascend_work_path";
  mmSetEnv("ASCEND_WORK_PATH", path.c_str(), MMPA_MAX_PATH);
  config.InitParamFromEnv();
  std::string fusion_result_json_path = path + "/FE" + "/" + std::to_string(mmGetPid()) + "/fusion_result.json";
  CreateFileAndFillContent(fusion_result_json_path);
  
  FusionStatisticWriter::Instance().fusion_result_path_ = "";
  FusionStatisticWriter::Instance().ClearHistoryFile();
  system(("rm -rf " + path).c_str());
}

TEST_F(BUFFER_FUSION_INFO_COLLECTOR_STEST, write_fusion_info_and_clear_history_file_and_dir_without_env) {
  Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
  unsetenv("ASCEND_WORK_PATH");
  std::array<string, static_cast<size_t>(ENV_STR_PARAM::EnvStrParamBottom)> env_str_param_vec_tmp;
  config.env_str_param_vec_.swap(env_str_param_vec_tmp);
  config.InitParamFromEnv();
  FusionStatisticWriter::Instance().fusion_result_path_ = "";
  FusionStatisticRecorder& fs_instance = FusionStatisticRecorder::Instance();
  fs_instance.graph_fusion_info_map_.clear();
  fs_instance.buffer_fusion_info_map_.clear();
  FusionInfo fusion_info1(0, "0", "pass1", 1, 1);
  FusionInfo fusion_info2(0, "0", "pass2", 2, 2);
  FusionInfo fusion_info3(0, "0", "pass3", 3, 3);
  FusionInfo fusion_info4(0, "0", "pass4", 4, 4);

  FusionInfo fusion_info5(1, "1", "pass5", 5, 5);
  FusionInfo fusion_info6(1, "1", "pass6", 6, 6);
  FusionInfo fusion_info7(1, "1", "pass7", 7, 7);
  FusionInfo fusion_info8(1, "1", "pass8", 8, 8);

  fs_instance.UpdateGraphFusionMatchTimes(fusion_info1);
  fs_instance.UpdateGraphFusionEffectTimes(fusion_info1);
  fs_instance.UpdateGraphFusionMatchTimes(fusion_info2);
  fs_instance.UpdateGraphFusionEffectTimes(fusion_info2);
  fs_instance.UpdateBufferFusionMatchTimes(fusion_info3);
  fs_instance.UpdateBufferFusionEffectTimes(fusion_info3);
  fs_instance.UpdateBufferFusionMatchTimes(fusion_info4);
  fs_instance.UpdateBufferFusionEffectTimes(fusion_info4);
  FusionStatisticWriter::Instance().WriteAllFusionInfoToJsonFile();

  fs_instance.UpdateGraphFusionMatchTimes(fusion_info5);
  fs_instance.UpdateGraphFusionEffectTimes(fusion_info5);
  fs_instance.UpdateGraphFusionMatchTimes(fusion_info6);
  fs_instance.UpdateGraphFusionEffectTimes(fusion_info6);
  fs_instance.UpdateBufferFusionMatchTimes(fusion_info7);
  fs_instance.UpdateBufferFusionEffectTimes(fusion_info7);
  fs_instance.UpdateBufferFusionMatchTimes(fusion_info8);
  fs_instance.UpdateBufferFusionEffectTimes(fusion_info8);
  FusionStatisticWriter::Instance().WriteAllFusionInfoToJsonFile();
  // std::string path = GetCurpath() + "../../../../../../../fusion_result.json";
  // std::ifstream file(path);
  // EXPECT_EQ(file.is_open(), true);
  // json js;
  // file >> js;
  // file.close();
  // EXPECT_EQ(js["session_and_graph_id_0_0"]["graph_fusion"]["pass1"]["effect_times"].get<std::string>(), "1");
  // EXPECT_EQ(js["session_and_graph_id_1_1"]["graph_fusion"]["pass1"]["effect_times"].get<std::string>(), "5");
  FusionStatisticWriter::Instance().ClearHistoryFile();
}
