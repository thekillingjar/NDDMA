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
#include <gmock/gmock.h>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "macro_utils/dt_public_scope.h"
#include "analyzer/analyzer.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "mmpa/mmpa_api.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;

namespace ge {

class UtestAnalyzer : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  	  Analyzer::GetInstance()->is_json_file_create_ = false;
      for (auto &session_resource : Analyzer::GetInstance()->graph_infos_){
          session_resource.second.clear();
      }
      Analyzer::GetInstance()->graph_infos_.clear();
  }
};

TEST_F(UtestAnalyzer, Normal) {
    Analyzer *instance = Analyzer::GetInstance();
    EXPECT_EQ(instance->IsEnableNetAnalyzeDebug(), false);
    EXPECT_EQ(instance->Initialize(), SUCCESS);
    instance->graph_infos_[1] = std::map<uint64_t, std::shared_ptr<analyzer::GraphInfo>>();
    instance->json_file_.open("test.json");
    instance->Finalize();
}


TEST_F(UtestAnalyzer, BuildJsonObject) {
    Analyzer *instance = Analyzer::GetInstance();
    EXPECT_EQ(instance->BuildJsonObject(1, 1), SUCCESS);
    instance->graph_infos_[1] = std::map<uint64_t, std::shared_ptr<analyzer::GraphInfo>>();
    EXPECT_EQ(instance->BuildJsonObject(1, 1), SUCCESS);
    std::map<uint64_t, std::shared_ptr<analyzer::GraphInfo>> am;
    am[2] = std::make_shared<analyzer::GraphInfo>();
    instance->graph_infos_[2] = am;
    EXPECT_EQ(instance->BuildJsonObject(2, 2), SUCCESS);
}

TEST_F(UtestAnalyzer, DestroySessionJsonObject) {
    Analyzer *instance = Analyzer::GetInstance();
    instance->DestroySessionJsonObject(1);
    instance->graph_infos_[1] = std::map<uint64_t, std::shared_ptr<analyzer::GraphInfo>>();
    EXPECT_EQ(instance->graph_infos_.size(), 1);
    instance->DestroySessionJsonObject(1);
    EXPECT_EQ(instance->graph_infos_.size(), 0);
}

TEST_F(UtestAnalyzer, DestroyGraphJsonObject) {
    Analyzer *instance = Analyzer::GetInstance();
    instance->DestroyGraphJsonObject(1, 1);
    instance->graph_infos_[1] = std::map<uint64_t, std::shared_ptr<analyzer::GraphInfo>>();
    instance->DestroyGraphJsonObject(1, 1);
    EXPECT_EQ(instance->graph_infos_.size(), 1);
    instance->graph_infos_[1][1] = std::make_shared<analyzer::GraphInfo>();
    instance->DestroyGraphJsonObject(1, 1);
    EXPECT_EQ(instance->graph_infos_.size(), 1);
}

TEST_F(UtestAnalyzer, GetJsonObject) {
    Analyzer *instance = Analyzer::GetInstance();
    EXPECT_EQ(instance->GetJsonObject(1, 1), nullptr);
    instance->graph_infos_[1] = std::map<uint64_t, std::shared_ptr<analyzer::GraphInfo>>();
    EXPECT_EQ(instance->GetJsonObject(1, 1), nullptr);
    instance->graph_infos_[1][1] = std::make_shared<analyzer::GraphInfo>();
    EXPECT_NE(instance->GetJsonObject(1, 1), nullptr);
}

TEST_F(UtestAnalyzer, ClearHistoryFile) {
    Analyzer *instance = Analyzer::GetInstance();
    EXPECT_NO_THROW(instance->ClearHistoryFile());
}

TEST_F(UtestAnalyzer, CreateAnalyzerFile) {
    Analyzer *instance = Analyzer::GetInstance();
    EXPECT_EQ(instance->CreateAnalyzerFile(), SUCCESS);
    instance->is_json_file_create_ = true;
    EXPECT_EQ(instance->CreateAnalyzerFile(), SUCCESS);
}

TEST_F(UtestAnalyzer, SaveAnalyzerDataToFile) {
    Analyzer *instance = Analyzer::GetInstance();
    instance->graph_infos_[1] = std::map<uint64_t, std::shared_ptr<analyzer::GraphInfo>>();
    instance->graph_infos_[1][1] = std::make_shared<analyzer::GraphInfo>();
    EXPECT_EQ(instance->SaveAnalyzerDataToFile(1, 1), SUCCESS);
    instance->graph_infos_[1][1]->op_info.push_back(analyzer::OpInfo());
    EXPECT_EQ(instance->SaveAnalyzerDataToFile(1, 1), SUCCESS);
}

TEST_F(UtestAnalyzer, DoAnalyze) {
    Analyzer *instance = Analyzer::GetInstance();
    auto di = analyzer::DataInfo();
    EXPECT_EQ(instance->DoAnalyze(di), PARAM_INVALID);
    auto graph = std::make_shared<ComputeGraph>("graph");
    auto tensor_desc = std::make_shared<GeTensorDesc>();

    auto op_desc = std::make_shared<OpDesc>("name", "type");
    op_desc->AddInputDesc(tensor_desc->Clone());
    op_desc->AddOutputDesc(tensor_desc->Clone());
    di.node_ptr = graph->AddNode(op_desc);
    di.session_id = 1;
    di.graph_id = 1;
    instance->graph_infos_[1] = std::map<uint64_t, std::shared_ptr<analyzer::GraphInfo>>();
    instance->graph_infos_[1][1] = std::make_shared<analyzer::GraphInfo>();
    EXPECT_EQ(instance->DoAnalyze(di), SUCCESS);
    di.analyze_type = (analyzer::AnalyzeType)1000;
    EXPECT_EQ(instance->DoAnalyze(di), FAILED);
}

TEST_F(UtestAnalyzer, SaveOpInfo) {
    Analyzer *instance = Analyzer::GetInstance();
    auto di0 = analyzer::DataInfo();
    di0.analyze_type = (analyzer::AnalyzeType)1000;
    auto de0 = std::make_shared<OpDesc>("name", "type");
    auto gi0 = std::make_shared<analyzer::GraphInfo>();
    EXPECT_EQ(instance->SaveOpInfo(de0, di0, gi0), PARAM_INVALID);
    auto graph = std::make_shared<ComputeGraph>("graph");
    auto tensor_desc = std::make_shared<GeTensorDesc>();
    std::vector<int64_t> shape;
    shape.push_back(1);
    shape.push_back(1);
    shape.push_back(224);
    shape.push_back(224);
    tensor_desc->SetShape(GeShape(shape));
    tensor_desc->SetFormat(FORMAT_NCHW);
    tensor_desc->SetDataType(DT_FLOAT);
    tensor_desc->SetOriginFormat(FORMAT_NCHW);
    tensor_desc->SetOriginShape(GeShape(shape));
    tensor_desc->SetOriginDataType(DT_FLOAT);

    auto op_desc = std::make_shared<OpDesc>("name", "type");
    for (int i = 0; i < 1; ++i) {
        op_desc->AddInputDesc(tensor_desc->Clone());
    }
    for (int i = 0; i < 1; ++i) {
        op_desc->AddOutputDesc(tensor_desc->Clone());
    }

    auto node = graph->AddNode(op_desc);
    auto di = analyzer::DataInfo(1, 1, analyzer::PARSER, node, "error");
    auto de = std::make_shared<OpDesc>("name", "type");
    auto gi = std::make_shared<analyzer::GraphInfo>();
    EXPECT_EQ(instance->SaveOpInfo(de, di, gi), SUCCESS);
}

TEST_F(UtestAnalyzer, TensorInfoToJson) {
    Analyzer *instance = Analyzer::GetInstance();
    nlohmann::json j;
    auto ti = analyzer::TensorInfo();
    ti.shape.push_back(1);
    ti.d_type = "tp";
    ti.layout = "lo";
    EXPECT_NO_THROW(instance->TensorInfoToJson(j, ti));
}

TEST_F(UtestAnalyzer, OpInfoToJson) {
    Analyzer *instance = Analyzer::GetInstance();
    nlohmann::json j;
    auto oi = analyzer::OpInfo();
    oi.input_info.push_back(analyzer::TensorInfo());
    oi.output_info.push_back(analyzer::TensorInfo());
    EXPECT_NO_THROW(instance->OpInfoToJson(j, oi));
}

TEST_F(UtestAnalyzer, GraphInfoToJson) {
    Analyzer *instance = Analyzer::GetInstance();
    nlohmann::json j;
    auto gi = analyzer::GraphInfo();
    EXPECT_NO_THROW(instance->GraphInfoToJson(j, gi));
}


} // namespace ge
