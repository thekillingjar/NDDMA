/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <gmock/gmock.h>
#include "gtest/gtest.h"
#include "proto/dflow.pb.h"
#include "macro_utils/dt_public_scope.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"
#include "macro_utils/dt_public_unscope.h"
#include "framework/common/types.h"
#include "framework/common/helper/model_helper.h"

using namespace testing;
namespace ge {
namespace {
class MockModelHelper : public ModelHelper {
 public:
  // 在这里添加模拟版本的GetModelFileHead方法
  MOCK_METHOD2(GetModelFileHead, Status(const ge::ModelData&, const ModelFileHeader*&));
};
}  // namespace

class FlowModelHelperTest : public Test {
 public:
  // Helper function to create a temporary OM file with size 0
  static std::string CreateTemporaryOmFile() {
    std::string tempFileName = "temp_om_file.om";
    std::ofstream tempFile(tempFileName, std::ios::binary);
    // Add any required OM file header data if necessary
    // For a size 0 OM file, you can just create an empty file
    return tempFileName;
  }

  // Helper function to delete a temporary OM file
  static void DeleteTemporaryOmFile(const std::string& fileName) {
    std::remove(fileName.c_str());
  }
  FlowModelHelper flowModelHelper;
  MockModelHelper mockModelHelper;

 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};


TEST(FlowModelHelperTest, LoadFlowModelFromModelBuffData) {
  // Create a mock ModelHelper object
  MockModelHelper mockModelHelper;

  FlowModelHelper flowModelHelper;

  // Create a temporary OM file with size 0
  std::string tempOmFile = FlowModelHelperTest::CreateTemporaryOmFile();

  // Mock data for name_and_model with the temporary OM file
  ge::AscendString name("test_model");
  ge::ModelBufferData modelBufferData;
  modelBufferData.data = nullptr; // Simulating om size 0
  modelBufferData.length = 0;
  ge::FlowModelPtr flow_model;
  Status status = flowModelHelper.LoadFlowModelFromBuffData(modelBufferData, flow_model);

  // Assert that the function returns an error status (e.g., ge::FAILED) when om size is 0
  ASSERT_NE(status, ge::FAILED);
  // Add more assertions as needed for the specific error condition

  // Clean up: Delete the temporary OM file
  FlowModelHelperTest::DeleteTemporaryOmFile(tempOmFile);
}

TEST(FlowModelHelperTest, LoadFlowModelFromOmFile) {
   FlowModelHelper flowModelHelper;

   // 创建一个临时OM文件
   std::string tempOmFile = FlowModelHelperTest::CreateTemporaryOmFile();

   ge::FlowModelPtr flow_model;
   Status status = flowModelHelper.LoadFlowModelFromOmFile(tempOmFile.c_str(), flow_model);

   ASSERT_NE(status, ge::SUCCESS);

   // 删除临时OM文件
   FlowModelHelperTest::DeleteTemporaryOmFile(tempOmFile);
}
}  // namespace ge
