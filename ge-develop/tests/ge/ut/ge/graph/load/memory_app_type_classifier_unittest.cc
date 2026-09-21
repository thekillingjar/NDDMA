/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/memory_app_type_classifier.h"
#include <gtest/gtest.h>
#include "faker/mem_allocation_faker.h"
#include "faker/task_run_param_faker.h"
namespace ge {
class MemoryAppTypeClassifierUT : public testing::Test {};
TEST_F(MemoryAppTypeClassifierUT, ClassifyByTaskRunParams_Ok_InputOutputWs) {
  auto allocation_table = MemAllocationFaker()
                              .FeatureMap(0x100, 1024 * 1024 * 1024)
                              .ModelInput(0, 0x80000000, 1024 * 1024)
                              .ModelInput(0, 0x81000000, 1024 * 1024)
                              .Build();

  std::vector<TaskRunParam> params;
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x200)
                          .RefreshHbmWorkspace(0x300)
                          .RefreshInput(0x400, RT_MEMORY_TS)
                          .Build());
  auto ret = MemoryAppTypeClassifier(allocation_table, 0).ClassifyByTaskRunParams(params);

  ASSERT_EQ(ret.size(), 4U);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x100))), MemoryAppType::kMemoryTypeFeatureMap);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x200))), MemoryAppType::kMemoryTypeFeatureMap);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x300))), MemoryAppType::kMemoryTypeFeatureMap);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_TS, 0x400))), MemoryAppType::kMemoryTypeFeatureMap);
}
TEST_F(MemoryAppTypeClassifierUT, ClassifyByTaskRunParams_Ok_ModelIo) {
  auto allocation_table = MemAllocationFaker()
                              .FeatureMap(0x100, 1024 * 1024 * 1024)
                              .ModelInput(0, 0x90000000, 1024 * 1024)
                              .ModelInput(0, 0x91000000, 1024 * 1024)
                              .ModelOutput(0, 0x92000000, 1024 * 1024)
                              .ModelOutput(0, 0x93000000, 1024 * 1024)
                              .ModelOutput(0, 0x94000000, 1024 * 1024)
                              .Build();

  std::vector<TaskRunParam> params;
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x90000000)
                          .Outputs({0x91000000, 0x91000000 + 1024, 0x92000000 + 1024, 0x93000000 + 1024, 0x94000000},
                                   RT_MEMORY_HBM, true)
                          .RefreshHbmWorkspace(0x100)
                          .Build());
  auto ret = MemoryAppTypeClassifier(allocation_table, 0).ClassifyByTaskRunParams(params);

  ASSERT_EQ(ret.size(), 7U);
  EXPECT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x90000000))), MemoryAppType::kMemoryTypeModelIo);
  EXPECT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x91000000))), MemoryAppType::kMemoryTypeModelIo);
  EXPECT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x91000000 + 1024))),
            MemoryAppType::kMemoryTypeModelIo);
  EXPECT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x92000000 + 1024))),
            MemoryAppType::kMemoryTypeModelIo);
  EXPECT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x93000000 + 1024))),
            MemoryAppType::kMemoryTypeModelIo);
  EXPECT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x94000000))), MemoryAppType::kMemoryTypeModelIo);
  EXPECT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x100))), MemoryAppType::kMemoryTypeFeatureMap);
}
TEST_F(MemoryAppTypeClassifierUT, ClassifyByTaskRunParams_Ok_WeightAbs) {
  auto allocation_table = MemAllocationFaker()
                              .FeatureMap(0x100, 1024 * 1024 * 1024)
                              .ModelInput(0, 0x90000000, 1024 * 1024)
                              .ModelInput(0, 0x91000000, 1024 * 1024)
                              .ModelOutput(0, 0x92000000, 1024 * 1024)
                              .ModelOutput(0, 0x94000000, 1024 * 1024)
                              .Build();

  std::vector<TaskRunParam> params;
  params.emplace_back(TaskRunParamFaker()
                          .RefreshWeightInput(0x100 + 1024 * 1024 * 1024)
                          .RefreshWeightInput(0x90000000 + 1024 * 1024)
                          .RefreshWeightOutput(0x91000000 + 1024 * 1024)
                          .RefreshWeightOutput(0x92000000 + 1024 * 1024)
                          .RefreshWeightOutput(0x94000000 + 1024 * 1024)
                          .RefreshWeightOutput(0xa0000000)
                          .RefreshHbmWorkspace(0x100)
                          .Build());
  auto ret = MemoryAppTypeClassifier(allocation_table, 0).ClassifyByTaskRunParams(params);

  ASSERT_EQ(ret.size(), 7U);
  EXPECT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(kWeightMemType, 0x100 + 1024 * 1024 * 1024))),
            MemoryAppType::kMemoryTypeFix);
  EXPECT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(kWeightMemType, 0x90000000 + 1024 * 1024))),
            MemoryAppType::kMemoryTypeFix);
  EXPECT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(kWeightMemType, 0x91000000 + 1024 * 1024))),
            MemoryAppType::kMemoryTypeFix);
  EXPECT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(kWeightMemType, 0x92000000 + 1024 * 1024))),
            MemoryAppType::kMemoryTypeFix);
  EXPECT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(kWeightMemType, 0x94000000 + 1024 * 1024))),
            MemoryAppType::kMemoryTypeFix);
  EXPECT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(kWeightMemType, 0xa0000000))), MemoryAppType::kMemoryTypeFix);
  EXPECT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x100))), MemoryAppType::kMemoryTypeFeatureMap);
}
TEST_F(MemoryAppTypeClassifierUT, ClassifyByTaskRunParams_Ok_ModelIoFmAbs) {
  auto allocation_table = MemAllocationFaker()
                              .FeatureMap(0x100, 1024 * 1024 * 1024)
                              .ModelInput(0, 0x90000000, 1024 * 1024)
                              .ModelInput(0, 0x91000000, 1024 * 1024)
                              .Build();

  std::vector<TaskRunParam> params;
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x90000000)
                          .RefreshHbmOutput(0x91000000)
                          .RefreshHbmOutput(0x91000000 + 1024)
                          .RefreshHbmWorkspace(0x300)
                          .Build());
  params.emplace_back(TaskRunParamFaker()
                          .RefreshWeightInput(0xa0000000)
                          .RefreshWeightOutput(0xa0000000 + 1024)
                          .RefreshWeightOutput(0xa0000000 + 2048)
                          .RefreshHbmWorkspace(0x400)
                          .Build());
  auto ret = MemoryAppTypeClassifier(allocation_table, 0).ClassifyByTaskRunParams(params);

  ASSERT_EQ(ret.size(), 8U);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x90000000))), MemoryAppType::kMemoryTypeModelIo);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x91000000))), MemoryAppType::kMemoryTypeModelIo);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x91000000 + 1024))),
            MemoryAppType::kMemoryTypeModelIo);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x300))), MemoryAppType::kMemoryTypeFeatureMap);

  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(kWeightMemType, 0xa0000000))), MemoryAppType::kMemoryTypeFix);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(kWeightMemType, 0xa0000000 + 1024))),
            MemoryAppType::kMemoryTypeFix);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(kWeightMemType, 0xa0000000 + 2048))),
            MemoryAppType::kMemoryTypeFix);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x400))), MemoryAppType::kMemoryTypeFeatureMap);
}
TEST_F(MemoryAppTypeClassifierUT, ClassifyByTaskRunParams_Ok_SameAddrBetweenParams) {
  auto allocation_table = MemAllocationFaker()
                              .FeatureMap(0x100, 1024 * 1024 * 1024)
                              .ModelInput(0, 0x90000000, 1024 * 1024)
                              .ModelInput(0, 0x91000000, 1024 * 1024)
                              .Build();

  std::vector<TaskRunParam> params;
  params.emplace_back(TaskRunParamFaker()
                          .RefreshInput(0x90000000, kFmMemType)
                          .RefreshHbmOutput(0x91000000)
                          .RefreshHbmOutput(0x91000000 + 1024)
                          .RefreshHbmWorkspace(0x300)
                          .Build());
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x91000000)
                          .RefreshWeightOutput(0xa0000000 + 1024)
                          .RefreshWeightOutput(0xa0000000 + 2048)
                          .RefreshHbmWorkspace(0x300)
                          .Build());
  auto ret = MemoryAppTypeClassifier(allocation_table, 0).ClassifyByTaskRunParams(params);

  ASSERT_EQ(ret.size(), 6U);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(kFmMemType, 0x90000000))), MemoryAppType::kMemoryTypeModelIo);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x91000000))), MemoryAppType::kMemoryTypeModelIo);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x91000000 + 1024))),
            MemoryAppType::kMemoryTypeModelIo);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x300))), MemoryAppType::kMemoryTypeFeatureMap);

  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(kWeightMemType, 0xa0000000 + 1024))),
            MemoryAppType::kMemoryTypeFix);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(kWeightMemType, 0xa0000000 + 2048))),
            MemoryAppType::kMemoryTypeFix);
}
TEST_F(MemoryAppTypeClassifierUT, ClassifyByTaskRunParams_Ok_HasAbsoluteAllocation) {
  auto allocation_table = MemAllocationFaker()
                              .FeatureMap(0x100, 1024 * 1024 * 1024)
                              .ModelInput(0, 0x80000000, 1024 * 1024)
                              .ModelInput(0, 0x81000000, 1024 * 1024)
                              .Absolute(0x91000000, 1024 * 1024)
                              .Build();

  std::vector<TaskRunParam> params;
  params.emplace_back(
      TaskRunParamFaker().RefreshHbmInput(0x100).RefreshHbmOutput(0x200).RefreshHbmWorkspace(0x300).Build());
  auto ret = MemoryAppTypeClassifier(allocation_table, 0).ClassifyByTaskRunParams(params);

  ASSERT_EQ(ret.size(), 3U);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x100))), MemoryAppType::kMemoryTypeFeatureMap);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x200))), MemoryAppType::kMemoryTypeFeatureMap);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x300))), MemoryAppType::kMemoryTypeFeatureMap);
}
TEST_F(MemoryAppTypeClassifierUT, GetMemoryAppTypeStr_Ok) {
  ASSERT_STREQ(GetMemoryAppTypeStr(MemoryAppType::kMemoryTypeModelIo), "model io");
  ASSERT_STREQ(GetMemoryAppTypeStr(MemoryAppType::kMemoryTypeFix), "weight");
  ASSERT_STREQ(GetMemoryAppTypeStr(MemoryAppType::kMemoryTypeFeatureMap), "feature map");
  ASSERT_STREQ(GetMemoryAppTypeStr(MemoryAppType::kEnd), "unknown");
}
TEST_F(MemoryAppTypeClassifierUT, ClassifyByTaskRunParams_UseMemoryType_AbsoluteAndFmAddrConflict) {
  constexpr uint64_t fm_base = 0x100;
  constexpr uint64_t model_io_base = 0x100 + 1024 * 1024;
  constexpr uint64_t model_io_1 = model_io_base;
  constexpr uint64_t model_io_2 = model_io_base + 0x100;
  auto allocation_table = MemAllocationFaker()
                              .FeatureMap(fm_base, 1024 * 1024)
                              .ModelInput(0, model_io_1, 0x100)
                              .ModelInput(1, model_io_2, 0x100)
                              .Absolute(0x100 + 1024, 1024 * 1024)
                              .Build();

  std::vector<TaskRunParam> params;
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(model_io_1)           // model io
                          .RefreshHbmInput(model_io_2)           // model io
                          .Input(model_io_2, kVarMemType, true)  // model io
                          .RefreshHbmOutput(0x200)
                          .RefreshHbmWorkspace(0x300)
                          .Build());
  auto ret = MemoryAppTypeClassifier(allocation_table, 0).ClassifyByTaskRunParams(params);

  ASSERT_EQ(ret.size(), 5U);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x100 + 1024 * 1024))),
            MemoryAppType::kMemoryTypeModelIo);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x100 + 1024 * 1024 + 0x100))),
            MemoryAppType::kMemoryTypeModelIo);
  // todo 当前ModelUtils里面认为Var也是可刷新的，会带来误判
  // ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(kVarMemType, 0x100 + 1024 * 1024 + 0x100))),
  // MemoryAppType::kMemoryTypeFix);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x200))), MemoryAppType::kMemoryTypeFeatureMap);
  ASSERT_EQ(ret.at((std::make_pair<uint64_t, uint64_t>(RT_MEMORY_HBM, 0x300))), MemoryAppType::kMemoryTypeFeatureMap);
}
}  // namespace ge
