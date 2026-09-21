/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/ffts_node_calculater_registry.h"
#include "register/ffts_node_converter_registry.h"
#include "register/op_ext_gentask_registry.h"
#include "proto/task.pb.h"
#include <gtest/gtest.h>

class FFTSNodeRegistryUnittest : public testing::Test {};

namespace TestFFTSNodeRegistry {
gert::LowerResult TestFftsLowFunc(const ge::NodePtr &node, const gert::FFTSLowerInput &lower_input) {
  return {};
}

ge::graphStatus TestFftsCalcFunc(const ge::NodePtr &node, const gert::LoweringGlobalData *global_data,
    size_t &total_size, size_t &pre_data_size, std::unique_ptr<uint8_t[]> &pre_data_ptr) {
  return ge::GRAPH_SUCCESS;
}

TEST_F(FFTSNodeRegistryUnittest, ConverterRegisterSuccess_Test) {
  EXPECT_EQ(gert::FFTSNodeConverterRegistry::GetInstance().FindNodeConverter("RegisterSuccess1"), nullptr);
  FFTS_REGISTER_NODE_CONVERTER("RegisterSuccess1", TestFftsLowFunc);
  EXPECT_EQ(gert::FFTSNodeConverterRegistry::GetInstance().FindNodeConverter("RegisterSuccess1"), TestFftsLowFunc);
}

TEST_F(FFTSNodeRegistryUnittest, CalculaterRegisterSuccess_Test) {
  EXPECT_EQ(gert::FFTSNodeCalculaterRegistry::GetInstance().FindNodeCalculater("RegisterSuccess2"), nullptr);
  FFTS_REGISTER_NODE_CALCULATER("RegisterSuccess2", TestFftsCalcFunc);
  EXPECT_EQ(gert::FFTSNodeCalculaterRegistry::GetInstance().FindNodeCalculater("RegisterSuccess2"), TestFftsCalcFunc);
}

TEST_F(FFTSNodeRegistryUnittest, SkipCtxRecord_Test) {
  gert::SkipCtxRecord skip_record;
  uint32_t ctx_id = 0;
  uint32_t ctx_type = 1;
  EXPECT_EQ(skip_record.SetSkipCtx(ctx_id, ctx_type), false);
  skip_record.Init();
  skip_record.SetSkipCtx(1, 2);
  skip_record.SetSkipCtx(2, 3);
  EXPECT_EQ(skip_record.GetCtxNum(), 2);
  skip_record.GetSkipCtx(1, ctx_id, ctx_type);
  EXPECT_EQ(ctx_id, 2);
  EXPECT_EQ(ctx_type, 3);
  skip_record.ClearRecord();
  EXPECT_EQ(skip_record.GetCtxNum(), 0);
}

ge::Status TestOpExtGenTask(const ge::Node &node, ge::RunContext &context, std::vector<domi::TaskDef> &tasks) {
  return ge::SUCCESS;
}

TEST_F(FFTSNodeRegistryUnittest, OpExtGenTask_test1) {
  EXPECT_EQ(fe::OpExtGenTaskRegistry::GetInstance().FindRegisterFunc("Conv2D"), nullptr);
  REGISTER_NODE_EXT_GENTASK("Conv2D", TestOpExtGenTask);
  auto func = fe::OpExtGenTaskRegistry::GetInstance().FindRegisterFunc("Conv2D");
  EXPECT_EQ(func, TestOpExtGenTask);
}

ge::Status TestSKExtGenTaskFunc(const ge::Node &node, std::vector<std::vector<domi::TaskDef>> &subTasks,
  const std::vector<ge::Node *> &sub_nodes, std::vector<domi::TaskDef> &tasks) {
  return ge::SUCCESS;
}

TEST_F(FFTSNodeRegistryUnittest, OpExtGenTask_test2) {
  EXPECT_EQ(fe::OpExtGenTaskRegistry::GetInstance().FindSKRegisterFunc("Conv2D"), nullptr);
  REGISTER_SK_EXT_GENTASK("Conv2D", TestSKExtGenTaskFunc);
  auto func = fe::OpExtGenTaskRegistry::GetInstance().FindSKRegisterFunc("Conv2D");
  EXPECT_EQ(func, TestSKExtGenTaskFunc);
}

TEST_F(FFTSNodeRegistryUnittest, ExtTaskTypeReg_test) {
  REGISTER_EXT_TASK_TYPE(MoeDistributeCombine, fe::ExtTaskType::kAicoreTask);
  fe::ExtTaskType taskType = fe::OpExtGenTaskRegistry::GetInstance().GetExtTaskType("MoeDistributeCombine");
  EXPECT_EQ(taskType, fe::ExtTaskType::kAicoreTask);
}
}
