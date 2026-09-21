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

#include "macro_utils/dt_public_scope.h"
#include "common/dump/dump_properties.h"
#include "ge_local_context.h"
#include "ge/ge_api_types.h"
#include "common/debug/log.h"
#include "common/ge_inner_error_codes.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
class UTEST_dump_properties : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UTEST_dump_properties, check_dump_step) {
  DumpProperties dp;

  std::string dump_step{"0|3-5|10"};
  EXPECT_EQ(dp.CheckDumpStepInner(dump_step), StepCheckErrorCode::kNoError);
  EXPECT_EQ(dp.CheckDumpStep(dump_step), SUCCESS);

  std::string unsupport_input1{"0|5-3|10"};
  EXPECT_EQ(dp.CheckDumpStepInner(unsupport_input1), StepCheckErrorCode::kReverseStep);
  EXPECT_NE(dp.CheckDumpStep(unsupport_input1), SUCCESS);

  std::string unsupport_input2{"one"};
  EXPECT_EQ(dp.CheckDumpStepInner(unsupport_input2), StepCheckErrorCode::kIncorrectFormat);
  EXPECT_NE(dp.CheckDumpStep(unsupport_input2), SUCCESS);

  std::string unsupport_input3;
  for (int i = 0; i < 200; ++i) {
    unsupport_input3 += std::to_string(i) + "|";
  }
  unsupport_input3.pop_back();
  EXPECT_EQ(dp.CheckDumpStepInner(unsupport_input3), StepCheckErrorCode::kTooManySets);
  EXPECT_NE(dp.CheckDumpStep(unsupport_input3), SUCCESS);

  std::string support_input1{"001|002"};
  EXPECT_EQ(dp.CheckDumpStepInner(support_input1), StepCheckErrorCode::kNoError);
  EXPECT_EQ(dp.CheckDumpStep(support_input1), SUCCESS);

  std::string support_input2{"001"};
  EXPECT_EQ(dp.CheckDumpStepInner(support_input2), StepCheckErrorCode::kNoError);
  EXPECT_EQ(dp.CheckDumpStep(support_input2), SUCCESS);

  std::string unsupport_input4{"5a|0-1|10a"};
  EXPECT_EQ(dp.CheckDumpStepInner(unsupport_input4), StepCheckErrorCode::kIncorrectFormat);
  EXPECT_NE(dp.CheckDumpStep(unsupport_input4), SUCCESS);

  std::string unsupport_input5{"5|0-1-2|10"};
  EXPECT_EQ(dp.CheckDumpStepInner(unsupport_input5), StepCheckErrorCode::kIncorrectFormat);
  EXPECT_NE(dp.CheckDumpStep(unsupport_input5), SUCCESS);

  std::string unsupport_input6{"5|0-\n1|10"};
  EXPECT_EQ(dp.CheckDumpStepInner(unsupport_input6), StepCheckErrorCode::kIncorrectFormat);
  EXPECT_NE(dp.CheckDumpStep(unsupport_input6), SUCCESS);

  std::string unsupport_input7{"5|"};
  EXPECT_EQ(dp.CheckDumpStepInner(unsupport_input7), StepCheckErrorCode::kIncorrectFormat);
  EXPECT_NE(dp.CheckDumpStep(unsupport_input7), SUCCESS);

  std::string unsupport_input8{""};
  EXPECT_EQ(dp.CheckDumpStepInner(unsupport_input8), StepCheckErrorCode::kIncorrectFormat);
  EXPECT_NE(dp.CheckDumpStep(unsupport_input8), SUCCESS);

  std::string unsupport_input9{"-1"};
  EXPECT_EQ(dp.CheckDumpStepInner(unsupport_input8), StepCheckErrorCode::kIncorrectFormat);
  EXPECT_NE(dp.CheckDumpStep(unsupport_input8), SUCCESS);

  std::string dos = std::string(54773, '0');
  EXPECT_EQ(dp.CheckDumpStepInner(dos), StepCheckErrorCode::kNoError);
  EXPECT_EQ(dp.CheckDumpStep(dos), SUCCESS);
}

TEST_F(UTEST_dump_properties, check_dump_mode) {
  DumpProperties dp;
  std::string dump_mode_1{"input"};
  std::string dump_mode_2{"output"};
  std::string dump_mode_3{"all"};
  std::string unsupport_input1{"mode1"};
  Status st = dp.CheckDumpMode(dump_mode_1);
  EXPECT_EQ(st, SUCCESS);
  st = dp.CheckDumpMode(dump_mode_2);
  EXPECT_EQ(st, SUCCESS);
  st = dp.CheckDumpMode(dump_mode_3);
  EXPECT_EQ(st, SUCCESS);
  st = dp.CheckDumpMode(unsupport_input1);
  EXPECT_NE(st, SUCCESS);
}

TEST_F(UTEST_dump_properties, check_dump_path) {
  DumpProperties dp;
  std::string dump_path{"/tmp/"};
  std::string unsupport_input1{"  \\unsupported"};
  Status st = dp.CheckDumpPathValid(dump_path);
  EXPECT_EQ(st, SUCCESS);
  st = dp.CheckDumpPathValid(unsupport_input1);
  EXPECT_NE(st, SUCCESS);
}

TEST_F(UTEST_dump_properties, check_enable_dump) {
  DumpProperties dp;
  std::string enable_dump_t{"1"};
  std::string enable_dump_f{"0"};
  std::string unsupport_input1{"true"};
  std::string unsupport_input2{"false"};
  Status st = dp.CheckEnableDump(enable_dump_t);
  EXPECT_EQ(st, SUCCESS);
  st = dp.CheckEnableDump(enable_dump_f);
  EXPECT_EQ(st, SUCCESS);
  st = dp.CheckEnableDump(unsupport_input1);
  EXPECT_NE(st, SUCCESS);
  st = dp.CheckEnableDump(unsupport_input2);
  EXPECT_NE(st, SUCCESS);
}

TEST_F(UTEST_dump_properties, init_by_options_success_1) {
  DumpProperties dp;
  std::map<std::string, std::string> options {{OPTION_EXEC_ENABLE_DUMP, "1"},
                                              {OPTION_EXEC_DUMP_PATH, "/tmp/"},
                                              {OPTION_EXEC_DUMP_STEP, "0|1-3|10"},
                                              {OPTION_EXEC_DUMP_MODE, "all"},
                                              {OPTION_EXEC_DUMP_DATA, "tensor"},
                                              {OPTION_EXEC_DUMP_LAYER, "Conv2D"}};
  GetThreadLocalContext().SetGlobalOption(options);
  Status st = dp.InitByOptions();
  EXPECT_EQ(st, SUCCESS);
}

TEST_F(UTEST_dump_properties, init_by_options_success_2) {
  DumpProperties dp;
  std::map<std::string, std::string> options {{OPTION_EXEC_ENABLE_DUMP_DEBUG, "1"},
                                              {OPTION_EXEC_DUMP_PATH, "/tmp/"},
                                              {OPTION_EXEC_DUMP_DEBUG_MODE, "aicore_overflow"}};
  GetThreadLocalContext().SetGlobalOption(options);
  Status st = dp.InitByOptions();
  EXPECT_EQ(st, SUCCESS);
}

TEST_F(UTEST_dump_properties, init_by_options_success_3) {
  DumpProperties dp;
  std::map<std::string, std::string> options {{OPTION_EXEC_ENABLE_DUMP_DEBUG, "1"},
                                              {OPTION_EXEC_DUMP_PATH, "/tmp/"}};
  GetThreadLocalContext().SetGlobalOption(options);
  Status st = dp.InitByOptions();
  EXPECT_EQ(st, SUCCESS);
}

TEST_F(UTEST_dump_properties, check_SetModelBlacklist) {
  DumpProperties dp;
  std::string model = "model1";
  ModelOpBlacklist blacklist;

  // 设置op_type blacklist
  OpBlacklist conv_blacklist;
  conv_blacklist.input_indices = {0, 1};
  conv_blacklist.output_indices = {0};
  blacklist.dump_optype_blacklist["Conv2D"] = conv_blacklist;

  // 设置op_name blacklist
  OpBlacklist specific_op_blacklist;
  specific_op_blacklist.input_indices = {2};
  specific_op_blacklist.output_indices = {1, 2};
  blacklist.dump_opname_blacklist["conv1"] = specific_op_blacklist;

  dp.SetModelBlacklist(model, blacklist);

  // 验证结果
  const auto& blacklist_map = dp.GetModelDumpBlacklistMap();
  ASSERT_EQ(blacklist_map.size(), 1);
  ASSERT_TRUE(blacklist_map.find("model1") != blacklist_map.end());

  const auto& model_blacklist = blacklist_map.at("model1");
  ASSERT_EQ(model_blacklist.dump_optype_blacklist.size(), 1);
  ASSERT_EQ(model_blacklist.dump_opname_blacklist.size(), 1);

  // 验证op_type blacklist
  const auto& conv_blacklist_result = model_blacklist.dump_optype_blacklist.at("Conv2D");
  EXPECT_EQ(conv_blacklist_result.input_indices, std::set<uint32_t>({0, 1}));
  EXPECT_EQ(conv_blacklist_result.output_indices, std::set<uint32_t>({0}));

  // 验证op_name blacklist
  const auto& specific_blacklist_result = model_blacklist.dump_opname_blacklist.at("conv1");
  EXPECT_EQ(specific_blacklist_result.input_indices, std::set<uint32_t>({2}));
  EXPECT_EQ(specific_blacklist_result.output_indices, std::set<uint32_t>({1, 2}));
}

// 测试合并现有模型的blacklist
TEST_F(UTEST_dump_properties, check_MergeExistingModelBlacklist) {
    DumpProperties dp;
    // 第一次设置
    ModelOpBlacklist first_blacklist;
    OpBlacklist first_conv_blacklist;
    first_conv_blacklist.input_indices = {0, 1};
    first_conv_blacklist.output_indices = {0};
    first_blacklist.dump_optype_blacklist["Conv2D"] = first_conv_blacklist;

    OpBlacklist first_specific_blacklist;
    first_specific_blacklist.input_indices = {2};
    first_specific_blacklist.output_indices = {1};
    first_blacklist.dump_opname_blacklist["conv1"] = first_specific_blacklist;

    dp.SetModelBlacklist("model1", first_blacklist);

    // 第二次设置（合并）
    ModelOpBlacklist second_blacklist;
    OpBlacklist second_conv_blacklist;
    second_conv_blacklist.input_indices = {2}; // 新增input index
    second_conv_blacklist.output_indices = {1}; // 新增output index
    second_blacklist.dump_optype_blacklist["Conv2D"] = second_conv_blacklist;

    OpBlacklist second_specific_blacklist;
    second_specific_blacklist.input_indices = {3}; // 新增input index
    second_specific_blacklist.output_indices = {2}; // 新增output index
    second_blacklist.dump_opname_blacklist["conv1"] = second_specific_blacklist;

    // 添加新的op_type和op_name
    OpBlacklist new_optype_blacklist;
    new_optype_blacklist.input_indices = {0};
    new_optype_blacklist.output_indices = {0};
    second_blacklist.dump_optype_blacklist["MatMul"] = new_optype_blacklist;

    OpBlacklist new_opname_blacklist;
    new_opname_blacklist.input_indices = {1};
    new_opname_blacklist.output_indices = {1};
    second_blacklist.dump_opname_blacklist["fc1"] = new_opname_blacklist;

    dp.SetModelBlacklist("model1", second_blacklist);

    // 验证合并结果
    const auto& blacklist_map = dp.GetModelDumpBlacklistMap();
    ASSERT_EQ(blacklist_map.size(), 1);

    const auto& model_blacklist = blacklist_map.at("model1");
    ASSERT_EQ(model_blacklist.dump_optype_blacklist.size(), 2);
    ASSERT_EQ(model_blacklist.dump_opname_blacklist.size(), 2);

    // 验证合并后的Conv2D op_type
    const auto& merged_conv_blacklist = model_blacklist.dump_optype_blacklist.at("Conv2D");
    EXPECT_EQ(merged_conv_blacklist.input_indices, std::set<uint32_t>({0, 1, 2}));
    EXPECT_EQ(merged_conv_blacklist.output_indices, std::set<uint32_t>({0, 1}));

    // 验证合并后的conv1 op_name
    const auto& merged_specific_blacklist = model_blacklist.dump_opname_blacklist.at("conv1");
    EXPECT_EQ(merged_specific_blacklist.input_indices, std::set<uint32_t>({2, 3}));
    EXPECT_EQ(merged_specific_blacklist.output_indices, std::set<uint32_t>({1, 2}));

    // 验证新添加的op_type
    const auto& new_optype_result = model_blacklist.dump_optype_blacklist.at("MatMul");
    EXPECT_EQ(new_optype_result.input_indices, std::set<uint32_t>({0}));
    EXPECT_EQ(new_optype_result.output_indices, std::set<uint32_t>({0}));

    // 验证新添加的op_name
    const auto& new_opname_result = model_blacklist.dump_opname_blacklist.at("fc1");
    EXPECT_EQ(new_opname_result.input_indices, std::set<uint32_t>({1}));
    EXPECT_EQ(new_opname_result.output_indices, std::set<uint32_t>({1}));
}

TEST_F(UTEST_dump_properties, check_SetOpDumpRange) {
  DumpProperties dp;
  std::string model = "model_test";
  std::vector<std::pair<std::string, std::string>> op_ranges = {{"a", "b"}, {"c", "d"}};
  dp.SetOpDumpRange("", op_ranges);
  EXPECT_EQ(dp.model_dump_op_ranges_map_.size(), 0);

  dp.SetOpDumpRange(model, {});
  EXPECT_EQ(dp.model_dump_op_ranges_map_.size(), 0);

  dp.SetOpDumpRange(model, op_ranges);
  EXPECT_EQ(dp.model_dump_op_ranges_map_.size(), 1);
  EXPECT_EQ(dp.model_dump_op_ranges_map_[model].size(), 2);
}

TEST_F(UTEST_dump_properties, check_GetDumpOpRangeSize) {
  DumpProperties dp;
  std::string model = "model_test";
  EXPECT_EQ(dp.GetDumpOpRangeSize(model, ""), 0);

  std::vector<std::pair<std::string, std::string>> op_ranges = {{"a", "b"}, {"c", "d"}};
  dp.SetOpDumpRange(model, op_ranges);
  EXPECT_EQ(dp.GetDumpOpRangeSize(model, ""), 2);
}

TEST_F(UTEST_dump_properties, check_DumpProcFsmTransition) {
  EXPECT_EQ(DumpProperties::DumpProcFsmTransition(DumpProcState::kInit, DumpProcEvent::kRangeStart), DumpProcState::kStart);
  EXPECT_EQ(DumpProperties::DumpProcFsmTransition(DumpProcState::kInit, DumpProcEvent::kRangeEnd), DumpProcState::kError);
  EXPECT_EQ(DumpProperties::DumpProcFsmTransition(DumpProcState::kInit, DumpProcEvent::kOutOfRange), DumpProcState::kInit);

  EXPECT_EQ(DumpProperties::DumpProcFsmTransition(DumpProcState::kStart, DumpProcEvent::kRangeStart), DumpProcState::kStart);
  EXPECT_EQ(DumpProperties::DumpProcFsmTransition(DumpProcState::kStart, DumpProcEvent::kRangeEnd), DumpProcState::kStop);
  EXPECT_EQ(DumpProperties::DumpProcFsmTransition(DumpProcState::kStart, DumpProcEvent::kOutOfRange), DumpProcState::kStart);

  EXPECT_EQ(DumpProperties::DumpProcFsmTransition(DumpProcState::kStop, DumpProcEvent::kRangeStart), DumpProcState::kError);
  EXPECT_EQ(DumpProperties::DumpProcFsmTransition(DumpProcState::kStop, DumpProcEvent::kRangeEnd), DumpProcState::kStop);
  EXPECT_EQ(DumpProperties::DumpProcFsmTransition(DumpProcState::kStop, DumpProcEvent::kOutOfRange), DumpProcState::kInit);

  EXPECT_EQ(DumpProperties::DumpProcFsmTransition(DumpProcState::kError, DumpProcEvent::kRangeStart), DumpProcState::kError);
  EXPECT_EQ(DumpProperties::DumpProcFsmTransition(DumpProcState::kError, DumpProcEvent::kRangeEnd), DumpProcState::kError);
  EXPECT_EQ(DumpProperties::DumpProcFsmTransition(DumpProcState::kError, DumpProcEvent::kOutOfRange), DumpProcState::kError);

  EXPECT_EQ(DumpProperties::DumpProcFsmTransition(DumpProcState::kEnd, DumpProcEvent::kRangeStart), DumpProcState::kError);
  EXPECT_EQ(DumpProperties::DumpProcFsmTransition(DumpProcState::kInit, DumpProcEvent::kEnd), DumpProcState::kError);
}

TEST_F(UTEST_dump_properties, check_SetDumpFsmState) {
  DumpProperties dp;
  std::string model = "model_test";
  std::vector<std::pair<std::string, std::string>> op_ranges = {{"b", "d"}, {"c", "e"}};
  dp.SetOpDumpRange("", op_ranges);

  // op list:a, b, c, d, e, f need dump op:b, c, e
  std::vector<DumpProcState> dump_fsm_state = {DumpProcState::kInit, DumpProcState::kInit};
  std::unordered_set<std::string> dump_op_in_range;
  dp.SetOpDumpRange(model, op_ranges);

  // 未配置的model
  EXPECT_EQ(dp.SetDumpFsmState("test", "", "a", dump_fsm_state, dump_op_in_range), ge::SUCCESS);

  // 配置model
  EXPECT_EQ(dp.SetDumpFsmState(model, "", "a", dump_fsm_state, dump_op_in_range), ge::SUCCESS);
  EXPECT_EQ(dump_fsm_state[0], DumpProcState::kInit);
  EXPECT_EQ(dump_fsm_state[1], DumpProcState::kInit);
  EXPECT_EQ(dump_op_in_range.size(), 0);

  EXPECT_EQ(dp.SetDumpFsmState(model, "", "b", dump_fsm_state, dump_op_in_range), ge::SUCCESS);
  EXPECT_EQ(dump_fsm_state[0], DumpProcState::kStart);
  EXPECT_EQ(dump_fsm_state[1], DumpProcState::kInit);
  EXPECT_EQ(dump_op_in_range.size(), 1);

  EXPECT_EQ(dp.SetDumpFsmState(model, "", "c", dump_fsm_state, dump_op_in_range), ge::SUCCESS);
  EXPECT_EQ(dump_fsm_state[0], DumpProcState::kStart);
  EXPECT_EQ(dump_fsm_state[1], DumpProcState::kStart);
  EXPECT_EQ(dump_op_in_range.size(), 2);

  // is_update_dump_op_range is false
  EXPECT_EQ(dp.SetDumpFsmState(model, "", "d", dump_fsm_state, dump_op_in_range, false), ge::SUCCESS);
  EXPECT_EQ(dump_fsm_state[0], DumpProcState::kStop);
  EXPECT_EQ(dump_fsm_state[1], DumpProcState::kStart);
  EXPECT_EQ(dump_op_in_range.size(), 2);

  EXPECT_EQ(dp.SetDumpFsmState(model, "", "e", dump_fsm_state, dump_op_in_range), ge::SUCCESS);
  EXPECT_EQ(dump_fsm_state[0], DumpProcState::kInit);
  EXPECT_EQ(dump_fsm_state[1], DumpProcState::kStop);
  EXPECT_EQ(dump_op_in_range.size(), 3);

  EXPECT_EQ(dp.SetDumpFsmState(model, "", "f", dump_fsm_state, dump_op_in_range), ge::SUCCESS);
  EXPECT_EQ(dump_fsm_state[0], DumpProcState::kInit);
  EXPECT_EQ(dump_fsm_state[1], DumpProcState::kInit);
  EXPECT_EQ(dump_op_in_range.size(), 3);

  // range start the same with range end
  std::vector<std::pair<std::string, std::string>> op_ranges_only_one_op = {{"a", "a"}};
  dump_fsm_state.assign(1, DumpProcState::kInit);
  dump_op_in_range.clear();
  dp.SetOpDumpRange(model, op_ranges_only_one_op);
  EXPECT_EQ(dp.SetDumpFsmState(model, "", "a", dump_fsm_state, dump_op_in_range), ge::SUCCESS);
  EXPECT_EQ(dump_fsm_state[0], DumpProcState::kStop);

  // kError1
  std::vector<std::pair<std::string, std::string>> op_ranges_error1 = {{"b", "a"}};
  dump_fsm_state.assign(1, DumpProcState::kInit);
  dump_op_in_range.clear();
  dp.SetOpDumpRange(model, op_ranges_error1);
  EXPECT_EQ(dp.SetDumpFsmState(model, "", "a", dump_fsm_state, dump_op_in_range), ge::SUCCESS);
  EXPECT_EQ(dump_fsm_state[0], DumpProcState::kError);

  EXPECT_EQ(dp.SetDumpFsmState(model, "", "b", dump_fsm_state, dump_op_in_range), ge::SUCCESS);
  EXPECT_EQ(dump_fsm_state[0], DumpProcState::kError);

  EXPECT_EQ(dp.SetDumpFsmState(model, "", "a", dump_fsm_state, dump_op_in_range), ge::SUCCESS);
  EXPECT_EQ(dump_fsm_state[0], DumpProcState::kError);

  EXPECT_EQ(dp.SetDumpFsmState(model, "", "d", dump_fsm_state, dump_op_in_range), ge::SUCCESS);
  EXPECT_EQ(dump_fsm_state[0], DumpProcState::kError);

  // kError2
  std::vector<std::pair<std::string, std::string>> op_ranges_error2 = {{"a", "b"}};
  dump_fsm_state.assign(1, DumpProcState::kInit);
  dump_op_in_range.clear();
  dp.SetOpDumpRange(model, op_ranges_error2);
  EXPECT_EQ(dp.SetDumpFsmState(model, "", "a", dump_fsm_state, dump_op_in_range), ge::SUCCESS);
  EXPECT_EQ(dump_fsm_state[0], DumpProcState::kStart);

  EXPECT_EQ(dp.SetDumpFsmState(model, "", "b", dump_fsm_state, dump_op_in_range), ge::SUCCESS);
  EXPECT_EQ(dump_fsm_state[0], DumpProcState::kStop);

  EXPECT_EQ(dp.SetDumpFsmState(model, "", "a", dump_fsm_state, dump_op_in_range), ge::SUCCESS);
  EXPECT_EQ(dump_fsm_state[0], DumpProcState::kError);

  // kError3
  std::vector<std::pair<std::string, std::string>> op_ranges_error3 = {{"a", "b"}};
  dump_fsm_state.assign(2, DumpProcState::kInit);
  dump_op_in_range.clear();
  dp.SetOpDumpRange(model, op_ranges_error3);
  EXPECT_NE(dp.SetDumpFsmState(model, "", "a", dump_fsm_state, dump_op_in_range), ge::SUCCESS);

  // kError4
  std::vector<std::pair<std::string, std::string>> op_ranges_error4 = {{"a", "b"},{"c", "d"}};
  dump_fsm_state.assign(1, DumpProcState::kInit);
  dump_op_in_range.clear();
  dp.SetOpDumpRange(model, op_ranges_error4);
  EXPECT_NE(dp.SetDumpFsmState(model, "", "a", dump_fsm_state, dump_op_in_range), ge::SUCCESS);
}
}  // namespace ge
