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
#include "graph/arg_desc_info.h"
#include "graph/utils/args_format_desc_utils.h"
#include "ge_common/ge_api_error_codes.h"


namespace ge {
class TestArgDescInfo : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

// 验证ArgDescInfo的反序列化能力
TEST_F(TestArgDescInfo, ArgDescInfoDeserialize) {
  std::vector<ArgDesc> args;
  ArgsFormatDescUtils::InsertCustomValue(args, 0, 2);
  ArgsFormatDescUtils::InsertHiddenInputs(args, 1, HiddenInputsType::HCOM, 2);
  ArgsFormatDescUtils::Append(args, AddrType::INPUT, 0);
  ArgsFormatDescUtils::Append(args, AddrType::INPUT_DESC, 1, true);
  ArgsFormatDescUtils::Append(args, AddrType::OUTPUT, 0);
  ArgsFormatDescUtils::Append(args, AddrType::OUTPUT_DESC, 1, true);
  ArgsFormatDescUtils::Append(args, AddrType::WORKSPACE);
  ArgsFormatDescUtils::Append(args, AddrType::TILING);
  auto args_format_str = ArgsFormatDescUtils::Serialize(args);
  auto args_infos = ArgsFormatSerializer::Deserialize(args_format_str.c_str());
  EXPECT_EQ(args_infos.size(), 9);
  EXPECT_EQ(args_infos[0].GetType(), ArgDescType::kCustomValue);
  EXPECT_EQ(args_infos[0].GetIrIndex(), -1);
  EXPECT_EQ(args_infos[0].GetCustomValue(), 2);
  EXPECT_EQ(args_infos[0].GetHiddenInputSubType(), HiddenInputSubType::kEnd);
  EXPECT_EQ(args_infos[0].IsFolded(), false);

  EXPECT_EQ(args_infos[1].GetType(), ArgDescType::kHiddenInput);
  EXPECT_EQ(args_infos[1].GetIrIndex(), -1);
  EXPECT_EQ(args_infos[1].GetCustomValue(), 0);
  EXPECT_EQ(args_infos[1].GetHiddenInputSubType(), HiddenInputSubType::kHcom);
  EXPECT_EQ(args_infos[1].IsFolded(), false);

  EXPECT_EQ(args_infos[2].GetType(), ArgDescType::kHiddenInput);
  EXPECT_EQ(args_infos[2].GetIrIndex(), -1);
  EXPECT_EQ(args_infos[2].GetCustomValue(), 0);
  EXPECT_EQ(args_infos[2].GetHiddenInputSubType(), HiddenInputSubType::kHcom);
  EXPECT_EQ(args_infos[2].IsFolded(), false);

  EXPECT_EQ(args_infos[3].GetType(), ArgDescType::kIrInput);
  EXPECT_EQ(args_infos[3].GetIrIndex(), 0);
  EXPECT_EQ(args_infos[3].GetCustomValue(), 0);
  EXPECT_EQ(args_infos[3].GetHiddenInputSubType(), HiddenInputSubType::kEnd);
  EXPECT_EQ(args_infos[3].IsFolded(), false);

  EXPECT_EQ(args_infos[4].GetType(), ArgDescType::kIrInputDesc);
  EXPECT_EQ(args_infos[4].GetIrIndex(), 1);
  EXPECT_EQ(args_infos[4].GetCustomValue(), 0);
  EXPECT_EQ(args_infos[4].GetHiddenInputSubType(), HiddenInputSubType::kEnd);
  EXPECT_EQ(args_infos[4].IsFolded(), true);

  EXPECT_EQ(args_infos[5].GetType(), ArgDescType::kIrOutput);
  EXPECT_EQ(args_infos[5].GetIrIndex(), 0);
  EXPECT_EQ(args_infos[5].GetCustomValue(), 0);
  EXPECT_EQ(args_infos[5].GetHiddenInputSubType(), HiddenInputSubType::kEnd);
  EXPECT_EQ(args_infos[5].IsFolded(), false);

  EXPECT_EQ(args_infos[6].GetType(), ArgDescType::kIrOutputDesc);
  EXPECT_EQ(args_infos[6].GetIrIndex(), 1);
  EXPECT_EQ(args_infos[6].GetCustomValue(), 0);
  EXPECT_EQ(args_infos[6].GetHiddenInputSubType(), HiddenInputSubType::kEnd);
  EXPECT_EQ(args_infos[6].IsFolded(), true);

  EXPECT_EQ(args_infos[7].GetType(), ArgDescType::kWorkspace);
  EXPECT_EQ(args_infos[7].GetIrIndex(), -1);
  EXPECT_EQ(args_infos[7].GetCustomValue(), 0);
  EXPECT_EQ(args_infos[7].GetHiddenInputSubType(), HiddenInputSubType::kEnd);
  EXPECT_EQ(args_infos[7].IsFolded(), false);

  EXPECT_EQ(args_infos[8].GetType(), ArgDescType::kTiling);
  EXPECT_EQ(args_infos[8].GetIrIndex(), -1);
  EXPECT_EQ(args_infos[8].GetCustomValue(), 0);
  EXPECT_EQ(args_infos[8].GetHiddenInputSubType(), HiddenInputSubType::kEnd);
  EXPECT_EQ(args_infos[8].IsFolded(), false);
}

// 验证ArgDescInfo的序列化能力
TEST_F(TestArgDescInfo, ArgDescInfoSerialize) {
  std::vector<ArgDescInfo> args_desc;
  auto custom_value_arg = ArgDescInfo(ArgDescType::kCustomValue);
  EXPECT_EQ(custom_value_arg.SetCustomValue(2), SUCCESS);
  args_desc.emplace_back(custom_value_arg);
  args_desc.emplace_back(ArgDescInfo::CreateCustomValue(0));
  auto hidden_input_arg = ArgDescInfo(ArgDescType::kHiddenInput, 0);
  EXPECT_EQ(hidden_input_arg.SetHiddenInputSubType(HiddenInputSubType::kHcom), SUCCESS);
  args_desc.emplace_back(hidden_input_arg);
  args_desc.emplace_back(ArgDescInfo::CreateHiddenInput(HiddenInputSubType::kHcom));
  args_desc.emplace_back(ArgDescInfo(ArgDescType::kIrInput, 0));
  auto input_desc_arg = ArgDescInfo(ArgDescType::kIrInputDesc, 1);
  input_desc_arg.SetFolded(true);
  args_desc.emplace_back(input_desc_arg);
  args_desc.emplace_back(ArgDescInfo(ArgDescType::kIrOutput, 0));
  args_desc.emplace_back(ArgDescInfo(ArgDescType::kIrOutputDesc, 1, true));
  args_desc.emplace_back(ArgDescInfo(ArgDescType::kWorkspace));
  args_desc.emplace_back(ArgDescInfo(ArgDescType::kTiling));
  auto args_format_str = ArgsFormatSerializer::Serialize(args_desc);
  EXPECT_EQ(std::string(args_format_str.GetString()), "{#2}{#0}{hi.hcom0*}{hi.hcom1*}{i0*}{i_desc1}{o0*}{o_desc1}{ws*}{t}");
}

// 验证ArgDescInfo的反序列化能力,args序列中有InputInstance和OutputInstance
TEST_F(TestArgDescInfo, ArgDescInfoWithInstanceDeserialize) {
  std::vector<ArgDesc> args;
  ArgsFormatDescUtils::InsertCustomValue(args, 0, 2);
  ArgsFormatDescUtils::InsertHiddenInputs(args, 1, HiddenInputsType::HCOM, 2);
  ArgsFormatDescUtils::Append(args, AddrType::INPUT_INSTANCE, 0);
  ArgsFormatDescUtils::Append(args, AddrType::INPUT_INSTANCE, 1);
  ArgsFormatDescUtils::Append(args, AddrType::OUTPUT_INSTANCE, 0);
  ArgsFormatDescUtils::Append(args, AddrType::OUTPUT_INSTANCE, 1);
  ArgsFormatDescUtils::Append(args, AddrType::WORKSPACE);
  ArgsFormatDescUtils::Append(args, AddrType::TILING);
  auto args_format_str = ArgsFormatDescUtils::Serialize(args);
  auto args_infos = ArgsFormatSerializer::Deserialize(args_format_str.c_str());
  EXPECT_EQ(args_infos.size(), 9);
  EXPECT_EQ(args_infos[0].GetType(), ArgDescType::kCustomValue);
  EXPECT_EQ(args_infos[0].GetIrIndex(), -1);
  EXPECT_EQ(args_infos[0].GetCustomValue(), 2);
  EXPECT_EQ(args_infos[0].GetHiddenInputSubType(), HiddenInputSubType::kEnd);
  EXPECT_EQ(args_infos[0].IsFolded(), false);

  EXPECT_EQ(args_infos[1].GetType(), ArgDescType::kHiddenInput);
  EXPECT_EQ(args_infos[1].GetIrIndex(), -1);
  EXPECT_EQ(args_infos[1].GetCustomValue(), 0);
  EXPECT_EQ(args_infos[1].GetHiddenInputSubType(), HiddenInputSubType::kHcom);
  EXPECT_EQ(args_infos[1].IsFolded(), false);

  EXPECT_EQ(args_infos[2].GetType(), ArgDescType::kHiddenInput);
  EXPECT_EQ(args_infos[2].GetIrIndex(), -1);
  EXPECT_EQ(args_infos[2].GetCustomValue(), 0);
  EXPECT_EQ(args_infos[2].GetHiddenInputSubType(), HiddenInputSubType::kHcom);
  EXPECT_EQ(args_infos[2].IsFolded(), false);

  EXPECT_EQ(args_infos[3].GetType(), ArgDescType::kInputInstance);
  EXPECT_EQ(args_infos[3].GetIrIndex(), -1);
  EXPECT_EQ(args_infos[3].GetCustomValue(), 0);
  EXPECT_EQ(args_infos[3].GetHiddenInputSubType(), HiddenInputSubType::kEnd);
  EXPECT_EQ(args_infos[3].IsFolded(), false);

  EXPECT_EQ(args_infos[4].GetType(), ArgDescType::kInputInstance);
  EXPECT_EQ(args_infos[4].GetIrIndex(), -1);
  EXPECT_EQ(args_infos[4].GetCustomValue(), 0);
  EXPECT_EQ(args_infos[4].GetHiddenInputSubType(), HiddenInputSubType::kEnd);
  EXPECT_EQ(args_infos[4].IsFolded(), false);

  EXPECT_EQ(args_infos[5].GetType(), ArgDescType::kOutputInstance);
  EXPECT_EQ(args_infos[5].GetIrIndex(), -1);
  EXPECT_EQ(args_infos[5].GetCustomValue(), 0);
  EXPECT_EQ(args_infos[5].GetHiddenInputSubType(), HiddenInputSubType::kEnd);
  EXPECT_EQ(args_infos[5].IsFolded(), false);

  EXPECT_EQ(args_infos[6].GetType(), ArgDescType::kOutputInstance);
  EXPECT_EQ(args_infos[6].GetIrIndex(), -1);
  EXPECT_EQ(args_infos[6].GetCustomValue(), 0);
  EXPECT_EQ(args_infos[6].GetHiddenInputSubType(), HiddenInputSubType::kEnd);
  EXPECT_EQ(args_infos[6].IsFolded(), false);

  EXPECT_EQ(args_infos[7].GetType(), ArgDescType::kWorkspace);
  EXPECT_EQ(args_infos[7].GetIrIndex(), -1);
  EXPECT_EQ(args_infos[7].GetCustomValue(), 0);
  EXPECT_EQ(args_infos[7].GetHiddenInputSubType(), HiddenInputSubType::kEnd);
  EXPECT_EQ(args_infos[7].IsFolded(), false);

  EXPECT_EQ(args_infos[8].GetType(), ArgDescType::kTiling);
  EXPECT_EQ(args_infos[8].GetIrIndex(), -1);
  EXPECT_EQ(args_infos[8].GetCustomValue(), 0);
  EXPECT_EQ(args_infos[8].GetHiddenInputSubType(), HiddenInputSubType::kEnd);
  EXPECT_EQ(args_infos[8].IsFolded(), false);
}

// 验证ArgDescInfo的序列化能力,args中存在InputInstance和OutputInstance
TEST_F(TestArgDescInfo, ArgDescInfoWithInstanceSerialize) {
  std::vector<ArgDescInfo> args_desc;
  auto custom_value_arg = ArgDescInfo(ArgDescType::kCustomValue);
  EXPECT_EQ(custom_value_arg.SetCustomValue(2), SUCCESS);
  args_desc.emplace_back(custom_value_arg);
  args_desc.emplace_back(ArgDescInfo::CreateCustomValue(0));
  auto hidden_input_arg = ArgDescInfo(ArgDescType::kHiddenInput, 0);
  EXPECT_EQ(hidden_input_arg.SetHiddenInputSubType(HiddenInputSubType::kHcom), SUCCESS);
  args_desc.emplace_back(hidden_input_arg);
  args_desc.emplace_back(ArgDescInfo::CreateHiddenInput(HiddenInputSubType::kHcom));
  args_desc.emplace_back(ArgDescInfo(ArgDescType::kInputInstance));
  args_desc.emplace_back(ArgDescInfo(ArgDescType::kInputInstance));
  args_desc.emplace_back(ArgDescInfo(ArgDescType::kOutputInstance));
  args_desc.emplace_back(ArgDescInfo(ArgDescType::kOutputInstance));
  args_desc.emplace_back(ArgDescInfo(ArgDescType::kWorkspace));
  args_desc.emplace_back(ArgDescInfo(ArgDescType::kTiling));
  auto args_format_str = ArgsFormatSerializer::Serialize(args_desc);
  EXPECT_EQ(std::string(args_format_str.GetString()),
      "{#2}{#0}{hi.hcom0*}{hi.hcom1*}{i_instance0*}{i_instance1*}{o_instance0*}{o_instance1*}{ws*}{t}");
}

// 验证ArgDescInfo的拷贝赋值函数
TEST_F(TestArgDescInfo, ArgDescInfoCopyAssignFunc) {
  ArgDescInfo args_desc_1(ArgDescType::kIrInput, 0, true);
  args_desc_1 = ArgDescInfo(ArgDescType::kIrInput, 1, true);
  EXPECT_EQ(args_desc_1.GetType(), ArgDescType::kIrInput);
  EXPECT_EQ(args_desc_1.GetIrIndex(), 1);
  EXPECT_EQ(args_desc_1.IsFolded(), true);
  EXPECT_EQ(args_desc_1.GetCustomValue(), 0);
  EXPECT_EQ(args_desc_1.GetHiddenInputSubType(), HiddenInputSubType::kEnd);

  args_desc_1 = ArgDescInfo::CreateCustomValue(2);
  EXPECT_EQ(args_desc_1.GetType(), ArgDescType::kCustomValue);
  EXPECT_EQ(args_desc_1.GetIrIndex(), -1);
  EXPECT_EQ(args_desc_1.IsFolded(), false);
  EXPECT_EQ(args_desc_1.GetCustomValue(), 2);
  EXPECT_EQ(args_desc_1.GetHiddenInputSubType(), HiddenInputSubType::kEnd);

  args_desc_1 = ArgDescInfo::CreateHiddenInput(HiddenInputSubType::kHcom);
  EXPECT_EQ(args_desc_1.GetType(), ArgDescType::kHiddenInput);
  EXPECT_EQ(args_desc_1.GetIrIndex(), -1);
  EXPECT_EQ(args_desc_1.IsFolded(), false);
  EXPECT_EQ(args_desc_1.GetCustomValue(), 0);
  EXPECT_EQ(args_desc_1.GetHiddenInputSubType(), HiddenInputSubType::kHcom);
}

// 验证ArgDescInfo的拷贝构造函数
TEST_F(TestArgDescInfo, ArgDescInfoCopyConstructFunc) {
  ArgDescInfo args_desc_input_desc(ArgDescType::kIrInput, 1, true);
  ArgDescInfo args_desc_1(args_desc_input_desc);
  EXPECT_EQ(args_desc_1.GetType(), ArgDescType::kIrInput);
  EXPECT_EQ(args_desc_1.GetIrIndex(), 1);
  EXPECT_EQ(args_desc_1.IsFolded(), true);
  EXPECT_EQ(args_desc_1.GetCustomValue(), 0);
  EXPECT_EQ(args_desc_1.GetHiddenInputSubType(), HiddenInputSubType::kEnd);
  ArgDescInfo args_desc_custom_value = ArgDescInfo::CreateCustomValue(2);
  ArgDescInfo args_desc_2(args_desc_custom_value);
  EXPECT_EQ(args_desc_2.GetType(), ArgDescType::kCustomValue);
  EXPECT_EQ(args_desc_2.GetIrIndex(), -1);
  EXPECT_EQ(args_desc_2.IsFolded(), false);
  EXPECT_EQ(args_desc_2.GetCustomValue(), 2);
  EXPECT_EQ(args_desc_2.GetHiddenInputSubType(), HiddenInputSubType::kEnd);

  auto args_desc_hidden_input = ArgDescInfo::CreateHiddenInput(HiddenInputSubType::kHcom);
  ArgDescInfo args_desc_3(args_desc_hidden_input);
  EXPECT_EQ(args_desc_3.GetType(), ArgDescType::kHiddenInput);
  EXPECT_EQ(args_desc_3.GetIrIndex(), -1);
  EXPECT_EQ(args_desc_3.IsFolded(), false);
  EXPECT_EQ(args_desc_3.GetCustomValue(), 0);
  EXPECT_EQ(args_desc_3.GetHiddenInputSubType(), HiddenInputSubType::kHcom);
}

// 验证ArgDescInfo的移动赋值函数
TEST_F(TestArgDescInfo, ArgDescInfoMoveAssignFunc) {
  ArgDescInfo args_desc_1(ArgDescType::kIrInput, 0, true);
  auto args_desc_input_desc = ArgDescInfo(ArgDescType::kIrInput, 1, true);
  args_desc_1 = std::move(args_desc_input_desc);
  EXPECT_EQ(args_desc_1.GetType(), ArgDescType::kIrInput);
  EXPECT_EQ(args_desc_1.GetIrIndex(), 1);
  EXPECT_EQ(args_desc_1.IsFolded(), true);
  EXPECT_EQ(args_desc_1.GetCustomValue(), 0);
  EXPECT_EQ(args_desc_1.GetHiddenInputSubType(), HiddenInputSubType::kEnd);

  auto args_desc_custom_value = ArgDescInfo::CreateCustomValue(2);
  args_desc_1 = std::move(args_desc_custom_value);
  EXPECT_EQ(args_desc_1.GetType(), ArgDescType::kCustomValue);
  EXPECT_EQ(args_desc_1.GetIrIndex(), -1);
  EXPECT_EQ(args_desc_1.IsFolded(), false);
  EXPECT_EQ(args_desc_1.GetCustomValue(), 2);
  EXPECT_EQ(args_desc_1.GetHiddenInputSubType(), HiddenInputSubType::kEnd);

  auto args_desc_hidden_input = ArgDescInfo::CreateHiddenInput(HiddenInputSubType::kHcom);
  args_desc_1 = std::move(args_desc_hidden_input);
  EXPECT_EQ(args_desc_1.GetType(), ArgDescType::kHiddenInput);
  EXPECT_EQ(args_desc_1.GetIrIndex(), -1);
  EXPECT_EQ(args_desc_1.IsFolded(), false);
  EXPECT_EQ(args_desc_1.GetCustomValue(), 0);
  EXPECT_EQ(args_desc_1.GetHiddenInputSubType(), HiddenInputSubType::kHcom);
}

// 验证ArgDescInfo的移动构造函数
TEST_F(TestArgDescInfo, ArgDescInfoMoveConstructFunc) {
  ArgDescInfo args_desc_input_desc(ArgDescType::kIrInput, 1, true);
  ArgDescInfo args_desc_1(std::move(args_desc_input_desc));
  EXPECT_EQ(args_desc_1.GetType(), ArgDescType::kIrInput);
  EXPECT_EQ(args_desc_1.GetIrIndex(), 1);
  EXPECT_EQ(args_desc_1.IsFolded(), true);
  EXPECT_EQ(args_desc_1.GetCustomValue(), 0);
  EXPECT_EQ(args_desc_1.GetHiddenInputSubType(), HiddenInputSubType::kEnd);
  ArgDescInfo args_desc_custom_value = ArgDescInfo::CreateCustomValue(2);
  ArgDescInfo args_desc_2(std::move(args_desc_custom_value));
  EXPECT_EQ(args_desc_2.GetType(), ArgDescType::kCustomValue);
  EXPECT_EQ(args_desc_2.GetIrIndex(), -1);
  EXPECT_EQ(args_desc_2.IsFolded(), false);
  EXPECT_EQ(args_desc_2.GetCustomValue(), 2);
  EXPECT_EQ(args_desc_2.GetHiddenInputSubType(), HiddenInputSubType::kEnd);

  auto args_desc_hidden_input = ArgDescInfo::CreateHiddenInput(HiddenInputSubType::kHcom);
  ArgDescInfo args_desc_3(std::move(args_desc_hidden_input));
  EXPECT_EQ(args_desc_3.GetType(), ArgDescType::kHiddenInput);
  EXPECT_EQ(args_desc_3.GetIrIndex(), -1);
  EXPECT_EQ(args_desc_3.IsFolded(), false);
  EXPECT_EQ(args_desc_3.GetCustomValue(), 0);
  EXPECT_EQ(args_desc_3.GetHiddenInputSubType(), HiddenInputSubType::kHcom);
}
}