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
#include <vector>
#include "inc/ffts_type.h"
#include "common/fe_log.h"
#include "common/aicore_util_constants.h"
#include "common/aicore_util_attr_define.h"
#include "framework/common/types.h"
#include "runtime/rt_model.h"
#include "rt_error_codes.h"
#define private public
#define protected public
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/compute_graph.h"
#include "ops_store/sub_op_info_store.h"
#include "ops_store/ops_kernel_manager.h"
#include "common/configuration.h"
#undef private
#undef protected
#include "ops_kernel_builder/task_builder/args_format_constructor.h"
#include "fe_llt_utils.h"
#include "../fe_test_utils.h"
using namespace std;
using namespace testing;
using namespace ge;
using namespace fe;
using namespace ffts;

class UTEST_ArgsFormat : public testing::Test {
 protected:

  void SetUp() {
    FEOpsStoreInfo TIK_CUSTOM_OPINFO_STUB  = {
        1,
        "tik-custom",
        EN_IMPL_CUSTOM_TIK,
        GetCodeDir() + "/tests/engines/nn_engine/st/stub/fe_config/tik_custom_opinfo",
        ""
    };
    FEOpsStoreInfo TBE_CUSTOM_OPINFO_STUB = {
        2,
        "tbe-custom",
        EN_IMPL_CUSTOM_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/st/stub/fe_config/tbe_custom_opinfo",
        ""
    };
    FEOpsStoreInfo TIK_OPINFO_STUB = {
        5,
        "tik-builtin",
        EN_IMPL_HW_TIK,
        GetCodeDir() + "/tests/engines/nn_engine/st/stub/fe_config/tik_opinfo",
        ""
    };
    FEOpsStoreInfo TBE_OPINFO_STUB = {
        6,
        "tbe-builtin",
        EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/st/stub/fe_config/tbe_opinfo",
        ""
    };
    FEOpsStoreInfo RL_OPINFO_STUB = {
        7,
        "rl-builtin",
        EN_IMPL_RL,
        GetCodeDir() + "/tests/engines/nn_engine/st/stub/fe_config/rl_opinfo",
        ""
    };
    std::vector<FEOpsStoreInfo> cfg_info_in_l2;
    cfg_info_in_l2.push_back(TIK_CUSTOM_OPINFO_STUB);
    cfg_info_in_l2.push_back(TBE_CUSTOM_OPINFO_STUB);
    cfg_info_in_l2.push_back(TIK_OPINFO_STUB);
    cfg_info_in_l2.push_back(TBE_OPINFO_STUB);
    cfg_info_in_l2.push_back(RL_OPINFO_STUB);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = cfg_info_in_l2;
    OpsKernelManager::Instance(fe::AI_CORE_NAME).Finalize();
    OpsKernelManager::Instance(fe::AI_CORE_NAME).Initialize();
  }

  void TearDown() {
  }
};

TEST_F(UTEST_ArgsFormat, args_by_instance_test1)
{
  FeTestOpDescBuilder builder;
  builder.SetName("test_tvm");
  builder.SetType("addn");
  builder.SetInputs({1, 2, 3, 4});
  builder.SetOutputs({1});
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
  builder.AddOutputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
  auto node = builder.Finish();
  auto op_desc = node->GetOpDesc();
  ArgsFormatConstructor args_construct(op_desc, false);
  auto ret = args_construct.ConstructNodeArgsDesc();
  EXPECT_EQ(ret, fe::SUCCESS);
  auto args_format_str = args_construct.GetArgsFormatString();
  EXPECT_STREQ(args_format_str.c_str(), "{i_instance0*}{i_instance1*}{i_instance2*}{i_instance3*}{o_instance0*}");
}

TEST_F(UTEST_ArgsFormat, dynamic_args_test)
{
  FeTestOpDescBuilder builder;
  builder.SetName("test_tvm");
  builder.SetType("addn");
  builder.SetInputs({1, 2, 3, 4});
  builder.SetOutputs({1});
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "x0");
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "x1");
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "w0");
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "b0");
  builder.AddOutputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "y0");
  auto node = builder.Finish();
  auto op_desc = node->GetOpDesc();

  size_t in_num = node->GetOpDesc()->GetAllInputsSize();
  size_t out_num = node->GetOpDesc()->GetOutputsSize();
  std::vector<uint32_t> input_type_list = {2, 2, 2, 0};
  (void)ge::AttrUtils::SetListInt(op_desc, kInputParaTypeList, input_type_list);
  std::vector<std::string> input_name_list = {"err", "err", "w", "b"};
  (void)ge::AttrUtils::SetListStr(op_desc, kInputNameList, input_name_list);
  std::vector<uint32_t> output_type_list(out_num, static_cast<uint32_t>(0));
  (void)ge::AttrUtils::SetListInt(op_desc, kOutputParaTypeList, output_type_list);
  std::vector<std::string> output_name_list = {"y"};
  (void)ge::AttrUtils::SetListStr(op_desc, kOutputNameList, output_name_list);

  (void)ge::AttrUtils::SetStr(op_desc, fe::kAttrDynamicParamMode, fe::kFoldedWithDesc);

  auto pure_op_desc = node->GetOpDescBarePtr();
  pure_op_desc->AppendIrInput("x", ge::kIrInputDynamic); // not equal with ops kernel
  pure_op_desc->AppendIrInput("w", ge::kIrInputDynamic);
  pure_op_desc->AppendIrInput("b", ge::kIrInputRequired);
  pure_op_desc->AppendIrOutput("y", ge::kIrOutputRequired);
  ArgsFormatConstructor args_construct(op_desc, false);
  auto ret = args_construct.ConstructNodeArgsDesc();
  EXPECT_NE(ret, fe::SUCCESS);

  input_name_list = {"x", "x", "w", "b"};
  (void)ge::AttrUtils::SetListStr(op_desc, kInputNameList, input_name_list);
  ArgsFormatConstructor args_construct1(op_desc, false);
  ret = args_construct1.ConstructNodeArgsDesc();
  EXPECT_EQ(ret, fe::SUCCESS);
  auto args_format_str = args_construct1.GetArgsFormatString();
  EXPECT_STREQ(args_format_str.c_str(), "{i_desc0}{i_desc1}{i2*}{o0*}");
}

TEST_F(UTEST_ArgsFormat, option_and_dynamic_args_test)
{
  FeTestOpDescBuilder builder;
  builder.SetName("test_tvm");
  builder.SetType("addn");
  builder.SetInputs({1, 2, 3, 4});
  builder.SetOutputs({1});
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "x0");
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "y0");
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "z0");
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "h0");
  builder.AddOutputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "o0");
  auto node = builder.Finish();
  auto op_desc = node->GetOpDesc();

  size_t in_num = node->GetOpDesc()->GetAllInputsSize();
  size_t out_num = node->GetOpDesc()->GetOutputsSize();
  std::vector<uint32_t> input_type_list = {0, 2, 1, 1};
  (void)ge::AttrUtils::SetListInt(op_desc, kInputParaTypeList, input_type_list);
  std::vector<std::string> input_name_list = {"x", "y", "err", "h"};
  (void)ge::AttrUtils::SetListStr(op_desc, kInputNameList, input_name_list);
  std::vector<uint32_t> output_type_list(out_num, static_cast<uint32_t>(0));
  (void)ge::AttrUtils::SetListInt(op_desc, kOutputParaTypeList, output_type_list);
  std::vector<std::string> output_name_list = {"o"};
  (void)ge::AttrUtils::SetListStr(op_desc, kOutputNameList, output_name_list);

  (void)ge::AttrUtils::SetStr(op_desc, fe::kAttrDynamicParamMode, fe::kFoldedWithDesc);
  (void)ge::AttrUtils::SetStr(op_desc, fe::kAttrOptionalInputMode, fe::kGenPlaceholder);

  auto pure_op_desc = node->GetOpDescBarePtr();
  pure_op_desc->AppendIrInput("x", ge::kIrInputRequired);
  pure_op_desc->AppendIrInput("y", ge::kIrInputDynamic);
  pure_op_desc->AppendIrInput("z", ge::kIrInputOptional);
  pure_op_desc->AppendIrInput("h", ge::kIrInputOptional);
  pure_op_desc->AppendIrOutput("o", ge::kIrOutputRequired);
  ArgsFormatConstructor args_construct(op_desc, false);
  auto ret = args_construct.ConstructNodeArgsDesc();
  EXPECT_NE(ret, fe::SUCCESS);


  input_name_list = {"x", "y", "z", "h"};
  (void)ge::AttrUtils::SetListStr(op_desc, kInputNameList, input_name_list);
  ArgsFormatConstructor args_construct1(op_desc, false);
  ret = args_construct1.ConstructNodeArgsDesc();
  EXPECT_EQ(ret, fe::SUCCESS);
  auto args_format_str = args_construct1.GetArgsFormatString();
  EXPECT_STREQ(args_format_str.c_str(), "{i0*}{i_desc1}{i2*}{i3*}{o0*}");

  // 动态输入y对应图上个数为0，报错
  auto &input_name_map = pure_op_desc->MutableAllInputName();
  auto index = input_name_map["y0"];
  input_name_map["err0"] = index;
  ArgsFormatConstructor args_construct2(op_desc, false);
  ret = args_construct2.ConstructNodeArgsDesc();
  EXPECT_NE(ret, fe::SUCCESS);
}

TEST_F(UTEST_ArgsFormat, IFA_append_by_ops_kernel_test)
{
  FeTestOpDescBuilder builder;
  builder.SetName("test_tvm");
  builder.SetType("IncreFlashAttention");
  builder.SetInputs({1, 2, 3, 6});
  builder.SetOutputs({1});
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "a");
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "b0");
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "c");
  // builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "d");
  // builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "e");
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "f");
  // builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "g");
  builder.AddOutputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "out");
  auto node = builder.Finish();
  auto op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, kOpKernelAllInputSize, 7);
  ge::AttrUtils::SetInt(op_desc, "_fe_imply_type", static_cast<int>(EN_IMPL_HW_TBE));

  size_t in_num = node->GetOpDesc()->GetAllInputsSize();
  size_t out_num = node->GetOpDesc()->GetOutputsSize();
  std::vector<uint32_t> input_type_list = {0, 2, 1, 1};
  (void)ge::AttrUtils::SetListInt(op_desc, kInputParaTypeList, input_type_list);
  std::vector<std::string> input_name_list = {"a", "b", "c", "f"};
  (void)ge::AttrUtils::SetListStr(op_desc, kInputNameList, input_name_list);
  std::vector<uint32_t> output_type_list(out_num, static_cast<uint32_t>(0));
  (void)ge::AttrUtils::SetListInt(op_desc, kOutputParaTypeList, output_type_list);
  std::vector<std::string> output_name_list = {"out"};
  (void)ge::AttrUtils::SetListStr(op_desc, kOutputNameList, output_name_list);

  (void)ge::AttrUtils::SetStr(op_desc, fe::kAttrDynamicParamMode, fe::kFoldedWithDesc);
  (void)ge::AttrUtils::SetStr(op_desc, fe::kAttrOptionalInputMode, fe::kGenPlaceholder);

  auto pure_op_desc = node->GetOpDescBarePtr();
  pure_op_desc->AppendIrInput("a", ge::kIrInputRequired);
  pure_op_desc->AppendIrInput("b", ge::kIrInputDynamic);
  pure_op_desc->AppendIrInput("c", ge::kIrInputOptional);
  pure_op_desc->AppendIrInput("d", ge::kIrInputOptional);
  pure_op_desc->AppendIrInput("e", ge::kIrInputOptional);
  pure_op_desc->AppendIrInput("f", ge::kIrInputOptional);
  pure_op_desc->AppendIrInput("g", ge::kIrInputOptional);
  pure_op_desc->AppendIrOutput("out", ge::kIrOutputRequired);

  ArgsFormatConstructor args_construct1(op_desc, false);
  auto ret = args_construct1.ConstructNodeArgsDesc();
  EXPECT_EQ(ret, fe::SUCCESS);
  auto args_format_str = args_construct1.GetArgsFormatString();
  EXPECT_STREQ(args_format_str.c_str(), "{i0*}{i_desc1}{i2*}{i3*}{i4*}{i5*}{i6*}{o0*}");
}

TEST_F(UTEST_ArgsFormat, IFA_append_by_ops_kernel_test2)
{
  FeTestOpDescBuilder builder;
  builder.SetName("test_tvm");
  builder.SetType("IncreFlashAttention");
  builder.SetInputs({1, 2, 3, 6});
  builder.SetOutputs({1});
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "a");
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "b0");
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "c");
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "d");
  // builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "e");
  // builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "f");
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "g");
  builder.AddOutputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "out");
  auto node = builder.Finish();
  auto op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, kOpKernelAllInputSize, 7);
  ge::AttrUtils::SetInt(op_desc, "_fe_imply_type", static_cast<int>(EN_IMPL_HW_TBE));

  size_t in_num = node->GetOpDesc()->GetAllInputsSize();
  size_t out_num = node->GetOpDesc()->GetOutputsSize();
  std::vector<uint32_t> input_type_list = {0, 2, 1, 1, 1};
  (void)ge::AttrUtils::SetListInt(op_desc, kInputParaTypeList, input_type_list);
  std::vector<std::string> input_name_list = {"a", "b", "c", "d", "g"};
  (void)ge::AttrUtils::SetListStr(op_desc, kInputNameList, input_name_list);
  std::vector<uint32_t> output_type_list(out_num, static_cast<uint32_t>(0));
  (void)ge::AttrUtils::SetListInt(op_desc, kOutputParaTypeList, output_type_list);
  std::vector<std::string> output_name_list = {"out"};
  (void)ge::AttrUtils::SetListStr(op_desc, kOutputNameList, output_name_list);

  (void)ge::AttrUtils::SetStr(op_desc, fe::kAttrDynamicParamMode, fe::kFoldedWithDesc);
  (void)ge::AttrUtils::SetStr(op_desc, fe::kAttrOptionalInputMode, fe::kGenPlaceholder);

  auto pure_op_desc = node->GetOpDescBarePtr();
  pure_op_desc->AppendIrInput("a", ge::kIrInputRequired);
  pure_op_desc->AppendIrInput("b", ge::kIrInputDynamic);
  pure_op_desc->AppendIrInput("c", ge::kIrInputOptional);
  pure_op_desc->AppendIrInput("d", ge::kIrInputOptional);
  pure_op_desc->AppendIrInput("e", ge::kIrInputOptional);
  pure_op_desc->AppendIrInput("f", ge::kIrInputOptional);
  pure_op_desc->AppendIrInput("g", ge::kIrInputOptional);
  pure_op_desc->AppendIrOutput("out", ge::kIrOutputRequired);

  ArgsFormatConstructor args_construct1(op_desc, false);
  auto ret = args_construct1.ConstructNodeArgsDesc();
  EXPECT_EQ(ret, fe::SUCCESS);
  auto args_format_str = args_construct1.GetArgsFormatString();
  EXPECT_STREQ(args_format_str.c_str(), "{i0*}{i_desc1}{i2*}{i3*}{i4*}{i5*}{i6*}{o0*}");
}

TEST_F(UTEST_ArgsFormat, IFA_append_by_ops_kernel_test3)
{
  FeTestOpDescBuilder builder;
  builder.SetName("test_tvm");
  builder.SetType("IncreFlashAttention");
  builder.SetInputs({1, 2, 3, 6});
  builder.SetOutputs({1});
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "a");
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "b0");
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "b1");
  // builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "c");
  // builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "d");
  // builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "e");
  // builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "f");
  builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "g");
  builder.AddOutputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "out");
  auto node = builder.Finish();
  auto op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, kOpKernelAllInputSize, 7);
  ge::AttrUtils::SetInt(op_desc, "_fe_imply_type", static_cast<int>(EN_IMPL_HW_TBE));

  size_t in_num = node->GetOpDesc()->GetAllInputsSize();
  size_t out_num = node->GetOpDesc()->GetOutputsSize();
  std::vector<uint32_t> input_type_list = {0, 2, 2, 1};
  (void)ge::AttrUtils::SetListInt(op_desc, kInputParaTypeList, input_type_list);
  std::vector<std::string> input_name_list = {"a", "b", "b","g"};
  (void)ge::AttrUtils::SetListStr(op_desc, kInputNameList, input_name_list);
  std::vector<uint32_t> output_type_list(out_num, static_cast<uint32_t>(0));
  (void)ge::AttrUtils::SetListInt(op_desc, kOutputParaTypeList, output_type_list);
  std::vector<std::string> output_name_list = {"out"};
  (void)ge::AttrUtils::SetListStr(op_desc, kOutputNameList, output_name_list);

  (void)ge::AttrUtils::SetStr(op_desc, fe::kAttrDynamicParamMode, fe::kFoldedWithDesc);
  (void)ge::AttrUtils::SetStr(op_desc, fe::kAttrOptionalInputMode, fe::kGenPlaceholder);

  auto pure_op_desc = node->GetOpDescBarePtr();
  pure_op_desc->AppendIrInput("a", ge::kIrInputRequired);
  pure_op_desc->AppendIrInput("b", ge::kIrInputDynamic);
  pure_op_desc->AppendIrInput("c", ge::kIrInputOptional);
  pure_op_desc->AppendIrInput("d", ge::kIrInputOptional);
  pure_op_desc->AppendIrInput("e", ge::kIrInputOptional);
  pure_op_desc->AppendIrInput("f", ge::kIrInputOptional);
  pure_op_desc->AppendIrInput("g", ge::kIrInputOptional);
  pure_op_desc->AppendIrOutput("out", ge::kIrOutputRequired);

  ArgsFormatConstructor args_construct1(op_desc, false);
  auto ret = args_construct1.ConstructNodeArgsDesc();
  EXPECT_EQ(ret, fe::SUCCESS);
  auto args_format_str = args_construct1.GetArgsFormatString();
  EXPECT_STREQ(args_format_str.c_str(), "{i0*}{i_desc1}{i2*}{i3*}{i4*}{i5*}{i6*}{o0*}");
}

TEST_F(UTEST_ArgsFormat, MC2_opt_output_test) {
  FeTestOpDescBuilder builder;
  builder.SetName("test_tvm");
  builder.SetType("MatmulAllReduce");
  builder.SetInputs({1, 2, 3, 4});
  builder.SetOutputs({1});
  builder.AddInputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "x0");
  builder.AddInputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "y0");
  builder.AddInputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "z0");
  builder.AddInputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "h0");
  builder.AddOutputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "o0");
  builder.AddOutputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "opt_x0");
  builder.AddOutputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "opt_y0");
  auto node = builder.Finish();
  auto op_desc = node->GetOpDesc();
  ge::AttrUtils::SetInt(op_desc, "_fe_imply_type", static_cast<int>(EN_IMPL_HW_TBE));
  size_t in_num = node->GetOpDesc()->GetAllInputsSize();
  size_t out_num = node->GetOpDesc()->GetOutputsSize();
  std::vector<uint32_t> input_type_list = {0, 2, 1, 1};
  (void)ge::AttrUtils::SetListInt(op_desc, kInputParaTypeList, input_type_list);
  std::vector<std::string> input_name_list = {"x0", "y", "z0", "h0"};
  (void)ge::AttrUtils::SetListStr(op_desc, kInputNameList, input_name_list);
  std::vector<uint32_t> output_type_list = {0, 1, 1};
  (void)ge::AttrUtils::SetListInt(op_desc, kOutputParaTypeList, output_type_list);
  std::vector<std::string> output_name_list = {"o0", "opt_x0", "opt_y0"};
  (void)ge::AttrUtils::SetListStr(op_desc, kOutputNameList, output_name_list);

  (void)ge::AttrUtils::SetStr(op_desc, fe::kAttrOptionalInputMode, fe::kGenPlaceholder);

  auto pure_op_desc = node->GetOpDescBarePtr();
  pure_op_desc->AppendIrInput("x0", ge::kIrInputRequired);
  pure_op_desc->AppendIrInput("y", ge::kIrInputDynamic);
  pure_op_desc->AppendIrInput("z0", ge::kIrInputOptional);
  pure_op_desc->AppendIrInput("h0", ge::kIrInputOptional);
  pure_op_desc->AppendIrOutput("o0", ge::kIrOutputRequired);
  pure_op_desc->AppendIrOutput("opt_x0", ge::kIrOutputRequired);
  pure_op_desc->AppendIrOutput("opt_y0", ge::kIrOutputRequired);
  (void)ge::AttrUtils::SetInt(op_desc, kOpKernelAllInputSize, 4);

  ArgsFormatConstructor args_construct(op_desc, false);
  auto ret = args_construct.ConstructNodeArgsDesc();
  EXPECT_EQ(ret, fe::SUCCESS);
  auto args_format_str = args_construct.GetArgsFormatString();
  EXPECT_STREQ(args_format_str.c_str(), "{i0*}{i1*}{i2*}{i3*}{o0*}{o1*}{o2*}");

  // input err roolback test: kInputParaTypeList != kInputNameList
  (void)ge::AttrUtils::SetListInt(op_desc, kInputParaTypeList, {0});
  ArgsFormatConstructor args_construct2(op_desc, false);
  ret = args_construct2.ConstructNodeArgsDesc();
  EXPECT_EQ(ret, fe::SUCCESS);
  (void)ge::AttrUtils::SetListInt(op_desc, kInputParaTypeList, input_type_list);

  // tbe always empty option output
  auto output_desc_ptr = op_desc->MutableOutputDesc(1);
  (void)ge::AttrUtils::SetInt(output_desc_ptr, ge::ATTR_NAME_MEMORY_SIZE_CALC_TYPE,
      static_cast<int64_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY));
  ArgsFormatConstructor args_construct1(op_desc, false);
  ret = args_construct1.ConstructNodeArgsDesc();
  EXPECT_EQ(ret, fe::SUCCESS);
  args_format_str = args_construct1.GetArgsFormatString();
  EXPECT_STREQ(args_format_str.c_str(), "{i0*}{i1*}{i2*}{i3*}{o0*}{o2*}");

  // dynamic in num in ir over total num in graph
  auto &name_index = pure_op_desc->MutableAllInputName();
  name_index.clear();
  name_index["x0"] = 0;
  name_index["y0"] = 1;
  name_index["y1"] = 2;
  name_index["y2"] = 3;
  name_index["y3"] = 4;
  name_index["y4"] = 5;
  name_index["y5"] = 6;
  name_index["y6"] = 7;
  name_index["y7"] = 8;
  name_index["z0"] = 9;
  name_index["h0"] = 10;
  ArgsFormatConstructor args_construct3(op_desc, false);
  ret = args_construct3.ConstructNodeArgsDesc();
  EXPECT_EQ(ret, fe::SUCCESS);
  args_format_str = args_construct3.GetArgsFormatString();
  EXPECT_STREQ(args_format_str.c_str(), "{i_instance0*}{i_instance1*}{i_instance2*}{i_instance3*}{o_instance0*}{o_instance1*}");
}

TEST_F(UTEST_ArgsFormat, MC2_opt_output_place_test) {
  FeTestOpDescBuilder builder;
  builder.SetName("test_tvm");
  builder.SetType("MatmulAllReduce");
  builder.SetInputs({1, 2, 3, 4});
  builder.SetOutputs({1});
  builder.AddInputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "x0");
  builder.AddInputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "y0");
  builder.AddInputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "z0");
  builder.AddInputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "h0");
  builder.AddOutputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "o0");
  builder.AddOutputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "opt_x0");
  builder.AddOutputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "opt_y0");
  auto node = builder.Finish();
  auto op_desc = node->GetOpDesc();
  ge::AttrUtils::SetInt(op_desc, "_fe_imply_type", static_cast<int>(EN_IMPL_HW_TBE));
  std::vector<uint32_t> input_type_list = {0, 2, 1, 1};
  (void)ge::AttrUtils::SetListInt(op_desc, kInputParaTypeList, input_type_list);
  std::vector<std::string> input_name_list = {"x0", "y", "z0", "h0"};
  (void)ge::AttrUtils::SetListStr(op_desc, kInputNameList, input_name_list);
  std::vector<uint32_t> output_type_list = {0, 1, 1};
  (void)ge::AttrUtils::SetListInt(op_desc, kOutputParaTypeList, output_type_list);
  std::vector<std::string> output_name_list = {"o0", "opt_x0", "opt_y0"};
  (void)ge::AttrUtils::SetListStr(op_desc, kOutputNameList, output_name_list);

  (void)ge::AttrUtils::SetStr(op_desc, fe::kAttrOptionalInputMode, fe::kGenPlaceholder);

  auto pure_op_desc = node->GetOpDescBarePtr();
  pure_op_desc->AppendIrInput("x0", ge::kIrInputRequired);
  pure_op_desc->AppendIrInput("y", ge::kIrInputDynamic);
  pure_op_desc->AppendIrInput("z0", ge::kIrInputOptional);
  pure_op_desc->AppendIrInput("h0", ge::kIrInputOptional);
  pure_op_desc->AppendIrOutput("o0", ge::kIrOutputRequired);
  pure_op_desc->AppendIrOutput("opt_x0", ge::kIrOutputRequired);
  pure_op_desc->AppendIrOutput("opt_y0", ge::kIrOutputRequired);
  (void)ge::AttrUtils::SetInt(op_desc, kOpKernelAllInputSize, 4);

  ArgsFormatConstructor args_construct(op_desc, false);
  auto ret = args_construct.ConstructNodeArgsDesc();
  EXPECT_EQ(ret, fe::SUCCESS);
  auto args_format_str = args_construct.GetArgsFormatString();
  EXPECT_STREQ(args_format_str.c_str(), "{i0*}{i1*}{i2*}{i3*}{o0*}{o1*}{o2*}");

  // tbe always empty option output
  (void)ge::AttrUtils::SetStr(op_desc, fe::kAttrOptionalOutputMode, fe::kGenPlaceholder);
  auto output_desc_ptr = op_desc->MutableOutputDesc(1);
  (void)ge::AttrUtils::SetInt(output_desc_ptr, ge::ATTR_NAME_MEMORY_SIZE_CALC_TYPE,
      static_cast<int64_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY));
  ArgsFormatConstructor args_construct1(op_desc, false);
  ret = args_construct1.ConstructNodeArgsDesc();
  EXPECT_EQ(ret, fe::SUCCESS);
  args_format_str = args_construct1.GetArgsFormatString();
  EXPECT_STREQ(args_format_str.c_str(), "{i0*}{i1*}{i2*}{i3*}{o0*}{}{o2*}");
}

TEST_F(UTEST_ArgsFormat, MC2_opt_output_place_test2) {
  FeTestOpDescBuilder builder;
  builder.SetName("test_tvm");
  builder.SetType("MatmulAllReduce");
  builder.SetInputs({1, 2, 3, 4});
  builder.SetOutputs({1});
  builder.AddInputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "x0");
  builder.AddInputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "y0");
  builder.AddInputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "z0");
  builder.AddInputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "h0");
  builder.AddOutputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "o0");
  builder.AddOutputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "opt_x0");
  builder.AddOutputDesc({1, 1, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, "opt_y0");
  auto node = builder.Finish();
  auto op_desc = node->GetOpDesc();
  ge::AttrUtils::SetInt(op_desc, "_fe_imply_type", static_cast<int>(EN_IMPL_HW_TBE));
  std::vector<uint32_t> input_type_list = {0, 2, 1, 1};
  (void)ge::AttrUtils::SetListInt(op_desc, kInputParaTypeList, input_type_list);
  std::vector<std::string> input_name_list = {"x0", "y", "z0", "h0"};
  (void)ge::AttrUtils::SetListStr(op_desc, kInputNameList, input_name_list);
  std::vector<uint32_t> output_type_list = {0, 1, 1};
  (void)ge::AttrUtils::SetListInt(op_desc, kOutputParaTypeList, output_type_list);
  std::vector<std::string> output_name_list = {"o0", "opt_x0", "opt_y0"};
  (void)ge::AttrUtils::SetListStr(op_desc, kOutputNameList, output_name_list);

  (void)ge::AttrUtils::SetStr(op_desc, fe::kAttrOptionalInputMode, fe::kGenPlaceholder);

  auto pure_op_desc = node->GetOpDescBarePtr();
  pure_op_desc->AppendIrInput("x0", ge::kIrInputRequired);
  pure_op_desc->AppendIrInput("y", ge::kIrInputDynamic);
  pure_op_desc->AppendIrInput("z0", ge::kIrInputOptional);
  pure_op_desc->AppendIrInput("h0", ge::kIrInputOptional);
  pure_op_desc->AppendIrOutput("o0", ge::kIrOutputRequired);
  pure_op_desc->AppendIrOutput("opt_x0", ge::kIrOutputRequired);
  pure_op_desc->AppendIrOutput("opt_y0", ge::kIrOutputRequired);
  (void)ge::AttrUtils::SetInt(op_desc, kOpKernelAllInputSize, 4);

  ArgsFormatConstructor args_construct(op_desc, false);
  auto ret = args_construct.ConstructNodeArgsDesc();
  EXPECT_EQ(ret, fe::SUCCESS);
  auto args_format_str = args_construct.GetArgsFormatString();
  EXPECT_STREQ(args_format_str.c_str(), "{i0*}{i1*}{i2*}{i3*}{o0*}{o1*}{o2*}");

  // tbe always empty option output
  (void)ge::AttrUtils::SetStr(op_desc, fe::kAttrOptionalOutputMode, fe::kGenPlaceholder);
  auto output_desc_ptr = op_desc->MutableOutputDesc(1);
  (void)ge::AttrUtils::SetInt(output_desc_ptr, ge::ATTR_NAME_MEMORY_SIZE_CALC_TYPE,
  static_cast<int64_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY));
  ArgsFormatConstructor args_construct1(op_desc, false);
  ret = args_construct1.ConstructNodeArgsDesc();
  EXPECT_EQ(ret, fe::SUCCESS);
  args_format_str = args_construct1.GetArgsFormatString();
  EXPECT_STREQ(args_format_str.c_str(), "{i0*}{i1*}{i2*}{i3*}{o0*}{}{o2*}");
}
