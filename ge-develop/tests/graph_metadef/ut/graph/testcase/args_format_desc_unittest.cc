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
#include <memory>
#include <string>

#include "ge_ir.pb.h"
#include "graph/args_format_desc.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/normal_graph/ge_tensor_impl.h"
#include "graph/tensor.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "register/hidden_inputs_func_registry.h"
#include "graph/operator_factory.h"
#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "common/graph_builder_utils.h"

using namespace std;
using namespace ge;
namespace ge {
class UtestArgsFormatDesc : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestArgsFormatDesc, serialize_simple_args) {
  auto op_desc = std::make_shared<OpDesc>("tmp_op", "Mul");
  ArgsFormatDesc desc;
  desc.Append(AddrType::FFTS_ADDR);
  desc.Append(AddrType::OVERFLOW_ADDR);
  desc.Append(AddrType::TILING);
  desc.Append(AddrType::TILING_FFTS, 0);
  desc.Append(AddrType::TILING_FFTS, 1);
  std::string res = desc.ToString();
  std::string expect_res = "{ffts_addr}{overflow_addr}{t}{t_ffts.non_tail}{t_ffts.tail}";
  EXPECT_EQ(expect_res, res);
  size_t args_size{0UL};
  EXPECT_EQ(desc.GetArgsSize(op_desc, args_size), ge::GRAPH_SUCCESS);
  EXPECT_EQ(args_size, 40UL);
  std::vector<ArgDesc> descs;
  EXPECT_EQ(ArgsFormatDesc::Parse(op_desc, expect_res, descs), SUCCESS);
}

TEST_F(UtestArgsFormatDesc, serialize_simple_args1) {
  auto op_desc = std::make_shared<OpDesc>("tmp_op", "Mul");
  GeTensorDesc desc;
  op_desc->AddOutputDesc(desc);
  op_desc->AddOutputDesc(desc);
  op_desc->AddOutputDesc(desc);
  op_desc->AddOutputDesc(desc);
  op_desc->AddOutputDesc(desc);
  op_desc->AddInputDesc(desc);
  op_desc->AddInputDesc(desc);
  op_desc->AddInputDesc(desc);
  op_desc->AddInputDesc(desc);
  op_desc->AddInputDesc(desc);
  op_desc->AddInputDesc(desc);

  std::string expect_res =
      "{i_instance0}{i_instance0*}{i_instance2*}{o_instance3}{o_instance1}{o_instance3*}{i_instance2}{o_instance4}";
  std::vector<ArgDesc> descs;
  EXPECT_EQ(ArgsFormatDesc::Parse(op_desc, expect_res, descs), SUCCESS);
  ArgsFormatDesc format_desc;
  format_desc.Append(AddrType::INPUT_INSTANCE, 0, true);
  format_desc.Append(AddrType::INPUT_INSTANCE, 0);
  format_desc.Append(AddrType::INPUT_INSTANCE, 2);
  format_desc.Append(AddrType::OUTPUT_INSTANCE, 3, true);
  format_desc.Append(AddrType::OUTPUT_INSTANCE, 1, true);
  format_desc.Append(AddrType::OUTPUT_INSTANCE, 3);
  format_desc.Append(AddrType::INPUT_INSTANCE, 2, true);
  format_desc.Append(AddrType::OUTPUT_INSTANCE, 4, true);
  std::string res = format_desc.ToString();
  EXPECT_EQ(res, expect_res);
  std::size_t arg_size{0UL};
  EXPECT_EQ(format_desc.GetArgsSize(op_desc, arg_size), SUCCESS);
  EXPECT_EQ(arg_size, 104);

  std::string expect_res1 = "{i_instance*}{o_instance*}";
  std::vector<ArgDesc> descs1;
  EXPECT_EQ(ArgsFormatDesc::Parse(op_desc, expect_res1, descs1), SUCCESS);
  EXPECT_EQ(descs1.size(), 11UL);
}

REG_OP(SimpleOP)
    .INPUT(a, TensorType(DT_FLOAT32))
    .INPUT(b, TensorType(DT_FLOAT32))
    .INPUT(c, TensorType(DT_FLOAT32))
    .OUTPUT(x, TensorType(DT_FLOAT32))
    .OUTPUT(y, TensorType(DT_FLOAT32))
    .OUTPUT(z, TensorType(DT_FLOAT32))
    .OP_END_FACTORY_REG(SimpleOP);

TEST_F(UtestArgsFormatDesc, common_args) {
  auto op = OperatorFactory::CreateOperator("test1", "SimpleOP");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 40, 1024, 256});
  GeTensorDesc desc(shape);
  op_desc->AddInputDesc(desc);
  op_desc->AddInputDesc(desc);
  op_desc->AddInputDesc(desc);

  op_desc->AddOutputDesc(desc);
  op_desc->AddOutputDesc(desc);
  op_desc->AddOutputDesc(desc);

  ArgsFormatDesc args_des;
  args_des.Append(AddrType::INPUT, -1);
  args_des.Append(AddrType::OUTPUT, -1);
  args_des.Append(AddrType::WORKSPACE, -1);
  std::string res = args_des.ToString();
  std::string expect_res = "{i*}{o*}{ws*}";
  EXPECT_EQ(expect_res, res);

  size_t args_size{0UL};
  EXPECT_EQ(args_des.GetArgsSize(op_desc, args_size), ge::GRAPH_SUCCESS);
  EXPECT_EQ(args_size, 176UL);
  std::vector<ArgDesc> descs;
  EXPECT_EQ(ArgsFormatDesc::Parse(op_desc, expect_res, descs), SUCCESS);
  EXPECT_EQ(descs.size(), 7UL);
  EXPECT_EQ(descs[2].addr_type, AddrType::INPUT);
  EXPECT_EQ(descs[2].ir_idx, 2);
}

REG_OP(DynOP)
    .INPUT(a, TensorType(DT_FLOAT32))
    .DYNAMIC_INPUT(b, TensorType(DT_FLOAT32))
    .INPUT(c, TensorType(DT_FLOAT32))
    .OUTPUT(x, TensorType(DT_FLOAT32))
    .DYNAMIC_OUTPUT(y, TensorType(DT_FLOAT32))
    .OUTPUT(z, TensorType(DT_FLOAT32))
    .OP_END_FACTORY_REG(DynOP);

TEST_F(UtestArgsFormatDesc, common_args_dynamic_folded) {
  auto op = OperatorFactory::CreateOperator("test1", "DynOP");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 40, 1024, 256});
  GeTensorDesc desc(shape);
  op_desc->UpdateInputDesc(0, desc);
  op_desc->AddDynamicInputDescByIndex("b", 1, 1);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->UpdateOutputDesc(0, desc);
  op_desc->AddDynamicOutputDesc("y", 1, true);
  op_desc->UpdateOutputDesc("y0", desc);
  op_desc->UpdateOutputDesc(0, desc);

  ArgsFormatDesc args_des;
  args_des.Append(AddrType::INPUT, -1);
  args_des.Append(AddrType::OUTPUT, -1);
  args_des.Append(AddrType::WORKSPACE, 0);
  args_des.Append(AddrType::WORKSPACE, 1);
  std::string res = args_des.ToString();
  std::string expect_res = "{i*}{o*}{ws0*}{ws1*}";
  EXPECT_EQ(expect_res, res);

  size_t args_size{0UL};
  EXPECT_EQ(args_des.GetArgsSize(op_desc, args_size), ge::GRAPH_SUCCESS);
  EXPECT_EQ(args_size, 80UL);
  std::vector<ArgDesc> descs;
  EXPECT_EQ(ArgsFormatDesc::Parse(op_desc, expect_res, descs), SUCCESS);

  std::string expanded_res = "{i0}{i1}{i2}{o0}{o1}{o2}{ws0}{ws1}";
  std::vector<ArgDesc> expanded_descs;
  EXPECT_EQ(ArgsFormatDesc::Parse(op_desc, expanded_res, expanded_descs), SUCCESS);
  EXPECT_EQ(descs.size(), expanded_descs.size());
  for (auto i = 0UL; i < descs.size(); ++i) {
    EXPECT_EQ(descs[i].addr_type, expanded_descs[i].addr_type);
    EXPECT_EQ(descs[i].ir_idx, expanded_descs[i].ir_idx);
    EXPECT_EQ(descs[i].folded, expanded_descs[i].folded);
  }
}

TEST_F(UtestArgsFormatDesc, common_args_size_equal) {
  auto op = OperatorFactory::CreateOperator("test1", "DynOP");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 40, 1024, 256});
  GeTensorDesc desc(shape);
  op_desc->UpdateInputDesc(0, desc);
  op_desc->AddDynamicInputDescByIndex("b", 1, 1);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->UpdateOutputDesc(0, desc);
  op_desc->AddDynamicOutputDesc("y", 1, true);
  op_desc->UpdateOutputDesc("y0", desc);
  op_desc->UpdateOutputDesc(0, desc);

  ArgsFormatDesc args_des;
  args_des.Append(AddrType::INPUT, -1);
  args_des.Append(AddrType::OUTPUT, -1);
  args_des.Append(AddrType::WORKSPACE, 0);
  args_des.Append(AddrType::WORKSPACE, 1);
  size_t args_size{0UL};
  EXPECT_EQ(args_des.GetArgsSize(op_desc, args_size), ge::GRAPH_SUCCESS);
  ArgsFormatDesc args_des1;
  args_des1.Append(AddrType::INPUT, 0);
  args_des1.Append(AddrType::INPUT, 1, true);
  args_des1.Append(AddrType::INPUT, 2);
  args_des1.Append(AddrType::OUTPUT, 0);
  args_des1.Append(AddrType::OUTPUT, 1, true);
  args_des1.Append(AddrType::OUTPUT, 2);
  args_des1.Append(AddrType::WORKSPACE, 0);
  args_des1.Append(AddrType::WORKSPACE, 1);
  size_t args_size1{0UL};
  EXPECT_EQ(args_des1.GetArgsSize(op_desc, args_size1), ge::GRAPH_SUCCESS);
  EXPECT_EQ(args_size, args_size1);
}

REG_OP(IFA)
    .INPUT(query, TensorType(DT_FLOAT32))
    .DYNAMIC_INPUT(key, TensorType(DT_FLOAT32))
    .DYNAMIC_INPUT(value, TensorType(DT_FLOAT32))
    .OPTIONAL_INPUT(padding_mask, TensorType(DT_FLOAT32))
    .OPTIONAL_INPUT(atten_mask, TensorType(DT_FLOAT32))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType(DT_FLOAT32))
    .DYNAMIC_OUTPUT(attention_out, TensorType(DT_FLOAT32))
    .OP_END_FACTORY_REG(IFA);

TEST_F(UtestArgsFormatDesc, serialize_dynamic_args) {
  auto op = OperatorFactory::CreateOperator("test1", "IFA");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 40, 1024, 256});
  GeTensorDesc desc(shape);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape);
  op_desc->UpdateInputDesc(0, desc);
  op_desc->AddDynamicInputDescByIndex("key", 2, 1);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->AddDynamicInputDescByIndex("value", 2, 3);
  op_desc->UpdateInputDesc(3, desc);
  op_desc->UpdateInputDesc(4, scalar_desc);
  op_desc->UpdateInputDesc("atten_mask", desc);

  op_desc->AddDynamicOutputDesc("attention_out", 2, true);
  op_desc->UpdateOutputDesc("attention_out0", desc);
  op_desc->UpdateOutputDesc("attention_out1", scalar_desc);

  ArgsFormatDesc args_des;
  args_des.Append(AddrType::FFTS_ADDR);
  args_des.Append(AddrType::INPUT, 0);
  args_des.Append(AddrType::INPUT_DESC, 1, true);
  args_des.Append(AddrType::INPUT_DESC, 2, true);
  args_des.Append(AddrType::INPUT, 4);
  args_des.Append(AddrType::OUTPUT_DESC, 0, true);
  args_des.Append(AddrType::WORKSPACE, 0);
  args_des.Append(AddrType::TILING_FFTS, 1);
  std::string res = args_des.ToString();
  std::string expect_res = "{ffts_addr}{i0*}{i_desc1}{i_desc2}{i4*}{o_desc0}{ws0*}{t_ffts.tail}";
  EXPECT_EQ(expect_res, res);

  size_t args_size{0UL};
  EXPECT_EQ(args_des.GetArgsSize(op_desc, args_size), ge::GRAPH_SUCCESS);
  EXPECT_EQ(args_size, 328UL);
  std::vector<ArgDesc> descs;
  EXPECT_EQ(ArgsFormatDesc::Parse(op_desc, expect_res, descs), SUCCESS);
  EXPECT_EQ(descs.size(), 8UL);
  EXPECT_EQ(descs[2].addr_type, AddrType::INPUT_DESC);
}

TEST_F(UtestArgsFormatDesc, serialize_hidden_input) {
  auto op_desc = std::make_shared<OpDesc>("tmp_op", "Mul");
  ArgsFormatDesc desc;
  ArgsFormatDescUtils::InsertHiddenInputs(desc.arg_descs_, -1, HiddenInputsType::HCOM);
  desc.Append(AddrType::PLACEHOLDER);
  EXPECT_EQ(desc.ToString(), "{hi.hcom0*}{}");
  size_t args_size{0UL};
  EXPECT_EQ(desc.GetArgsSize(op_desc, args_size), ge::GRAPH_SUCCESS);
  EXPECT_EQ(args_size, 16UL);
}

TEST_F(UtestArgsFormatDesc, serialize_ascendcpp_hidden_input) {
  auto op_desc = std::make_shared<OpDesc>("tmp_op", "Mul");
  ArgsFormatDesc desc;
  ArgsFormatDescUtils::InsertHiddenInputs(desc.arg_descs_, -1, HiddenInputsType::TILEFWK);
  desc.Append(AddrType::PLACEHOLDER);
  EXPECT_EQ(desc.ToString(), "{hi.tilefwk0*}{}");
  size_t args_size{0UL};
  EXPECT_EQ(desc.GetArgsSize(op_desc, args_size), ge::GRAPH_SUCCESS);
  EXPECT_EQ(args_size, 16UL);
}

TEST_F(UtestArgsFormatDesc, deserialzie_hidden_input) {
  auto op_desc = std::make_shared<OpDesc>("tmp_op", "Mul");
  std::vector<ArgDesc> descs;
  EXPECT_NE(ArgsFormatDesc::Parse(op_desc, "{hi.unsupported}", descs), SUCCESS);
  EXPECT_NE(ArgsFormatDesc::Parse(op_desc, "{hi.hcom}", descs), SUCCESS);
  EXPECT_NE(ArgsFormatDesc::Parse(op_desc, "{hi.hcom[xx]}", descs), SUCCESS);

  EXPECT_EQ(ArgsFormatDesc::Parse(op_desc, "{hi.hcom0*}", descs), SUCCESS);
  EXPECT_EQ(descs.size(), 1UL);
  EXPECT_EQ(descs[0UL].addr_type, AddrType::HIDDEN_INPUT);
  EXPECT_EQ(descs[0UL].ir_idx, 0);

  EXPECT_EQ(ArgsFormatDesc::Parse(op_desc, "{hi.tilefwk0*}", descs), SUCCESS);
  EXPECT_EQ(descs.size(), 1UL);
  EXPECT_EQ(descs[0UL].addr_type, AddrType::HIDDEN_INPUT);
  EXPECT_EQ(descs[0UL].ir_idx, 0);
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(descs[0UL].reserved), static_cast<uint32_t>(HiddenInputsType::TILEFWK));

  EXPECT_EQ(ArgsFormatDesc::Parse(op_desc, "{hi.hcom0*}{hi.hcom1*}", descs), SUCCESS);
  EXPECT_EQ(descs.size(), 2UL);
  EXPECT_EQ(descs[0UL].addr_type, AddrType::HIDDEN_INPUT);
  EXPECT_EQ(descs[0UL].ir_idx, 0);
  EXPECT_EQ(descs[1UL].addr_type, AddrType::HIDDEN_INPUT);
  EXPECT_EQ(descs[1UL].ir_idx, 1);
}

TEST_F(UtestArgsFormatDesc, serialize_custom_val) {
  auto op_desc = std::make_shared<OpDesc>("tmp_op", "Mul");
  ArgsFormatDesc desc;
  size_t args_size = 0;

  EXPECT_EQ(ArgsFormatDescUtils::InsertCustomValue(desc.arg_descs_, -1, 0), GRAPH_SUCCESS);
  EXPECT_EQ(desc.ToString(), "{#0}");
  EXPECT_EQ(desc.GetArgsSize(op_desc, args_size), ge::GRAPH_SUCCESS);
  EXPECT_EQ(args_size, 8UL);
}

TEST_F(UtestArgsFormatDesc, deserialzie_custom_val) {
  auto op_desc = std::make_shared<OpDesc>("tmp_op", "Mul");
  std::vector<ArgDesc> descs;
  EXPECT_NE(ArgsFormatDesc::Parse(op_desc, "{xxx}", descs), SUCCESS);
  EXPECT_NE(ArgsFormatDesc::Parse(op_desc, "{#}", descs), SUCCESS);

  EXPECT_EQ(ArgsFormatDesc::Parse(op_desc, "{#18446744073709551615}", descs), SUCCESS); // 0xFFFFFFFFFFFFFFFF
  EXPECT_EQ(descs[0UL].addr_type, AddrType::CUSTOM_VALUE);
  EXPECT_EQ(*(uint64_t *)descs[0UL].reserved, 0xFFFFFFFFFFFFFFFF);

  EXPECT_EQ(ArgsFormatDesc::Parse(op_desc, "{#0}", descs), SUCCESS);
  EXPECT_EQ(descs[0UL].addr_type, AddrType::CUSTOM_VALUE);
  EXPECT_EQ(*(uint64_t *)descs[0UL].reserved, 0);
}

TEST_F(UtestArgsFormatDesc, deserialzie_placeholder) {
  auto op_desc = std::make_shared<OpDesc>("tmp_op", "Mul");
  std::string format1 = "{}";
  std::vector<ArgDesc> descs;
  EXPECT_EQ(ArgsFormatDesc::Parse(op_desc, format1, descs), SUCCESS);
  EXPECT_EQ(descs.size(), 1UL);
  EXPECT_EQ(descs[0UL].addr_type, AddrType::PLACEHOLDER);
}

TEST_F(UtestArgsFormatDesc, placeholder_serdes) {
  auto op_desc = std::make_shared<OpDesc>("op", "Dummy");
  ArgsFormatDesc desc, desc2;
  size_t size = 0;
  desc.Append(AddrType::PLACEHOLDER);
  desc.AppendPlaceholder();
  desc.AppendPlaceholder(ArgsFormatWidth::BIT32);
  desc.AppendPlaceholder(ArgsFormatWidth::BIT64);
  EXPECT_EQ(desc.GetArgsSize(op_desc, size), GRAPH_SUCCESS);
  EXPECT_EQ(size, 8 + 8 + 4 + 8);
  auto str = desc.ToString();
  EXPECT_EQ(str, "{}{}{.32b}{}");

  EXPECT_EQ(ArgsFormatDesc::FromString(desc2, op_desc, str), GRAPH_SUCCESS);
  size_t idx = 0;
  for (const auto &iter : desc) {
    const auto &iter2 = desc2.arg_descs_[idx++];
    EXPECT_EQ(iter.addr_type, AddrType::PLACEHOLDER);
    EXPECT_EQ(iter2.addr_type, AddrType::PLACEHOLDER);
    EXPECT_EQ(iter.ir_idx, iter2.ir_idx);
  }
}

TEST_F(UtestArgsFormatDesc, custom_value_serdes) {
  auto op_desc = std::make_shared<OpDesc>("op", "Dummy");
  ArgsFormatDesc desc, desc2;
  size_t size = 0;
  desc.AppendCustomValue(42);
  desc.AppendCustomValue(114, ArgsFormatWidth::BIT32);
  desc.AppendCustomValue(514, ArgsFormatWidth::BIT64);
  EXPECT_EQ(desc.GetArgsSize(op_desc, size), GRAPH_SUCCESS);
  EXPECT_EQ(size, 8 + 4 + 8);
  auto str = desc.ToString();
  EXPECT_EQ(str, "{#42}{#.32b114}{#514}");

  EXPECT_EQ(ArgsFormatDesc::FromString(desc2, op_desc, str), GRAPH_SUCCESS);
  size_t idx = 0;
  for (const auto &iter : desc) {
    const auto &iter2 = desc2.arg_descs_[idx++];
    EXPECT_EQ(iter.addr_type, AddrType::CUSTOM_VALUE);
    EXPECT_EQ(iter2.addr_type, AddrType::CUSTOM_VALUE);
    EXPECT_EQ(iter.ir_idx, iter2.ir_idx);
    EXPECT_EQ(*reinterpret_cast<const uint64_t *>(iter.reserved),
              *reinterpret_cast<const uint64_t *>(iter2.reserved));
  }
}

TEST_F(UtestArgsFormatDesc, invalid_args_format_width) {
  auto op_desc = std::make_shared<OpDesc>("op", "Dummy");
  auto invalid_width = static_cast<ArgsFormatWidth>(0);
  ArgsFormatDesc desc;
  size_t size = 0;
  desc.AppendPlaceholder(invalid_width);
  EXPECT_NE(desc.GetArgsSize(op_desc, size), GRAPH_SUCCESS);

  desc.Clear();
  desc.AppendCustomValue(42, invalid_width);
  EXPECT_NE(desc.GetArgsSize(op_desc, size), GRAPH_SUCCESS);
}

TEST_F(UtestArgsFormatDesc, deserialzie_unsupported) {
auto op_desc = std::make_shared<OpDesc>("tmp_op", "Mul");
  std::string format1 = "{hehe}";
  std::vector<ArgDesc> descs1;
  EXPECT_NE(ArgsFormatDesc::Parse(op_desc, format1, descs1), SUCCESS);

  std::string format2 = "{ }";
  std::vector<ArgDesc> descs2;
  EXPECT_NE(ArgsFormatDesc::Parse(op_desc, format2, descs2), SUCCESS);

  std::string format3 = "{hi.unsupported}";
  std::vector<ArgDesc> descs3;
  EXPECT_NE(ArgsFormatDesc::Parse(op_desc, format3, descs3), SUCCESS);
}

TEST_F(UtestArgsFormatDesc, deserialzie_tiling_context) {
  auto op_desc = std::make_shared<OpDesc>("tmp_op", "Mul");
  std::string format1 =
      "{tiling_context}{*op_type}{tiling_context.tiling_key}{tiling_context.tiling_data}{tiling_context.block_dim}";
  std::vector<ArgDesc> descs;
  EXPECT_EQ(ArgsFormatDesc::Parse(op_desc, format1, descs), SUCCESS);
  EXPECT_EQ(descs.size(), 5UL);
  EXPECT_EQ(descs[0UL].addr_type, AddrType::TILING_CONTEXT);
  EXPECT_EQ(descs[1UL].addr_type, AddrType::OP_TYPE);
  EXPECT_EQ(descs[2UL].ir_idx, static_cast<int32_t>(TilingContextSubType::TILING_KEY));
  EXPECT_EQ(descs[3UL].ir_idx, static_cast<int32_t>(TilingContextSubType::TILING_DATA));
  EXPECT_EQ(descs[4UL].ir_idx, static_cast<int32_t>(TilingContextSubType::BLOCK_DIM));
}

TEST_F(UtestArgsFormatDesc, serialzie_tiling_context) {
  auto op_desc = std::make_shared<OpDesc>("tmp_op", "Mul");
  std::string format1 =
      "{tiling_context}{*op_type}{tiling_context.tiling_key}{tiling_context.tiling_data}{tiling_context.block_dim}";
  ArgsFormatDesc desc;
  desc.AppendTilingContext();
  desc.Append(AddrType::OP_TYPE);
  desc.AppendTilingContext(TilingContextSubType::TILING_KEY);
  desc.AppendTilingContext(TilingContextSubType::TILING_DATA);
  desc.AppendTilingContext(TilingContextSubType::BLOCK_DIM);
  size_t target_size{0UL};
  EXPECT_EQ(desc.GetArgsSize(op_desc, target_size), SUCCESS);
  EXPECT_EQ(target_size, 40UL);
  std::string res = desc.ToString();
  EXPECT_EQ(format1, res);
}

TEST_F(UtestArgsFormatDesc, single_arg_size_calc) {
  auto op = OperatorFactory::CreateOperator("test1", "IFA");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 40, 1024, 256});
  GeTensorDesc desc(shape);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape);
  op_desc->UpdateInputDesc(0, desc);
  op_desc->AddDynamicInputDescByIndex("key", 2, 1);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->AddDynamicInputDescByIndex("value", 2, 3);
  op_desc->UpdateInputDesc(3, desc);
  op_desc->UpdateInputDesc(4, scalar_desc);
  op_desc->UpdateInputDesc("atten_mask", desc);

  op_desc->AddDynamicOutputDesc("attention_out", 2, true);
  op_desc->UpdateOutputDesc("attention_out0", desc);
  op_desc->UpdateOutputDesc("attention_out1", scalar_desc);

  ArgsFormatDesc args_des;
  args_des.Append(AddrType::FFTS_ADDR);
  args_des.Append(AddrType::INPUT, 0);
  args_des.Append(AddrType::INPUT_DESC, 1, true);
  args_des.Append(AddrType::INPUT_DESC, 2, true);
  args_des.Append(AddrType::INPUT, 4);
  args_des.Append(AddrType::OUTPUT_DESC, 0, true);
  args_des.Append(AddrType::WORKSPACE, 0);
  args_des.Append(AddrType::TILING_FFTS, 1);
  args_des.Append(AddrType::HIDDEN_INPUT);
  std::string res = args_des.ToString();
  std::string expect_res = "{ffts_addr}{i0*}{i_desc1}{i_desc2}{i4*}{o_desc0}{ws0*}{t_ffts.tail}{hi.hcom0*}";
  EXPECT_EQ(expect_res, res);

  std::vector<ArgDesc> descs;
  EXPECT_EQ(ArgsFormatDesc::Parse(op_desc, expect_res, descs), SUCCESS);
  EXPECT_EQ(descs.size(), 9UL);

  size_t arg_size{0UL};
  EXPECT_EQ(args_des.GetArgSize(op_desc, descs[1], arg_size), ge::GRAPH_SUCCESS);
  EXPECT_EQ(arg_size, 8UL);
  arg_size = 0UL;
  EXPECT_EQ(args_des.GetArgSize(op_desc, descs[2], arg_size), ge::GRAPH_SUCCESS);
  EXPECT_EQ(arg_size, 112UL);
  arg_size = 0UL;
  EXPECT_EQ(args_des.GetArgSize(op_desc, descs[3], arg_size), ge::GRAPH_SUCCESS);
  EXPECT_EQ(arg_size, 88UL);
  arg_size = 0UL;
  EXPECT_EQ(args_des.GetArgSize(op_desc, descs[4], arg_size), ge::GRAPH_SUCCESS);
  EXPECT_EQ(arg_size, 8UL);
  arg_size = 0UL;
  EXPECT_EQ(args_des.GetArgSize(op_desc, descs[5], arg_size), ge::GRAPH_SUCCESS);
  EXPECT_EQ(arg_size, 88UL);

  ArgDesc unsupport_desc;
  unsupport_desc.addr_type = static_cast<AddrType>(0xff);
  EXPECT_EQ(args_des.GetArgSize(op_desc, unsupport_desc, arg_size), ge::GRAPH_PARAM_INVALID);

  args_des.Clear();
  EXPECT_EQ(args_des.ToString(), "");
}

/*
 *          netoutput
 *         |    \    \
 *       node4 node5 node6
 *       |      \
 *     node2  node3
 *      \    /
 *      node1
 */
ge::ComputeGraphPtr BuildNormalGraph(const std::string &name) {
  auto builder = ge::ut::GraphBuilder(name);
  auto node1 = builder.AddNode("node1", "node1", 0, 2);
  auto node2 = builder.AddNode("node2", "node2", 1, 1);
  auto node3 = builder.AddNode("node3", "node3", 1, 1);
  auto node4 = builder.AddNode("node4", "node4", 1, 1);
  auto node5 = builder.AddNode("node5", "node5", 1, 1);
  auto node6 = builder.AddNode("node6", "node6", 0, 1);
  auto netoutput = builder.AddNode("netoutput", "netoutput", 3, 1);

  node1->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  node2->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  node2->GetOpDesc()->AppendIrOutput("y", kIrOutputRequired);

  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node1, 1, node3, 0);
  builder.AddDataEdge(node2, 0, node4, 0);
  builder.AddDataEdge(node3, 0, node5, 0);
  builder.AddDataEdge(node4, 0, netoutput, 0);
  builder.AddDataEdge(node5, 0, netoutput, 1);
  builder.AddDataEdge(node6, 0, netoutput, 2);
  return builder.GetGraphWithoutSort();
}

TEST_F(UtestArgsFormatDesc, SknArgDescTest) {
  auto sub_graph = BuildNormalGraph("test");
  auto op_desc = std::make_shared<OpDesc>("sk", "SuperKernel");
  EXPECT_NE(sub_graph, nullptr);
  EXPECT_NE(op_desc, nullptr);
  op_desc->SetExtAttr("_sk_sub_graph", sub_graph);
  SkArgDesc sk_desc = {AddrType::SUPER_KERNEL_SUB_NODE, 1, false, AddrType::INPUT, 0};
  ArgDesc sub_desc = *reinterpret_cast<ArgDesc *>(&sk_desc);
  std::vector<ArgDesc> args_desc_vec;
  args_desc_vec.emplace_back(sub_desc);
  auto str = ArgsFormatDesc::Serialize(args_desc_vec);
  EXPECT_EQ(str, "{skn1i0*}");

  std::vector<ArgDesc> target_sub_desc_vec;
  auto ret = ArgsFormatDesc::Parse(nullptr, str, target_sub_desc_vec, false);
  EXPECT_NE(ret, GRAPH_SUCCESS);

  ret = ArgsFormatDesc::Parse(op_desc, str, target_sub_desc_vec, false);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(target_sub_desc_vec.size(), 1);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(target_sub_desc_vec.size(), 1);
  EXPECT_EQ(target_sub_desc_vec[0].ir_idx, sub_desc.ir_idx);
  EXPECT_EQ(target_sub_desc_vec[0].addr_type, sub_desc.addr_type);
  EXPECT_EQ(target_sub_desc_vec[0].folded, sub_desc.folded);

  EXPECT_EQ(reinterpret_cast<SkArgDesc *>(&target_sub_desc_vec[0])->sub_idx,
            reinterpret_cast<SkArgDesc *>(&sub_desc)->sub_idx);
  EXPECT_EQ(reinterpret_cast<SkArgDesc *>(&target_sub_desc_vec[0])->sub_addr_type,
            reinterpret_cast<SkArgDesc *>(&sub_desc)->sub_addr_type);

  ArgDesc arg_desc_normal{};
  arg_desc_normal.addr_type = AddrType::INPUT;
  ArgDesc tmp_arg_desc{};
  int32_t sub_op_id = 0;
  EXPECT_EQ(ArgsFormatDesc::ConvertArgDescSkToNormal(arg_desc_normal, tmp_arg_desc, sub_op_id), GRAPH_SUCCESS);
}

TEST_F(UtestArgsFormatDesc, SknArgDescTestHiddenInput) {
  auto sub_graph = BuildNormalGraph("test");
  auto op_desc = std::make_shared<OpDesc>("sk", "SuperKernel");
  EXPECT_NE(sub_graph, nullptr);
  EXPECT_NE(op_desc, nullptr);
  op_desc->SetExtAttr("_sk_sub_graph", sub_graph);
  SkArgDescV2 sk_desc = {AddrType::SUPER_KERNEL_SUB_NODE, 1,
                         static_cast<uint32_t>(HiddenInputsType::HCOM), AddrType::HIDDEN_INPUT, 0};
  auto sub_desc = *reinterpret_cast<ArgDesc *>(&sk_desc);
  std::vector<ArgDesc> args_desc_vec;
  args_desc_vec.emplace_back(*reinterpret_cast<ArgDesc *>(&sk_desc));

  sk_desc.reserved = static_cast<uint32_t>(HiddenInputsType::TILEFWK);
  args_desc_vec.emplace_back(*reinterpret_cast<ArgDesc *>(&sk_desc));
  sk_desc.reserved = static_cast<uint32_t>(HiddenInputsType::HCCLSUPERKERNEL);
  args_desc_vec.emplace_back(*reinterpret_cast<ArgDesc *>(&sk_desc));

  auto str = ArgsFormatDesc::Serialize(args_desc_vec);
  EXPECT_EQ(str, "{skn1hi.hcom0*}{skn1hi.tilefwk0*}{skn1hi.hcclsk0*}");

  std::vector<ArgDesc> target_sub_desc_vec;
  auto ret = ArgsFormatDesc::Parse(op_desc, str, target_sub_desc_vec, false);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(target_sub_desc_vec.size(), 3);
  EXPECT_EQ(target_sub_desc_vec[0].ir_idx, sub_desc.ir_idx);
  EXPECT_EQ(target_sub_desc_vec[0].addr_type, sub_desc.addr_type);
  EXPECT_EQ(target_sub_desc_vec[0].folded, sub_desc.folded);

  EXPECT_EQ(reinterpret_cast<SkArgDesc *>(&target_sub_desc_vec[0])->sub_idx,
            reinterpret_cast<SkArgDesc *>(&sub_desc)->sub_idx);
  EXPECT_EQ(reinterpret_cast<SkArgDesc *>(&target_sub_desc_vec[0])->sub_idx,
            0);
  EXPECT_EQ(reinterpret_cast<SkArgDesc *>(&target_sub_desc_vec[0])->sub_addr_type,
            reinterpret_cast<SkArgDesc *>(&sub_desc)->sub_addr_type);
  EXPECT_EQ(reinterpret_cast<SkArgDesc *>(&target_sub_desc_vec[0])->sub_addr_type,
            AddrType::HIDDEN_INPUT);
  EXPECT_EQ(reinterpret_cast<SkArgDescV2 *>(&target_sub_desc_vec[0])->reserved,
            static_cast<uint32_t>(HiddenInputsType::HCOM));

  EXPECT_EQ(reinterpret_cast<SkArgDesc *>(&target_sub_desc_vec[1])->sub_addr_type,
            AddrType::HIDDEN_INPUT);
  EXPECT_EQ(reinterpret_cast<SkArgDescV2 *>(&target_sub_desc_vec[1])->reserved,
            static_cast<uint32_t>(HiddenInputsType::TILEFWK));

  EXPECT_EQ(reinterpret_cast<SkArgDesc *>(&target_sub_desc_vec[2])->sub_addr_type,
            AddrType::HIDDEN_INPUT);
  EXPECT_EQ(reinterpret_cast<SkArgDescV2 *>(&target_sub_desc_vec[2])->reserved,
            static_cast<uint32_t>(HiddenInputsType::HCCLSUPERKERNEL));

  ArgDesc tmp_arg_desc{};
  int32_t sub_op_id = 0;
  EXPECT_EQ(ArgsFormatDesc::ConvertArgDescSkToNormal(target_sub_desc_vec[0], tmp_arg_desc, sub_op_id), GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int32_t *>(tmp_arg_desc.reserved), static_cast<int32_t>(HiddenInputsType::HCOM));

  EXPECT_EQ(ArgsFormatDesc::ConvertArgDescSkToNormal(target_sub_desc_vec[1], tmp_arg_desc, sub_op_id), GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int32_t *>(tmp_arg_desc.reserved), static_cast<int32_t>(HiddenInputsType::TILEFWK));

  EXPECT_EQ(ArgsFormatDesc::ConvertArgDescSkToNormal(target_sub_desc_vec[2], tmp_arg_desc, sub_op_id), GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int32_t *>(tmp_arg_desc.reserved), static_cast<int32_t>(HiddenInputsType::HCCLSUPERKERNEL));

  size_t arg_size = 0;
  ret = ArgsFormatDesc::GetArgSize(op_desc, sub_desc, arg_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(arg_size, 8);
}

TEST_F(UtestArgsFormatDesc, SknArgDesceEventAddr) {
  auto sub_graph = BuildNormalGraph("test");
  auto op_desc = std::make_shared<OpDesc>("sk", "SuperKernel");
  EXPECT_NE(sub_graph, nullptr);
  EXPECT_NE(op_desc, nullptr);
  op_desc->SetExtAttr("_sk_sub_graph", sub_graph);
  SkArgDesc sk_desc = {AddrType::SUPER_KERNEL_SUB_NODE, 1, false, AddrType::EVENT_ADDR, 10};
  ArgDesc sub_desc = *reinterpret_cast<ArgDesc *>(&sk_desc);
  std::vector<ArgDesc> args_desc_vec;
  args_desc_vec.emplace_back(sub_desc);
  auto str = ArgsFormatDesc::Serialize(args_desc_vec);

  EXPECT_EQ(str, "{skn1event_addr10*}");

  std::vector<ArgDesc> target_sub_desc_vec;
  auto ret = ArgsFormatDesc::Parse(op_desc, str, target_sub_desc_vec, false);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(target_sub_desc_vec.size(), 1);
  EXPECT_EQ(target_sub_desc_vec[0].ir_idx, sub_desc.ir_idx);
  EXPECT_EQ(target_sub_desc_vec[0].addr_type, sub_desc.addr_type);
  EXPECT_EQ(target_sub_desc_vec[0].folded, sub_desc.folded);

  EXPECT_EQ(reinterpret_cast<SkArgDesc *>(&target_sub_desc_vec[0])->sub_idx,
            reinterpret_cast<SkArgDesc *>(&sub_desc)->sub_idx);
  EXPECT_EQ(reinterpret_cast<SkArgDesc *>(&target_sub_desc_vec[0])->sub_idx,
            10);
  EXPECT_EQ(reinterpret_cast<SkArgDesc *>(&target_sub_desc_vec[0])->sub_addr_type,
            reinterpret_cast<SkArgDesc *>(&sub_desc)->sub_addr_type);
  EXPECT_EQ(reinterpret_cast<SkArgDesc *>(&target_sub_desc_vec[0])->sub_addr_type,
            AddrType::EVENT_ADDR);
  size_t arg_size = 0;
  ret = ArgsFormatDesc::GetArgSize(op_desc, sub_desc, arg_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(arg_size, 8);
}

TEST_F(UtestArgsFormatDesc, ConvertToSuperKernelArgFormat) {
  auto sub_graph = BuildNormalGraph("test");
  auto op_desc = std::make_shared<OpDesc>("sk", "SuperKernel");
  EXPECT_NE(sub_graph, nullptr);
  EXPECT_NE(op_desc, nullptr);
  op_desc->SetExtAttr("_sk_sub_graph", sub_graph);

  auto sk_node = std::shared_ptr<Node>(new (std::nothrow) Node(op_desc, nullptr));
  EXPECT_NE(sk_node, nullptr);
  (void)sk_node->Init();

  NodePtr sub_node = std::shared_ptr<Node>(new (std::nothrow) Node(op_desc, nullptr));;
  std::string sub_node_arg_format = "{i0*}";
  std::string sk_arg_format;
  sub_node->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  EXPECT_EQ(ArgsFormatDesc::ConvertToSuperKernelArgFormat(
                sk_node, sub_node, sub_node_arg_format, sk_arg_format), ge::GRAPH_SUCCESS);
  EXPECT_EQ(sk_arg_format, "{skn0i0*}");
}
}  // namespace ge
