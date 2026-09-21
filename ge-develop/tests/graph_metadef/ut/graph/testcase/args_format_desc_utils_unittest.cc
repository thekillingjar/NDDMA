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
#include "graph/utils/args_format_desc_utils.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/normal_graph/ge_tensor_impl.h"
#include "graph/tensor.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "register/hidden_inputs_func_registry.h"
#include "dlog_pub.h"
#include "graph/operator_factory.h"
#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"

using namespace std;
using namespace ge;
namespace ge {
class UtestArgsFormatDescUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestArgsFormatDescUtils, serialize_simple_args) {
  std::vector<ArgDesc> desc;
  ArgsFormatDescUtils::Append(desc, AddrType::FFTS_ADDR);
  ArgsFormatDescUtils::Append(desc, AddrType::OVERFLOW_ADDR);
  ArgsFormatDescUtils::Append(desc, AddrType::TILING);
  ArgsFormatDescUtils::Append(desc, AddrType::TILING_FFTS, 0);
  ArgsFormatDescUtils::Append(desc, AddrType::TILING_FFTS, 1);
  std::string res = ArgsFormatDescUtils::ToString(desc);
  std::string expect_res = "{ffts_addr}{overflow_addr}{t}{t_ffts.non_tail}{t_ffts.tail}";
  EXPECT_EQ(expect_res, res);

  std::vector<ArgDesc> descs;
  EXPECT_EQ(ArgsFormatDescUtils::Parse(expect_res, descs), SUCCESS);
  EXPECT_EQ(descs.size(), 5);
  EXPECT_EQ(desc[0].addr_type, AddrType::FFTS_ADDR);
  EXPECT_EQ(desc[1].addr_type, AddrType::OVERFLOW_ADDR);
  EXPECT_EQ(desc[2].addr_type, AddrType::TILING);
  EXPECT_EQ(desc[3].addr_type, AddrType::TILING_FFTS);
  EXPECT_EQ(desc[3].ir_idx, 0);
  EXPECT_EQ(desc[4].addr_type, AddrType::TILING_FFTS);
  EXPECT_EQ(desc[4].ir_idx, 1);
}

TEST_F(UtestArgsFormatDescUtils, common_args) {
  std::vector<ArgDesc> arg_desc;
  ArgsFormatDescUtils::Append(arg_desc, AddrType::INPUT, -1);
  ArgsFormatDescUtils::Append(arg_desc, AddrType::OUTPUT, -1);
  ArgsFormatDescUtils::Append(arg_desc, AddrType::WORKSPACE, -1);

  std::string res = ArgsFormatDescUtils::ToString(arg_desc);
  std::string expect_res = "{i*}{o*}{ws*}";
  EXPECT_EQ(expect_res, res);

  // 对外开放的接口，parser时不会处理动态输入的场景，因此：
  // i* -> [AddrType::INPUT, -1, false]
  // o* -> [AddrType::INPUT, -1, false]
  // ws* -> [AddrType::INPUT, -1, false]
  std::vector<ArgDesc> descs;
  EXPECT_EQ(ArgsFormatDescUtils::Parse(expect_res, descs), SUCCESS);
  EXPECT_EQ(descs.size(), 3UL);
  EXPECT_EQ(descs[0].addr_type, AddrType::INPUT);
  EXPECT_EQ(descs[0].ir_idx, -1);
  EXPECT_EQ(descs[0].folded, false);

  EXPECT_EQ(descs[1].addr_type, AddrType::OUTPUT);
  EXPECT_EQ(descs[1].ir_idx, -1);
  EXPECT_EQ(descs[1].folded, false);

  EXPECT_EQ(descs[2].addr_type, AddrType::WORKSPACE);
  EXPECT_EQ(descs[2].ir_idx, -1);
  EXPECT_EQ(descs[2].folded, false);
}

REG_OP(DynOP)
    .INPUT(a, TensorType(DT_FLOAT32))
    .DYNAMIC_INPUT(b, TensorType(DT_FLOAT32))
    .INPUT(c, TensorType(DT_FLOAT32))
    .OUTPUT(x, TensorType(DT_FLOAT32))
    .DYNAMIC_OUTPUT(y, TensorType(DT_FLOAT32))
    .OUTPUT(z, TensorType(DT_FLOAT32))
    .OP_END_FACTORY_REG(DynOP);

TEST_F(UtestArgsFormatDescUtils, common_args_dynamic_folded) {
  std::vector<ArgDesc> args_desc;
  ArgsFormatDescUtils::Append(args_desc, AddrType::INPUT, -1);
  ArgsFormatDescUtils::Append(args_desc, AddrType::OUTPUT, -1);
  ArgsFormatDescUtils::Append(args_desc, AddrType::WORKSPACE, 0);
  ArgsFormatDescUtils::Append(args_desc, AddrType::WORKSPACE, 1);
  std::string res = ArgsFormatDescUtils::ToString(args_desc);
  std::string expect_res = "{i*}{o*}{ws0*}{ws1*}";
  EXPECT_EQ(expect_res, res);

  std::vector<ArgDesc> descs;
  EXPECT_EQ(ArgsFormatDescUtils::Parse(expect_res, descs), SUCCESS);
  EXPECT_EQ(descs[0].addr_type, AddrType::INPUT);
  EXPECT_EQ(descs[0].ir_idx, -1);
  EXPECT_EQ(descs[0].folded, false);

  EXPECT_EQ(descs[2].addr_type, AddrType::WORKSPACE);
  EXPECT_EQ(descs[2].ir_idx, 0);
  EXPECT_EQ(descs[2].folded, false);

  std::string expanded_res = "{i0}{i1}{i2}{o0}{o1}{o2}{ws0}{ws1}{i0}{i_desc0}{i_instance0}{o_instance*}";
  std::vector<ArgDesc> expanded_descs;
  EXPECT_EQ(ArgsFormatDescUtils::Parse(expanded_res, expanded_descs), SUCCESS);
  // 对外开放的接口，由于没有op_desc，因此不会将i* o* ws*类似的展开,因此不会相等
  EXPECT_NE(descs.size(), expanded_descs.size());

  EXPECT_EQ(expanded_descs[0].addr_type, AddrType::INPUT);
  EXPECT_EQ(expanded_descs[0].ir_idx, 0);
  EXPECT_EQ(expanded_descs[0].folded, true);

}

TEST_F(UtestArgsFormatDescUtils, serialize_dynamic_args) {
  std::vector<ArgDesc> args_des;
  ArgsFormatDescUtils::Append(args_des, AddrType::FFTS_ADDR);
  ArgsFormatDescUtils::Append(args_des, AddrType::INPUT, 0);
  ArgsFormatDescUtils::Append(args_des, AddrType::INPUT_DESC, 1, true);
  ArgsFormatDescUtils::Append(args_des, AddrType::INPUT_DESC, 2, true);
  ArgsFormatDescUtils::Append(args_des, AddrType::INPUT, 4);
  ArgsFormatDescUtils::Append(args_des, AddrType::OUTPUT_DESC, 0, true);
  ArgsFormatDescUtils::Append(args_des, AddrType::WORKSPACE, 0);
  ArgsFormatDescUtils::Append(args_des, AddrType::TILING_FFTS, 1);
  std::string res = ArgsFormatDescUtils::ToString(args_des);
  std::string expect_res = "{ffts_addr}{i0*}{i_desc1}{i_desc2}{i4*}{o_desc0}{ws0*}{t_ffts.tail}";
  EXPECT_EQ(expect_res, res);

  std::vector<ArgDesc> descs;
  EXPECT_EQ(ArgsFormatDescUtils::Parse(expect_res, descs), SUCCESS);
  EXPECT_EQ(descs.size(), 8UL);
  EXPECT_EQ(descs[0].addr_type, AddrType::FFTS_ADDR);
  EXPECT_EQ(descs[0].ir_idx, -1);
  EXPECT_EQ(descs[0].folded, false);

  EXPECT_EQ(descs[1].addr_type, AddrType::INPUT);
  EXPECT_EQ(descs[1].ir_idx, 0);
  EXPECT_EQ(descs[1].folded, false);

  EXPECT_EQ(descs[2].addr_type, AddrType::INPUT_DESC);
  EXPECT_EQ(descs[2].ir_idx, 1);
  EXPECT_EQ(descs[2].folded, true);

  EXPECT_EQ(descs[3].addr_type, AddrType::INPUT_DESC);
  EXPECT_EQ(descs[3].ir_idx, 2);
  EXPECT_EQ(descs[3].folded, true);

  EXPECT_EQ(descs[4].addr_type, AddrType::INPUT);
  EXPECT_EQ(descs[4].ir_idx, 4);
  EXPECT_EQ(descs[4].folded, false);

  EXPECT_EQ(descs[5].addr_type, AddrType::OUTPUT_DESC);
  EXPECT_EQ(descs[5].ir_idx, 0);
  EXPECT_EQ(descs[5].folded, true);

  EXPECT_EQ(descs[6].addr_type, AddrType::WORKSPACE);
  EXPECT_EQ(descs[6].ir_idx, 0);
  EXPECT_EQ(descs[6].folded, false);

  EXPECT_EQ(descs[7].addr_type, AddrType::TILING_FFTS);
  // ffts的ir_index无效，默认值-1
  EXPECT_EQ(descs[7].ir_idx, -1);
  EXPECT_EQ(descs[7].folded, false);
}

TEST_F(UtestArgsFormatDescUtils, deserialzie_placeholder) {
  auto op_desc = std::make_shared<OpDesc>("tmp_op", "Mul");
  std::string format1 = "{}";
  std::vector<ArgDesc> descs;
  EXPECT_EQ(ArgsFormatDescUtils::Parse(format1, descs), SUCCESS);
  EXPECT_EQ(descs.size(), 1UL);
  EXPECT_EQ(descs[0UL].addr_type, AddrType::PLACEHOLDER);
}

TEST_F(UtestArgsFormatDescUtils, deserialzie_unsupported) {
auto op_desc = std::make_shared<OpDesc>("tmp_op", "Mul");
  std::string format1 = "{hehe}";
  std::vector<ArgDesc> descs1;
  EXPECT_NE(ArgsFormatDescUtils::Parse(format1, descs1), SUCCESS);

  std::string format2 = "{ }";
  std::vector<ArgDesc> descs2;
  EXPECT_NE(ArgsFormatDescUtils::Parse(format2, descs2), SUCCESS);

  std::string format3 = "{hi.unsupported}";
  std::vector<ArgDesc> descs3;
  EXPECT_NE(ArgsFormatDescUtils::Parse(format3, descs3), SUCCESS);
}

TEST_F(UtestArgsFormatDescUtils, deserialzie_tiling_context) {
  auto op_desc = std::make_shared<OpDesc>("tmp_op", "Mul");
  std::string format1 =
      "{tiling_context}{*op_type}{tiling_context.tiling_key}{tiling_context.tiling_data}{tiling_context.block_dim}";
  std::vector<ArgDesc> descs;
  EXPECT_EQ(ArgsFormatDescUtils::Parse(format1, descs), SUCCESS);
  EXPECT_EQ(descs.size(), 5UL);
  EXPECT_EQ(descs[0UL].addr_type, AddrType::TILING_CONTEXT);
  EXPECT_EQ(descs[1UL].addr_type, AddrType::OP_TYPE);
  EXPECT_EQ(descs[2UL].ir_idx, static_cast<int32_t>(TilingContextSubType::TILING_KEY));
  EXPECT_EQ(descs[3UL].ir_idx, static_cast<int32_t>(TilingContextSubType::TILING_DATA));
  EXPECT_EQ(descs[4UL].ir_idx, static_cast<int32_t>(TilingContextSubType::BLOCK_DIM));
}

TEST_F(UtestArgsFormatDescUtils, serialzie_tiling_context) {
  auto op_desc = std::make_shared<OpDesc>("tmp_op", "Mul");
  std::string format1 =
      "{tiling_context}{*op_type}{tiling_context.tiling_key}{tiling_context.tiling_data}{tiling_context.block_dim}";
  std::vector<ArgDesc> args_desc;
  ArgsFormatDescUtils::AppendTilingContext(args_desc);
  ArgsFormatDescUtils::Append(args_desc, AddrType::OP_TYPE);
  ArgsFormatDescUtils::AppendTilingContext(args_desc, TilingContextSubType::TILING_KEY);
  ArgsFormatDescUtils::AppendTilingContext(args_desc, TilingContextSubType::TILING_DATA);
  ArgsFormatDescUtils::AppendTilingContext(args_desc, TilingContextSubType::BLOCK_DIM);

  std::string res = ArgsFormatDescUtils::ToString(args_desc);
  EXPECT_EQ(format1, res);
}

TEST_F(UtestArgsFormatDescUtils, insert_hidden_input) {
  auto op_desc = std::make_shared<OpDesc>("tmp_op", "Mul");
  std::vector<ArgDesc> descs;
  ArgsFormatDescUtils::Append(descs, AddrType::PLACEHOLDER);
  ArgsFormatDescUtils::InsertHiddenInputs(descs, -1, HiddenInputsType::HCOM);
  EXPECT_EQ(ArgsFormatDescUtils::ToString(descs), "{}{hi.hcom0*}");

  EXPECT_EQ(ArgsFormatDescUtils::InsertHiddenInputs(descs, 1, HiddenInputsType::HCOM, 3), GRAPH_SUCCESS);
  EXPECT_EQ(ArgsFormatDescUtils::ToString(descs), "{}{hi.hcom0*}{hi.hcom1*}{hi.hcom2*}{hi.hcom0*}");

  EXPECT_EQ(ArgsFormatDescUtils::InsertHiddenInputs(descs, 5, HiddenInputsType::HCOM, 3), GRAPH_SUCCESS);
  EXPECT_EQ(ArgsFormatDescUtils::ToString(descs), "{}{hi.hcom0*}{hi.hcom1*}{hi.hcom2*}{hi.hcom0*}{hi.hcom0*}{hi.hcom1*}{hi.hcom2*}");

  EXPECT_NE(ArgsFormatDescUtils::InsertHiddenInputs(descs, 9, HiddenInputsType::HCOM, 1), GRAPH_SUCCESS);
}

TEST_F(UtestArgsFormatDescUtils, insert_custom_val) {
  auto op_desc = std::make_shared<OpDesc>("tmp_op", "Mul");
  std::vector<ArgDesc> descs;

  EXPECT_EQ(ArgsFormatDescUtils::InsertCustomValue(descs, -1, 0), GRAPH_SUCCESS);
  EXPECT_EQ(ArgsFormatDescUtils::ToString(descs), "{#0}");

  EXPECT_EQ(ArgsFormatDescUtils::InsertCustomValue(descs, 0, UINT64_MAX), GRAPH_SUCCESS);
  EXPECT_EQ(ArgsFormatDescUtils::ToString(descs), "{#18446744073709551615}{#0}");

  EXPECT_NE(ArgsFormatDescUtils::InsertCustomValue(descs, 3, UINT64_MAX), GRAPH_SUCCESS);
}

TEST_F(UtestArgsFormatDescUtils, test_debug_log) {
  dlog_setlevel(0, 0, 0);
  std::vector<ArgDesc> desc;
  ArgsFormatDescUtils::Append(desc, AddrType::INPUT);
  ArgsFormatDescUtils::Append(desc, AddrType::INPUT, 0);
  ArgsFormatDescUtils::Append(desc, AddrType::INPUT, 0, true);
  ArgsFormatDescUtils::Append(desc, AddrType::FFTS_ADDR);
  ArgsFormatDescUtils::Append(desc, AddrType::OVERFLOW_ADDR);
  ArgsFormatDescUtils::Append(desc, AddrType::TILING);
  ArgsFormatDescUtils::Append(desc, AddrType::TILING_FFTS, 0);
  ArgsFormatDescUtils::Append(desc, AddrType::TILING_FFTS, 1);
  std::string res = ArgsFormatDescUtils::ToString(desc);
  std::string expect_res = "{i*}{i0*}{i0}{ffts_addr}{overflow_addr}{t}{t_ffts.non_tail}{t_ffts.tail}";
  EXPECT_EQ(expect_res, res);
  dlog_setlevel(0, 3, 0);
}

TEST_F(UtestArgsFormatDescUtils, AddrTypeValue) {
  // 增加新的枚举值，需要 l0 exception dump 适配
  EXPECT_EQ(static_cast<int>(AddrType::MAX), static_cast<int>(AddrType::EVENT_ADDR) + 1);
  EXPECT_EQ(static_cast<int>(TilingContextSubType::MAX), static_cast<int>(TilingContextSubType::BLOCK_DIM) + 1);
}
}  // namespace ge
