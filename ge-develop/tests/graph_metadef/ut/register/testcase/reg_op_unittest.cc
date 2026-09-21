/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/operator_reg.h"
#include <gtest/gtest.h>

#include "graph/utils/op_desc_utils.h"
namespace {
const std::string kStr1 = "abc";
const std::string kStr2 = "abcd";
const std::vector<std::string> kStrs = {"a", "bc"};
const std::vector<ge::AscendString> kAscendStrs = {ge::AscendString("a"), ge::AscendString("b")};
}

namespace ge {
class RegisterOpUnittest : public testing::Test {};

REG_OP(AttrIrNameRegSuccess1)
    .ATTR(AttrInt, Int, 0)
    .ATTR(AttrFloat, Float, 0.0)
    .ATTR(AttrBool, Bool, true)
    .ATTR(AttrTensor, Tensor, Tensor())
    .ATTR(AttrType, Type, DT_INT32)
    .ATTR(AttrString, String, "")
    .ATTR(AttrString1, String, kStr1)
    .ATTR(AttrString2, String, "ab")
    .ATTR(AttrAscendString, AscendString, "")
    .ATTR(AttrAscendString1, AscendString, AscendString("abc"))
    .ATTR(AttrListInt, ListInt, {})
    .ATTR(AttrListFloat, ListFloat, {})
    .ATTR(AttrListBool, ListBool, {})
    .ATTR(AttrListTensor, ListTensor, {})
    .ATTR(AttrListType, ListType, {})
    .ATTR(AttrListString, ListString, {})
    .ATTR(AttrListString1, ListString, {"", ""})
    .ATTR(AttrListString2, ListString, {kStr1, kStr2})
    .ATTR(AttrListString3, ListString, kStrs)
    .ATTR(AttrListAscendString, ListAscendString, {})
    .ATTR(AttrListAscendString1, ListAscendString, kAscendStrs)
    .ATTR(AttrBytes, Bytes, {})
    .ATTR(AttrListListInt, ListListInt, {})

    .REQUIRED_ATTR(ReqAttrInt, Int)
    .REQUIRED_ATTR(ReqAttrFloat, Float)
    .REQUIRED_ATTR(ReqAttrBool, Bool)
    .REQUIRED_ATTR(ReqAttrTensor, Tensor)
    .REQUIRED_ATTR(ReqAttrType, Type)
    .REQUIRED_ATTR(ReqAttrString, String)
    .REQUIRED_ATTR(ReqAttrAscendString, AscendString)

    .REQUIRED_ATTR(ReqAttrListInt, ListInt)
    .REQUIRED_ATTR(ReqAttrListFloat, ListFloat)
    .REQUIRED_ATTR(ReqAttrListBool, ListBool)
    .REQUIRED_ATTR(ReqAttrListTensor, ListTensor)
    .REQUIRED_ATTR(ReqAttrListType, ListType)
    .REQUIRED_ATTR(ReqAttrListString, ListString)
    .REQUIRED_ATTR(ReqAttrListAscendString, ListAscendString)
    .REQUIRED_ATTR(ReqAttrBytes, Bytes)
    .REQUIRED_ATTR(ReqAttrListListInt, ListListInt)

    //.ATTR(AttrNamedAttrs, NamedAttrs, NamedAttrs())
    //.ATTR(AttrListNamedAttrs, ListNamedAttrs, {})
    .OP_END_FACTORY_REG(AttrIrNameRegSuccess1);

REG_OP(InputIrNameRegSuccess1)
    .INPUT(fix_input1, TensorType({DT_INT32, DT_INT64}))
    .INPUT(fix_input2, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(opi1, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(opi2, TensorType({DT_INT32, DT_INT64}))
    .DYNAMIC_INPUT(dyi1, TensorType({DT_INT32, DT_INT64}))
    .DYNAMIC_INPUT(dyi2, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(InputIrNameRegSuccess1);

TEST_F(RegisterOpUnittest, AttrIrNameRegSuccess) {
  auto op = OperatorFactory::CreateOperator("AttrIrNameRegSuccess1Op", "AttrIrNameRegSuccess1");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  const auto &ir_names = op_desc->GetIrAttrNames();
  EXPECT_EQ(ir_names,
            std::vector<std::string>({"AttrInt",
                                      "AttrFloat",
                                      "AttrBool",
                                      "AttrTensor",
                                      "AttrType",
                                      "AttrString",
                                      "AttrString1",
                                      "AttrString2",
                                      "AttrAscendString",
                                      "AttrAscendString1",
                                      "AttrListInt",
                                      "AttrListFloat",
                                      "AttrListBool",
                                      "AttrListTensor",
                                      "AttrListType",
                                      "AttrListString",
                                      "AttrListString1",
                                      "AttrListString2",
                                      "AttrListString3",
                                      "AttrListAscendString",
                                      "AttrListAscendString1",
                                      "AttrBytes",
                                      "AttrListListInt",
                                      "ReqAttrInt",
                                      "ReqAttrFloat",
                                      "ReqAttrBool",
                                      "ReqAttrTensor",
                                      "ReqAttrType",
                                      "ReqAttrString",
                                      "ReqAttrAscendString",
                                      "ReqAttrListInt",
                                      "ReqAttrListFloat",
                                      "ReqAttrListBool",
                                      "ReqAttrListTensor",
                                      "ReqAttrListType",
                                      "ReqAttrListString",
                                      "ReqAttrListAscendString",
                                      "ReqAttrBytes",
                                      "ReqAttrListListInt"}));
}

TEST_F(RegisterOpUnittest, InputIrNameRegSuccess) {
  auto op = OperatorFactory::CreateOperator("InputIrNameRegSuccess1", "InputIrNameRegSuccess1");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  const auto &inputs = op_desc->GetIrInputs();
  std::vector<std::pair<std::string, IrInputType>> expect_inputs({{"fix_input1", kIrInputRequired},
                                                                  {"fix_input2", kIrInputRequired},
                                                                  {"opi1", kIrInputOptional},
                                                                  {"opi2", kIrInputOptional},
                                                                  {"dyi1", kIrInputDynamic},
                                                                  {"dyi2", kIrInputDynamic}});
  EXPECT_EQ(inputs, expect_inputs);

  EXPECT_EQ(op_desc->GetValidInputNameByIndex(0), "fix_input1");
  EXPECT_EQ(op_desc->GetValidInputNameByIndex(1), "fix_input2");
  EXPECT_EQ(op_desc->GetValidInputNameByIndex(2), "");
  EXPECT_EQ(op_desc->GetValidInputNameByIndex(3), "");
  EXPECT_EQ(op_desc->GetValidInputNameByIndex(4), "");
  EXPECT_EQ(op_desc->GetValidInputNameByIndex(5), "");
}

}  // namespace ge
