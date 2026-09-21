/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/op_desc.h"
#include "register/hidden_input_func_registry.h"
#include <gtest/gtest.h>
#include <memory>
namespace ge {
ge::graphStatus HcomHiddenInputFunc(const ge::OpDescPtr &op_desc, void *&addr) {
  addr = reinterpret_cast<void *>(0xf1);
  return ge::GRAPH_SUCCESS;
}
class HiddenInputFuncRegistryUnittest : public testing::Test {};

TEST_F(HiddenInputFuncRegistryUnittest, HcomHiddenFuncRegisterSuccess_Test) {
  EXPECT_EQ(ge::HiddenInputFuncRegistry::GetInstance().FindHiddenInputFunc(ge::HiddenInputType::HCOM), nullptr);
  REG_HIDDEN_INPUT_FUNC(ge::HiddenInputType::HCOM, HcomHiddenInputFunc);
  ge::GetHiddenAddr func = nullptr;
  func = ge::HiddenInputFuncRegistry::GetInstance().FindHiddenInputFunc(ge::HiddenInputType::HCOM);
  EXPECT_EQ(func, HcomHiddenInputFunc);
  const ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  void *res = nullptr;
  EXPECT_EQ(func(op_desc, res), ge::GRAPH_SUCCESS);
  EXPECT_EQ(reinterpret_cast<uint64_t>(res), 0xf1);
}
}  // namespace ge
