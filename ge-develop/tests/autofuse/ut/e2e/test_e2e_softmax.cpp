/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "e2e_softmax.h"

#include "gtest/gtest.h"

#include "optimize.h"
#include "codegen.h"
#include "ascir_utils.h"

using namespace ascir;
class E2E_Softmax : public ::testing::Test {
 protected:
  optimize::Optimizer optimizer;
  codegen::Codegen codegen;

  E2E_Softmax() : optimizer(optimize::OptimizerOptions{}), codegen(codegen::CodegenOptions{}) {}
};

TEST_F(E2E_Softmax, ConstructGraphWithAscir) {
  ge::AscGraph graph("graph");
  Softmax_BeforeAutofuse(graph);

  std::cout << utils::DebugStr(graph) << std::endl;
}
