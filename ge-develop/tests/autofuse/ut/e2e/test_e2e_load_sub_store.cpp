/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

#include "ascir_utils.h"
#include "optimize.h"
#include "codegen.h"

class E2E_LoadSubStore : public ::testing::Test {
protected:
    optimize::Optimizer optimizer;
    codegen::Codegen codegen;

    E2E_LoadSubStore() : optimizer(optimize::OptimizerOptions{}), codegen(codegen::CodegenOptions{}) {}
};
