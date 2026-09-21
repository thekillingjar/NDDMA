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
#include <nlohmann/json.hpp>
#include <iostream>

#include <list>

#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#define protected public
#define private public
#include "common/configuration.h"
#include "pass_manager.h"
//#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/relu_fusion_pass.h"
#undef private
#undef protected

using namespace std;
using namespace fe;
using namespace ge;

class pass_manager_ut: public testing::Test
{
public:


protected:
    void SetUp()
    {
    }

    void TearDown()
    {

    }
// AUTO GEN PLEASE DO NOT MODIFY IT
};

namespace{

}

//TEST_F(pass_manager_ut, run_with_opstore)
//{
//  PassManager fusion_passes(nullptr);
//  fusion_passes.AddPass("ReluFusionPass", AI_CORE_NAME, new(std::nothrow) ReluFusionPass, GRAPH_FUSION);
//  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
//  Status ret = fusion_passes.Run(*(graph.get()), nullptr);
//  EXPECT_EQ(ret, fe::NOT_CHANGED);
//}
//
//TEST_F(pass_manager_ut, run_without_opstore)
//{
//  PassManager fusion_passes(nullptr);
//  fusion_passes.AddPass("ReluFusionPass", AI_CORE_NAME, new(std::nothrow) ReluFusionPass, GRAPH_FUSION);
//  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
//  Status ret = fusion_passes.Run(*(graph.get()));
//  EXPECT_EQ(ret, fe::NOT_CHANGED);
//}
