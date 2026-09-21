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

#define protected public
#define private public

#include "fusion_rule_manager/fusion_rule_manager.h"
#include "common/fe_log.h"
#include "common/configuration.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"

#undef private
#undef protected

using namespace std;
using namespace fe;

class fusion_rule_manager_unittest : public testing::Test
{
protected:
    void SetUp()
    {
        ops_kernel_info_store_ptr_ = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
    }

    void TearDown()
    {
    }
    FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr_;
};

TEST_F(fusion_rule_manager_unittest, initialize_success)
{
    auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    frm->init_flag_ = true;
    Status ret = frm->Initialize(fe::AI_CORE_NAME);
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(fusion_rule_manager_unittest, finalize_success_not_init)
{
    auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    Status ret = frm->Finalize();
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(fusion_rule_manager_unittest, finalize_success_init)
{
    auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    frm->init_flag_ = true;
    Status ret = frm->Finalize();
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(fusion_rule_manager_unittest, get_fusion_rules_by_rule_type_failed_not_init1)
{
    auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    RuleType rule_type;
    rule_type = RuleType::CUSTOM_GRAPH_RULE;
    vector<FusionRulePatternPtr> out_rule_vector = {};
    Status ret = frm->GetFusionRulesByRuleType(rule_type, out_rule_vector);
    EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(fusion_rule_manager_unittest, get_fusion_rules_by_rule_type_failed_not_init2)
{
    auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    frm->init_flag_ = false;
    RuleType rule_type;
    std::string rule_name = "rule_name";
    Status ret = frm->RunGraphFusionRuleByType(*graph, rule_type, rule_name);
    EXPECT_EQ(fe::FAILED, ret);
}

