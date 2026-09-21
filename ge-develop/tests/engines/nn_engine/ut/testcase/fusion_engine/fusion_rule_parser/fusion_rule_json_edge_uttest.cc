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
#include <fstream>
#include <nlohmann/json.hpp>
#include "fe_llt_utils.h"
#define protected public
#define private   public

#include "fusion_rule_manager/fusion_rule_parser/fusion_rule_json_pattern.h"

using namespace testing;
using namespace fe;
using namespace std;
using namespace nlohmann;

class UTEST_FUSION_RULE_JSON_EDGE : public testing::Test {
protected:
    void SetUp()
    {
        string file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_rule_parser/fusion_rule_json_edge_test.json";

        ifstream ifs(file_path);
        if (!ifs.is_open()) {
            cout << "Open file:" << file_path << " failed." << endl;
            return;
        }
        try {
            ifs >> test_file;
            ifs.close();
        } catch(const exception& e) {
            cout << "Convert file:" << file_path << " to Json failed." << endl;
            cout << e.what() << endl;
            ifs.close();
            return;
        }
    }

    void TearDowm() {}

    void Dump(const FusionRuleJsonAnchor& object, string space = "")
    {
        if(!object.has_index_) {
            printf("%s\"%s\"", space.c_str() ,object.name_.c_str());
        } else {
            printf("%s\"%s : %d\"", space.c_str() ,object.src_node_.c_str(), object.src_index_);
        }
    }

    void Dump(const FusionRuleJsonEdge& object, string space = "")
    {
        printf("%s{\n", space.c_str());

        if (object.src_ != nullptr) {
            printf("    %s\"src\" : ", space.c_str());
            Dump(*(object.src_.get()));
            printf(",\n");
        }

        if (object.dst_ != nullptr) {
            printf("    %s\"dst\" : ",space.c_str());
            Dump(*(object.dst_.get()));
            printf("\n");
        }

        printf("%s}", space.c_str());
    }
private:
    json test_file;
};

TEST_F(UTEST_FUSION_RULE_JSON_EDGE, test_001)
{
    FusionRuleJsonEdge test_object;
    Status ret = test_object.ParseToJsonEdge(test_file["test_001"]);
    Dump(test_object);
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
TEST_F(UTEST_FUSION_RULE_JSON_EDGE, test_002)
{
    FusionRuleJsonEdge test_object;
    Status ret = test_object.ParseToJsonEdge(test_file["test_002"]);
    Dump(test_object);
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
TEST_F(UTEST_FUSION_RULE_JSON_EDGE, test_003)
{
    FusionRuleJsonEdge test_object;
    Status ret = test_object.ParseToJsonEdge(test_file["test_003"]);
    Dump(test_object);
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
TEST_F(UTEST_FUSION_RULE_JSON_EDGE, test_004)
{
    FusionRuleJsonEdge test_object;
    Status ret = test_object.ParseToJsonEdge(test_file["test_004"]);
    Dump(test_object);
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
TEST_F(UTEST_FUSION_RULE_JSON_EDGE, test_005)
{
    FusionRuleJsonEdge test_object;
    Status ret = test_object.ParseToJsonEdge(test_file["test_005"]);
    Dump(test_object);
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
TEST_F(UTEST_FUSION_RULE_JSON_EDGE, test_006)
{
    FusionRuleJsonEdge test_object;
    Status ret = test_object.ParseToJsonEdge(test_file["test_006"]);
    Dump(test_object);
    EXPECT_EQ(ret, fe::SUCCESS);
}
TEST_F(UTEST_FUSION_RULE_JSON_EDGE, test_007)
{
    FusionRuleJsonEdge test_object;
    Status ret = test_object.ParseToJsonEdge(test_file["test_007"]);
    Dump(test_object);
    EXPECT_EQ(ret, fe::SUCCESS);
}
TEST_F(UTEST_FUSION_RULE_JSON_EDGE, test_008)
{
    FusionRuleJsonEdge test_object;
    Status ret = test_object.ParseToJsonEdge(test_file["test_008"]);
    Dump(test_object);
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
TEST_F(UTEST_FUSION_RULE_JSON_EDGE, test_009)
{
    FusionRuleJsonEdge test_object;
    Status ret = test_object.ParseToJsonEdge(test_file["test_009"]);
    Dump(test_object);
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
