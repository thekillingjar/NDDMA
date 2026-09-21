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


//#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/matmul_cast_fusion_pass.h"

#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "common/util/op_info_util.h"

using namespace std;
using namespace fe;
static const uint8_t MATMUL_INPUT_NUM = 2;
static const uint8_t MATMUL_OUTPUT_NUM = 1;
static const uint8_t CAST_INPUT_NUM = 1;
static const uint8_t CAST_OUTPUT_NUM = 1;
static const uint8_t SUB_INPUT_NUM = 2;
static const uint8_t SUB_OUTPUT_NUM = 1;
static const int32_t TENSORFLOW_DATATYPE_FLOAT32 = 0;
static const int32_t TENSORFLOW_DATATYPE_FLOAT16 = 19;
static const string MATMUL_DATATYPE_ATTR_KEY = "T";
static const string CAST_DATATYPE_DES_ATTR_KEY = "DstT";

static const char *TF_MATMUL = "Matmul";
static const char *SUB = "Sub";

class UTEST_fusion_engine_matmul_cast_fusion_unittest :public testing::Test {
protected:
    void SetUp()
    {
    }
    void TearDown()
    {

    }

protected:
    ge::NodePtr AddNode(ge::ComputeGraphPtr graph,
                        const string &name,
                        const string &type,
                        unsigned int in_anchor_num,
                        unsigned int out_anchor_num,
                        ge::DataType input_type,
                        ge::DataType output_type)
    {
        ge::GeTensorDesc input_tensor_desc(ge::GeShape({1, 2, 3}), ge::FORMAT_NHWC, input_type);
        ge::GeTensorDesc output_tensor_desc(ge::GeShape({1, 2, 3}), ge::FORMAT_NHWC, output_type);
        ge::OpDescPtr op_desc = make_shared<ge::OpDesc>(name, type);
        for (unsigned int i = 0; i < in_anchor_num; ++i) {
            op_desc->AddInputDesc(input_tensor_desc);
        }
        for (unsigned int i = 0; i < out_anchor_num; ++i) {
            op_desc->AddOutputDesc(output_tensor_desc);
        }
        ge::NodePtr node = graph->AddNode(op_desc);
        return node;
    }

    ge::ComputeGraphPtr CreateMatmulCastGraph()
    {
        // create compute graph
        ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("test_matmul_cast_fusion_graph");
        // create matmul node
        ge::NodePtr matmul_node = AddNode(graph, "matmul", TF_MATMUL, MATMUL_INPUT_NUM, MATMUL_OUTPUT_NUM,
                                         ge::DataType::DT_FLOAT16,
                                         ge::DataType::DT_FLOAT16);
        // create cast node
        ge::NodePtr cast_node = AddNode(graph, "cast", fe::CAST, CAST_INPUT_NUM, CAST_OUTPUT_NUM,
                                       ge::DataType::DT_FLOAT16,
                                       ge::DataType::DT_FLOAT);
        // create sub node
        ge::NodePtr sub_node = AddNode(graph, "sub", SUB, SUB_INPUT_NUM, SUB_OUTPUT_NUM,
                                      ge::DataType::DT_FLOAT,
                                      ge::DataType::DT_FLOAT);
        // link matmul node and cast node
        ge::GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0));
        // linkd cast node and sub node
        ge::GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), sub_node->GetInDataAnchor(0));
        return graph;
    }
};
