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
#include "all_ops_stub.h"

#define protected public
#define private   public

//#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/logsoftmaxgrad_fusion_pass.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/types.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/util/op_info_util.h"
#include "graph/compute_graph.h"
using namespace std;

namespace fe {
  static const char *SUB = "Sub";
  static const char *MUL ="Mul";
  static const char *EXP = "Exp";
  static const char *TF_MATMUL = "Matmul";

class UTEST_omg_optimizer_fusion_logsoftmaxgrad_fusion_unittest :public testing::Test {
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
    /*
                                                     add
                                                    /    \
                                                   /      \
                                                  /        \
                                                sum---mul---sub----matmul
                                                /      |
                                               /       |
                                              /        |
                                         constant     exp-----constant	
     */
    ge::ComputeGraphPtr CreateGraph()
    {
        // create compute graph
        ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("test_logsoftmaxgrad_fusion_graph");
        // create mul node
        ge::NodePtr mul_node = AddNode(graph, "mul", MUL, 2, 1, ge::DataType::DT_FLOAT16,
                                      ge::DataType::DT_FLOAT16);
        // create sum node
        ge::NodePtr sum_node = AddNode(graph, "sum", "Sum", 1, 1, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT);
        // create sub node
        ge::NodePtr sub_node = AddNode(graph, "/sub", SUB, 2, 2, ge::DataType::DT_FLOAT,
                                      ge::DataType::DT_FLOAT);
        // create exp node
        ge::NodePtr exp_node = AddNode(graph, "exp", EXP, 1, 1, ge::DataType::DT_FLOAT,
                                      ge::DataType::DT_FLOAT);
        // create cosnt node
        ge::NodePtr const_node = AddNode(graph, "constant", fe::CONSTANT, 1, 1, ge::DataType::DT_FLOAT,
                                        ge::DataType::DT_FLOAT);

        // create matmul node
        ge::NodePtr matmul_node = AddNode(graph, "matmul", TF_MATMUL, 1, 2, ge::DataType::DT_FLOAT,
                                         ge::DataType::DT_FLOAT);
        // create add node
        ge::NodePtr add_node = AddNode(graph, "add", "Add", 1, 2, ge::DataType::DT_FLOAT,
                                      ge::DataType::DT_FLOAT);
        // link sub node and matmul node
        ge::GraphUtils::AddEdge(sub_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0));
        // link sub node and mul node
        ge::GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0), sub_node->GetInDataAnchor(1));
        // link add node and sub node
        ge::GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), sub_node->GetInDataAnchor(0));
        // link add node and sum node
        ge::GraphUtils::AddEdge(add_node->GetOutDataAnchor(1), sum_node->GetInDataAnchor(0));
        // link sum node and mul node
        ge::GraphUtils::AddEdge(sum_node->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
        // link exp node and mul node
        ge::GraphUtils::AddEdge(exp_node->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));
        // link exp node and const node
        ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), exp_node->GetInDataAnchor(0));
        // set sum input const node
        // ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), sum_node->GetInDataAnchor(1));
        ge::GeTensorDesc tensor_desc(ge::GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
        vector<int32_t> axis = {-1};
        ge::AttrUtils::SetListInt(sum_node->GetOpDesc(), "axes", axis);
        return graph;
    }
};
}
