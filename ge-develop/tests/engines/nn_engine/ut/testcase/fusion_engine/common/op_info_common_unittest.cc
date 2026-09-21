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
#include <string>
#include <vector>

#define protected public
#define private public
#include "common/fe_op_info_common.h"
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "compute_graph.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#undef protected
#undef private

using namespace std;
using namespace fe;
using namespace ge;

class UTEST_op_info_common_unittest : public testing::Test
{
 protected:
  void SetUp()
  {
  }

  void TearDown()
  {
  }

  static void CreateGraph(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BN");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Cast");
    OpDescPtr out_op = std::make_shared<OpDesc>("out", "NetOutput");

    // add descriptor
    vector<int64_t> dims = {1,2,3,4};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_FRACTAL_NZ);
    in_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_FRACTAL_NZ);
    out_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_NZ);
    in_desc2.SetDataType(DT_FLOAT16);
    relu_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_FRACTAL_NZ);
    out_desc2.SetDataType(DT_FLOAT16);
    relu_op->AddOutputDesc("y", out_desc2);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_FRACTAL_NZ);
    in_desc3.SetDataType(DT_FLOAT16);
    out_op->AddInputDesc("x", in_desc3);

    GeTensorDesc out_desc3(shape);
    out_desc3.SetFormat(FORMAT_FRACTAL_NZ);
    out_desc3.SetDataType(DT_FLOAT16);
    out_op->AddOutputDesc("y", out_desc3);


    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr relu_node = graph->AddNode(relu_op);
    NodePtr out_node = graph->AddNode(out_op);

    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));

  }
};
