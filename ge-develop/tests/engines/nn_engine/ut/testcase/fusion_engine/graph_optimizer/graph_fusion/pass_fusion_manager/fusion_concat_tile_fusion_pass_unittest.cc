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

#define private public
#define protected public
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/concat_tile_fusion_pass.h"
#include "common/fe_utils.h"
#include "pass_manager.h"
#include "common/util/op_info_util.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

#undef protected
#undef private

using namespace std;
using namespace ge;
using namespace fe;

namespace fe {

class fusion_concat_tile_fusion_cast_ut : public testing::Test {
public:
  const string GRAPH_NAME = "test";
  const string CONCATV2 = "ConcatV2";
  const string CONCAT = "Concat";
  const string TILED = "TileD";
  const string TILE = "Tile";
protected:
  void SetUp() {}

  void TearDown() {}

  void InitGraph(ComputeGraphPtr &graph) {
    OpDescPtr tile1 = std::make_shared<OpDesc>("tile0", TILED);
    OpDescPtr tile2 = std::make_shared<OpDesc>("tile1", TILED);
    OpDescPtr tile3 = std::make_shared<OpDesc>("tile2", TILE);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATV2);
    OpDescPtr end = std::make_shared<OpDesc>("end", "End");
    ge::AttrUtils::SetListInt(tile1, "multiples", vector<int64_t>{128, 1,1,1});
    ge::AttrUtils::SetListInt(tile2, "multiples", vector<int64_t>{128, 1,1,1});

    OpDescPtr data1 = std::make_shared<OpDesc>("data0", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr data3 = std::make_shared<OpDesc>("data2", "Data");

    OpDescPtr const_data = std::make_shared<OpDesc>("const", "Constant");

    // add descriptor
    ge::GeShape shape0({1, 64, 7, 17777});
    GeTensorDesc input_desc0(shape0, ge::FORMAT_NCHW, ge::DT_FLOAT);
    input_desc0.SetOriginShape(shape0);
    tile1->AddInputDesc(input_desc0);
    tile2->AddInputDesc(input_desc0);
    tile3->AddInputDesc(input_desc0);

    data1->AddOutputDesc(input_desc0);
    data2->AddOutputDesc(input_desc0);
    data3->AddOutputDesc(input_desc0);

    ge::GeShape shape1({128, 64, 7, 17777});
    GeTensorDesc out_desc(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc.SetOriginShape(shape1);
    tile1->AddOutputDesc(out_desc);
    tile2->AddOutputDesc(out_desc);
    tile3->AddOutputDesc(out_desc);

    concat->AddInputDesc(out_desc);
    concat->AddInputDesc(out_desc);
    concat->AddInputDesc(out_desc);


    ge::GeShape shape2({128, 64, 21, 1777});
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);
    concat->AddOutputDesc(out_desc2);
    end->AddInputDesc(out_desc2);

    ge::GeShape shape3({1});
    GeTensorDesc out_desc3(shape2, ge::FORMAT_ND, ge::DT_INT32);
    concat->AddInputDesc(out_desc3);
    const_data->AddOutputDesc(out_desc3);


    NodePtr tile1_node = graph->AddNode(tile1);
    NodePtr tile2_node = graph->AddNode(tile2);
    NodePtr tile3_node = graph->AddNode(tile3);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr data3_node = graph->AddNode(data3);
    NodePtr concat_node = graph->AddNode(concat);
    NodePtr const_op = graph->AddNode(const_data);
    
    
    NodePtr end_node = graph->AddNode(end);

    ge::GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), tile1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), tile2_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data3_node->GetOutDataAnchor(0), tile3_node->GetInDataAnchor(0));

    ge::GraphUtils::AddEdge(tile1_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(tile2_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(tile3_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(const_op->GetOutDataAnchor(0), concat_node->GetInDataAnchor(3));
    ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), end_node->GetInDataAnchor(0));
  }

  void InitGraph1(ComputeGraphPtr &graph) {
    OpDescPtr tile1 = std::make_shared<OpDesc>("tile0", TILED);
    OpDescPtr tile2 = std::make_shared<OpDesc>("tile1", TILED);
    OpDescPtr tile3 = std::make_shared<OpDesc>("tile2", TILE);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATV2);
    OpDescPtr end = std::make_shared<OpDesc>("end", "End");
    ge::AttrUtils::SetListInt(tile1, "multiples", vector<int64_t>{128, 1,1,1});
    ge::AttrUtils::SetListInt(tile2, "multiples", vector<int64_t>{128, 1,1,1});

    OpDescPtr data1 = std::make_shared<OpDesc>("data0", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr data3 = std::make_shared<OpDesc>("data2", "Data");

    OpDescPtr const_data = std::make_shared<OpDesc>("const", "Constant");

    // add descriptor
    ge::GeShape shape0({1, 6, 1, 1});
    GeTensorDesc input_desc0(shape0, ge::FORMAT_NCHW, ge::DT_FLOAT);
    input_desc0.SetOriginShape(shape0);
    tile1->AddInputDesc(input_desc0);
    tile2->AddInputDesc(input_desc0);
    tile3->AddInputDesc(input_desc0);
    data1->AddOutputDesc(input_desc0);
    data2->AddOutputDesc(input_desc0);
    data3->AddOutputDesc(input_desc0);

    ge::GeShape shape1({128, 6, 1, 1});
    GeTensorDesc out_desc(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc.SetOriginShape(shape1);
    tile1->AddOutputDesc(out_desc);
    tile2->AddOutputDesc(out_desc);
    tile3->AddOutputDesc(out_desc);

    concat->AddInputDesc(out_desc);
    concat->AddInputDesc(out_desc);
    concat->AddInputDesc(out_desc);


    ge::GeShape shape2({128, 6, 3, 1});
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);
    concat->AddOutputDesc(out_desc2);
    end->AddInputDesc(out_desc2);

    ge::GeShape shape3({1});
    GeTensorDesc out_desc3(shape2, ge::FORMAT_ND, ge::DT_INT32);
    concat->AddInputDesc(out_desc3);
    const_data->AddOutputDesc(out_desc3);


    NodePtr tile1_node = graph->AddNode(tile1);
    NodePtr tile2_node = graph->AddNode(tile2);
    NodePtr tile3_node = graph->AddNode(tile3);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr data3_node = graph->AddNode(data3);
    NodePtr concat_node = graph->AddNode(concat);
    NodePtr const_op = graph->AddNode(const_data);
    
    
    NodePtr end_node = graph->AddNode(end);

    ge::GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), tile1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), tile2_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data3_node->GetOutDataAnchor(0), tile3_node->GetInDataAnchor(0));

    ge::GraphUtils::AddEdge(tile1_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(tile2_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(tile3_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(const_op->GetOutDataAnchor(0), concat_node->GetInDataAnchor(3));
    ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), end_node->GetInDataAnchor(0));
  }

  void InitGraph2(ComputeGraphPtr &graph) {
    OpDescPtr tile1 = std::make_shared<OpDesc>("tile0", TILED);
    OpDescPtr tile2 = std::make_shared<OpDesc>("tile1", TILED);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATV2);
    OpDescPtr end = std::make_shared<OpDesc>("end", "End");
    ge::AttrUtils::SetListInt(tile1, "multiples", vector<int64_t>{128, 1,1,1});
    ge::AttrUtils::SetListInt(tile2, "multiples", vector<int64_t>{128, 1,1,1});

    OpDescPtr data1 = std::make_shared<OpDesc>("data0", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("data1", "Data");

    OpDescPtr const_data = std::make_shared<OpDesc>("const", "Constant");

    // add descriptor
    ge::GeShape shape0({1, 6, 1, 1});
    GeTensorDesc input_desc0(shape0, ge::FORMAT_NCHW, ge::DT_FLOAT);
    input_desc0.SetOriginShape(shape0);
    tile1->AddInputDesc(input_desc0);
    tile2->AddInputDesc(input_desc0);
    data1->AddOutputDesc(input_desc0);
    data2->AddOutputDesc(input_desc0);


    ge::GeShape shape1({128, 6, 1, 1});
    GeTensorDesc out_desc(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc.SetOriginShape(shape1);
    tile1->AddOutputDesc(out_desc);
    tile2->AddOutputDesc(out_desc);

    concat->AddInputDesc(out_desc);
    concat->AddInputDesc(out_desc);


    ge::GeShape shape2({128, 6, 2, 1});
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);
    concat->AddOutputDesc(out_desc2);
    end->AddInputDesc(out_desc2);

    ge::GeShape shape3({1});
    GeTensorDesc out_desc3(shape2, ge::FORMAT_ND, ge::DT_INT32);
    concat->AddInputDesc(out_desc3);
    const_data->AddOutputDesc(out_desc3);


    NodePtr tile1_node = graph->AddNode(tile1);
    NodePtr tile2_node = graph->AddNode(tile2);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr concat_node = graph->AddNode(concat);
    NodePtr const_op = graph->AddNode(const_data);
    
    
    NodePtr end_node = graph->AddNode(end);

    ge::GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), tile1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), tile2_node->GetInDataAnchor(0));

    ge::GraphUtils::AddEdge(tile1_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(tile2_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(const_op->GetOutDataAnchor(0), concat_node->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), end_node->GetInDataAnchor(0));
  }

  void InitGraph3(ComputeGraphPtr &graph) {
    OpDescPtr tile1 = std::make_shared<OpDesc>("tile0", TILED);
    OpDescPtr tile2 = std::make_shared<OpDesc>("tile1", TILED);
    OpDescPtr tile3 = std::make_shared<OpDesc>("tile2", TILED);

    OpDescPtr tile4 = std::make_shared<OpDesc>("tile3", TILED);
    OpDescPtr tile5 = std::make_shared<OpDesc>("tile4", TILED);
    OpDescPtr tile6 = std::make_shared<OpDesc>("tile5", TILED);
    ge::AttrUtils::SetListInt(tile1, "multiples", vector<int64_t>{128, 1,1,1});
    ge::AttrUtils::SetListInt(tile2, "multiples", vector<int64_t>{128, 1,1,1});
    ge::AttrUtils::SetListInt(tile3, "multiples", vector<int64_t>{128, 1,1,1});

    ge::AttrUtils::SetListInt(tile4, "multiples", vector<int64_t>{64, 1,1,1});
    ge::AttrUtils::SetListInt(tile5, "multiples", vector<int64_t>{64, 1,1,1});
    ge::AttrUtils::SetListInt(tile6, "multiples", vector<int64_t>{64, 1,1,1});

    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATV2);
    OpDescPtr end = std::make_shared<OpDesc>("end", "End");

    OpDescPtr data1 = std::make_shared<OpDesc>("data0", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr data3 = std::make_shared<OpDesc>("data2", "Data");

    OpDescPtr data4 = std::make_shared<OpDesc>("data3", "Data");
    OpDescPtr data5 = std::make_shared<OpDesc>("data4", "Data");
    OpDescPtr data6 = std::make_shared<OpDesc>("data5", "Data");

    OpDescPtr const_data = std::make_shared<OpDesc>("const", "Constant");
    OpDescPtr contrl_data = std::make_shared<OpDesc>("contrlconst", "Constant");

    // add descriptor
    ge::GeShape shape0({1, 6, 1, 1});
    GeTensorDesc input_desc0(shape0, ge::FORMAT_NCHW, ge::DT_FLOAT);
    input_desc0.SetOriginShape(shape0);
    tile1->AddInputDesc(input_desc0); 
    tile2->AddInputDesc(input_desc0);
    tile3->AddInputDesc(input_desc0);

    data1->AddOutputDesc(input_desc0);
    data2->AddOutputDesc(input_desc0);
    data3->AddOutputDesc(input_desc0);

    ge::GeShape shape_0({2, 6, 1, 1});
    GeTensorDesc input_desc_0(shape_0, ge::FORMAT_NCHW, ge::DT_FLOAT);
    input_desc_0.SetOriginShape(shape_0);
    tile4->AddInputDesc(input_desc_0);
    tile5->AddInputDesc(input_desc_0);
    tile6->AddInputDesc(input_desc_0);

    data4->AddOutputDesc(input_desc_0);
    data5->AddOutputDesc(input_desc_0);
    data6->AddOutputDesc(input_desc_0);

    ge::GeShape shape1({128, 6, 6, 1});
    GeTensorDesc out_desc(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc.SetOriginShape(shape1);
    tile1->AddOutputDesc(out_desc);
    tile2->AddOutputDesc(out_desc);
    tile3->AddOutputDesc(out_desc);

    tile4->AddOutputDesc(out_desc);
    tile5->AddOutputDesc(out_desc);
    tile6->AddOutputDesc(out_desc);

    concat->AddInputDesc(out_desc);
    concat->AddInputDesc(out_desc);
    concat->AddInputDesc(out_desc);

    concat->AddInputDesc(out_desc);
    concat->AddInputDesc(out_desc);
    concat->AddInputDesc(out_desc);


    ge::GeShape shape2({128, 6, 42, 7});
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);
    concat->AddOutputDesc(out_desc2);
    end->AddInputDesc(out_desc2);

    ge::GeShape shape3({1});
    GeTensorDesc out_desc3(shape2, ge::FORMAT_ND, ge::DT_INT32);
    concat->AddInputDesc(out_desc3);
    const_data->AddOutputDesc(out_desc3);


    NodePtr tile1_node = graph->AddNode(tile1);
    NodePtr tile2_node = graph->AddNode(tile2);
    NodePtr tile3_node = graph->AddNode(tile3);
    NodePtr tile4_node = graph->AddNode(tile4);
    NodePtr tile5_node = graph->AddNode(tile5);
    NodePtr tile6_node = graph->AddNode(tile6);

    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr data3_node = graph->AddNode(data3);
    NodePtr data4_node = graph->AddNode(data4);
    NodePtr data5_node = graph->AddNode(data5);
    NodePtr data6_node = graph->AddNode(data6);


    NodePtr concat_node = graph->AddNode(concat);
    NodePtr const_op = graph->AddNode(const_data);
    NodePtr contrl_op = graph->AddNode(contrl_data);
    
    NodePtr end_node = graph->AddNode(end);

    ge::GraphUtils::AddEdge(contrl_op->GetOutControlAnchor(), tile1_node->GetInControlAnchor());
    ge::GraphUtils::AddEdge(contrl_op->GetOutControlAnchor(), tile2_node->GetInControlAnchor());
    ge::GraphUtils::AddEdge(contrl_op->GetOutControlAnchor(), tile3_node->GetInControlAnchor());

    ge::GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), tile1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), tile2_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data3_node->GetOutDataAnchor(0), tile3_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data4_node->GetOutDataAnchor(0), tile4_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data5_node->GetOutDataAnchor(0), tile5_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data6_node->GetOutDataAnchor(0), tile6_node->GetInDataAnchor(0));

    ge::GraphUtils::AddEdge(tile1_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(tile2_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(tile3_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(tile4_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(3));
    ge::GraphUtils::AddEdge(tile5_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(4));
    ge::GraphUtils::AddEdge(tile6_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(5));

    ge::GraphUtils::AddEdge(const_op->GetOutDataAnchor(0), concat_node->GetInDataAnchor(6));
    ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), end_node->GetInDataAnchor(0));
  }

  void InitGraph4(ComputeGraphPtr &graph) {
    OpDescPtr tile1 = std::make_shared<OpDesc>("tile0", TILED);
    OpDescPtr tile2 = std::make_shared<OpDesc>("tile1", TILED);
    OpDescPtr tile3 = std::make_shared<OpDesc>("tile2", TILED);

    OpDescPtr tile4 = std::make_shared<OpDesc>("tile3", TILED);
    OpDescPtr tile5 = std::make_shared<OpDesc>("tile4", TILED);
    OpDescPtr tile6 = std::make_shared<OpDesc>("tile5", TILED);
    ge::AttrUtils::SetListInt(tile1, "multiples", vector<int64_t>{128, 1,1,1});
    ge::AttrUtils::SetListInt(tile2, "multiples", vector<int64_t>{128, 1,1,1});
    ge::AttrUtils::SetListInt(tile3, "multiples", vector<int64_t>{128, 1,1,1});

    ge::AttrUtils::SetListInt(tile4, "multiples", vector<int64_t>{128, 1,1,1});
    ge::AttrUtils::SetListInt(tile5, "multiples", vector<int64_t>{128, 1,1,1});
    ge::AttrUtils::SetListInt(tile6, "multiples", vector<int64_t>{128, 1,1,1});

    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATV2);
    OpDescPtr end = std::make_shared<OpDesc>("end", "End");

    OpDescPtr data1 = std::make_shared<OpDesc>("data0", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr data3 = std::make_shared<OpDesc>("data2", "Data");

    OpDescPtr data4 = std::make_shared<OpDesc>("data3", "Data");
    OpDescPtr data5 = std::make_shared<OpDesc>("data4", "Data");
    OpDescPtr data6 = std::make_shared<OpDesc>("data5", "Data");
    OpDescPtr data7 = std::make_shared<OpDesc>("data6", "Data");

    OpDescPtr const_data = std::make_shared<OpDesc>("const", "Constant");
    OpDescPtr contrl_data = std::make_shared<OpDesc>("contrlconst", "Constant");

    // add descriptor
    ge::GeShape shape0({1, 6, 1, 1});
    GeTensorDesc input_desc0(shape0, ge::FORMAT_NCHW, ge::DT_FLOAT);
    input_desc0.SetOriginShape(shape0);
    tile1->AddInputDesc(input_desc0); 
    tile2->AddInputDesc(input_desc0);
    tile3->AddInputDesc(input_desc0);

    data1->AddOutputDesc(input_desc0);
    data2->AddOutputDesc(input_desc0);
    data3->AddOutputDesc(input_desc0);

    ge::GeShape shape_0({1, 6, 1, 1});
    GeTensorDesc input_desc_0(shape_0, ge::FORMAT_NCHW, ge::DT_FLOAT);
    input_desc_0.SetOriginShape(shape_0);
    tile4->AddInputDesc(input_desc_0);
    tile5->AddInputDesc(input_desc_0);
    tile6->AddInputDesc(input_desc_0);

    data4->AddOutputDesc(input_desc_0);
    data5->AddOutputDesc(input_desc_0);
    data6->AddOutputDesc(input_desc_0);

    ge::GeShape shape1({128, 6, 6, 1});
    GeTensorDesc out_desc(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc.SetOriginShape(shape1);
    tile1->AddOutputDesc(out_desc);
    tile2->AddOutputDesc(out_desc);
    tile3->AddOutputDesc(out_desc);

    tile4->AddOutputDesc(out_desc);
    tile5->AddOutputDesc(out_desc);
    tile6->AddOutputDesc(out_desc);

    concat->AddInputDesc(out_desc);
    concat->AddInputDesc(out_desc);
    concat->AddInputDesc(out_desc);

    concat->AddInputDesc(out_desc);
    concat->AddInputDesc(out_desc);
    concat->AddInputDesc(out_desc);
    concat->AddInputDesc(out_desc);
    data7->AddOutputDesc(out_desc);


    ge::GeShape shape2({128, 6, 42, 7});
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);
    concat->AddOutputDesc(out_desc2);
    end->AddInputDesc(out_desc2);

    ge::GeShape shape3({1});
    GeTensorDesc out_desc3(shape2, ge::FORMAT_ND, ge::DT_INT32);
    concat->AddInputDesc(out_desc3);
    const_data->AddOutputDesc(out_desc3);


    NodePtr tile1_node = graph->AddNode(tile1);
    NodePtr tile2_node = graph->AddNode(tile2);
    NodePtr tile3_node = graph->AddNode(tile3);
    NodePtr tile4_node = graph->AddNode(tile4);
    NodePtr tile5_node = graph->AddNode(tile5);
    NodePtr tile6_node = graph->AddNode(tile6);

    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr data3_node = graph->AddNode(data3);
    NodePtr data4_node = graph->AddNode(data4);
    NodePtr data5_node = graph->AddNode(data5);
    NodePtr data6_node = graph->AddNode(data6);
    NodePtr data7_node = graph->AddNode(data7);


    NodePtr concat_node = graph->AddNode(concat);
    NodePtr const_op = graph->AddNode(const_data);
    NodePtr contrl_op = graph->AddNode(contrl_data);
    
    NodePtr end_node = graph->AddNode(end);

    ge::GraphUtils::AddEdge(contrl_op->GetOutControlAnchor(), tile1_node->GetInControlAnchor());
    ge::GraphUtils::AddEdge(contrl_op->GetOutControlAnchor(), tile2_node->GetInControlAnchor());
    ge::GraphUtils::AddEdge(contrl_op->GetOutControlAnchor(), tile3_node->GetInControlAnchor());

    ge::GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), tile1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), tile2_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data3_node->GetOutDataAnchor(0), tile3_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data4_node->GetOutDataAnchor(0), tile4_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data5_node->GetOutDataAnchor(0), tile5_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data6_node->GetOutDataAnchor(0), tile6_node->GetInDataAnchor(0));

    ge::GraphUtils::AddEdge(tile1_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(tile2_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(tile3_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(data7_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(3));
    ge::GraphUtils::AddEdge(tile4_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(4));
    ge::GraphUtils::AddEdge(tile5_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(5));
    ge::GraphUtils::AddEdge(tile6_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(6));

    ge::GraphUtils::AddEdge(const_op->GetOutDataAnchor(0), concat_node->GetInDataAnchor(7));
    ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), end_node->GetInDataAnchor(0));
  }

  void InitGraph6(ComputeGraphPtr &graph, bool flag=false) {
    OpDescPtr tile1 = std::make_shared<OpDesc>("tile0", TILED);
    OpDescPtr tile2 = std::make_shared<OpDesc>("tile1", TILED);
    OpDescPtr tile3 = std::make_shared<OpDesc>("tile2", TILE);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATV2);
    OpDescPtr end = std::make_shared<OpDesc>("end", "End");
    OpDescPtr end1 = std::make_shared<OpDesc>("end1", "End");
    ge::AttrUtils::SetListInt(tile1, "multiples", vector<int64_t>{128, 1, 1, 4});
    ge::AttrUtils::SetListInt(tile2, "multiples", vector<int64_t>{128, 1, 1 ,4});
    ge::AttrUtils::SetListInt(tile3, "multiples", vector<int64_t>{128, 1, 1, 4});

    OpDescPtr data1 = std::make_shared<OpDesc>("data0", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr data3 = std::make_shared<OpDesc>("data2", "Data");

    OpDescPtr const_data = std::make_shared<OpDesc>("const", "Constant");

    // add descriptor
    ge::GeShape shape0({1, 6, 1, 1});
    GeTensorDesc input_desc0(shape0, ge::FORMAT_NCHW, ge::DT_FLOAT);
    input_desc0.SetOriginShape(shape0);
    tile1->AddInputDesc(input_desc0);
    tile2->AddInputDesc(input_desc0);
    tile3->AddInputDesc(input_desc0);
    data1->AddOutputDesc(input_desc0);
    data2->AddOutputDesc(input_desc0);
    data3->AddOutputDesc(input_desc0);

    ge::GeShape shape1({4, 6, 4, 4});
    GeTensorDesc out_desc(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc.SetOriginShape(shape1);
    tile1->AddOutputDesc(out_desc);
    tile2->AddOutputDesc(out_desc);
    tile3->AddOutputDesc(out_desc);

    concat->AddInputDesc(out_desc);
    concat->AddInputDesc(out_desc);
    concat->AddInputDesc(out_desc);


    ge::GeShape shape2({128, 6, 3, 8});
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);
    concat->AddOutputDesc(out_desc2);
    end->AddInputDesc(out_desc2);
    end1->AddInputDesc(out_desc2);

    ge::GeShape shape3({1});
    GeTensorDesc out_desc3(shape2, ge::FORMAT_ND, ge::DT_INT32);
    concat->AddInputDesc(out_desc3);
    const_data->AddOutputDesc(out_desc3);


    NodePtr tile1_node = graph->AddNode(tile1);
    NodePtr tile2_node = graph->AddNode(tile2);
    NodePtr tile3_node = graph->AddNode(tile3);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr data3_node = graph->AddNode(data3);
    NodePtr concat_node = graph->AddNode(concat);
    NodePtr const_op = graph->AddNode(const_data);
    
    
    NodePtr end_node = graph->AddNode(end);
    NodePtr end_node1 = graph->AddNode(end1);

    ge::GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), tile1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), tile2_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data3_node->GetOutDataAnchor(0), tile3_node->GetInDataAnchor(0));

    ge::GraphUtils::AddEdge(tile1_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(tile2_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(tile3_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(2));
    if (flag) {
      ge::GraphUtils::AddEdge(tile3_node->GetOutDataAnchor(0), end_node1->GetInDataAnchor(0));
    }
    ge::GraphUtils::AddEdge(const_op->GetOutDataAnchor(0), concat_node->GetInDataAnchor(3));
    ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), end_node->GetInDataAnchor(0));
  }

  void InitGraph7(ComputeGraphPtr &graph) {
    OpDescPtr tile1 = std::make_shared<OpDesc>("tile0", TILED);
    OpDescPtr tile2 = std::make_shared<OpDesc>("tile1", TILED);
    OpDescPtr tile3 = std::make_shared<OpDesc>("tile2", TILED);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCAT);
    OpDescPtr end = std::make_shared<OpDesc>("end", "End");
    ge::AttrUtils::SetListInt(tile1, "multiples", vector<int64_t>{128, 1,1,1});
    ge::AttrUtils::SetListInt(tile2, "multiples", vector<int64_t>{128, 1,1,1});
    ge::AttrUtils::SetListInt(tile3, "multiples", vector<int64_t>{128, 1,1,1});

    OpDescPtr data1 = std::make_shared<OpDesc>("data0", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr data3 = std::make_shared<OpDesc>("data2", "Data");

    OpDescPtr const_data = std::make_shared<OpDesc>("const", "Constant");

    // add descriptor
    ge::GeShape shape0({1, 6, 1, 1});
    GeTensorDesc input_desc0(shape0, ge::FORMAT_NCHW, ge::DT_FLOAT);
    input_desc0.SetOriginShape(shape0);
    tile1->AddInputDesc(input_desc0);
    tile2->AddInputDesc(input_desc0);
    tile3->AddInputDesc(input_desc0);
    data1->AddOutputDesc(input_desc0);
    data2->AddOutputDesc(input_desc0);
    data3->AddOutputDesc(input_desc0);

    ge::GeShape shape1({128, 6, 1, 1});
    GeTensorDesc out_desc(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc.SetOriginShape(shape1);
    tile1->AddOutputDesc(out_desc);
    tile2->AddOutputDesc(out_desc);
    tile3->AddOutputDesc(out_desc);

    concat->AddInputDesc(out_desc);
    concat->AddInputDesc(out_desc);
    concat->AddInputDesc(out_desc);


    ge::GeShape shape2({128, 6, 3, 1});
    GeTensorDesc out_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    out_desc2.SetOriginShape(shape2);
    concat->AddOutputDesc(out_desc2);
    end->AddInputDesc(out_desc2);

    ge::GeShape shape3({1});
    GeTensorDesc out_desc3(shape2, ge::FORMAT_ND, ge::DT_INT32);
    concat->AddInputDesc(out_desc3);
    const_data->AddOutputDesc(out_desc3);


    NodePtr tile1_node = graph->AddNode(tile1);
    NodePtr tile2_node = graph->AddNode(tile2);
    NodePtr tile3_node = graph->AddNode(tile3);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr data3_node = graph->AddNode(data3);
    NodePtr concat_node = graph->AddNode(concat);
    NodePtr const_op = graph->AddNode(const_data);
    
    
    NodePtr end_node = graph->AddNode(end);

    ge::GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), tile1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), tile2_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data3_node->GetOutDataAnchor(0), tile3_node->GetInDataAnchor(0));

    ge::GraphUtils::AddEdge(tile1_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(tile2_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(tile3_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(3));
    ge::GraphUtils::AddEdge(const_op->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), end_node->GetInDataAnchor(0));
  }

static void DumpGraph(const ge::ComputeGraphPtr graph, string graph_name) {
    printf("start to dump graph %s...\n", graph_name.c_str());
    for (ge::NodePtr node : graph->GetAllNodes()) {
      printf("node name = %s.\n", node->GetName().c_str());
      for (ge::OutDataAnchorPtr anchor : node->GetAllOutDataAnchors()) {
        for (ge::InDataAnchorPtr peer_in_anchor : anchor->GetPeerInDataAnchors()) {
          printf("    node name = %s[%d], out data node name = %s[%d].\n",
                 node->GetName().c_str(),
                 anchor->GetIdx(),
                 peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                 peer_in_anchor->GetIdx());
        }
      }
      if (node->GetOutControlAnchor() != nullptr) {
        for (ge::InControlAnchorPtr peer_in_anchor : node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
          printf("    node name = %s, out control node name = %s.\n", node->GetName().c_str(),
                 peer_in_anchor->GetOwnerNode()->GetName().c_str());
        }
      }
    }

    return;
  }
};


TEST_F(fusion_concat_tile_fusion_cast_ut, test_ParseConcatNode) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph7(graph);
  ConcatTileFusionPass pass;
  vector<ge::NodePtr> fusion_nodes = {};
  PatternFusionBasePass::Mapping mapping;
  NodePtr node1 = graph->FindNode("concat");
  Status status= pass.ParseConcatNode(node1);
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(fusion_concat_tile_fusion_cast_ut, concat_tile_not_changed0) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph(graph);
  ConcatTileFusionPass pass;
  vector<ge::NodePtr> fusion_nodes = {};
  PatternFusionBasePass::Mapping mapping;
  std::shared_ptr<FusionPattern::OpDesc> op_desc1 = std::make_shared<FusionPattern::OpDesc>();
  op_desc1->id = "concat";
  NodePtr node1 = graph->FindNode("concat");
  mapping[op_desc1] = std::vector<ge::NodePtr>{node1};
  Status status= pass.Fusion(*graph, mapping, fusion_nodes);
  DumpGraph(graph, "test");
  EXPECT_EQ(fe::NOT_CHANGED, status);
}

TEST_F(fusion_concat_tile_fusion_cast_ut, concat_tile_Success0) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph1(graph);
  ConcatTileFusionPass pass;
  vector<ge::NodePtr> fusion_nodes = {};
  PatternFusionBasePass::Mapping mapping;
  std::shared_ptr<FusionPattern::OpDesc> op_desc1 = std::make_shared<FusionPattern::OpDesc>();
  op_desc1->id = "concat";
  NodePtr node1 = graph->FindNode("concat");
  mapping[op_desc1] = std::vector<ge::NodePtr>{node1};
  Status status= pass.Fusion(*graph, mapping, fusion_nodes);
  DumpGraph(graph, "test");
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(fusion_concat_tile_fusion_cast_ut, concat_tile_not_changed1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph2(graph);
  ConcatTileFusionPass pass;
  vector<ge::NodePtr> fusion_nodes = {};
  PatternFusionBasePass::Mapping mapping;
  std::shared_ptr<FusionPattern::OpDesc> op_desc1 = std::make_shared<FusionPattern::OpDesc>();
  op_desc1->id = "concat";
  NodePtr node1 = graph->FindNode("concat");
  mapping[op_desc1] = std::vector<ge::NodePtr>{node1};
  Status status= pass.Fusion(*graph, mapping, fusion_nodes);
  DumpGraph(graph, "test");
  EXPECT_EQ(fe::NOT_CHANGED, status);
}

TEST_F(fusion_concat_tile_fusion_cast_ut, concat_tile_Success1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph3(graph);
  ConcatTileFusionPass pass;
  vector<ge::NodePtr> fusion_nodes = {};
  PatternFusionBasePass::Mapping mapping;
  std::shared_ptr<FusionPattern::OpDesc> op_desc1 = std::make_shared<FusionPattern::OpDesc>();
  op_desc1->id = "concat";
  NodePtr node1 = graph->FindNode("concat");
  mapping[op_desc1] = std::vector<ge::NodePtr>{node1};
  Status status= pass.Fusion(*graph, mapping, fusion_nodes);
  DumpGraph(graph, "test");
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(fusion_concat_tile_fusion_cast_ut, concat_tile_Success2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph4(graph);
  ConcatTileFusionPass pass;
  vector<ge::NodePtr> fusion_nodes = {};
  PatternFusionBasePass::Mapping mapping;
  std::shared_ptr<FusionPattern::OpDesc> op_desc1 = std::make_shared<FusionPattern::OpDesc>();
  op_desc1->id = "concat";
  NodePtr node1 = graph->FindNode("concat");
  mapping[op_desc1] = std::vector<ge::NodePtr>{node1};
  Status status= pass.Fusion(*graph, mapping, fusion_nodes);
  DumpGraph(graph, "test");
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(fusion_concat_tile_fusion_cast_ut, concat_tile_Failed01) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph6(graph);
  ConcatTileFusionPass pass;
  vector<ge::NodePtr> fusion_nodes = {};
  PatternFusionBasePass::Mapping mapping;
  std::shared_ptr<FusionPattern::OpDesc> op_desc1 = std::make_shared<FusionPattern::OpDesc>();
  op_desc1->id = "concat";
  NodePtr node1 = graph->FindNode("concat");
  mapping[op_desc1] = std::vector<ge::NodePtr>{node1};
  Status status= pass.Fusion(*graph, mapping, fusion_nodes);
  DumpGraph(graph, "test");
}

TEST_F(fusion_concat_tile_fusion_cast_ut, concat_tile_Failed02) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(GRAPH_NAME);
  InitGraph6(graph, true);
  ConcatTileFusionPass pass;
  ge::NodePtr concat_node = graph->FindNode("concat");
  ge::NodePtr tile_node = graph->FindNode("tile2");
  vector<ge::NodePtr> input_nodes;
  input_nodes.emplace_back(tile_node);
  Status status = pass.RemoveConcatEdges(concat_node, input_nodes);
  EXPECT_EQ(status, fe::FAILED);
}

}
