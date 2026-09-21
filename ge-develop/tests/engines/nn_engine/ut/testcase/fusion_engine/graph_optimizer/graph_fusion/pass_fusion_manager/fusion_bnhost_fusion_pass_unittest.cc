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


#include <memory>
#include <string>
#include <vector>

#define protected public
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "pass_manager.h"
#include "common/util/op_info_util.h"
#include "graph/op_desc.h"
#undef protected
#undef private

namespace fe {
  static const string EPSILON = "epsilon";
  static const string USE_GLOBAL_STATS = "use_global_stats";
  static const string MODE = "mode";
  class STEST_optimizer_fusion_bnhost_fusion_pass : public testing::Test {

  protected:
    void SetUp() {
    }

    void TearDown() {

    }

  protected:
    ge::NodePtr AddNode(ge::ComputeGraphPtr graph, const string &name, const string &type,
                        int32_t out_anchors_num = 1, int32_t in_anchors_num = 1) {
      ge::GeTensorDesc tensor_desc;
      ge::OpDescPtr opdesc = std::make_shared<ge::OpDesc>(name, type);
      for (int32_t i = 0; i < out_anchors_num; i++) {
        opdesc->AddOutputDesc(tensor_desc);
      }
      for (int32_t i = 0; i < in_anchors_num; i++) {
        opdesc->AddInputDesc(tensor_desc);
      }
      ge::NodePtr node = graph->AddNode(opdesc);
      return node;
    }

    ge::ComputeGraphPtr CreateSucessGraph() {
      ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
      ge::NodePtr data_node = AddNode(graph, "data", fe::DATA);
      ge::NodePtr mean_node = AddNode(graph, "mean", fe::CONSTANT);
      ge::NodePtr var_node = AddNode(graph, "variance", fe::CONSTANT);
      ge::NodePtr mu_node = AddNode(graph, "momentum", fe::CONSTANT);
      ge::NodePtr bninference_node = AddNode(graph, "bninference", "BNInference",1,4);
      ge::NodePtr out_node = AddNode(graph, "out", "Upsample");
      std::vector<int64_t> shape_vec{1, 32, 4, 4};
      ge::GeShape shape_desc = ge::GeShape(shape_vec);
      std::vector<int64_t> shape_mean{32};
      ge::GeShape shapemean = ge::GeShape(shape_mean);
      std::vector<int64_t> shape_mu{1};
      ge::GeShape shapemu = ge::GeShape(shape_mu);
      data_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(shape_desc);
      data_node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
      data_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT16);
      mean_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(shapemean);
      mean_node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
      mean_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT16);
      var_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(shapemean);
      var_node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
      var_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT16);
      mu_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(shapemu);
      mu_node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
      mu_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT16);
      bninference_node->GetOpDesc()->MutableInputDesc(0)->SetShape(shape_desc);
      bninference_node->GetOpDesc()->MutableInputDesc(0)->SetFormat(ge::FORMAT_NCHW);
      bninference_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_FLOAT16);
      bninference_node->GetOpDesc()->MutableInputDesc(1)->SetShape(shapemean);
      bninference_node->GetOpDesc()->MutableInputDesc(1)->SetFormat(ge::FORMAT_NCHW);
      bninference_node->GetOpDesc()->MutableInputDesc(1)->SetDataType(ge::DT_FLOAT16);
      bninference_node->GetOpDesc()->MutableInputDesc(2)->SetShape(shapemean);
      bninference_node->GetOpDesc()->MutableInputDesc(2)->SetFormat(ge::FORMAT_NCHW);
      bninference_node->GetOpDesc()->MutableInputDesc(2)->SetDataType(ge::DT_FLOAT16);
      bninference_node->GetOpDesc()->MutableInputDesc(3)->SetShape(shapemu);
      bninference_node->GetOpDesc()->MutableInputDesc(3)->SetFormat(ge::FORMAT_NCHW);
      bninference_node->GetOpDesc()->MutableInputDesc(3)->SetDataType(ge::DT_FLOAT16);
      bninference_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(shape_desc);
      bninference_node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
      bninference_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT16);
      float eps = 0.00001;
      ge::AttrUtils::SetFloat(bninference_node->GetOpDesc(), EPSILON, eps);
      bool use_global_stats_ = true;
      ge::AttrUtils::SetBool(bninference_node->GetOpDesc(), USE_GLOBAL_STATS, use_global_stats_);
      int mode_ = 1;
      ge::AttrUtils::SetInt(bninference_node->GetOpDesc(), MODE, mode_);
      out_node->GetOpDesc()->MutableInputDesc(0)->SetShape(shape_desc);
      out_node->GetOpDesc()->MutableInputDesc(0)->SetFormat(ge::FORMAT_NCHW);
      out_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_FLOAT16);
      ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), bninference_node->GetInDataAnchor(0));
      ge::GraphUtils::AddEdge(mean_node->GetOutDataAnchor(0), bninference_node->GetInDataAnchor(1));
      ge::GraphUtils::AddEdge(var_node->GetOutDataAnchor(0), bninference_node->GetInDataAnchor(2));
      ge::GraphUtils::AddEdge(mu_node->GetOutDataAnchor(0), bninference_node->GetInDataAnchor(3));
      ge::GraphUtils::AddEdge(bninference_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
      return graph;
    }

    ge::ComputeGraphPtr CreateSucessGraph2() {
      ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
      ge::NodePtr data_node = AddNode(graph, "data", fe::DATA);
      ge::NodePtr mean_node = AddNode(graph, "mean", fe::CONSTANT);
      ge::NodePtr var_node = AddNode(graph, "variance", fe::CONSTANT);
      ge::NodePtr mu_node = AddNode(graph, "momentum", fe::CONSTANT);
      ge::NodePtr bninference_node = AddNode(graph, "bninference", "BNInference",1,4);
      ge::NodePtr out_node = AddNode(graph, "out", "Upsample");
      std::vector<int64_t> shape_vec{1, 32, 4, 4};
      ge::GeShape shape_desc = ge::GeShape(shape_vec);
      std::vector<int64_t> shape_mean{32};
      ge::GeShape shapemean = ge::GeShape(shape_mean);
      std::vector<int64_t> shape_mu{1};
      ge::GeShape shapemu = ge::GeShape(shape_mu);
      data_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(shape_desc);
      data_node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
      data_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
      mean_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(shapemean);
      mean_node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
      mean_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
      var_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(shapemean);
      var_node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
      var_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
      mu_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(shapemu);
      mu_node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
      mu_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
      bninference_node->GetOpDesc()->MutableInputDesc(0)->SetShape(shape_desc);
      bninference_node->GetOpDesc()->MutableInputDesc(0)->SetFormat(ge::FORMAT_NCHW);
      bninference_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_FLOAT);
      bninference_node->GetOpDesc()->MutableInputDesc(1)->SetShape(shapemean);
      bninference_node->GetOpDesc()->MutableInputDesc(1)->SetFormat(ge::FORMAT_NCHW);
      bninference_node->GetOpDesc()->MutableInputDesc(1)->SetDataType(ge::DT_FLOAT);
      bninference_node->GetOpDesc()->MutableInputDesc(2)->SetShape(shapemean);
      bninference_node->GetOpDesc()->MutableInputDesc(2)->SetFormat(ge::FORMAT_NCHW);
      bninference_node->GetOpDesc()->MutableInputDesc(2)->SetDataType(ge::DT_FLOAT);
      bninference_node->GetOpDesc()->MutableInputDesc(3)->SetShape(shapemu);
      bninference_node->GetOpDesc()->MutableInputDesc(3)->SetFormat(ge::FORMAT_NCHW);
      bninference_node->GetOpDesc()->MutableInputDesc(3)->SetDataType(ge::DT_FLOAT);
      bninference_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(shape_desc);
      bninference_node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
      bninference_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
      float eps = 0.00001;
      ge::AttrUtils::SetFloat(bninference_node->GetOpDesc(), EPSILON, eps);
      bool use_global_stats_ = true;
      ge::AttrUtils::SetBool(bninference_node->GetOpDesc(), USE_GLOBAL_STATS, use_global_stats_);
      int mode_ = 1;
      ge::AttrUtils::SetInt(bninference_node->GetOpDesc(), MODE, mode_);
        int index = 1;
        string dtindex = "_output_dt_index";
        ge::AttrUtils::SetInt(bninference_node->GetOpDesc(), dtindex, index);
      out_node->GetOpDesc()->MutableInputDesc(0)->SetShape(shape_desc);
      out_node->GetOpDesc()->MutableInputDesc(0)->SetFormat(ge::FORMAT_NCHW);
      out_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_FLOAT);
      ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), bninference_node->GetInDataAnchor(0));
      ge::GraphUtils::AddEdge(mean_node->GetOutDataAnchor(0), bninference_node->GetInDataAnchor(1));
      ge::GraphUtils::AddEdge(var_node->GetOutDataAnchor(0), bninference_node->GetInDataAnchor(2));
      ge::GraphUtils::AddEdge(mu_node->GetOutDataAnchor(0), bninference_node->GetInDataAnchor(3));
      ge::GraphUtils::AddEdge(bninference_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
      return graph;
    }

  };
}
