/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "utils/model_factory.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "common/types.h"
#include "graph/model.h"
#include "graph/model_serialize.h"
#include "ge_running_env/path_utils.h"
#include "ge_running_env/tensor_utils.h"
#include "mmpa/mmpa_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"

#include <unistd.h>

#define RETURN_WHEN_FOUND(name) do { \
      auto iter = model_names_to_path_.find(name); \
      if (iter != model_names_to_path_.end()) { \
        return iter->second; \
      }                                \
    } while(0)

FAKE_NS_BEGIN
namespace {
const std::string &GetModelPath() {
  static std::string path;
  if (!path.empty()) {
    return path;
  }
  path = PathJoin(GetRunPath().c_str(), "models");
  if (!IsDir(path.c_str())) {
    auto ret = mmMkdir(path.c_str(),
                       S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IWGRP | S_IXGRP | S_IROTH | S_IXOTH); // 775
    if (ret != EN_OK) {
      path = "";
    }
  }
  return path;
}

}
std::unordered_map<std::string, std::string> ModelFactory::model_names_to_path_;
const std::string &ge::ModelFactory::GenerateModel_1(bool is_dynamic, const bool with_fusion) {
  const std::string name = "ms1_" + std::to_string(is_dynamic);
  RETURN_WHEN_FOUND(name);

  std::vector<int64_t> shape = {2,2,3,2};  // HWCN
  auto data_tensor = GenerateTensor(shape);

  DEF_GRAPH(dynamic_op) {
    ge::OpDescPtr data1;
    if (is_dynamic) {
      data1 = OP_CFG(DATA)
                  .Attr(ATTR_NAME_INDEX, 0)
                  .InCnt(1)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, 3, 16, 16})
                  .OutCnt(1)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, 3, 16, 16})
                  .Build("data1");

    } else {
      data1 = OP_CFG(DATA)
                  .Attr(ATTR_NAME_INDEX, 0)
                  .InCnt(1)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
                  .OutCnt(1)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
                  .Build("data1");
    }

    auto const1 = OP_CFG(CONSTANT)
                      .Weight(data_tensor)
                      .TensorDesc(FORMAT_HWCN, DT_FLOAT, shape)
                      .InCnt(1)
                      .OutCnt(1)
                      .Build("const1");

    auto conv2d1 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d1");

    auto relu1 = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("relu1");

    auto netoutput1 = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("netoutput1");

    CHAIN(NODE(data1)->NODE(conv2d1)->NODE(relu1))->NODE(netoutput1);
    CHAIN(NODE(const1)->EDGE(0, 1)->NODE(conv2d1));
  };
  auto graph = ToGeGraph(dynamic_op);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data = compute_graph->FindNode("data1");
  if (data != nullptr && data->GetOpDesc()->MutableInputDesc(0) != nullptr) {
    data->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({1,2,3,4,5}));
    data->GetOpDesc()->MutableInputDesc(0)->SetOriginFormat(FORMAT_NC1HWC0);
    bool enable_storage_format_spread = true;
    (void)AttrUtils::SetBool(data->GetOpDesc(), "_enable_storage_format_spread", enable_storage_format_spread);
    bool is_origin_format_set = true;
    (void)AttrUtils::SetBool(data->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, is_origin_format_set);
  }
  return SaveAsModel(name, graph, with_fusion);
}

const std::string &ge::ModelFactory::GenerateModel_2(bool is_dynamic, const bool with_fusion) {
  const std::string name = "ms2_" + std::to_string(is_dynamic);
  RETURN_WHEN_FOUND(name);

  std::vector<int64_t> shape = {2,2,3,2};  // HWCN
  auto data_tensor = GenerateTensor(shape);
  // two input data
  DEF_GRAPH(dynamic_op) {
    ge::OpDescPtr data1;
    if (is_dynamic) {
      data1 = OP_CFG(DATA)
          .Attr(ATTR_NAME_INDEX, 0)
          .InCnt(1)
          .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, 3, 16, 16})
          .OutCnt(1)
          .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, 3, 16, 16})
          .Build("data1");

    } else {
      data1 = OP_CFG(DATA)
          .Attr(ATTR_NAME_INDEX, 0)
          .InCnt(1)
          .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
          .OutCnt(1)
          .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
          .Build("data1");
    }
    auto data2 = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 1)
        .InCnt(1)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
        .OutCnt(1)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
        .Build("data2");
    auto const1 = OP_CFG(CONSTANT)
        .Weight(data_tensor)
        .TensorDesc(FORMAT_HWCN, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Build("const1");

    auto conv2d1 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d1");

    auto relu1 = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("relu1");

    auto relu2 = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("relu2");

    auto netoutput1 = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("netoutput1");

    CHAIN(NODE(data1)->NODE(conv2d1)->NODE(relu1))->NODE(netoutput1);
    CHAIN(NODE(data2)->NODE(relu2)->EDGE(0, 1)->NODE(netoutput1));
    CHAIN(NODE(const1)->EDGE(0, 1)->NODE(conv2d1));
  };

  return SaveAsModel(name, ToGeGraph(dynamic_op), with_fusion);
}

const std::string &ge::ModelFactory::GenerateModel_switch(bool is_dynamic, const bool with_fusion) {
  const std::string name = "ms3_" + std::to_string(is_dynamic);
  RETURN_WHEN_FOUND(name);

  GeTensorDesc tensor_4_desc(ge::GeShape({-1,3,4,5}), FORMAT_NCHW, DT_INT32);
  GeTensorDesc tensor_1_desc(ge::GeShape({1}), FORMAT_ND, DT_BOOL);

  auto data1 = std::make_shared<OpDesc>("data1", DATA);
  data1->AddInputDesc(tensor_4_desc);
  data1->AddOutputDesc(tensor_4_desc);

  auto data2 = std::make_shared<OpDesc>("data2", DATA);
  data2->AddInputDesc(tensor_1_desc);
  data2->AddOutputDesc(tensor_1_desc);

  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE("switch", SWITCH)->EDGE(0, 0)->NODE("relu1", RELU)->
            NODE("merge", MERGE)->EDGE(0, 0)->NODE("relu3", RELU)->NODE("output", NETOUTPUT));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE("switch")->EDGE(1, 0)->NODE("relu2", RELU)->EDGE(0, 1)->
            NODE("merge")->EDGE(1, 0)->NODE("relu4", RELU)->NODE("output"));
  };

  return SaveAsModel(name, ToGeGraph(g1), with_fusion);
}

/*
 *+ -------------------------- +
 *|        |                  |
 *|  Subgraph00-NetOutput     |
 *|        |                  |
 *|   conv_node00             |                net_output_root
 *|     /    \                |                   |
 *|  data000 data001          |   + -------------------------------- +
 *+ --------------------------+   |              |                  |
 *                  \             |     Subgraph0-NetOutput         |
 *                   \            |           |          \          |
 *               subgraph00<-------partitioned_call00 conv_node1    | -------> (partitionedcall | subgraph0)
 *                                |    /     \          /   \       |
 *                                | data00 data01    data10 data11    |
 *                                + ------------------------------- +
 *                                  /    \            /    \
 *                                data1 data2       data3 data4
 */
const std::string &ge::ModelFactory::GenerateModel_4(bool is_dynamic, const bool with_fusion) {
  const std::string name = "StaticSubGraphAipp_" + std::to_string(is_dynamic);
  RETURN_WHEN_FOUND(name);

  // four input data
  ge::OpDescPtr partitioncall0;
  ge::OpDescPtr partitioncall00;
  ge::OpDescPtr net_output_root;
  ge::OpDescPtr net_output_subgraph0;
  ge::OpDescPtr net_output_subgraph00;
  DEF_GRAPH(root_graph_name) {
    auto data1 = OP_CFG(DATA)
                     .Attr(ATTR_NAME_INDEX, 0)
                     .InCnt(1)
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                     .OutCnt(1)
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                     .Build("data1");
    auto data2 = OP_CFG(DATA)
                     .Attr(ATTR_NAME_INDEX, 1)
                     .InCnt(1)
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                     .OutCnt(1)
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                     .Build("data2");
    auto data3 = OP_CFG(DATA)
                     .Attr(ATTR_NAME_INDEX, 2)
                     .InCnt(1)
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                     .OutCnt(1)
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                     .Build("data3");
    auto data4 = OP_CFG(DATA)
                     .Attr(ATTR_NAME_INDEX, 3)
                     .InCnt(1)
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                     .OutCnt(1)
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                     .Build("data4");

    partitioncall0 = OP_CFG(PARTITIONEDCALL).InCnt(4).OutCnt(1).Build("partitioncall0");
    net_output_root = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("net_output_root");

    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(partitioncall0));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(partitioncall0));
    CHAIN(NODE(data3)->EDGE(0, 2)->NODE(partitioncall0));
    CHAIN(NODE(data4)->EDGE(0, 3)->NODE(partitioncall0));
    CHAIN(NODE(partitioncall0)->EDGE(0, 0)->NODE(net_output_root));
  };

  DEF_GRAPH(subgraph0_name) {
    auto data00 = OP_CFG(DATA)
                      .Attr(ATTR_NAME_INDEX, 0)
                      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                      .InCnt(1)
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                      .OutCnt(1)
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                      .Build("data00");
    auto data01 = OP_CFG(DATA)
                      .Attr(ATTR_NAME_INDEX, 1)
                      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 1)
                      .InCnt(1)
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                      .OutCnt(1)
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                      .Build("data01");
    auto data10 = OP_CFG(DATA)
                      .Attr(ATTR_NAME_INDEX, 2)
                      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 2)
                      .InCnt(1)
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                      .OutCnt(1)
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                      .Build("data10");
    auto data11 = OP_CFG(DATA)
                      .Attr(ATTR_NAME_INDEX, 3)
                      .Attr(ATTR_NAME_PARENT_NODE_INDEX, 3)
                      .InCnt(1)
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                      .OutCnt(1)
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                      .Build("data11");
    partitioncall00 = OP_CFG(PARTITIONEDCALL).InCnt(2).OutCnt(1).Build("partitioncall00");
    auto conv_node0 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv_node0");
    net_output_subgraph0 = OP_CFG(NETOUTPUT)
                               .InCnt(2)
                               .OutCnt(1)
                               .Build("net_output_subgraph0");
    for (size_t i = 0U; i < net_output_subgraph0->GetInputsSize(); ++i) {
      AttrUtils::SetInt(net_output_subgraph0->MutableInputDesc(i), ATTR_NAME_PARENT_NODE_INDEX, 0);
    }
    CHAIN(NODE(data00)->EDGE(0, 0)->NODE(partitioncall00));
    CHAIN(NODE(data01)->EDGE(0, 1)->NODE(partitioncall00));
    CHAIN(NODE(data10)->EDGE(0, 0)->NODE(conv_node0));
    CHAIN(NODE(data11)->EDGE(0, 1)->NODE(conv_node0));
    CHAIN(NODE(partitioncall00)->EDGE(0, 0)->NODE(net_output_subgraph0));
    CHAIN(NODE(conv_node0)->EDGE(0, 1)->NODE(net_output_subgraph0));
  };

  DEF_GRAPH(subgraph00_name) {
    auto data000 = OP_CFG(DATA)
                       .Attr(ATTR_NAME_INDEX, 0)
                       .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                       .InCnt(1)
                       .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                       .OutCnt(1)
                       .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                       .Build("data000");
    auto data001 = OP_CFG(DATA)
                       .Attr(ATTR_NAME_INDEX, 1)
                       .Attr(ATTR_NAME_PARENT_NODE_INDEX, 1)
                       .InCnt(1)
                       .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                       .OutCnt(1)
                       .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128,3,224,224})
                       .Build("data001");

    auto conv_node00 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv_node00");
    net_output_subgraph00 = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("net_output_subgraph00");
    for (size_t i = 0U; i < net_output_subgraph00->GetInputsSize(); ++i) {
      AttrUtils::SetInt(net_output_subgraph00->MutableInputDesc(i), ATTR_NAME_PARENT_NODE_INDEX, 0);
    }
    CHAIN(NODE(data000)->EDGE(0, 0)->NODE(conv_node00));
    CHAIN(NODE(data001)->EDGE(0, 1)->NODE(conv_node00));
    CHAIN(NODE(conv_node00)->EDGE(0, 0)->NODE(net_output_subgraph00));
  };

  Graph root_graph = ToGeGraph(root_graph_name);
  Graph subgraph0 = ToGeGraph(subgraph0_name);
  Graph subgraph00 = ToGeGraph(subgraph00_name);

  ComputeGraphPtr comput_root_graph = ge::GraphUtilsEx::GetComputeGraph(root_graph);
  ComputeGraphPtr comput_subgraph0 = ge::GraphUtilsEx::GetComputeGraph(subgraph0);
  ComputeGraphPtr comput_subgraph00 = ge::GraphUtilsEx::GetComputeGraph(subgraph00);

  NodePtr partitioncall0_node = comput_root_graph->FindFirstNodeMatchType(PARTITIONEDCALL);
  NodePtr partitioncall00_node = comput_subgraph0->FindFirstNodeMatchType(PARTITIONEDCALL);

  partitioncall00_node->GetOpDesc()->AddSubgraphName(comput_subgraph00->GetName());
  partitioncall00_node->GetOpDesc()->SetSubgraphInstanceName(0, comput_subgraph00->GetName());
  comput_subgraph00->SetParentNode(partitioncall00_node);
  comput_subgraph00->SetParentGraph(comput_subgraph0);
  comput_root_graph->AddSubgraph(comput_subgraph00);

  partitioncall0_node->GetOpDesc()->AddSubgraphName(comput_subgraph0->GetName());
  partitioncall0_node->GetOpDesc()->SetSubgraphInstanceName(0, comput_subgraph0->GetName());
  comput_subgraph0->SetParentNode(partitioncall0_node);
  comput_subgraph0->SetParentGraph(comput_root_graph);
  comput_root_graph->AddSubgraph(comput_subgraph0);

  return SaveAsModel(name, root_graph, with_fusion);
}

/**
 *             data
 *              |
 *            netouput
 */
const std::string &ge::ModelFactory::GenerateModel_data_to_netoutput(bool is_dynamic, const bool with_fusion) {
  const std::string name = "ms_data_to_netoutput" + std::to_string(is_dynamic);
  RETURN_WHEN_FOUND(name);

  std::vector<int64_t> shape = {2,2,3,2};  // HWCN
  auto data_tensor = GenerateTensor(shape);

  DEF_GRAPH(dynamic_op) {
    ge::OpDescPtr data1;
    if (is_dynamic) {
      data1 = OP_CFG(DATA)
                  .Attr(ATTR_NAME_INDEX, 0)
                  .InCnt(1)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, 3, 16, 16})
                  .OutCnt(1)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, 3, 16, 16})
                  .Build("data1");

    } else {
      data1 = OP_CFG(DATA)
                  .Attr(ATTR_NAME_INDEX, 0)
                  .InCnt(1)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
                  .OutCnt(1)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
                  .Build("data1");
    }

    auto netoutput1 = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("netoutput1");

    CHAIN(NODE(data1)->NODE(netoutput1));
  };

  return SaveAsModel(name, ToGeGraph(dynamic_op), with_fusion);
}

const std::string &ge::ModelFactory::GenerateModel_refdata(bool is_dynamic, const bool with_fusion) {
  (void)is_dynamic;
  (void)with_fusion;
  const std::string name = "ms_ref_data" + std::to_string(is_dynamic);
  RETURN_WHEN_FOUND(name);

  std::vector<int64_t> shape = {2, 2, 3, 2};  // HWCN
  auto data_tensor = GenerateTensor(shape);

  DEF_GRAPH(dynamic_op) {
    auto ref_data1 = OP_CFG(REFDATA)
                         .Attr(ATTR_NAME_INDEX, 1)
                         .InCnt(1)
                         .OutCnt(1)
                         .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
                         .Build("ref_data1");

    auto data1 = OP_CFG(DATA)
                     .Attr(ATTR_NAME_INDEX, 0)
                     .InCnt(1)
                     .OutCnt(1)
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
                     .Build("data1");

    auto add1 = OP_CFG(ADD).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16}).Build("add1");

    auto netoutput1 = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("netoutput1");

    CHAIN(NODE(ref_data1)->NODE(add1)->NODE(netoutput1));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(add1)->NODE(netoutput1));
  };

  return SaveAsModel(name, ToGeGraph(dynamic_op), with_fusion);
}


const std::string &ModelFactory::SaveAsModel(const string &name, const Graph &graph, const bool with_fusion) {
  Model model;
  model.SetName(name);
  model.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  model.SetVersion(1);

  if (with_fusion) {
    std::vector<Buffer> b_list(2U);
    ge::proto::OpDef fusion_op_def;
    auto &fusion_attr_map = *fusion_op_def.mutable_attr();
    fusion_attr_map["fusion_scope"].set_i(0x00010000);
    const auto group_op_l1_id = fusion_op_def.SerializeAsString();
    b_list[0] = Buffer::CopyFrom(reinterpret_cast<const uint8_t *>(group_op_l1_id.data()), group_op_l1_id.length());
    fusion_attr_map["fusion_scope"].set_i(0x00000001);
    const auto group_op_ub_id = fusion_op_def.SerializeAsString();
    b_list[1] = Buffer::CopyFrom(reinterpret_cast<const uint8_t *>(group_op_ub_id.data()), group_op_ub_id.length());
    (void)AttrUtils::SetListBytes(model, MODEL_ATTR_FUSION_MODEL_DEF, b_list);
  }

  const auto &file_dir = GetModelPath();
  {
    const auto text_path = PathJoin(file_dir.c_str(), name.c_str()) + ".txt";
    if (IsFile(text_path.c_str())) {
      RemoveFile(text_path.c_str());
    }
    ModelSerialize serialize;
    proto::ModelDef model_def;
    (void)serialize.SerializeModel(model, false, model_def);
    (void)GraphUtils::WriteProtoToTextFile(model_def, text_path.c_str());
  }

  const auto file_path = PathJoin(file_dir.c_str(), name.c_str()) + ".pbtxt";
  if (IsFile(file_path.c_str())) {
    RemoveFile(file_path.c_str());
  }
  model.SaveToFile(file_path);

  return model_names_to_path_[name] = file_path;
}
FAKE_NS_END
