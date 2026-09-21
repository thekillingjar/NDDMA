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
#include <string>
#include "graph/ge_local_context.h"
#include "graph/ge_context.h"

#include "macro_utils/dt_public_scope.h"
#include "opskernel/ops_kernel_info_types.h"
#include "graph/passes/standard_optimize/same_transdata_breadth_fusion_pass.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "api/gelib/gelib.h"
#include "graph/operator_reg.h"

using namespace ge;
using namespace std;

namespace {
REG_OP(Cast)
.INPUT(x, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64,
                      DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,
                      DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32})) /* input tensor */
.OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64,
                       DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,
                       DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32})) /* output tensor */
.ATTR(dst_type, Int, 0)
.ATTR(truncate, Bool, false)
.OP_END_FACTORY_REG(Cast)

struct NodeIO {
  std::string node_nmae;
  std::vector<uint32_t> in_indexes;
  std::vector<uint32_t> out_indexes;
};

NodePtr FindNodeByName(const std::vector<NodePtr> &all_nodes, const std::string &node_name) {
  for (const auto &node : all_nodes) {
    if (node->GetName() == node_name) {
      return node;
    }
  }
  return nullptr;
}

bool CheckTensorDesc(const ComputeGraphPtr &root_graph, std::vector<NodeIO> &nodes) {
  std::vector<NodePtr> all_nodes;
  for (auto &node : root_graph->GetAllNodes()) {
    all_nodes.emplace_back(node);
  }

  for (const auto &node_io : nodes) {
    auto node = FindNodeByName(all_nodes, node_io.node_nmae);
    if (node == nullptr) {
      std::cout << "========================================" << std::endl;
      std::cout << node_io.node_nmae << " is not found." << std::endl;
      std::cout << "========================================" << std::endl;
      GE_DUMP(root_graph, "CheckTensorDesc_failed");
      return false;
    }
    for (const auto index : node_io.in_indexes) {
      if (index >= node->GetOpDesc()->GetInputsSize()) {
        std::cout << "========================================" << std::endl;
        std::cout << node_io.node_nmae << " input[" << index << "] is larger than actual inputs size: "
                  <<  node->GetOpDesc()->GetInputsSize() << std::endl;
        std::cout << "========================================" << std::endl;
        GE_DUMP(root_graph, "CheckTensorDesc_failed");
        return false;
      }
      const auto in_tensor_desc = node->GetOpDesc()->GetInputDesc(index);
      if (in_tensor_desc.GetFormat() != FORMAT_NCL) {
        std::cout << "========================================" << std::endl;
        std::cout << node_io.node_nmae << " input[" << index << "] format[" << in_tensor_desc.GetFormat()
                  << "] is different from [" << FORMAT_NCL << "]" << std::endl;
        std::cout << "========================================" << std::endl;
        GE_DUMP(root_graph, "CheckTensorDesc_failed");
        return false;
      }
    }
    for (const auto index : node_io.out_indexes) {
      if (index >= node->GetOpDesc()->GetOutputsSize()) {
        std::cout << "========================================" << std::endl;
        std::cout << node_io.node_nmae << " output[" << index << "] is larger than actual outputs size: "
                  <<  node->GetOpDesc()->GetOutputsSize() << std::endl;
        std::cout << "========================================" << std::endl;
        GE_DUMP(root_graph, "CheckTensorDesc_failed");
        return false;
      }
      const auto out_tensor_desc = node->GetOpDesc()->GetOutputDesc(index);
      if (out_tensor_desc.GetFormat() != FORMAT_NCL) {
        std::cout << "========================================" << std::endl;
        std::cout << node_io.node_nmae << " output[" << index << "] format[" << out_tensor_desc.GetFormat()
                  << "] is different from [" << FORMAT_NCL << "]" << std::endl;
        std::cout << "========================================" << std::endl;
        GE_DUMP(root_graph, "CheckTensorDesc_failed");
        return false;
      }
    }
  }
  return true;
}

bool CheckNotChangedTensorDesc(const ComputeGraphPtr &root_graph, std::vector<NodeIO> &nodes) {
  std::vector<NodePtr> all_nodes;
  for (auto &node : root_graph->GetAllNodes()) {
    all_nodes.emplace_back(node);
  }

  for (const auto &node_io : nodes) {
    auto node = FindNodeByName(all_nodes, node_io.node_nmae);
    if (node == nullptr) {
      std::cout << "========================================" << std::endl;
      std::cout << node_io.node_nmae << " is not found." << std::endl;
      std::cout << "========================================" << std::endl;
      GE_DUMP(root_graph, "CheckTensorDesc_failed");
      return false;
    }
    for (const auto index : node_io.in_indexes) {
      const auto in_tensor_desc = node->GetOpDesc()->GetInputDesc(index);
      if (in_tensor_desc.GetFormat() == FORMAT_NCL) {
        std::cout << "========================================" << std::endl;
        std::cout << node_io.node_nmae << " input[" << index << "] format[" << in_tensor_desc.GetFormat()
                  << "] should not be [" << FORMAT_NCL << "]" << std::endl;
        std::cout << "========================================" << std::endl;
        GE_DUMP(root_graph, "CheckTensorDesc_failed");
        return false;
      }
    }
    for (const auto index : node_io.out_indexes) {
      const auto out_tensor_desc = node->GetOpDesc()->GetOutputDesc(index);
      if (out_tensor_desc.GetFormat() == FORMAT_NCL) {
        std::cout << "========================================" << std::endl;
        std::cout << node_io.node_nmae << " output[" << index << "] format[" << out_tensor_desc.GetFormat()
                  << "] should not be [" << FORMAT_NCL << "]" << std::endl;
        std::cout << "========================================" << std::endl;
        GE_DUMP(root_graph, "CheckTensorDesc_failed");
        return false;
      }
    }
  }
  return true;
}

bool CheckNodesDataType(const ComputeGraphPtr &root_graph, std::vector<NodeIO> &nodes, const DataType type) {
  std::vector<NodePtr> all_nodes;
  for (auto &node : root_graph->GetAllNodes()) {
    all_nodes.emplace_back(node);
  }

  for (const auto &node_io : nodes) {
    auto node = FindNodeByName(all_nodes, node_io.node_nmae);
    if (node == nullptr) {
      std::cout << "========================================" << std::endl;
      std::cout << node_io.node_nmae << " is not found." << std::endl;
      std::cout << "========================================" << std::endl;
      GE_DUMP(root_graph, "CheckTensorDesc_failed");
      return false;
    }
    for (const auto index : node_io.in_indexes) {
      const auto in_tensor_desc = node->GetOpDesc()->GetInputDesc(index);
      if (in_tensor_desc.GetDataType() != type) {
        std::cout << "========================================" << std::endl;
        std::cout << node_io.node_nmae << " input[" << index << "] data_type[" << in_tensor_desc.GetDataType()
                  << "] should be [" << type << "]" << std::endl;
        std::cout << "========================================" << std::endl;
        GE_DUMP(root_graph, "CheckNodesDataType_failed");
        return false;
      }
    }
    for (const auto index : node_io.out_indexes) {
      const auto out_tensor_desc = node->GetOpDesc()->GetOutputDesc(index);
      if (out_tensor_desc.GetDataType() != type) {
        std::cout << "========================================" << std::endl;
        std::cout << node_io.node_nmae << " output[" << index << "] data_type[" << out_tensor_desc.GetDataType()
                  << "] should be [" << type << "]" << std::endl;
        std::cout << "========================================" << std::endl;
        GE_DUMP(root_graph, "CheckNodesDataType_failed");
        return false;
      }
    }
  }
  return true;
}

void SetDataTypeForNode(ComputeGraphPtr &graph, const std::vector<std::string> &node_names, const DataType type) {
  std::map<std::string, NodePtr> graph_all_nodes;
  for (const auto &node : graph->GetAllNodes()) {
    graph_all_nodes[node->GetName()] = node;
  }

  for (const auto &name : node_names) {
    const auto iter = graph_all_nodes.find(name);
    if (iter != graph_all_nodes.end()) {
      auto op_desc = iter->second->GetOpDesc();
      for (size_t i = 0U; i < op_desc->GetInputsSize(); ++i) {
        auto in_tensor_desc = op_desc->MutableInputDesc(i);
        if (in_tensor_desc != nullptr) {
          in_tensor_desc->SetDataType(type);
          in_tensor_desc->SetOriginDataType(type);
        }
      }
      for (size_t i = 0U; i < op_desc->GetOutputsSize(); ++i) {
        auto out_tensor_desc = op_desc->MutableOutputDesc(i);
        if (out_tensor_desc != nullptr) {
          out_tensor_desc->SetDataType(type);
          out_tensor_desc->SetOriginDataType(type);
        }
      }
    } else {
      std::cout << "SetDataTypeForNode can not find " << name << std::endl;
    }
  }
}

void SetDataTypeForNodeInputs(ComputeGraphPtr &graph, const std::vector<std::string> &node_names, const DataType type) {
  std::map<std::string, NodePtr> graph_all_nodes;
  for (const auto &node : graph->GetAllNodes()) {
    graph_all_nodes[node->GetName()] = node;
  }

  for (const auto &name : node_names) {
    const auto iter = graph_all_nodes.find(name);
    if (iter != graph_all_nodes.end()) {
      auto op_desc = iter->second->GetOpDesc();
      for (size_t i = 0U; i < op_desc->GetInputsSize(); ++i) {
        auto in_tensor_desc = op_desc->MutableInputDesc(i);
        if (in_tensor_desc != nullptr) {
          in_tensor_desc->SetDataType(type);
          in_tensor_desc->SetOriginDataType(type);
        }
      }
    } else {
      std::cout << "SetDataTypeForNodeInputs can not find " << name << std::endl;
    }
  }
}

using NodeOutIndex = std::pair<std::string, uint32_t>;
using UtPath = std::vector<NodeOutIndex>;
using UtPaths = std::vector<UtPath>;
bool CheckConnection(const ComputeGraphPtr &graph, std::vector<NodeOutIndex> &path) {
  for (size_t i = 0U; i < path.size() - 1U; ++i) {
    auto firt_node = graph->FindNode(path[i].first);
    if (firt_node == nullptr) {
      std::cout << "========================================" << std::endl;
      std::cout << path[i].first << " is not found." << std::endl;
      std::cout << "========================================" << std::endl;
      GE_DUMP(graph, "CheckConnection_failed");
      return false;
    }
    if (path[i].second >= firt_node->GetOutDataNodesAndAnchors().size()) {
      std::cout << "========================================" << std::endl;
      std::cout << path[i].first << " index: " << path[i].second << " is larger than  actrual output size: "
                << firt_node->GetOutDataNodesAndAnchors().size()
                << ", i: " << i << std::endl;
      std::cout << "========================================" << std::endl;
      GE_DUMP(graph, "CheckConnection_failed");
      return false;
    }
    auto first_out = firt_node->GetOutDataAnchor(path[i].second);
    if (first_out == nullptr) {
      std::cout << "========================================" << std::endl;
      std::cout << path[i].first << " index: " << path[i].second << " out_anchor is null. output size: "
                << firt_node->GetOutDataNodesAndAnchors().size()
                << ", i: " << i << std::endl;
      std::cout << "========================================" << std::endl;
      GE_DUMP(graph, "CheckConnection_failed");
      return false;
    }
    bool find = false;
    for (const auto &in_anchor : first_out->GetPeerInDataAnchors()) {
      if (in_anchor->GetOwnerNode()->GetName() == path[i + 1].first) {
        find = true;
      }
    }

    if (!find) {
      std::cout << "========================================" << std::endl;
      std::cout << path[i].first << "[" << path[i].second << "] is not connected to " << path[i + 1].first << std::endl;
      std::cout << "========================================" << std::endl;
      GE_DUMP(graph, "CheckConnection_failed");
      return false;
    }
  }
  return true;
}

bool CheckConnection(const ComputeGraphPtr &graph, std::vector<std::vector<NodeOutIndex>> &paths) {
  for (auto &path : paths) {
    if (!CheckConnection(graph, path)) {
      return false;
    }
  }
  return true;
}

bool SetTransDataTensorDesc(const ComputeGraphPtr &root_graph, const std::vector<std::string> &node_names, Format format = FORMAT_NCL) {
  GeTensorDesc tensor_desc{GeShape{{2022, 2023}}, format, DT_FLOAT16};
  std::map<std::string, NodePtr> all_transdata_map;
  for (auto &node : root_graph->GetAllNodes()) {
    if (node->GetType() == TRANSDATA) {
      all_transdata_map[node->GetName()] = node;
    }
  }
  for (const auto &node_name : node_names) {
    const auto iter = all_transdata_map.find(node_name);
    if (iter != all_transdata_map.end()) {
      iter->second->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
    } else {
      std::cout << "========================================" << std::endl;
      std::cout << "can not find " << node_name << std::endl;
      std::cout << "========================================" << std::endl;
      GE_DUMP(root_graph, "SetTransDataTensorDesc_failed");
      return false;
    }
  }
  return true;
}

using NetoutputParentIndexes = std::vector<std::pair<std::string, std::vector<uint32_t>>>;
bool AddParentIndexForNetoutput(ComputeGraphPtr &root_graph, NetoutputParentIndexes &indexes) {
  std::map<std::string, NodePtr> netoutput_map;
  for (auto &node : root_graph->GetAllNodes()) {
    netoutput_map[node->GetName()] = node;
  }
  for (auto &name_indexes_pair : indexes) {
    const auto iter = netoutput_map.find(name_indexes_pair.first);
    if (iter == netoutput_map.end()) {
      std::cout << "========================================" << std::endl;
      std::cout << "can not find " << name_indexes_pair.first << std::endl;
      std::cout << "========================================" << std::endl;
      GE_DUMP(root_graph, "AddParentIndexForNetoutput_failed");
      return false;
    }
    auto op_desc = iter->second->GetOpDesc();
    size_t input_index = 0U;
    if (name_indexes_pair.second.size() != op_desc->GetInputsSize()) {
      std::cout << "========================================" << std::endl;
      std::cout << name_indexes_pair.first << " real inputs size: " << op_desc->GetInputsSize()
                << ", but name_indexes_pair.second.size(): " << name_indexes_pair.second.size() << std::endl;
      std::cout << "========================================" << std::endl;
      GE_DUMP(root_graph, "AddParentIndexForNetoutput_failed");
      return false;
    }
    for (auto parent_index : name_indexes_pair.second) {
      auto tensor_desc = op_desc->MutableInputDesc(input_index++);
      AttrUtils::SetInt(tensor_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index);
    }
  }
  return true;
}
}

class StubKernelInfoStore : public ge::OpsKernelInfoStore {
 public:
  Status Initialize(const std::map<std::string, std::string> &options) override {
    return 0U;
  }
  Status Finalize() override {return true;};
  void GetAllOpsKernelInfo(std::map<std::string, OpInfo> &infos) const override {}
  virtual bool CheckSupported(const OpDescPtr &opDescPtr, std::string &un_supported_reason) const override {
    return true;
  }
  bool CheckAccuracySupported(const OpDescPtr &opDescPtr, std::string &un_supported_reason,
                              bool realQuery = false) const override {
    if ((opDescPtr->GetType() == TRANSDATA) && (opDescPtr->GetInputDesc(0).GetDataType() == DT_INT16)) {
      std::cout << std::endl << "transdata not support new datatype" << std::endl;
      return false;
    }
    if (opDescPtr->GetType() == CAST && (opDescPtr->GetInputDesc(0).GetFormat() == FORMAT_FRACTAL_Z_C04)) {
      std::cout << std::endl << "cast not support new format" << std::endl;
      return false;
    }
    return true;
  }

  bool CheckAccuracySupported(const ge::NodePtr &node, std::string &un_supported_reason,
                              bool realQuery = false) const override {
    const auto opDescPtr = node->GetOpDesc();
    return CheckAccuracySupported(opDescPtr, un_supported_reason, realQuery);
  }
};

class UtestGraphPassesSameTransdataBreadthFusionPass : public testing::Test {
 public:
  static void SetUpTestCase() {
    auto instance = GELib::GetInstance();
    map<string, string> options;
    instance->Initialize(options);
    OpsKernelManager &ops_kernel_manager = instance->OpsKernelManagerObj();
    OpInfo op_info;
    op_info.opKernelLib = "test_support";
    ops_kernel_manager.ops_kernel_info_[TRANSDATA].emplace_back(op_info);
    ops_kernel_manager.ops_kernel_info_[CAST].emplace_back(op_info);
    ops_kernel_manager.ops_kernel_store_["test_support"] = std::make_shared<StubKernelInfoStore>();
  }
  static void TearDownTestCase() {
    auto instance = GELib::GetInstance();
    instance->Finalize();
  }
 protected:
  void SetUp() {
    global_options_ = ge::GetThreadLocalContext().GetAllGlobalOptions();
    graph_options_ = ge::GetThreadLocalContext().GetAllGraphOptions();
    session_options_ = ge::GetThreadLocalContext().GetAllSessionOptions();
    ge::GetThreadLocalContext().SetGlobalOption({});
    ge::GetThreadLocalContext().SetGraphOption({});
    ge::GetThreadLocalContext().SetSessionOption({});
  }

  void TearDown() {
    ge::GetThreadLocalContext().SetGlobalOption(global_options_);
    ge::GetThreadLocalContext().SetGraphOption(graph_options_);
    ge::GetThreadLocalContext().SetSessionOption(session_options_);
  }
  std::map<std::string, std::string> global_options_;
  std::map<std::string, std::string> graph_options_;
  std::map<std::string, std::string> session_options_;
};

class NodeBuilder {
 public:
  NodeBuilder(const std::string &name, const std::string &type) { op_desc_ = std::make_shared<OpDesc>(name, type); }

  NodeBuilder &AddInputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                            ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddInputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  NodeBuilder &AddOutputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                             ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddOutputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  ge::NodePtr Build(const ge::ComputeGraphPtr &graph) { return graph->AddNode(op_desc_); }

 private:
  ge::GeTensorDescPtr CreateTensorDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                                       ge::DataType data_type = DT_FLOAT) {
    GeShape ge_shape{std::vector<int64_t>(shape)};
    ge::GeTensorDescPtr tensor_desc = std::make_shared<ge::GeTensorDesc>();
    tensor_desc->SetShape(ge_shape);
    tensor_desc->SetFormat(format);
    tensor_desc->SetDataType(data_type);
    return tensor_desc;
  }

  ge::OpDescPtr op_desc_;
};
// Node4D(NCHW)->cast1(DT_BOOL->FP16)->transdata1(NCHW->FORMAT_FRACTAL_Z_C04)->sinh1
//           /
//          --->cast2(DT_BOOL->FP16)->transdata2(NCHW->FORMAT_FRACTAL_Z_C04)->sinh2
static void CreateGraph(ComputeGraphPtr &graph) {
  // Node4D
  ge::NodePtr node_data = NodeBuilder("Data4D", DATA).AddOutputDesc({2, 16, 2, 2}, FORMAT_NCHW, DT_BOOL).Build(graph);
  // cast1
  ge::NodePtr node_cast_1 = NodeBuilder("node_cast_1", CAST)
                                .AddInputDesc({2, 16, 2, 2}, FORMAT_NCHW, DT_BOOL)
                                .AddOutputDesc({2, 16, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  auto src_name = node_data->GetName();
  node_cast_1->GetOpDesc()->SetSrcName({src_name});
  node_cast_1->GetOpDesc()->SetInputName({src_name});
  AttrUtils::SetInt(node_cast_1->GetOpDesc(), CAST_ATTR_SRCT, DT_FLOAT);

  // trandata1
  ge::NodePtr node_transdata_1 = NodeBuilder("node_transdata_1", TRANSDATA)
                                     .AddInputDesc({2, 16, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                     .AddOutputDesc({2, 1, 2, 2, 16}, FORMAT_FRACTAL_Z_C04, DT_FLOAT16)
                                     .Build(graph);
  // sinh1
  ge::NodePtr node_sinh_1 = NodeBuilder("node_sinh_1", SINH)
                              .AddInputDesc({2, 1, 2, 2, 16}, FORMAT_NC1HWC0, DT_FLOAT16)
                              .AddOutputDesc({2, 1, 2, 2, 16}, FORMAT_NC1HWC0, DT_FLOAT16)
                              .Build(graph);
  // cast2
  ge::NodePtr node_cast_2 = NodeBuilder("node_cast_2", CAST)
                                .AddInputDesc({2, 16, 2, 2}, FORMAT_NCHW, DT_BOOL)
                                .AddOutputDesc({2, 16, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  node_cast_2->GetOpDesc()->SetSrcName({src_name});
  node_cast_2->GetOpDesc()->SetInputName({src_name});
  // transdata2
  ge::NodePtr node_transdata_2 = NodeBuilder("node_transdata_2", TRANSDATA)
                                     .AddInputDesc({2, 16, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                     .AddOutputDesc({2, 1, 2, 2, 16}, FORMAT_FRACTAL_Z_C04, DT_FLOAT16)
                                     .Build(graph);

  // sinh2
  ge::NodePtr node_sinh_2 = NodeBuilder("node_sinh_2", SINH)
                              .AddInputDesc({2, 1, 2, 2, 16}, FORMAT_NC1HWC0, DT_FLOAT16)
                              .AddOutputDesc({2, 1, 2, 2, 16}, FORMAT_NC1HWC0, DT_FLOAT16)
                              .Build(graph);

  // add edge
  ge::GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_cast_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_1->GetOutDataAnchor(0), node_transdata_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_transdata_1->GetOutDataAnchor(0), node_sinh_1->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_cast_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_2->GetOutDataAnchor(0), node_transdata_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_transdata_2->GetOutDataAnchor(0), node_sinh_2->GetInDataAnchor(0));
}

/*
 *  data--cast--transdata--netoutput
 *    |
 * partitioned_call
 *
 *  +--------------------------------------------------------+
 *  | sub_data--sub_cast--sub_transdata-sub_add0--netoutput  |
 *  |   |                                                    |
 *  |  sub_add1                                              |
 *  +--------------------------------------------------------+
 *
 *             ||
 *             \/
 *
 *  data--transdata--cast--netoutput
 *     \      |
 *      partitioned_call
 *
 *  +---------------------------------------------------------------+
 *  | sub_1_transdata_fusion_arg_1--sub_cast--sub_add0--netoutput   |
 *  |                                                               |
 *  | sub_data--sub_add1                                            |
 *  +---------------------------------------------------------------+
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, DiffGraph_ExtractTransdataFromSubGraph_AddNewData_DoFusion) {
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub_data", sub_data)->NODE("sub_cast", CAST)->NODE("sub_transdata", TRANSDATA)
                               ->NODE("sub_add0", ADD)->NODE("netoutput", NETOUTPUT));
                     CHAIN(NODE("sub_data", sub_data)->EDGE(0, 0)->NODE("sub_add1", ADD)->NODE("netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("cast", CAST)->NODE("transdata", TRANSDATA)
                            ->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
                            ->NODE("netoutput", NETOUTPUT));
                };

  sub_1.Layout();
  const auto compute_graph = ToComputeGraph(g1);
  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"transdata", "sub_transdata"}));

  SameTransdataBreadthFusionPass pass;

  ge::graphStatus status = pass.Run(compute_graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);
  std::vector<NodeOutIndex> path = {{"data", 0}, {"sub_transdata", 0}, {"cast", 0}};
  ASSERT_TRUE(CheckConnection(compute_graph, path));

  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);
  std::vector<NodeOutIndex> path2 = {{"sub_1_transdata_fusion_arg_1", 0}, {"sub_cast", 0}, {"sub_add0", 0}};
  ASSERT_TRUE(CheckConnection(sub_graph_1, path2));

  std::vector<NodeOutIndex> path3 = {{"sub_data", 0}, {"sub_add1", 0}};
  ASSERT_TRUE(CheckConnection(sub_graph_1, path3));

  std::vector<NodeIO> check_nodes = {{"cast", {0}, {0}},
                                     {"partitioned_call", {1}, {}},
                                     {"sub_transdata", {}, {0}},
                                     {"sub_1_transdata_fusion_arg_1", {0}, {0}},
                                     {"sub_cast", {0}, {0}}};
  ASSERT_TRUE(CheckTensorDesc(compute_graph, check_nodes));

  std::vector<NodeIO> not_changed_nodes = {{"data", {}, {0}},
                                           {"partitioned_call", {0}, {}},
                                           {"sub_transdata", {0}, {}},
                                           {"sub_data", {0}, {0}},
                                           {"sub_add1", {0}, {0}}};
  ASSERT_TRUE(CheckNotChangedTensorDesc(compute_graph, not_changed_nodes));
}

/*
 *  data--cast--transdata--netoutput
 *    |
 * partitioned_call
 *    |
 * +----------------------------------------------+
 * | data--partitioned_call--netoutput            |
 * |              |                               |
 * |  +----------------------------------------+  |
 * |  | data--cast--transdata--add--netoutput  |  |
 * |  |   |                                    |  |
 * |  |  add                                   |  |
 * |  +----------------------------------------+  |
 * +----------------------------------------------+
 *             ||
 *             \/
 *  data--transdata--cast--netoutput
 *    |      |
 * partitioned_call
 *    |
 * +----------------------------------------------+
 * | new_data---+                                 |
 * |            |                                 |
 * | data--partitioned_call--netoutput            |
 * |              |                               |
 * |  +----------------------------------------+  |
 * |  | new_data--cast--add--netoutput         |  |
 * |  |                                        |  |
 * |  | data--add                              |  |
 * |  +----------------------------------------+  |
 * +----------------------------------------------+
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, DiffGraph_ExtractTransdataFromSubSubGraph_AddNewData_DoFusion) {
  const auto sub_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_sub_1) {
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->NODE("sub_sub_cast", CAST)
                                   ->NODE("sub_sub_transdata", TRANSDATA)
                                   ->NODE("sub_sub_add0", ADD)->NODE("sub_sub_netoutput", NETOUTPUT));
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->EDGE(0, 0)->NODE("sub_sub_add1", ADD)
                                   ->NODE("sub_sub_netoutput", NETOUTPUT));
                   };
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub_data", sub_data)->NODE("sub_partitioned_call", PARTITIONEDCALL, sub_sub_1)
                               ->NODE("sub_netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("cast", CAST)->NODE("transdata", TRANSDATA)
                            ->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
                            ->NODE("netoutput", NETOUTPUT));
                };
  auto sub_sub_1_graph = ToComputeGraph(sub_sub_1);
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);
  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  auto sub_partitioned_call_node = sub_graph_1->FindNode("sub_partitioned_call");
  ASSERT_NE(sub_partitioned_call_node, nullptr);
  sub_sub_1_graph->SetParentGraph(compute_graph);
  sub_sub_1_graph->SetParentNode(sub_partitioned_call_node);
  compute_graph->AddSubGraph(sub_sub_1_graph);


  const auto sub_sub_graph_1 = compute_graph->GetSubgraph("sub_sub_1");
  ASSERT_NE(sub_sub_graph_1, nullptr);

  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"transdata", "sub_sub_transdata"}));

  NetoutputParentIndexes indexes{{"sub_netoutput", {0}},
                                 {"sub_sub_netoutput", {0, 1}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));
  SameTransdataBreadthFusionPass pass;

  ge::graphStatus status = pass.Run(compute_graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);

  UtPaths root_graph_path = {{{"data", 0}, {"sub_sub_transdata", 0}, {"cast", 0}},
                             {{"data", 0}, {"partitioned_call", 0}},
                             {{"sub_sub_transdata", 0}, {"partitioned_call", 0}}
  };
  ASSERT_TRUE(CheckConnection(compute_graph, root_graph_path));

  std::vector<NodeOutIndex> path2 = {{"sub_data", 0}, {"sub_partitioned_call", 0}, {"sub_netoutput", 0}};
  ASSERT_TRUE(CheckConnection(sub_graph_1, path2));

  UtPaths paths3{{{"sub_sub_data", 0}, {"sub_sub_add1", 0}},
                 {{"sub_sub_1_transdata_fusion_arg_1", 0}, {"sub_sub_cast", 0}, {"sub_sub_add0", 0}}
  };
  ASSERT_TRUE(CheckConnection(sub_sub_graph_1, paths3));

  std::vector<NodeIO> check_nodes = {{"cast", {0}, {0}},
                                     {"partitioned_call", {1}, {}},
                                     {"sub_partitioned_call", {1}, {}},
                                     {"sub_1_transdata_fusion_arg_1", {0}, {0}},
                                     {"sub_sub_1_transdata_fusion_arg_1", {0}, {0}},
                                     {"sub_sub_cast", {0}, {0}}};
  ASSERT_TRUE(CheckTensorDesc(compute_graph, check_nodes));

  std::vector<NodeIO> not_changed_nodes = {
      {"data", {}, {0}},
      {"sub_sub_data", {}, {0}},
      {"sub_data", {0}, {0}},
      {"partitioned_call", {0}, {0}},
      {"sub_partitioned_call", {0}, {0}}};
  ASSERT_TRUE(CheckNotChangedTensorDesc(compute_graph, not_changed_nodes));
}

/*
 *  data--transdata--cast--netoutput
 *    |
 *   if (has two subgraph)
 *
 * if_subgraph               then_subgraph
 * +------------------+      +-------------------------------------------------+
 * | data--cast-add   |      |             +--transdata-add                    |
 * +------------------+      |             |                                   |
 *                           | data--cast--+--partitioned_call--cast-netoutput |
 *                           |                    |                            |
 *                           |  +-----------------------------------------+    |
 *                           |  | data--transdata--cast--add--netoutput   |    |
 *                           |  +-----------------------------------------+    |
 *                           +-------------------------------------------------+
 *
 *             ||
 *             \/
 *  data--transdata--cast--netoutput
 *     \  /
 *     if (has two subgraph) [改动点]新增了一个输入
 *
 * if_subgraph               then_subgraph
 * +------------------+      +-------------------------------------------------+ [改动点]新增了一个data, transdata被提取出去
 * | data--cast-add   |      |             +--add                              |
 * +------------------+      |             |                                   |
 *                           | data--cast--+--partitioned_call--cast-netoutput |
 *                           |                    |                            |
 *                           |  +-----------------------------------------+    | [改动点]新增了一个data, transdata被提取出去
 *                           |  | data------------ cast--add--netoutput   |    |
 *                           |  +-----------------------------------------+    |
 *                           +-------------------------------------------------+
 *
 * 关注点：data节点，对应父节点的输入有可能是新增的，校验数据类型
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass,
       DiffGraph_ExtractTransdataFromIfSubSubGraph_AddNewData_CheckDataType) {
  ge::GetThreadLocalContext().SetGlobalOption({});
  ge::GetThreadLocalContext().SetGraphOption({});
  ge::GetThreadLocalContext().SetSessionOption({});
  const auto sub_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_sub_1) {
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->NODE("sub_sub_transdata", TRANSDATA)
                                   ->NODE("sub_sub_cast", CAST)
                                   ->NODE("sub_sub_add", ADD)->NODE("sub_sub_netoutput", NETOUTPUT));
                       };
  const auto if_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(if_sub) {
                     CHAIN(NODE("if_sub_data", if_sub_data)->NODE("if_cast", CAST)->NODE("if_add", ADD)
                               ->NODE("if_sub_netoutput", NETOUTPUT));
                   };
  const auto then_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub) {
                        CHAIN(NODE("then_sub_data", then_sub_data)->NODE("then_cast1", CAST)->EDGE(0, 0)
                                  ->NODE("then_transdata", TRANSDATA)
                                  ->NODE("then_add", ADD)->NODE("then_sub_netoutput", NETOUTPUT));
                        CHAIN(NODE("then_cast1", CAST)->EDGE(0, 0)
                                  ->NODE("sub_partitioned_call", PARTITIONEDCALL, sub_sub_1)
                                  ->NODE("then_cast2", CAST)->NODE("then_sub_netoutput", NETOUTPUT));
                    };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("transdata", TRANSDATA)->NODE("cast", CAST)
                            ->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("if", IF, if_sub, then_sub)
                            ->NODE("netoutput", NETOUTPUT));
                };
  auto sub_sub_1_graph = ToComputeGraph(sub_sub_1);
  if_sub.Layout();
  auto compute_graph = ToComputeGraph(g1);
  const auto then_sub_graph = compute_graph->GetSubgraph("then_sub");
  ASSERT_NE(then_sub_graph, nullptr);

  // 最内层子图
  auto sub_partitioned_call_node = then_sub_graph->FindNode("sub_partitioned_call");
  ASSERT_NE(sub_partitioned_call_node, nullptr);
  sub_sub_1_graph->SetParentGraph(compute_graph);
  sub_sub_1_graph->SetParentNode(sub_partitioned_call_node);
  compute_graph->AddSubGraph(sub_sub_1_graph);

  compute_graph->TopologicalSorting();
  std::vector<std::string> fp16_nodes {"data", "transdata", "if_sub_data", "then_sub_data"};
  SetDataTypeForNode(compute_graph, fp16_nodes, DT_FLOAT16);

  std::vector<std::string> fp16_nodes_inputs {"cast", "if", "if_cast", "then_cast1"};
  SetDataTypeForNodeInputs(compute_graph, fp16_nodes_inputs, DT_FLOAT16);

  const auto sub_sub_graph_1 = compute_graph->GetSubgraph("sub_sub_1");
  ASSERT_NE(sub_sub_graph_1, nullptr);

  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"sub_sub_transdata", "then_transdata", "transdata"}));

  NetoutputParentIndexes indexes{{"sub_sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));
  SameTransdataBreadthFusionPass pass;
  ge::graphStatus status = pass.Run(compute_graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);

  UtPaths root_graph_path = {{{"data", 0}, {"transdata", 0}, {"cast", 0}},
                             {{"data", 0}, {"if", 0}},
                             {{"transdata", 0}, {"if", 0}}
  };
  ASSERT_TRUE(CheckConnection(compute_graph, root_graph_path));

  const auto if_sub_graph = compute_graph->GetSubgraph("if_sub");
  ASSERT_NE(if_sub_graph, nullptr);
  std::vector<NodeOutIndex> path2 = {{"if_sub_data", 0}, {"if_cast", 0}, {"if_add", 0}};
  ASSERT_TRUE(CheckConnection(if_sub_graph, path2));

  UtPaths paths3{{{"sub_sub_data", 0}, {"sub_sub_cast", 0}, {"sub_sub_add", 0}}
  };
  ASSERT_TRUE(CheckConnection(sub_sub_graph_1, paths3));

  UtPaths path4 = {{{"then_sub_transdata_fusion_arg_1", 0}, {"then_cast1", 0}, {"then_add", 0}},
                   {{"then_cast1", 0}, {"sub_partitioned_call", 0}}};
  ASSERT_TRUE(CheckConnection(then_sub_graph, path4));


  std::vector<NodeIO> check_nodes = {
                                     {"if", {1}, {}},
                                     {"then_sub_transdata_fusion_arg_1", {0}, {0}},
                                     {"then_cast1", {0}, {0}},
                                     {"sub_partitioned_call", {0}, {}},
                                     {"sub_sub_data", {0}, {0}}};
  ASSERT_TRUE(CheckTensorDesc(compute_graph, check_nodes));

  std::vector<NodeIO> not_changed_nodes = {
      {"data", {}, {0}},
      {"if", {0}, {}},
      {"if_sub_data", {0}, {0}}};
  ASSERT_TRUE(CheckNotChangedTensorDesc(compute_graph, not_changed_nodes));

  std::vector<NodeIO> data_type_fp16_nodes = {
      {"if", {1}, {}},
      {"then_sub_transdata_fusion_arg_1", {0}, {0}}};
  ASSERT_TRUE(CheckNodesDataType(compute_graph, data_type_fp16_nodes, DT_FLOAT16));
}

/*
 *  data--if--transdata
 *
 * if_subgraph                            then_subgraph
 * +-------------------------------+      +------------------------+
 * | data-+--if_transdata-if_relu  |      | data--relu---netoutput |
 * |      |                        |      +------------------------+
 * |     netoutput                 |
 * +-------------------------------+
 * 关注点：由于then_subgraph中没有到达transdata的路径，所以两个transdata不能融合
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, IfTwoSubgraphs_NotFuse) {
  ge::GetThreadLocalContext().SetGlobalOption({});
  ge::GetThreadLocalContext().SetGraphOption({});
  ge::GetThreadLocalContext().SetSessionOption({});

  const auto if_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(if_sub) {
                      CHAIN(NODE("if_sub_data", if_sub_data)->EDGE(0, 0)
                                ->NODE("if_sub_netoutput", NETOUTPUT));
                      CHAIN(NODE("if_sub_data", if_sub_data)->EDGE(0, 0)->NODE("if_transdata", TRANSDATA)->NODE("if_relu", RELU)->Ctrl()
                                ->NODE("if_sub_netoutput", NETOUTPUT));
                    };
  const auto then_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub) {
                        CHAIN(NODE("then_sub_data", then_sub_data)->NODE("then_relu", RELU)->NODE("then_sub_netoutput", NETOUTPUT));
                      };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("if", IF, if_sub, then_sub)->NODE("transdata", TRANSDATA)
                            ->NODE("netoutput", NETOUTPUT));
                };

  auto compute_graph = ToComputeGraph(g1);
  const auto then_sub_graph = compute_graph->GetSubgraph("then_sub");
  ASSERT_NE(then_sub_graph, nullptr);

  compute_graph->TopologicalSorting();

  NetoutputParentIndexes indexes{{"if_sub_netoutput", {0}}, {"then_sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  SameTransdataBreadthFusionPass pass;
  ge::graphStatus status = pass.Run(compute_graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);

  UtPaths root_graph_path = {{{"data", 0}, {"if", 0}, {"transdata", 0}}};
  ASSERT_TRUE(CheckConnection(compute_graph, root_graph_path));

  const auto if_sub_graph = compute_graph->GetSubgraph("if_sub");
  ASSERT_NE(if_sub_graph, nullptr);
  std::vector<NodeOutIndex> path2 = {{"if_sub_data", 0}, {"if_transdata", 0}, {"if_relu", 0}};
  ASSERT_TRUE(CheckConnection(if_sub_graph, path2));

  UtPaths path4 = {{{"then_sub_data", 0}, {"then_relu", 0}, {"then_sub_netoutput", 0}}};
  ASSERT_TRUE(CheckConnection(then_sub_graph, path4));
}

/*
 *  data--cast--transdata--netoutput
 *    |
 * partitioned_call
 *    |
 * +----------------------------------------------+
 * | data--cast-partitioned_call--netoutput       |
 * |                       |                      |
 * |  +----------------------------------------+  |
 * |  | data--cast--transdata--add--netoutput  |  |
 * |  +----------------------------------------+  |
 * +----------------------------------------------+
 *             ||
 *             \/
 *  data--transdata--cast--netoutput
 *          |
 * partitioned_call
 *    |
 * +----------------------------------------------+
 * | data--cast-partitioned_call--netoutput       |
 * |                       |                      |
 * |  +----------------------------------------+  |
 * |  | data--cast--add--netoutput             |  |
 * |  +----------------------------------------+  |
 * +----------------------------------------------+
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, DiffGraph_ExtractTransdataFromSubSubGraph_DoFusion) {
  const auto sub_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_sub_1) {
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->NODE("sub_sub_cast", CAST)
                                   ->NODE("sub_sub_transdata", TRANSDATA)
                                   ->NODE("sub_sub_add0", ADD)->NODE("sub_sub_netoutput", NETOUTPUT));
                       };
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub_data", sub_data)->NODE("sub_cast", CAST)->NODE("sub_partitioned_call", PARTITIONEDCALL, sub_sub_1)
                               ->NODE("sub_netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("cast", CAST)->NODE("transdata", TRANSDATA)
                            ->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
                            ->NODE("netoutput", NETOUTPUT));
                };
  auto sub_sub_1_graph = ToComputeGraph(sub_sub_1);
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);
  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  auto sub_partitioned_call_node = sub_graph_1->FindNode("sub_partitioned_call");
  ASSERT_NE(sub_partitioned_call_node, nullptr);
  sub_sub_1_graph->SetParentGraph(compute_graph);
  sub_sub_1_graph->SetParentNode(sub_partitioned_call_node);
  compute_graph->AddSubGraph(sub_sub_1_graph);


  const auto sub_sub_graph_1 = compute_graph->GetSubgraph("sub_sub_1");
  ASSERT_NE(sub_sub_graph_1, nullptr);

  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"transdata", "sub_sub_transdata"}));

  NetoutputParentIndexes indexes{{"sub_netoutput", {0}},
                                 {"sub_sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));
  SameTransdataBreadthFusionPass pass;

  ge::graphStatus status = pass.Run(compute_graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);

  UtPaths root_graph_path = {{{"data", 0}, {"sub_sub_transdata", 0}, {"cast", 0}},
                             {{"sub_sub_transdata", 0}, {"partitioned_call", 0}}
  };
  ASSERT_TRUE(CheckConnection(compute_graph, root_graph_path));

  std::vector<NodeOutIndex> path2 = {{"sub_data", 0}, {"sub_cast", 0}, {"sub_partitioned_call", 0}};
  ASSERT_TRUE(CheckConnection(sub_graph_1, path2));

  UtPaths paths3{{{"sub_sub_data", 0}, {"sub_sub_cast", 0}, {"sub_sub_add0", 0}}};
  ASSERT_TRUE(CheckConnection(sub_sub_graph_1, paths3));

  std::vector<NodeIO> check_nodes = {{"cast", {0}, {0}},
                                     {"sub_sub_data", {}, {0}},
                                     {"sub_data", {0}, {0}},
                                     {"partitioned_call", {0}, {}},
                                     {"sub_partitioned_call", {0}, {}},
                                     {"sub_sub_cast", {0}, {0}}};
  ASSERT_TRUE(CheckTensorDesc(compute_graph, check_nodes));

  std::vector<NodeIO> not_changed_nodes = {
      {"data", {}, {0}}};
  ASSERT_TRUE(CheckNotChangedTensorDesc(compute_graph, not_changed_nodes));
}


/*
 *  data--cast--transdata--netoutput
 *    |
 * partitioned_call
 *    |
 * +----------------------------------------------+
 * |            +-transdata-cast                  |
 * |            |                                 |
 * | data--cast-+-partitioned_call--netoutput     |
 * |                       |                      |
 * |  +----------------------------------------+  |
 * |  | data--cast--transdata--add--netoutput  |  |
 * |  +----------------------------------------+  |
 * +----------------------------------------------+
 *             ||
 *             \/
 *  data--transdata--cast--netoutput
 *          |
 * partitioned_call
 *    |
 * +----------------------------------------------+
 * |            +-cast                            |
 * |            |                                 |
 * | data--cast-+-partitioned_call--netoutput     |
 * |                       |                      |
 * |  +----------------------------------------+  |
 * |  | data--cast--add--netoutput             |  |
 * |  +----------------------------------------+  |
 * +----------------------------------------------+
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, DiffGraph_ExtractTransdataFromSubGraphAndSubSubGraph_DoFusion) {
  const auto sub_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_sub_1) {
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->NODE("sub_sub_cast", CAST)
                                   ->NODE("sub_sub_transdata", TRANSDATA)
                                   ->NODE("sub_sub_add0", ADD)->NODE("sub_sub_netoutput", NETOUTPUT));
                       };
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub_data", sub_data)->NODE("sub_cast1", CAST)->NODE("sub_partitioned_call", PARTITIONEDCALL, sub_sub_1)
                               ->NODE("sub_netoutput", NETOUTPUT));
                     CHAIN(NODE("sub_cast1", CAST)->EDGE(0, 0)->NODE("sub_transdata", TRANSDATA)->NODE("sub_cast2", CAST));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("cast", CAST)->NODE("transdata", TRANSDATA)
                            ->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
                            ->NODE("netoutput", NETOUTPUT));
                };
  auto sub_sub_1_graph = ToComputeGraph(sub_sub_1);
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);
  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  auto sub_partitioned_call_node = sub_graph_1->FindNode("sub_partitioned_call");
  ASSERT_NE(sub_partitioned_call_node, nullptr);
  sub_sub_1_graph->SetParentGraph(compute_graph);
  sub_sub_1_graph->SetParentNode(sub_partitioned_call_node);
  compute_graph->AddSubGraph(sub_sub_1_graph);


  const auto sub_sub_graph_1 = compute_graph->GetSubgraph("sub_sub_1");
  ASSERT_NE(sub_sub_graph_1, nullptr);

  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"transdata", "sub_transdata", "sub_sub_transdata"}));

  NetoutputParentIndexes indexes{{"sub_netoutput", {0}},
                                 {"sub_sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));
  SameTransdataBreadthFusionPass pass;

  ge::graphStatus status = pass.Run(compute_graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);

  UtPaths root_graph_path = {{{"data", 0}, {"sub_transdata", 0}, {"cast", 0}},
                             {{"sub_transdata", 0}, {"partitioned_call", 0}}
  };
  ASSERT_TRUE(CheckConnection(compute_graph, root_graph_path));

  UtPaths path2 = {{{"sub_data", 0}, {"sub_cast1", 0}, {"sub_partitioned_call", 0}},
                   {{"sub_data", 0}, {"sub_cast1", 0}, {"sub_cast2", 0}}};
  ASSERT_TRUE(CheckConnection(sub_graph_1, path2));

  UtPaths paths3{{{"sub_sub_data", 0}, {"sub_sub_cast", 0}, {"sub_sub_add0", 0}}};
  ASSERT_TRUE(CheckConnection(sub_sub_graph_1, paths3));

  std::vector<NodeIO> check_nodes = {{"cast", {0}, {0}},
                                     {"sub_sub_data", {}, {0}},
                                     {"sub_data", {0}, {0}},
                                     {"partitioned_call", {0}, {}},
                                     {"sub_partitioned_call", {0}, {}},
                                     {"sub_sub_cast", {0}, {0}}};
  ASSERT_TRUE(CheckTensorDesc(compute_graph, check_nodes));

  std::vector<NodeIO> not_changed_nodes = {
      {"data", {}, {0}}};
  ASSERT_TRUE(CheckNotChangedTensorDesc(compute_graph, not_changed_nodes));
}

/*
 *  data--cast--transdata--netoutput
 *    |
 * partitioned_call
 *    |
 * +--------------------------------------------------+
 * | data--cast-partitioned_call-transdata--netoutput |
 * |                       |                          |
 * |               +-----------------+                |
 * |               | data--netoutput |                |
 * |               +-----------------+                |
 * +--------------------------------------------------+
 *             ||
 *             \/
 *  data--transdata--cast--netoutput
 *             |
 * partitioned_call
 *    |
 * +--------------------------------------------------+
 * | data--cast-partitioned_call--netoutput           |
 * |                       |                          |
 * |               +-----------------+                |
 * |               | data--netoutput |                |
 * |               +-----------------+                |
 * +--------------------------------------------------+
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, DiffGraph_ExtractTransdataThroughSubSubGraph_DoFusion) {
  const auto sub_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_sub_1) {
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->NODE("sub_sub_netoutput", NETOUTPUT));
                       };
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub_data", sub_data)->NODE("sub_cast", CAST)->NODE("sub_partitioned_call", PARTITIONEDCALL, sub_sub_1)
                               ->NODE("sub_transdata", TRANSDATA)->NODE("sub_netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("cast", CAST)->NODE("transdata", TRANSDATA)
                            ->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
                            ->NODE("netoutput", NETOUTPUT));
                };
  auto sub_sub_1_graph = ToComputeGraph(sub_sub_1);
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);
  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  auto sub_partitioned_call_node = sub_graph_1->FindNode("sub_partitioned_call");
  ASSERT_NE(sub_partitioned_call_node, nullptr);
  sub_sub_1_graph->SetParentGraph(compute_graph);
  sub_sub_1_graph->SetParentNode(sub_partitioned_call_node);
  compute_graph->AddSubGraph(sub_sub_1_graph);


  const auto sub_sub_graph_1 = compute_graph->GetSubgraph("sub_sub_1");
  ASSERT_NE(sub_sub_graph_1, nullptr);

  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"transdata", "sub_transdata"}));

  NetoutputParentIndexes indexes{{"sub_netoutput", {0}},
                                 {"sub_sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  SameTransdataBreadthFusionPass pass;
  ge::graphStatus status = pass.Run(compute_graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);

  UtPaths root_graph_path = {{{"data", 0}, {"sub_transdata", 0}, {"cast", 0}},
                             {{"sub_transdata", 0}, {"partitioned_call", 0}}
  };
  ASSERT_TRUE(CheckConnection(compute_graph, root_graph_path));

  std::vector<NodeOutIndex> path2 = {{"sub_data", 0}, {"sub_cast", 0}, {"sub_partitioned_call", 0}, {"sub_netoutput", 0}};
  ASSERT_TRUE(CheckConnection(sub_graph_1, path2));

  UtPaths paths3{{{"sub_sub_data", 0}, {"sub_sub_netoutput", 0}}};
  ASSERT_TRUE(CheckConnection(sub_sub_graph_1, paths3));

  std::vector<NodeIO> check_nodes = {{"cast", {0}, {0}},
                                     {"sub_sub_data", {}, {0}},
                                     {"sub_sub_netoutput", {0}, {}},
                                     {"sub_data", {0}, {0}},
                                     {"partitioned_call", {0}, {}},
                                     {"sub_partitioned_call", {0}, {0}}};
  ASSERT_TRUE(CheckTensorDesc(compute_graph, check_nodes));

  std::vector<NodeIO> not_changed_nodes = {
      {"data", {}, {0}},
      {"partitioned_call", {}, {0}}};
  ASSERT_TRUE(CheckNotChangedTensorDesc(compute_graph, not_changed_nodes));
}

/*
 *  data--cast--transdata--netoutput
 *    |
 * partitioned_call
 *    |
 * +--------------------------------------------------+
 * |                            mul                   |
 * |                            /                     |
 * | data----partitioned_call--+-transdata--netoutput |
 * |                       |                          |
 * |               +-----------------+                |
 * |               | data--netoutput |                |
 * |               +-----------------+                |
 * +--------------------------------------------------+
 *             ||
 *             \/
 *  data--transdata--cast--netoutput
 *             |
 * partitioned_call
 *    |
 * +--------------------------------------------------+
 * | new_data--+                    mul               |
 * |            \                  /                  |
 * | data-------partitioned_call--+--netoutput        |
 * |                       |                          |
 * |               +-----------------+                |
 * |               | new_data--+     |                |
 * |               |           |     |                |
 * |               | data--netoutput |                |
 * |               +-----------------+                |
 * +--------------------------------------------------+
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, DiffGraph_ExtractTransdataThroughSubSubGraph_AddNewData_DoFusion) {
  const auto sub_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_sub_1) {
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->NODE("sub_sub_netoutput", NETOUTPUT));
                       };
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub_data", sub_data)->NODE("sub_partitioned_call", PARTITIONEDCALL, sub_sub_1)
                               ->NODE("sub_transdata", TRANSDATA)->NODE("sub_netoutput", NETOUTPUT));
                     CHAIN(NODE("sub_partitioned_call", PARTITIONEDCALL)->EDGE(0, 0)->NODE("mul", MUL));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("cast", CAST)->NODE("transdata", TRANSDATA)
                            ->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
                            ->NODE("netoutput", NETOUTPUT));
                };
  auto sub_sub_1_graph = ToComputeGraph(sub_sub_1);
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);
  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  auto sub_partitioned_call_node = sub_graph_1->FindNode("sub_partitioned_call");
  ASSERT_NE(sub_partitioned_call_node, nullptr);
  sub_sub_1_graph->SetParentGraph(compute_graph);
  sub_sub_1_graph->SetParentNode(sub_partitioned_call_node);
  compute_graph->AddSubGraph(sub_sub_1_graph);


  const auto sub_sub_graph_1 = compute_graph->GetSubgraph("sub_sub_1");
  ASSERT_NE(sub_sub_graph_1, nullptr);

  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"transdata", "sub_transdata"}));
  SetDataTypeForNode(compute_graph, {"data", "partitioned_call", "sub_data", "sub_partitioned_call", "sub_transdata",
                                    "sub_sub_data", "sub_sub_netoutput"}, DT_FLOAT16);

  NetoutputParentIndexes indexes{{"sub_netoutput", {0}},
                                 {"sub_sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  SameTransdataBreadthFusionPass pass;
  ge::graphStatus status = pass.Run(compute_graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);

  UtPaths root_graph_path = {{{"data", 0}, {"sub_transdata", 0}, {"cast", 0}},
                             {{"sub_transdata", 0}, {"partitioned_call", 0}},
                             {{"data", 0}, {"partitioned_call", 1}},
  };
  ASSERT_TRUE(CheckConnection(compute_graph, root_graph_path));

  UtPaths path2 = {{{"sub_data", 0}, {"sub_partitioned_call", 1}, {"sub_netoutput", 0}},
                   {{"sub_1_transdata_fusion_arg_1", 0}, {"sub_partitioned_call", 0}, {"mul", 0}}
  };
  ASSERT_TRUE(CheckConnection(sub_graph_1, path2));

  UtPaths paths3{{{"sub_sub_data", 0}, {"sub_sub_netoutput", 0}},
                 {{"sub_sub_1_transdata_fusion_arg_1", 0}, {"sub_sub_netoutput", 1}}};
  ASSERT_TRUE(CheckConnection(sub_sub_graph_1, paths3));

  std::vector<NodeIO> check_nodes = {{"cast", {0}, {0}},
                                     {"partitioned_call", {1}, {}},
                                     {"sub_1_transdata_fusion_arg_1", {0}, {0}},
                                     {"sub_partitioned_call", {1}, {1}},
                                     {"sub_sub_1_transdata_fusion_arg_1", {0}, {0}},
                                     {"sub_sub_netoutput", {1}, {}}};
  ASSERT_TRUE(CheckTensorDesc(compute_graph, check_nodes));

  std::vector<NodeIO> not_changed_nodes = {
      {"data", {}, {0}},
      {"partitioned_call", {0}, {}},
      {"sub_data", {0}, {0}},
      {"sub_partitioned_call", {0}, {0}},
      {"sub_netoutput", {0}, {}},
      {"sub_sub_data", {0}, {0}},
      {"sub_sub_netoutput", {0}, {0}}
  };
  ASSERT_TRUE(CheckNotChangedTensorDesc(compute_graph, not_changed_nodes));

  std::vector<NodeIO> check_dtype_nodes = {{"sub_sub_netoutput", {1}, {}},
                                           {"sub_partitioned_call", {1}, {1}},
                                           {"sub_sub_1_transdata_fusion_arg_1", {0}, {0}},
                                           {"sub_1_transdata_fusion_arg_1", {0}, {0}}};
  ASSERT_TRUE(CheckNodesDataType(compute_graph, check_dtype_nodes, DT_FLOAT16));
}

/*
 *  data--cast--transdata--netoutput
 *    |
 * partitioned_call-transdta
 *    |
 * +-------------------------------------+
 * | data--partitioned_call---netoutput  |
 * |              |                      |
 * |      +-----------------+            |
 * |      | data--netoutput |            |
 * |      +-----------------+            |
 * +-------------------------------------+
 *             ||
 *             \/
 *  data--transdata--cast--netoutput
 *            |
 * partitioned_call
 *    |
 * +-------------------------------------+
 * | data--partitioned_call---netoutput  |
 * |              |                      |
 * |      +-----------------+            |
 * |      | data--netoutput |            |
 * |      +-----------------+            |
 * +-------------------------------------+
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, DiffGraph_ExtractTransdataThroughDoubleSubGraph_DoFusion) {
  const auto sub_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_sub_1) {
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->NODE("sub_sub_netoutput", NETOUTPUT));
                       };
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub_data", sub_data)->NODE("sub_partitioned_call", PARTITIONEDCALL, sub_sub_1)
                               ->NODE("sub_netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("cast", CAST)->NODE("transdata", TRANSDATA)
                            ->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
                            ->NODE("transdata1", TRANSDATA)->NODE("netoutput", NETOUTPUT));
                };
  auto sub_sub_1_graph = ToComputeGraph(sub_sub_1);
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);
  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  auto sub_partitioned_call_node = sub_graph_1->FindNode("sub_partitioned_call");
  ASSERT_NE(sub_partitioned_call_node, nullptr);
  sub_sub_1_graph->SetParentGraph(compute_graph);
  sub_sub_1_graph->SetParentNode(sub_partitioned_call_node);
  compute_graph->AddSubGraph(sub_sub_1_graph);


  const auto sub_sub_graph_1 = compute_graph->GetSubgraph("sub_sub_1");
  ASSERT_NE(sub_sub_graph_1, nullptr);

  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"transdata", "transdata1"}));

  NetoutputParentIndexes indexes{{"sub_netoutput", {0}},
                                 {"sub_sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  SameTransdataBreadthFusionPass pass;
  ge::graphStatus status = pass.Run(compute_graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);

  UtPaths root_graph_path = {{{"data", 0}, {"transdata1", 0}, {"cast", 0}},
                             {{"transdata1", 0}, {"partitioned_call", 0}, {"netoutput", 0}}
  };
  ASSERT_TRUE(CheckConnection(compute_graph, root_graph_path));

  std::vector<NodeOutIndex> path2 = {{"sub_data", 0}, {"sub_partitioned_call", 0}, {"sub_netoutput", 0}};
  ASSERT_TRUE(CheckConnection(sub_graph_1, path2));

  UtPaths paths3{{{"sub_sub_data", 0}, {"sub_sub_netoutput", 0}}};
  ASSERT_TRUE(CheckConnection(sub_sub_graph_1, paths3));

  std::vector<NodeIO> check_nodes = {{"cast", {0}, {0}},
                                     {"sub_sub_data", {}, {0}},
                                     {"sub_sub_netoutput", {0}, {}},
                                     {"sub_data", {}, {0}},
                                     {"sub_netoutput", {0}, {}},
                                     {"partitioned_call", {0}, {0}},
                                     {"sub_partitioned_call", {0}, {0}}};
  ASSERT_TRUE(CheckTensorDesc(compute_graph, check_nodes));

  std::vector<NodeIO> not_changed_nodes = {
      {"data", {}, {0}}};
  ASSERT_TRUE(CheckNotChangedTensorDesc(compute_graph, not_changed_nodes));
}

/*
 *  data--cast--transdata--netoutput
 *    |
 * partitioned_call--+--transdta
 *    |              |
 *    |              +--mul
 *    |
 * +-------------------------------------+
 * | data--partitioned_call---netoutput  |
 * |              |                      |
 * |      +-----------------+            |
 * |      | data--netoutput |            |
 * |      +-----------------+            |
 * +-------------------------------------+
 *             ||
 *             \/
 *  data--transdata--cast--netoutput
 *            |
 * partitioned_call
 *    |
 * +-------------------------------------+
 * | data--partitioned_call---netoutput  |
 * |              |                      |
 * |      +-----------------+            |
 * |      | data--netoutput |            |
 * |      +-----------------+            |
 * +-------------------------------------+
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, DiffGraph_ExtractTransdataThroughDoubleSubGraph_AddNewData_DoFusion) {
  const auto sub_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_sub_1) {
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->NODE("sub_sub_netoutput", NETOUTPUT));
                       };
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub_data", sub_data)->NODE("sub_partitioned_call", PARTITIONEDCALL, sub_sub_1)
                               ->NODE("sub_netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("cast", CAST)->NODE("transdata", TRANSDATA)
                            ->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
                            ->NODE("transdata1", TRANSDATA)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("partitioned_call", PARTITIONEDCALL)->EDGE(0, 0)->NODE("mul", MUL)->NODE("netoutput", NETOUTPUT));
                };
  auto sub_sub_1_graph = ToComputeGraph(sub_sub_1);
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);
  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  auto sub_partitioned_call_node = sub_graph_1->FindNode("sub_partitioned_call");
  ASSERT_NE(sub_partitioned_call_node, nullptr);
  sub_sub_1_graph->SetParentGraph(compute_graph);
  sub_sub_1_graph->SetParentNode(sub_partitioned_call_node);
  compute_graph->AddSubGraph(sub_sub_1_graph);


  const auto sub_sub_graph_1 = compute_graph->GetSubgraph("sub_sub_1");
  ASSERT_NE(sub_sub_graph_1, nullptr);

  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"transdata", "transdata1"}));

  NetoutputParentIndexes indexes{{"sub_netoutput", {0}},
                                 {"sub_sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  SameTransdataBreadthFusionPass pass;
  ge::graphStatus status = pass.Run(compute_graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);

  UtPaths root_graph_path = {{{"data", 0}, {"transdata1", 0}, {"cast", 0}},
                             {{"transdata1", 0}, {"partitioned_call", 1}, {"netoutput", 0}}
  };
  ASSERT_TRUE(CheckConnection(compute_graph, root_graph_path));

  UtPaths path2 = {{{"sub_data", 0}, {"sub_partitioned_call", 0}, {"sub_netoutput", 0}},
                   {{"sub_1_transdata_fusion_arg_1", 0}, {"sub_partitioned_call", 1}, {"sub_netoutput", 1}}};
  ASSERT_TRUE(CheckConnection(sub_graph_1, path2));

  UtPaths paths3{{{"sub_sub_data", 0}, {"sub_sub_netoutput", 0}},
                 {{"sub_sub_1_transdata_fusion_arg_1", 0}, {"sub_sub_netoutput", 1}}};
  ASSERT_TRUE(CheckConnection(sub_sub_graph_1, paths3));

  std::vector<NodeIO> check_nodes = {{"cast", {0}, {0}},
                                     {"partitioned_call", {1}, {1}},
                                     {"sub_netoutput", {1}, {}},
                                     {"sub_partitioned_call", {1}, {1}},
                                     {"sub_1_transdata_fusion_arg_1", {0}, {0}},
                                     {"sub_sub_netoutput", {1}, {}},
                                     {"sub_sub_1_transdata_fusion_arg_1", {0}, {0}}};
  ASSERT_TRUE(CheckTensorDesc(compute_graph, check_nodes));

  std::vector<NodeIO> not_changed_nodes = {
      {"data", {}, {0}},
      {"sub_sub_data", {}, {0}},
      {"sub_sub_netoutput", {0}, {}},
      {"sub_data", {0}, {0}},
      {"sub_partitioned_call", {0}, {0}},
      {"sub_netoutput", {0}, {}},
      {"partitioned_call", {0}, {0}},
      {"netoutput", {0,2}, {}}
  };
  ASSERT_TRUE(CheckNotChangedTensorDesc(compute_graph, not_changed_nodes));
}

/*
 *                                  0
 * data--partitioned_call-+-----mul--netoutput
 *          |  3| 2| 1|              1| 2| 3|
 *          |   |  |  +------transdata+  |  |
 *          |   |  |                     |  |
 *          |   |  +---------mul---------+  |
 *          |   |                           |
 *          |   +------------transdata------+
 *         \|/
 *  +-----------------------------------+
 *  |                       0           |
 *  | data-mul-+--transdata--netoutput  |
 *  |          |             1| 2| 3|   |
 *  |          +---cast-------+  |  |   |
 *  |          |                 |  |   |
 *  |          +---add-----------+  |   |
 *  |          |                    |   |
 *  |          +--------------------+   |
 *  +-----------------------------------+
 *
 *             ||
 *             \/
 *                       0     0
 * data--partitioned_call-+-----mul--netoutput
 *          |  3|  2|  1|           1| 2| 3|
 *          |   |   |   +------------+  |  |
 *          |   |   |                   |  |
 *          |   |   +-------mul---------+  |
 *          |   |                          |
 *          |   +--------------------------+
 *         \|/
 *  +----------------------------------------------+
 *  |          +------------add-----------+        |
 *  |          |                     0   2|        |
 *  | data-mul-+--transdata--+----------netoutput  |
 *  |                        |          1|   3|    |
 *  |                        +---cast----+    |    |
 *  |                        |                |    |
 *  |                        |                |    |
 *  |                        +----------------+    |
 *  +----------------------------------------------+
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, DiffGraph_ExtractTransdataToSubGraph_DoFusion) {
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub_data", sub_data)->NODE("sub_mul", MUL)->NODE("sub_transdata", TRANSDATA)
                               ->NODE("sub_netoutput", NETOUTPUT));
                     CHAIN(NODE("sub_mul")->EDGE(0, 0)->NODE("sub_cast", CAST)->NODE("sub_netoutput", NETOUTPUT));
                     CHAIN(NODE("sub_mul")->EDGE(0, 0)->NODE("sub_add", ADD)->NODE("sub_netoutput", NETOUTPUT));
                     CHAIN(NODE("sub_mul")->EDGE(0, 3)->NODE("sub_netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
                            ->NODE("mul1", MUL)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("partitioned_call", PARTITIONEDCALL)
                            ->NODE("transdata1", TRANSDATA)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("partitioned_call", PARTITIONEDCALL)->NODE("mul2", MUL)
                            ->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("partitioned_call", PARTITIONEDCALL)->NODE("transdata2", TRANSDATA)
                            ->NODE("netoutput", NETOUTPUT));
                };
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"sub_transdata", "transdata1", "transdata2"}));

  NetoutputParentIndexes indexes{{"sub_netoutput", {0, 1, 2, 3}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  SameTransdataBreadthFusionPass pass;
  ge::graphStatus status = pass.Run(compute_graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);

  UtPaths root_graph_path = {{{"data", 0}, {"partitioned_call", 0}, {"mul1", 0}, {"netoutput", 0}},
                             {{"partitioned_call", 1}, {"netoutput", 0}},
                             {{"partitioned_call", 3}, {"netoutput", 0}},
                             {{"partitioned_call", 2}, {"mul2", 0}, {"netoutput", 0}}
  };
  ASSERT_TRUE(CheckConnection(compute_graph, root_graph_path));

  UtPaths path2 = {{{"sub_data", 0}, {"sub_mul", 0}, {"sub_add", 0}, {"sub_netoutput", 0}},
                   {{"sub_mul", 0}, {"sub_transdata", 1}},
                   {{"sub_transdata", 0}, {"sub_cast", 0}, {"sub_netoutput", 0}},
                   {{"sub_transdata", 0}, {"sub_netoutput", 0}}};
  ASSERT_TRUE(CheckConnection(sub_graph_1, path2));

  std::vector<NodeIO> check_nodes = {{"sub_cast", {0}, {0}},
                                     {"partitioned_call", {}, {1,3}},
                                     {"sub_netoutput", {1,3}, {}}};
  ASSERT_TRUE(CheckTensorDesc(compute_graph, check_nodes));

  std::vector<NodeIO> not_changed_nodes = {
      {"sub_data", {0}, {0}},
      {"sub_mul", {0}, {0}},
      {"sub_add", {0}, {0}},
      {"sub_netoutput", {2}, {}},
      {"partitioned_call", {}, {0,2}},
      {"mul2", {0}, {0}}
  };
  ASSERT_TRUE(CheckNotChangedTensorDesc(compute_graph, not_changed_nodes));
}

/*
 *                                  0
 * data--partitioned_call-+-----mul--netoutput
 *          |  3| 2| 1|              1| 2| 3|
 *          |   |  |  +------transdata+  |  |
 *          |   |  |                     |  |
 *          |   |  +---------mul---------+  |
 *          |   |                           |
 *          |   +------------transdata------+
 *         \|/
 *  +---------------------------------------+
 *  | data--partitioned_call---netoutput    |
 *  |              |                        |
 *  |             \|/                       |
 *  | +-----------------------------------+ |
 *  | |                       0           | |
 *  | | data-mul-+--transdata--netoutput  | |
 *  | |          |             1| 2| 3|   | |
 *  | |          +---cast-------+  |  |   | |
 *  | |          |                 |  |   | |
 *  | |          +---add-----------+  |   | |
 *  | |          |                    |   | |
 *  | |          +--------------------+   | |
 *  | +-----------------------------------+ |
 *  +---------------------------------------+
 *             ||
 *             \/
 *                       0     0
 * data--partitioned_call-+-----mul--netoutput
 *          |  3|  2|  1|           1| 2| 3|
 *          |   |   |   +------------+  |  |
 *          |   |   |                   |  |
 *          |   |   +-------mul---------+  |
 *          |   |                          |
 *          |   +--------------------------+
 *         \|/
 *  +--------------------------------------------------+
 *  | data--partitioned_call---netoutput               |
 *  |              |                                   |
 *  |             \|/                                  |
 *  | +----------------------------------------------+ |
 *  | |          +------------add-----------+        | |
 *  | |          |                     0   2|        | |
 *  | | data-mul-+--transdata--+----------netoutput  | |
 *  | |                        |          1|   3|    | |
 *  | |                        +---cast----+    |    | |
 *  | |                        |                |    | |
 *  | |                        |                |    | |
 *  | |                        +----------------+    | |
 *  | +----------------------------------------------+ |
 *  +--------------------------------------------------+
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, DiffGraph_ExtractTransdataToSubSubGraph_DoFusion) {
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub_data", sub_data)->NODE("sub_mul", MUL)->NODE("sub_transdata", TRANSDATA)
                               ->NODE("sub_netoutput", NETOUTPUT));
                     CHAIN(NODE("sub_mul")->EDGE(0, 0)->NODE("sub_cast", CAST)->NODE("sub_netoutput", NETOUTPUT));
                     CHAIN(NODE("sub_mul")->EDGE(0, 0)->NODE("sub_add", ADD)->NODE("sub_netoutput", NETOUTPUT));
                     CHAIN(NODE("sub_mul")->EDGE(0, 3)->NODE("sub_netoutput", NETOUTPUT));
                   };
  const auto middle_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(middle_sub) {
                          CHAIN(NODE("middle_data", middle_data)->NODE("middle_partitioned_call", PARTITIONEDCALL, sub_1)
                                    ->NODE("middle_netoutput",NETOUTPUT));
                          CHAIN(NODE("middle_partitioned_call", PARTITIONEDCALL)->NODE("middle_netoutput",NETOUTPUT));
                          CHAIN(NODE("middle_partitioned_call", PARTITIONEDCALL)->NODE("middle_netoutput",NETOUTPUT));
                          CHAIN(NODE("middle_partitioned_call", PARTITIONEDCALL)->NODE("middle_netoutput",NETOUTPUT));
                        };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->NODE("partitioned_call", PARTITIONEDCALL, middle_sub)
                            ->NODE("mul1", MUL)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("partitioned_call", PARTITIONEDCALL)
                            ->NODE("transdata1", TRANSDATA)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("partitioned_call", PARTITIONEDCALL)->NODE("mul2", MUL)
                            ->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("partitioned_call", PARTITIONEDCALL)->NODE("transdata2", TRANSDATA)
                            ->NODE("netoutput", NETOUTPUT));
                };
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);
  auto sub1_graph = ToComputeGraph(sub_1);
  const auto middle_sub_graph = compute_graph->GetSubgraph("middle_sub");
  ASSERT_NE(middle_sub_graph, nullptr);

  auto middle_partitioned_call_node = middle_sub_graph->FindNode("middle_partitioned_call");
  ASSERT_NE(middle_partitioned_call_node, nullptr);
  sub1_graph->SetParentGraph(compute_graph);
  sub1_graph->SetParentNode(middle_partitioned_call_node);
  compute_graph->AddSubGraph(sub1_graph);

  compute_graph->TopologicalSorting();

  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"sub_transdata", "transdata1", "transdata2"}));

  NetoutputParentIndexes indexes{{"sub_netoutput", {0, 1, 2, 3}}, {"middle_netoutput", {0, 1, 2, 3}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));
  SameTransdataBreadthFusionPass pass;
  ge::graphStatus status = pass.Run(compute_graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);

  UtPaths root_graph_path = {{{"data", 0}, {"partitioned_call", 0}, {"mul1", 0}, {"netoutput", 0}},
                             {{"partitioned_call", 1}, {"netoutput", 0}},
                             {{"partitioned_call", 3}, {"netoutput", 0}},
                             {{"partitioned_call", 2}, {"mul2", 0}, {"netoutput", 0}}
  };
  ASSERT_TRUE(CheckConnection(compute_graph, root_graph_path));

  UtPaths path2 = {{{"sub_data", 0}, {"sub_mul", 0}, {"sub_add", 0}, {"sub_netoutput", 0}},
                   {{"sub_mul", 0}, {"sub_transdata", 1}},
                   {{"sub_transdata", 0}, {"sub_cast", 0}, {"sub_netoutput", 0}},
                   {{"sub_transdata", 0}, {"sub_netoutput", 0}}};
  ASSERT_TRUE(CheckConnection(sub1_graph, path2));

  std::vector<NodeIO> check_nodes = {{"sub_cast", {0}, {0}},
                                     {"partitioned_call", {}, {1,3}},
                                     {"sub_netoutput", {1,3}, {}}};
  ASSERT_TRUE(CheckTensorDesc(compute_graph, check_nodes));

  std::vector<NodeIO> not_changed_nodes = {
      {"sub_data", {0}, {0}},
      {"sub_mul", {0}, {0}},
      {"sub_add", {0}, {0}},
      {"sub_netoutput", {2}, {}},
      {"partitioned_call", {}, {0,2}},
      {"mul2", {0}, {0}}
  };
  ASSERT_TRUE(CheckNotChangedTensorDesc(compute_graph, not_changed_nodes));
}

/*
 *  data--case--trans-netoutput
 *           \
 *            +--add
 *
 * case has three sub graphs
 *
 *   +-------------------+  +-------------------+  +-------------------+
 *   |   case1           |  |   case2           |  |   case3           |
 *   |                   |  |                   |  |                   |
 *   |  sub1_data        |  |  sub2_data        |  |  sub2_data        |
 *   |    / \            |  |    / \            |  |    / \            |
 *   |    |  sub1_trans  |  |    |  mul         |  |    |  mul         |
 *   |    |   |          |  |    |   |          |  |    |   |          |
 *   |    |   mul        |  |    |   mul        |  |    |   mul        |
 *   |    |    |         |  |    |    |         |  |    |    |         |
 *   |  NetOutput        |  |  NetOutput        |  |  NetOutput        |
 *   +-------------------+  +-------------------+  +-------------------+
 *
 * GetRealInAnchors
 * case有3个子图，head节点在其中一个子图内，并且transdata在子图外，校验不能将transdata提取到其中一个子图内。
 * 如果父节点有多个子图，并且head在子图内，要求path不能通过netoutput延伸到子图外。因为如果transdata提取到某一子图内，那么其他子图就不等价了
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, MultiSubGraph_CanNotExtractTransdataToSubGraph_NoFusion) {
  const auto sub1_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub1_data", sub1_data)->NODE("sub1_netoutput", NETOUTPUT));
                     CHAIN(NODE("sub1_data", sub1_data)->EDGE(0, 0)->NODE("sub1_trans", ADD)
                               ->NODE("sub1_mul", ADD)->NODE("sub1_netoutput", NETOUTPUT));
                   };

  const auto sub2_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_2) {
                     CHAIN(NODE("sub2_data", sub2_data)->NODE("sub2_netoutput", NETOUTPUT));
                     CHAIN(NODE("sub2_data", sub2_data)->EDGE(0, 0)->NODE("sub2_mul1", ADD)
                               ->NODE("sub2_mul2", ADD)->NODE("sub2_netoutput", NETOUTPUT));
                   };

  const auto sub3_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_3) {
                     CHAIN(NODE("sub3_data", sub3_data)->NODE("sub3_netoutput", NETOUTPUT));
                     CHAIN(NODE("sub3_data", sub3_data)->EDGE(0, 0)->NODE("sub3_mul1", ADD)
                               ->NODE("sub3_mul2", ADD)->NODE("sub3_netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("case", CASE, sub_1, sub_2, sub_3)
                            ->NODE("transdata", TRANSDATA)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("case", CASE)->EDGE(1, 0)->NODE("add", ADD));
                };
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);

  NetoutputParentIndexes parent_indexes{{"sub1_netoutput", {0, 1}},
                                        {"sub2_netoutput", {0, 1}},
                                        {"sub3_netoutput", {0, 1}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, parent_indexes));
  SameTransdataBreadthFusionPass pass;
//  ge::graphStatus status = pass.Run(compute_graph);
//  EXPECT_EQ(ge::GRAPH_SUCCESS, status);

  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);
  std::vector<NodeOutIndex> path2 = {{"sub1_data", 0}, {"sub1_trans", 0}, {"sub1_mul", 0}};
  ASSERT_TRUE(CheckConnection(sub_graph_1, path2));
}

/*
 *
 * data--partitioned_call--+--transdata--netoutput
 *                         |                |
 *                         +----------------+
 *  +------------------------------------------------+
 *  | data--case----netoutput                        |
 *  |          \        |                            |
 *  |           +--add--+                            |
 *  |                                                |
 *  | case has two sub graphs                        |
 *  |                                                |
 *  |   +-------------------+  +-------------------+ |
 *  |   |   case1           |  |   case2           | |
 *  |   |                   |  |                   | |
 *  |   |  sub1_data        |  |  sub2_data        | |
 *  |   |     |             |  |    / \            | |
 *  |   |    add            |  |    |  |           | |
 *  |   |    / \            |  |    |  |           | |
 *  |   |    |  sub1_trans  |  |    |  mul         | |
 *  |   |    |   |          |  |    |   |          | |
 *  |   |   cast  mul        |  |    |   mul       | |
 *  |   |    |    |         |  |    |    |         | |
 *  |   |  NetOutput        |  |  NetOutput        | |
 *  |   +-------------------+  +-------------------+ |
 *  +------------------------------------------------+
 *
 * GetRealInAnchors
 * case有2个子图，head节点在其中一个子图内，并且transdata在子图外，校验不能将transdata提取到其中一个子图内。
 * 如果父节点有多个子图，并且head在子图内，要求path不能通过netoutput延伸到子图外。因为如果transdata提取到某一子图内，那么其他子图就不等价了
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, MultiSubGraph_CanNotExtractTransdataToSubSubGraph_NoFusion) {
  const auto sub1_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub1_data", sub1_data)->NODE("sub1_add", ADD));
                     CHAIN(NODE("sub1_add", ADD)->NODE("sub1_cast", CAST)->NODE("sub1_netoutput", NETOUTPUT));
                     CHAIN(NODE("sub1_add", ADD)->EDGE(0, 0)->NODE("sub1_trans", TRANSDATA)
                               ->NODE("sub1_mul", ADD)->NODE("sub1_netoutput", NETOUTPUT));
                   };

  const auto sub2_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_2) {
                     CHAIN(NODE("sub2_data", sub2_data)->NODE("sub2_netoutput", NETOUTPUT));
                     CHAIN(NODE("sub2_data", sub2_data)->EDGE(0, 0)->NODE("sub2_mul1", ADD)
                               ->NODE("sub2_mul2", ADD)->NODE("sub2_netoutput", NETOUTPUT));
                   };

  const auto middle_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(middle_sub) {
                  CHAIN(NODE("middle_data", middle_data)->EDGE(0, 0)->NODE("middle_case", CASE, sub_1, sub_2)
                            ->NODE("middle_netoutput", NETOUTPUT));
                  CHAIN(NODE("middle_case", CASE)->EDGE(1, 0)->NODE("middle_add", ADD)
                            ->NODE("middle_netoutput", NETOUTPUT));
                };

  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->NODE("partitioned_call", PARTITIONEDCALL, middle_sub)
                            ->NODE("transdata", TRANSDATA)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("partitioned_call", PARTITIONEDCALL)->NODE("netoutput", NETOUTPUT));
                };
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);
  const auto middle_sub_graph = compute_graph->GetSubgraph("middle_sub");
  ASSERT_NE(middle_sub_graph, nullptr);

  auto middle_case_node = middle_sub_graph->FindNode("middle_case");
  ASSERT_NE(middle_case_node, nullptr);

  auto sub1_graph = ToComputeGraph(sub_1);
  ASSERT_NE(sub1_graph, nullptr);
  sub1_graph->SetParentGraph(compute_graph);
  sub1_graph->SetParentNode(middle_case_node);
  compute_graph->AddSubGraph(sub1_graph);

  auto sub2_graph = ToComputeGraph(sub_2);
  ASSERT_NE(sub2_graph, nullptr);
  sub2_graph->SetParentGraph(compute_graph);
  sub2_graph->SetParentNode(middle_case_node);
  compute_graph->AddSubGraph(sub2_graph);

  NetoutputParentIndexes parent_indexes{{"sub1_netoutput", {0, 1}},
                                        {"sub2_netoutput", {0, 1}},
                                        {"middle_netoutput", {0, 1}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, parent_indexes));
  SameTransdataBreadthFusionPass pass;
  ge::graphStatus status = pass.Run(compute_graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);

  std::vector<NodeOutIndex> path2 = {{"sub1_add", 0}, {"sub1_cast", 0}};
  ASSERT_TRUE(CheckConnection(sub1_graph, path2));
}

/*
 *  data--case--mul--netoutput
 *
 * case has three sub graphs, 子图1和2中的transdata可融合，子图3和4中的transdata可融合
 *
 *   +-------------+   +-------------+    +-------------+   +-------------+
 *   |   case1     |   |   case2     |    |   case3     |   |   case4     |
 *   |             |   |             |    |             |   |             |
 *   |   data      |   |   data      |    |   data      |   |   data      |
 *   |    |        |   |    |        |    |    |        |   |    |        |
 *   |   cast      |   |   cast      |    | transdata   |   | transdata   |
 *   |    |        |   |    |        |    |    |        |   |    |        |
 *   |   transdata |   |   cast      |    |   add       |   |   add       |
 *   |    |        |   |    |        |    |    |        |   |    |        |
 *   |  netOutput  |   |  transdata  |    | transdata   |   |    |        |
 *   |             |   |    |        |    |   |         |   |    |        |
 *   |             |   |  netoutput  |    | netoutput   |   |  netoutput  |
 *   +-------------+   +-------------+    +-------------+   +-------------+
 *
 *           ||
 *           \/
 *        +--transdata1---+
 *        |               |
 *  data--+--transdata2--case--mul--netoutput
 *
 *   +-------------+   +-------------+    +-------------+   +-------------+
 *   |   case1     |   |   case2     |    |   case3     |   |   case4     |
 *   |             |   |             |    |             |   |             |
 *   | data  data  |   | data  data  |    | data  data  |   | data  data  |
 *   |    \        |   |   \         |    |      /      |   |      /      |
 *   |   cast      |   |   cast      |    |     |       |   |     |       |
 *   |    |        |   |    |        |    |     |       |   |     |       |
 *   |    |        |   |   cast      |    |    add      |   |    add      |
 *   |    |        |   |    |        |    |     |       |   |     |       |
 *   |  netOutput  |   |    |        |    | transdata   |   |     |       |
 *   |             |   |    |        |    |   |         |   |     |       |
 *   |             |   |  netoutput  |    | netoutput   |   |  netoutput  |
 *   +-------------+   +-------------+    +-------------+   +-------------+
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, MultiSubGraph_TwoSameTransdataGroups_DoFusion) {
  const auto sub1_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub1_data", sub1_data)->NODE("sub1_cast", CAST)->NODE("sub1_transdata", TRANSDATA)
                               ->NODE("sub1_netoutput", NETOUTPUT));
                   };

  const auto sub2_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_2) {
                     CHAIN(NODE("sub2_data", sub2_data)->NODE("sub2_cast1", CAST)->NODE("sub2_cast2", CAST)
                               ->NODE("sub2_transdata", TRANSDATA)->NODE("sub2_netoutput", NETOUTPUT));
                   };

  const auto sub3_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_3) {
                     CHAIN(NODE("sub3_data", sub3_data)->NODE("sub3_transdata", TRANSDATA)->NODE("sub3_add", ADD)
                               ->NODE("sub3_transdata2", TRANSDATA)->NODE("sub3_netoutput", NETOUTPUT));
                   };
  const auto sub4_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_4) {
                     CHAIN(NODE("sub4_data", sub4_data)->NODE("sub4_transdata", TRANSDATA)->NODE("sub4_add", ADD)
                               ->NODE("sub4_netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("case", CASE, sub_1, sub_2, sub_3, sub_4)
                            ->NODE("mul", MUL)->NODE("netoutput", NETOUTPUT));
                };
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();

  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"sub1_transdata", "sub2_transdata"}));
  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"sub3_transdata", "sub4_transdata"}, FORMAT_CN));

  NetoutputParentIndexes parent_indexes{{"sub1_netoutput", {0}},
                                        {"sub2_netoutput", {0}},
                                        {"sub3_netoutput", {0}},
                                        {"sub4_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, parent_indexes));
  SameTransdataBreadthFusionPass pass;
  ge::graphStatus status = pass.Run(compute_graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);

  UtPaths paths1{{{"data", 0}, {"sub1_transdata", 0}, {"case", 0}, {"mul", 0}},
                 {{"data", 0}, {"sub3_transdata", 0}, {"case", 0}, {"mul", 0}}};
  ASSERT_TRUE(CheckConnection(compute_graph, paths1));

  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);
  std::vector<NodeOutIndex> path2 = {{"sub1_data", 0}, {"sub1_cast", 0}, {"sub1_netoutput", 0}};
  ASSERT_TRUE(CheckConnection(sub_graph_1, path2));

  const auto sub_graph_2 = compute_graph->GetSubgraph("sub_2");
  ASSERT_NE(sub_graph_2, nullptr);
  std::vector<NodeOutIndex> path3 = {{"sub2_data", 0}, {"sub2_cast1", 0}, {"sub2_cast2", 0}, {"sub2_netoutput", 0}};
  ASSERT_TRUE(CheckConnection(sub_graph_2, path3));

  const auto sub_graph_3 = compute_graph->GetSubgraph("sub_3");
  ASSERT_NE(sub_graph_3, nullptr);
  std::vector<NodeOutIndex> path4 = {{"sub_3_transdata_fusion_arg_1", 0}, {"sub3_add", 0}, {"sub3_transdata2", 0}, {"sub3_netoutput", 0}};
  ASSERT_TRUE(CheckConnection(sub_graph_3, path4));

  const auto sub_graph_4 = compute_graph->GetSubgraph("sub_4");
  ASSERT_NE(sub_graph_4, nullptr);
  std::vector<NodeOutIndex> path5 = {{"sub_4_transdata_fusion_arg_1", 0}, {"sub4_add", 0}, {"sub4_netoutput", 0}};
  ASSERT_TRUE(CheckConnection(sub_graph_4, path5));
}


/*
 *  data--case--mul--netoutput
 *
 *   +-------------+   +-------------+    +-----------------+   +-----------------+
 *   |   case1     |   |   case2     |    |   case3         |   |   case4         |
 *   |             |   |             |    |                 |   |                 |
 *   |   data      |   |   data      |    |     data        |   |     data        |
 *   |    |        |   |    |        |    |  /   |   \      |   |  /   |   \      |
 *   |   cast      |   |   cast      |    |cast trans trans |   |cast trans trans |
 *   |    |        |   |    |        |    |  |    |    |    |   |  |    |    |    |
 *   |   transdata |   |   transdata |    |trans add  add   |   |trans add  add   |
 *   +-------------+   +-------------+    +-----------------+   +-----------------+
 *
 *           ||
 *           \/
 *  data--transdata-case--mul--netoutput
 *
 *   +-------------+   +-------------+    +-----------------+   +-----------------+
 *   |   case1     |   |   case2     |    |   case3         |   |   case3         |
 *   |             |   |             |    |                 |   |                 |
 *   |   data      |   |   data      |    |     data        |   |     data        |
 *   |    |        |   |    |        |    |  /    |   \     |   |  /    |   \     |
 *   |   cast      |   |   cast      |    |cast  add  add   |   |cast  add  add   |
 *   +-------------+   +-------------+    +-----------------+   +-----------------+
 *
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, MultiSubGraph_OneTransdataGroup_DoFusion) {
  const auto sub1_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub1_data", sub1_data)->NODE("sub1_cast", CAST)->NODE("sub1_transdata", TRANSDATA)
                               ->NODE("sub1_netoutput", NETOUTPUT));
                   };

  const auto sub2_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_2) {
                     CHAIN(NODE("sub2_data", sub2_data)->NODE("sub2_cast", CAST)->NODE("sub2_transdata", TRANSDATA)
                               ->NODE("sub2_netoutput", NETOUTPUT));
                   };

  const auto sub3_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_3) {
                     CHAIN(NODE("sub3_data", sub3_data)->NODE("sub3_cast", CAST)->NODE("sub3_transdata1", TRANSDATA)
                               ->NODE("sub3_netoutput", NETOUTPUT));
                     CHAIN(NODE("sub3_data", sub3_data)->EDGE(0, 0)->NODE("sub3_transdata2", TRANSDATA)->NODE("sub3_add", ADD));
                     CHAIN(NODE("sub3_data", sub3_data)->EDGE(0, 0)->NODE("sub3_transdata3", TRANSDATA)->NODE("sub3_apply_momentum", ADD));
                   };
  const auto sub4_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_4) {
                     CHAIN(NODE("sub4_data", sub4_data)->NODE("sub4_cast", CAST)->NODE("sub4_transdata1", TRANSDATA)
                               ->NODE("sub4_netoutput", NETOUTPUT));
                     CHAIN(NODE("sub4_data", sub4_data)->EDGE(0, 0)->NODE("sub4_transdata2", TRANSDATA)->NODE("sub4_add", ADD));
                     CHAIN(NODE("sub4_data", sub4_data)->EDGE(0, 0)->NODE("sub4_transdata3", TRANSDATA)->NODE("sub4_apply_momentum", ADD));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("case", CASE, sub_1, sub_2, sub_3, sub_4)
                            ->NODE("mul", MUL)->NODE("netoutput", NETOUTPUT));
                };
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"sub1_transdata", "sub2_transdata",
                                                     "sub3_transdata1", "sub3_transdata2", "sub3_transdata3",
                                                     "sub4_transdata1", "sub4_transdata2", "sub4_transdata3"}));

  NetoutputParentIndexes parent_indexes{{"sub1_netoutput", {0}},
                                        {"sub2_netoutput", {0}},
                                        {"sub3_netoutput", {0}},
                                        {"sub4_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, parent_indexes));
  SameTransdataBreadthFusionPass pass;
  ge::graphStatus status = pass.Run(compute_graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);

  UtPaths paths1{{{"data", 0}, {"sub1_transdata", 0}, {"case", 0}, {"mul", 0}}};
  ASSERT_TRUE(CheckConnection(compute_graph, paths1));

  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);
  std::vector<NodeOutIndex> path2 = {{"sub1_data", 0}, {"sub1_cast", 0}, {"sub1_netoutput", 0}};
  ASSERT_TRUE(CheckConnection(sub_graph_1, path2));

  const auto sub_graph_2 = compute_graph->GetSubgraph("sub_2");
  ASSERT_NE(sub_graph_2, nullptr);
  std::vector<NodeOutIndex> path3 = {{"sub2_data", 0}, {"sub2_cast", 0}, {"sub2_netoutput", 0}};
  ASSERT_TRUE(CheckConnection(sub_graph_2, path3));

  const auto sub_graph_3 = compute_graph->GetSubgraph("sub_3");
  ASSERT_NE(sub_graph_3, nullptr);
  UtPaths path4 = {{{"sub3_data", 0}, {"sub3_cast", 0}},
                   {{"sub3_data", 0}, {"sub3_add", 0}},
                   {{"sub3_data", 0}, {"sub3_apply_momentum", 0}}};
  ASSERT_TRUE(CheckConnection(sub_graph_3, path4));

  const auto sub_graph_4 = compute_graph->GetSubgraph("sub_4");
  ASSERT_NE(sub_graph_4, nullptr);
  UtPaths path5 = {{{"sub4_data", 0}, {"sub4_cast", 0}},
                   {{"sub4_data", 0}, {"sub4_add", 0}},
                   {{"sub4_data", 0}, {"sub4_apply_momentum", 0}}};
  ASSERT_TRUE(CheckConnection(sub_graph_4, path5));
}

/*
 *  data--case--mul--netoutput
 *
 * case has three sub graphs, 子图1和2中的transdata可融合，子图3中的transdataformat不一致，不可融合
 *
 *   +-------------+   +-------------+    +-------------+
 *   |   case1     |   |   case2     |    |   case3     |
 *   |             |   |             |    |             |
 *   |   data      |   |   data      |    |   data      |
 *   |    |        |   |    |        |    |    |        |
 *   |   cast      |   |   cast      |    | transdata   |
 *   |    |        |   |    |        |    |    |        |
 *   |   transdata |   |   cast      |    |   add       |
 *   |    |        |   |    |        |    |    |        |
 *   |  netOutput  |   |  transdata  |    | transdata   |
 *   |             |   |    |        |    |   |         |
 *   |             |   |  netoutput  |    | netoutput   |
 *   +-------------+   +-------------+    +-------------+
 *
 *           ||
 *           \/
 *        +--------------+
 *        |              |
 *  data--+--transdata2--case--mul--netoutput
 *
 *   +-------------+   +-------------+    +-------------+
 *   |   case1     |   |   case2     |    |   case3     |
 *   |             |   |             |    |             |
 *   | data  data  |   | data  data  |    | data  data  |
 *   |    \        |   |    \        |    |        /    |
 *   |   cast      |   |   cast      |    | transdata   |
 *   |    |        |   |    |        |    |    |        |
 *   |    |        |   |   cast      |    |   add       |
 *   |    |        |   |    |        |    |    |        |
 *   |  netOutput  |   |    |        |    | transdata   |
 *   |             |   |    |        |    |   |         |
 *   |             |   |  netoutput  |    | netoutput   |
 *   +-------------+   +-------------+    +-------------+
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, MultiSubGraph_HaveNotFusedTransdata_DoFusion) {
  const auto sub1_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub1_data", sub1_data)->NODE("sub1_cast", CAST)->NODE("sub1_transdata", TRANSDATA)
                               ->NODE("sub1_netoutput", NETOUTPUT));
                   };

  const auto sub2_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_2) {
                     CHAIN(NODE("sub2_data", sub2_data)->NODE("sub2_cast1", CAST)->NODE("sub2_cast2", CAST)
                               ->NODE("sub2_transdata", TRANSDATA)->NODE("sub2_netoutput", NETOUTPUT));
                   };

  const auto sub3_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_3) {
                     CHAIN(NODE("sub3_data", sub3_data)->NODE("sub3_transdata1", TRANSDATA)->NODE("sub3_add", ADD)
                               ->NODE("sub3_transdata2", TRANSDATA)->NODE("sub3_netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("case", CASE, sub_1, sub_2, sub_3)
                            ->NODE("mul", MUL)->NODE("netoutput", NETOUTPUT));
                };
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();

  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"sub1_transdata", "sub2_transdata"}));
  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"sub3_transdata1"}, FORMAT_CN));

  NetoutputParentIndexes parent_indexes{{"sub1_netoutput", {0}},
                                        {"sub2_netoutput", {0}},
                                        {"sub3_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, parent_indexes));
  SameTransdataBreadthFusionPass pass;
  ge::graphStatus status = pass.Run(compute_graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);

  UtPaths paths1{{{"data", 0}, {"sub1_transdata", 0}, {"case", 0}, {"mul", 0}},
                 {{"data", 0}, {"case", 0}, {"mul", 0}}};
  ASSERT_TRUE(CheckConnection(compute_graph, paths1));

  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);
  std::vector<NodeOutIndex> path2 = {{"sub_1_transdata_fusion_arg_1", 0}, {"sub1_cast", 0}, {"sub1_netoutput", 0}};
  ASSERT_TRUE(CheckConnection(sub_graph_1, path2));

  const auto sub_graph_2 = compute_graph->GetSubgraph("sub_2");
  ASSERT_NE(sub_graph_2, nullptr);
  std::vector<NodeOutIndex> path3 = {{"sub_2_transdata_fusion_arg_1", 0}, {"sub2_cast1", 0}, {"sub2_cast2", 0}, {"sub2_netoutput", 0}};
  ASSERT_TRUE(CheckConnection(sub_graph_2, path3));

  const auto sub_graph_3 = compute_graph->GetSubgraph("sub_3");
  ASSERT_NE(sub_graph_3, nullptr);
  std::vector<NodeOutIndex> path4 = {{"sub3_data", 0}, {"sub3_transdata1", 0}, {"sub3_add", 0}, {"sub3_transdata2", 0}, {"sub3_netoutput", 0}};
  ASSERT_TRUE(CheckConnection(sub_graph_3, path4));
}

/*
 *         +--add--+
 *         |ctrl   |ctrl
 * op--+--cast---transdata1--mul--netoutput
 *     |
 *     +--transdata2
 *
 *         ||
 *         \/
 *                    +--add--+
 *                    |ctrl   |ctrl
 * op--transdata----cast------mul--netoutput
 *
 * transdata should not fusion cause in contrl not same
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, SameGraphDoFusion_CheckNoLoop) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("cast", CAST)->NODE("transdata1", TRANSDATA)
                            ->NODE("mul", MUL)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data", DATA)->EDGE(0,0))->NODE("transdata2", TRANSDATA)->NODE("netoutput", NETOUTPUT);
                  CHAIN(NODE("data2", DATA)->NODE("add", ADD)->NODE("netoutput", NETOUTPUT));
                };
  auto compute_graph = ToComputeGraph(g1);
  auto cast_node = compute_graph->FindNode("cast");
  ASSERT_NE(cast_node, nullptr);
  auto add_node = compute_graph->FindNode("add");
  ASSERT_NE(add_node, nullptr);
  auto trans1_node = compute_graph->FindNode("transdata1");
  ASSERT_NE(trans1_node, nullptr);
  cast_node->GetOutControlAnchor()->LinkTo(add_node->GetInControlAnchor());
  add_node->GetOutControlAnchor()->LinkTo(trans1_node->GetInControlAnchor());
  compute_graph->TopologicalSorting();

  SameTransdataBreadthFusionPass pass;
  ge::graphStatus status = pass.Run(compute_graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);

  // will fail if has loop
  EXPECT_EQ(compute_graph->TopologicalSorting(), SUCCESS);

  UtPaths paths1{{{"data", 0}, {"cast", 0}, {"transdata1", 0}, {"mul", 0}},
                 {{"data", 0}, {"transdata2", 0}, {"netoutput", 0}}};
  ASSERT_TRUE(CheckConnection(compute_graph, paths1));
}


/*
 *                     ctrl
 * a--+--transdata1---b---transdata2--c
 *    |                     |
 *    +---------------------+
 *
 *         ||
 *         \/
 *
 * a---transdata1--+---b
 *                 |    \ctrl
 *                 +-----c
 * 如果保留transdata2会成环
 *
 * transdata should not fusion cause in contrl not same
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, SameGraphDoFusion_GetKeepTransdataPathIndex_CheckNoLoop) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", DATA)->EDGE(0, 0)->NODE("transdata1", TRANSDATA)
                            ->NODE("b", MUL)->Ctrl()->NODE("transdata2", TRANSDATA)->EDGE(0, 0)->NODE("c", MUL));
                  CHAIN(NODE("a", DATA)->EDGE(0, 0))->NODE("transdata2", TRANSDATA);
                };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();

  SameTransdataBreadthFusionPass pass;
  ge::graphStatus status = pass.Run(compute_graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);

  // will fail if has loop
  EXPECT_EQ(compute_graph->TopologicalSorting(), SUCCESS);
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 5);
}

/*
 * op--+--cast---transdata--netoutput
 *     |
 *     +--cast---transdata
 *
 *         ||
 *         \/
 *
 * op--transdata--+--cast--mul--netoutput
 *                |
 *                +--cast
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, SameGraphDoFusion_CheckDataType) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("cast1", CAST)->NODE("transdata1", TRANSDATA)
                            ->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data", DATA)->EDGE(0,0)->NODE("cast2", CAST)->NODE("transdata2", TRANSDATA)
                      ->NODE("netoutput", NETOUTPUT));
                };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();

  auto data_node = compute_graph->FindNode("data");
  ASSERT_NE(data_node, nullptr);
  data_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);

  auto cast1_node = compute_graph->FindNode("cast1");
  ASSERT_NE(cast1_node, nullptr);
  cast1_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  cast1_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);

  auto cast2_node = compute_graph->FindNode("cast2");
  ASSERT_NE(cast2_node, nullptr);
  cast2_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  cast2_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);

  auto trans1_node = compute_graph->FindNode("transdata1");
  ASSERT_NE(trans1_node, nullptr);
  trans1_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  trans1_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);

  auto trans2_node = compute_graph->FindNode("transdata2");
  ASSERT_NE(trans2_node, nullptr);
  trans2_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  trans2_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);

  SameTransdataBreadthFusionPass pass;
  ge::graphStatus status = pass.Run(compute_graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);

  UtPaths paths1{{{"data", 0}, {"transdata2", 0}, {"cast1", 0}, {"netoutput", 0}},
                 {{"data", 0}, {"transdata2", 0}, {"cast2", 0}}};
  ASSERT_TRUE(CheckConnection(compute_graph, paths1));

  ASSERT_EQ(trans2_node->GetOpDesc()->GetInputDesc(0).GetDataType(), DT_FLOAT);
  ASSERT_EQ(trans2_node->GetOpDesc()->GetOutputDesc(0).GetDataType(), DT_FLOAT);
}

/*
 * todo:
 * AddNewInputForWrapper
 * test1
 * head连接case，case有2个子图，一个子图中data一个输入直连netoutput的两个输入，case的输出连接transdata和一个不可融合节点。
 * 另外一个子图data不直连netoutput外。校验所有子图中都有一个新增data节点，并且校验tensordesc
 *
 * test2
 * case的2个子图，子图1中data连接不可融合节点，子图2中data连接可融合节点。校验子图2中的data的parent index得修改
 * test3
 * case的2个子图，子图1中data连接不可融合节点，子图2中data连接partitinoed_call,对应子图只有一个data连接netoutput，
 * partitioned_call只有一个输出连接可融合节点。校验子图2中的data的parent index得修改
 *
 * test4
 * case的2个子图，子图1中data连接一个可融合节点和一个不可融合节点，校验在子图1中新增data节点连接到可融合节点上
 *
 * test5
 * case的两个子图，子图1中data连接netoutput/可融合节点/不可融合节点，子图2 data连接Netoutput。
 * 子图1netoutput对应的实际输出为不可融合节点。子图2netoutput对应的实际输出为可融合节点。校验子图1中新增了data节点，且parent index正确，
 * 子图1中netoutput没有新增输入。子图2中也新增了data节点连接到netoutput上，且parent index正确，子图2中原data节点如果没有输出边就删除。
 *
 *
 * AddNewPathToTransdataForDiffGraph
 * test1
 * head连接两个partitioned_call，也就对应两个子图，子图1中data连接可融合节点，子图2中data连接不可融合节点，
 * 校验partitioned_call都新增了输入边，但是子图中没有新增data节点，子图1中data的parent index修改成新增的index
 *
 * test2
 * 基本场景，head连接一个partitioned_call，对应1个子图，子图内data连接两个节点，一个是可融合节点，一个是不可融合节点，
 * 校验新增了data节点连接到可融合节点上
 *
 * test3
 * 递归场景，head连接partitioned_call1，子图1中data连接netoutput，实际输出连接partitioned_call2,
 * 子图2中data连接可融合和不可融合共两个节点。校验两个partioned_call都新增了输入，且两个子图中都新增了data节点，且parent index正确。
 *
 */

TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, CheckOpSupported_Failed_WhenCastNotSupportNewFormat) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
  (void)CreateGraph(graph);

  ge::SameTransdataBreadthFusionPass pass;
  ge::graphStatus status = pass.Run(graph);
  const auto head = graph->FindNode("Data4D");
  ASSERT_NE(head, nullptr);
  ASSERT_EQ(head->GetOutDataNodesSize(), 2U);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);
}

TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, CheckOpSupported_Failed_WhenTransdataNotSupportNewDataType) {
  /*
            |---cast1---cast2---transdata1
            |
    Node4D--|---cast3---transdata2
  */
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // Node4D
  ge::NodePtr node_4d = NodeBuilder("node_4d", DATA)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NCHW, DT_INT16)
      .Build(graph);

  ge::NodePtr cast1 = NodeBuilder("cast1", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT16)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .Build(graph);

  ge::NodePtr cast2 = NodeBuilder("cast2", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast3 = NodeBuilder("cast3", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT16)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .Build(graph);

  ge::NodePtr transdata1 = NodeBuilder("transdata1", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr transdata2 = NodeBuilder("transdata2", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_softmax_2 = NodeBuilder("softemax_2", SOFTMAX)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast1->GetOutDataAnchor(0), cast2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast2->GetOutDataAnchor(0), transdata1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata1->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast3->GetOutDataAnchor(0), transdata2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata2->GetOutDataAnchor(0), node_softmax_2->GetInDataAnchor(0));

  //set tensor desc
  ASSERT_TRUE(SetTransDataTensorDesc(graph, {"transdata1", "transdata2"}));

  ge::SameTransdataBreadthFusionPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(node_4d->GetOutDataNodes().size(), 2);
}

TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, test_same_transop_fusion_pass_not_cause_loop_unittest) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("cast1", CAST));
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("cast2", CAST));
    CHAIN(NODE("cast1", CAST)->EDGE(0, 0)->NODE("transdata1", TRANSDATA));
    CHAIN(NODE("cast2", CAST)->EDGE(0, 0)->NODE("transdata2", TRANSDATA));
    CHAIN(NODE("transdata1", TRANSDATA)->EDGE(0, 0)->NODE("transdata3", TRANSDATA));
    CHAIN(NODE("transdata3", TRANSDATA)->EDGE(0, 0)->NODE("netoutput1", NETOUTPUT));
    CHAIN(NODE("transdata2", TRANSDATA)->EDGE(0, 0)->NODE("abs", ABSVAL));
    CHAIN(NODE("abs", ABSVAL)->CTRL_EDGE()->NODE("cast1", CAST));
    CHAIN(NODE("abs2", ABSVAL)->CTRL_EDGE()->NODE("transdata2", TRANSDATA));
    CHAIN(NODE("abs1", ABSVAL)->CTRL_EDGE()->NODE("transdata1", TRANSDATA));
  };
  auto graph = ToComputeGraph(g1);
  ge::SameTransdataBreadthFusionPass pass;
  ge::graphStatus status = pass.Run(graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);
  EXPECT_EQ(graph->TopologicalSorting(), SUCCESS);
}

TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, SameGraph_DoFusion_HeadConnectToTwoTransDataDirectly) {
  /*
            |---transdata---cast---A
            |
    Node4D--|---transdata---cast---B

            ||
            \/
                            |---cast---A
                            |
    Node4D--|---transdata---|---cast---B
  */
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // Node4D
  ge::NodePtr node_4d = NodeBuilder("node_4d", DATA)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
      .Build(graph);
  // cast fp32 -> fp16
  ge::NodePtr cast_fp32_2_fp16_1 = NodeBuilder("cast_4d_fp32_2_fp16_1", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16_2 = NodeBuilder("cast_4d_fp32_2_fp16_2", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_5d_2_4hd_1 = NodeBuilder("5d_2_4d_1", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .Build(graph);
  ge::NodePtr node_5d_2_4hd_2 = NodeBuilder("5d_2_4d_2", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_softmax_2 = NodeBuilder("softemax_2", SOFTMAX)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), node_5d_2_4hd_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), node_5d_2_4hd_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_5d_2_4hd_1->GetOutDataAnchor(0), cast_fp32_2_fp16_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_5d_2_4hd_2->GetOutDataAnchor(0), cast_fp32_2_fp16_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_1->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_2->GetOutDataAnchor(0), node_softmax_2->GetInDataAnchor(0));

  //set tensor desc
  ASSERT_TRUE(SetTransDataTensorDesc(graph, {"5d_2_4d_1", "5d_2_4d_2"}));

  ge::SameTransdataBreadthFusionPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(node_4d->GetOutDataNodes().size(), 1);

  // check connetion
  std::vector<NodeOutIndex> path = {{"node_4d", 0}, {"5d_2_4d_2", 0}, {"cast_4d_fp32_2_fp16_2", 0}};
  ASSERT_TRUE(CheckConnection(graph, path));
  std::vector<NodeOutIndex> path2 = {{"node_4d", 0}, {"5d_2_4d_2", 0}, {"cast_4d_fp32_2_fp16_1", 0}};
  ASSERT_TRUE(CheckConnection(graph, path2));

  // check tensor desc
  std::vector<NodeIO> check_nodes = {{"5d_2_4d_2", {}, {0}}};
  ASSERT_TRUE(CheckTensorDesc(graph, check_nodes));

  std::vector<NodeIO> not_changed_nodes = {{"node_4d", {}, {0}},
                                           {"5d_2_4d_2", {0}, {}}};
  ASSERT_TRUE(CheckNotChangedTensorDesc(graph, not_changed_nodes));
}

TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, test_transdata_no_out) {
  /*
            |---cast--transdata---A
            |
    Node4D--|---transdata---B
            |
            |---cast---transdata

            ||
            \/

                            |---cast---A
                            |
    Node4D--|---transdata---|---B
            |
            |---cast---transdata

  */
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // Node4D
  ge::NodePtr node_4d = NodeBuilder("node_4d", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
      .Build(graph);
  // cast fp32 -> fp16
  ge::NodePtr cast_fp32_2_fp16_1 = NodeBuilder("cast_4d_fp32_2_fp16_1", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16_2 = NodeBuilder("cast_4d_fp32_2_fp16_2", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_4d_2_5hd_1 = NodeBuilder("4d_2_5hd_1", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr node_4d_2_5hd_2 = NodeBuilder("4d_2_5hd_2", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
      .Build(graph);
  ge::NodePtr node_4d_2_5hd_3 = NodeBuilder("4d_2_5hd_3", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), node_4d_2_5hd_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_1->GetOutDataAnchor(0), node_4d_2_5hd_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_1->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_2->GetOutDataAnchor(0), node_4d_2_5hd_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_2->GetOutDataAnchor(0), node_relu_2->GetInDataAnchor(0));

  ge::SameTransdataBreadthFusionPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(node_4d->GetOutDataNodes().size(), 2);
  EXPECT_EQ(node_4d_2_5hd_2->GetOutDataNodes().size(), 2);
  EXPECT_EQ(cast_fp32_2_fp16_2->GetOutDataNodes().size(), 1);
}

TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, SeGraph_CastConnectToOtherNode_NotFusion) {
  /*
                      |---C
                      |
            |---cast--|---transdata---A
            |
    Node4D--|---cast---cast---transdata---B

  */
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // Node4D
  ge::NodePtr node_4d = NodeBuilder("node_4d", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .Build(graph);
  // cast fp32 -> fp16
  ge::NodePtr cast_fp32_2_fp16_1 = NodeBuilder("cast_4d_fp32_2_fp16_1", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16_2 = NodeBuilder("cast_4d_fp32_2_fp16_2", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr cast_fp16_2_int8 = NodeBuilder("cast_fp16_2_int8", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT8)
      .Build(graph);

  ge::NodePtr node_4d_2_5hd_1 = NodeBuilder("4d_2_5hd_1", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr node_4d_2_5hd_2 = NodeBuilder("4d_2_5hd_2", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_1->GetOutDataAnchor(0), node_4d_2_5hd_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_1->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_2->GetOutDataAnchor(0), cast_fp16_2_int8->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_int8->GetOutDataAnchor(0), node_4d_2_5hd_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_2->GetOutDataAnchor(0), node_relu_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_1->GetOutDataAnchor(0), node_relu_3->GetInDataAnchor(0));

  ge::SameTransdataBreadthFusionPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(node_4d->GetOutDataNodes().size(), 2);
  EXPECT_EQ(node_4d_2_5hd_1->GetOutDataNodes().size(), 1);
  EXPECT_EQ(cast_fp32_2_fp16_1->GetOutDataNodes().size(), 2);
}

TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, test_no_transdata) {
  /*
            |---cast---transpose---A
            |
    Node4D--|---transpose---B
            |
            |---C

            ||
            \/

            |---cast---transpose---A
            |
    Node4D--|---transpose---B
            |
            |---C

  */
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // Node4D
  ge::NodePtr node_4d = NodeBuilder("node_4d", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .Build(graph);

  // cast fp32 -> fp16
  ge::NodePtr cast_fp32_2_fp16_1 = NodeBuilder("cast_4d_fp32_2_fp16_1", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr transpose_1 = NodeBuilder("transpose_1", TRANSPOSE)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT16)
      .Build(graph);


  ge::NodePtr transpose_2 = NodeBuilder("transpose_2", TRANSPOSE)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT)
      .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
      .Build(graph);

  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), transpose_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), node_relu_3->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_1->GetOutDataAnchor(0), transpose_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transpose_1->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transpose_2->GetOutDataAnchor(0), node_relu_2->GetInDataAnchor(0));

  ge::SameTransdataBreadthFusionPass pass;
  Status status = pass.Run(graph);

  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(node_4d->GetOutDataNodes().size(), 3);
}

TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, test_different_transdata) {
  /*
            |---cast---transdata---A
            |
    Node4D--|---cast---transdata---B
            |
            |---transdata---C

            ||
            \/

            |---cast---transdata---A
            |
    Node4D--|---transdata---|---cast---B
                            |---C

  */
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // Node4D
  ge::NodePtr node_4d = NodeBuilder("node_4d", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .Build(graph);

  // cast fp32 -> fp16
  ge::NodePtr cast_fp32_2_fp16_1 = NodeBuilder("cast_4d_fp32_2_fp16_1", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr cast_fp32_2_fp16_2 = NodeBuilder("cast_4d_fp32_2_fp16_2", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);


  ge::NodePtr node_4d_2_5hd_1 = NodeBuilder("4d_2_5hd_1", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_FRACTAL_Z, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr node_4d_2_5hd_2 = NodeBuilder("4d_2_5hd_2", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_4d_2_5hd_3 = NodeBuilder("4d_2_5hd_3", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
      .Build(graph);

  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), node_4d_2_5hd_3->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_1->GetOutDataAnchor(0), node_4d_2_5hd_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_1->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_2->GetOutDataAnchor(0), node_4d_2_5hd_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_2->GetOutDataAnchor(0), node_relu_2->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_4d_2_5hd_3->GetOutDataAnchor(0), node_relu_3->GetInDataAnchor(0));

  ge::SameTransdataBreadthFusionPass pass;
  Status status = pass.Run(graph);

  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(node_4d->GetOutDataNodes().size(), 2);
  EXPECT_EQ(node_4d_2_5hd_3->GetOutDataNodes().size(), 2);
}

TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, test_transdata2_cast1_simple) {
  /*

            |---cast---transdata---A
    Node4D--|
            |---transdata---B
  */
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // Node4D
  ge::NodePtr node_4d = NodeBuilder("node_4d", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .Build(graph);

  // cast fp32 -> fp16
  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_4d_fp32_2_fp16", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);


  ge::NodePtr node_4d_2_5hd_1 = NodeBuilder("4d_2_5hd_1", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_4d_2_5hd_2 = NodeBuilder("4d_2_5hd_2", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
      .Build(graph);

  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), node_4d_2_5hd_2->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), node_4d_2_5hd_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_1->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_2->GetOutDataAnchor(0), node_relu_2->GetInDataAnchor(0));

  ge::SameTransdataBreadthFusionPass pass;
  Status status = pass.Run(graph);

  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(node_4d->GetOutDataNodes().size(), 1);
  EXPECT_EQ(node_4d_2_5hd_2->GetOutDataNodes().size(), 2);
}

TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, test_transdata3_cast2_simple) {
  /*
            |---cast---transdata---A
            |
    Node4D--|---cast---transdata---B
            |
            |---transdata---C

            ||
            \/

            |               |---cast---A
            |               |
    Node4D--|---transdata---|---cast---B
                            |---C

  */
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // Node4D
  ge::NodePtr node_4d = NodeBuilder("node_4d", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .Build(graph);

  // cast fp32 -> fp16
  ge::NodePtr cast_fp32_2_fp16_1 = NodeBuilder("cast_4d_fp32_2_fp16_1", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr cast_fp32_2_fp16_2 = NodeBuilder("cast_4d_fp32_2_fp16_2", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);


  ge::NodePtr node_4d_2_5hd_1 = NodeBuilder("4d_2_5hd_1", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr node_4d_2_5hd_2 = NodeBuilder("4d_2_5hd_2", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_4d_2_5hd_3 = NodeBuilder("4d_2_5hd_3", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
      .Build(graph);

  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), node_4d_2_5hd_3->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_1->GetOutDataAnchor(0), node_4d_2_5hd_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_1->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_2->GetOutDataAnchor(0), node_4d_2_5hd_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_2->GetOutDataAnchor(0), node_relu_2->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_4d_2_5hd_3->GetOutDataAnchor(0), node_relu_3->GetInDataAnchor(0));

  ge::SameTransdataBreadthFusionPass pass;
  Status status = pass.Run(graph);

  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(node_4d->GetOutDataNodes().size(), 1);
  EXPECT_EQ(node_4d_2_5hd_3->GetOutDataNodes().size(), 3);
}

TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, test_transdata3_cast2_anchors_out3) {
  /*
            |---cast---transdata---A
            |
    Node4D--|---cast---transdata---B
            |
            |---transdata---C

            ||
            \/

            |---cast---transdata---A
            |
    Node4D--|---cast---transdata---B
            |
            |---transdata---C
  */
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // Node4D
  ge::NodePtr node_4d = NodeBuilder("node_4d", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .Build(graph);

  // cast fp32 -> fp16
  ge::NodePtr cast_fp32_2_fp16_1 = NodeBuilder("cast_4d_fp32_2_fp16_1", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr cast_fp32_2_fp16_2 = NodeBuilder("cast_4d_fp32_2_fp16_2", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);


  ge::NodePtr node_4d_2_5hd_1 = NodeBuilder("4d_2_5hd_1", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr node_4d_2_5hd_2 = NodeBuilder("4d_2_5hd_2", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_4d_2_5hd_3 = NodeBuilder("4d_2_5hd_3", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
      .Build(graph);

  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(1), cast_fp32_2_fp16_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(2), node_4d_2_5hd_3->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_1->GetOutDataAnchor(0), node_4d_2_5hd_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_1->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_2->GetOutDataAnchor(0), node_4d_2_5hd_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_2->GetOutDataAnchor(0), node_relu_2->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_4d_2_5hd_3->GetOutDataAnchor(0), node_relu_3->GetInDataAnchor(0));

  ge::SameTransdataBreadthFusionPass pass;
  Status status = pass.Run(graph);

  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(node_4d->GetOutDataNodes().size(), 3);
  EXPECT_EQ(node_4d_2_5hd_1->GetOutDataNodes().size(), 1);
  EXPECT_EQ(node_4d_2_5hd_2->GetOutDataNodes().size(), 1);
  EXPECT_EQ(node_4d_2_5hd_3->GetOutDataNodes().size(), 1);
}

TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, test_transdata2_cast3) {
  /*
            |---cast---transdata---A
            |
    Node4D--|---cast---cast---transdata---B

            ||
            \/

                        |---cast---A
                        |
    Node4D--transdata---|---cast---cast---B

  */
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // Node4D
  ge::NodePtr node_4d = NodeBuilder("node_4d", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .Build(graph);
  // cast fp32 -> fp16
  ge::NodePtr cast_fp32_2_fp16_1 = NodeBuilder("cast_4d_fp32_2_fp16_1", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16_2 = NodeBuilder("cast_4d_fp32_2_fp16_2", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr cast_fp16_2_int8 = NodeBuilder("cast_fp16_2_int8", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT8)
      .Build(graph);

  ge::NodePtr node_4d_2_5hd_1 = NodeBuilder("4d_2_5hd_1", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr node_4d_2_5hd_2 = NodeBuilder("4d_2_5hd_2", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_1->GetOutDataAnchor(0), node_4d_2_5hd_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_1->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_2->GetOutDataAnchor(0), cast_fp16_2_int8->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_int8->GetOutDataAnchor(0), node_4d_2_5hd_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_2->GetOutDataAnchor(0), node_relu_2->GetInDataAnchor(0));

  ge::SameTransdataBreadthFusionPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(node_4d->GetOutDataNodes().size(), 1);
  EXPECT_EQ(node_4d_2_5hd_2->GetOutDataNodes().size(), 2);
}


TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, SameGraph_HeadConnectToThreeCast_DoFusion) {
  /*
            |---cast---transdata---A
            |
    Node4D--|---cast---cast---transdata---B
            |
            |---cast---transdata---C
            ||
            \/

                        |---cast---A
                        |
    Node4D--transdata---|---cast---cast---B
                        |
                        |---cast---C

  */
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // Node4D
  ge::NodePtr node_4d = NodeBuilder("node_4d", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .Build(graph);
  // cast fp32 -> fp16
  ge::NodePtr cast_fp32_2_fp16_1 = NodeBuilder("cast_4d_fp32_2_fp16_1", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16_3 = NodeBuilder("cast_4d_fp32_2_fp16_3", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16_2 = NodeBuilder("cast_4d_fp32_2_fp16_2", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast_fp16_2_int8 = NodeBuilder("cast_fp16_2_int8", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT8)
      .Build(graph);

  ge::NodePtr node_4d_2_5hd_1 = NodeBuilder("4d_2_5hd_1", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr node_4d_2_5hd_2 = NodeBuilder("4d_2_5hd_2", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);
  ge::NodePtr node_4d_2_5hd_3 = NodeBuilder("4d_2_5hd_3", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_1->GetOutDataAnchor(0), node_4d_2_5hd_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_1->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_2->GetOutDataAnchor(0), cast_fp16_2_int8->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_int8->GetOutDataAnchor(0), node_4d_2_5hd_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_2->GetOutDataAnchor(0), node_relu_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_3->GetOutDataAnchor(0), node_4d_2_5hd_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_3->GetOutDataAnchor(0), node_relu_3->GetInDataAnchor(0));

  ASSERT_TRUE(SetTransDataTensorDesc(graph, {"4d_2_5hd_1", "4d_2_5hd_3", "4d_2_5hd_2"}));

  ge::SameTransdataBreadthFusionPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(node_4d->GetOutDataNodes().size(), 1);
  EXPECT_EQ(node_4d_2_5hd_3->GetOutDataNodes().size(), 3);

  std::vector<NodeOutIndex> path = {{"node_4d", 0}, {"4d_2_5hd_3", 0}, {"cast_4d_fp32_2_fp16_1", 0}};
  ASSERT_TRUE(CheckConnection(graph, path));

  std::vector<NodeOutIndex> path2 = {{"node_4d", 0}, {"4d_2_5hd_3", 0}, {"cast_4d_fp32_2_fp16_3", 0}};
  ASSERT_TRUE(CheckConnection(graph, path2));

  std::vector<NodeOutIndex> path3 = {{"node_4d", 0}, {"4d_2_5hd_3", 0}, {"cast_4d_fp32_2_fp16_2", 0}, {"cast_fp16_2_int8", 0}};
  ASSERT_TRUE(CheckConnection(graph, path3));

  std::vector<NodeIO> check_nodes = {{"cast_4d_fp32_2_fp16_1", {0}, {0}},
                                     {"cast_4d_fp32_2_fp16_3", {0}, {0}},
                                     {"cast_4d_fp32_2_fp16_2", {0}, {0}},
                                     {"cast_fp16_2_int8", {0}, {0}}};
  ASSERT_TRUE(CheckTensorDesc(graph, check_nodes));

  std::vector<NodeIO> not_changed_nodes = {{"node_4d", {}, {0}},
                                           {"4d_2_5hd_3", {0}, {}}};
  ASSERT_TRUE(CheckNotChangedTensorDesc(graph, not_changed_nodes));
}


TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, SameGraph_CastConnectToOtherNode_NoFusion) {
  /*
            |---cast---transdata---A
            |
    Node4D--|---cast---cast---transdata---B
            |           |
            |           |---C
            ||
            \/

  */
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // Node4D
  ge::NodePtr node_4d = NodeBuilder("node_4d", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .Build(graph);
  // cast fp32 -> fp16
  ge::NodePtr cast_fp32_2_fp16_1 = NodeBuilder("cast_4d_fp32_2_fp16_1", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16_2 = NodeBuilder("cast_4d_fp32_2_fp16_2", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast_fp16_2_int8 = NodeBuilder("cast_fp16_2_int8", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT8)
      .Build(graph);

  ge::NodePtr node_4d_2_5hd_1 = NodeBuilder("4d_2_5hd_1", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr node_4d_2_5hd_2 = NodeBuilder("4d_2_5hd_2", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_1->GetOutDataAnchor(0), node_4d_2_5hd_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_1->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_2->GetOutDataAnchor(0), cast_fp16_2_int8->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_int8->GetOutDataAnchor(0), node_4d_2_5hd_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_2->GetOutDataAnchor(0), node_relu_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_int8->GetOutDataAnchor(0), node_relu_3->GetInDataAnchor(0));

  ge::SameTransdataBreadthFusionPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(node_4d->GetOutDataNodes().size(), 2);
  EXPECT_EQ(node_4d_2_5hd_1->GetOutDataNodes().size(), 1);
}

TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, test_transdata_out_control) {
  /*
            |---cast---transdata---A
            |
    Node4D--|---cast---cast---transdata---B
                                       |
                                       |---C
            ||
            \/

                            |---cast---A
                            |
    Node4D--|---transdata---|---cast---cast---B
                                          |
                                          |---C


  */
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // Node4D
  ge::NodePtr node_4d = NodeBuilder("node_4d", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .Build(graph);
  // cast fp32 -> fp16
  ge::NodePtr cast_fp32_2_fp16_1 = NodeBuilder("cast_4d_fp32_2_fp16_1", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16_2 = NodeBuilder("cast_4d_fp32_2_fp16_2", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast_fp16_2_int8 = NodeBuilder("cast_fp16_2_int8", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT8)
      .Build(graph);

  ge::NodePtr node_4d_2_5hd_1 = NodeBuilder("4d_2_5hd_1", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr node_4d_2_5hd_2 = NodeBuilder("4d_2_5hd_2", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_1->GetOutDataAnchor(0), node_4d_2_5hd_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_1->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_2->GetOutDataAnchor(0), cast_fp16_2_int8->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_int8->GetOutDataAnchor(0), node_4d_2_5hd_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_2->GetOutDataAnchor(0), node_relu_2->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_4d_2_5hd_2->GetOutControlAnchor(), node_relu_3->GetInControlAnchor());

  ge::SameTransdataBreadthFusionPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(node_4d->GetOutDataNodes().size(), 1);
  EXPECT_EQ(node_4d_2_5hd_2->GetOutDataNodes().size(), 2);
  EXPECT_EQ(cast_fp16_2_int8->GetOutControlNodes().size(), 1);
}

TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, test_transdata_out_data_control) {
  /*
            |---cast---transdata---A
            |
    Node4D--|---cast---cast---transdata---B
                                      |
                                      |---C

            ||
            \/

                            |---cast---A
                            |
    Node4D--|---transdata---|---cast---cast---B
                                          |
                                          |---C


  */
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // Node4D
  ge::NodePtr node_4d = NodeBuilder("node_4d", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .Build(graph);
  // cast fp32 -> fp16
  ge::NodePtr cast_fp32_2_fp16_1 = NodeBuilder("cast_4d_fp32_2_fp16_1", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16_2 = NodeBuilder("cast_4d_fp32_2_fp16_2", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast_fp16_2_int8 = NodeBuilder("cast_fp16_2_int8", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT8)
      .Build(graph);

  ge::NodePtr node_4d_2_5hd_1 = NodeBuilder("4d_2_5hd_1", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr node_4d_2_5hd_2 = NodeBuilder("4d_2_5hd_2", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_1->GetOutDataAnchor(0), node_4d_2_5hd_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_1->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_2->GetOutDataAnchor(0), cast_fp16_2_int8->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_int8->GetOutDataAnchor(0), node_4d_2_5hd_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_2->GetOutDataAnchor(0), node_relu_2->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_4d_2_5hd_2->GetOutControlAnchor(), node_relu_3->GetInControlAnchor());

  ge::SameTransdataBreadthFusionPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(node_4d->GetOutDataNodes().size(), 1);
  EXPECT_EQ(node_4d_2_5hd_2->GetOutDataNodes().size(), 2);
  EXPECT_EQ(cast_fp16_2_int8->GetOutControlNodes().size(), 1);
}

TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, test_transdata_out_control_peer_in_data) {
  /*
            |---cast---transdata---A
            |
    Node4D--|---cast---cast---transdata---B
                                      |
                                      |---C

            ||
            \/

                            |---cast---A
                            |
    Node4D--|---transdata---|---cast---cast---B
                                          |
                                          |---C


  */
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // Node4D
  ge::NodePtr node_4d = NodeBuilder("node_4d", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .Build(graph);
  // cast fp32 -> fp16
  ge::NodePtr cast_fp32_2_fp16_1 = NodeBuilder("cast_4d_fp32_2_fp16_1", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16_2 = NodeBuilder("cast_4d_fp32_2_fp16_2", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast_fp16_2_int8 = NodeBuilder("cast_fp16_2_int8", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT8)
      .Build(graph);

  ge::NodePtr node_4d_2_5hd_1 = NodeBuilder("4d_2_5hd_1", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr node_4d_2_5hd_2 = NodeBuilder("4d_2_5hd_2", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp16_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_1->GetOutDataAnchor(0), node_4d_2_5hd_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_1->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_2->GetOutDataAnchor(0), cast_fp16_2_int8->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_int8->GetOutDataAnchor(0), node_4d_2_5hd_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_2->GetOutDataAnchor(0), node_relu_2->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_4d_2_5hd_2->GetOutControlAnchor(), node_relu_3->GetInDataAnchor(0));

  ge::SameTransdataBreadthFusionPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(node_4d->GetOutDataNodes().size(), 1);
  EXPECT_EQ(node_4d_2_5hd_2->GetOutDataNodes().size(), 2);
  EXPECT_EQ(cast_fp16_2_int8->GetOutControlNodes().size(), 1);
}

/*
如下图，Node4D作为normal node，输出anchor对端为1个cast算子，
获取子图的逻辑不能够正常工作，此类场景暂不支持
              |---cast---transdata---A
              |
Node4D--cast--|---cast---cast---transdata---B
                                      |
                                      |---C
 */
TEST_F(UtestGraphPassesSameTransdataBreadthFusionPass, test_one_more_transop_between_normal_op_and_subgraph_not_change) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // Node4D
  ge::NodePtr node_4d = NodeBuilder("node_4d", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .Build(graph);
  // cast fp32 -> fp32
  ge::NodePtr cast_fp32_2_fp32_1 = NodeBuilder("cast_4d_fp32_2_fp32_1", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .Build(graph);

  // cast fp32 -> fp16
  ge::NodePtr cast_fp32_2_fp16_1 = NodeBuilder("cast_4d_fp32_2_fp16_1", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16_2 = NodeBuilder("cast_4d_fp32_2_fp16_2", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast_fp16_2_int8 = NodeBuilder("cast_fp16_2_int8", CAST)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT8)
      .Build(graph);

  ge::NodePtr node_4d_2_5hd_1 = NodeBuilder("4d_2_5hd_1", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr node_4d_2_5hd_2 = NodeBuilder("4d_2_5hd_2", TRANSDATA)
      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT8)
      .Build(graph);
      
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), cast_fp32_2_fp32_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp32_1->GetOutDataAnchor(0), cast_fp32_2_fp16_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp32_1->GetOutDataAnchor(0), cast_fp32_2_fp16_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_1->GetOutDataAnchor(0), node_4d_2_5hd_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_1->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_2->GetOutDataAnchor(0), cast_fp16_2_int8->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_int8->GetOutDataAnchor(0), node_4d_2_5hd_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_2_5hd_2->GetOutDataAnchor(0), node_relu_2->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_4d_2_5hd_2->GetOutDataAnchor(0), node_relu_3->GetInControlAnchor());
  EXPECT_EQ(graph->GetDirectNodesSize(), 10);

  ge::SameTransdataBreadthFusionPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 10);
}
