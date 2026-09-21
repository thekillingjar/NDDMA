/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <unordered_set>
#include "es_c_graph_builder.h"

#include <utility>
#include "utils/extern_math_util.h"
#include "es_c_tensor_holder.h"
#include "graph/operator_factory.h"
#include "graph/operator_reg.h"
#include "common/checker.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "compliant_node_builder.h"

namespace ge {
namespace {
const std::unordered_set<std::string> kDataOpSet = {"Data", "RefData", "AippData", "AnyData"};

template <typename Iterator>
std::string Join(Iterator begin, Iterator end, const std::string &sep = ",") {
  if (begin == end) {
    return "";
  }
  std::stringstream ss;
  ss << *begin;
  for (auto iter = std::next(begin); iter != end; ++iter) {
    ss << sep << *iter;
  }
  return ss.str();
}

GNode CreateCompliantData(const char_t *name, const char_t *type, int64_t index, Graph *const graph,
                          C_DataType data_type, C_Format format, const int64_t *dims, int64_t dim_num) {
  std::vector<int64_t> dims_vec;
  if (dims != nullptr) {
    dims_vec.assign(dims, dims + dim_num);
  }
  auto av = ge::AttrValue();
  (void)av.SetAttrValue(index);
  auto node = ge::es::CompliantNodeBuilder(graph)
                  .OpType(type)
                  .Name(name)
                  .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, "x"}})
                  .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, "y"}})
                  .IrDefAttrs({{"index", ge::es::CompliantNodeBuilder::kEsAttrOptional, "Int", av}})
                  .InstanceOutputDataType("y", static_cast<ge::DataType>(data_type))
                  .InstanceOutputFormat("y", static_cast<ge::Format>(format))
                  .InstanceOutputShape("y", dims_vec)
                  .Build();
  return node;
}
GNode CreateCompliantNetOutput(int32_t output_num, Graph *const graph) {
  AscendString graph_name;
  graph->GetName(graph_name);
  std::string net_output_name = "NetOutput_" + std::string(graph_name.GetString());

  return ge::es::CompliantNodeBuilder(graph)
      .OpType("NetOutput")
      .Name(net_output_name.c_str())
      .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputDynamic, ""}})
      .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputDynamic, ""}})
      .InstanceDynamicInputNum("x", output_num)
      .InstanceDynamicOutputNum("y", output_num)
      .Build();
}

graphStatus EsMoveSubgraphToRoot(Graph &root_graph) {
  AscendString root_graph_name;
  GE_ASSERT_GRAPH_SUCCESS(root_graph.GetName(root_graph_name));
  for (const auto &graph : root_graph.GetAllSubgraphs()) {
    GE_ASSERT_NOTNULL(graph);
    AscendString graph_name;
    GE_ASSERT_GRAPH_SUCCESS(graph->GetName(graph_name));
    for (const auto &subgraph : graph->GetAllSubgraphs()) {
      GE_ASSERT_NOTNULL(subgraph);
      AscendString subgraph_name;
      GE_ASSERT_GRAPH_SUCCESS(subgraph->GetName(subgraph_name));
      const char *subgraph_name_cstr = subgraph_name.GetString();
      if (root_graph.GetSubGraph(subgraph_name_cstr) != nullptr) {
        REPORT_INNER_ERR_MSG("E18888", "Es move subgraph %s to root graph %s failed: subgraph existed",
          subgraph_name_cstr, root_graph_name.GetString());
        GELOGE(GRAPH_FAILED, "[EsRemove][SubgraphToRoot] The subgraph %s existed in root graph %s ",
          subgraph_name_cstr, root_graph_name.GetString());
        return GRAPH_FAILED;
      }
      GE_ASSERT_GRAPH_SUCCESS(graph->RemoveSubgraph(subgraph_name_cstr));
      GE_ASSERT_GRAPH_SUCCESS(root_graph.AddSubGraph(*subgraph));
      GELOGI("Move subgraph %s from graph %s to root graph %s",
             subgraph_name_cstr, graph_name.GetString(), root_graph_name.GetString());
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus EsUpdateDataNodeInputDesc(const Graph &graph) {
  AscendString graph_name;
  graph.GetName(graph_name);
  auto all_nodes = graph.GetDirectNode();
  for (auto &node : all_nodes) {
    AscendString node_type;
    GE_ASSERT_GRAPH_SUCCESS(node.GetType(node_type));
    if (kDataOpSet.find(node_type.GetString()) != kDataOpSet.end()) {
      TensorDesc tensor_desc;
      GE_ASSERT_GRAPH_SUCCESS(node.GetOutputDesc(0, tensor_desc));
      GE_ASSERT_GRAPH_SUCCESS(node.UpdateInputDesc(0, tensor_desc));
      AscendString node_name;
      GE_ASSERT_GRAPH_SUCCESS(node.GetName(node_name));
      GELOGI("Update input desc from output desc for %s type node %s in graph %s",
             node_type.GetString(), node_name.GetString(), graph_name.GetString());
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus NormalizeEsToGe(Graph &graph) {
  GE_ASSERT_GRAPH_SUCCESS(EsUpdateDataNodeInputDesc(graph));
  GE_ASSERT_GRAPH_SUCCESS(EsMoveSubgraphToRoot(graph));
  return GRAPH_SUCCESS;
}
}  // namespace
}  // namespace ge

struct EsCGraphBuilder::EsCGraphBuilderImpl {
 public:
  EsCGraphBuilderImpl(const char *name) : graph_(std::make_unique<ge::Graph>(name)), nodes_num_(0) {
    if (graph_ != nullptr) {
      graph_->SetValid();
    }
  }
  ge::Status SetGraphOutput(EsCTensorHolder *tensor, int64_t output_index) {
    GE_ASSERT_NOTNULL(tensor);
    GE_ASSERT_TRUE(output_index >= 0, "Invalid output index %d, must be non-negative", output_index);
    GE_ASSERT_TRUE(output_indexes_to_tensor_.count(output_index) == 0, "Duplicated output index %d", output_index);
    output_indexes_to_tensor_[output_index] = tensor;
    return ge::SUCCESS;
  }
  bool IsGraphValid() const {
    if (!graph_input_indexes_.empty()) {
      GE_ASSERT_TRUE(*graph_input_indexes_.begin() == 0, "Invalid graph, graph input index must starts with 0");
      if (static_cast<size_t>(*graph_input_indexes_.rbegin()) + 1U != graph_input_indexes_.size()) {
        std::stringstream ss;
        ss << "Invalid graph, graph input indexes are not continuous: ";
        bool first = true;
        for (auto index : graph_input_indexes_) {
          if (!first) {
            ss << ", ";
          }
          ss << index;
          first = false;
        }
        GELOGE(ge::GRAPH_FAILED, "%s", ss.str().c_str());
        return false;
      }
    }

    if (!output_indexes_to_tensor_.empty()) {
      GE_ASSERT_TRUE(output_indexes_to_tensor_.begin()->first == 0, "Invalid graph, output index must starts with 0");
      if (static_cast<size_t>(output_indexes_to_tensor_.rbegin()->first) + 1U != output_indexes_to_tensor_.size()) {
        std::stringstream ss;
        ss << "Invalid graph, output indexes are not continuous: ";
        bool first = true;
        for (auto index : output_indexes_to_tensor_) {
          if (!first) {
            ss << ", ";
          }
          ss << index.first;
          first = false;
        }
        GELOGE(ge::GRAPH_FAILED, "%s", ss.str().c_str());
        return false;
      }
    }

    return true;
  }

  std::unique_ptr<ge::Graph> BuildGraphAndReset() {
    GE_ASSERT_NOTNULL(graph_, "Cannot build graph again");
    GE_ASSERT_TRUE(IsGraphValid());

    auto node = ge::CreateCompliantNetOutput(static_cast<int32_t>(output_indexes_to_tensor_.size()), graph_.get());
    std::vector<std::pair<ge::GNode, int32_t>> output_node_and_index;
    for (const auto &output : output_indexes_to_tensor_) {
      auto tensor = output.second;
      GE_ASSERT_NOTNULL(tensor);
      auto src_node = tensor->GetProducer();
      auto src_node_output_idx = tensor->GetOutIndex();
      GE_ASSERT_GRAPH_SUCCESS(graph_->AddDataEdge(src_node, src_node_output_idx, node, output.first));
      output_node_and_index.emplace_back(src_node, src_node_output_idx);
    }
    GE_ASSERT_GRAPH_SUCCESS(graph_->SetOutputs(output_node_and_index));
    GE_ASSERT_GRAPH_SUCCESS(ge::NormalizeEsToGe(*graph_));
    return std::move(graph_);
  }

  ge::AscendString GenerateNodeName(const ge::char_t *node_type) {
    return ge::AscendString((std::string(node_type) + "_" + std::to_string(NextNodeIndex())).c_str());
  }
  ge::Graph *GetGraph() {
    return graph_.get();
  }
  int64_t NextNodeIndex() {
    return nodes_num_++;
  }
  int32_t GetGraphInputSize() {
    return static_cast<int32_t>(graph_input_indexes_.size());
  }
  void *AddResource(void *resource_ptr, std::function<void(void *)> deleter) {
    auto unique_dynamic_outputs = ge::ComGraphMakeUnique<ResourceHolder>(resource_ptr, std::move(deleter));
    resource_holder_.emplace_back(std::move(unique_dynamic_outputs));
    return resource_holder_.back().get()->resource_ptr_;
  }

  ge::Status GetAddGraphInputNode(ge::GNode &node, int32_t index, const ge::char_t *name, const ge::char_t *type,
                                  C_DataType data_type, C_Format format, const int64_t *dims, int64_t dim_num) {
    GE_ASSERT_NOTNULL(graph_, "Graph has been build and reset, cannot add graph input");
    GE_ASSERT_TRUE(index >= 0, "Invalid input index %d, must be non-negative", index);
    GE_ASSERT_TRUE(graph_input_indexes_.insert(index).second, "Duplicated graph input index %d", index);

    if (type == nullptr) {
      type = "Data";
    } else {
      GE_ASSERT_TRUE(ge::kDataOpSet.find(type) != ge::kDataOpSet.end(), "Input type must be one of {%s}",
                     ge::Join(ge::kDataOpSet.begin(), ge::kDataOpSet.end()).c_str());
    }
    if (name == nullptr) {
      std::string cpp_name = "input_" + std::to_string(index);
      node = ge::CreateCompliantData(cpp_name.c_str(), type, index, graph_.get(), data_type, format, dims, dim_num);
    } else {
      node = ge::CreateCompliantData(name, type, index, graph_.get(), data_type, format, dims, dim_num);
    }
    return ge::SUCCESS;
  }

  /**
   * 资源管理struct
   * resource_ptr_ 资源指针
   * deleter_ 析构函数
   */
  struct ResourceHolder {
    void *resource_ptr_;
    std::function<void(void *)> deleter_;
    ResourceHolder(void *resource_ptr, std::function<void(void *)> deleter)
        : resource_ptr_(resource_ptr), deleter_(std::move(deleter)) {}
    ~ResourceHolder() {
      if (resource_ptr_ != nullptr) {
        deleter_(resource_ptr_);
      }
    }
  };
  std::unique_ptr<ge::Graph> graph_;
  std::list<std::unique_ptr<ResourceHolder>> resource_holder_;
  std::set<int32_t> graph_input_indexes_;
  std::map<int32_t, EsCTensorHolder *> output_indexes_to_tensor_;
  int64_t nodes_num_;  // Node name 自生成计数
};


EsCGraphBuilder::EsCGraphBuilder() {
  // new失败说明业务无法正常进行，主动抛出异常
  impl_ = std::make_unique<EsCGraphBuilderImpl>("graph");
}
EsCGraphBuilder::EsCGraphBuilder(const char *name) {
  // nnew失败说明业务无法正常进行，主动抛出异常
  impl_ = std::make_unique<EsCGraphBuilderImpl>(name);
}
EsCGraphBuilder::~EsCGraphBuilder() = default;
EsCGraphBuilder::EsCGraphBuilder(EsCGraphBuilder &&) noexcept = default;
EsCGraphBuilder &EsCGraphBuilder::operator=(EsCGraphBuilder &&) noexcept = default;

EsCTensorHolder *EsCGraphBuilder::GetTensorHolderFromNode(const ge::GNode &node, int32_t output_index) {
  ge::AscendString op_type;
  GE_ASSERT_GRAPH_SUCCESS(node.GetType(op_type));
  GE_ASSERT(ge::kDataOpSet.find(op_type.GetString()) == ge::kDataOpSet.end(),
            "Call `AppendGraphInput` or `CreateInput` to add graph input");
  return GetTensorHolderFromNodeInner(node, output_index);
}
EsCTensorHolder *EsCGraphBuilder::AddGraphInput(int32_t index, const ge::char_t *name, const ge::char_t *type,
                                                C_DataType data_type, C_Format format, const int64_t *dims,
                                                int64_t dim_num) {
  ge::GNode node;
  GE_ASSERT_SUCCESS(impl_->GetAddGraphInputNode(node, index, name, type, data_type, format, dims, dim_num));
  return GetTensorHolderFromNodeInner(node, 0);
}

ge::Status EsCGraphBuilder::SetGraphOutput(EsCTensorHolder *tensor, int64_t output_index) {
  return impl_->SetGraphOutput(tensor, output_index);
}

std::unique_ptr<ge::Graph> EsCGraphBuilder::BuildGraphAndReset() {
  return impl_->BuildGraphAndReset();
}

ge::AscendString EsCGraphBuilder::GenerateNodeName(const ge::char_t *node_type) {
  return impl_->GenerateNodeName(node_type);
}
EsCTensorHolder *EsCGraphBuilder::AppendGraphInput(const ge::char_t *name, const ge::char_t *type) {
  return AddGraphInput(impl_->GetGraphInputSize(), name, type);
}
ge::Graph *EsCGraphBuilder::GetGraph() {
  return impl_->GetGraph();
}

std::vector<EsCTensorHolder *> *EsCGraphBuilder::CreateDynamicTensorHolderFromNode(const ge::GNode &node,
                                                                                   int32_t start_idx,
                                                                                   int32_t output_num) {
  auto output_tensors = ge::ComGraphMakeUnique<std::vector<EsCTensorHolder *>>();
  GE_ASSERT_NOTNULL(output_tensors);
  output_tensors->reserve(output_num);
  for (int32_t output_idx = start_idx; output_idx < start_idx + output_num; ++output_idx) {
    output_tensors->push_back(GetTensorHolderFromNode(node, output_idx));
  }

  return AddResource(std::move(output_tensors));
}

void *EsCGraphBuilder::AddResource(void *resource_ptr, std::function<void(void *)> deleter) {
  return impl_->AddResource(resource_ptr, deleter);
}

EsCTensorHolder *EsCGraphBuilder::GetTensorHolderFromNodeInner(const ge::GNode &node, int32_t output_index) {
  auto es_c_tensor_holder = ge::ComGraphMakeUnique<EsCTensorHolder>(*this, node, output_index);
  GE_ASSERT_NOTNULL(es_c_tensor_holder);
  return AddResource(std::move(es_c_tensor_holder));
}