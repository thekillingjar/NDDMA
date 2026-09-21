/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "llm_flow_service.h"
#include "llm_utils.h"
#include "common/llm_log.h"
#include "common/llm_checker.h"
#include "llm_test_helper.h"

// mock for df, just rewrite the impl, no need the flow_graph
namespace ge {
class DataFlowInfoImpl {
 public:
  DataFlowInfoImpl() = default;
  ~DataFlowInfoImpl() = default;

  uint64_t GetTransactionId() const {
    LLMLOGI("stub DataFlowInfoImpl::GetTransactionId");
    return transaction_id_;
  }

  void SetTransactionId(uint64_t transaction_id) {
    LLMLOGI("stub DataFlowInfoImpl::SetTransactionId");
    transaction_id_ = transaction_id;
  }

 private:
  uint64_t transaction_id_ = 0UL;
};

DataFlowInfo::DataFlowInfo() {
  LLMLOGI("stub DataFlowInfo::DataFlowInfo");
  impl_ = llm::MakeShared<DataFlowInfoImpl>();
}
DataFlowInfo::~DataFlowInfo() {
  LLMLOGI("stub DataFlowInfo::~DataFlowInfo");
}

uint64_t DataFlowInfo::GetTransactionId() const {
  LLMLOGI("stub DataFlowInfo::GetTransactionId");
  if (impl_ != nullptr) {
    return impl_->GetTransactionId();
  }
  return 0UL;
}

void DataFlowInfo::SetTransactionId(uint64_t transaction_id) {
  LLMLOGI("stub DataFlowInfo::SetTransactionId");
  if (impl_ != nullptr) {
    impl_->SetTransactionId(transaction_id);
  }
}

template<typename T, typename... Args>
static inline std::shared_ptr<T> ComGraphMakeShared(Args &&...args) {
  using T_nc = typename std::remove_const<T>::type;
  std::shared_ptr<T> ret = nullptr;
  try {
    ret = std::make_shared<T_nc>(std::forward<Args>(args)...);
  } catch (const std::bad_alloc &) {
    ret = nullptr;
    LLMLOGE(ge::FAILED, "Make shared failed", "");
  }
  return ret;
}

// mock for Operator
Operator::Operator(const char *name, const char *type) {
  LLMLOGI("stub Operator::Operator");
}

graphStatus Operator::UpdateInputDesc(const std::string &name, const ge::TensorDesc &tensor_desc) {
  LLMLOGI("stub Operator::UpdateInputDesc");
  return GRAPH_SUCCESS;
}

graphStatus Operator::UpdateInputDesc(const char_t *name, const ge::TensorDesc &tensor_desc) {
  LLMLOGI("stub Operator::UpdateInputDesc");
  return GRAPH_SUCCESS;
}

graphStatus Operator::UpdateOutputDesc(const std::string &name, const ge::TensorDesc &tensor_desc) {
  LLMLOGI("stub Operator::UpdateOutputDesc");
  return GRAPH_SUCCESS;
}

graphStatus Operator::UpdateOutputDesc(const char_t *name, const ge::TensorDesc &tensor_desc) {
  LLMLOGI("stub Operator::UpdateOutputDesc");
  return GRAPH_SUCCESS;
}

// mock for graph
class GraphImpl {
public:
explicit GraphImpl(const std::string &name) : name_(name) {
  LLMLOGI("stub GraphImpl::GraphImpl");
}
~GraphImpl() {
  LLMLOGI("stub GraphImpl::~GraphImpl");
};

 private:
  std::string name_;
};

Graph::Graph(const std::string &name) {
  LLMLOGI("stub Graph::Graph");
  impl_ = ComGraphMakeShared<GraphImpl>(name);
  if (impl_ == nullptr) {
    LLMLOGW("[Check][Impl] Make graph impl failed");
  }
}

Graph::Graph(const char_t *name) {
  LLMLOGI("stub Graph::Graph");
  if (name != nullptr) {
    std::string graph_name = name;
    impl_ = ComGraphMakeShared<GraphImpl>(graph_name);
    if (impl_ == nullptr) {
      LLMLOGW("[Check][Impl] Make graph impl failed");
    }
  } else {
    LLMLOGW("[Check][Param] Input graph name is nullptr.");
  }
}

namespace dflow {
// mock for FlowOperator
FlowOperator::FlowOperator(const char *name, const char *type) : ge::Operator(name, type) {
  LLMLOGI("stub FlowOperator::FlowOperator");
}

FlowOperator::~FlowOperator() = default;

// mock for FlowData
FlowData::FlowData(const char *name, int64_t index) : FlowOperator(name, "Data") {
  LLMLOGI("stub FlowData::FlowData");
};

FlowData::~FlowData() = default;

class ProcessPointImpl {
public:
  ProcessPointImpl(const char_t *pp_name, ProcessPointType pp_type)
      : pp_name_(pp_name), pp_type_(pp_type), json_file_path_() {}
  ~ProcessPointImpl() = default;

private:
  std::string pp_name_;
  const ProcessPointType pp_type_;
  std::string json_file_path_;
};

ProcessPoint::ProcessPoint(const char_t *pp_name, ProcessPointType pp_type) {
  LLMLOGI("ProcessPoint::ProcessPoint");
}
ProcessPoint::~ProcessPoint() {}

// mock for FunctionPp
FunctionPp::FunctionPp(const char_t *pp_name) : ProcessPoint(pp_name, ProcessPointType::FUNCTION) {
  LLMLOGI("stub FunctionPp::FunctionPp");
}

FunctionPp::~FunctionPp() = default;

FunctionPp &FunctionPp::SetCompileConfig(const char_t *json_file_path) {
  LLMLOGI("stub FunctionPp::SetCompileConfig");
  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const ge::AscendString &value) {
  LLMLOGI("stub FunctionPp::SetInitParam");
  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const std::vector<ge::AscendString> &value) {
  LLMLOGI("stub FunctionPp::SetInitParam");
  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const int64_t &value) {
  LLMLOGI("stub FunctionPp::SetInitParam");
  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const std::vector<int64_t> &value) {
  LLMLOGI("stub FunctionPp::SetInitParam");
  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const char_t *value) {
  LLMLOGI("stub FunctionPp::SetInitParam");
  return *this;
}

void FunctionPp::Serialize(ge::AscendString &str) const {
  LLMLOGI("stub FunctionPp::Serialize");
}

// mock for FlowNode
FlowNode::FlowNode(const char *name, uint32_t input_num, uint32_t output_num) : FlowOperator(name, "FlowNode")  {
  LLMLOGI("stub FlowNode::FlowNode");
}
FlowNode::~FlowNode() = default;

FlowNode &FlowNode::SetInput(uint32_t dst_index, const FlowOperator &src_op, uint32_t src_index) {
  LLMLOGI("stub FlowNode::SetInput");
  return *this;
}

FlowNode &FlowNode::AddPp(const ProcessPoint &pp) {
  LLMLOGI("stub FlowNode::AddPp");
  return *this;
}

// mock for FlowGraph
class FlowGraphImpl {
public:
  explicit FlowGraphImpl(const char *name) : name_(name), graph_(Graph(name)) {
    LLMLOGI("stub FlowGraphImpl::FlowGraphImpl");
  }
  ~FlowGraphImpl() = default;

  const ge::Graph &ToGeGraph() const {

    return graph_;
  }

  const char *GetName() const {
    LLMLOGI("stub FlowGraphImpl::GetName");
    return name_.c_str();
  }

private:
  const std::string name_;
  ge::Graph graph_;
};

FlowGraph::FlowGraph(const char *name) {
  LLMLOGI("stub FlowGraph::FlowGraph");
  if (name != nullptr) {
    impl_ = ComGraphMakeShared<FlowGraphImpl>(name);
    if (impl_ == nullptr) {
      LLMLOGW("FlowGraphImpl make shared failed.");
    }
  } else {
    impl_ = nullptr;
    LLMLOGW("Input graph name is nullptr.");
  }
}

FlowGraph::~FlowGraph() = default;

FlowGraph &FlowGraph::SetInputs(const std::vector<FlowOperator> &inputs) {
  LLMLOGI("stub FlowGraph::SetInputs");
  return *this;
}

FlowGraph &FlowGraph::SetOutputs(const std::vector<FlowOperator> &outputs) {
  LLMLOGI("stub FlowGraph::SetOutputs");
  return *this;
}

FlowGraph &FlowGraph::SetOutputs(const std::vector<std::pair<FlowOperator, std::vector<size_t>>> &output_indexes) {
  LLMLOGI("stub FlowGraph::SetOutputs");
  return *this;
}

const ge::Graph &FlowGraph::ToGeGraph() const {
  LLMLOGI("stub FlowGraph::ToGeGraph");
  if (impl_ == nullptr) {
    LLMLOGW("ToGeGraph failed: graph can not be used, impl is nullptr.");
    static ge::Graph graph;
    return graph;
  }
  return impl_->ToGeGraph();
}

const char *FlowGraph::GetName() const {
  LLMLOGI("stub FlowGraph::GetName");
  if (impl_ == nullptr) {
    LLMLOGW("GetName failed: graph can not be used, impl is nullptr.");
    return nullptr;
  }
  return impl_->GetName();
}

} // namespace dflow
} // namespace ge