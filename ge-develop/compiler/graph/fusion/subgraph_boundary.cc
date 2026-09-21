/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <utility>

#include "ge/fusion/subgraph_boundary.h"
#include "ge_common/ge_api_error_codes.h"
#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/anchor.h"
#include "graph/utils/node_adapter.h"

namespace ge {
namespace fusion {
class SubgraphInputImpl {
 public:
  SubgraphInputImpl() = default;
  explicit SubgraphInputImpl(std::vector<NodeIo> &&inputs) : inputs_(std::move(inputs)) {}
  Status AddInput(const NodeIo &node_input) {
    const auto in_node = NodeAdapter::GNode2Node(node_input.node);
    const auto in_data_anchor = in_node->GetInDataAnchor(node_input.index);
    GE_ASSERT_NOTNULL(in_data_anchor, "Node [%s][%s] input[%d] is not exist", in_node->GetNamePtr(),
                      in_node->GetTypePtr(), node_input.index);
    if (tensor_producer_ == nullptr) {
      tensor_producer_ = in_data_anchor->GetPeerOutAnchor();
      GE_ASSERT_NOTNULL(tensor_producer_, "The tensor producer of node [%s][%s] input[%d] is not exist",
                        in_node->GetNamePtr(), in_node->GetTypePtr(), node_input.index);
    } else {
      // check node_input has same tensor producer
      GE_ASSERT_TRUE(in_data_anchor->GetPeerOutAnchor() == tensor_producer_,
                     "The tensor producer of node [%s][%s] input[%d] is not same one with others",
                     in_node->GetNamePtr(), in_node->GetTypePtr(), node_input.index);
    }
    inputs_.emplace_back(node_input);
    return SUCCESS;
  }
  std::vector<NodeIo> GetAllInputs() const {
    return inputs_;
  }

 private:
  OutDataAnchorPtr tensor_producer_{nullptr};
  std::vector<NodeIo> inputs_;
};

class SubgraphOutputImpl {
 public:
  SubgraphOutputImpl() = default;
  explicit SubgraphOutputImpl(NodeIo node_output) : output_(std::move(node_output)) {}
  Status SetOutput(const NodeIo &node_output) {
    GE_ASSERT_TRUE(NodeAdapter::GNode2Node(output_.node) == nullptr, "SubgraphOutput has already set");
    auto node = NodeAdapter::GNode2Node(node_output.node);
    GE_ASSERT_NOTNULL(node);
    GE_ASSERT_NOTNULL(node->GetOutDataAnchor(node_output.index), "Node [%s][%s] output [%ld] is not exsit",
                      node->GetNamePtr(), node->GetTypePtr(), node_output.index);
    output_ = node_output;
    return SUCCESS;
  }
  NodeIo GetOutput() const {
    return output_;
  }

 private:
  NodeIo output_;
};

class SubgraphBoundaryImpl {
 public:
  Status AddInput(int64_t index, SubgraphInput &&input) {
    const auto iter = idx_2_subgraph_input_.find(index);
    if (iter != idx_2_subgraph_input_.cend()) {
      GELOGE(FAILED, "Input %ld has already added.", index);
      return FAILED;
    }
    idx_2_subgraph_input_[index] = std::move(input);
    return SUCCESS;
  }

  Status AddOutput(int64_t index, SubgraphOutput &&output) {
    const auto iter = idx_2_subgraph_output_.find(index);
    if (iter != idx_2_subgraph_output_.cend()) {
      GELOGE(FAILED, "Output %ld has already added.", index);
      return FAILED;
    }
    idx_2_subgraph_output_[index] = std::move(output);
    return SUCCESS;
  }

  Status GetInput(int64_t index, SubgraphInput &subgraph_input) const {
    const auto iter = idx_2_subgraph_input_.find(index);
    if (iter == idx_2_subgraph_input_.cend()) {
      return FAILED;
    }
    subgraph_input = iter->second;
    return SUCCESS;
  }

  Status GetAllInputs(std::vector<SubgraphInput> &subgraph_input) const {
    for (const auto &id_2_input : idx_2_subgraph_input_) {
      subgraph_input.emplace_back(id_2_input.second);
    }
    return SUCCESS;
  }
  Status GetOutput(int64_t index, SubgraphOutput &subgraph_output) const {
    const auto iter = idx_2_subgraph_output_.find(index);
    if (iter == idx_2_subgraph_output_.cend()) {
      GELOGE(FAILED, "Output of index %ld is not exist", index);
      return FAILED;
    }
    subgraph_output = iter->second;
    return SUCCESS;
  }

  Status GetAllOutputs(std::vector<SubgraphOutput> &subgraph_outputs) const {
    for (const auto &id_2_input : idx_2_subgraph_output_) {
      subgraph_outputs.emplace_back(id_2_input.second);
    }
    return SUCCESS;
  }
 private:
  std::map<int64_t, SubgraphInput> idx_2_subgraph_input_;
  std::map<int64_t, SubgraphOutput> idx_2_subgraph_output_;
};

SubgraphInput::SubgraphInput() : impl_(MakeUnique<SubgraphInputImpl>()) {}

SubgraphInput::SubgraphInput(std::vector<NodeIo> node_inputs)
    : impl_(MakeUnique<SubgraphInputImpl>(std::move(node_inputs))) {}

SubgraphInput::~SubgraphInput() = default;

Status SubgraphInput::AddInput(const NodeIo &node_input) {
  return (impl_ == nullptr) ? FAILED : impl_->AddInput(node_input);
}

std::vector<NodeIo> SubgraphInput::GetAllInputs() const {
  std::vector<NodeIo> empty;
  return (impl_ == nullptr) ? empty : impl_->GetAllInputs();
}

SubgraphInput &SubgraphInput::operator=(SubgraphInput &&other) noexcept {
  if (this != &other) {
    impl_ = std::move(other.impl_);
  }
  return *this;
}

SubgraphInput &SubgraphInput::operator=(const SubgraphInput &other) noexcept {
  if (this != &other) {
    impl_ = (other.impl_ != nullptr) ? MakeUnique<SubgraphInputImpl>(*other.impl_) : nullptr;
  }
  return *this;
}

SubgraphInput::SubgraphInput(const SubgraphInput &other) noexcept {
  if (other.impl_ != nullptr) {
    impl_ = MakeUnique<SubgraphInputImpl>(*other.impl_);
  }
}

SubgraphOutput::SubgraphOutput() : impl_(MakeUnique<SubgraphOutputImpl>()) {}
SubgraphOutput::SubgraphOutput(const NodeIo &node_output) : impl_(MakeUnique<SubgraphOutputImpl>(node_output)) {}

SubgraphOutput::~SubgraphOutput() = default;

Status SubgraphOutput::SetOutput(const NodeIo &node_output) {
  return (impl_ == nullptr) ? FAILED : impl_->SetOutput(node_output);
}

Status SubgraphOutput::GetOutput(NodeIo &node_output) const {
  if (impl_ == nullptr) {
    return FAILED;
  }
  node_output = impl_->GetOutput();
  return SUCCESS;
}

SubgraphOutput &SubgraphOutput::operator=(SubgraphOutput &&other) noexcept {
  if (this != &other) {
    impl_ = std::move(other.impl_);
  }
  return *this;
}
SubgraphOutput::SubgraphOutput(const SubgraphOutput &other) noexcept {
  if (other.impl_ != nullptr) {
    impl_ = MakeUnique<SubgraphOutputImpl>(*other.impl_);
  }
}

SubgraphOutput &SubgraphOutput::operator=(const SubgraphOutput &other) noexcept {
  if (this != &other) {
    impl_ = (other.impl_ != nullptr) ? MakeUnique<SubgraphOutputImpl>(*other.impl_) : nullptr;
  }
  return *this;
}

SubgraphBoundary::SubgraphBoundary() : impl_(MakeUnique<SubgraphBoundaryImpl>()) {}

SubgraphBoundary::SubgraphBoundary(std::vector<SubgraphInput> inputs, std::vector<SubgraphOutput> outputs)
    : impl_(MakeUnique<SubgraphBoundaryImpl>()) {
  if (impl_ == nullptr) {
    return;
  }

  for (int64_t i = 0; i < static_cast<int64_t>(inputs.size()); ++i) {
    impl_->AddInput(i, std::move(inputs[i]));
  }
  for (int64_t i = 0U; i < static_cast<int64_t>(outputs.size()); ++i) {
    impl_->AddOutput(i, std::move(outputs[i]));
  }
}

SubgraphBoundary::~SubgraphBoundary() = default;

Status SubgraphBoundary::AddInput(int64_t index, SubgraphInput input) {
  GE_ASSERT_NOTNULL(impl_, "SubgraphBoundary is invalid");
  return impl_->AddInput(index, std::move(input));
}

Status SubgraphBoundary::AddOutput(int64_t index, SubgraphOutput output) {
  GE_ASSERT_NOTNULL(impl_, "SubgraphBoundary is invalid");
  return impl_->AddOutput(index, std::move(output));
}

Status SubgraphBoundary::GetInput(int64_t index, SubgraphInput &subgraph_input) const {
  GE_ASSERT_NOTNULL(impl_, "SubgraphBoundary is invalid");
  return impl_->GetInput(index, subgraph_input);
}

Status SubgraphBoundary::GetAllInputs(std::vector<SubgraphInput> &subgraph_inputs) const {
  GE_ASSERT_NOTNULL(impl_, "SubgraphBoundary is invalid");
  return impl_->GetAllInputs(subgraph_inputs);
}

Status SubgraphBoundary::GetOutput(int64_t index, SubgraphOutput &subgraph_output) const {
  GE_ASSERT_NOTNULL(impl_, "SubgraphBoundary is invalid");
  return impl_->GetOutput(index, subgraph_output);
}

Status SubgraphBoundary::GetAllOutputs(std::vector<SubgraphOutput> &subgraph_outputs) const {
  GE_ASSERT_NOTNULL(impl_, "SubgraphBoundary is invalid");
  return impl_->GetAllOutputs(subgraph_outputs);
}

SubgraphBoundary::SubgraphBoundary(const SubgraphBoundary &other) noexcept {
  if (other.impl_ != nullptr) {
    impl_ = MakeUnique<SubgraphBoundaryImpl>(*other.impl_);
  }
}

SubgraphBoundary &SubgraphBoundary::operator=(SubgraphBoundary &&other) noexcept {
  if (this != &other) {
    impl_ = std::move(other.impl_);
  }
  return *this;
}

SubgraphBoundary &SubgraphBoundary::operator=(const SubgraphBoundary &other) noexcept {
  if (this != &other) {
    impl_ = (other.impl_ != nullptr) ? MakeUnique<SubgraphBoundaryImpl>(*other.impl_) : nullptr;
  }
  return *this;
}
} // namespace fusion
}  // namespace ge