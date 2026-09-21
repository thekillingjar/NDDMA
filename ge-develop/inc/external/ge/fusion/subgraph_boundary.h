/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_GE_FUSION_SUBGRAPH_BOUNDARY_H_
#define INC_EXTERNAL_GE_FUSION_SUBGRAPH_BOUNDARY_H_

#include <memory>
#include "graph/graph.h"
#include "ge_common/ge_api_types.h"

namespace ge {
namespace fusion {
/**
 * 标记节点的输入或输出索引
 */
struct NodeIo {
  GNode node;
  int64_t index;
};

class SubgraphInputImpl;
/**
 * 用于描述subgraph的一个输入tensor
 * （1）考虑一个tensor可能有多个消费者，因此一个subgraph input可能对应多个node input
 *      tensor1
 *       /  \
 *     op1   op2
 *     如上示例，tensor1是一个subgraph的input, 其在边界上对应op1和op2两个node。tensor1在subgraph边界外，op1和op2在subgraph边界内
 *
 * （2）再考虑tensor1有2个消费者，其中op1在subgraph边界内，op2在边界外。
 *     因此当描述subgraph input的时候，明确边界上的node input是有必要的
 */
class SubgraphInput {
 public:
  SubgraphInput();
  explicit SubgraphInput(std::vector<NodeIo> node_inputs);
  ~SubgraphInput();
  SubgraphInput(const SubgraphInput &other) noexcept;
  SubgraphInput &operator=(SubgraphInput &&other) noexcept;
  SubgraphInput &operator=(const SubgraphInput &other) noexcept;

  Status AddInput(const NodeIo &node_input);
  [[nodiscard]] std::vector<NodeIo> GetAllInputs() const;
 private:
  std::unique_ptr<SubgraphInputImpl> impl_;
};

class SubgraphOutputImpl;
/**
 * 用于描述subgraph的一个输出tensor
 */
class SubgraphOutput {
  public:
  SubgraphOutput();
  explicit SubgraphOutput(const NodeIo &node_output);
  ~SubgraphOutput();
  SubgraphOutput(const SubgraphOutput &other) noexcept;
  SubgraphOutput &operator=(SubgraphOutput &&other) noexcept;
  SubgraphOutput &operator=(const SubgraphOutput &other) noexcept;

  Status SetOutput(const NodeIo &node_output);
  Status GetOutput(NodeIo &node_output) const;
 private:
  std::unique_ptr<SubgraphOutputImpl> impl_;
};

/**
 * 用于圈定subgraph边界
 * (1) subgraph 需要自包含
 *     除了边界的输出算子，边界内所有算子的数据输出消费者都要在边界内。若有边界外的消费者，会导致消费者在重写后找不到原始tensor
 * (2) subgraph 中的节点不能跨图
 * (3) subgraph 不能包含所在图的边界算子（data/netoutput/variable等）
 */
class SubgraphBoundaryImpl;
class SubgraphBoundary {
 public:
  SubgraphBoundary();
  SubgraphBoundary(std::vector<SubgraphInput> inputs, std::vector<SubgraphOutput> outputs);
  ~SubgraphBoundary();
  SubgraphBoundary(const SubgraphBoundary &other) noexcept;
  SubgraphBoundary &operator=(SubgraphBoundary &&other) noexcept;
  SubgraphBoundary &operator=(const SubgraphBoundary &other) noexcept;

  Status AddInput(int64_t index, SubgraphInput input);
  Status AddOutput(int64_t index, SubgraphOutput output);

  Status GetInput(int64_t index, SubgraphInput &subgraph_input) const;
  Status GetAllInputs(std::vector<SubgraphInput> &subgraph_input) const;
  Status GetOutput(int64_t index, SubgraphOutput &subgraph_output) const;
  Status GetAllOutputs(std::vector<SubgraphOutput> &subgraph_outputs) const;
 private:
  std::unique_ptr<SubgraphBoundaryImpl> impl_;
};
}  // namespace fusion
} // namespace ge

#endif  // INC_EXTERNAL_GE_FUSION_SUBGRAPH_BOUNDARY_H_
