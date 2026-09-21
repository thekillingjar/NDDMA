/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PREPROCESS_GRAPH_LINT_H_
#define GE_GRAPH_PREPROCESS_GRAPH_LINT_H_
#include "graph/compute_graph.h"
#include "graph/utils/connection_matrix.h"
#include "common/checker.h"
namespace ge {
class GraphLint {
 public:
  GraphLint() = default;
  graphStatus Verify(const ComputeGraphPtr &root_graph);
  GraphLint(const GraphLint &) = delete;
  GraphLint &operator=(const GraphLint &other) = delete;

  enum class RWType : uint32_t { kReadOnly = 0u, kWritable, kCanIgnore, kInvalid };

  struct NodeInputRWDesc {
   public:
    NodeInputRWDesc() {
      input_rw_type = {};
      is_marked = false;
      is_init = false;
    };
    void Init(size_t input_size, RWType init_value = RWType::kInvalid) {
      input_rw_type.resize(input_size);
      for (size_t i = 0u; i < input_size; ++i) {
        input_rw_type[i] = init_value;
      }
      is_init = true;
    }

    bool IsInit() const {
      return is_init;
    }

    graphStatus SetInputRwType(uint64_t input_index, RWType rw_type) {
      GE_ASSERT_TRUE(input_index < input_rw_type.size(), "Input index %ld should not large than inputs size %zu",
                     input_index, input_rw_type.size());
      if (input_rw_type[input_index] == RWType::kWritable) {
        return GRAPH_SUCCESS;
      }
      input_rw_type[input_index] = rw_type;
      return GRAPH_SUCCESS;
    }

    graphStatus GetInputRwType(uint64_t input_index, RWType &rw_type) const {
      GE_ASSERT_TRUE(input_index < input_rw_type.size());
      rw_type = input_rw_type[input_index];
      return GRAPH_SUCCESS;
    }

    graphStatus SetIsMarked() {
      for (size_t i = 0U; i < input_rw_type.size(); ++i) {
        if (input_rw_type[i] == RWType::kInvalid) {
          GELOGW("Input %zu is invalid.", i);
          return GRAPH_FAILED;
        }
      }
      is_marked = true;
      return GRAPH_SUCCESS;
    }
    bool IsMarked() const {
      return is_marked;
    }

   private:
    bool is_init = false;
    bool is_marked = false;
    std::vector<RWType> input_rw_type{};
  };

 private:
  graphStatus Initialize(const ComputeGraphPtr &root_graph);
  graphStatus VerifyRwConflictPerOutAnchor(const OutDataAnchorPtr &out_anchor) const;
  graphStatus VerifyRwConflictPerGraph(const ComputeGraphPtr &graph) const;

  std::vector<NodeInputRWDesc> nodes_2_rw_descs_{};
  std::unordered_map<const ComputeGraph *, std::unique_ptr<ConnectionMatrix>> graph_2_connection_matrixes_{};
};
}  // namespace ge

#endif  // GE_GRAPH_PREPROCESS_GRAPH_LINT_H_
