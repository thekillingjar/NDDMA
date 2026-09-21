/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_CONSTANT_FUSE_SAME_PASS_H_
#define GE_GRAPH_PASSES_CONSTANT_FUSE_SAME_PASS_H_

#include <map>
#include <set>
#include <utility>
#include <vector>
#include "graph_metadef/graph/aligned_ptr.h"
#include "graph/types.h"
#include "graph/passes/graph_pass.h"

namespace ge {
struct SameConstKey {
  int32_t data_size;
  std::shared_ptr<AlignedPtr> aligned_ptr;
  DataType data_type;
  Format format;
  std::vector<int64_t> shape;

 public:
  bool operator< (const SameConstKey &key) const {
    if (data_size != key.data_size) {
      return data_size < key.data_size;
    }
    if (data_size != 0) {
      int32_t ret = memcmp(aligned_ptr->Get(), key.aligned_ptr->Get(), data_size);
      if (ret != 0) {
        return ret < 0;
      }
    }
    if (data_type != key.data_type) {
      return data_type < key.data_type;
    }
    if (format != key.format) {
      return format < key.format;
    }
    size_t shape_size = shape.size();
    if (shape_size != key.shape.size()) {
      return shape_size < key.shape.size();
    }
    for (size_t i = 0; i < shape_size; ++i) {
      if (shape.at(i) != key.shape.at(i)) {
        return shape.at(i) < key.shape.at(i);
      }
    }
    return false;
  }
};

class ConstantFuseSamePass : public GraphPass {
 public:
  Status Run(ge::ComputeGraphPtr graph) override;

 private:
  void GetFuseConstNodes(ComputeGraphPtr &graph,
      std::map<SameConstKey, std::vector<NodePtr>> &fuse_nodes) const;
  Status MoveOutDataEdges(NodePtr &src_node, NodePtr &dst_node) const;
  Status FuseConstNodes(ComputeGraphPtr &graph,
      std::map<SameConstKey, std::vector<NodePtr>> &fuse_nodes) const;
};
} // namespace ge
#endif // GE_GRAPH_PASSES_CONSTANT_FUSE_SAME_PASS_H_
