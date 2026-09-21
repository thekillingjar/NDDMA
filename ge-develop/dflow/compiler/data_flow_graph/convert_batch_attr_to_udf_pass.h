/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMPILER_PNE_DATA_FLOW_GRAPH_CONVERT_BATCH_ATTR_TO_UDF_PASS_H_
#define GE_COMPILER_PNE_DATA_FLOW_GRAPH_CONVERT_BATCH_ATTR_TO_UDF_PASS_H_

#include "graph/graph.h"
#include "graph/passes/graph_pass.h"

namespace ge {
struct CountBatchPara {
  int64_t batch_size;
  int64_t timeout;
  bool padding;
  int64_t slide_stride;
};

enum class BatchType {
  kTimeBatch = 0,
  kCountBatch = 1,
  kInvalidBatch,
};

class ConvertBatchAttrToUdfPass : public GraphPass {
 public:
  Status Run(ge::ComputeGraphPtr graph);

 private:
  bool HasPeerOutAnchor(const Node &node, const int32_t index) const;

  bool CanBatch(const Node &node) const;

  Status GetTimeBatchAttrs(const GeTensorDesc &tensor_desc, int64_t &window, int64_t &dim, bool &drop_remainder) const;

  Status GetCountBatchAttrs(const GeTensorDesc &tensor_desc, CountBatchPara &batch_para) const;

  Status CreateTimeBatchDesc(const Node &node, int32_t index, OpDescPtr &time_batch_desc) const;

  Status CreateCountBatchDesc(const Node &node, int32_t index, OpDescPtr &count_batch_desc) const;

  Status UpdateAnchor(const Node &node, int32_t node_index, const Node &batch, int32_t batch_index) const;

  Status InsertBatch(const Node &node, int32_t index, BatchType batch_type) const;

  Status CheckTimeBatchAttrs(int64_t window) const;
};
}  // namespace ge

#endif  // GE_COMPILER_PNE_DATA_FLOW_GRAPH_CONVERT_BATCH_ATTR_TO_UDF_PASS_H_
