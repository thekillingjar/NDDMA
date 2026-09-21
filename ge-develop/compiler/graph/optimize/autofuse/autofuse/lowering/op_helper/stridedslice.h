/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_STRIDEDSLICE_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_STRIDEDSLICE_H_

#include "graph/graph.h"
#include "graph/node.h"
#include "graph/symbolizer/symbolic.h"

namespace ge {
struct StridedSliceMaskAttr {
  int64_t begin_mask{0};
  int64_t end_mask{0};
  int64_t ellipsis_mask{0};
  int64_t new_axis_mask{0};
  int64_t shrink_axis_mask{0};
};

struct StrdedSliceIndexInputs {
  std::vector<Expression> start_indexes;
  std::vector<Expression> strides_indexes;
  std::vector<bool> is_new_axis;
};

graphStatus InferShapeStridedSlice(const std::vector<Expression> &input_x_dims, const StridedSliceMaskAttr &mask_attr,
                                   StrdedSliceIndexInputs &index_input);
}

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_STRIDEDSLICE_H_