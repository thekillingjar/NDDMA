/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_GRAPH_STAGE_CACHE_H
#define AIR_CXX_GRAPH_STAGE_CACHE_H

#include <map>
#include <mutex>
#include <vector>
#include "ge/ge_api_error_codes.h"
#include "node_shape_propagator.h"

namespace ge {
namespace hybrid {
struct CacheTensorDescObserver;
struct NodeItem;

struct GraphStageCache {
  GraphStageCache() = default;
  Status CreatePropagator(NodeItem &cur_node, const int32_t output_index, NodeItem &next_node,
                          const int32_t input_index);
  Status DoPropagate(const int32_t stage);

private:
  struct StageCache {
    StageCache() = default;
    ~StageCache();
    TensorDescObserver CreateTensorDescObserver(const std::shared_ptr<CacheTensorDescObserver> &observer);
    Status DoPropagate();

  private:
    std::vector<std::shared_ptr<CacheTensorDescObserver>> tensor_desc_observers_;
    std::mutex sync_;
  };

private:
  StageCache &GetOrCreate(const int32_t stage);

private:
  std::map<int32_t, StageCache> stage_caches_;
};
} // namespace hybrid
} // namespace ge
#endif
