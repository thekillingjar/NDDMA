/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_stage_cache.h"
#include "graph/ge_tensor.h"
#include "graph/utils/tensor_utils.h"
#include "common/debug/ge_log.h"
#include "hybrid/model/node_item.h"
#include "hybrid/model/infer/tensor_desc_holder.h"
#include "hybrid/model/infer/shape_utils.h"
#include "hybrid/model/infer/node_shape_infer.h"
#include "common/plugin/ge_make_unique_util.h"

namespace ge {
namespace hybrid {

struct CacheTensorDescObserver : ChangeObserver, private GeTensorDesc {
  explicit CacheTensorDescObserver(const TensorDescHolder &dest)
      : ChangeObserver(), GeTensorDesc(), dest_(dest), is_changed_(false) {}
  Status DoPropagate() {
    if (is_changed_.load()) {
      GE_CHK_STATUS_RET_NOLOG(ShapeUtils::CopyShapeAndTensorSize(*this, dest_.Input()));
      is_changed_.store(false);
    }
    return SUCCESS;
  }

  GeTensorDesc &GetCacheTensorDesc() {
    return *this;
  }

private:
  virtual void Changed() override {
    is_changed_.store(true);
  };

private:
  TensorDescHolder dest_;
  std::atomic<bool> is_changed_;
};

namespace {
bool DoNotNeedShapePropagate(const NodeItem &next_node, const int32_t input_index) {
  if (next_node.IsInputShapeStatic(input_index)) {
    return true;
  }
  const auto &input_desc = next_node.MutableInputDesc(input_index);
  return input_desc == nullptr;
}
} // namespace

Status GraphStageCache::CreatePropagator(NodeItem &cur_node,
                                         const int32_t output_index,
                                         NodeItem &next_node,
                                         const int32_t input_index) {
  if (DoNotNeedShapePropagate(next_node, input_index)) {
    return SUCCESS;
  }

  const TensorDescHolder src_tensor_desc(cur_node, output_index);
  const TensorDescHolder dest_tensor_desc(next_node, input_index);
  cur_node.CreatePropagator(src_tensor_desc, TensorDescObserver(dest_tensor_desc));
  GELOGD("Create stage propagator for [%s] stage id[%d] ", next_node.NodeName().c_str(), next_node.group);
  return SUCCESS;
}

GraphStageCache::StageCache::~StageCache() {}

GraphStageCache::StageCache &GraphStageCache::GetOrCreate(const int32_t stage) {
  return stage_caches_[stage];
}

TensorDescObserver GraphStageCache::StageCache::CreateTensorDescObserver(
    const std::shared_ptr<CacheTensorDescObserver> &observer) {
  tensor_desc_observers_.push_back(observer);
  return TensorDescObserver(observer->GetCacheTensorDesc(), *observer);
}

Status GraphStageCache::StageCache::DoPropagate() {
  const std::lock_guard<std::mutex> lk(sync_);
  for (const auto &server : tensor_desc_observers_) {
    GE_CHK_STATUS_RET_NOLOG(server->DoPropagate());
  }
  return SUCCESS;
}

Status GraphStageCache::DoPropagate(const int32_t stage) {
  if (stage < 0) {
    return SUCCESS;
  }
  return GetOrCreate(stage).DoPropagate();
}
} // namespace hybrid
} // namespace ge
