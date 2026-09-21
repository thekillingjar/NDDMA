/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H46CE8BDB_A361_403B_AB8B_D4CFAE40604E
#define H46CE8BDB_A361_403B_AB8B_D4CFAE40604E

#include <vector>
#include <set>
#include "kernel/memory/span/span_layer.h"
#include "kernel/memory/span/span_allocator.h"
#include "kernel/memory/device/device_allocator.h"

namespace gert {
class SpanLayerLut {
  using SpanLayerIdIterator = std::set<SpanLayerId>::iterator;
 public:
  explicit SpanLayerLut(const std::vector<SpanLayer *> &span_layers)
      : span_layers_{span_layers} {
  }

 public:
  virtual void OnLayerCreated(const SpanLayer &) = 0;
  virtual void OnLayerAddSpan(const SpanLayer &) = 0;
  virtual void OnLayerRemoveSpan(const SpanLayer &) = 0;
  virtual SpanLayerId FindFitLayerId(PageLen, size_t maxLiftLevel) const = 0;
  virtual void Recycle(SpanAllocator &, DeviceAllocator &) = 0;
  virtual void Release(SpanAllocator &, DeviceAllocator &) = 0;
  virtual ~SpanLayerLut() = default;

 public:
  SpanLayerIdIterator begin() {
    return span_layer_ids_.begin();
  }

  SpanLayerIdIterator end() {
    return span_layer_ids_.end();
  }

  SpanLayerIdIterator begin() const {
    return span_layer_ids_.begin();
  }

  SpanLayerIdIterator end() const {
    return span_layer_ids_.end();
  }

  size_t size() {
    return span_layer_ids_.size();
  }

 protected:
  std::set<SpanLayerId> span_layer_ids_;
  const std::vector<SpanLayer *> &span_layers_;
};

class SpanLayerSeqLut : public SpanLayerLut {
 public:
  using SpanLayerLut::SpanLayerLut;

 private:
  void OnLayerCreated(const SpanLayer &layer) override {
    span_layer_ids_.emplace(layer.GetLayerId());
  }
  void OnLayerAddSpan(const SpanLayer &) override {}
  void OnLayerRemoveSpan(const SpanLayer &) override {}

  SpanLayerId FindFitLayerId(PageLen page_len, size_t max_lift_Level) const override {
    if (max_lift_Level == 0U) {
      return SPAN_LAYER_ID_INVALID;
    }
    auto layer_idIter = span_layer_ids_.lower_bound(page_len);
    for (size_t lift_level = 0U; (lift_level < max_lift_Level) && (layer_idIter != span_layer_ids_.end());
         lift_level++) {
      if ((*layer_idIter < span_layers_.size()) && (span_layers_[*layer_idIter] != nullptr)
          && (!span_layers_[*layer_idIter]->IsEmpty())) {
        return *layer_idIter;
      }
      ++layer_idIter;
    }
    return SPAN_LAYER_ID_INVALID;
  }

  void Recycle(SpanAllocator &span_allocator, DeviceAllocator &device) override {
    for (auto layer_id : span_layer_ids_) {
      if ((layer_id < span_layers_.size()) && (span_layers_[layer_id] != nullptr)) {
        span_layers_[layer_id]->Recycle(span_allocator, device);
      }
    }
  }

  void Release(SpanAllocator &span_allocator, DeviceAllocator &device) override {
    for (auto layer_id : span_layer_ids_) {
      if ((layer_id < span_layers_.size()) && (span_layers_[layer_id] != nullptr)) {
        span_layers_[layer_id]->Release(span_allocator, device);
      }
    }
  }
};

class SpanLayerQuickLut : public SpanLayerLut {
 public:
  using SpanLayerLut::SpanLayerLut;

 private:
  void OnLayerCreated(const SpanLayer &) override {
  }
  void OnLayerAddSpan(const SpanLayer &layer) override {
    if (layer.GetSize() == 1) {
      span_layer_ids_.emplace(layer.GetLayerId());
    }
  }
  void OnLayerRemoveSpan(const SpanLayer &layer) override {
    if (layer.IsEmpty()) {
      span_layer_ids_.erase(layer.GetLayerId());
    }
  }
  SpanLayerId FindFitLayerId(PageLen page_len, size_t max_lift_level) const override {
    if (max_lift_level == 0U) {
      return SPAN_LAYER_ID_INVALID;
    }
    auto layer_idIter = span_layer_ids_.lower_bound(page_len);
    return (layer_idIter == span_layer_ids_.end()) ? SPAN_LAYER_ID_INVALID : *layer_idIter;
  }
  void Recycle(SpanAllocator &span_allocator, DeviceAllocator &device) override {
    for (auto layer_idIter = span_layer_ids_.begin(); layer_idIter != span_layer_ids_.end();) {
      if ((*layer_idIter >= span_layers_.size()) || (span_layers_[*layer_idIter] == nullptr)) {
        continue;
      }

      span_layers_[*layer_idIter]->Recycle(span_allocator, device);
      if (span_layers_[*layer_idIter]->IsEmpty()) {
        layer_idIter = span_layer_ids_.erase(layer_idIter);
      } else {
        ++layer_idIter;
      }
    }
  }
  void Release(SpanAllocator &span_allocator, DeviceAllocator &device) override {
    for (auto &layer : span_layers_) {
      if (layer != nullptr) {
        layer->Release(span_allocator, device);
      }
    }
    span_layer_ids_.clear();
  }
};
}

#endif
