/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/memory/allocator/scalable_allocator.h"
#include <limits>
#include <map>
#include "common/checker.h"

namespace gert {
constexpr size_t kSafeTryCount = 1000U;
std::atomic_size_t ScalableAllocator::global_allocator_id_(0U);

ScalableAllocator::ScalableAllocator(SpanAllocator &span_allocator, DeviceMemAllocator &device_allocator,
    const ScalableConfig &cfg,
    const std::string &graph_name)
    : MemoryPool(device_allocator),
      device_allocator_{device_allocator},
      allocator_id_(++global_allocator_id_),
      allocator_id_with_type_("[allocator_" + std::to_string(allocator_id_) + "]"),
      config_{cfg},
      page_mem_size_{PageLen_GetMemSize(1, cfg.page_idem_num)},
      span_layer_capacity_{1 + SpanLayerId_GetIdFromSize(cfg.page_mem_size_total_threshold, cfg.page_idem_num)},
      uncacheable_layer_start_{SpanLayerId_GetIdFromSize(cfg.uncacheable_size_threshold, cfg.page_idem_num)},
      span_layer_page_capacity_{PageLen_GetLenFromSize(cfg.page_mem_size_total_threshold, cfg.page_idem_num)},
      span_allocator_{span_allocator},
      layer_allocator_{cfg.span_layer_prepared_count},
      graph_name_(graph_name) {
  span_layers_.resize(span_layer_capacity_);
  new_span_layers_.resize(span_layer_capacity_);
  // 不加nothrow，理由：由于构造函数无法返回失败，且这是关键资源申请，如果申请失败允许进程退出。
  auto layer_lut = config_.enable_quick_layer_mode
                           ? static_cast<SpanLayerLut *>(new SpanLayerQuickLut{span_layers_})
                           : static_cast<SpanLayerLut *>(new SpanLayerSeqLut{span_layers_});
  if (layer_lut != nullptr) {
    span_layer_lut_.reset(layer_lut);
  }
}

ScalableAllocator::~ScalableAllocator() {
  try {
    Finalize(true);
    span_layers_.clear();
    new_span_layers_.clear();
    new_span_layer_ids_.clear();
  } catch (...) {
    // do nothing
  }
}

ge::Status ScalableAllocator::Finalize(bool no_log) {
  if (span_layer_lut_ == nullptr) {
    return ge::FAILED;
  }
  if (!occupied_spans_.empty()) {
    if (!no_log) {
      GELOGW("%s [ScalableAllocator]: memory leaks, occupied spans = %zu", GetId().c_str(), occupied_spans_.size());
      for (auto &span : occupied_spans_) {
        GELOGW("%s [ScalableAllocator]: memory leaks, occupied spans = %p", GetId().c_str(), span.GetAddr());
      }
    }
    return ge::FAILED;
  }
  if (!no_log) {
    PrintDetails(GeLogLevel::kEvent);
  }

  if (!graph_name_.empty()) {
    GEEVENT("model_metrics:name=%s, max_alloc_dev_mem=%lu B, device_id=%u",
                graph_name_.c_str(), max_occupied_size_, device_allocator_.GetDeviceId());
  }

  for (auto layer_id : new_span_layer_ids_) {
    auto layer = FetchNewVaSpanLayer(layer_id);
    if (layer != nullptr) {
      layer->Release(span_allocator_, device_allocator_);
      delete layer;
    }
  }
  new_span_layers_.clear();
  new_span_layer_ids_.clear();

  if (!span_layers_.empty()) {
    occupied_spans_.clear();
    span_layer_lut_->Release(span_allocator_, device_allocator_);
    for (auto layer : span_layers_) {
      if (layer != nullptr) {
        if (layer->GetLayerId() < span_layers_.size()) {
          span_layers_[layer->GetLayerId()] = nullptr;
        }
        delete layer;
      }
    }
    device_allocator_.GetExpandableAllocator().ReleaseVirtualMemory();
  }
  return ge::SUCCESS;
}

ge::MemBlock *ScalableAllocator::ConvertToRootBlock(ge::MemBlock *block) {
  auto span = reinterpret_cast<PageSpan *>(block);
  if (span->HasSplited()) {
    // todo 被切分过的block也可以转换成root block
    return nullptr;
  }
  auto root_block = span->GetBlockAddr();
  occupied_spans_.remove(*span);
  span_allocator_.Free(*span);
  return root_block;
}

MemSize ScalableAllocator::GetAllocSize(const MemSize size) const {
  return (size < page_mem_size_) ? page_mem_size_ : MemSize_GetAlignedOf(size, page_mem_size_);
}

PageLen ScalableAllocator::GetlayerSpanCapacity(const SpanLayerId layer_id) const {
  if ((layer_id >= uncacheable_layer_start_) || (layer_id == 0U)) {
    return 0U;
  }
  return std::min(span_layer_page_capacity_ / layer_id, static_cast<uint32_t>(config_.span_count_in_layer_threshold));
}

bool ScalableAllocator::IsLayerCacheable(const SpanLayerId layer_id) const {
  return layer_id < uncacheable_layer_start_;
}

bool ScalableAllocator::IsThresholdExceeded(const MemSize size) const {
  return device_allocator_.GetOccupiedSize() > (config_.page_mem_size_total_threshold - size);
}

PageSpan *ScalableAllocator::FetchLayerSpan(const SpanLayerId layer_id) {
  if (span_layers_.empty() || (span_layers_[layer_id] == nullptr)) {
    return nullptr;
  }

  auto span = PopSpanFromLayer(*span_layers_[layer_id]);
  if (span != nullptr) {
    OccupySpan(*span, layer_id);
  }
  return span;
}

PageSpan *ScalableAllocator::FetchNewVaLayerSpan(const SpanLayerId layer_id) {
  if (new_span_layers_.empty() || (layer_id >= span_layer_capacity_) || (new_span_layers_[layer_id] == nullptr)) {
    return nullptr;
  }

  auto span = new_span_layers_[layer_id]->PopSpan();
  if (span != nullptr) {
    OccupySpan(*span, layer_id);
  }
  return span;
}

SpanLayer *ScalableAllocator::FetchNewVaSpanLayer(const SpanLayerId layer_id) {
  if (new_span_layers_.empty() || (layer_id >= span_layer_capacity_)) {
    return nullptr;
  }
  if (new_span_layers_[layer_id] != nullptr) {
    return new_span_layers_[layer_id];
  }
  auto new_layer = new(layer_allocator_.Alloc()) SpanLayer(layer_id, GetlayerSpanCapacity(layer_id));
  if (new_layer == nullptr) {
    return nullptr;
  }
  new_span_layers_[layer_id] = new_layer;
  new_span_layer_ids_.insert(layer_id);
  return new_layer;
}

BlockAddr ScalableAllocator::DevAlloc(const MemSize size) {
  return device_allocator_.Alloc(size);
}

PageSpan *ScalableAllocator::BlockAlloc(ge::Allocator &allocator, const BlockAddr block_addr, const MemAddr addr,
                                        const size_t size) {
  auto span = new(span_allocator_.Alloc()) PageSpan{allocator, *this, block_addr, addr, size};
  return span;
}

PageSpan *ScalableAllocator::FetchNewVaSpan(ge::Allocator &allocator, const MemSize size,
                                            std::vector<size_t> &pa_list) {
  size_t alloc_size = pa_list.size() << ge::kLargePageSizeBits;
  SpanLayerId fix_layer_id = SpanLayerId_GetIdFromSize(alloc_size, config_.page_idem_num);
  PageSpan *span = nullptr;
  std::list<PageSpan *> spans;
  bool reuse = false;
  size_t try_count = 0U;
  while ((try_count < kSafeTryCount) && ((span = FetchNewVaLayerSpan(fix_layer_id)) != nullptr)) {
    try_count++;
    if (ge::SUCCESS == device_allocator_.GetExpandableAllocator().MallocPhysicalMemory(
        reinterpret_cast<MemAddr>(span->GetAddr()), span->GetPaList())) {
      reuse = true;
      break;
    }
    spans.push_back(span);
  }
  for (auto s : spans) {
    GELOGI("Free using block device_id:%u size:%llu alloc_size:%zu mem_addr:%p.",
                device_allocator_.GetDeviceId(), size, s->GetSize(), s->GetAddr());
    FreeSpanEx(s);
  }

  if (reuse) {
    GE_ASSERT_NOTNULL(span);
    GELOGI("reuse new_va_span try_count:%zu size:%zu span_size:%zu mem_addr_:%p",
                try_count, size, span->GetSize(), span->GetAddr());
    // 复用已有的span，释放新占用的内存
    device_allocator_.GetExpandableAllocator().FreePhysicalMemory(reinterpret_cast<MemAddr>(span->GetAddr()),
                                                                  pa_list, false, false);
    return span;
  }
  // 新的span需要新预留虚拟地址并做map
  void *addr = nullptr;
  device_allocator_.GetExpandableAllocator().GetPhysicalMemoryAllocator().ReserveMemAddress(&addr, alloc_size);
  if (addr == nullptr) {
    // 返回nullptr，上层触发回收释放空闲new_va
    return nullptr;
  }

  new_va_size_ += alloc_size;
  span = BlockAlloc(allocator, nullptr, reinterpret_cast<MemAddr>(addr), alloc_size);
  if (span == nullptr) {
    device_allocator_.GetExpandableAllocator().GetPhysicalMemoryAllocator().ReleaseMemAddress(addr, alloc_size);
    return nullptr;
  }

  span->GetPaList().insert(span->GetPaList().begin(), pa_list.begin(), pa_list.end());
  if (device_allocator_.GetExpandableAllocator().MallocPhysicalMemory(reinterpret_cast<MemAddr>(span->GetAddr()),
                                                                      span->GetPaList(), true) != ge::SUCCESS) {
    span_allocator_.Free(*span);
    device_allocator_.GetExpandableAllocator().GetPhysicalMemoryAllocator().ReleaseMemAddress(addr, alloc_size);
    return nullptr;
  }
  OccupySpan(*span, fix_layer_id);
  GELOGI("block device_id:%u size:%llu alloca_size:%zu mem_addr:%p",
              device_allocator_.GetDeviceId(), size, span->GetSize(), span->GetAddr());
  return span;
}

PageSpan *ScalableAllocator::FetchNewSpan(ge::Allocator &allocator, const MemSize size, const PageLen page_len) {
  if (IsThresholdExceeded(size)) {
    GELOGI("OccupiedSize:%llu add size:%llu exceed total_thresold:%llu.",
                device_allocator_.GetOccupiedSize(), size, config_.page_mem_size_total_threshold);

    // has freed memory, return nullptr and try recycle
    if (GetIdleMemSize() > 0U) {
      return nullptr;
    }
  }
  auto addr = DevAlloc(size);
  if (addr == nullptr) {
    return nullptr;
  }

  auto span = BlockAlloc(allocator, addr, reinterpret_cast<MemAddr>(addr->GetAddr()),
                         static_cast<size_t>(size));
  if (span == nullptr) {
    device_allocator_.Free(addr);
    return nullptr;
  }
  OccupySpan(*span, page_len);
  return span;
}

SpanLayer *ScalableAllocator::FetchSpanLayer(const SpanLayerId layer_id) {
  if (span_layers_.empty() || (layer_id >= span_layer_capacity_)) {
    return nullptr;
  }

  if (span_layers_[layer_id] != nullptr) {
    return span_layers_[layer_id];
  }

  auto newLayer = new(layer_allocator_.Alloc()) SpanLayer(layer_id, GetlayerSpanCapacity(layer_id));
  if (newLayer == nullptr) {
    return nullptr;
  }

  span_layers_[layer_id] = newLayer;
  span_layer_lut_->OnLayerCreated(*newLayer);
  return newLayer;
}

PageSpan *ScalableAllocator::FetchSplitedSpan(ge::Allocator &allocator, const SpanLayerId fix_layer_id,
                                              const SpanLayerId fit_layer_id, const MemSize size, int32_t &ret) {
  if (span_layers_.empty() || (fit_layer_id <= fix_layer_id)) {
    return nullptr;
  }
  if (span_layers_[fit_layer_id] == nullptr) {
    return nullptr;
  }

  auto span = PopSpanFromLayer(*span_layers_[fit_layer_id]);
  if (span == nullptr) {
    return nullptr;
  }

  if ((try_count_ > set_try_count_) && (span->GetAddr() != base_addr_)) {
    OccupySpan(*span, fit_layer_id);
    ret = ge::PHYSICAL_MEM_USING;
    try_count_ = 0U;
    return span;
  }
  if ((!device_allocator_.GetExpandableAllocator().IsValidVirtualAddr(reinterpret_cast<MemAddr>(span->GetAddr())))
      && (!span->IsSplitable(fix_layer_id))) {
    const auto layer = FetchSpanLayer(fit_layer_id);
    if (layer != nullptr) {
      GELOGI("Delay split block device_id:%u size:%zu allocate_size:%lu.",
                  device_allocator_.GetDeviceId(), span->GetSize(), size);
      PushSpanToLayer(*layer, *span);
    }
    return FetchNewSpan(allocator, size, fix_layer_id);
  }
  
  return SplitSpan(allocator, fix_layer_id, fit_layer_id, span, size);
}

PageSpan *ScalableAllocator::SplitSpan(ge::Allocator &allocator, const SpanLayerId fix_layer_id,
                                       const SpanLayerId fit_layer_id, PageSpan *const span, const MemSize size) {
  GE_ASSERT_NOTNULL(span);
  PageLen left_page_len = fit_layer_id - fix_layer_id;
  const auto buddy_addr = PageLen_ForwardAddr(left_page_len, config_.page_idem_num,
                                              reinterpret_cast<MemAddr>(span->GetAddr()));
  const auto buddy_span = BlockAlloc(allocator, nullptr, buddy_addr, static_cast<size_t>(size));
  if (buddy_span == nullptr) {
    return nullptr;
  }

  span->SetBuddy(*buddy_span);
  span->SetPageLen(left_page_len);

  const auto layer = FetchSpanLayer(left_page_len);
  if (layer == nullptr) {
    span_allocator_.Free(*span);
    return nullptr;
  }
  PushSpanToLayer(*layer, *span);
  OccupySpan(*buddy_span, fix_layer_id);
  return buddy_span;
}

PageSpan *ScalableAllocator::ProcessNewVaSpan(ge::Allocator &allocator, PageSpan *span, const MemSize size,
                                              size_t reuse_size, const MemSize alloc_size) {
  while (span->GetCount() > 0U) {
    span->SubCount();
  }
  occupied_spans_.remove(*span);
  span = TryMergeNext(*span);
  span = TryMergePrev(*span);
  PageSpan *free_span = nullptr;
  auto free_size = reuse_size & ge::kLargePageSizeMask;
  // |********physical page**********|********physical page**********|
  // |********************span**********************|
  // |********using_span*************|**free_span***|
  // 最后一块物理内存对齐部分大小，切成一小块free_span继续使用,using_span关联到new_var_span，最后一起释放
  if ((reuse_size > ge::kLargePageSize) && (free_size > 0U)) {
    SpanLayerId free_fix_layer_id = SpanLayerId_GetIdFromSize(free_size, config_.page_idem_num);
    SpanLayerId free_fit_layer_id = SpanLayerId_GetIdFromSize(span->GetSize(), config_.page_idem_num);
    free_span = SplitSpan(allocator, free_fix_layer_id, free_fit_layer_id, span, free_size);
    GE_ASSERT_NOTNULL(free_span);
    GELOGI("free span size:%zu realsize:%zu left_size:%zu next_size:%zu",
                free_span->GetSize(), free_size, span->GetSize(), reuse_size - free_size);
  } else {
    free_size = 0U;
  }

  SpanLayerId using_fix_layer_id = SpanLayerId_GetIdFromSize(reuse_size - free_size, config_.page_idem_num);
  SpanLayerId using_fit_layer_id = SpanLayerId_GetIdFromSize(span->GetSize(), config_.page_idem_num);

  if (free_span != nullptr) {
    span = PopSpanFromLayer(*span_layers_[using_fit_layer_id]);
  }
  auto using_span = SplitSpan(allocator, using_fix_layer_id, using_fit_layer_id, span, reuse_size - free_size);
  if (free_span != nullptr) {
    FreeSpanEx(free_span);
  }
  GE_ASSERT_NOTNULL(using_span);
  GELOGI("using span size:%zu realsize:%zu memaddr:%p index:%zu offset:%zu",
            using_span->GetSize(), reuse_size - free_size, using_span->GetAddr(),
            (reinterpret_cast<MemAddr>(using_span->GetAddr()) - reinterpret_cast<MemAddr>(base_addr_))
                / ge::kLargePageSize,
            (reinterpret_cast<MemAddr>(using_span->GetAddr()) - reinterpret_cast<MemAddr>(base_addr_))
                & ge::kLargePageSizeMask);
  auto new_va_span = FetchNewVaSpan(allocator, alloc_size, device_allocator_.GetExpandableAllocator().GetPaList());
  if (new_va_span == nullptr) {
    FreeSpanEx(using_span);
  } else {
    new_va_span->SetNewVaSpan(true);
    new_va_span->SetRealSize(size);
    new_va_span->SetRefSpan(using_span);
  }
  return new_va_span;
}
PageSpan *ScalableAllocator::ProcessPaUsingSpan(ge::Allocator &allocator, PageSpan *span, const MemSize size,
                                                size_t reuse_size, const MemSize alloc_size) {
  if (reuse_size > 0U) {
    while (span->GetCount() > 0U) {
      span->SubCount();
    }
    occupied_spans_.remove(*span);
    span = TryMergeNext(*span);
    span = TryMergePrev(*span);
    SpanLayerId using_fix_layer_id = SpanLayerId_GetIdFromSize(reuse_size, config_.page_idem_num);
    SpanLayerId using_fit_layer_id = SpanLayerId_GetIdFromSize(span->GetSize(), config_.page_idem_num);
    span = SplitSpan(allocator, using_fix_layer_id, using_fit_layer_id, span, reuse_size);
    GE_ASSERT_NOTNULL(span);
    span->SetRealSize(reuse_size);
    GELOGI("using span size:%zu realsize:%zu alloc_size:%zu", span->GetSize(), reuse_size,
                alloc_size);
    return span;
  }
  span->SetRealSize(size);
  return span;
}

PageSpan *ScalableAllocator::AllocImp(ge::Allocator &allocator, const MemSize size, int32_t &ret) {
  try_count_++;
  MemSize alloc_size = GetAllocSize(size);
  SpanLayerId fix_layer_id = SpanLayerId_GetIdFromSize(alloc_size, config_.page_idem_num);
  if ((fix_layer_id >= span_layer_capacity_) || (span_layer_lut_ == nullptr)) {
    return nullptr;
  }
  auto span = FetchLayerSpan(fix_layer_id);
  SpanLayerId fit_layer_id = 0U;
  if (span == nullptr) {
    fit_layer_id = span_layer_lut_->FindFitLayerId(fix_layer_id + 1, config_.span_layer_lift_max);
    if (fit_layer_id >= span_layer_capacity_) {
      span = FetchNewSpan(allocator, alloc_size, fix_layer_id);
    } else {
      span = FetchSplitedSpan(allocator, fix_layer_id, fit_layer_id, alloc_size, ret);
    }
  }

  if (span != nullptr) {
    GELOGI("try_count:%zu size:%zu block_size:%zu", try_count_, size, span->GetSize());
    if (ret == ge::PHYSICAL_MEM_USING) {
      span->SetRealSize(size);
      return span;
    }
    static const std::string purpose = "page caching";
    size_t reuse_size = 0U;
    ret = device_allocator_.GetExpandableAllocator().MallocPhysicalMemory(purpose,
        reinterpret_cast<MemAddr>(span->GetAddr()), alloc_size, reuse_size);
    if (ret != ge::SUCCESS) {
      if (ret == ge::NEW_VA) {
        ret = ge::SUCCESS;
        return ProcessNewVaSpan(allocator, span, size, reuse_size, alloc_size);
      } else if (ret == ge::PHYSICAL_MEM_USING) {
        return ProcessPaUsingSpan(allocator, span, size, reuse_size, alloc_size);
      } else {}
      FreeSpanEx(span);
      return nullptr;
    }
    span->SetRealSize(size);
  }
  return span;
}

PageSpan *ScalableAllocator::Alloc(ge::Allocator &allocator, const MemSize size) {
  if (size > config_.page_mem_size_total_threshold) {
    GELOGE(ge::FAILED, "%s [ScalableAllocator]: size:%lu > mem_size_total_thresold:%lu", GetId().c_str(), size,
           config_.page_mem_size_total_threshold);
    return nullptr;
  }

  try_count_ = 0U;
  size_t total_try_count = 0U;
  std::vector<PageSpan *> spans;
  PageSpan *span = nullptr;
  while ((total_try_count < kSafeTryCount)) {
    int32_t ret = ge::SUCCESS;
    span = AllocImp(allocator, size, ret);
    if (ret != ge::PHYSICAL_MEM_USING) {
      break;
    }
    if (span != nullptr) {
      spans.emplace_back(span);
    }
    total_try_count++;
  }
  for (auto s : spans) {
    GELOGI("Free using block device_id:%u size:%llu allocate_size:%zu mem_addr:%p.",
                device_allocator_.GetDeviceId(), size, s->GetSize(), s->GetAddr());
    FreeSpanEx(s);
  }
  if (span != nullptr) {
    if (!spans.empty()) {
      GELOGI("Free using block device_id:%u size:%llu allocate_size:%zu mem_addr:%p count:%zu.",
                  device_allocator_.GetDeviceId(), size, span->GetSize(), span->GetAddr(), spans.size());
    }

    theory_size_ += span->GetSize();
    real_theory_size_ += span->GetRealSize();
    device_allocator_.GetExpandableAllocator().SetTheorySize(theory_size_);
    if (real_theory_size_ > real_theory_min_size_) {
      real_theory_min_size_ = real_theory_size_;
    }

    if (theory_size_ > theory_min_size_) {
      theory_min_size_ = theory_size_;
      device_allocator_.GetExpandableAllocator().SetTheoryMinSize(theory_min_size_);
    }
    if ((device_allocator_.GetOccupiedSize() > max_occupied_size_) || span->IsNewVaSpan()) {
      max_occupied_size_ = device_allocator_.GetOccupiedSize();
      PrintDetails(GeLogLevel::kInfo);
    }
    GELOGI("Malloc block device_id:%u size:%llu allocate_size:%zu mem_addr:%p. span addr %p",
                device_allocator_.GetDeviceId(), size, span->GetSize(), span->GetAddr(), span);
  }
  return span;
}

PageSpan *ScalableAllocator::PickOutBuddy(PageSpan *const buddy_span) {
  if (span_layers_.empty() || (buddy_span == nullptr)) {
    return nullptr;
  }
  if (buddy_span->GetCount() != 0U) {
    return nullptr;
  }

  PageLen page_len = buddy_span->GetPageLen();
  if (page_len >= span_layer_capacity_) {
    return nullptr;
  }
  if (span_layers_[page_len] == nullptr) {
    return nullptr;
  }
  RemoveFromLayer(*span_layers_[page_len], *buddy_span);
  return buddy_span;
}

PageSpan *ScalableAllocator::TryMergePrev(PageSpan &span) {
  auto prev_buddy = PickOutBuddy(span.GetPrevBuddy());
  if (prev_buddy == nullptr) {
    return &span;
  }

  prev_buddy->MergeBuddy(span);
  span_allocator_.Free(span);
  return prev_buddy;
}

PageSpan *ScalableAllocator::TryMergeNext(PageSpan &span) {
  auto next_buddy = PickOutBuddy(span.GetNextBuddy());
  if (next_buddy == nullptr) {
    return &span;
  }

  span.MergeBuddy(*next_buddy);
  span_allocator_.Free(*next_buddy);
  return &span;
}

void ScalableAllocator::Free(ge::MemBlock *block) {
  auto span = reinterpret_cast<PageSpan *>(block);
  if ((span == nullptr) || (span_layer_lut_ == nullptr)) {
    return;
  }

  if (theory_size_ >= span->GetRealSize()) {
    real_theory_size_ -= span->GetRealSize();
    theory_size_ -= span->GetSize();
    device_allocator_.GetExpandableAllocator().SetTheorySize(theory_size_);
  }
  LOG_BY_TYPE(GeLogLevel::kInfo, "Free block device_id:%u theory_size_:%zu theory_min_size_:%zu allock_size:%zu mem_addr:%p.",
              device_allocator_.GetDeviceId(), theory_size_, theory_min_size_, span->GetSize(), span->GetAddr());

  if (span->IsNewVaSpan()) {
    (void) device_allocator_.GetExpandableAllocator().FreePhysicalMemory(reinterpret_cast<MemAddr>(span->GetAddr()),
                                                                         span->GetPaList(), true, false);
  } else {
    // 减物理内存引用计数，不真正释放
    (void) device_allocator_.GetExpandableAllocator().FreePhysicalMemory(reinterpret_cast<MemAddr>(span->GetAddr()),
                                                                         span->GetSize(), true, false);
  }

  auto ref_span = span->GetRefSpan();
  if (ref_span != nullptr) {
    GELOGI("Free ref span device_id:%u allocate_size:%zu mem_addr:%p.",
                device_allocator_.GetDeviceId(), ref_span->GetSize(), ref_span->GetAddr());
    FreeSpanEx(ref_span);
    span->SetRefSpan(nullptr);
  }

  occupied_spans_.remove(*span);
  span = TryMergeNext(*span);
  span = TryMergePrev(*span);
  FreeSpan(*span);
}

PageSpan *ScalableAllocator::PopSpanFromLayer(SpanLayer &layer) {
  PageSpan *span = layer.PopSpan();
  span_layer_lut_->OnLayerRemoveSpan(layer);
  return span;
}

void ScalableAllocator::PushSpanToLayer(SpanLayer &layer, PageSpan &span) {
  layer.PushSpan(span);
  span_layer_lut_->OnLayerAddSpan(layer);
}

void ScalableAllocator::RemoveFromLayer(SpanLayer &layer, PageSpan &span) {
  layer.Remove(span);
  span_layer_lut_->OnLayerRemoveSpan(layer);
}

void ScalableAllocator::OccupySpan(PageSpan &span, PageLen page_len) {
  span.Alloc(page_len);
  occupied_spans_.push_front(span);
  alloc_succ_count_++;
}

void ScalableAllocator::FreeSpan(PageSpan &span) {
  SpanLayer *layer = nullptr;
  if (span.IsNewVaSpan()) {
    layer = FetchNewVaSpanLayer(span.GetPageLen());
    if (layer != nullptr) {
      layer->PushSpan(span);
    }
    return;
  } else {
    layer = FetchSpanLayer(span.GetPageLen());
  }
  if (layer == nullptr) {
    return;
  }

  PushSpanToLayer(*layer, span);
  free_succ_count_++;
}

void ScalableAllocator::FreeSpanEx(PageSpan *span) {
  while (span->GetCount() > 0U) {
    span->SubCount();
  }
  occupied_spans_.remove(*span);
  span = TryMergeNext(*span);
  span = TryMergePrev(*span);
  FreeSpan(*span);
}

void ScalableAllocator::Recycle() {
  if (span_layer_lut_ == nullptr) {
    return;
  }
  size_t new_span_count = 0U;
  if (new_va_size_ > 0U) {
    // 必须先回收new_va，物理内存释放是在va回收时处理
    for (auto layer_id : new_span_layer_ids_) {
      auto layer = FetchNewVaSpanLayer(layer_id);
      if (layer != nullptr) {
        layer->Recycle(span_allocator_, device_allocator_);
        new_span_count++;
      }
    }
  }
  GELOGI("Total count:%zu new_va count:%zu", span_layer_lut_->size() + new_span_count, new_span_count);
  span_layer_lut_->Recycle(span_allocator_, device_allocator_);
  recycle_count_++;
  PrintDetails(GeLogLevel::kInfo);
}

size_t ScalableAllocator::GetIdleSpanCountOfLayer(const SpanLayerId layer_id) const {
  if (span_layers_.empty() || (layer_id >= span_layer_capacity_)) {
    return 0U;
  }
  return (span_layers_[layer_id] != nullptr) ? span_layers_[layer_id]->GetSize() : 0U;
}

size_t ScalableAllocator::GetIdleSpanCount() const {
  size_t span_sum = 0U;
  for (const auto &layer_id : *span_layer_lut_) {
    span_sum += GetIdleSpanCountOfLayer(layer_id);
  }
  return span_sum;
}

size_t ScalableAllocator::GetIdlePageCountOfLayer(const SpanLayerId layer_id) const {
  if (span_layers_.empty() || (layer_id >= span_layer_capacity_)) {
    return 0U;
  }
  return (span_layers_[layer_id] != nullptr) ? span_layers_[layer_id]->GetPageSize() : 0U;
}

size_t ScalableAllocator::GetIdlePageCount() const {
  size_t page_sum = 0U;
  for (const auto &layer_id : *span_layer_lut_) {
    page_sum += GetIdlePageCountOfLayer(layer_id);
  }
  return page_sum;
}

MemSize ScalableAllocator::GetIdleMemSizeOfLayer(const SpanLayerId layer_id) const {
  return PageLen_GetMemSize(GetIdlePageCountOfLayer(layer_id), config_.page_idem_num);
}

MemSize ScalableAllocator::GetIdleMemSize() const {
  MemSize mem_size = 0U;
  for (const auto &layer_id : *span_layer_lut_) {
    mem_size += GetIdleMemSizeOfLayer(layer_id);
  }
  return mem_size;
}

size_t ScalableAllocator::GetOccupiedSpanCount() const {
  return occupied_spans_.size();
}

size_t ScalableAllocator::GetOccupiedPageCount() const {
  size_t page_sum = 0U;
  for (auto &span : occupied_spans_) {
    page_sum += span.GetPageLen();
  }
  return page_sum;
}

size_t ScalableAllocator::GetRecycleCount() const {
  return recycle_count_;
}

MemSize ScalableAllocator::GetOccupiedMemSize() const {
  return PageLen_GetMemSize(GetOccupiedPageCount(), config_.page_idem_num);
}

void ScalableAllocator::PrintDetails(const int32_t level) {
  if (((level != GeLogLevel::kEvent) && (!IsLogEnable(GE_MODULE_NAME, level))) ||
      ((device_allocator_.GetOccupiedSize() == 0) && (!is_fix_sized_))) {
    return;
  }

  LOG_BY_TYPE(level, "Device memory   : [alloc count:%zu free count:%zu total size:%s new_va_size:%s real_theory_min:%s"
      " theory_min:%s reach theory rate:%.2f%s]", device_allocator_.GetAllocCount(), device_allocator_.GetFreeCount(),
      ge::ActiveMemoryUtil::SizeToString(device_allocator_.GetOccupiedSize()).c_str(),
      ge::ActiveMemoryUtil::SizeToString(new_va_size_).c_str(),
      ge::ActiveMemoryUtil::SizeToString(real_theory_min_size_).c_str(),
      ge::ActiveMemoryUtil::SizeToString(theory_min_size_).c_str(), GetReachTheoryRate(), "%");
  LOG_BY_TYPE(level, "Allocator memory: [alloc count:%zu free count:%zu recycle count:%zu]", alloc_succ_count_,
              free_succ_count_, recycle_count_);

  std::map<size_t, size_t> occupied_span_stat;
  size_t total_page_count = 0U;
  for (const auto &span : occupied_spans_) {
    if (!span.IsNewVaSpan()) {
      total_page_count += span.GetPageLen();
      occupied_span_stat[span.GetPageLen()]++;
    }
  }

  LOG_BY_TYPE(level, "Using: [span count:%zu page count:%zu total size:%llu]",
              occupied_spans_.size(), total_page_count, PageLen_GetMemSize(total_page_count, config_.page_idem_num));
  for (const auto &stat : occupied_span_stat) {
    LOG_BY_TYPE(level, "    |-span: [size:%-11llu count:%-5zu]",
                PageLen_GetMemSize(stat.first, config_.page_idem_num), stat.second);
  }

  size_t total_span_count = 0U;
  MemSize total_mem_size = 0U;
  total_page_count = 0U;
  LOG_BY_TYPE(level, "Freed span layers: [level:%u, count:%zu]", span_layer_capacity_, span_layer_lut_->size());
  for (const auto &layer_id : *span_layer_lut_) {
    if ((!span_layers_.empty()) && (layer_id < span_layer_capacity_) && (span_layers_[layer_id] != nullptr)) {
      total_span_count += span_layers_[layer_id]->GetSize();
      total_page_count += span_layers_[layer_id]->GetPageSize();
      total_mem_size += PageLen_GetMemSize(span_layers_[layer_id]->GetPageSize(), config_.page_idem_num);
    }
  }

  LOG_BY_TYPE(level, "Freed: [span count:%zu page count:%zu total size:%llu]", total_span_count, total_page_count,
              total_mem_size);
  for (const auto &layer_id : *span_layer_lut_) {
    if ((!span_layers_.empty()) && (layer_id < span_layer_capacity_) && (span_layers_[layer_id] != nullptr)) {
      LOG_BY_TYPE(level, "    |-layer: [id:%-5u capacity:%-5zu size:%-11llu count:%-5zu]",
                  span_layers_[layer_id]->GetLayerId(), span_layers_[layer_id]->GetCapacity(),
                  PageLen_GetMemSize(span_layers_[layer_id]->GetLayerId(), config_.page_idem_num),
                  span_layers_[layer_id]->GetSize());
    }
  }
  if (new_va_size_ > 0U) {
    PrintDetailsNewVa(level);
  }
}

void ScalableAllocator::PrintDetailsNewVa(const int32_t level) {
  std::map<size_t, size_t> occupied_span_stat;
  size_t total_page_count = 0U;
  size_t total_count = 0U;
  for (const auto &span : occupied_spans_) {
    if (span.IsNewVaSpan()) {
      total_page_count += span.GetPageLen();
      occupied_span_stat[span.GetPageLen()]++;
    }
  }

  LOG_BY_TYPE(level, "Using: [new_va_span count:%zu page count:%zu total size:%llu]",
              total_count, total_page_count, PageLen_GetMemSize(total_page_count, config_.page_idem_num));
  for (const auto &stat : occupied_span_stat) {
    LOG_BY_TYPE(level, "    |-new_va_span: [size:%-11llu count:%-5zu]",
                PageLen_GetMemSize(stat.first, config_.page_idem_num), stat.second);
  }

  size_t total_span_count = 0U;
  MemSize total_mem_size = 0U;
  total_page_count = 0U;
  for (auto layer_id : new_span_layer_ids_) {
    auto layer = FetchNewVaSpanLayer(layer_id);
    if (layer != nullptr) {
      total_span_count += layer->GetSize();
      total_page_count += layer->GetPageSize();
      total_mem_size += PageLen_GetMemSize(layer->GetPageSize(), config_.page_idem_num);
    }
  }
  LOG_BY_TYPE(level, "Freed new_va_span layers: [level:%u, count:%zu]", span_layer_capacity_, total_span_count);
  LOG_BY_TYPE(level, "Freed: [new_va_span count:%zu page count:%zu total size:%llu]", total_span_count, total_page_count,
              total_mem_size);
  for (auto layer_id : new_span_layer_ids_) {
    auto layer = FetchNewVaSpanLayer(layer_id);
    if (layer != nullptr) {
      LOG_BY_TYPE(level, "    |-new_va_layer: [id:%-5u capacity:%-5zu size:%-11llu count:%-5zu]", layer->GetLayerId(),
                  layer->GetCapacity(), PageLen_GetMemSize(layer->GetLayerId(), config_.page_idem_num),
                  layer->GetSize());
    }
  }
}

size_t ScalableAllocator::GetAllocatorId() const {
  return allocator_id_;
}

const std::string &ScalableAllocator::GetId() const {
  return allocator_id_with_type_;
}

void ScalableAllocator::InitExpandableAllocator(ge::Allocator &allocator, const rtMemType_t memory_type) {
  size_t virtual_memory_size = static_cast<size_t>(config_.page_mem_size_total_threshold - ge::kLargePageSize);
  const auto virtual_active_addr = device_allocator_.GetExpandableAllocator().ReserveVirtualMemory(virtual_memory_size,
      device_allocator_.GetDeviceId(), memory_type);
  if (virtual_active_addr != nullptr) {
    const auto span = BlockAlloc(allocator, nullptr, virtual_active_addr, virtual_memory_size);
    if (span != nullptr) {
      base_addr_ = span->GetAddr();
      GELOGI("base_addr:%p size:%zu", base_addr_, span->GetSize());
      // span is idle, so the usage count needs to be 0
      while (span->GetCount() > 0U) {
        span->SubCount();
      }

      const SpanLayerId fit_layer_id =
          SpanLayerId_GetIdFromSize(virtual_memory_size, ScalableAllocator::config_.page_idem_num);
      const auto layer = FetchSpanLayer(fit_layer_id);
      if (layer != nullptr) {
        PushSpanToLayer(*layer, *span);
        PrintDetails(GeLogLevel::kEvent);
        return;
      }
      span_allocator_.Free(*span);
    }
    device_allocator_.GetExpandableAllocator().ReleaseVirtualMemory();
  }
  return;
}

ge::Status ScalableAllocator::InitFixSizedAllocator(ge::Allocator &allocator, void *base_addr, size_t size) {
  device_allocator_.SetLogErrorIfAllocFailed(false);
  is_fix_sized_ = true;
  const auto span = BlockAlloc(allocator, nullptr, static_cast<uint8_t *>(base_addr), size);
  if (span != nullptr) {
    span->SetSplitable(true);
    base_addr_ = span->GetAddr();
    GELOGI("base_addr:%p size:%zu", base_addr_, span->GetSize());
    // span is idle, so the usage count needs to be 0
    while (span->GetCount() > 0U) {
      span->SubCount();
    }

    const SpanLayerId fit_layer_id =
        SpanLayerId_GetIdFromSize(size, ScalableAllocator::config_.page_idem_num);
    const auto layer = FetchSpanLayer(fit_layer_id);
    if (layer != nullptr) {
      PushSpanToLayer(*layer, *span);
      PrintDetails(GeLogLevel::kEvent);
      return ge::SUCCESS;
    }
    span_allocator_.Free(*span);
  }
  return ge::FAILED;
}

const std::string ScalableAllocator::GetStatics() const {
  return "new_va_size:" + ge::ActiveMemoryUtil::SizeToString(new_va_size_);
}

float ScalableAllocator::GetReachTheoryRate() const {
  if (device_allocator_.GetOccupiedSize() != 0U) {
    if (device_allocator_.GetOccupiedSize() >= real_theory_min_size_) {
      return (ge::kRatioBase * static_cast<float>(real_theory_min_size_))
          / static_cast<float>(device_allocator_.GetOccupiedSize());
    } else {
      // 回收后实际占用会比理论最小值小
      return static_cast<float>(ge::kRatioBase);
    }
  } else {
    return 0.0;
  }
}
}
