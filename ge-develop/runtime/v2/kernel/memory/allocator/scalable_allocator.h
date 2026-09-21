/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H5CF96432_BE55_46BE_B9E1_8F7A5C662D50
#define H5CF96432_BE55_46BE_B9E1_8F7A5C662D50

#include <memory>
#include "runtime/mem_allocator.h"
#include "kernel/memory/allocator/scalable_config.h"
#include "kernel/memory/device/device_allocator.h"
#include "kernel/memory/span/span_layer_allocator.h"
#include "kernel/memory/span/span_allocator.h"
#include "kernel/memory/span/span_layer_lut.h"
#include "memory_pool.h"

namespace gert {
enum GeLogLevel : int32_t {
  kDebug = 0,
  kInfo = 1,
  kWarn = 2,
  kError = 3,
  kNull = 4,
  kEvent = 10
};
#define LOG_BY_TYPE(type, fmt, ...)                           \
  do {                                                        \
    if (type == GeLogLevel::kInfo) {          \
      dlog_info(GE_MODULE_NAME, "%lu %s %s:" fmt, GeLog::GetTid(), GetId().c_str(), __FUNCTION__, ##__VA_ARGS__);  \
    } else if (type == GeLogLevel::kError) {  \
      GELOGE(ge::FAILED, fmt, ##__VA_ARGS__);                 \
    } else if (type == GeLogLevel::kEvent){   \
      GEEVENT(fmt, ##__VA_ARGS__);                            \
    } else {}                                                 \
  } while (false)
class VISIBILITY_EXPORT ScalableAllocator : public MemoryPool {
 public:
  explicit ScalableAllocator(SpanAllocator &span_allocator, DeviceMemAllocator &device_allocator,
                             const ScalableConfig &cfg = ScalableConfig(),
                             const std::string &graph_name = "");
  ~ScalableAllocator() override;

  PageSpan *Alloc(ge::Allocator &allocator, const MemSize size) override;
  void Free(ge::MemBlock *block) override;
  void Recycle() override;
  void PrintDetails(const int32_t level = GeLogLevel::kError) override;
  ge::Status Finalize(bool no_log = false) override;
  ge::MemBlock *ConvertToRootBlock(ge::MemBlock *block) override;
  void InitExpandableAllocator(ge::Allocator &allocator, const rtMemType_t memory_type = RT_MEMORY_HBM);
  ge::Status InitFixSizedAllocator(ge::Allocator &allocator, void *base_addr, size_t size);

  size_t GetIdleSpanCountOfLayer(const SpanLayerId layer_id) const;
  size_t GetIdlePageCountOfLayer(const SpanLayerId layer_id) const;
  MemSize GetIdleMemSizeOfLayer(const SpanLayerId layer_id) const;

  size_t GetIdleSpanCount() const;
  size_t GetIdlePageCount() const;
  MemSize GetIdleMemSize() const;

  size_t GetOccupiedSpanCount() const;
  size_t GetOccupiedPageCount() const;
  MemSize GetOccupiedMemSize() const;

  size_t GetRecycleCount() const;
  const ScalableConfig &GetScalableConfig() const { return config_; }
  const std::string &GetId() const override;
  size_t GetAllocatorId() const;
  const std::string GetStatics() const;
  float GetReachTheoryRate() const;

 protected:
  virtual BlockAddr DevAlloc(const MemSize size);
  virtual PageSpan *BlockAlloc(ge::Allocator &allocator, const BlockAddr block_addr, const MemAddr addr,
                               const size_t size);
  virtual bool IsThresholdExceeded(const MemSize size) const;
  virtual PageSpan *FetchNewSpan(ge::Allocator &allocator, const MemSize size, const PageLen page_len);
  virtual PageSpan *SplitSpan(ge::Allocator &allocator, const SpanLayerId fix_layer_id,
                              const SpanLayerId fit_layer_id, PageSpan *const span, const MemSize size);
 private:
  PageSpan *FetchLayerSpan(const SpanLayerId layer_id);
  PageSpan *FetchSplitedSpan(ge::Allocator &allocator, const SpanLayerId fix_layer_id, const SpanLayerId fit_layer_id,
                             const MemSize size, int32_t &ret);
  PageSpan *PickOutBuddy(PageSpan *const buddy_span);
  PageSpan *TryMergePrev(PageSpan &span);
  PageSpan *TryMergeNext(PageSpan &span);

  PageSpan *PopSpanFromLayer(SpanLayer &layer);
  void PushSpanToLayer(SpanLayer &layer, PageSpan &span);
  void RemoveFromLayer(SpanLayer &layer, PageSpan &span);

  void OccupySpan(PageSpan &span, PageLen page_len);
  void FreeSpan(PageSpan &span);
  PageSpan *AllocImp(ge::Allocator &allocator, const MemSize size, int32_t &ret);
  PageSpan *FetchNewVaSpan(ge::Allocator &allocator, const MemSize size, std::vector<size_t> &pa_list);
  PageSpan *FetchNewVaLayerSpan(const SpanLayerId layer_id);
  SpanLayer *FetchNewVaSpanLayer(const SpanLayerId layer_id);
  void FreeSpanEx(PageSpan *span);
  PageSpan *ProcessNewVaSpan(ge::Allocator &allocator, PageSpan *span, const MemSize size, size_t reuse_size,
                             const MemSize alloc_size);
  PageSpan *ProcessPaUsingSpan(ge::Allocator &allocator, PageSpan *span, const MemSize size, size_t reuse_size,
                               const MemSize alloc_size);
  void PrintDetailsNewVa(const int32_t level);
  SpanLayer *FetchSpanLayer(const SpanLayerId layer_id);
  MemSize GetAllocSize(const MemSize size) const;
  PageLen GetlayerSpanCapacity(const SpanLayerId layer_id) const;
  bool IsLayerCacheable(const SpanLayerId layer_id) const;

 protected:
  DeviceAllocator device_allocator_;
  size_t allocator_id_;
  std::string allocator_id_with_type_;
  const ScalableConfig config_;

 private:
  MemSize page_mem_size_;
  SpanLayerId span_layer_capacity_;
  SpanLayerId uncacheable_layer_start_;
  PageLen span_layer_page_capacity_;
  size_t recycle_count_{0U};
  size_t alloc_succ_count_{0U};
  size_t free_succ_count_{0U};
  Link<PageSpan> occupied_spans_;
  std::vector<SpanLayer *> span_layers_;
  std::vector<SpanLayer *> new_span_layers_;
  std::set<SpanLayerId> new_span_layer_ids_;
  std::unique_ptr<SpanLayerLut> span_layer_lut_;
  SpanAllocator &span_allocator_;
  SpanLayerAllocator layer_allocator_;
  static std::atomic_size_t global_allocator_id_;
  std::string graph_name_;
  MemSize max_occupied_size_{0U};
  void *base_addr_{nullptr};
  size_t theory_size_{0U};
  size_t theory_min_size_{0U};
  size_t real_theory_size_{0U};
  size_t real_theory_min_size_{0U};
  size_t try_count_{0U};
  size_t set_try_count_{10U};
  size_t new_va_size_{0U};
  bool is_fix_sized_{false};
};
}

#endif
