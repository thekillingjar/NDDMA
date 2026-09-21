/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/memory/binary_block_mem_assigner.h"
#include <algorithm>
#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils.h"
#include "common/checker.h"
#include "base/err_msg.h"

namespace {
const uint32_t kRangeCeilInterval = 2;
const uint32_t kLogBase = 2;
const int64_t kLargeBlockSize = 8388608;   // 8 * 1024 * 1024
const int64_t kLargeBlockRangeSize = 2;
}  // namespace

namespace ge {
void BinaryBlockMemAssigner::PlanRanges(size_t range_number_limit, std::vector<std::vector<int64_t>> &ranges) const {
  /// range delete and merge
  /// result after delete and merge is: [[6,12],[16,24,24],[30,32,48],[60,256]]
  bool changed = false;
  std::vector<int64_t> temp;
  do {
    changed = false;
    for (auto iter = ranges.begin(); iter != ranges.end();) {
      if (!temp.empty()) {
        iter->insert(iter->end(), temp.begin(), temp.end());
        temp.clear();
      }
      if (iter->empty()) {
        iter = ranges.erase(iter);
        changed = true;
      } else if ((iter->size() < range_number_limit) && (ranges.end() - iter > 1) &&
                 !(iter->at(0) >= kLargeBlockSize && iter->size() >= kLargeBlockRangeSize)) {
        temp.insert(temp.end(), iter->begin(), iter->end());
        iter = ranges.erase(iter);
        changed = true;
      } else {
        ++iter;
      }
    }
  } while (changed);
}

///
/// @ingroup domi_omg
/// @brief memory size fixed for reuse. this function determines memory types and sizes
/// @param [out] range_ceils return memory sizes
/// @return Status result
/// @author
///
Status BinaryBlockMemAssigner::GetMemoryRanges(std::vector<int64_t> &range_ceils) {
  std::vector<int64_t> all_memory_size;
  GE_ASSERT_SUCCESS(GetOutAndWorkSpaceMem(all_memory_size));
  if (all_memory_size.empty()) {
    GELOGW("Vector all_memory_size is empty!");
    return SUCCESS;
  }
  if (all_memory_size.front() <= 0) {
    GELOGE(FAILED, "[Check][MemRangeStep]first mem_range_step:%ld less than 0,invalid,"
          "maybe has dynamic shape in graph", all_memory_size.front());
    REPORT_INNER_ERR_MSG("E19999", "first mem_range_step:%ld less than 0,invalid,"
          "maybe has dynamic shape in graph", all_memory_size.front());
    return FAILED;
  }
  // Memory size is 512 aligned, so it is not necessary to take less than 512
  int64_t min_memory_size = (all_memory_size.back() > MEM_ALIGN_SIZE) ? MEM_ALIGN_SIZE : all_memory_size.front();
  auto range_number = static_cast<size_t>(
    ceil(log(all_memory_size.back() / static_cast<double>(min_memory_size)) / log(kLogBase)));
  range_number = (range_number == 0) ? 1 : range_number;
  GELOGD("Range number: %zu", range_number);

  std::vector<std::vector<int64_t>> ranges(range_number);
  GE_CHK_BOOL_EXEC((range_number != 0),
    REPORT_INNER_ERR_MSG("E19999", "inner data[range_number] is 0, judge invalid");
    return PARAM_INVALID,
    "[Check][RangeNumber]inner data is 0, judge invalid.");
  size_t range_number_limit = all_memory_size.size() / range_number;
  int64_t range_ceil = all_memory_size.back();
  const int64_t align_size = (range_ceil > MEM_ALIGN_SIZE) ? MEM_ALIGN_SIZE : 0;
  for (size_t i = range_number - 1; i > 0; --i) {
    if (range_ceil <= 0) {
      break;
    }
    range_ceil /= kRangeCeilInterval;  // The block size of each interval is doubled every time.
    for (auto iter = all_memory_size.rbegin(); iter != all_memory_size.rend();) {
      if (*iter > (range_ceil + align_size)) {
        ranges[i].push_back(*iter);
        all_memory_size.erase((++iter).base());
      } else {
        break;
      }
    }
  }

  GELOGD("Origin ranges:");
  for (auto &v : ranges) {
    GELOGD("__%s", ToString(v).c_str());
  }

  PlanRanges(range_number_limit, ranges);
  GELOGD("Origin ranges:");
  for (auto &v : ranges) {
    GELOGD("__%s", ToString(v).c_str());
  }

  for (auto &range : ranges) {
    std::sort(range.begin(), range.end());
    if (!range.empty()) {
      range_ceils.push_back(range.back());
    }
  }
  std::sort(range_ceils.begin(), range_ceils.end());
  GELOGI("Range ceils: %s", ToString(range_ceils).c_str());

  return SUCCESS;
}
}  // namespace ge
