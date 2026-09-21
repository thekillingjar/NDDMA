/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_PROFILING_AUTO_CHECKER_H_
#define GE_PROFILING_AUTO_CHECKER_H_
#include "profiling_test_util.h"
#include "runtime/subscriber/global_profiler.h"
#include "aprof_pub.h"
#include <iostream>
namespace ge {
inline void DefaultProfilingTest(void (*test_func)()) {
  auto request_id = 1;
  auto event_count = 0;
  auto default_check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) -> int32_t {
    if (type == InfoType::kInfo) {
      auto info = reinterpret_cast<MsprofAdditionalInfo *>(data);
      EXPECT_NE(info->dataLen, 0);
    }

    if (type == InfoType::kEvent) {
      auto event = reinterpret_cast<MsprofEvent *>(data);
      request_id ^= (event->requestId);
      ++event_count;
    }
    return 0;
  };
  ProfilingTestUtil::Instance().SetProfFunc(default_check_func);
  test_func();
  EXPECT_EQ(request_id, 1);
  EXPECT_EQ(event_count % 2, 0);
  ProfilingTestUtil::Instance().Clear();
}

inline void EXPECT_DefaultProfilingTestWithExpectedCallTimes(void (*test_func)(), size_t api_count, size_t info_count,
                                                      size_t event_count, size_t compact_info_count) {
  auto request_id = 1;
  auto cur_event_count = 0;
  auto cur_info_count = 0;
  auto cur_api_count = 0;
  auto cur_compact_info_count = 0;
  auto default_check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) -> int32_t {
    if (type == InfoType::kInfo) {
      auto info = reinterpret_cast<MsprofAdditionalInfo *>(data);
      EXPECT_NE(info->dataLen, 0);
      ++cur_info_count;
      if (info->type == MSPROF_REPORT_NODE_TENSOR_INFO_TYPE) {
        auto prof_tensor_info = reinterpret_cast<MsprofTensorInfo *>(info->data);
        EXPECT_TRUE(prof_tensor_info->tensorNum > 0);
        for (size_t i=0; i<prof_tensor_info->tensorNum; ++i) {
          EXPECT_NE(prof_tensor_info->tensorData[i].shape[0], 0);
        }
      }
      if (info->type == MSPROF_REPORT_NODE_TASK_MEMORY_TYPE) {
        auto memory_info_data = reinterpret_cast<MsprofMemoryInfo *>(info->data);
        std::cout << "Report memory info: node_id: " << memory_info_data->nodeId
                  << ", addr: " << memory_info_data->addr
                  << ", size: " << memory_info_data->size
                  << ", total allocate size: " << memory_info_data->totalAllocateMemory
                  << ", total reserve size: " << memory_info_data->totalReserveMemory << std::endl;
      }
    }

    if (type == InfoType::kApi) {
      ++cur_api_count;
    }

    if (type == InfoType::kCompactInfo) {
      ++cur_compact_info_count;
    }

    if (type == InfoType::kEvent) {
      auto event = reinterpret_cast<MsprofEvent *>(data);
      request_id ^= (event->requestId);
      ++cur_event_count;
    }
    return 0;
  };
  ProfilingTestUtil::Instance().SetProfFunc(default_check_func);
  test_func();
  EXPECT_EQ(request_id, 1);
  EXPECT_EQ(cur_event_count % 2, 0);
  EXPECT_EQ(cur_event_count, event_count);
  EXPECT_EQ(cur_api_count, api_count);
  EXPECT_EQ(cur_info_count, info_count);
  EXPECT_EQ(cur_compact_info_count, compact_info_count);
  ProfilingTestUtil::Instance().Clear();
}

inline void AutoProfilingTestWithExpectedFunc(void (*test_func)(), ProfilingTestUtil::ProfFunc func) {
  ProfilingTestUtil::Instance().SetProfFunc(func);
  test_func();
  ProfilingTestUtil::Instance().Clear();
}
}
#endif  // GE_PROFILING_AUTO_CHECKER_H_
