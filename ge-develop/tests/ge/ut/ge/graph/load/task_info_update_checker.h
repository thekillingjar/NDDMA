/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_UT_GE_GRAPH_LOAD_TASK_INFO_UPDATE_CHECKER_H_
#define AIR_CXX_TESTS_UT_GE_GRAPH_LOAD_TASK_INFO_UPDATE_CHECKER_H_
#include "task_info_stubs.h"
namespace ge {
class TaskInfoUpdateChecker {
 public:
  TaskInfoUpdateChecker &CallTimes(size_t call_times);
};
}
#endif  // AIR_CXX_TESTS_UT_GE_GRAPH_LOAD_TASK_INFO_UPDATE_CHECKER_H_
