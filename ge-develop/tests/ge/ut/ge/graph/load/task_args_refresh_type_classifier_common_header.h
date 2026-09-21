/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_UT_GE_GRAPH_LOAD_TASK_ARGS_REFRESH_TYPE_CLASSIFIER_COMMON_HEADER_H_
#define AIR_CXX_TESTS_UT_GE_GRAPH_LOAD_TASK_ARGS_REFRESH_TYPE_CLASSIFIER_COMMON_HEADER_H_
#include "graph/load/model_manager/task_args_refresh_type_classifier.h"

namespace ge {
inline bool operator<(const TaskArgsRefreshTypeClassifier::TaskFixedAddr &lhs,
               const TaskArgsRefreshTypeClassifier::TaskFixedAddr &rhs) {
  if (lhs.task_index != rhs.task_index) {
    return lhs.task_index < rhs.task_index;
  }
  if (lhs.iow_index_type != rhs.iow_index_type) {
    return lhs.iow_index_type < rhs.iow_index_type;
  }
  return lhs.iow_index < rhs.iow_index;
}
inline bool operator==(const TaskArgsRefreshTypeClassifier::TaskFixedAddr &lhs,
                const TaskArgsRefreshTypeClassifier::TaskFixedAddr &rhs) {
  return lhs.task_index == rhs.task_index && lhs.iow_index_type == rhs.iow_index_type && lhs.iow_index == rhs.iow_index;
}
}  // namespace ge
#endif  // AIR_CXX_TESTS_UT_GE_GRAPH_LOAD_TASK_ARGS_REFRESH_TYPE_CLASSIFIER_COMMON_HEADER_H_
