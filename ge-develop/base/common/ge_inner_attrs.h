/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_GE_INNER_ATTRS_H_
#define AIR_CXX_BASE_COMMON_GE_INNER_ATTRS_H_
#include "graph/types.h"

namespace ge {
constexpr char_t const *kAttrNameSingleOpType = "GE_INNER_SINGLE_OP_TYPE";
// profiling name
constexpr char_t const *kProfilingDeviceConfigData = "PROFILING_DEVICE_CONFIG_DATA";
constexpr char_t const *kProfilingIsExecuteOn = "PROFILING_IS_EXECUTE_ON";
// helper option
constexpr char_t const *kHostMasterPidName = "HOST_MASTER_PID";
constexpr char_t const *kExecutorDevId = "EXECUTOR_DEVICE_ID";
// runtime 2.0
constexpr char_t const *kRequestWatcher = "_request_watcher";
constexpr char_t const *kWatcherAddress = "_watcher_address";
constexpr char_t const *kSubgraphInput = "_subgraph_input";
constexpr char_t const *kSubgraphOutput = "_subgraph_output";
constexpr char_t const *kKnownSubgraph = "_known_subgraph";
constexpr char_t const *kRelativeBranch = "branch";
constexpr char_t const *kConditionGraph = "CondGraph";
constexpr char_t const *kThenGraph = "then_graph";
constexpr char_t const *kElseGraph = "else_graph";
}
#endif // AIR_CXX_BASE_COMMON_GE_INNER_ATTRS_H_
