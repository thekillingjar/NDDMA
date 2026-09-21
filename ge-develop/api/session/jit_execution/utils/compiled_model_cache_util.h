/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMPILED_MODEL_CACHE_UTIL_H
#define COMPILED_MODEL_CACHE_UTIL_H

#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include "api/session/jit_execution/exe_points/execution_point.h"
#include "api/session/jit_execution/exe_points/guarded_execution_point.h"
#include "graph/graph.h"
#include "graph/ge_local_context.h"
#include "graph/utils/graph_utils_ex.h"
#include "mmpa/mmpa_api.h"
#include "graph_metadef/graph/utils/file_utils.h"

namespace ge {
namespace {
constexpr const char *kCompiledModelCacheDirName = "jit";
constexpr const char *kSlicingHierarchySubDirName = "slicing_hierarchy";
constexpr const char *kSlicingResultJsonFileName = "slicing_result.json";
constexpr const char *kGEPListJsonFileName = "gep_list.json";
constexpr const char *kSliceGraphPbFileName = "slice_graph.pb";
constexpr const char *kRemGraphPbFileName = "rem_graph.pb";

constexpr const char *kSliceGraphInfoSliceGraphIDKeyName = "slice_graph_id";
constexpr const char *kSliceGraphInfoMaxCntKeyName = "max_cnt";
constexpr const char *kSlicingResultUserGraphKeyKeyName = "user_graph_key";
constexpr const char *kSlicingResultUserGraphIDKeyName = "user_graph_id";
constexpr const char *kSlicingResultSliceGraphListKeyName = "slice_graph_list";
constexpr const char *kGuardedExecutionPointInfoKeyName = "gep_graph_key";
constexpr const char *kGuardedExecutionPointListKeyName = "gep_list";

constexpr const uint8_t kMaxErrStrLen = 128u;
}

struct SliceGraphInfo {
  int64_t slice_graph_id;
  uint32_t max_cnt;
};

struct SlicingResult {
  std::string user_graph_key;
  uint32_t user_graph_id;
  std::vector<SliceGraphInfo> slice_graph_infos;
};

struct GuardedExecutionPointInfo {
  std::string gep_graph_key;
};

struct GuardedExecutionPointInfoList {
  std::vector<GuardedExecutionPointInfo> gep_list;
};

using Json = nlohmann::json;
}
#endif // COMPILED_MODEL_CACHE_UTIL_H