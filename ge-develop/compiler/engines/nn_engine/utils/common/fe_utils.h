/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_UTILS_COMMON_FE_UTIL_H_
#define FUSION_ENGINE_UTILS_COMMON_FE_UTIL_H_

#include <memory>
#include <vector>
#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/aicore_util_types.h"
#include "common/util/constants.h"
#include "common/util/op_info_util.h"
#include "graph/utils/type_utils.h"
#include "common/lxfusion_json_util.h"

namespace fe {
// Save error message to error manager
void SaveErrorMessage(const std::string &error_code, const std::string &key, const std::string &value);

int64_t GetMicroSecondTime();

uint64_t GetCurThreadId();

std::string GetCurThreadIdStr();

uint64_t GetAtomicId();

std::string FormatToStr(ge::Format format);
std::string DTypeToStr(ge::DataType d_type);

std::string GetImplTypeString(OpImplType op_impl_type);

std::string GetGeImplTypeString(domi::ImplyType ge_impl_type);

bool IsInvalidImplType(const std::string &engine_name, const OpImplType &op_impl_type);

std::string GetPassTypeString(GraphFusionPassType pass_type);

std::string GetBufferFusionPassTypeString(BufferFusionPassType pass_type);

constexpr uint32_t DefaultGroupID = 0xFFFFFFFFU;
bool IsHeavyFormatMatmul(const ge::NodePtr &node);

/**
 * check file is empty or not
 * @param file_name
 * @return
 */
bool CheckFileEmpty(const std::string &file_name);

bool IsRootGraphData(const std::string &op_node_type);

bool IsHeavyDataOp(const std::string &op_type);

int GetDataTypeBits(const ge::DataType data_type);

std::string GetStrTaskIdByMap(const std::map<uint64_t, int64_t> &task_scope_id_map);

void PrintOutputInplace(const ge::OpDescPtr &opdesc, const vector<vector<int64_t>> &inplace);

void GetIrIdexInstance(const ge::OpDescPtr &opdesc, 
                       std::map<size_t, std::pair<size_t, size_t>> &input_ir_real_index,
                       std::map<size_t, std::pair<size_t, size_t>> &output_ir_real_index);

Status TransDtypeToString(const ge::DataType &dtype, string &dtype_string);

Status TransStringToDtype(const string &dtype_str, ge::DataType &dtype);

template <typename T, typename... Args>
static inline std::shared_ptr<T> FeComGraphMakeShared(Args &&... args) {
  using T_nc = typename std::remove_const<T>::type;
  std::shared_ptr<T> ret(new (std::nothrow) T_nc(std::forward<Args>(args)...));
  return ret;
}

// Record the start time of stage.
#define FE_TIMECOST_START(stage) int64_t start_usec_##stage = GetMicroSecondTime();

// Print the log of time cost of stage to event.
#define FE_TIMECOST_END(stage, stage_name)                                              \
  {                                                                                     \
    int64_t end_usec_##stage = GetMicroSecondTime();                                    \
    FE_LOGI("[FE_PERFORMANCE]The time cost of %s is [%ld] micro second.", (stage_name), \
            (end_usec_##stage - start_usec_##stage));                                   \
  }

// Print the log of time cost of stage to info.
#define FE_TIMECOST_END_LOGI(stage, stage_name)                                         \
  do {                                                                                     \
    int64_t end_usec_##stage = GetMicroSecondTime();                                    \
    FE_LOGI("[FE_PERFORMANCE]The time cost of %s is [%ld] micro second.", (stage_name), \
            (end_usec_##stage - start_usec_##stage));                                   \
  } while (false)

#define RUN_AND_DUMP_WITH_TIMESTAMP_NAME(var_name, prefix, name, func, ...)              \
  do {                                                                                   \
    FE_TIMECOST_START(var_name);                                                         \
    const auto ret_inner_macro = (func)(__VA_ARGS__);                                    \
    FE_TIMECOST_END(var_name, #prefix "::" #func);                                       \
    if (ret_inner_macro != SUCCESS) {                                                    \
      FE_LOGE("[Process][" #prefix "_" #func "] failed");                                \
      FE_DUMP(graph, name "_Failed");                                                    \
      return ret_inner_macro;                                                            \
    };                                                                                   \
    FE_DUMP(graph, name);                                                                \
  } while (false)

#define RUN_WITH_TIMESTAMP_NAME(var_name, prefix, func, ...)                             \
  do {                                                                                   \
    FE_TIMECOST_START(var_name);                                                         \
    const auto ret_inner_macro = (func)(__VA_ARGS__);                                    \
    FE_TIMECOST_END(var_name, #prefix "::" #func);                                       \
    if (ret_inner_macro != SUCCESS) {                                                    \
      FE_LOGE("[Process][" #prefix "_" #func "] failed");                                \
      return ret_inner_macro;                                                            \
    }                                                                                    \
  } while (false)

#define FE_DUMP(graph, name)                                                             \
  do {                                                                                   \
    FeGraphUtils::DumpGraphAndOnnx(graph, name);                                         \
    FeGraphUtils::DumpSubGraphAndOnnx(graph, name "_Subgraph");                          \
  } while (false)

#define JOIN_NAME_INNER(a, b) a##b
#define JOIN_NAME(a, b) JOIN_NAME_INNER(a, b)
#define COUNTER_NAME(a) JOIN_NAME(a, __COUNTER__)
#define FE_RUN_AND_DUMP(prefix, name, func, ...) \
  RUN_AND_DUMP_WITH_TIMESTAMP_NAME(COUNTER_NAME(fe_timestamp_##prefix), prefix, name, func, __VA_ARGS__)
#define FE_RUN(prefix, func, ...) \
  RUN_WITH_TIMESTAMP_NAME(COUNTER_NAME(fe_timestamp_##prefix), prefix, func, __VA_ARGS__)
}  // namespace fe
#endif  // FUSION_ENGINE_UTILS_COMMON_FE_UTIL_H_
