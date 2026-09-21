/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_ENGINE_INC_FFTS_UTIL_H_
#define FFTS_ENGINE_INC_FFTS_UTIL_H_

#include <memory>
#include <vector>
#include "inc/ffts_log.h"
#include "inc/ffts_type.h"
#include "graph/op_desc.h"
#include "graph/node.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/math_util.h"
#include "common/string_utils.h"

namespace ffts {
#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define FFTS_FUNC_VISIBILITY _declspec(dllexport)
#else
#define FFTS_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define FFTS_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define FFTS_FUNC_VISIBILITY
#endif
#endif

// send error message to error manager
void LogErrorMessage(std::string error_code, const std::map<std::string, std::string> &args_map);

int64_t GetMicroSecondTime();

FFTSMode GetPlatformFFTSMode();

void DumpGraph(const ge::ComputeGraph &graph, const std::string &suffix);

std::string RealPath(const std::string &path);

bool IsSubGraphData(const ge::OpDescPtr &op_desc_ptr);

bool IsSubGraphDataWithControlEdge(const ge::NodePtr &node, uint32_t recurise_cnt);

bool IsSubGraphNetOutput(const ge::OpDescPtr &op_desc_ptr);

bool IsMemoryEmpty(const ge::GeTensorDesc &tensor_desc);

bool IsPhonyOp(const ge::OpDescPtr &op_desc_ptr);
Status IsNoEdge(const ge::NodePtr &parant_node, const ge::NodePtr &child_node);
void PrintNode(const ge::NodePtr &node);
void DeleteNode(ge::NodePtr &node);
void PrintNodeAttrExtNode(const ge::NodePtr &node, std::string attrname);
void PrintNodeAttrExtNodes(const ge::NodePtr &node, std::string attrname);
Status UnfoldPartionCallOnlyOneDepth(ge::ComputeGraph& graph, const std::string &node_type);
// Record the start time of stage.
#define FFTS_TIMECOST_START(stage) int64_t start_usec_##stage = GetMicroSecondTime();

// Print the log of time cost of stage.
#define FFTS_TIMECOST_END(stage, stage_name)                                                \
  do {                                                                                      \
    int64_t end_usec_##stage = GetMicroSecondTime();                                        \
    FFTS_LOGI("[FFTS_PERFORMANCE]The time cost of %s is [%ld] micro second.", (stage_name), \
              (end_usec_##stage - start_usec_##stage));                                     \
  } while (false)

#define FFTS_TIMECOST_END_LOGI(stage, stage_name)                                           \
  do {                                                                                      \
    int64_t end_usec_##stage = GetMicroSecondTime();                                        \
    FFTS_LOGI("[FFTS_PERFORMANCE]The time cost of %s is [%ld] micro second.", (stage_name), \
              (end_usec_##stage - start_usec_##stage));                                     \
  } while (false)

template <typename T, typename... Args>
inline std::shared_ptr<T> FFTSComGraphMakeShared(Args &&... args) {
  using T_nc = typename std::remove_const<T>::type;
  const std::shared_ptr<T> ret(new (std::nothrow) T_nc(std::forward<Args>(args)...));
  return ret;
}

inline Status CheckInt64AddOverflow(const int64_t &a, const int64_t &b) {
  if (((b > 0) && (a > (static_cast<int64_t>(INT64_MAX) - b))) ||
      ((b < 0) && (a < (static_cast<int64_t>(INT64_MIN) - b)))) {
    return FAILED;
  }
  return SUCCESS;
}

inline Status CheckInt64MulOverflow(const int64_t &m, const int64_t &n) {
  if (m > 0) {
    if (n > 0) {
      if (m > (static_cast<int64_t>(INT64_MAX) / n)) {
        return FAILED;
      }
    } else {
      if (n < (static_cast<int64_t>(INT64_MIN) / m)) {
        return FAILED;
      }
    }
  } else {
    if (n > 0) {
      if (m < (static_cast<int64_t>(INT64_MIN) / n)) {
        return FAILED;
      }
    } else {
      if ((m != 0) && (n < (static_cast<int64_t>(INT64_MAX) / m))) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

#define FFTS_INT64_ADDCHECK(a, b)                                              \
  if (CheckInt64AddOverflow((a), (b)) != SUCCESS) {                            \
    FFTS_LOGE("Int64 %ld and %ld addition can result in overflow!", (a), (b)); \
    return FAILED;                                                             \
  }

#define FFTS_INT64_MULCHECK(a, b)                                                                  \
  if (CheckInt64MulOverflow((a), (b)) != SUCCESS) {                                                \
    FFTS_LOGE("INT64 %ld and %ld multiplication can result in overflow!", static_cast<int64_t>(a), \
              static_cast<int64_t>(b));                                                            \
    return FAILED;                                                                                 \
  }

#define FFTS_ADD_OVERFLOW(x, y, z)        \
  if (ge::AddOverflow((x), (y), (z))) {   \
    return FAILED;                        \
  }                                       \

#define FFTS_MUL_OVERFLOW(x, y, z)        \
  if (ge::MulOverflow((x), (y), (z))) {   \
    return FAILED;                        \
  }                                       \

constexpr uint32_t kOneByteBitNum = 8U;
void SetBitOne(const uint32_t pos, uint32_t &bm);
void SetBitOne(const uint32_t pos, uint64_t &bm);

inline Status GetDataTypeSize(const ge::DataType &data_type, uint32_t &data_type_size) {
  const int32_t res = ge::GetSizeByDataType(data_type);
  if (res < 0) {
    return FAILED;
  }
  data_type_size = static_cast<uint32_t>(res);
  return SUCCESS;
}

template <typename T>
void LoopPrintIntergerVec(const std::vector<T> &interger_vec, const char *fmt, ...) {
  if (CheckLogLevel(static_cast<int32_t>(FFTS), DLOG_DEBUG) != 1) {
    return;
  }
  static constexpr int32_t fmt_buff_size = 1024;
  static constexpr int32_t print_vec_num = 100;
  va_list args;
  va_start(args, fmt);
  char fmt_buff[fmt_buff_size] = {0};
  if (vsnprintf_s(fmt_buff, fmt_buff_size, fmt_buff_size - 1, fmt, args) == -1) {
    return;
  }
  va_end(args);

  size_t vec_size = interger_vec.size();
  size_t start = 0;
  size_t end = start + print_vec_num;
  while (vec_size >= end) {
    std::vector<T> slice(interger_vec.begin() + start, interger_vec.begin() + end);
    std::string vec_str = fe::StringUtils::IntegerVecToString(slice);
    FFTS_LOGD("%s%s", fmt_buff, vec_str.c_str());
    start = end;
    end += print_vec_num;
  }
  std::vector<T> slice(interger_vec.begin() + start, interger_vec.end());
  std::string vec_str = fe::StringUtils::IntegerVecToString(slice);
  FFTS_LOGD("%s%s", fmt_buff, vec_str.c_str());
}

using SameAtomicNodeMap = std::map<std::vector<std::string>, std::vector<ge::NodePtr>>;
void GenerateSameAtomicNodesMap(std::vector<ge::NodePtr> &graph_nodes, SameAtomicNodeMap &same_memset_nodes_map);
}  // namespace ffts
#endif  // FFTS_ENGINE_INC_FFTS_UTIL_H_
