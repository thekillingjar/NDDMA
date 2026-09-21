/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_ENGINE_FFTS_FE_RT2_COMMON_H_
#define AIR_CXX_RUNTIME_V2_ENGINE_FFTS_FE_RT2_COMMON_H_
#include "base/err_msg.h"
#include "common/checker.h"
#include "framework/common/debug/ge_log.h"
#include "graph_builder/converter_checker.h"
namespace gert {
const std::string EM_INNER_ERROR = "E29999";
const std::string EM_INNER_WARN = "W29999";
constexpr const ge::char_t *kUnknownShapeFromFe = "_unknown_shape";
const std::vector<std::string> CMO_OPERATION = {"prefetch", "invalidate", "writeback"};

constexpr const ge::char_t *kContextId = "_context_id";
constexpr const ge::char_t *kContextIdList = "_context_id_list";
constexpr const ge::char_t *kTensorCtxId = "_tensor_ctx_id";
constexpr const ge::char_t *kThreadAddrOffset = "_thread_addr_offset";
constexpr const ge::char_t *kFFTSThreadDim = "_ffts_thread_dim";
constexpr const ge::char_t *kFFTSWindowSize = "_ffts_window_size";
constexpr const ge::char_t *kCoreTypeAIC = "AIC";
constexpr const ge::char_t *kCoreTypeAIV = "AIV";
constexpr const ge::char_t *kCoreTypeMixAIC = "MIX_AIC";
constexpr const ge::char_t *kCoreTypeMixAIV = "MIX_AIV";
constexpr const ge::char_t *kNeedModeAddr = "_need_mode_addr";
constexpr const ge::char_t *kAutoAtEndCtxIdList = "_at_end_ctx_id_list";
constexpr const ge::char_t *kThreadId = "_thread_id";
// this attr define by SGT to mark class 2 op
constexpr const ge::char_t *kThreadDependInputVal = "thread_depend_input_val";

constexpr const ge::char_t *kFFTSProfCtxType = "_ffts_prof_ctx_type";
constexpr const ge::char_t *kDataProfCtxIdV = "_data_prof_ctx_id_v";
constexpr const ge::char_t *kDataProfName = "_data_prof_name_v";
constexpr const ge::char_t *kDataProfType = "_data_prof_type";
constexpr const ge::char_t *kFESingleOpScene = "_fe_single_op_scene";
constexpr const ge::char_t *kAttrKernelBinId = "_kernel_bin_id";
constexpr const ge::char_t *kAttrMemsetKernelBinId = "_memset_kernel_bin_id";
constexpr const ge::char_t *kOptionalInputMode = "optionalInputMode";
constexpr const ge::char_t *kOptionalOutputMode = "optionalOutputMode";
constexpr const ge::char_t *kGenPlaceHolder = "gen_placeholder";

constexpr const ge::char_t *kOpKernelAllInputSize = "_op_kernel_all_input_size";
constexpr const ge::char_t *kOpKernelAllOutputSize = "_op_kernel_all_output_size";
constexpr const ge::char_t *kInputInsertOptPosList = "_input_insert_opt_pos_list";

constexpr const ge::char_t *kAttrDynamicParamMode = "dynamicParamMode";
constexpr const ge::char_t *kFoldedWithDesc = "folded_with_desc";
constexpr const ge::char_t *kDynamicInputsIndexes = "_dynamic_inputs_indexes";
constexpr const ge::char_t *kDynamicOutputsIndexes = "_dynamic_outputs_indexes";
constexpr const ge::char_t *kDisableMixVectorCore = "_disable_mix_vector_core";
constexpr const ge::char_t *kEnableMixVectorCore = "_enable_mix_vector_core";
constexpr const ge::char_t *kMixCoreNumVec = "_mix_core_num_vec";
constexpr const ge::char_t *kCoreTypeMixVectorCore = "MIX_VECTOR_CORE";
constexpr const ge::char_t *kCoreTypeMixAICore = "MIX_AICORE";
constexpr const ge::char_t *kLocalMemorySize = "local_memory_size";
struct DfxExeArg {
  bool need_assert{false};
  bool need_print{false};
  int64_t buffer_size{0U};
};
}
#define REPORT_FE_ERROR(fmt, ...)                               \
  do {                                                          \
    REPORT_INNER_ERR_MSG(EM_INNER_ERROR.c_str(), fmt, ##__VA_ARGS__);     \
    GELOGE(ge::FAILED, fmt, ##__VA_ARGS__);                     \
  } while (0)

#define REPORT_FE_WARN(fmt, ...)                                \
  do {                                                          \
    REPORT_INNER_ERR_MSG(EM_INNER_WARN.c_str(), fmt, ##__VA_ARGS__);      \
    GELOGW(fmt, ##__VA_ARGS__);                                 \
  } while (0)

#define CHECK_HOLDERS_ALL_OK_RET(holders, expect_num, return_expr)    \
  do {                                                                \
    if ((holders).size() != (expect_num)) {                           \
      GELOGE(ge::FAILED, "Size[%zu] not expect.", (holders).size());  \
      return_expr;                                                    \
    }                                                                 \
    for (const auto &holder_inter : (holders)) {                            \
      if (holder_inter == nullptr) {                                        \
        GELOGE(ge::FAILED, "Holder is null.");                        \
        return_expr;                                                  \
      }                                                               \
      if (!holder_inter->IsOk()) {                                          \
        GELOGE(ge::FAILED, "Holder is not ok.");                      \
        return_expr;                                                  \
      }                                                               \
    }                                                                 \
  } while (0)

#define FE_LOWER_REQUIRE(exp, ...)                               \
  do {                                                           \
    if (!(exp)) {                                                \
      auto msg = CreateErrorMsg(__VA_ARGS__);                    \
      if (msg.empty()) {                                         \
        REPORT_FE_ERROR("Assert %s failed", #exp);               \
        GELOGE(ge::FAILED, "Assert %s failed", #exp);            \
        return CreateErrorLowerResult("Assert %s failed", #exp); \
      }                                                          \
      REPORT_FE_ERROR("%s", msg.data());                         \
      GELOGE(ge::FAILED, "%s", msg.data());                      \
      return CreateErrorLowerResult(msg.data());                 \
    }                                                            \
  } while (false)

#define FE_LOWER_REQUIRE_NOTNULL(v, ...) FE_LOWER_REQUIRE(((v) != nullptr), __VA_ARGS__)
#define FE_LOWER_REQUIRE_SUCCESS(v, ...) FE_LOWER_REQUIRE(((v) == ge::SUCCESS), __VA_ARGS__)

#define FE_RET_NULL_RET_IF(ctx, ...)                              \
  do {                                                            \
    if (ctx) {                                                    \
      auto msg = CreateErrorMsg(__VA_ARGS__);                     \
      if (!msg.empty()) {                                         \
        REPORT_FE_ERROR("%s", msg.data());                        \
        GELOGE(ge::FAILED, "%s", msg.data());                     \
      }                                                           \
      return nullptr;                                             \
    }                                                             \
  } while (false)

#define FE_RET_ERR_RET_IF(ctx, msg)                          \
  do {                                                       \
    if (ctx) {                                               \
      if ((msg) != nullptr) {                                \
        REPORT_FE_ERROR("%s", msg);                          \
        GELOGE(ge::FAILED, "%s", msg);                       \
      }                                                      \
      return {HyperStatus::ErrorStatus(msg), {}, {}, {}};    \
    }                                                        \
  } while (false)


#define FE_ASSERT(exp, ...)                                               \
  do {                                                                    \
    if (!(exp)) {                                                         \
      auto msg = CreateErrorMsg(__VA_ARGS__);                             \
      if (msg.empty()) {                                                  \
        REPORT_FE_ERROR("Assert %s failed", #exp);                        \
        GELOGE(ge::FAILED, "Assert %s failed", #exp);                     \
      } else {                                                            \
        REPORT_FE_ERROR("%s", msg.data());                                \
        GELOGE(ge::FAILED, "%s", msg.data());                             \
      }                                                                   \
      return ::ErrorResult();                                             \
    }                                                                     \
  } while (false)

#define FE_CHK_RT_RET(expr)                                                   \
  do {                                                                        \
    const rtError_t _rt_ret = (expr);                                         \
    if (_rt_ret != RT_ERROR_NONE) {                                           \
      REPORT_FE_ERROR("Call %s fail, ret: 0x%X", #expr, _rt_ret);              \
      return RT_ERROR_TO_GE_STATUS(_rt_ret);                                  \
    }                                                                         \
  } while (false)

#define FE_ASSERT_NOTNULL(v, ...) FE_ASSERT(((v) != nullptr), __VA_ARGS__)
#define FE_ASSERT_SUCCESS(v, ...) FE_ASSERT(((v) == ge::SUCCESS), __VA_ARGS__)
#define FE_ASSERT_TRUE(v, ...) FE_ASSERT((v), __VA_ARGS__)
#define FE_ASSERT_RT_OK(v, ...) FE_ASSERT(((v) == 0), __VA_ARGS__)
#define FE_ASSERT_HYPER_SUCCESS(v, ...) FE_ASSERT(((v).IsSuccess()), __VA_ARGS__)
#define FE_ASSERT_GRAPH_SUCCESS(v, ...) FE_ASSERT(((v) == ge::GRAPH_SUCCESS), __VA_ARGS__)
#define FE_ASSERT_EOK(v, ...) FE_ASSERT(((v) == EOK), __VA_ARGS__)

#define FE_RETURN_IF_ERROR(expr)            \
  do {                                      \
    const ge::Status _chk_status = (expr);  \
    if (_chk_status != ge::SUCCESS) {       \
      return _chk_status;                   \
    }                                       \
  } while (false)

constexpr std::size_t MAX_DIM_NUM = 16;
constexpr std::size_t MAX_PRINT_SIZE = 36;

#endif  // AIR_CXX_RUNTIME_V2_ENGINE_FFTS_FE_RT2_COMMON_H_
