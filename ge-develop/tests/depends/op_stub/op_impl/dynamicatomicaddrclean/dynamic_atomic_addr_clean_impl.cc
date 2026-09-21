/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dynamic_atomic_addr_clean_impl.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/storage_shape.h"
#include "framework/common/debug/ge_log.h"
#include <nlohmann/json.hpp>
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/atomic_clean_tiling_context.h"

using namespace ge;

namespace gert {
constexpr uint32_t MIN_ELE_SIZE_USING_ALL_CORE = 1024;
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t BYTE_FP32 = 4;
constexpr uint32_t MASK_FP32 = 64;
constexpr uint32_t MAX_REPEAT_TIME = 255;

#define CHECK(cond, log_func, expr)                                                                                    \
  do {                                                                                                                 \
    if (cond) {                                                                                                        \
      log_func;                                                                                                        \
      expr;                                                                                                            \
    }                                                                                                                  \
  } while (0)

ge::graphStatus InferShapeForDynamicAtomicAddrClean(InferShapeContext *context) {
  return ge::GRAPH_SUCCESS;
}

int32_t CeilDiv(const int32_t &num, const int32_t &factor) {
  int32_t res = (num % factor == 0) ? num / factor : num / factor + 1;
  return res;
}

void ComputeParamsOneCore(const int32_t &ele_num_one_core,
                          const int32_t &ele_num_full_mask_full_repeat_time_input_scalar,
                          int32_t &init_times_full_mask_full_repeat_time_input_scalar,
                          int32_t &ele_num_front_part_input_scalar, int32_t &burst_len_last_part_input_scalar,
                          int32_t &repeat_time_last_part_input_scalar) {
  init_times_full_mask_full_repeat_time_input_scalar =
      ele_num_one_core / ele_num_full_mask_full_repeat_time_input_scalar;
  ele_num_front_part_input_scalar =
      init_times_full_mask_full_repeat_time_input_scalar * ele_num_full_mask_full_repeat_time_input_scalar;
  uint32_t ele_num_last_part = ele_num_one_core - ele_num_front_part_input_scalar;
  burst_len_last_part_input_scalar = CeilDiv(ele_num_last_part * BYTE_FP32, BYTE_BLOCK);
  if (ele_num_last_part % MASK_FP32 == 0) {
    repeat_time_last_part_input_scalar = ele_num_last_part / MASK_FP32;
  } else {
    repeat_time_last_part_input_scalar = ele_num_last_part / MASK_FP32 + 1;
  }
}

void InitTilingParams(DynamicAtomicAddrCleanParam &params) {
  params.select_key_input_scalar = 0;
  params.need_core_num_input_scalar = 0;
  params.ele_num_full_mask_full_repeat_time_input_scalar = 0;
  params.burst_len_full_mask_full_repeat_time_input_scalar = 0;

  // init input scalar
  // front core
  params.ele_num_front_core_input_scalar = 0;
  params.init_times_full_mask_full_repeat_time_front_core_input_scalar = 0;
  params.ele_num_front_part_front_core_input_scalar = 0;
  params.burst_len_last_part_front_core_input_scalar = 0;
  params.repeat_time_last_part_front_core_input_scalar = 0;
  // last core
  params.ele_num_last_core_input_scalar = 0;
  params.init_times_full_mask_full_repeat_time_last_core_input_scalar = 0;
  params.ele_num_front_part_last_core_input_scalar = 0;
  params.burst_len_last_part_last_core_input_scalar = 0;
  params.repeat_time_last_part_last_core_input_scalar = 0;
}

bool CheckSize(const int64_t &size) {
  if (size < 0) {
    GELOGE(ge::GRAPH_FAILED, "op: workspace size must be greater than 0!");
    return false;
  }
  if (size % 32 != 0) {
    GELOGE(ge::GRAPH_FAILED, "op : workspace size must be able to be divided by 32!");
    return false;
  }
  return true;
}

ge::graphStatus WriteAtomicTilingData(TilingContext *context, DynamicAtomicAddrCleanParam &params, uint64_t tensor_size,
                                      uint32_t core_num) {
  int64_t addr_tensor_size = static_cast<int64_t>(tensor_size);
  bool flag = CheckSize(addr_tensor_size);
  if (!flag) {
    GELOGE(ge::GRAPH_FAILED, "op: CheckSize %ld error!", addr_tensor_size);
    return ge::GRAPH_FAILED;
  }
  uint32_t ele_num_fp32 = addr_tensor_size / BYTE_FP32;
  // init tiling params
  InitTilingParams(params);
  params.select_key_input_scalar = 1;
  // is using all core
  if (addr_tensor_size >= MIN_ELE_SIZE_USING_ALL_CORE) {
    params.need_core_num_input_scalar = core_num;
  } else {
    params.need_core_num_input_scalar = 1;
  }
  // compute tiling params
  params.ele_num_full_mask_full_repeat_time_input_scalar = MASK_FP32 * MAX_REPEAT_TIME;
  params.burst_len_full_mask_full_repeat_time_input_scalar =
      params.ele_num_full_mask_full_repeat_time_input_scalar * BYTE_FP32 / BYTE_BLOCK;
  if (params.need_core_num_input_scalar == 1) {
    // use one core
    params.ele_num_front_core_input_scalar = ele_num_fp32;
    ComputeParamsOneCore(params.ele_num_front_core_input_scalar, params.ele_num_full_mask_full_repeat_time_input_scalar,
                         params.init_times_full_mask_full_repeat_time_front_core_input_scalar,
                         params.ele_num_front_part_front_core_input_scalar,
                         params.burst_len_last_part_front_core_input_scalar,
                         params.repeat_time_last_part_front_core_input_scalar);

    params.ele_num_last_core_input_scalar = params.ele_num_front_core_input_scalar;
    ComputeParamsOneCore(params.ele_num_last_core_input_scalar, params.ele_num_full_mask_full_repeat_time_input_scalar,
                         params.init_times_full_mask_full_repeat_time_last_core_input_scalar,
                         params.ele_num_front_part_last_core_input_scalar,
                         params.burst_len_last_part_last_core_input_scalar,
                         params.repeat_time_last_part_last_core_input_scalar);
  } else if (params.need_core_num_input_scalar > 1) {
    // use all core
    // front core
    params.ele_num_front_core_input_scalar = ele_num_fp32 / params.need_core_num_input_scalar;
    ComputeParamsOneCore(params.ele_num_front_core_input_scalar, params.ele_num_full_mask_full_repeat_time_input_scalar,
                         params.init_times_full_mask_full_repeat_time_front_core_input_scalar,
                         params.ele_num_front_part_front_core_input_scalar,
                         params.burst_len_last_part_front_core_input_scalar,
                         params.repeat_time_last_part_front_core_input_scalar);
    // last core
    params.ele_num_last_core_input_scalar =
        ele_num_fp32 - params.ele_num_front_core_input_scalar * (params.need_core_num_input_scalar - 1);
    ComputeParamsOneCore(params.ele_num_last_core_input_scalar, params.ele_num_full_mask_full_repeat_time_input_scalar,
                         params.init_times_full_mask_full_repeat_time_last_core_input_scalar,
                         params.ele_num_front_part_last_core_input_scalar,
                         params.burst_len_last_part_last_core_input_scalar,
                         params.repeat_time_last_part_last_core_input_scalar);
  }
  context->SetBlockDim(params.need_core_num_input_scalar);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForDynamicAtomicAddrClean(TilingContext *context) {
  auto compute_node_info = context->GetComputeNodeInfo();
  if (compute_node_info == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "op: compile_node_info nullptr!");
    return ge::GRAPH_FAILED;
  }
  auto compile_info = reinterpret_cast<const DynamicAtomicAddrCleanCompileInfo *>(context->GetCompileInfo());
  if (compile_info == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "op: compile_info nullptr!");
    return ge::GRAPH_FAILED;
  }
  uint32_t core_num = compile_info->core_num;
  const std::vector<int64_t> &workspace_idx = compile_info->_workspace_index_list;

  size_t clean_tensor_num = compute_node_info->GetInputsNum() - 1;
  size_t total_clean_num = clean_tensor_num + workspace_idx.size();

  TilingData *tiling_data = context->GetRawTilingData();
  if (tiling_data == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "op: tiling_data nullptr!");
    return ge::GRAPH_FAILED;
  }

  tiling_data->SetDataSize(sizeof(DynamicAtomicAddrCleanParam) * total_clean_num);
  auto param = reinterpret_cast<DynamicAtomicAddrCleanParam *>(tiling_data->GetData());
  auto atomic_clean_context = reinterpret_cast<AtomicCleanTilingContext *>(context);
  size_t idx = 0U;
  for (; idx < clean_tensor_num; ++idx) {
    auto tensor_size = atomic_clean_context->GetCleanOutputSize(idx);
    if (WriteAtomicTilingData(context, *(param + idx), tensor_size, core_num) != ge::GRAPH_SUCCESS) {
      GELOGE(ge::GRAPH_FAILED, "op: write atomic tiling data failed!");
      return ge::GRAPH_FAILED;
    }
  }

  if (!workspace_idx.empty()) {
    auto ws_sizes = atomic_clean_context->GetCleanWorkspaceSizes();
    if (ws_sizes == nullptr) {
      GELOGE(ge::GRAPH_FAILED, "op: ws_size nullptr!");
      return ge::GRAPH_FAILED;
    }
    auto ws_size_data = reinterpret_cast<const uint64_t *>(ws_sizes->GetData());
    for (size_t i = 0U; i < workspace_idx.size(); ++i, ++idx) {
      auto tensor_size = ws_size_data[workspace_idx[i]];
      if (WriteAtomicTilingData(context, *(param + idx), tensor_size, core_num) != ge::GRAPH_SUCCESS) {
        GELOGE(ge::GRAPH_FAILED, "op: write atomic tiling data failed!");
        return ge::GRAPH_FAILED;
      }
    }
  }

  return ge::GRAPH_SUCCESS;
}

template<typename T>
bool GetCompileValue(const nlohmann::json &all_vars, const char *name, T &value) {
  if (all_vars.count(name) == 0) {
    return false;
  }

  value = all_vars[name].get<T>();
  return true;
}

#define GET_COMPILE_VALUE(vars, compile_info, name) GetCompileValue(vars, #name, (*compile_info).name)

//TODO hardcode DynamicAtomicAddrClean
ge::graphStatus TilingPrepareForDynamicAtomicAddrClean(KernelContext *context) {
  auto compile_info = context->GetOutputPointer<DynamicAtomicAddrCleanCompileInfo>(0);
  auto json_str = context->GetInputStrPointer(0);
  if (compile_info == nullptr || json_str == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "op: compile_info or json_str nullptr!");
    return ge::GRAPH_FAILED;
  }
  std::unique_ptr<nlohmann::json> parsed_object_cinfo(new nlohmann::json(nlohmann::json::parse(json_str)));
  if (parsed_object_cinfo == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "op: parsed_object_cinfo nullptr!");
    return ge::GRAPH_FAILED;
  }
  GET_COMPILE_VALUE(*parsed_object_cinfo, compile_info, _workspace_index_list);
  const nlohmann::json &vars = (*parsed_object_cinfo)["vars"];
  if (vars.empty()) {
    GELOGE(ge::GRAPH_FAILED, "DynamicAtomicAddrClean3TilingParse: get vars failed.");
    return ge::GRAPH_FAILED;
  }
  GET_COMPILE_VALUE(vars, compile_info, ub_size);
  GET_COMPILE_VALUE(vars, compile_info, core_num);
  GET_COMPILE_VALUE(vars, compile_info, workspace_num);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(DynamicAtomicAddrClean)
    .InferShape(InferShapeForDynamicAtomicAddrClean)
    .Tiling(TilingForDynamicAtomicAddrClean)
    .TilingParse<DynamicAtomicAddrCleanCompileInfo>(TilingPrepareForDynamicAtomicAddrClean);
IMPL_OP(MemSet)
    .InferShape(InferShapeForDynamicAtomicAddrClean)
    .Tiling(TilingForDynamicAtomicAddrClean)
    .TilingParse<DynamicAtomicAddrCleanCompileInfo>(TilingPrepareForDynamicAtomicAddrClean);
} // namespace gert
