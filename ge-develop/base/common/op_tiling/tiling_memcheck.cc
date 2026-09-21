/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling_memcheck.h"
#include "framework/common/debug/ge_log.h"
#include "graph/args_format_desc.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/math_util.h"
#include "common/checker.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "exe_graph/runtime/tiling_data.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/types.h"

namespace optiling {
namespace {
constexpr size_t kMaxTilingDataSize = 16UL * 1024UL;
constexpr ge::char_t const *kMaxTilingSize = "op_para_size";

ge::graphStatus AppendInputSizeAndDfxShapeInfo(const ge::OpDescPtr &op_desc,
                                               std::vector<int64_t> &tiling_vec) {
  size_t index = 0UL;
  for (const auto &tensor : op_desc->GetAllInputsDescPtr()) {
    GE_ASSERT_NOTNULL(tensor);
    int64_t tensor_size = 0L;
    GE_ASSERT_SUCCESS(ge::TensorUtils::GetSize(*tensor, tensor_size));
    GELOGI("Append input tensor size, node[%s], index: %zu, tensor size: %lld",
        op_desc->GetNamePtr(), index, tensor_size);
    (void)tiling_vec.emplace_back(tensor_size);
    index++;
  }
  return ge::SUCCESS;
}

ge::graphStatus AppendOutputSizeAndDfxShapeInfo(const ge::OpDescPtr &op_desc,
                                                std::vector<int64_t> &tiling_vec) {
  for (size_t i = 0UL; i < op_desc->GetOutputsSize(); i++) {
    const auto tensor = op_desc->GetOutputDesc(static_cast<uint32_t>(i));
    int64_t tensor_size = 0L;
    GE_ASSERT_SUCCESS(ge::TensorUtils::GetSize(tensor, tensor_size));
    GELOGI("Append output tensor size, node[%s], index: %zu, tensor size: %lld",
        op_desc->GetNamePtr(), i, tensor_size);
    (void)tiling_vec.emplace_back(tensor_size);
  }
  return ge::SUCCESS;
}

ge::graphStatus AppendInputSizeAndDfxShapeInfo(const ge::OpDescPtr &op_desc,
    const std::vector<ge::ArgDesc> &arg_descs, std::vector<int64_t> &tiling_vec) {
  std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
  (void)(ge::OpDescUtils::GetIrInputInstanceDescRange(op_desc, ir_input_2_range));
  size_t index = 0UL;
  auto input_descs = op_desc->GetAllInputsDescPtr();
  for (const auto &arg_desc : arg_descs) {
    switch(arg_desc.addr_type) {
      case ge::AddrType::INPUT_DESC: {
        size_t arg_size = 0UL;
        const auto &ir_range = ir_input_2_range[static_cast<size_t>(arg_desc.ir_idx)];
        GE_ASSERT_GRAPH_SUCCESS(ge::ArgsFormatDesc::GetArgSize(op_desc, arg_desc, arg_size));
        GELOGI("Append input args size, node[%s], ir_idx:%d, arg_size:%zu input num:%zu",
            op_desc->GetNamePtr(), arg_desc.ir_idx, arg_size, ir_range.second);
        GE_ASSERT_TRUE(arg_size > 0UL);
        (void)tiling_vec.emplace_back(static_cast<int64_t>(arg_size));
        index += ir_range.second;
        break;
      }
      case ge::AddrType::INPUT: {
        const auto &ir_range = ir_input_2_range[static_cast<size_t>(arg_desc.ir_idx)];
        if (ir_range.second == 0UL) {
          GELOGI("Append empty optional input tensor size, node[%s], ir_idx:%d",
              op_desc->GetNamePtr(), arg_desc.ir_idx);
          (void)tiling_vec.emplace_back(0L);
          continue;
        }
        const auto tensor = input_descs.at(index);
        GE_ASSERT_NOTNULL(tensor);
        int64_t tensor_size = 0L;
        GE_ASSERT_SUCCESS(ge::TensorUtils::GetSize(*tensor, tensor_size));
        GELOGI("Append input tensor size, node[%s], ir_idx:%d, index: %zu, tensor size: %lld",
            op_desc->GetNamePtr(), arg_desc.ir_idx, index, tensor_size);
        (void)tiling_vec.emplace_back(tensor_size);
        index++;
        break;
      }
      case ge::AddrType::INPUT_INSTANCE: {
        const size_t instance_index = static_cast<size_t>(arg_desc.ir_idx);
        GE_ASSERT_TRUE(instance_index < input_descs.size());
        const auto tensor = input_descs.at(instance_index);
        GE_ASSERT_NOTNULL(tensor);
        int64_t tensor_size = 0L;
        GE_ASSERT_SUCCESS(ge::TensorUtils::GetSize(*tensor, tensor_size));
        GELOGI("Append input tensor size, node[%s], instance_idx:%d, tensor size: %lld", op_desc->GetNamePtr(),
               arg_desc.ir_idx, tensor_size);
        (void)tiling_vec.emplace_back(tensor_size);
        break;
      }
      case ge::AddrType::HIDDEN_INPUT: {
        constexpr const int64_t hidden_input_size = 32L; // HiddenInput固定拼入32字节跳过校验
        GELOGI("Append hidden input size, node[%s], size: %lld", op_desc->GetNamePtr(), hidden_input_size);
        (void)tiling_vec.emplace_back(hidden_input_size);
        break;
      }
      default: // workspace tiling等无需处理
        break;
    }
  }
  return ge::SUCCESS;
}

ge::graphStatus AppendOutputSizeAndDfxShapeInfo(
    const ge::OpDescPtr &op_desc, const std::vector<ge::ArgDesc> &arg_descs,
    std::vector<int64_t> &tiling_vec) {
  // tensor size
  std::map<size_t, std::pair<size_t, size_t>> ir_output_2_range;
  (void)(ge::OpDescUtils::GetIrOutputDescRange(op_desc, ir_output_2_range));
  size_t index = 0UL;
  for (const auto &arg_desc : arg_descs) {
    switch(arg_desc.addr_type) {
      case ge::AddrType::OUTPUT_DESC: {
        size_t arg_size = 0UL;
        const auto &ir_range = ir_output_2_range[static_cast<size_t>(arg_desc.ir_idx)];
        GE_ASSERT_GRAPH_SUCCESS(ge::ArgsFormatDesc::GetArgSize(op_desc, arg_desc, arg_size));
        GELOGI("Append output args size, node[%s], ir_idx:%d, arg_size:%zu output num:%zu",
              op_desc->GetNamePtr(), arg_desc.ir_idx, arg_size, ir_range.second);
        GE_ASSERT_TRUE(arg_size > 0UL);
        (void)tiling_vec.emplace_back(static_cast<int64_t>(arg_size));
        index += ir_range.second;
        break;
      }
      case ge::AddrType::OUTPUT: {
        const auto tensor = op_desc->GetOutputDesc(static_cast<uint32_t>(index));
        int64_t tensor_size = 0L;
        GE_ASSERT_SUCCESS(ge::TensorUtils::GetSize(tensor, tensor_size));
        GELOGI("Append output tensor size, node[%s], ir_idx:%d, index: %zu, tensor size: %lld",
            op_desc->GetNamePtr(), arg_desc.ir_idx, index, tensor_size);
        (void)tiling_vec.emplace_back(tensor_size);
        index++;
        break;
      }
      case ge::AddrType::OUTPUT_INSTANCE: {
        const auto tensor = op_desc->GetOutputDesc(static_cast<uint32_t>(arg_desc.ir_idx));
        int64_t tensor_size = 0L;
        GE_ASSERT_SUCCESS(ge::TensorUtils::GetSize(tensor, tensor_size));
        GELOGI("Append output tensor size, node[%s], instance_idx:%d, tensor size: %lld", op_desc->GetNamePtr(),
               arg_desc.ir_idx, tensor_size);
        (void)tiling_vec.emplace_back(tensor_size);
        break;
      }
      default:  // workspace tiling等无需处理
        break;
    }
  }
  return ge::SUCCESS;
}

ge::graphStatus GetMemCheckStartSize(const ge::OpDescPtr &op_desc,
                                     const int64_t origin_tiling_data_size,
                                     int64_t &memcheck_start_size) {
  int64_t ori_param_size = 0LL;
  (void)ge::AttrUtils::GetInt(op_desc, kOriOpParaSize, ori_param_size);
  if (ori_param_size > 0LL) {
    // tik场景下TilingAppendMem添加的数据需要从偏移为ori_param_size的地址开始添加，此处需要将DataSize设置成ori_param_size
    GE_ASSERT_TRUE(origin_tiling_data_size <= ori_param_size);
    GELOGI("Current tiling data size: %zu, set ori_para_size to %lld by attr, op_name: %s",
          origin_tiling_data_size, ori_param_size, op_desc->GetNamePtr());
  } else {
    ori_param_size = ((origin_tiling_data_size + sizeof(int64_t) - 1UL) / sizeof(int64_t)) * sizeof(int64_t);
    GELOGI("Current tiling data size: %zu, set ori_param_size to %lld by aligned by %zu, op_name: %s",
          origin_tiling_data_size, ori_param_size, sizeof(int64_t), op_desc->GetNamePtr());
  }
  memcheck_start_size = ori_param_size - origin_tiling_data_size;
  return ge::SUCCESS;
}
}

ge::graphStatus TilingMemCheck::ConstructMemCheckInfo(const ge::OpDescPtr &op_desc,
                                                      const OpRunInfoV2 &run_info,
                                                      const std::vector<ge::ArgDesc> &arg_descs,
                                                      std::string &memcheck_info) {
  GE_ASSERT_NOTNULL(op_desc);
  bool value = false;
  if ((!ge::AttrUtils::GetBool(op_desc, kMemoryCheck, value)) || (!value)) {
    GELOGI("Memcheck is not enable, op name: %s", op_desc->GetNamePtr());
    return ge::SUCCESS;
  }
  const int64_t tiling_data_size = static_cast<int64_t>(run_info.GetAllTilingData().str().size());
  if (tiling_data_size == 0) {
    return ge::SUCCESS;
  }
  // tiling
  int64_t max_size = -1;
  if (!ge::AttrUtils::GetInt(op_desc, kMaxTilingSize, max_size) || max_size < 0) {
    GELOGI("No max tiling size in opdesc.");
    max_size = static_cast<int64_t>(kMaxTilingDataSize);
  }
  const auto memcheck_info_capcity =
      ge::RoundUp(static_cast<uint64_t>(max_size), sizeof(uintptr_t));
  GELOGI("Get memcheck info capcity: %zu, op_name: %s", memcheck_info_capcity, op_desc->GetNamePtr());
  const auto memcheck_data_holder = gert::TilingData::CreateCap(memcheck_info_capcity);
  auto memcheck_data = reinterpret_cast<gert::TilingData *>(memcheck_data_holder.get());
  int64_t memcheck_start_size = 0L;
  GE_ASSERT_SUCCESS(GetMemCheckStartSize(op_desc, tiling_data_size, memcheck_start_size));
  memcheck_data->SetDataSize(static_cast<size_t>(memcheck_start_size));
  std::vector<int64_t> tiling_vec;
  if (!arg_descs.empty()) {
    GELOGI("Get memcheck info from args format, node: %s", op_desc->GetNamePtr());
    GE_ASSERT_SUCCESS(AppendInputSizeAndDfxShapeInfo(op_desc, arg_descs, tiling_vec));
    GE_ASSERT_SUCCESS(AppendOutputSizeAndDfxShapeInfo(op_desc, arg_descs, tiling_vec));
  } else {
    GELOGI("Get memcheck info from graph ir, node: %s", op_desc->GetNamePtr());
    GE_ASSERT_SUCCESS(AppendInputSizeAndDfxShapeInfo(op_desc, tiling_vec));
    GE_ASSERT_SUCCESS(AppendOutputSizeAndDfxShapeInfo(op_desc, tiling_vec));
  }

  (void)tiling_vec.insert(tiling_vec.cend(),
      run_info.GetAllWorkspaces().cbegin(), run_info.GetAllWorkspaces().cend());
  // append tensor size
  if (!tiling_vec.empty()) {
    GE_ASSERT_SUCCESS(memcheck_data->Append(tiling_vec.data(), tiling_vec.size()));
  }
  // append total size
  const int64_t current_data_size = static_cast<int64_t>(memcheck_data->GetDataSize());
  GE_ASSERT_SUCCESS(memcheck_data->Append(current_data_size + tiling_data_size));
  GELOGI("Op name[%s] memcheck info size: %lld", op_desc->GetNamePtr(), memcheck_data->GetDataSize());
  memcheck_info = std::string(reinterpret_cast<ge::char_t *>(memcheck_data->GetData()), memcheck_data->GetDataSize());
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling