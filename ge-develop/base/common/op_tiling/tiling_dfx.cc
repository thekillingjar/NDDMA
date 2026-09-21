/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling_dfx.h"
#include "framework/common/debug/ge_log.h"
#include "graph/args_format_desc.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/math_util.h"
#include "common/checker.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "exe_graph/runtime/tiling_data.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {

ge::Status TilingDfx::GetArgsSizeWithArgsFormat(const ge::OpDescPtr &op_desc,
                                                const std::vector<ge::ArgDesc> &arg_descs,
                                                std::vector<int64_t> &args_size_list,
                                                std::vector<ArgsIndexToIoIndex> &args_index_to_io_index) {
  std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
  (void)(ge::OpDescUtils::GetIrInputInstanceDescRange(op_desc, ir_input_2_range));
  std::map<size_t, std::pair<size_t, size_t>> ir_output_2_range;
  (void)(ge::OpDescUtils::GetIrOutputDescRange(op_desc, ir_output_2_range));
  for (size_t idx = 0U; idx < arg_descs.size(); idx++) {
    switch (arg_descs[idx].addr_type) {
      case ge::AddrType::INPUT_DESC: {
        const size_t ir_idx = static_cast<size_t>(arg_descs[idx].ir_idx);
        const auto iter = ir_input_2_range.find(ir_idx);
        GE_ASSERT(iter != ir_input_2_range.end(), "input ir idx [%zu] is not found", ir_idx);
        size_t arg_size = 0UL;
        GE_ASSERT_GRAPH_SUCCESS(ge::ArgsFormatDesc::GetArgSize(op_desc, arg_descs[idx], arg_size)); // max dim的大小是25还是16
        GE_ASSERT_TRUE(arg_size > 0UL);
        (void)args_size_list.emplace_back(static_cast<int64_t>(arg_size));
        break;
      }
      case ge::AddrType::OUTPUT_DESC: {
        const size_t ir_idx = static_cast<size_t>(arg_descs[idx].ir_idx);
        const auto iter = ir_output_2_range.find(ir_idx);
        GE_ASSERT(iter != ir_output_2_range.end(), "output ir idx [%zu] is not found", ir_idx);
        size_t arg_size = 0UL;
        GE_ASSERT_GRAPH_SUCCESS(ge::ArgsFormatDesc::GetArgSize(op_desc, arg_descs[idx], arg_size));
        GE_ASSERT_TRUE(arg_size > 0UL);
        (void)args_size_list.emplace_back(static_cast<int64_t>(arg_size));
        break;
      }
      case ge::AddrType::INPUT_INSTANCE: {
        const size_t instance_index = static_cast<size_t>(arg_descs[idx].ir_idx);
        ArgsIndexToIoIndex args_idx_to_io_idx = {ArgsRole::kInput, args_size_list.size(), instance_index};
        (void)args_index_to_io_index.emplace_back(std::move(args_idx_to_io_idx));
        (void)args_size_list.emplace_back(0);
        break;
      }
      case ge::AddrType::OUTPUT_INSTANCE: {
        const size_t instance_index = static_cast<size_t>(arg_descs[idx].ir_idx);
        ArgsIndexToIoIndex args_idx_to_io_idx = {ArgsRole::kOutput, args_size_list.size(), instance_index};
        (void)args_index_to_io_index.emplace_back(std::move(args_idx_to_io_idx));
        (void)args_size_list.emplace_back(0);
        break;
      }
      case ge::AddrType::INPUT: {
        const size_t ir_idx = static_cast<size_t>(arg_descs[idx].ir_idx);
        const auto iter = ir_input_2_range.find(ir_idx);
        GE_ASSERT(iter != ir_input_2_range.end(), "input ir idx [%zu] is not found", ir_idx);
        const auto &range_pair = iter->second;
        if (range_pair.second == 0UL) {
          // optional input placeholder
          (void)args_size_list.emplace_back(0);
          break;
        }
        size_t begin_idx = range_pair.first;
        size_t loop_times = range_pair.second;
        while (loop_times-- > 0UL) {
          ArgsIndexToIoIndex args_idx_to_io_idx = {ArgsRole::kInput, args_size_list.size(), begin_idx};
          (void)args_index_to_io_index.emplace_back(std::move(args_idx_to_io_idx));
          (void)args_size_list.emplace_back(0);
          ++begin_idx;
        }
        break;
      }
      case ge::AddrType::OUTPUT: {
        const size_t ir_idx = static_cast<size_t>(arg_descs[idx].ir_idx);
        const auto iter = ir_output_2_range.find(ir_idx);
        GE_ASSERT(iter != ir_output_2_range.end(), "output ir idx [%zu] is not found", ir_idx);

        const auto &range_pair = iter->second;
        size_t begin_idx = range_pair.first;
        size_t loop_times = range_pair.second;
        while (loop_times-- > 0UL) {
          ArgsIndexToIoIndex args_idx_to_io_idx = {ArgsRole::kOutput, args_size_list.size(), begin_idx};
          (void)args_index_to_io_index.emplace_back(std::move(args_idx_to_io_idx));
          (void)args_size_list.emplace_back(0);
          ++begin_idx;
        }
        break;
      }
      case ge::AddrType::FFTS_ADDR: // 不占位
        break;
      case ge::AddrType::HIDDEN_INPUT:
      case ge::AddrType::PLACEHOLDER:
        (void)args_size_list.emplace_back(0); // 占位
        break;
      default:
        // iow之后的地址格式不再解析：TILING,OVERFLOW_ADDR,TILING_FFTS,TILING_CONTEXT
        // workspace num在lowering阶段无法确定，动静态的workspace size在io后统一添加,worksapce直接跳过：WORKSPACE
        // aicpu流程不涉及：CUSTOM_VALUE,OP_TYPE
        return ge::SUCCESS;
    }
  }

  return ge::SUCCESS;
}

ge::Status TilingDfx::GetArgsSizeWithoutArgsFormat(size_t input_size,
                                                   size_t output_size,
                                                   std::vector<int64_t> &args_size_list,
                                                   std::vector<ArgsIndexToIoIndex> &args_index_to_io_index) {
  size_t io_size = 0U;
  GE_ASSERT_TRUE(!ge::AddOverflow(input_size, output_size, io_size));
  (void) args_size_list.insert(args_size_list.cend(), io_size, 0);

  for (size_t index = 0U; index < input_size; index++) {
    ArgsIndexToIoIndex args_idx_to_io_idx = {ArgsRole::kInput, index, index};
    (void)args_index_to_io_index.emplace_back(std::move(args_idx_to_io_idx));
  }

  for (size_t index = 0U; index < output_size; index++) {
    ArgsIndexToIoIndex args_idx_to_io_idx = {ArgsRole::kOutput, input_size + index, index};
    (void)args_index_to_io_index.emplace_back(std::move(args_idx_to_io_idx));
  }

  return ge::SUCCESS;
}

}  // namespace optiling