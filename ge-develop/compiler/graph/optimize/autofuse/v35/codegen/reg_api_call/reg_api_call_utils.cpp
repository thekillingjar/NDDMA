/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reg_api_call_utils.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace {
constexpr size_t kDmaMaxLen = 2U;
constexpr size_t kFourAxisNum = 4U;
constexpr size_t kFiveAxisNum = 5U;
constexpr char kNormalPddingMode[] = "AscendC::PaddingMode::Normal";
constexpr char kCompactPddingMode[] = "AscendC::PaddingMode::Compact";
}  // namespace

namespace codegen {
// A5场景，拷贝指令增强，可以通过loop_mode_params消掉另外两层for循环
void SetLoopModeParams(const TPipe &tpipe, const DataCopyParams &data_copy_param, LoopModeParams &loop_mode_param,
                       bool copy_in) {
  int64_t total_len = data_copy_param.repeats.size();
  if (total_len <= static_cast<int64_t>(kDmaMaxLen)) {
    return;
  }

  int64_t loop_idx = 0;
  for (int64_t idx = total_len - 1 - kDmaMaxLen; idx >= 0; idx--) {
    if (loop_idx > 1) {
      break;
    }
    loop_mode_param.loop_size[loop_idx] = tpipe.tiler.ActualSize(data_copy_param.repeats[idx]);
    loop_mode_param.loop_src_stride[loop_idx] =
        copy_in ? tpipe.tiler.Size(data_copy_param.gm_strides[idx]) : tpipe.tiler.Size(data_copy_param.ub_strides[idx]);
    loop_mode_param.loop_dst_stride[loop_idx] =
        copy_in ? tpipe.tiler.Size(data_copy_param.ub_strides[idx]) : tpipe.tiler.Size(data_copy_param.gm_strides[idx]);
    loop_idx++;
  }
}

// 根据ub上最后一维的对齐信息以及切分轴信息判断，是否使用Compact模式，如果能明确判断出来，stride与repeat相同且ub切分轴是首轴，则使用compact模式，否则使用normal模式
std::string GetPaddingMode(const TPipe &tpipe, const Tensor &ub_tensor, const DataCopyParams &data_copy_param) {
  for (auto axis_pos : ub_tensor.vectorized_axis_pos) {
    ascir::AxisId axis_id = ub_tensor.axis[axis_pos];
    const Axis &axis = tpipe.tiler.GetAxis(axis_id);
    if (axis.type == ascir::Axis::Type::kAxisTypeTileInner && ub_tensor.vectorized_axis[0] != axis_id) {
      GELOGD("The TileInner axis is not the first axis， use normal mode.");
      return kNormalPddingMode;
    }
  }
  if (data_copy_param.repeats.size() <= 1) {
    return kNormalPddingMode;
  }
  ascir::SizeExpr repeat = data_copy_param.repeats.back();
  ascir::SizeExpr stride = data_copy_param.ub_strides[data_copy_param.ub_strides.size() - kDmaMaxLen];
  bool status = ge::SymbolicUtils::StaticCheckEq(repeat, stride) == ge::TriBool::kTrue;
  return status ? kCompactPddingMode : kNormalPddingMode;
}

std::string GenLoopModeParams(const LoopModeParams &loop_mode_param, int64_t input_dtype_size,
                              int64_t output_dtype_size) {
  std::stringstream ss;
  ss << "{static_cast<uint32_t>(" << loop_mode_param.loop_size[0] << "), static_cast<uint32_t>("
     << loop_mode_param.loop_size[1] << "), static_cast<uint64_t>(" << loop_mode_param.loop_src_stride[0] << " * "
     << input_dtype_size << "), static_cast<uint64_t>(" << loop_mode_param.loop_dst_stride[0] << " * "
     << output_dtype_size << "), static_cast<uint64_t>(" << loop_mode_param.loop_src_stride[1] << " * "
     << input_dtype_size << "), static_cast<uint64_t>(" << loop_mode_param.loop_dst_stride[1] << " * "
     << output_dtype_size << ")}";
  return ss.str();
}

void CreateBaseDmaCall(const Tensor &input, const Tensor &output, const DmaParams &dma_param,
                       std::string &padding_mode, std::stringstream &ss, bool copy_in) {
  std::string dtype_name;
  Tensor::DtypeName(input.dtype, dtype_name);
  ss << "DataCopyPadExtend<" << dtype_name << ", " << padding_mode << ">(";
  if (copy_in) {
    ss << output << ", " << input << "[" << dma_param.gm_offset << "], ";
  } else {
    ss << output << "[" << dma_param.gm_offset << "], " << input << ", ";
  }
  ss << dma_param.block_count << ", " << dma_param.block_len << ", " << dma_param.src_stride << ", "
     << dma_param.dst_stride << ");" << std::endl;
}

void CreateBaseEnhanceDmaCall(const Tensor &input, const Tensor &output, const DmaParams &dma_param,
                              std::string padding_mode, const LoopModeParams &loop_mode_param, std::stringstream &ss,
                              bool copy_in) {
  std::string dtype_name;
  Tensor::DtypeName(input.dtype, dtype_name);
  ss << "DataCopyPadExtend<" << dtype_name << ", " << padding_mode << ">(";
  if (copy_in) {
    ss << output << "[" << dma_param.ub_offset << "], " << input << "[" << dma_param.gm_offset << "], ";
  } else {
    ss << output << "[" << dma_param.gm_offset << "], " << input << "[" << dma_param.ub_offset << "], ";
  }
  ss << dma_param.block_count << ", " << dma_param.block_len << ", " << dma_param.src_stride << ", "
     << dma_param.dst_stride << ", "
     << GenLoopModeParams(loop_mode_param, ge::GetSizeByDataType(input.dtype), ge::GetSizeByDataType(output.dtype))
     << ");" << std::endl;
}

void CreateEnhanceDmaCall(const TPipe &tpipe, const Tensor &input, const Tensor &output, const string &gm_offset,
                          const DataCopyParams &data_copy_param, const ascir::SizeExpr &offset, std::stringstream &ss,
                          bool copy_in) {
  size_t total_len = data_copy_param.repeats.size();
  DmaParams dma_param;
  SetDmaParams(tpipe, data_copy_param, dma_param, copy_in);
  dma_param.gm_offset = gm_offset + " + " + tpipe.tiler.Size(offset);
  LoopModeParams loop_mode_param;
  SetLoopModeParams(tpipe, data_copy_param, loop_mode_param, copy_in);
  const Tensor &ub_tensor = copy_in ? output : input;
  std::string padding_mode = GetPaddingMode(tpipe, ub_tensor, data_copy_param);
  if (total_len <= kDmaMaxLen) {
    CreateBaseDmaCall(input, output, dma_param, padding_mode, ss, copy_in);
    return;
  }

  if (total_len > kDmaMaxLen && total_len <= kFourAxisNum) {
    CreateBaseEnhanceDmaCall(input, output, dma_param, padding_mode, loop_mode_param, ss, copy_in);
    return;
  }

  // 超过四层for循环，需要外抛
  std::vector<ascir::SizeExpr> gm_stride(data_copy_param.gm_strides.begin(),
                                         data_copy_param.gm_strides.end() - kFourAxisNum);
  std::vector<ascir::SizeExpr> ub_stride(data_copy_param.ub_strides.begin(),
                                         data_copy_param.ub_strides.end() - kFourAxisNum);
  std::vector<ascir::SizeExpr> repeats(data_copy_param.repeats.begin(), data_copy_param.repeats.end() - kFourAxisNum);
  std::string gm_inner_offset = CalcInnerOffset(tpipe, gm_stride);
  std::string ub_inner_offset = CalcInnerOffset(tpipe, ub_stride);
  std::stringstream ss1;
  dma_param.gm_offset = gm_offset + " + " + tpipe.tiler.Size(offset) + " + " + gm_inner_offset;
  dma_param.ub_offset = ub_inner_offset;
  CreateBaseEnhanceDmaCall(input, output, dma_param, padding_mode, loop_mode_param, ss1, copy_in);
  CreateOuterFor(tpipe, repeats, ss1, ss, 0);
}

void SetNddmaParams(const TPipe &tpipe, const DataCopyParams &data_copy_param, NddmaParams &nddma_param,
                    const int64_t &tensor_id, std::stringstream &ss) {
  nddma_param.ss_output_dims << "const int64_t output_dims_" << tensor_id << "[5] = {";
  nddma_param.ss_output_stride << "const int64_t output_stride_" << tensor_id << "[5] = {";
  nddma_param.ss_input_stride << "const int64_t input_stride_" << tensor_id << "[5] = {";
  for (size_t i = 0UL;
       data_copy_param.repeats.size() < kFiveAxisNum && i < kFiveAxisNum - data_copy_param.repeats.size(); i++) {
    if (data_copy_param.repeats.empty() && i == kFiveAxisNum - 1) {
      nddma_param.ss_output_dims << "1";
      nddma_param.ss_output_stride << "1";
      nddma_param.ss_input_stride << "1";
      break;
    }
    nddma_param.ss_output_dims << "1, ";
    nddma_param.ss_output_stride << "1, ";
    nddma_param.ss_input_stride << "1, ";
  }
  size_t i = data_copy_param.repeats.size() > kFiveAxisNum ? data_copy_param.repeats.size() - kFiveAxisNum : 0UL;
  size_t j = std::min(data_copy_param.repeats.size(), kFiveAxisNum);
  while (j > 0UL) {
    if (j == 1UL) {
      nddma_param.ss_output_dims << data_copy_param.repeats_str[i];
      nddma_param.ss_output_stride << tpipe.tiler.Size(data_copy_param.ub_strides[i]);
      nddma_param.ss_input_stride << tpipe.tiler.Size(data_copy_param.gm_strides[i]);
      break;
    }
    nddma_param.ss_output_dims << data_copy_param.repeats_str[i] << ", ";
    nddma_param.ss_output_stride << tpipe.tiler.Size(data_copy_param.ub_strides[i]) << ", ";
    nddma_param.ss_input_stride << tpipe.tiler.Size(data_copy_param.gm_strides[i]) << ", ";
    i++;
    j--;
  }
  nddma_param.ss_output_dims << "};" << std::endl;
  nddma_param.ss_output_stride << "};" << std::endl;
  nddma_param.ss_input_stride << "};" << std::endl;
  ss << nddma_param.ss_output_dims.str() << nddma_param.ss_output_stride.str() << nddma_param.ss_input_stride.str();
}

void CreateNddmaCall(const TPipe &tpipe, const Tensor &input, const Tensor &output, const string &gm_offset,
                     const DataCopyParams &data_copy_param, const ascir::SizeExpr &offset, std::stringstream &ss) {
  const std::vector<ascir::SizeExpr> gm_stride(data_copy_param.gm_strides.begin(),
                                               data_copy_param.gm_strides.end() - kFiveAxisNum);
  const std::vector<ascir::SizeExpr> ub_stride(data_copy_param.ub_strides.begin(),
                                               data_copy_param.ub_strides.end() - kFiveAxisNum);
  const std::vector<ascir::SizeExpr> repeats(data_copy_param.repeats.begin(),
                                             data_copy_param.repeats.end() - kFiveAxisNum);
  const std::string gm_inner_offset = CalcInnerOffset(tpipe, gm_stride);
  const std::string ub_inner_offset = CalcInnerOffset(tpipe, ub_stride);
  std::stringstream ss1;
  NddmaParams nddma_param;
  SetNddmaParams(tpipe, data_copy_param, nddma_param, output.id, ss);
  ss1 << "DataCopyNddma(" << output << "[" << ub_inner_offset << "], " << input << "[" << gm_offset << " + "
      << tpipe.tiler.Size(offset) << " + " << gm_inner_offset << "], "
      << "output_dims_" << output.id << ", " << "output_stride_" << output.id << ", " << "input_stride_" << output.id
      << ");" << std::endl;
  CreateOuterFor(tpipe, repeats, ss1, ss, 0UL);
}

}  // namespace codegen