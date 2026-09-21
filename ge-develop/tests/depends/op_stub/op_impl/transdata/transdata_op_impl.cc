/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transdata_op_impl.h"
#include "graph/ge_error_codes.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"

#include "graph/utils/math_util.h"
#include "common/table_driven.h"
#include "trans_data_positive_source_tc_1010.h"
#include "trans_data_negative_target_tc_201.h"
#include "exe_graph/runtime/tiling_context.h"
#include "nlohmann/json.hpp"

using ge::CeilDiv;

namespace gert {
namespace kernel {
namespace {
constexpr int64_t BLOCK_BYTE_SIZE = 32;
constexpr int64_t NI_16 = 16;
constexpr int64_t C0_16 = 16;
constexpr int64_t C0_32 = 32;

int64_t GetC0SizeWithType(ge::DataType &dtype) {
  if (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8) {
    return C0_32;
  }
  return C0_16;
}
}  // namespace
namespace transdata {
ge::graphStatus ConvertShapeHNC2HCNT(const Shape &in_shape, const Shape &out_shape, int64_t c0_size,
                                     Shape &in_shape_new, Shape &out_shape_new) {
  int64_t axis_n;
  int64_t axis_h;
  int64_t axis_c;
  if (in_shape.GetDimNum() == 1) {
    axis_h = 1;
    axis_n = 1;
    axis_c = in_shape[0];
  } else if (in_shape.GetDimNum() == 2) {
    axis_h = 1;
    axis_n = in_shape[0];
    axis_c = in_shape[1];
  } else {
    int64_t shape_size = 1;
    for (size_t i = 0; i < in_shape.GetDimNum() - 2; i++) {
      shape_size *= in_shape[i];
    }
    axis_h = shape_size;
    axis_n = in_shape[in_shape.GetDimNum() - 2];
    axis_c = in_shape[in_shape.GetDimNum() - 1];
  }
  in_shape_new.AppendDim(axis_h);
  in_shape_new.AppendDim(axis_n);
  in_shape_new.AppendDim(axis_c);
  int64_t axis_c0 = c0_size;
  int64_t axis_c1 = CeilDiv(axis_c, axis_c0);
  int64_t axis_ni = 16;
  int64_t axis_no = CeilDiv(axis_n, axis_ni);
  out_shape_new.AppendDim(axis_h);
  out_shape_new.AppendDim(axis_c1);
  out_shape_new.AppendDim(axis_no * axis_ni);
  out_shape_new.AppendDim(axis_c0);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConvertShapeHCNT2HNC(const Shape &in_shape, const Shape &out_shape, int64_t c0_size,
                                     Shape &in_shape_new, Shape &out_shape_new) {

  int64_t axis_n;
  int64_t axis_h;
  int64_t axis_c;
  size_t out_shape_size = out_shape.GetDimNum();
  if (out_shape_size == 1) {
    axis_h = 1;
    axis_n = 1;
    axis_c = out_shape[0];
  } else if (out_shape_size == 2) {
    axis_h = 1;
    axis_n = out_shape[0];
    axis_c = out_shape[1];
  } else {
    int64_t shape_size = 1;
    for (size_t i = 0; i < out_shape_size - 2; i++) {
      shape_size *= out_shape[i];
    }
    axis_h = shape_size;
    axis_n = out_shape[out_shape_size - 2];
    axis_c = out_shape[out_shape_size - 1];
  }
  int64_t axis_c0 = in_shape[in_shape.GetDimNum() - 1];
  int64_t axis_c1 = CeilDiv(axis_c, axis_c0);
  int64_t axis_ni = 16;
  int64_t axis_no = CeilDiv(axis_n, axis_ni);
  in_shape_new.AppendDim(axis_h);
  in_shape_new.AppendDim(axis_c1);
  in_shape_new.AppendDim(axis_no * axis_ni);
  in_shape_new.AppendDim(axis_c0);
  out_shape_new.AppendDim(axis_h);
  out_shape_new.AppendDim(axis_n);
  out_shape_new.AppendDim(axis_c);
  return ge::GRAPH_SUCCESS;
}

RealShapeConvertFunc GetRealShapeConvertFunc(ge::Format src_format, ge::Format dst_format) {
  static auto table = TableDriven2<ge::FORMAT_END, ge::FORMAT_END, RealShapeConvertFunc>(nullptr)
                          .Add(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ConvertShapeHNC2HCNT)
                          .Add(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_NZ, ConvertShapeHNC2HCNT)
                          .Add(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ConvertShapeHCNT2HNC);
  return table.Find(src_format, dst_format);
}

struct RealSrcDstFormat {
  RealSrcDstFormat() = default;
  RealSrcDstFormat(const RealSrcDstFormat &other) = default;
  RealSrcDstFormat(RealFormat src, RealFormat dst) : src(src), dst(dst) {}
  RealFormat src;
  RealFormat dst;
};

const RealSrcDstFormat *GetRealFormat(ge::Format src_format, ge::Format dst_format) {
  static RealSrcDstFormat default_value = {RF_END, RF_END};
  static auto table = TableDriven2<ge::FORMAT_END, ge::FORMAT_END, RealSrcDstFormat>(default_value)
                          .Add(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, RF_HNC, RF_HCNT)
                          .Add(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_NZ, RF_HNC, RF_HCNT)
                          .Add(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, RF_HCNT, RF_HNC);

  return table.FindPointer(src_format, dst_format);
}
}  // namespace transdata

ge::graphStatus TilingForTransData(TilingContext *context) {
  auto in_shape = context->GetInputShape(0);
  auto out_shape = context->GetOutputShape(0);
  auto compile_info = reinterpret_cast<const TransDataCompileInfo *>(context->GetCompileInfo());
  auto src_td = context->GetInputDesc(0);
  auto dst_td = context->GetOutputDesc(0);
  if (in_shape == nullptr || out_shape == nullptr || src_td == nullptr || dst_td == nullptr ||
      compile_info == nullptr) {
    return ge::FAILED;
  }
  auto src_format = src_td->GetStorageFormat();
  auto dst_format = dst_td->GetStorageFormat();

  auto real_formats = transdata::GetRealFormat(src_format, dst_format);
  auto real_shape_converter = transdata::GetRealShapeConvertFunc(src_format, dst_format);
  if (real_formats == nullptr || real_shape_converter == nullptr) {
    return ge::FAILED;
  }

  auto data_type = src_td->GetDataType();
  auto c0_size = GetC0SizeWithType(data_type);
  Shape real_in_shape{}, real_out_shape{};

  auto ret = real_shape_converter(in_shape->GetStorageShape(), out_shape->GetStorageShape(), c0_size, real_in_shape,
                                  real_out_shape);
  if (ret != ge::GRAPH_SUCCESS) {
    return ret;
  }

  int64_t block_elem_cnt = BLOCK_BYTE_SIZE / GetSizeByDataType(data_type);

  context->SetBlockDim(compile_info->block_dim);

  if (real_formats->src == transdata::RF_HNC && real_formats->dst == transdata::RF_HCNT) {
    return transdata::TillingPositiveMode1010(context, real_in_shape, real_out_shape, real_formats->src,
                                              real_formats->dst, compile_info->block_dim, block_elem_cnt,
                                              compile_info->ub_size);
  }
  if (real_formats->src == transdata::RF_HCNT && real_formats->dst == transdata::RF_HNC) {
    return transdata::TilingNegativeTc201(context, real_in_shape, real_out_shape, real_formats->src, real_formats->dst,
                                          compile_info->block_dim, block_elem_cnt, compile_info->ub_size, data_type);
  }

  GELOGE(ge::FAILED, "Unsupported src and dst format %d -> %d", real_formats->src, real_formats->dst);
  return ge::GRAPH_FAILED;
}

ge::graphStatus InferShapeForTransData(InferShapeContext *context) {
  auto shape = context->GetInputShape(0);
  auto output_shape = context->GetOutputShape(0);
  if (shape == nullptr || output_shape == nullptr) {
    // log error
    return ge::GRAPH_FAILED;
  }
  *output_shape = *shape;
  return ge::GRAPH_SUCCESS;
}

template<typename T>
bool GetCompileValue(const nlohmann::json &all_vars, const std::string &name, T &value) {
  if (all_vars.empty()) {
    return false;
  }

  if (all_vars.count(name) == 0) {
    return false;
  }

  value = all_vars[name].get<T>();
  return true;
}

ge::graphStatus TilingPrepareForTransdata(KernelContext *context) {
  auto compile_info = context->GetOutputPointer<TransDataCompileInfo>(0);
  auto json_str = context->GetInputStrPointer(0);
  if (compile_info == nullptr || json_str == nullptr) {
    return ge::GRAPH_FAILED;
  }
  std::unique_ptr<nlohmann::json> parsed_object_cinfo(new nlohmann::json(nlohmann::json::parse(json_str)));
  if (parsed_object_cinfo == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const nlohmann::json &allVars = (*parsed_object_cinfo)["vars"];

  if (!GetCompileValue(allVars, "ub_size", compile_info->ub_size)) {
    GELOGE(ge::PARAM_INVALID, "Failed to find ub_size in json %s", json_str);
    return ge::PARAM_INVALID;
  }

  if (!GetCompileValue(allVars, "block_dim", compile_info->block_dim)) {
    GELOGE(ge::PARAM_INVALID, "Failed to find block_dim in json %s", json_str);
    return ge::PARAM_INVALID;
  }

  if (!GetCompileValue(allVars, "group", compile_info->group)) {
    GELOGE(ge::PARAM_INVALID, "Failed to find group in json %s", json_str);
    return ge::PARAM_INVALID;
  }
  compile_info->vnc_fp32_flag = 0;
  if (!GetCompileValue(allVars, "vnc_fp32_flag", compile_info->vnc_fp32_flag)) {
    GELOGD("Can not find vnc_fp32_flag in json, use default value %ld", compile_info->vnc_fp32_flag);
  }

  GELOGD("Parsed TransData compile info(ub, block, group, vnc): %ld, %ld, %ld, %ld", compile_info->ub_size,
         compile_info->block_dim, compile_info->group, compile_info->vnc_fp32_flag);

  return ge::GRAPH_SUCCESS;
}

IMPL_OP(TransData)
    .InferShape(InferShapeForTransData)
    .Tiling(TilingForTransData)
    .TilingParse<TransDataCompileInfo>(TilingPrepareForTransdata);

}  // namespace kernel
}  // namespace gert
