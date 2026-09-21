/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file trans_data_positive_source_tc_1010.cc
 * \brief dynamic TransData op tiling
 */
#include "trans_data_positive_source_tc_1010.h"
#include <algorithm>
#include "graph/ge_error_codes.h"

#include "graph/utils/math_util.h"
#include "exe_graph/runtime/tiling_context.h"

namespace gert {
namespace kernel {
namespace transdata {
constexpr int64_t C0_16 = 16;
constexpr int64_t VNC_LINES = 16;

int64_t GetCeilFillB(int64_t u_value, int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }

  res_value = (u_value + d_value - 1) / d_value * d_value;

  return res_value;
}

int64_t GetShapeSize(const Shape &in_shape, int32_t pos) {
  int32_t n = in_shape.GetDimNum();
  int64_t shape_size = 1;
  if (pos < 0) {
    pos = n + pos;
  }
  for (int32_t i = pos; i < n; i++) {
    shape_size *= in_shape[i];
  }
  return shape_size;
}

void GetMcInfoPositive1010(int64_t &dst_cl_lp_cnt, int64_t &vnc_row_cl_left, int64_t &ll_dst_cl_left, int64_t &c_lp_cnt,
                           int64_t c_left, int64_t &dst_cr_lp_cnt, int64_t vnc_row_left, int64_t ll_dst_cr_left,
                           int64_t &core_num, TransDataMode1010Param &params) {
  int64_t tmp_full_loop_cnt_cr = ge::FloorDiv(dst_cr_lp_cnt, core_num) > 0 ? core_num : 0;
  int64_t reminder_loop_cnt_cr = dst_cr_lp_cnt % core_num;
  if (reminder_loop_cnt_cr == 0) {
    tmp_full_loop_cnt_cr += core_num;
  }
  int64_t full_loop_cnt_cr = tmp_full_loop_cnt_cr + reminder_loop_cnt_cr;

  int64_t tmp_full_loop_cnt_c = ge::FloorDiv(c_lp_cnt, core_num) > 0 ? core_num : 0;

  int64_t reminder_loop_cnt_c = c_lp_cnt % core_num;
  if (reminder_loop_cnt_c == 0) {
    tmp_full_loop_cnt_c += core_num;
  }
  int64_t full_loop_cnt_c = tmp_full_loop_cnt_c + reminder_loop_cnt_c;

  int64_t tmp_full_loop_cnt_left = ge::FloorDiv(dst_cl_lp_cnt, core_num) > 0 ? core_num : 0;
  int64_t reminder_loop_cnt_left = dst_cl_lp_cnt % core_num;
  if (reminder_loop_cnt_left == 0) {
    tmp_full_loop_cnt_left += core_num;
  }
  int64_t full_loop_cnt_left = tmp_full_loop_cnt_left + reminder_loop_cnt_left;

  auto max_value = std::max(std::max(full_loop_cnt_left, full_loop_cnt_cr), full_loop_cnt_c);
  if (max_value == full_loop_cnt_left) {
    params.used_core_cnt = ge::CeilDiv(dst_cl_lp_cnt, ge::CeilDiv(dst_cl_lp_cnt, core_num));
    params.nlc_dst_cl_lp_cnt = ge::CeilDiv(dst_cl_lp_cnt, params.used_core_cnt);
    params.lc_dst_cl_lp_cnt = dst_cl_lp_cnt - params.nlc_dst_cl_lp_cnt * (params.used_core_cnt - 1);
    params.core_step_in = params.nlc_dst_cl_lp_cnt * params.dst_cl_lp_step_in;
    params.core_step_out = params.nlc_dst_cl_lp_cnt * params.dst_cl_lp_step_out;
    params.nlc_vnc_row_cl_left = 0;
    params.lc_vnc_row_cl_left = vnc_row_cl_left;
    params.nlc_last_line_cl_cnt = ll_dst_cl_left;
    params.lc_last_line_cl_cnt = ll_dst_cl_left;
    params.nlc_c_lp_cnt = c_lp_cnt;
    params.lc_c_lp_cnt = c_lp_cnt;
    params.nlc_c_left = c_left;
    params.lc_c_left = c_left;
    params.nlc_dst_cr_lp_cnt = dst_cr_lp_cnt;
    params.lc_dst_cr_lp_cnt = dst_cr_lp_cnt;
    params.nlc_vnc_row_left = vnc_row_left;
    params.lc_vnc_row_left = vnc_row_left;
    params.nlc_last_line_cr_cnt = ll_dst_cr_left;
    params.lc_last_line_cr_cnt = ll_dst_cr_left;
  } else if (max_value == full_loop_cnt_cr) {
    params.used_core_cnt = ge::CeilDiv(dst_cr_lp_cnt, ge::CeilDiv(dst_cr_lp_cnt, core_num));
    params.nlc_dst_cr_lp_cnt = ge::CeilDiv(dst_cr_lp_cnt, params.used_core_cnt);
    params.lc_dst_cr_lp_cnt = dst_cr_lp_cnt - params.nlc_dst_cr_lp_cnt * (params.used_core_cnt - 1);
    params.core_step_in = params.nlc_dst_cr_lp_cnt * params.dst_cr_lp_step_in;
    params.core_step_out = params.nlc_dst_cr_lp_cnt * params.dst_cr_lp_step_out;
    params.nlc_vnc_row_left = 0;
    params.lc_vnc_row_left = vnc_row_left;
    params.nlc_last_line_cr_cnt = params.pln_dst_cr_size;
    params.lc_last_line_cr_cnt = ll_dst_cr_left;
    params.nlc_c_lp_cnt = c_lp_cnt;
    params.lc_c_lp_cnt = c_lp_cnt;
    params.nlc_c_left = c_left;
    params.lc_c_left = c_left;
    params.nlc_dst_cl_lp_cnt = dst_cl_lp_cnt;
    params.lc_dst_cl_lp_cnt = dst_cl_lp_cnt;
    params.nlc_vnc_row_cl_left = vnc_row_cl_left;
    params.lc_vnc_row_cl_left = vnc_row_cl_left;
    params.nlc_last_line_cl_cnt = ll_dst_cl_left;
    params.lc_last_line_cl_cnt = ll_dst_cl_left;
  } else {
    params.used_core_cnt = ge::CeilDiv(c_lp_cnt, ge::CeilDiv(c_lp_cnt, core_num));
    params.nlc_c_lp_cnt = ge::CeilDiv(c_lp_cnt, params.used_core_cnt);
    params.lc_c_lp_cnt = c_lp_cnt - params.nlc_c_lp_cnt * (params.used_core_cnt - 1);
    params.core_step_in = params.nlc_c_lp_cnt * params.c_lp_step_in;
    params.core_step_out = params.nlc_c_lp_cnt * params.c_lp_step_out;
    params.nlc_c_left = 0;
    params.lc_c_left = c_left;
    params.nlc_dst_cl_lp_cnt = dst_cl_lp_cnt;
    params.lc_dst_cl_lp_cnt = dst_cl_lp_cnt;
    params.nlc_vnc_row_cl_left = vnc_row_cl_left;
    params.lc_vnc_row_cl_left = vnc_row_cl_left;
    params.nlc_last_line_cl_cnt = ll_dst_cl_left;
    params.lc_last_line_cl_cnt = ll_dst_cl_left;
    params.nlc_dst_cr_lp_cnt = dst_cr_lp_cnt;
    params.lc_dst_cr_lp_cnt = dst_cr_lp_cnt;
    params.nlc_vnc_row_left = vnc_row_left;
    params.lc_vnc_row_left = vnc_row_left;
    params.nlc_last_line_cr_cnt = ll_dst_cr_left;
    params.lc_last_line_cr_cnt = ll_dst_cr_left;
  }
}

constexpr int64_t TRANSDATA_TILING_FACTOR_2 = 2;
constexpr int64_t TRANSDATA_TILING_FACTOR_4 = 4;
constexpr int64_t TILING_MODE_1010 = 1010;

constexpr int64_t TRANSDATA_TILING_PARAM_2 = 2;
constexpr int64_t TRANSDATA_TILING_PARAM_3 = 3;
constexpr int64_t TRANSDATA_TILING_PARAM_4 = 4;
constexpr int64_t TRANSDATA_TILING_PARAM_5 = 5;

void GetCommonParam(int64_t ub_size, int64_t block_elem_cnt, int64_t c0_len, int64_t axis_c_size,
                    TransDataMode1010Param &params) {
  if (block_elem_cnt == 0) {
    // VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "block_elem_cnt shoule not be 0");
    return;
  }

  int64_t half_ub_size;
  if (c0_len == C0_16) {
    half_ub_size = ub_size / TRANSDATA_TILING_FACTOR_2;
  } else {
    half_ub_size = ub_size / TRANSDATA_TILING_FACTOR_4;
  }
  params.vnc_line_size = half_ub_size / VNC_LINES / block_elem_cnt * block_elem_cnt;
  int64_t tmp_ub_offset = params.vnc_line_size * VNC_LINES;
  if (c0_len == C0_16) {
    params.ub_offset = tmp_ub_offset;
  } else {
    params.ub_offset = tmp_ub_offset * TRANSDATA_TILING_FACTOR_2;
  }
  params.c_mod_c0 = axis_c_size % c0_len;
  params.c0_size = c0_len;
}

ge::graphStatus TillingPositiveMode1010(TilingContext *context, const Shape &in_shape, const Shape &out_shape,
                                        RealFormat src_format, RealFormat dst_format, int64_t core_num,
                                        int64_t block_elem_cnt, int64_t ub_size) {
  auto params = context->GetTilingData<TransDataMode1010Param>();
  if (params == nullptr) {
    return ge::GRAPH_FAILED;
  }
  /*
  if ((src_format.length() != in_shape.size()) || (dst_format.length() != out_shape.size())) {
    // VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "TillingPositiveMode1010 Failed.");
    return false;
  }
  */
  int64_t axis_c_size = in_shape[in_shape.GetDimNum() - 1];
  int64_t c0_len = out_shape[out_shape.GetDimNum() - 1];
  GetCommonParam(ub_size, block_elem_cnt, c0_len, axis_c_size, *params);
  params->tiling_mode = TILING_MODE_1010;
  params->vnc_line_size = params->vnc_line_size / c0_len * c0_len;

  // source axis c tiling parameters
  int32_t dst_axis_pos_c = GetAxisIndex(dst_format, RAT_C);
  if (axis_c_size < params->vnc_line_size) {
    params->c_lp_unit = axis_c_size;
  } else {
    params->c_lp_unit = params->vnc_line_size;
  }
  params->c_lp_step_in = params->c_lp_unit;
  int64_t lp_c1_cnt = ge::CeilDiv(params->c_lp_unit, c0_len);
  params->c_lp_step_out = lp_c1_cnt * GetShapeSize(out_shape, dst_axis_pos_c + 1);
  params->c_step_out = GetShapeSize(out_shape, dst_axis_pos_c + 1);
  int64_t c_lp_cnt = ge::CeilDiv(axis_c_size, params->c_lp_unit);
  int64_t c_left = axis_c_size % params->c_lp_unit;

  // target axis c-right tiling parameters
  int64_t axis_dst_cl_size = 1;
  for (int32_t i = 0; i < dst_axis_pos_c; i++) {
    axis_dst_cl_size *= out_shape[i];
  }

  auto tmp_src_pos = GetAxisIndex(src_format, GetAxisType(dst_format, -2));
  int64_t axis_dst_cr_size = GetShapeSize(in_shape, tmp_src_pos) / in_shape[in_shape.GetDimNum() - 1];
  params->pln_dst_cr_size = params->vnc_line_size / GetCeilFillB(params->c_lp_unit, c0_len);
  params->vnc_row_size = VNC_LINES;
  int64_t per_vnc_dst_cr_cnt = params->pln_dst_cr_size * params->vnc_row_size;
  if (per_vnc_dst_cr_cnt >= axis_dst_cr_size && core_num > 1 && axis_dst_cl_size == 1) {
    int64_t new_vnc_lines = ge::CeilDiv(axis_dst_cr_size, params->pln_dst_cr_size);
    if (new_vnc_lines > VNC_LINES) {
      new_vnc_lines = VNC_LINES;
    }
    int64_t vnc_per_core = new_vnc_lines > core_num ? ge::CeilDiv(new_vnc_lines, core_num) : 1;
    params->vnc_row_size = vnc_per_core;
    per_vnc_dst_cr_cnt = params->pln_dst_cr_size * params->vnc_row_size;
  }
  int64_t dst_cr_lp_cnt = ge::CeilDiv(axis_dst_cr_size, per_vnc_dst_cr_cnt);
  int64_t dst_cr_left = axis_dst_cr_size % per_vnc_dst_cr_cnt;
  int64_t vnc_row_left = ge::CeilDiv(dst_cr_left, params->pln_dst_cr_size);
  int64_t tmp_dst_cr_left = dst_cr_left % params->pln_dst_cr_size;
  int64_t ll_dst_cr_left;
  if (tmp_dst_cr_left > 0) {
    ll_dst_cr_left = tmp_dst_cr_left;
  } else {
    ll_dst_cr_left = params->pln_dst_cr_size;
  }

  params->dst_cr_lp_step_in = in_shape[in_shape.GetDimNum() - 1] * per_vnc_dst_cr_cnt;
  int32_t tmp_dst_pos = GetAxisIndex(dst_format, GetAxisType(src_format, -2));
  params->dst_cr_lp_step_out = GetShapeSize(out_shape, tmp_dst_pos + 1) * per_vnc_dst_cr_cnt;
  params->dst_cr_step_in = GetShapeSize(in_shape, -1);

  // target axis c-left tiling parameters
  int64_t per_vnc_dst_cl_cnt = 1;
  int64_t dst_cl_lp_cnt = 1;
  int64_t dst_cl_left = 0;
  int64_t vnc_row_cl_left = 0;
  int64_t tmp_dst_cl_left = 0;
  int64_t ll_dst_cl_left = 0;

  if ((axis_c_size % c0_len == 0 && ge::CeilDiv(params->c_lp_unit, block_elem_cnt) % C0_16 != 0) ||
      (axis_c_size % c0_len == 0 && params->pln_dst_cr_size % TRANSDATA_TILING_FACTOR_2 == 0)) {
    // move in cl_cr_c in together
    if (params->c_lp_unit == axis_c_size && per_vnc_dst_cr_cnt >= axis_dst_cr_size) {
      params->nc_le_vcol = TRANSDATA_TILING_PARAM_3;
      per_vnc_dst_cl_cnt = ge::FloorDiv(params->vnc_line_size * VNC_LINES, axis_c_size * axis_dst_cr_size);
    } else if (params->c_lp_unit == axis_c_size) {
      // move in cr_c in together
      params->nc_le_vcol = TRANSDATA_TILING_PARAM_4;
      per_vnc_dst_cl_cnt = 1;
    } else {
      // move in c
      params->nc_le_vcol = TRANSDATA_TILING_PARAM_5;
      per_vnc_dst_cl_cnt = 1;
    }
    params->pln_dst_cl_size = per_vnc_dst_cl_cnt;
    dst_cl_lp_cnt = ge::CeilDiv(axis_dst_cl_size, params->pln_dst_cl_size);
    vnc_row_cl_left = axis_dst_cl_size % params->pln_dst_cl_size;
    ll_dst_cl_left = axis_dst_cl_size % params->pln_dst_cl_size;
  } else if (dst_cr_lp_cnt == 1 && params->c_lp_unit == axis_c_size &&
             vnc_row_left <= ge::FloorDiv(VNC_LINES, TRANSDATA_TILING_FACTOR_2)) {
    // nc is less than vnchwconv col size
    if (vnc_row_left == 1) {
      params->nc_le_vcol = 1;
      params->pln_dst_cl_size = ge::FloorDiv(params->pln_dst_cr_size, axis_dst_cr_size);
    } else {
      params->nc_le_vcol = TRANSDATA_TILING_PARAM_2;
      params->pln_dst_cl_size = 1;
      // adjust c-right parameters
      dst_cr_lp_cnt = ge::CeilDiv(axis_dst_cr_size, params->pln_dst_cr_size);
      vnc_row_left = axis_dst_cr_size % params->pln_dst_cr_size;
      if (vnc_row_left > 0) {
        ll_dst_cr_left = vnc_row_left;
      } else {
        ll_dst_cr_left = params->pln_dst_cr_size;
      }
      params->dst_cr_lp_step_in = in_shape[in_shape.GetDimNum() - 1] * params->pln_dst_cr_size;
      params->dst_cr_lp_step_out = GetShapeSize(out_shape, tmp_dst_pos + 1) * params->pln_dst_cr_size;
    }

    per_vnc_dst_cl_cnt = params->pln_dst_cl_size * params->vnc_row_size;
    dst_cl_lp_cnt = ge::CeilDiv(axis_dst_cl_size, per_vnc_dst_cl_cnt);
    // adjust c-left parameters
    int64_t four_in_core_cnt = 4;
    int64_t pln_cl_gate = 64;
    if ((dst_cl_lp_cnt < ge::FloorDiv(core_num, four_in_core_cnt)) && (params->pln_dst_cl_size > pln_cl_gate)) {
      params->pln_dst_cl_size = ge::FloorDiv(params->pln_dst_cl_size, pln_cl_gate);
      per_vnc_dst_cl_cnt = params->pln_dst_cl_size * params->vnc_row_size;
      dst_cl_lp_cnt = ge::CeilDiv(axis_dst_cl_size, per_vnc_dst_cl_cnt);
    }
    dst_cl_left = axis_dst_cl_size % per_vnc_dst_cl_cnt;
    vnc_row_cl_left = ge::CeilDiv(dst_cl_left, params->pln_dst_cl_size);
    tmp_dst_cl_left = dst_cl_left % params->pln_dst_cl_size;
    if (tmp_dst_cl_left > 0) {
      ll_dst_cl_left = tmp_dst_cl_left;
    } else {
      ll_dst_cl_left = params->pln_dst_cl_size;
    }
  } else {
    params->nc_le_vcol = 0;
    params->pln_dst_cl_size = 1;
    dst_cl_lp_cnt = axis_dst_cl_size;
    vnc_row_cl_left = params->pln_dst_cl_size;
    ll_dst_cl_left = params->pln_dst_cl_size;
  }

  params->dst_cl_step_in = GetShapeSize(in_shape, GetAxisIndex(src_format, GetAxisType(dst_format, -1)) + 1);
  params->dst_cl_step_out = GetShapeSize(out_shape, dst_axis_pos_c);
  if (params->nc_le_vcol == 0) {
    params->dst_cl_lp_step_in = params->dst_cl_step_in;
    params->dst_cl_lp_step_out = params->dst_cl_step_out;
  } else {
    params->dst_cl_lp_step_in = params->dst_cl_step_in * per_vnc_dst_cl_cnt;
    params->dst_cl_lp_step_out = params->dst_cl_step_out * per_vnc_dst_cl_cnt;
  }

  GetMcInfoPositive1010(dst_cl_lp_cnt, vnc_row_cl_left, ll_dst_cl_left, c_lp_cnt, c_left, dst_cr_lp_cnt, vnc_row_left,
                        ll_dst_cr_left, core_num, *params);
  return ge::SUCCESS;
}
/*
void PrintTilingMode1010Params(const std::string& op_type, const TransDataMode1010Param& params) {
  OP_LOGD(op_type, "tiling_mode=%d", params.tiling_mode);
  OP_LOGD(op_type, "ub_offset=%d", params.ub_offset);
  OP_LOGD(op_type, "used_core_cnt=%d", params.used_core_cnt);
  OP_LOGD(op_type, "core_step_in=%d", params.core_step_in);
  OP_LOGD(op_type, "core_step_out=%d", params.core_step_out);

  OP_LOGD(op_type, "dst_cl_lp_step_in=%d", params.dst_cl_lp_step_in);
  OP_LOGD(op_type, "dst_cl_lp_step_out=%d", params.dst_cl_lp_step_out);
  OP_LOGD(op_type, "dst_cl_step_in=%d", params.dst_cl_step_in);
  OP_LOGD(op_type, "dst_cl_step_out=%d", params.dst_cl_step_out);
  OP_LOGD(op_type, "dst_cr_lp_step_in=%d", params.dst_cr_lp_step_in);
  OP_LOGD(op_type, "dst_cr_lp_step_out=%d", params.dst_cr_lp_step_out);
  OP_LOGD(op_type, "dst_cr_step_in=%d", params.dst_cr_step_in);
  OP_LOGD(op_type, "nc_le_vcol=%d", params.nc_le_vcol);
  OP_LOGD(op_type, "vnc_line_size=%d", params.vnc_line_size);

  OP_LOGD(op_type, "pln_dst_cl_size=%d", params.pln_dst_cl_size);
  OP_LOGD(op_type, "pln_dst_cr_size=%d", params.pln_dst_cr_size);
  OP_LOGD(op_type, "vnc_row_size=%d", params.vnc_row_size);
  OP_LOGD(op_type, "c_lp_step_in=%d", params.c_lp_step_in);
  OP_LOGD(op_type, "c_lp_step_out=%d", params.c_lp_step_out);
  OP_LOGD(op_type, "c_step_out=%d", params.c_step_out);
  OP_LOGD(op_type, "c0_size=%d", params.c0_size);
  OP_LOGD(op_type, "c_mod_c0=%d", params.c_mod_c0);
  OP_LOGD(op_type, "c_lp_unit=%d", params.c_lp_unit);

  OP_LOGD(op_type, "nlc_dst_cl_lp_cnt=%d", params.nlc_dst_cl_lp_cnt);
  OP_LOGD(op_type, "nlc_vnc_row_cl_left=%d", params.nlc_vnc_row_cl_left);
  OP_LOGD(op_type, "nlc_last_line_cl_cnt=%d", params.nlc_last_line_cl_cnt);
  OP_LOGD(op_type, "nlc_dst_cr_lp_cnt=%d", params.nlc_dst_cr_lp_cnt);
  OP_LOGD(op_type, "nlc_vnc_row_left=%d", params.nlc_vnc_row_left);
  OP_LOGD(op_type, "nlc_last_line_cr_cnt=%d", params.nlc_last_line_cr_cnt);
  OP_LOGD(op_type, "nlc_c_lp_cnt=%d", params.nlc_c_lp_cnt);
  OP_LOGD(op_type, "nlc_c_left=%d", params.nlc_c_left);

  OP_LOGD(op_type, "lc_dst_cl_lp_cnt=%d", params.lc_dst_cl_lp_cnt);
  OP_LOGD(op_type, "lc_vnc_row_cl_left=%d", params.lc_vnc_row_cl_left);
  OP_LOGD(op_type, "lc_last_line_cl_cnt=%d", params.lc_last_line_cl_cnt);
  OP_LOGD(op_type, "lc_dst_cr_lp_cnt=%d", params.lc_dst_cr_lp_cnt);
  OP_LOGD(op_type, "lc_vnc_row_left=%d", params.lc_vnc_row_left);
  OP_LOGD(op_type, "lc_last_line_cr_cnt=%d", params.lc_last_line_cr_cnt);
  OP_LOGD(op_type, "lc_c_lp_cnt=%d", params.lc_c_lp_cnt);
  OP_LOGD(op_type, "lc_c_left=%d", params.lc_c_left);
}
 */
}  // namespace transdata
}  // namespace kernel
}  // namespace gert
