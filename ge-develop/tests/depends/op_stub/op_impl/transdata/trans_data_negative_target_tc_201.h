/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSDATA_TRANS_DATA_NEGATIVE_TARGET_TC_201_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSDATA_TRANS_DATA_NEGATIVE_TARGET_TC_201_H_
#include "exe_graph/runtime/shape.h"
#include "transdata_real_format.h"
#include "exe_graph/runtime/tiling_context.h"
namespace gert {
namespace kernel {
namespace transdata {
struct TransDataTc201Param {
  int64_t tiling_mode;
  int64_t ub_offset;
  int64_t mc_pos;
  int64_t used_core_cnt;
  int64_t src_r2nd_dst_r2nd_same;
  int64_t c0_len;
  int64_t core_step_in;
  int64_t core_step_out;
  int64_t nlc_dst_r2nd_lp_cnt;
  int64_t nlc_src_cl_lp_cnt;
  int64_t nlc_src_left_lp_cnt;
  int64_t nlc_dst_r2nd_left;
  int64_t nlc_src_cl_left;
  int64_t nlc_src_left_left;
  int64_t lc_dst_r2nd_lp_cnt;
  int64_t lc_src_cl_lp_cnt;
  int64_t lc_src_left_lp_cnt;
  int64_t lc_dst_r2nd_left;
  int64_t lc_src_cl_left;
  int64_t lc_src_left_left;
  int64_t dst_r2nd_lp_unit;
  int64_t dst_r2nd_step_in;
  int64_t dst_r2nd_step_out;
  int64_t dst_r2nd_lp_step_in;
  int64_t dst_r2nd_lp_step_out;
  int64_t src_cl_lp_unit;
  int64_t all_c_in;
  int64_t src_cl_step_in;
  int64_t src_cl_step_out;
  int64_t src_cl_lp_step_in;
  int64_t src_cl_lp_step_out;
  int64_t c_mod_c0;
  int64_t src_left_lp_unit;
  int64_t src_left_step_in;
  int64_t src_left_step_out;
  int64_t src_left_lp_step_in;
  int64_t src_left_lp_step_out;
  int64_t dst_r2nd_in_0_size;
  int64_t dst_r2nd_in_0_src_rsize;
  int64_t dst_r2nd_in_0_src_asize;
  int64_t dst_r2nd_in_1_size;
  int64_t dst_r2nd_in_1_src_rsize;
  int64_t dst_r2nd_in_1_src_asize;
  int64_t dst_r2nd_dims;
  int64_t vnc_col_size;
  int64_t all_r2nd_in;
};

ge::graphStatus TilingNegativeTc201(TilingContext *context, const Shape &in_shape, const Shape &out_shape,
                                    RealFormat src_format, RealFormat dst_format,
                                    int64_t core_num, int64_t block_elem_cnt, int64_t ub_size,
                                    ge::DataType& dtype);
}
}
}
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSDATA_TRANS_DATA_NEGATIVE_TARGET_TC_201_H_
