/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSDATA_TRANS_DATA_POSITIVE_SOURCE_TC_1010_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSDATA_TRANS_DATA_POSITIVE_SOURCE_TC_1010_H_
#include "exe_graph/runtime/shape.h"
#include "transdata_real_format.h"
#include "exe_graph/runtime/tiling_context.h"
namespace gert {
namespace kernel {
namespace transdata {
struct TransDataMode1010Param {
  int64_t tiling_mode;
  int64_t ub_offset;
  int64_t used_core_cnt;
  int64_t core_step_in;
  int64_t core_step_out;
  int64_t dst_cl_lp_step_in;
  int64_t dst_cl_lp_step_out;
  int64_t dst_cl_step_in;
  int64_t dst_cl_step_out;
  int64_t dst_cr_lp_step_in;
  int64_t dst_cr_lp_step_out;
  int64_t dst_cr_step_in;
  int64_t nc_le_vcol;
  int64_t vnc_line_size;
  int64_t pln_dst_cl_size;
  int64_t pln_dst_cr_size;
  int64_t vnc_row_size;
  int64_t c_lp_step_in;
  int64_t c_lp_step_out;
  int64_t c_step_out;
  int64_t c0_size;
  int64_t c_mod_c0;
  int64_t c_lp_unit;
  int64_t nlc_dst_cl_lp_cnt;
  int64_t nlc_vnc_row_cl_left;
  int64_t nlc_last_line_cl_cnt;
  int64_t nlc_dst_cr_lp_cnt;
  int64_t nlc_vnc_row_left;
  int64_t nlc_last_line_cr_cnt;
  int64_t nlc_c_lp_cnt;
  int64_t nlc_c_left;
  int64_t lc_dst_cl_lp_cnt;
  int64_t lc_vnc_row_cl_left;
  int64_t lc_last_line_cl_cnt;
  int64_t lc_dst_cr_lp_cnt;
  int64_t lc_vnc_row_left;
  int64_t lc_last_line_cr_cnt;
  int64_t lc_c_lp_cnt;
  int64_t lc_c_left;
};

ge::graphStatus TillingPositiveMode1010(TilingContext *context, const Shape &in_shape, const Shape &out_shape,
                                        RealFormat src_format, RealFormat dst_format,
                                        int64_t core_num, int64_t block_elem_cnt, int64_t ub_size);
int64_t GetShapeSize(const Shape& in_shape, int32_t pos);
}
}
}
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSDATA_TRANS_DATA_POSITIVE_SOURCE_TC_1010_H_
