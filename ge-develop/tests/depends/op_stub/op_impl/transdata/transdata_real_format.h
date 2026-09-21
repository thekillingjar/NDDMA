/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSDATA_TRANSDATA_REAL_FORMAT_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSDATA_TRANSDATA_REAL_FORMAT_H_
#include "graph/ge_error_codes.h"
#include "graph/types.h"
namespace gert {
namespace kernel {
namespace transdata {
enum RealAxisType {
  RAT_C,
  RAT_H,
  RAT_N,
  RAT_T,

  RAT_END
};
enum RealFormat {
  RF_NHC,
  RF_NCHT,
  RF_HNC,
  RF_HCNT,

  RF_END
};
struct RealFormatAxisType {
  int32_t rank;
  RealAxisType axis_type[8];
};

// TODO: GetAxisType、GetAxisIndex的代码可以考虑自动生成
inline RealAxisType GetAxisType(RealFormat real_format, int32_t dim_index) {
  static RealFormatAxisType real_formats_axis_types[RF_END] = {
      {3, {RAT_N, RAT_H, RAT_C}},  // NHC
      {4, {RAT_N, RAT_C, RAT_H, RAT_T}},  // NCHT
      {3, {RAT_H, RAT_N, RAT_C}},  // HNC
      {4, {RAT_H, RAT_C, RAT_N,  RAT_T}},  // HCNT
  };
  if (real_format >= RF_END) {
    return RAT_END;
  }
  auto &axis_types = real_formats_axis_types[real_format];

  if (dim_index < 0) {
    if (axis_types.rank + dim_index < 0) {
      return RAT_END;
    }
    return axis_types.axis_type[axis_types.rank + dim_index];
  } else {
    if (dim_index >= axis_types.rank) {
      return RAT_END;
    }
    return axis_types.axis_type[dim_index];
  }
}
inline int32_t GetAxisIndex(RealFormat real_format, RealAxisType axis_type) {
  static int32_t real_formats_axis_index[RF_END][RAT_END] = {
      {2,1,0,-1}, // NHC
      {1,2,0,3},  // NCHT
      {2,0,1,-1},  // HNC
      {1,0,2,3}, // HCNT
  };
  if (real_format >= RF_END) {
    return -1;
  }
  if (axis_type >= RAT_END) {
    return -1;
  }
  return real_formats_axis_index[real_format][axis_type];
}

typedef ge::graphStatus (*RealShapeConvertFunc)(const Shape &in_shape, const Shape &out_shape, int64_t c0_size,
                                                Shape &real_in_shape, Shape &real_out_shape);
RealShapeConvertFunc GetRealShapeConvertFunc(ge::Format src_format, ge::Format dst_format);

}
}
}

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSDATA_TRANSDATA_REAL_FORMAT_H_
