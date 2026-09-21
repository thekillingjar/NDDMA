/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_UTILS_COMMON_FORMAT_AXIS_UTIL_H_
#define FUSION_ENGINE_UTILS_COMMON_FORMAT_AXIS_UTIL_H_

#include <functional>
#include <vector>
#include "common/fe_inner_error_codes.h"
#include "common/fe_utils.h"
#include "common/math_util.h"
#include "common/util/op_info_util.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"

namespace fe {
enum AxisValueType {
  AXIS_N = 0,
  AXIS_C = 1,
  AXIS_H = 2,
  AXIS_W = 3,
  AXIS_C1 = 4,
  AXIS_C0 = 5,
  AXIS_Co = 6,
  AXIS_D = 7,
  AXIS_G = 8,
  AXIS_INPUT_SIZE = 9,
  AXIS_HIDEEN_SIZE = 10,
  AXIS_STATE_SIZE = 11,
  AXIS_BOTTOM = 12
};

using AxisNameNumberMap = std::map<std::string, std::vector<int32_t>>;
const AxisNameNumberMap AXIS_NAME_NUMBER_MAP_5HD{{N_AXIS_NAME, {NC1HWC0_DIM_N}},
                                                 {H_AXIS_NAME, {NC1HWC0_DIM_H}},
                                                 {W_AXIS_NAME, {NC1HWC0_DIM_W}},
                                                 {C_AXIS_NAME, {NC1HWC0_DIM_C1, NC1HWC0_DIM_C0}}};

const AxisNameNumberMap AXIS_NAME_NUMBER_MAP_6HD{{N_AXIS_NAME, {NDC1HWC0_DIM_N}},
                                                 {D_AXIS_NAME, {NDC1HWC0_DIM_D}},
                                                 {H_AXIS_NAME, {NDC1HWC0_DIM_H}},
                                                 {W_AXIS_NAME, {NDC1HWC0_DIM_W}},
                                                 {C_AXIS_NAME, {NDC1HWC0_DIM_C1, NDC1HWC0_DIM_C0}}};

const AxisNameNumberMap AXIS_NAME_NUMBER_MAP_FZ{
    {H_AXIS_NAME, {C1HWNCoC0_DIM_H}}, {W_AXIS_NAME, {C1HWNCoC0_DIM_W}},
};

const AxisNameNumberMap AXIS_NAME_NUMBER_MAP_FZ_3D{
    {D_AXIS_NAME, {C1DHWNCoC0_DIM_D}}, {H_AXIS_NAME, {C1DHWNCoC0_DIM_H}}, {W_AXIS_NAME, {C1DHWNCoC0_DIM_W}},
};

const std::map<ge::Format, AxisNameNumberMap> FORMAT_AXIS_NAME_NUMBER_MAP{
    {ge::FORMAT_NC1HWC0, AXIS_NAME_NUMBER_MAP_5HD},
    {ge::FORMAT_NDC1HWC0, AXIS_NAME_NUMBER_MAP_6HD},
    {ge::FORMAT_FRACTAL_Z, AXIS_NAME_NUMBER_MAP_FZ},
    {ge::FORMAT_FRACTAL_Z_3D, AXIS_NAME_NUMBER_MAP_FZ_3D},
    /* Axis info for 6D is the same as Fractal_Z */
    {ge::FORMAT_C1HWNCoC0, AXIS_NAME_NUMBER_MAP_FZ}};

uint64_t DivisionCeiling(uint64_t dividend, uint64_t divisor);
int64_t DivisionCeiling(int64_t dividend, int64_t divisor);

/* Axis value is arranged as {N,C,H,W,C1,C0,...} */
/* The first parameter is old shape's dimension,
 * second is c0 and third is axis value. */
using GetAxisValueInfoByFormat =
        std::function<Status(const vector<int64_t>&, const uint32_t&, vector<int64_t>&, vector<int64_t>&)>;

using GetAxisValueInfoByFormatPtr = std::shared_ptr<GetAxisValueInfoByFormat>;

class AxisUtil {
 public:
  static Status GetAxisValueByOriginFormat(const ge::Format& format, const vector<int64_t>& dim_vec, const int64_t& c0,
                                           vector<int64_t>& axis_value, vector<int64_t>& nd_value);

  static bool HasAxisValueFunc(const ge::Format& format);
  static Status GetOriginAxisAttribute(const ge::OpDesc& op_desc, const ge::GeShape shape,
                                       vector<int64_t>& axis_index_vec);

 private:
  static Status CheckParams(const vector<int64_t>& original_dim_vec, const int64_t& c0, vector<int64_t>& nd_value,
                            const size_t& dim_default_size);

  static Status GetAxisValueByNCHW(const vector<int64_t>& original_dim_vec, const int64_t &c0,
                                   vector<int64_t>& axis_value_a, vector<int64_t>& nd_value);

  static Status GetAxisValueByNHWC(const vector<int64_t>& original_dim_vec, const int64_t &c0,
                                   vector<int64_t>& axis_value_c, vector<int64_t>& nd_value);

  static Status GetAxisValueByNC1HWC0(const vector<int64_t>& original_dim_vec_a, const int64_t &c0,
                                      vector<int64_t>& axis_value_j, vector<int64_t>& nd_value);

  static Status GetAxisValueByFz(const vector<int64_t>& original_dim_vec_b, const int64_t &c0,
                                 vector<int64_t>& axis_value_f, vector<int64_t>& nd_value);

  static Status GetAxisValueByHWCN(const vector<int64_t>& original_dim_vec, const int64_t &c0,
                                   vector<int64_t>& axis_value_d, vector<int64_t>& nd_value);

  static Status GetAxisValueByCHWN(const vector<int64_t>& original_dim_vec, const int64_t &c0,
                                   vector<int64_t>& axis_value_e, vector<int64_t>& nd_value);

  static Status GetAxisValueByND(const vector<int64_t>& original_dim_vec, const int64_t &c0,
                                 vector<int64_t>& axis_value_b, vector<int64_t>& nd_value);

  static Status GetAxisValueByNDHWC(const vector<int64_t>& original_dim_vec_c, const int64_t &c0,
                                    vector<int64_t>& axis_value_f, vector<int64_t>& nd_value);

  static Status GetAxisValueByNCDHW(const vector<int64_t>& original_dim_vec_d, const int64_t &c0,
                                    vector<int64_t>& axis_value_g, vector<int64_t>& nd_value);

  static Status GetAxisValueByDHWCN(const vector<int64_t>& original_dim_vec_e, const int64_t &c0,
                                    vector<int64_t>& axis_value_h, vector<int64_t>& nd_value);

  static Status GetAxisValueByDHWNC(const vector<int64_t>& original_dim_vec_f, const int64_t &c0,
                                    vector<int64_t>& axis_value_i, vector<int64_t>& nd_value);

  /* map of GetAxisValueInfoByFormat, get axis value by different original
   * formats. */
  static const std::map<ge::Format, GetAxisValueInfoByFormatPtr> get_axis_value_func_map;
};
}  // namespace fe

#endif  // FUSION_ENGINE_UTILS_COMMON_FORMAT_AXIS_UTIL_H_
