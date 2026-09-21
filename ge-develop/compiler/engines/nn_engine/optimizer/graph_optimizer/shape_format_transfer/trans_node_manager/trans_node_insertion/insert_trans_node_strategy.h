/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_MANAGER_TRANS_NODE_INSERTION_INSERT_TRANS_NODE_STRATEGY_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_MANAGER_TRANS_NODE_INSERTION_INSERT_TRANS_NODE_STRATEGY_H_

#include <map>
#include "common/math_util.h"
#include "common/fe_type_utils.h"
#include "common/util/op_info_util.h"
#include "graph/node.h"

namespace fe {
const uint64_t kStrategyIdExtDeft = 0x00000; /* 0000 00 00 */
const uint64_t kStrategyIdExtThree = 0x30000; /* 0011[C==1 && N==groups && groups > 1] 00 00 */

struct TransferInfo {
  ge::OpDescPtr src_op_desc;
  ge::OutDataAnchorPtr src_anchor;
  ge::OpDescPtr dst_op_desc;
  ge::InDataAnchorPtr dst_anchor;
  ge::NodePtr src_node_ptr;
  ge::NodePtr dst_node_ptr;
  ge::ConstGeTensorDescPtr src_out_tensor_desc_ptr;
  ge::ConstGeTensorDescPtr dst_in_tensor_desc_ptr;

  ge::Format src_out_primary_format;
  ge::Format init_src_out_primary_format;
  int32_t src_out_sub_format = 0;
  int32_t src_out_c0_format = 0;
  ge::DataType src_out_data_type;
  ge::GeShape src_out_shape;

  ge::Format dst_in_primary_format;
  ge::Format init_dst_in_primary_format;
  int32_t dst_in_sub_format = 0;
  int32_t dst_in_c0_format = 0;
  ge::DataType dst_in_data_type;
  ge::GeShape dst_in_shape;

  string src_op_desc_type;
  string dst_op_desc_type;

  string src_reshape_type;
  string dst_reshape_type;


  /* if dst_in_format is RNN-related format, need to use hidden_size and input_size to
   * calc new shape */
  int64_t hidden_size;
  int64_t input_size;
  int64_t state_size;

  /* src_imply_type is always pointed to the source normal op in the origin graph
   * instead of trans op which is inserted by this class TransOpInsert */
  OpImplType src_imply_type;
  /* dst_imply_type is always pointed to the destination normal op in the origin
   * graph instead of trans op which is inserted by this class TransOpInsert */
  OpImplType dst_imply_type;

  uint32_t insertion_mode;

  /* Set this source out orginal format and shape for getting 4D shape
   * without padding in 5->4 scenario */
  ge::Format src_out_original_format;
  ge::GeShape src_out_original_shape;

  ge::Format dst_in_original_format;
  ge::GeShape dst_in_original_shape;

  bool is_source_weight;
  bool is_dst_weight;

  OpPattern src_op_pattern;
  OpPattern dst_op_pattern;

  vector<std::pair<int64_t, int64_t>> src_out_range;
  vector<std::pair<int64_t, int64_t>> dst_in_range;

  uint64_t strategy_id;
};
using TransInfo = struct TransferInfo;
using TransInfoPtr = std::shared_ptr<TransInfo>;

/* strategy id constitute:
 * 20bits:     0000   00000000  00000000
 *   high 4bits[extra_val]: res[2 bits] C==1[1 bit] N==groups[1 bit]
 *            media 8bits[src_format]:
 *                         low 8bits[dst_format]:
*/
inline uint64_t CalcStrategyId(const ge::Format &src_format, const ge::Format &dst_format, const uint64_t extra_val) {
  return ((static_cast<uint64_t>(src_format)) << static_cast<uint32_t>(BitShift::BIT_SHIFT_8)) |
         (static_cast<uint64_t>(dst_format)) | extra_val;
}

/* strategy extra id constitute:
 * 16bits:     00000000  00000000
 *      high 8bits[src_format]:
 *                    low 8bits[dst_format]:
*/
inline uint64_t CalcBaseStrategyId(const ge::Format &src_format, const ge::Format &dst_format) {
  return ((static_cast<uint64_t>(src_format)) << static_cast<uint32_t>(BitShift::BIT_SHIFT_8)) |
         (static_cast<uint64_t>(dst_format));
}

enum InsertionMode {
  INSERT_TRANS_NODE_BY_ORIGINAL_FORMAT_FRONT = 0,
  INSERT_TRANS_NODE_BY_ORIGINAL_FORMAT_END = 1,
  INSERT_TRANS_NODE_BY_CONSECUTIVE_PRINCIPLE = 2,
  INSERTION_MODE_BOTTOM
};

const uint32_t DEFAULT_LOW_TO_HIGH_STRATEGY = 0xffffffff;
const uint32_t DEFAULT_HIGH_TO_LOW_STRATEGY = 0xfffffffe;

using strategy_id_map = std::map<uint64_t, std::vector<uint32_t>>;

enum TransFormIndex {
  RESHAPE_INDEX = 0,
  TRANSPOSE_INDEX = 1,
  CAST_INDEX = 2,
  TRANSDATA_INDEX = 3,
  TRANSDATARNN_INDEX = 4,
  REFORMAT_INEDX = 5,
  SQUEEZE_V2_INDEX = 6,
  UNSQUEEZE_V2_INDEX = 7,
  FORBIDDEN_INDEX = 8
};

using CheckExtValCondFunc = std::function<Status(const TransInfoPtr &)>;
using CheckExtValCondFuncPtr = std::shared_ptr<CheckExtValCondFunc>;

// <StrategyExtraId, <ext_val, checkFunc>>
using strategy_extra_val_map = std::unordered_map<uint64_t, std::unordered_map<uint64_t, CheckExtValCondFuncPtr>>;

inline Status CheckNCHWFzTrans(const TransInfoPtr &trans_info_ptr) {
  FE_CHECK_NOTNULL_WARNLOG(trans_info_ptr);
  /* NCHW  --> FFRACTAL_Z  condition is C==1 && N == groups && groups > 1 */
  int64_t groups = static_cast<int64_t>(trans_info_ptr->dst_in_sub_format);
  if (groups <= 1) {
    FE_LOGD("Groups is %ld.", groups);
    return FAILED;
  }

  if (trans_info_ptr->src_out_data_type != ge::DT_FLOAT16 &&
      trans_info_ptr->src_out_data_type != ge::DT_FLOAT &&
      trans_info_ptr->src_out_data_type != ge::DT_BF16) {
    FE_LOGD("Data type is %s.", DTypeToStr(trans_info_ptr->src_out_data_type).c_str());
    return FAILED;
  }

  vector<int64_t> dim_vec = trans_info_ptr->dst_in_original_shape.GetDims();
  if (dim_vec.empty()) {
    FE_LOGD("Dim_vec is empty.");
    return FAILED;
  }
  int32_t dim_vec_size = static_cast<int32_t>(dim_vec.size());
  int32_t index_c = GetAxisIndexByFormat(trans_info_ptr->dst_in_original_format, C_AXIS_NAME);
  if (index_c < 0 || index_c >= dim_vec_size) {
    REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][CheckNCHWFzTrans] Can not get C index[%d] of format [%s].",
        index_c, ge::TypeUtils::FormatToSerialString(trans_info_ptr->dst_in_original_format).c_str());
    return FAILED;
  }

  int32_t index_n = GetAxisIndexByFormat(trans_info_ptr->dst_in_original_format, N_AXIS_NAME);
  if (index_n < 0 || index_n >= dim_vec_size) {
    REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][CheckNCHWFzTrans] Can not get N index[%d] of format [%s].",
        index_n, ge::TypeUtils::FormatToSerialString(trans_info_ptr->dst_in_original_format).c_str());
    return FAILED;
  }

  FE_LOGD("C value is %ld, N value is %ld, dst in sub format is %ld.",
      dim_vec[index_c], dim_vec[index_n], groups);

  if (dim_vec[index_c] == 1 && dim_vec[index_n] == groups) {
    return SUCCESS;
  }
  return FAILED;
}

inline Status CheckFzNCHWTrans(const TransInfoPtr &trans_info_ptr) {
  FE_CHECK_NOTNULL_WARNLOG(trans_info_ptr);
  /* FFRACTAL_Z  --> NCHW  condition is C==1 && N == groups && groups > 1*/
  int64_t groups = static_cast<int64_t>(trans_info_ptr->src_out_sub_format);
  if (groups <= 1) {
    FE_LOGD("Groups %ld is <= 1.", groups);
    return FAILED;
  }

  if (trans_info_ptr->src_out_data_type != ge::DT_FLOAT16 && trans_info_ptr->src_out_data_type != ge::DT_FLOAT) {
    FE_LOGD("Dtype %s should be fp16/fp32.", DTypeToStr(trans_info_ptr->src_out_data_type).c_str());
    return FAILED;
  }

  vector<int64_t> dim_vec = trans_info_ptr->src_out_original_shape.GetDims();
  if (dim_vec.empty()) {
    FE_LOGD("Dim_vec is empty.");
    return FAILED;
  }
  int32_t dim_vec_size = static_cast<int32_t>(dim_vec.size());
  int32_t index_c = GetAxisIndexByFormat(trans_info_ptr->src_out_original_format, C_AXIS_NAME);
  if (index_c < 0 || index_c >= dim_vec_size) {
    REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][CheckFzNCHWTrans] Can not get C index[%d] of format [%s].",
        index_c, ge::TypeUtils::FormatToSerialString(trans_info_ptr->src_out_original_format).c_str());
    return FAILED;
  }

  int32_t index_n = GetAxisIndexByFormat(trans_info_ptr->src_out_original_format, N_AXIS_NAME);
  if (index_n < 0 || index_n >= dim_vec_size) {
    REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][CheckFzNCHWTrans] Can not get N index[%d] of format [%s].",
        index_n, ge::TypeUtils::FormatToSerialString(trans_info_ptr->src_out_original_format).c_str());
    return FAILED;
  }

  FE_LOGD("C value is %ld, N value is %ld, src out sub format is %ld.",
      dim_vec[index_c], dim_vec[index_n], groups);

  if (dim_vec[index_c] == 1 && dim_vec[index_n] == groups) {
    return SUCCESS;
  }
  return FAILED;
}

const strategy_extra_val_map kStrategyExtraValMap = {
    /* NCHW to Fractal_Z */
    {CalcBaseStrategyId(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z),
        {{kStrategyIdExtThree, std::make_shared<CheckExtValCondFunc>(CheckNCHWFzTrans)}}},
    /* Fractal_Z to NCHW */
    {CalcBaseStrategyId(ge::FORMAT_FRACTAL_Z, ge::FORMAT_NCHW),
        {{kStrategyIdExtThree, std::make_shared<CheckExtValCondFunc>(CheckFzNCHWTrans)}}}
};

inline uint64_t CalcStrategyIdExtraVal(const TransInfoPtr &trans_info_ptr) {
  uint64_t strategy_ext_id = CalcBaseStrategyId(trans_info_ptr->src_out_primary_format,
                                                trans_info_ptr->dst_in_primary_format);
  auto strategy_ext = kStrategyExtraValMap.find(strategy_ext_id);
  if (strategy_ext == kStrategyExtraValMap.end()) {
    FE_LOGD("No strategy extra value, src format %u, dst format %u.",
        trans_info_ptr->src_out_primary_format, trans_info_ptr->dst_in_primary_format);
    return kStrategyIdExtDeft;
  }

  uint64_t strategy_ext_val = kStrategyIdExtDeft;
  std::unordered_map<uint64_t, CheckExtValCondFuncPtr> check_map = strategy_ext->second;
  for (const auto &iter : check_map) {
    if (iter.second == nullptr) {
      continue;
    }
    CheckExtValCondFuncPtr check_ext_val_cond_func = nullptr;
    FE_MAKE_SHARED(check_ext_val_cond_func = iter.second, return kStrategyIdExtDeft);

    if ((*check_ext_val_cond_func)(trans_info_ptr) == FAILED) {
      continue;
    }
    strategy_ext_val = iter.first;
    break;
  }
  return strategy_ext_val;
}

const strategy_id_map STRATEGY_ORIGINAL_FORMAT_FRONT = {

};

const strategy_id_map STRATEGY_ORIGINAL_FORMAT_BACK = {

};

const strategy_id_map STRATEGY_CONSECUTIVE_PRINCIPLE = {
    /* insert all trans nodes from reshape */
    {DEFAULT_LOW_TO_HIGH_STRATEGY, {RESHAPE_INDEX, TRANSPOSE_INDEX, CAST_INDEX, TRANSDATA_INDEX, TRANSDATARNN_INDEX}},
    /* insert all trans nodes from transdata */
    {DEFAULT_HIGH_TO_LOW_STRATEGY, {TRANSDATARNN_INDEX, TRANSDATA_INDEX, CAST_INDEX, TRANSPOSE_INDEX, RESHAPE_INDEX}},

    /* NCHW to NHWC */
    {CalcStrategyId(ge::FORMAT_NCHW, ge::FORMAT_NHWC, kStrategyIdExtDeft), {TRANSPOSE_INDEX}},
    /* NCHW to HWCN */
    {CalcStrategyId(ge::FORMAT_NCHW, ge::FORMAT_HWCN, kStrategyIdExtDeft), {TRANSPOSE_INDEX}},
    /* NHWC to CHWN */
    {CalcStrategyId(ge::FORMAT_NCHW, ge::FORMAT_CHWN, kStrategyIdExtDeft), {TRANSPOSE_INDEX}},
    /* NCHW to NCHW */
    {CalcStrategyId(ge::FORMAT_NCHW, ge::FORMAT_NCHW, kStrategyIdExtDeft), {SQUEEZE_V2_INDEX, UNSQUEEZE_V2_INDEX}},
    /* NCHW to NC1HWC0 */
    {CalcStrategyId(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSDATA_INDEX}},
    /* NCHW to C1HWC0 */
    {CalcStrategyId(ge::FORMAT_NCHW, ge::FORMAT_C1HWC0, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSDATA_INDEX}},
    /* NCHW to Fractal_Z */
    {CalcStrategyId(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSDATA_INDEX, RESHAPE_INDEX}},
    /* NCHW to Fractal_Z_WINO */
    {CalcStrategyId(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_WINO, kStrategyIdExtDeft),
                    {UNSQUEEZE_V2_INDEX, TRANSDATA_INDEX, RESHAPE_INDEX}},
    /* Fractal_Z_WINO to Fractal_Z */
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z_WINO, ge::FORMAT_FRACTAL_Z, kStrategyIdExtDeft),
                    {TRANSDATA_INDEX, RESHAPE_INDEX}},
    /* NCHW to Fractal_Z, NCHW to HWCN first, then to Fractal_Z */
    {CalcStrategyId(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, kStrategyIdExtThree), {UNSQUEEZE_V2_INDEX, TRANSPOSE_INDEX, TRANSDATA_INDEX, RESHAPE_INDEX}},
    /* NCHW to NC1HWC0_C04 */
    {CalcStrategyId(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSDATA_INDEX}},
    /* NCHW to Fractal_Z_C04 */
    {CalcStrategyId(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSDATA_INDEX, RESHAPE_INDEX}},
    /* NCHW to C1HWNCoC0 */
    {CalcStrategyId(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, kStrategyIdExtDeft), {TRANSPOSE_INDEX, TRANSDATA_INDEX}},
    /* NCHW to FRACTAL_NZ */
    {CalcStrategyId(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_NZ, kStrategyIdExtDeft), {REFORMAT_INEDX, TRANSDATA_INDEX}},

    /* NHWC to NCHW */
    {CalcStrategyId(ge::FORMAT_NHWC, ge::FORMAT_NCHW, kStrategyIdExtDeft), {TRANSPOSE_INDEX}},
    /* NHWC to HWCN */
    {CalcStrategyId(ge::FORMAT_NHWC, ge::FORMAT_HWCN, kStrategyIdExtDeft), {TRANSPOSE_INDEX}},
    /* NHWC to NHWC */
    {CalcStrategyId(ge::FORMAT_NHWC, ge::FORMAT_NHWC, kStrategyIdExtDeft), {SQUEEZE_V2_INDEX, UNSQUEEZE_V2_INDEX}},
    /* NHWC to CHWN */
    {CalcStrategyId(ge::FORMAT_NHWC, ge::FORMAT_CHWN, kStrategyIdExtDeft), {TRANSPOSE_INDEX}},
    /* NHWC to NC1HWC0 */
    {CalcStrategyId(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSDATA_INDEX}},
    /* NHWC to Fractal_Z, NHWC to NCHW then to Fractal_Z,
     * Dtype does ot Change */
    {CalcStrategyId(ge::FORMAT_NHWC, ge::FORMAT_FRACTAL_Z, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSPOSE_INDEX, TRANSDATA_INDEX, RESHAPE_INDEX}},
    /* NHWC to NC1HWC0_C04 */
    {CalcStrategyId(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0_C04, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSDATA_INDEX}},
    /* NHWC to Fractal_Z_C04 */
    {CalcStrategyId(ge::FORMAT_NHWC, ge::FORMAT_FRACTAL_Z_C04, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSDATA_INDEX, RESHAPE_INDEX}},
    /* NHWC to C1HWNCoC0 */
    {CalcStrategyId(ge::FORMAT_NHWC, ge::FORMAT_C1HWNCoC0, kStrategyIdExtDeft), {TRANSPOSE_INDEX, TRANSDATA_INDEX}},
    /* NHWC to FRACTAL_NZ */
    {CalcStrategyId(ge::FORMAT_NHWC, ge::FORMAT_FRACTAL_NZ, kStrategyIdExtDeft), {REFORMAT_INEDX, TRANSDATA_INDEX}},

    /* HWCN to NC1HWC0 */
    {CalcStrategyId(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSPOSE_INDEX, TRANSDATA_INDEX}},
    /* HWCN to C1HWC0 */
    {CalcStrategyId(ge::FORMAT_HWCN, ge::FORMAT_C1HWC0, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSDATA_INDEX}},
    /* HWCN to Fractal_Z */
    {CalcStrategyId(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSDATA_INDEX, RESHAPE_INDEX}},
    /* HWCN to Fractal_Z_WINO */
    {CalcStrategyId(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z_WINO, kStrategyIdExtDeft),
                    {UNSQUEEZE_V2_INDEX, TRANSDATA_INDEX, RESHAPE_INDEX}},
    /* HWCN to NC1HWC0_C04 */
    {CalcStrategyId(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSDATA_INDEX}},
    /* HWCN to Fractal_Z_C04 */
    {CalcStrategyId(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z_C04, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSDATA_INDEX, RESHAPE_INDEX}},
    /* HWCN to HWCN */
    {CalcStrategyId(ge::FORMAT_HWCN, ge::FORMAT_HWCN, kStrategyIdExtDeft), {}},
    /* HWCN to C1HWNCoC0 */
    {CalcStrategyId(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, kStrategyIdExtDeft), {TRANSDATA_INDEX}},
    /* HWCN to NCHW */
    {CalcStrategyId(ge::FORMAT_HWCN, ge::FORMAT_NCHW, kStrategyIdExtDeft), {TRANSPOSE_INDEX}},
    /* HWCN to NHWC */
    {CalcStrategyId(ge::FORMAT_HWCN, ge::FORMAT_NHWC, kStrategyIdExtDeft), {TRANSPOSE_INDEX}},
    /* HWCN to FRACTAL_NZ */
    {CalcStrategyId(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_NZ, kStrategyIdExtDeft), {REFORMAT_INEDX, TRANSDATA_INDEX}},
    /* HWCN to FORMAT_FRACTAL_ZN_LSTM */
    {CalcStrategyId(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_ZN_LSTM, kStrategyIdExtDeft), {TRANSDATA_INDEX}},

    /* NC1HWC0 to NCHW */
    {CalcStrategyId(ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, kStrategyIdExtDeft), {TRANSDATA_INDEX, SQUEEZE_V2_INDEX}},
    /* NC1HWC0 to NHWC */
    {CalcStrategyId(ge::FORMAT_NC1HWC0, ge::FORMAT_NHWC, kStrategyIdExtDeft), {TRANSDATA_INDEX, SQUEEZE_V2_INDEX}},
    /* NC1HWC0 to HWCN, to NCHW then to HWCN */
    {CalcStrategyId(ge::FORMAT_NC1HWC0, ge::FORMAT_HWCN, kStrategyIdExtDeft), {TRANSDATA_INDEX, TRANSPOSE_INDEX, SQUEEZE_V2_INDEX}},
    /* NC1HWC0 to NC1HWC0 */
    {CalcStrategyId(ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, kStrategyIdExtDeft), {}},
    /* NC1HWC0 to Fractal_Z,
     * to NCHW first, then to Fractal_Z */
    {CalcStrategyId(ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, kStrategyIdExtDeft), {TRANSDATA_INDEX, TRANSDATA_INDEX, RESHAPE_INDEX}},
    /* NC1HWC0 to FRACTAL_NZ,
     * to NCHW first, then to FRACTAL_NZ */
    {CalcStrategyId(ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_NZ, kStrategyIdExtDeft), {TRANSDATA_INDEX, TRANSDATA_INDEX}},
    /* NC1HWC0 to ND */
    {CalcStrategyId(ge::FORMAT_NC1HWC0, ge::FORMAT_ND, kStrategyIdExtDeft), {FORBIDDEN_INDEX}},

    /* NC1HWC0_C04 to NCHW */
    {CalcStrategyId(ge::FORMAT_NC1HWC0_C04, ge::FORMAT_NCHW, kStrategyIdExtDeft), {TRANSDATA_INDEX, SQUEEZE_V2_INDEX}},
    /* NC1HWC0_C04 to NHWC */
    {CalcStrategyId(ge::FORMAT_NC1HWC0_C04, ge::FORMAT_NHWC, kStrategyIdExtDeft), {TRANSDATA_INDEX, SQUEEZE_V2_INDEX}},
    /* NC1HWC0_C04 to HWCN */
    {CalcStrategyId(ge::FORMAT_NC1HWC0_C04, ge::FORMAT_HWCN, kStrategyIdExtDeft), {TRANSDATA_INDEX, SQUEEZE_V2_INDEX}},
    /* NC1HWC0_C04 to NC1HWC0_C04 */
    {CalcStrategyId(ge::FORMAT_NC1HWC0_C04, ge::FORMAT_NC1HWC0_C04, kStrategyIdExtDeft), {}},

    /* Fractal_Z to HWCN */
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN, kStrategyIdExtDeft), {RESHAPE_INDEX, TRANSDATA_INDEX, SQUEEZE_V2_INDEX}},
    /* Fractal_Z to NCHW */
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z, ge::FORMAT_NCHW, kStrategyIdExtDeft), {RESHAPE_INDEX, TRANSDATA_INDEX, SQUEEZE_V2_INDEX}},
    /* Fractal_Z to NCHW, Fractal_Z to HWCN first, then to NCHW */
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z, ge::FORMAT_NCHW, kStrategyIdExtThree), {RESHAPE_INDEX, TRANSDATA_INDEX, TRANSPOSE_INDEX, SQUEEZE_V2_INDEX}},
    /* Fractal_Z to NHWC, Fractal_Z to NCHW first,
     * then to NHWC */
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z, ge::FORMAT_NHWC, kStrategyIdExtDeft), {RESHAPE_INDEX, TRANSDATA_INDEX, TRANSPOSE_INDEX, SQUEEZE_V2_INDEX}},
    /* Fractal_Z to CHWN, Fractal_Z to NCHW first,
     * then to CHWN Dtype does ot Change */
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z, ge::FORMAT_CHWN, kStrategyIdExtDeft), {RESHAPE_INDEX, TRANSDATA_INDEX, TRANSPOSE_INDEX, SQUEEZE_V2_INDEX}},
    /* Fractal_Z to Fractal_Z */
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, kStrategyIdExtDeft), {RESHAPE_INDEX}},
    /* Fractal_Z to NC1HWC0,
     * to NCHW first, then to NC1HWC0 */
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z, ge::FORMAT_NC1HWC0, kStrategyIdExtDeft), {TRANSDATA_INDEX, TRANSDATA_INDEX}},
    /* Fractal_Z to FRACTAL_NZ,
     * to NCHW first, then to FRACTAL_NZ */
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_NZ, kStrategyIdExtDeft), {TRANSDATA_INDEX, TRANSDATA_INDEX}},
    /* Fractal_Z to Fractal_Z_WINO */
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z_WINO, kStrategyIdExtDeft),
                    {RESHAPE_INDEX, TRANSDATA_INDEX}},

    /* Fractal_Z_C04 to NCHW */
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_NCHW, kStrategyIdExtDeft), {RESHAPE_INDEX, TRANSDATA_INDEX, SQUEEZE_V2_INDEX}},
    /* Fractal_Z_C04 to NHWC */
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_NHWC, kStrategyIdExtDeft), {RESHAPE_INDEX, TRANSDATA_INDEX, SQUEEZE_V2_INDEX}},
    /* Fractal_Z_C04 to HWCN */
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_HWCN, kStrategyIdExtDeft), {RESHAPE_INDEX, TRANSDATA_INDEX, SQUEEZE_V2_INDEX}},
    /* Fractal_Z_C04 to Fractal_Z_C04 */
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, kStrategyIdExtDeft), {RESHAPE_INDEX}},

    /* C1HWNCoC0 to C1HWNCoC0 */
    {CalcStrategyId(ge::FORMAT_C1HWNCoC0, ge::FORMAT_C1HWNCoC0, kStrategyIdExtDeft), {}},
    /* C1HWNCoC0 to HWCN */
    {CalcStrategyId(ge::FORMAT_C1HWNCoC0, ge::FORMAT_HWCN, kStrategyIdExtDeft), {TRANSDATA_INDEX}},
    /* C1HWNCoC0 to NCHW */
    {CalcStrategyId(ge::FORMAT_C1HWNCoC0, ge::FORMAT_NCHW, kStrategyIdExtDeft), {TRANSDATA_INDEX, TRANSPOSE_INDEX}},
    /* C1HWNCoC0 to NHWC */
    {CalcStrategyId(ge::FORMAT_C1HWNCoC0, ge::FORMAT_NHWC, kStrategyIdExtDeft), {TRANSDATA_INDEX, TRANSPOSE_INDEX}},
    /* C1HWNCoC0 to FRACTAL_NZ,
     * to NCHW first, then to FRACTAL_NZ */
    {CalcStrategyId(ge::FORMAT_C1HWNCoC0, ge::FORMAT_FRACTAL_NZ, kStrategyIdExtDeft), {TRANSDATA_INDEX, TRANSDATA_INDEX}},

    /* CHWN to NCHW */
    {CalcStrategyId(ge::FORMAT_CHWN, ge::FORMAT_NCHW, kStrategyIdExtDeft), {TRANSPOSE_INDEX}},
    /* CHWN to NHWC */
    {CalcStrategyId(ge::FORMAT_CHWN, ge::FORMAT_NHWC, kStrategyIdExtDeft), {TRANSPOSE_INDEX}},
    /* CHWN to HWCN */
    {CalcStrategyId(ge::FORMAT_CHWN, ge::FORMAT_HWCN, kStrategyIdExtDeft), {TRANSPOSE_INDEX}},
    /* CHWN to CHWN */
    {CalcStrategyId(ge::FORMAT_CHWN, ge::FORMAT_CHWN, kStrategyIdExtDeft), {}},
    /* CHWN to Fractal_Z, to NCHW first then to fz Dtype does ot Change */
    {CalcStrategyId(ge::FORMAT_CHWN, ge::FORMAT_FRACTAL_Z, kStrategyIdExtDeft), {TRANSPOSE_INDEX, TRANSDATA_INDEX}},
    /* CHWN to NC1HWC0, to NCHW first then to 5D Dtype does ot Change */
    {CalcStrategyId(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, kStrategyIdExtDeft), {TRANSPOSE_INDEX, TRANSDATA_INDEX}},

    /* NHWC to ND */
    {CalcStrategyId(ge::FORMAT_NHWC, ge::FORMAT_ND, kStrategyIdExtDeft), {}},
    /* NCHW to ND */
    {CalcStrategyId(ge::FORMAT_NCHW, ge::FORMAT_ND, kStrategyIdExtDeft), {}},
    /* HWCN to ND */
    {CalcStrategyId(ge::FORMAT_HWCN, ge::FORMAT_ND, kStrategyIdExtDeft), {}},
    /* NCDHW to ND */
    {CalcStrategyId(ge::FORMAT_NCDHW, ge::FORMAT_ND, kStrategyIdExtDeft), {}},
    /* NDHWC to ND */
    {CalcStrategyId(ge::FORMAT_NDHWC, ge::FORMAT_ND, kStrategyIdExtDeft), {}},
    /* DHWCN to ND */
    {CalcStrategyId(ge::FORMAT_DHWCN, ge::FORMAT_ND, kStrategyIdExtDeft), {}},
    /* DHWNC to ND */
    {CalcStrategyId(ge::FORMAT_DHWNC, ge::FORMAT_ND, kStrategyIdExtDeft), {}},
    /* FRACTAL_NZ to ND */
    {CalcStrategyId(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, kStrategyIdExtDeft), {TRANSDATA_INDEX}},

    /* ND to ND */
    {CalcStrategyId(ge::FORMAT_ND, ge::FORMAT_ND, kStrategyIdExtDeft), {}},
    /* ND to NCHW */
    {CalcStrategyId(ge::FORMAT_ND, ge::FORMAT_NCHW, kStrategyIdExtDeft), {}},
    /* ND to NHWC */
    {CalcStrategyId(ge::FORMAT_ND, ge::FORMAT_NHWC, kStrategyIdExtDeft), {}},
    /* ND to HWCN */
    {CalcStrategyId(ge::FORMAT_ND, ge::FORMAT_HWCN, kStrategyIdExtDeft), {}},
    /* ND to NCDHW */
    {CalcStrategyId(ge::FORMAT_ND, ge::FORMAT_NCDHW, kStrategyIdExtDeft), {}},
    /* ND to NDHWC */
    {CalcStrategyId(ge::FORMAT_ND, ge::FORMAT_NDHWC, kStrategyIdExtDeft), {}},
    /* ND to DHWCN */
    {CalcStrategyId(ge::FORMAT_ND, ge::FORMAT_DHWCN, kStrategyIdExtDeft), {}},
    /* ND to DHWNC */
    {CalcStrategyId(ge::FORMAT_ND, ge::FORMAT_DHWNC, kStrategyIdExtDeft), {}},
    /* ND to FRACTAL_NZ */
    {CalcStrategyId(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, kStrategyIdExtDeft), {TRANSDATA_INDEX}},
    /* ND to FRACTAL_Z */
    {CalcStrategyId(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, kStrategyIdExtDeft), {TRANSDATA_INDEX}},
    /* ND to NC1HWC0 */
    {CalcStrategyId(ge::FORMAT_ND, ge::FORMAT_NC1HWC0, kStrategyIdExtDeft), {FORBIDDEN_INDEX}},

    /* ND to FRACTAL_ZN_RNN, Dtype does ot Change */
    {CalcStrategyId(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, kStrategyIdExtDeft), {TRANSDATARNN_INDEX}},
    /* ND to ND_RNN_BIAS, Dtype does ot Change */
    {CalcStrategyId(ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, kStrategyIdExtDeft), {TRANSDATARNN_INDEX}},
    /* FRACTAL_ZN_RNN to ND, Dtype does ot Change */
    {CalcStrategyId(ge::FORMAT_FRACTAL_ZN_RNN, ge::FORMAT_ND, kStrategyIdExtDeft), {TRANSDATARNN_INDEX}},
    /* FRACTAL_ZN_RNN to ND, Dtype does ot Change */
    {CalcStrategyId(ge::FORMAT_FRACTAL_ZN_RNN, ge::FORMAT_NHWC, kStrategyIdExtDeft), {TRANSDATARNN_INDEX}},
    /* ND_RNN_BIAS to ND, Dtype does ot Change */
    {CalcStrategyId(ge::FORMAT_ND_RNN_BIAS, ge::FORMAT_ND, kStrategyIdExtDeft), {TRANSDATARNN_INDEX}},

    /* FRACTAL_NZ to NCHW */
    {CalcStrategyId(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NCHW, kStrategyIdExtDeft), {TRANSDATA_INDEX, REFORMAT_INEDX}},
    /* FRACTAL_NZ to NHWC */
    {CalcStrategyId(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NHWC, kStrategyIdExtDeft), {TRANSDATA_INDEX, REFORMAT_INEDX}},
    /* FRACTAL_NZ to HWCN */
    {CalcStrategyId(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_HWCN, kStrategyIdExtDeft), {TRANSDATA_INDEX, REFORMAT_INEDX}},
    /* FRACTAL_NZ to NCDHW */
    {CalcStrategyId(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NCDHW, kStrategyIdExtDeft), {TRANSDATA_INDEX, REFORMAT_INEDX}},
    /* FRACTAL_NZ to NDHWC */
    {CalcStrategyId(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NDHWC, kStrategyIdExtDeft), {TRANSDATA_INDEX, REFORMAT_INEDX}},

    /* FRACTAL_NZ to FRACTAL_Z
     * to original format first, then to FRACTAL_Z */
    {CalcStrategyId(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_Z, kStrategyIdExtDeft), {TRANSDATA_INDEX, TRANSPOSE_INDEX, TRANSDATA_INDEX}},
    /* FRACTAL_NZ to NC1HWC0
     * to original format first, then to NC1HWC0 */
    {CalcStrategyId(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NC1HWC0, kStrategyIdExtDeft), {TRANSDATA_INDEX, TRANSPOSE_INDEX, TRANSDATA_INDEX}},
    /* FRACTAL_NZ to C1HWNCoC0
     * to original format first, then to HWCN, then to C1HWNCoC0 */
    {CalcStrategyId(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_C1HWNCoC0, kStrategyIdExtDeft), {TRANSDATA_INDEX, TRANSPOSE_INDEX, TRANSDATA_INDEX}},
    /* FRACTAL_NZ to FRACTAL_NZ */
    {CalcStrategyId(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, kStrategyIdExtDeft), {}},

    /*************************************NCDHW*************************************/
    // NCDHW -> NDC1HWC0
    {CalcStrategyId(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSDATA_INDEX}},
    // NCDHW -> FORMAT_FRACTAL_Z_3D
    {CalcStrategyId(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, kStrategyIdExtDeft), {TRANSDATA_INDEX, RESHAPE_INDEX}},
    // NCDHW -> NCDHW
    {CalcStrategyId(ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, kStrategyIdExtDeft), {}},
    // NCDHW -> NDHWC
    {CalcStrategyId(ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSPOSE_INDEX, SQUEEZE_V2_INDEX}},
    // NCDHW -> DHWCN
    {CalcStrategyId(ge::FORMAT_NCDHW, ge::FORMAT_DHWCN, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSPOSE_INDEX, SQUEEZE_V2_INDEX}},
    // NCDHW -> DHWNC
    {CalcStrategyId(ge::FORMAT_NCDHW, ge::FORMAT_DHWNC, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSPOSE_INDEX, SQUEEZE_V2_INDEX}},
    // NCDHW -> FORMAT_FRACTAL_NZ
    {CalcStrategyId(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_NZ, kStrategyIdExtDeft), {REFORMAT_INEDX, TRANSDATA_INDEX}},

    /*************************************NDHWC*************************************/
    // NDHWC -> NDC1HWC0
    {CalcStrategyId(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSDATA_INDEX}},
    // NDHWC -> FRACTAL_NZ
    {CalcStrategyId(ge::FORMAT_NDHWC, ge::FORMAT_FRACTAL_NZ, kStrategyIdExtDeft), {REFORMAT_INEDX, TRANSDATA_INDEX}},
    // NDHWC -> FORMAT_FRACTAL_Z_3D
    {CalcStrategyId(ge::FORMAT_NDHWC, ge::FORMAT_FRACTAL_Z_3D, kStrategyIdExtDeft), {TRANSDATA_INDEX, RESHAPE_INDEX}},
    // NDHWC -> NDHWC
    {CalcStrategyId(ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, kStrategyIdExtDeft), {}},
    // NDHWC -> NCDHW
    {CalcStrategyId(ge::FORMAT_NDHWC, ge::FORMAT_NCDHW, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSPOSE_INDEX, SQUEEZE_V2_INDEX}},
    // NDHWC -> DHWCN
    {CalcStrategyId(ge::FORMAT_NDHWC, ge::FORMAT_DHWCN, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSPOSE_INDEX, SQUEEZE_V2_INDEX}},
    // NDHWC -> DHWNC
    {CalcStrategyId(ge::FORMAT_NDHWC, ge::FORMAT_DHWNC, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSPOSE_INDEX, SQUEEZE_V2_INDEX}},

    /*************************************NDC1HWC0*************************************/
    // NDC1HWC0 -> NDHWC
    {CalcStrategyId(ge::FORMAT_NDC1HWC0, ge::FORMAT_NDHWC, kStrategyIdExtDeft), {TRANSDATA_INDEX, SQUEEZE_V2_INDEX}},
    // NDC1HWC0 -> NCDHW
    {CalcStrategyId(ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, kStrategyIdExtDeft), {TRANSDATA_INDEX, SQUEEZE_V2_INDEX}},
    // NDC1HWC0 -> NDC1HWC0
    {CalcStrategyId(ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, kStrategyIdExtDeft), {}},

    /*************************************DHWCN*************************************/
    // DHWCN - > FRACTAL_Z_3D
    {CalcStrategyId(ge::FORMAT_DHWCN, ge::FORMAT_FRACTAL_Z_3D, kStrategyIdExtDeft), {TRANSDATA_INDEX, RESHAPE_INDEX}},
    // DHWCN -> DHWCN
    {CalcStrategyId(ge::FORMAT_DHWCN, ge::FORMAT_DHWCN, kStrategyIdExtDeft), {}},
    // DHWCN -> NCDHW
    {CalcStrategyId(ge::FORMAT_DHWCN, ge::FORMAT_NCDHW, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSPOSE_INDEX, SQUEEZE_V2_INDEX}},
    // DHWCN -> NDHWC
    {CalcStrategyId(ge::FORMAT_DHWCN, ge::FORMAT_NDHWC, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSPOSE_INDEX, SQUEEZE_V2_INDEX}},
    // DHWCN -> DHWNC
    {CalcStrategyId(ge::FORMAT_DHWCN, ge::FORMAT_DHWNC, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSPOSE_INDEX, SQUEEZE_V2_INDEX}},

    /*************************************DHWNC*************************************/
    // DHWNC -> FORMAT_FRACTAL_Z_3D_TRANSPOSE
    {CalcStrategyId(ge::FORMAT_DHWNC, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, kStrategyIdExtDeft), {TRANSDATA_INDEX}},
    // DHWNC -> DHWNC
    {CalcStrategyId(ge::FORMAT_DHWNC, ge::FORMAT_DHWNC, kStrategyIdExtDeft), {}},
    // DHWNC -> NCDHW
    {CalcStrategyId(ge::FORMAT_DHWNC, ge::FORMAT_NCDHW, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSPOSE_INDEX, SQUEEZE_V2_INDEX}},
    // DHWNC -> DHWCN
    {CalcStrategyId(ge::FORMAT_DHWNC, ge::FORMAT_NDHWC, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSPOSE_INDEX, SQUEEZE_V2_INDEX}},
    // DHWNC -> DHWCN
    {CalcStrategyId(ge::FORMAT_DHWNC, ge::FORMAT_DHWCN, kStrategyIdExtDeft), {UNSQUEEZE_V2_INDEX, TRANSPOSE_INDEX, SQUEEZE_V2_INDEX}},

    /*************************************FRACTAL_Z_3D*************************************/
    // FRACTAL_Z_3D -> NDHWC
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NDHWC, kStrategyIdExtDeft), {RESHAPE_INDEX, TRANSDATA_INDEX}},
    // FRACTAL_Z_3D -> NCDHW
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, kStrategyIdExtDeft), {RESHAPE_INDEX, TRANSDATA_INDEX}},
    // FRACTAL_Z_3D -> DHWCN
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_DHWCN, kStrategyIdExtDeft), {RESHAPE_INDEX, TRANSDATA_INDEX}},
    // FRACTAL_Z_3D -> FRACTAL_Z_3D
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D, kStrategyIdExtDeft), {}},

    /*************************************FRACTAL_Z_3D_TRANSPOSE***********************************/
    /* FRACTAL_Z_3D to DHWNC */
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, ge::FORMAT_DHWNC, kStrategyIdExtDeft), {TRANSDATA_INDEX}},
    /* FRACTAL_Z_3D to DHWCN */
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, kStrategyIdExtDeft), {}},

    /*************************************FRACTAL_ZN_LSTM***********************************/
    /* FORMAT_FRACTAL_ZN_LSTM to HWCN */
    {CalcStrategyId(ge::FORMAT_FRACTAL_ZN_LSTM, ge::FORMAT_HWCN, kStrategyIdExtDeft), {TRANSDATA_INDEX}},
    {CalcStrategyId(ge::FORMAT_FRACTAL_ZN_LSTM, ge::FORMAT_FRACTAL_ZN_LSTM, kStrategyIdExtDeft), {}},
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_MANAGER_TRANS_NODE_INSERTION_INSERT_TRANS_NODE_STRATEGY_H_
