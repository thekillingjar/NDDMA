/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/weight_compress_utils.h"
#include "common/fe_inner_attr_define.h"
#include "common/fe_utils.h"
#include "common/fe_type_utils.h"
#include "platform/platform_info.h"
#include "common/format/axis_util.h"
#include "common/op_info_common.h"
#include "common/configuration.h"

namespace fe {
namespace {
constexpr int64_t WEIGHT_INDEX_SHAPE = 4;
constexpr int64_t WEIGHT_SPECIAL_DIVISION = 2;
constexpr uint8_t ENABLE_SPARSITY = 1;
}

bool ReadWeightConfig() {
  FE_LOGD("Begin to read weight config from platform info.");
  PlatformInfo platform_info;
  OptionalInfo opti_compilation_info;
  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, opti_compilation_info) !=
      SUCCESS) {
    FE_LOGD("ReadWeightConfig failed to get platform info using current soc version.");
    return false;
  }
  FE_LOGD("ReadWeightConfig platform sparsity flag = %u.", platform_info.ai_core_spec.sparsity);
  if (platform_info.ai_core_spec.sparsity == ENABLE_SPARSITY) {
    return true;
  }
  else {
    return false;
  }
}

WEIGHCOMPRESSINNERFLAG JudgeIsSparsityFlag() {
  bool enable_sparsity = Configuration::Instance(AI_CORE_NAME).IsEnableSparseMatrixWeight();
  bool enable_config = ReadWeightConfig();
  FE_LOGD("matrixweight flag = %d, and config flag = %d.", enable_sparsity, enable_config);
  if (enable_sparsity && enable_config) {
    return WEIGHCOMPRESSINNERFLAG::FOUR_TO_TWO_FLAG;
  }
  return WEIGHCOMPRESSINNERFLAG::DISABLE_COMPRESS_FLAG;
}

ge::GeShape ReshapeConvWeight(const ge::GeShape &src_shape,
                              const ge::Format &src_format) {
  std::string reshape_type = "";
  ge::GeShape dst_shape;
  ExpandDimension(src_format, src_format,
                  reshape_type, src_shape, dst_shape);
  if (src_shape.GetDimNum() != dst_shape.GetDimNum()) {
    FE_LOGD("Cube weight tensor needs reshaping.");
    return dst_shape;
  }
  return src_shape;
}

Status TranseWeight2SparsityFZShapeAndFormat(const ge::OpDescPtr &conv_desc,
                                             const ge::GeShape &src_shape,
                                             const ge::Format &src_format,
                                             const ge::DataType &src_type,
                                             ge::GeShape &dst_shape,
                                             const std::pair<bool, bool> &flag) {
  bool need_divisonc0 = flag.first;
  bool origin_fzflag = flag.second;
  std::vector<int64_t> nd_value;
  int64_t c0 = GetC0ValByOpDescAndDtype(conv_desc, src_type);
  std::vector<int64_t> axis_value;
  std::vector<int64_t> new_dim_vec;
  for (uint32_t i = 0; i < AXIS_BOTTOM; i++) {
    axis_value.push_back(1);
  }
  auto reshape_src_shape = ReshapeConvWeight(src_shape, src_format);
  Status status = AxisUtil::GetAxisValueByOriginFormat(src_format,
                                                       reshape_src_shape.GetDims(), c0,
                                                       axis_value, nd_value);
  if (status != SUCCESS) {
    return status;
  }
  bool has_unknown_shape = axis_value[AXIS_W] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_H] == UNKNOWN_SHAPE_VALUE ||
                           axis_value[AXIS_C1] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_G] == UNKNOWN_SHAPE_VALUE;
  if (has_unknown_shape) {
    return FAILED;
  }
  int64_t axis_n_val = axis_value[AXIS_N];
  int64_t axis_c_val = axis_value[AXIS_C];
  int64_t axis_c1_val = DivisionCeiling(axis_c_val, axis_value[AXIS_C0]);
  if (!origin_fzflag) {
    axis_c1_val = DivisionCeiling(axis_c1_val, WEIGHT_SPECIAL_DIVISION);
  }
  FE_MUL_OVERFLOW(axis_c1_val, axis_value[AXIS_H], axis_c1_val);
  FE_MUL_OVERFLOW(axis_c1_val, axis_value[AXIS_W], axis_c1_val);
  new_dim_vec.push_back(axis_c1_val);
  new_dim_vec.push_back(DivisionCeiling(axis_n_val, NI));
  new_dim_vec.push_back(NI);
  if (need_divisonc0) {
    int64_t axis_c0_index_val = DivisionCeiling(axis_value[AXIS_C0], WEIGHT_INDEX_SHAPE);
    new_dim_vec.push_back(axis_c0_index_val);
  } else {
    new_dim_vec.push_back(axis_value[AXIS_C0]);
  }
  dst_shape = ge::GeShape(new_dim_vec);
  return SUCCESS;
}

Status SetSparsityCompressWeightShape(const ge::OpDescPtr &conv_desc, const ge::GeTensorDescPtr &tensor_desc_ptr,
                                      const std::pair<bool, bool> &flag) {
   // copy the weight tensor desc
  ge::GeShape dst_shape;
  ge::GeTensorDescPtr src_weight_tensor_desc = conv_desc->MutableInputDesc(TENSOR_INDEX_FILTER_COMPRESS);
  FE_CHECK(src_weight_tensor_desc == nullptr, FE_LOGW("Null pointer is unexpected."), return FAILED);
  Status status = TranseWeight2SparsityFZShapeAndFormat(conv_desc, src_weight_tensor_desc->GetOriginShape(),
                                                        src_weight_tensor_desc->GetOriginFormat(),
                                                        src_weight_tensor_desc->GetOriginDataType(),
                                                        dst_shape,
                                                        flag);
  if (status != SUCCESS) {
    return FAILED;
  }
  FE_CHECK(tensor_desc_ptr == nullptr,
           REPORT_FE_ERROR("[SubGraphOpt][Compile][UpdCmprsOp]The compress index tensor of compress node[%s] is null.",
                           conv_desc->GetName().c_str()),
           return FAILED);
  tensor_desc_ptr->SetShape(dst_shape);
  tensor_desc_ptr->SetFormat(ge::FORMAT_FRACTAL_Z);
  tensor_desc_ptr->SetOriginShape(src_weight_tensor_desc->GetOriginShape());
  tensor_desc_ptr->SetOriginFormat(src_weight_tensor_desc->GetOriginFormat());

  FE_LOGD("success in setting _compress flag attribute weight_sparse_4_2 on node [%s].", conv_desc->GetName().c_str());
  return SUCCESS;
}

Status RefreshConvShapeForSpasity(const ge::OpDescPtr &conv_desc) {
  ge::GeTensorDescPtr conv2d_compress_weight_desc = conv_desc->MutableInputDesc(TENSOR_INDEX_FILTER_COMPRESS);
  Status status = SetSparsityCompressWeightShape(conv_desc, conv2d_compress_weight_desc,
                                                 std::make_pair(false, false));
  if (status != SUCCESS) {
    return status;
  }
  ge::GeTensorDescPtr conv2d_compress_index_desc = conv_desc->MutableInputDesc(TENSOR_INDEX_COMPRESS_INDEX);
  status = SetSparsityCompressWeightShape(conv_desc, conv2d_compress_index_desc,
                                          std::make_pair(true, false));
  return status;
}

Status RefreshCompressShapeForSpasity(const ge::OpDescPtr &conv_desc, const ge::OpDescPtr &compress_opdesc) {
  FE_CHECK_NOTNULL(compress_opdesc);
  (void)ge::AttrUtils::SetStr(compress_opdesc, ATTR_NAME_COMPRESS_TYPE_FLAG, kWeightSparseFourToTwo);
  ge::GeTensorDescPtr compress_weight_desc = compress_opdesc->MutableInputDesc("weight");
  Status status = SetSparsityCompressWeightShape(conv_desc, compress_weight_desc,
                                                 std::make_pair(false, true));

  return status;
}
}
