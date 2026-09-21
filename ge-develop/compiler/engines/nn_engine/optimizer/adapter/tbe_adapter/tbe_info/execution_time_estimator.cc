/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter/tbe_adapter/tbe_info/execution_time_estimator.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"

namespace fe {
const std::unordered_set<string> kConvWhiteList {
  CONV2D, kConv2DCompress, DECONVOLUTION, DEPTHWISECONV2D, kFullyConnection
};

const std::unordered_set<string> kVectorWhiteList {
  POOLING, ASCEND_QUANT, CONCATD, BNINFERENCED, SCALE
};

const std::unordered_set<string> kMatmulWhiteList {
  "MatMul", "MatMulV2", "BatchMatMulV2", "BatchMatMul"
};

Status ExecutionTimeEstimator::GetExecTime(PlatFormInfos &platform_info, const ge::OpDesc &op_desc,
                                           const OpKernelInfoPtr &op_kernel_info_ptr,
                                           uint64_t &exec_time) {
  if (UnknownShapeUtils::IsUnknownShapeOp(op_desc)) {
    return FAILED;
  }

  string op_type = op_desc.GetType();
  OpPattern pattern = OP_PATTERN_OP_KERNEL;
  if (op_kernel_info_ptr != nullptr) {
    pattern = op_kernel_info_ptr->GetOpPattern();
  }

  if (pattern == OP_PATTERN_FORMAT_AGNOSTIC || pattern == OP_PATTERN_BROADCAST ||
      pattern == OP_PATTERN_BROADCAST_ENHANCED || kVectorWhiteList.count(op_type) != 0) {
    return VectorEstimator::EstimateTime(platform_info, op_desc, exec_time);
  } else if (kConvWhiteList.count(op_type) != 0) {
    return ConvEstimator::EstimateTime(platform_info, op_desc, exec_time);
  } else if (kMatmulWhiteList.count(op_type) != 0) {
    return MatMulEstimator::EstimateTime(platform_info, op_desc, exec_time);
  } else {
    FE_LOGD("Cannot find estimation function of op %s.", op_desc.GetName().c_str());
    return FAILED;
  }
}
}
