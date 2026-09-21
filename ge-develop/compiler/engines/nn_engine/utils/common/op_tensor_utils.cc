/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/op_tensor_utils.h"

#include <cmath>
#include "common/comm_error_codes.h"
#include "common/fe_log.h"
#include "common/math_util.h"
#include "common/constants_define.h"
#include "common/string_utils.h"
#include "common/platform_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "graph/ge_context.h"

namespace fe {
namespace {
// The align size of data memory
const uint32_t DATA_MEMORY_ALIGN_SIZE = 32;
const uint32_t DATA_MEMORY_ALIGN_SIZE_FOR_V350 = 16;
}

Status OpTensorUtils::VerifyTensor(const std::vector<int64_t> &dims, const ge::DataType &data_type) {
  if (data_type < ge::DT_FLOAT || data_type == ge::DT_UNDEFINED || data_type >= ge::DT_MAX) {
    REPORT_FE_ERROR("[GraphOpt][OptWholeGph][VerifyTensor] The data type of this tensor is invalid.");
    return TENSOR_DATATYPE_INVALID;
  }

  if (!dims.empty()) {
    for (const int64_t &dim : dims) {
      if (dim < 0) {
        REPORT_FE_ERROR("[GraphOpt][OptWholeGph][VerfyTensor] The dim value[%ld] is invalid.", dim);
        return DIM_VALUE_INVALID;
      }
    }
  }

  return SUCCESS;
}

Status OpTensorUtils::ArrayMultiplyInt64WithVerify(const std::vector<int64_t> &dims, int64_t &result) {
  result = 1;  // Initial value
  if (dims.empty()) {
    return SUCCESS;
  }
  for (const int64_t &num : dims) {
    if (num == 0) {
      FE_LOGD("num = 0; return 1;");
      result = 1;
      return SUCCESS;
    }
    FE_MUL_OVERFLOW(result, num, result);
  }
  return SUCCESS;
}

Status OpTensorUtils::CalibrateTensorSize(int64_t &tensor_size) {
  if (PlatformUtils::Instance().GetIsaArchVersion() != ISAArchVersion::EN_ISA_ARCH_V350) {
    tensor_size = (tensor_size + DATA_MEMORY_ALIGN_SIZE - 1) / DATA_MEMORY_ALIGN_SIZE;
    FE_MUL_OVERFLOW(tensor_size, DATA_MEMORY_ALIGN_SIZE, tensor_size);
    FE_ADD_OVERFLOW(tensor_size, DATA_MEMORY_ALIGN_SIZE, tensor_size);
  } else {
    tensor_size = (tensor_size + DATA_MEMORY_ALIGN_SIZE_FOR_V350 - 1) / DATA_MEMORY_ALIGN_SIZE_FOR_V350;
    FE_MUL_OVERFLOW(tensor_size, DATA_MEMORY_ALIGN_SIZE_FOR_V350, tensor_size);
    FE_ADD_OVERFLOW(tensor_size, DATA_MEMORY_ALIGN_SIZE_FOR_V350, tensor_size);
  }
  return SUCCESS;
}

Status OpTensorUtils::CalcTensorSize(const std::vector<int64_t> &dims, const ge::DataType &data_type,
                                     int32_t output_real_calc_flag, int64_t &tensor_size) {
  // verify the tensor
  if (VerifyTensor(dims, data_type) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][OptWholeGph][CalcTensorSize] Failed to verify this tensor.");
    return FAILED;
  }

  int64_t element_cnt;
  if (ArrayMultiplyInt64WithVerify(dims, element_cnt) != SUCCESS) {
    tensor_size = INT64_MAX;
    FE_LOGW("Tensor size is larger than the upper bound of int64, using INT64_MAX as the tensor size.");
    return SUCCESS;
  }

  tensor_size = ge::GetSizeInBytes(element_cnt, data_type);
  if (tensor_size < 0) {
    REPORT_FE_ERROR("GetSizeInBytes failed!");
    return FAILED;
  }
  if (!output_real_calc_flag) {
    return CalibrateTensorSize(tensor_size);
  }

  return SUCCESS;
}

Status OpTensorUtils::CalcTensorSize(const ge::GeTensorDesc &tensor_desc,
                                     const int32_t output_real_calc_flag, int64_t &tensor_size) {
  std::vector<int64_t> dims = tensor_desc.GetShape().GetDims();
  ge::DataType data_type = tensor_desc.GetDataType();
  return CalcTensorSize(dims, data_type, output_real_calc_flag, tensor_size);
}


bool OpTensorUtils::IsStaticReuseBinaryOp(const ge::OpDescPtr &op_desc) {
  bool stc_dyn_soft_sync = false;
  (void)ge::AttrUtils::GetBool(op_desc, kStaticToDynamicSoftSyncOp, stc_dyn_soft_sync);
  if (stc_dyn_soft_sync) {
    return true;
  }
  if (UnknownShapeUtils::IsUnknownShapeOp(*op_desc)) {
    FE_LOGD("Node [%s] has an unknown shape", op_desc->GetName().c_str());
    return false;
  }
  bool is_unknown_graph = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_OWNER_GRAPH_IS_UNKNOWN, is_unknown_graph);
  if (is_unknown_graph) {
    FE_LOGD("Node:%s is unknown graph", op_desc->GetName().c_str());
    return false;
  }
  bool stc_tiling_depend = false;
  (void)ge::AttrUtils::GetBool(op_desc, kDynamicTilingDependOp, stc_tiling_depend);
  if (stc_tiling_depend) {
    return false;
  }
  return op_desc->HasAttr(COMPILE_INFO_JSON);
}

bool OpTensorUtils::IsFuzzBuildOp(const ge::OpDesc &op_desc) {
  std::string build_mode;
  bool support_dyn_shape = false;
  if ((ge::GetContext().GetOption("ge.shape_generalized_build_mode", build_mode) == ge::GRAPH_SUCCESS) &&
    (build_mode == SHAPE_GENERALIZED) &&
    ge::AttrUtils::GetBool(op_desc, ATTR_NAME_SUPPORT_DYNAMIC_SHAPE, support_dyn_shape) && support_dyn_shape) {
    return true;
  } else {
    return false;
  }
}

}  // namespace fe
