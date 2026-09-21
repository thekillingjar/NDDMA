/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_TASKDOWN_COMMON_H_
#define INC_FRAMEWORK_COMMON_TASKDOWN_COMMON_H_

#include "runtime/rt.h"
#include "ge/framework/common/taskdown_common.h"

namespace ge {

constexpr int32_t CC_FUSION_OP_MAX = 32;

enum class ccStatus_t : uint32_t {
  CC_STATUS_SUCCESS = 0,         /**< succ */
  CC_STATUS_NOT_INITIALIZED = 1, /**< not init */
  CC_STATUS_ALLOC_FAILED = 2,    /**< alloc mem failed */
  CC_STATUS_BAD_PARAM = 3,       /**< para check failed */
  CC_STATUS_INTERNAL_ERROR = 4,  /**< internal error */
  CC_STATUS_KERNEL_ERROR = 5,    /**< kernel error */
  CC_STATUS_RUNTIME_ERROR = 6,   /**< runtime error */
  CC_STATUS_NOT_SUPPORTED = 7,   /**< unsupport error */
  CC_STATUS_INVALID_VALUE = 7,   /**< invalid value error for blas*/
  CC_STATUS_RESERVED      = 8,   /**< just for check */
};

using ccOpContext = struct tagOpContext {
  ccKernelType kernelType;
  uint32_t opId;
  uint32_t kernelFuncId;
  uint32_t opIndex;
  uint32_t opCount;
  uint32_t opIndex2[CC_FUSION_OP_MAX];
  bool isFlowtable;
  uint16_t *argsOffset;
  uint32_t argsCount;
  uint64_t genDataBaseAddr;
  uint64_t genDataBaseSize;
  uint64_t genWeightBaseAddr;
  uint64_t genWeightBaseSize;
  uint64_t genVariableBaseAddr;
  uint64_t genVariableBaseSize;
  uint64_t l2ctrlSize;
};

enum class tagOpTensorFormat : uint32_t {
  OP_TENSOR_FORMAT_NC1HWC0 = 0,
  OP_TENSOR_FORMAT_ND,
  OP_TENSOR_FORMAT_RESERVED
};

enum class tagOpDataType : uint32_t {
  OP_DATA_FLOAT = 0,            /**< float type */
  OP_DATA_HALF,                 /**< fp16 type */
  OP_DATA_INT8,                 /**< int8 type */
  OP_DATA_INT32,                /**< int32 type */
  OP_DATA_UINT8,                /**< uint8 type */
  OP_DATA_HALF_UINT16_PROPOSAL, /**< mixed type for proposal */
  OP_DATA_RESERVED
};

// AICPU Tensor
using ccAICPUTensor = struct tagOpTensor {
  // real dim info
  tagOpTensorFormat format;
  tagOpDataType data_type;
  int32_t dim_cnt;
  int32_t mm;
  int32_t dim[8];
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_TASKDOWN_COMMON_H_
