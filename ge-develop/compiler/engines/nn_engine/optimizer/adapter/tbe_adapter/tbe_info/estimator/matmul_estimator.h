/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_INFO_ESTIMATOR_MATMUL_ESTIMATOR_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_INFO_ESTIMATOR_MATMUL_ESTIMATOR_H_
#include "adapter/tbe_adapter/tbe_info/estimator/basic_estimator.h"

namespace fe {
class MatMulEstimator {
 public:
  static Status EstimateTime(PlatFormInfos &platform_info, const ge::OpDesc &op_desc,
                             uint64_t &exec_time);
};
}
#endif // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_INFO_ESTIMATOR_CONV_ESTIMATOR_H_
