/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_INFO_EXECUTION_TIME_ESTIMATOR_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_INFO_EXECUTION_TIME_ESTIMATOR_H_
#include <memory>
#include <string>

#include "adapter/tbe_adapter/tbe_info/estimator/conv_estimator.h"
#include "adapter/tbe_adapter/tbe_info/estimator/vector_estimator.h"
#include "adapter/tbe_adapter/tbe_info/estimator/matmul_estimator.h"
#include "common/util/op_info_util.h"
#include "ops_store/op_kernel_info.h"

namespace fe {
class ExecutionTimeEstimator {
 public:
  static Status GetExecTime(PlatFormInfos &platform_info, const ge::OpDesc& op_desc,
                            const OpKernelInfoPtr &op_kernel_info_ptr,
                            uint64_t &exec_time);
};
}
#endif // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_INFO_EXECUTION_TIME_ESTIMATOR_H_
