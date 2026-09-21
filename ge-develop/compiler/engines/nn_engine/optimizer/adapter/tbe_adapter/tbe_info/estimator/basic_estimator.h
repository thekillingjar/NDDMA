/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_INFO_ESTIMATOR_BASIC_ESTIMATOR_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_INFO_ESTIMATOR_BASIC_ESTIMATOR_H_
#include "platform/platform_info.h"
#include "graph/op_desc.h"
#include "common/fe_log.h"

namespace fe {
extern const std::string kAiCoreSpecLbl;

class BasicEstimator {
 public:
  BasicEstimator();

  virtual ~BasicEstimator();
  static inline uint64_t GetUintParam(PlatFormInfos &platform_info, const std::string &label, const std::string &key) {
    std::string val_str;
    (void)platform_info.GetPlatformRes(label, key, val_str);
    return static_cast<uint64_t>(std::atoi(val_str.c_str()));
  }

  static uint64_t CalcTimeByCycle(PlatFormInfos &platform_info, uint64_t cycle);
};
}
#endif // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_INFO_ESTIMATOR_BASIC_ESTIMATOR_H_
