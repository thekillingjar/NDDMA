/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_FUSION_COMMON_FUSION_CONFIG_INFO_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_FUSION_COMMON_FUSION_CONFIG_INFO_H_

#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
class FusionConfigInfo {
public:
  FusionConfigInfo(const FusionConfigInfo &) = delete;
  FusionConfigInfo &operator=(const FusionConfigInfo &) = delete;
  static FusionConfigInfo& Instance();
  Status Initialize();
  Status Finalize();
  bool IsEnableNetworkAnalysis() const;
private:
  FusionConfigInfo() = default;
  ~FusionConfigInfo() = default;
  void InitEnvParam();
  bool is_init_ = false;
  bool is_enable_network_analysis_ = false;
};
}
#endif  // INC_REGISTER_GRAPH_OPTIMIZER_FUSION_COMMON_FUSION_CONFIG_INFO_H_
