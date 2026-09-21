/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef AUTOFUSE_AUTOFUSER_H
#define AUTOFUSE_AUTOFUSER_H

#include "ge_common/ge_api_types.h"
#include "graph/compute_graph.h"
#include "ascir.h"
#include "common/autofuse_base_type.h"
#include "autofuse_frame/autofuse_frames.h"

namespace ge {
struct AutofuserOptions {
  AutoFuseFwkType fwk_type = AutoFuseFwkType::kGe;
};

class Autofuser {
 public:
  // 调用侧是非const类型
  explicit Autofuser(AutofuserOptions &options, CounterPtr counter = nullptr);
  explicit Autofuser(const AutofuserOptions &options) = delete;
  Autofuser() = delete;
  ge::Status Fuse(const ge::ComputeGraphPtr &graph) const;
  ge::Status FuseLite(const ge::ComputeGraphPtr &graph) const;

 private:
  AutofuserOptions options_;
  CounterPtr counter_ = nullptr;
};
}  // namespace ge

#endif  // AUTOFUSE_AUTOFUSER_H
