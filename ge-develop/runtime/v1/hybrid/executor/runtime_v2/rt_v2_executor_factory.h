/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_MANY_RT2_EXECUTORS_H_
#define GE_HYBRID_EXECUTOR_MANY_RT2_EXECUTORS_H_

#include "hybrid/model/hybrid_model.h"
#include "rt_v2_executor_interface.h"
#include "common/model/external_allocator_manager.h"

namespace gert {
class RtV2ExecutorFactory {
 public:
  static std::unique_ptr<RtV2ExecutorInterface> Create(const ge::GeRootModelPtr &model,
                                                       ge::DevResourceAllocator &allocator,
                                                       RtSession *session = nullptr);

 private:
  RtV2ExecutorFactory() = default;
};
}  // namespace gert

#endif  // GE_HYBRID_EXECUTOR_MANY_RT2_EXECUTORS_H_
