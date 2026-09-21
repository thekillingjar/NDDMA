/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_ENGINE_ASCEND910B_DVPP_INFO_STORE_910B_H_
#define DVPP_ENGINE_ASCEND910B_DVPP_INFO_STORE_910B_H_

#include "common/dvpp_info_store.h"

namespace dvpp {
class DvppInfoStore910B : public DvppInfoStore {
 public:
  /**
   * @brief convert op_desc to vpu_config or cmdlist
   * @param task task info
   * @return status whether success
   */
  DvppErrorCode LoadTask(ge::GETaskInfo& task) override;

  /**
   * @brief free vpu_config or cmdlist memory
   * @param task task info
   * @return status whether success
   */
  DvppErrorCode UnloadTask(ge::GETaskInfo& task) override;
}; // class DvppInfoStore910B
} // namespace dvpp

#endif // DVPP_ENGINE_ASCEND910B_DVPP_INFO_STORE_910B_H_