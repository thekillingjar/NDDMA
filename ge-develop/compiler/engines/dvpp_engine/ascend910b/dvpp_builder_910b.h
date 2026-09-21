/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_ENGINE_ASCEND910B_DVPP_BUILDER_910B_H_
#define DVPP_ENGINE_ASCEND910B_DVPP_BUILDER_910B_H_

#include "common/dvpp_builder.h"

namespace dvpp {
class DvppBuilder910B : public DvppBuilder {
 public:
  /**
   * @brief make the task info details
   *        then GE will alloc the address and send task by runtime
   * @param node node info
   * @param context run context from GE
   * @param tasks make the task return to GE
   * @return status whether success
   */
  DvppErrorCode GenerateTask(const ge::Node& node, ge::RunContext& context,
      std::vector<domi::TaskDef>& tasks) override;

 private:
  /**
   * @brief config dvpp task
   * @param node node info
   * @param dvpp_task output dvpp task
   * @return status whether success
   */
  DvppErrorCode BuildDvppTask(
      const ge::Node& node, domi::DvppTaskDef*& dvpp_task) const;

  /**
   * @brief set op channel resource
   * @param op_desc_ptr OpDesc pointer
   * @return status whether success
   */
  DvppErrorCode SetAttrResource(ge::OpDescPtr& op_desc_ptr) const override;
}; // class DvppBuilder910B
} // namespace dvpp

#endif // DVPP_ENGINE_ASCEND910B_DVPP_BUILDER_910B_H_