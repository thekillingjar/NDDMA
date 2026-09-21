/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_GE_GRAPH_FUSION_FUSION_PASS_REG_H
#define INC_EXTERNAL_GE_GRAPH_FUSION_FUSION_PASS_REG_H
#include "fusion_base_pass.h"
#include "register/register_custom_pass.h"
namespace ge {
namespace fusion {
using CreateFusionPassFn = FusionBasePass *(*)();

class FusionPassRegistrationDataImpl;
class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FusionPassRegistrationData {
 public:
  FusionPassRegistrationData() = default;
  ~FusionPassRegistrationData() = default;

  explicit FusionPassRegistrationData(const AscendString &pass_name);

  /**
   * 获取融合pass name
   * @return
   */
  AscendString GetPassName() const;

  /**
   * 融合pass阶段定义
   * 注意：kAfterAssignLogicStream阶段不可注册普通融合pass，此阶段不允许修改图结构，若误将pass注册到此阶段，将被忽略
   * @param stage
   * @return
   */
  FusionPassRegistrationData &Stage(CustomPassStage stage);

  /**
   * 获取融合pass所属阶段
   * @return
   */
  CustomPassStage GetStage() const;

  /**
   * 融合pass creator注册函数
   * @param create_fusion_pass_fn
   * @return
   */
  FusionPassRegistrationData &CreatePassFn(const CreateFusionPassFn &create_fusion_pass_fn);

  /**
   * 获取融合pass creator
   * @return
   */
  CreateFusionPassFn GetCreatePassFn() const;

  /**
   * 将融合pass的注册信息序列化，如pass name, stage等
   * @return
   */
  AscendString ToString() const;

 private:
  std::shared_ptr<FusionPassRegistrationDataImpl> impl_;
};

class PassRegistrar {
 public:
  /**
   * 该函数接收一个融合pass注册信息，注册到全局单例中
   * @param fusion_pass_reg_data
   */
  PassRegistrar(FusionPassRegistrationData &fusion_pass_reg_data);
  ~PassRegistrar() = default;
};
}  // namespace fusion
}  // namespace ge
#endif  // INC_EXTERNAL_GE_GRAPH_FUSION_FUSION_PASS_REG_H
