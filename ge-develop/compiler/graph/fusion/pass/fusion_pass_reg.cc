/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ge/fusion/pass/fusion_pass_reg.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/checker.h"

#include "pass_registry.h"
namespace ge {
namespace fusion {
class FusionPassRegistrationDataImpl {
 public:
  explicit FusionPassRegistrationDataImpl(const AscendString &pass_name) : pass_name_(pass_name) {}

  void SetPassName(const AscendString &pass_name) {
    pass_name_ = pass_name;
  }

  AscendString GetPassName() const {
    return pass_name_;
  }

  void Stage(CustomPassStage stage) {
    if (stage == CustomPassStage::kAfterAssignLogicStream || stage >= CustomPassStage::kInvalid) {
      GELOGE(PARAM_INVALID, "Fusion Pass[%s] register stage[%s] which is invalid.", pass_name_.GetString(),
             CustomPassStageToString(stage).c_str());
    }
    stage_ = stage;
  }

  CustomPassStage GetStage() const {
    return stage_;
  }

  void CreatePassFn(const CreateFusionPassFn &create_fusion_pass_fn) {
    create_pass_func_ = create_fusion_pass_fn;
  }

  CreateFusionPassFn GetCreatePassFn() const {
    return create_pass_func_;
  }

 private:
  AscendString pass_name_;
  CreateFusionPassFn create_pass_func_{};
  CustomPassStage stage_{};
};

FusionPassRegistrationData::FusionPassRegistrationData(const AscendString &pass_name) {
  impl_ = MakeUnique<FusionPassRegistrationDataImpl>(pass_name);
}

AscendString FusionPassRegistrationData::GetPassName() const {
  if (impl_ != nullptr) {
    return impl_->GetPassName();
  }
  return {};
}

FusionPassRegistrationData &FusionPassRegistrationData::Stage(CustomPassStage stage){
  if (impl_ != nullptr) {
    impl_->Stage(stage);
  }
  return *this;
}

CustomPassStage FusionPassRegistrationData::GetStage() const {
  if (impl_ != nullptr) {
    return impl_->GetStage();
  }
  return CustomPassStage::kBeforeInferShape;
}

FusionPassRegistrationData &FusionPassRegistrationData::CreatePassFn(const CreateFusionPassFn &create_fusion_pass_fn) {
  if (impl_ != nullptr) {
    impl_->CreatePassFn(create_fusion_pass_fn);
  }
  return *this;
}

CreateFusionPassFn FusionPassRegistrationData::GetCreatePassFn() const {
  if (impl_ != nullptr) {
    return impl_->GetCreatePassFn();
  }
  return nullptr;
}

AscendString FusionPassRegistrationData::ToString() const {
  AscendString reg_info;
  if (impl_ != nullptr) {
    std::stringstream ss;
    ss << "Pass Name[" << impl_->GetPassName().GetString() << "], stage[" << CustomPassStageToString(impl_->GetStage())
       << "]";
    reg_info = ss.str().c_str();
  }
  return reg_info;
}

PassRegistrar::PassRegistrar(FusionPassRegistrationData &fusion_pass_reg_data) {
  PassRegistry::GetInstance().RegisterFusionPass(fusion_pass_reg_data);
}
} // namespace fusion
}  // namespace ge