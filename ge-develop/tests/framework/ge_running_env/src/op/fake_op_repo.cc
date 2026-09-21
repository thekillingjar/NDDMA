/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/operator_factory_impl.h"
#include "ge_running_env/fake_op.h"
#include "fake_op_repo.h"

FAKE_NS_BEGIN

void FakeOpRepo::Reset() {
  if (OperatorFactoryImpl::operator_creators_) {
    OperatorFactoryImpl::operator_creators_->clear();
  }
  if (OperatorFactoryImpl::operator_infershape_funcs_) {
    OperatorFactoryImpl::operator_infershape_funcs_->clear();
  }
}

void FakeOpRepo::Regist(const std::string &operator_type, const OpCreator creator) {
  OperatorFactoryImpl::RegisterOperatorCreator(operator_type, creator);
}
void FakeOpRepo::Regist(const std::string &operator_type, const InferShapeFunc infer_fun) {
  OperatorFactoryImpl::RegisterInferShapeFunc(operator_type, infer_fun);
}

FAKE_NS_END
