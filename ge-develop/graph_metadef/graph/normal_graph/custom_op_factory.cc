/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/custom_op_factory.h"
#include "graph/custom_op.h"
#include "graph/custom_op_factory_impl.h"
#include "debug/ge_log.h"

namespace ge {
graphStatus CustomOpFactory::RegisterCustomOpCreator(const AscendString &op_type, const BaseOpCreator &op_creator) {
  return CustomOpFactoryImpl::GetInstance().RegisterCustomOpCreator(op_type, op_creator);
}

std::unique_ptr<BaseCustomOp> CustomOpFactory::CreateCustomOp(const AscendString &op_type) {
  return CustomOpFactoryImpl::GetInstance().CreateCustomOp(op_type);
}

graphStatus CustomOpFactory::GetAllRegisteredOps(std::vector<AscendString> &all_registered_ops) {
  return CustomOpFactoryImpl::GetInstance().GetAllRegisteredOps(all_registered_ops);
}
bool CustomOpFactory::IsExistOp(const AscendString &op_type) {
  return CustomOpFactoryImpl::GetInstance().IsExistOp(op_type);
}

CustomOpCreatorRegister::CustomOpCreatorRegister(const AscendString &operator_type, BaseOpCreator const &op_creator) {
  CustomOpFactoryImpl::GetInstance().RegisterCustomOpCreator(operator_type, op_creator);
}
}  // namespace ge
