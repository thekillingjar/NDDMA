/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/kernel_registry.h"
#include "common/checker.h"
#include "framework/runtime/model_desc.h"
namespace gert {
ge::graphStatus GetSpaceRegistry(KernelContext *context) {
  auto model_desc = context->GetInputValue<ModelDesc *>(0);
  GE_ASSERT_NOTNULL(model_desc);
  auto opp_impl_version = context->GetInputValue<size_t>(1);
  GE_ASSERT_TRUE(opp_impl_version < static_cast<size_t>(ge::OppImplVersion::kVersionEnd));
  auto space_registry = context->GetOutputPointer<gert::OpImplSpaceRegistryV2 *>(0);
  GE_ASSERT_NOTNULL(space_registry);
  GE_ASSERT_NOTNULL(model_desc->GetSpaceRegistries());
  *space_registry = model_desc->GetSpaceRegistries()->at(opp_impl_version).get();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetSpaceRegistries(KernelContext *context) {
  auto model_desc = context->GetInputValue<ModelDesc *>(0);
  GE_ASSERT_NOTNULL(model_desc);
  auto space_registry_array = context->GetOutputPointer<gert::OpImplSpaceRegistryV2Array *>(0);
  GE_ASSERT_NOTNULL(space_registry_array);
  GE_ASSERT_NOTNULL(model_desc->GetSpaceRegistries());
  *space_registry_array = model_desc->GetSpaceRegistries();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetFileConstantWeightDir(KernelContext *context) {
  auto model_desc = context->GetInputValue<ModelDesc *>(0);
  GE_ASSERT_NOTNULL(model_desc);
  auto file_constant_weight_dir = context->GetOutputPointer<const ge::char_t *>(0);
  GE_ASSERT_NOTNULL(file_constant_weight_dir);
  *file_constant_weight_dir = model_desc->GetFileConstantWeightDir();
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(GetSpaceRegistry).RunFunc(GetSpaceRegistry);
REGISTER_KERNEL(GetSpaceRegistries).RunFunc(GetSpaceRegistries);
REGISTER_KERNEL(GetFileConstantWeightDir).RunFunc(GetFileConstantWeightDir);
}  // namespace gert
