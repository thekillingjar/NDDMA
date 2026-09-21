/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/infer_axis_slice_registry.h"
#include "graph/types.h"
#include "graph/operator_factory_impl.h"

namespace ge {
InferAxisTypeInfoFuncRegister::InferAxisTypeInfoFuncRegister(const char_t *const operator_type,
                                                             const InferAxisTypeInfoFunc &infer_axis_type_info_func) {
  (void)OperatorFactoryImpl::RegisterInferAxisTypeInfoFunc(operator_type, infer_axis_type_info_func);
}

InferAxisSliceFuncRegister::InferAxisSliceFuncRegister(const char_t *const operator_type,
                                                       const InferAxisSliceFunc &infer_axis_slice_func) {
  (void)OperatorFactoryImpl::RegisterInferAxisSliceFunc(operator_type, infer_axis_slice_func);
}
}  // namespace ge
