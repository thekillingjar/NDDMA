/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/checker.h"
#include "graph/load/model_manager/davinci_model.h"
#include "register/kernel_registry.h"

namespace gert {
namespace kernel {
ge::graphStatus DavinciModelFinalizer(KernelContext *context) {
  const size_t num = context->GetInputNum();
  GELOGD("Total davinci model num: %zu.", num);
  GE_ASSERT_TRUE(num > 0U);
  std::vector<ge::DavinciModel *> davinci_models;
  for (size_t i = 0U; i < num; ++i) {
    const auto davinci_model = context->MutableInputPointer<ge::DavinciModel>(i);
    GE_ASSERT_NOTNULL(davinci_model);
    davinci_models.emplace_back(davinci_model);
  }

  for (const auto davinci_model : davinci_models) {
    davinci_model->UnbindTaskSinkStream();
  }
  for (const auto davinci_model : davinci_models) {
    davinci_model->DestroyStream();
  }
  for (const auto davinci_model : davinci_models) {
    davinci_model->DestroyResources();
  }
  GELOGD("Success to call all davinci models' finalize.");

  return ge::SUCCESS;
}
REGISTER_KERNEL(DavinciModelFinalizer).RunFunc(DavinciModelFinalizer);
}  // namespace kernel
}  // namespace gert
