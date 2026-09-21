/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/base_node_exe_faker.h"
#include "exe_graph/runtime/gert_tensor_data.h"

namespace gert {
ge::graphStatus BaseNodeExeFaker::OutputCreator(const ge::FastNode *node, KernelContext *context) const {
  auto output_num = context->GetOutputNum();

  size_t output_index = 0;
  for (; output_index < output_num / 2; ++output_index) {
    context->GetOutput(output_index)->SetWithDefaultDeleter(new StorageShape());
  }
  for (; output_index < output_num; ++output_index) {
    context->GetOutput(output_index)->SetWithDefaultDeleter(new GertTensorData());
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace gert
