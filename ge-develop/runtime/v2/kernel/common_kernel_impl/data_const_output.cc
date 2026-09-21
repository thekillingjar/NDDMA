/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ge_error_codes.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/kernel_registry.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/node.h"
#include "common/debug/ge_log.h"
#include "graph/utils/math_util.h"
#include "common/checker.h"

namespace gert {
namespace kernel {
ge::graphStatus DefaultFunc(KernelContext *context) {
  (void) context;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConstOutputCreator(const ge::FastNode *org_node, KernelContext *context) {
  const auto op_desc = org_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  ge::Buffer buffer;
  if (!ge::AttrUtils::GetZeroCopyBytes(op_desc, kConstValue, buffer)) {
    GELOGE(ge::GRAPH_FAILED, "%s get zero copy bytes fail.", org_node->GetName().c_str());
    return ge::GRAPH_FAILED;
  }

  // todo 为string考虑更好的办法
  bool alloc;
  bool is_string = false;
  if (ge::AttrUtils::GetBool(op_desc, "is_string", is_string) && is_string) {
    alloc = true;
  } else {
    alloc = !KernelContext::IsInlineSize(buffer.size());
  }
  auto av = context->GetOutput(0);
  if (av == nullptr) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Kernel output index 0 is nullptr");
    return ge::GRAPH_PARAM_INVALID;
  }
  if (alloc) {
    auto const_data = ge::MakeUnique<uint8_t[]>(buffer.size());
    if (const_data == nullptr) {
      GELOGE(ge::GRAPH_FAILED, "Failed to construct data, buffer size %zu", buffer.size());
      return ge::GRAPH_FAILED;
    }
    const auto ret = ge::GeMemcpy(const_data.get(), buffer.size(), buffer.data(), buffer.size());
    GE_ASSERT_TRUE((ret == ge::GRAPH_SUCCESS), "memcpy_s failed, copy size is %zu", buffer.size());
    av->SetWithDefaultDeleter<uint8_t[]>(const_data.release());
  } else {
    if (memcpy_s(av->GetPointer<void *>(), sizeof(void *), buffer.data(), buffer.size()) != EOK) {
      GELOGE(ge::GRAPH_FAILED, "Failed to memory copy, dest addr %u, src addr %u, buffer size %zu",
             av->GetPointer<void *>(), buffer.data(), buffer.size());
      return ge::GRAPH_FAILED;
    }
  }

  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(Data).RunFunc(DefaultFunc);
REGISTER_KERNEL(Const).RunFunc(DefaultFunc).OutputsCreator(ConstOutputCreator);
REGISTER_KERNEL(NetOutput).RunFunc(DefaultFunc);
}  // namespace kernel
}  // namespace gert
