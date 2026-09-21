/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "exe_graph/runtime/extended_kernel_context.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/tensor.h"
#include "register/kernel_registry_impl.h"
#include "tensor_sequence.h"
#include "resource_mgr.h"
#include "framework/common/debug/ge_log.h"

namespace gert {
ge::graphStatus SequenceEraseCompute(KernelContext* context) {
  constexpr int32_t kSessionIdIndex = 0;
  constexpr int32_t kContainerIdIndex = 1;
  constexpr int32_t kInputNumIndex = 2;
  constexpr int32_t kInputHandleIdx = 3;
  constexpr int32_t kInputIndexIdx = 4;
  GELOGD("Enter SequenceEraseCompute");
  auto input_num = context->GetInputValue<uint32_t>(kInputNumIndex);
  if (input_num < 1) {
    GELOGE(ge::PARAM_INVALID, "input num must be at least 1, but now is %d", input_num);
    REPORT_INNER_ERR_MSG("E39999", "input num must be at least 1, but now is %d", input_num);
    return ge::PARAM_INVALID;
  }

  auto input_handle_data =
      context->GetInputPointer<TensorData>(kInputHandleIdx);
  if (input_handle_data == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Get input handle tensor data failed.");
    REPORT_INNER_ERR_MSG("E39999", "Get input handle tensor data failed.");
    return ge::PARAM_INVALID;
  }

  if (input_handle_data->GetAddr() == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Get input handle tensor data nullptr.");
    REPORT_INNER_ERR_MSG("E39999", "Get input handle tensor data nullptr.");
    return ge::PARAM_INVALID;
  }

  // handle's type is DT_RESOURCE(aka uint64_t)
  auto handle = static_cast<uint64_t*>(input_handle_data->GetAddr());
  GELOGD("handle is %llu", *handle);

  auto session_id = context->GetInputValue<size_t>(kSessionIdIndex);
  auto container_id = context->GetInputValue<size_t>(kContainerIdIndex);
  GELOGD("Session_id is %llu, Container_id is %llu", session_id, container_id);
  ResourceMgrPtr rm;
  SessionMgr::GetInstance()->GetRm(session_id, container_id, rm);
  TensorSeqPtr tensor_seq_ptr;
  rm->Lookup(*handle, &tensor_seq_ptr);
  int64_t index = tensor_seq_ptr->Size() - 1;
  int32_t output_idx = kInputIndexIdx;
  constexpr int32_t least_input_number = 2;
  if (input_num == least_input_number) {
    auto extend_ctx = reinterpret_cast<ExtendedKernelContext*>(context);
    auto erase_index_desc = extend_ctx->GetInputDesc(1);
    if (erase_index_desc == nullptr) {
      GELOGE(ge::PARAM_INVALID, "erase_index_desc is nullptr");
      REPORT_INNER_ERR_MSG("E39999", "erase_index_desc is nullptr");
      return ge::PARAM_INVALID;
    }

    auto erase_index_type = erase_index_desc->GetDataType();
    auto erase_index_data =
        context->GetInputPointer<TensorData>(kInputIndexIdx);
    if (erase_index_data == nullptr) {
      GELOGE(ge::PARAM_INVALID, "Get erase index failed.");
      REPORT_INNER_ERR_MSG("E39999", "Get erase index failed.");
      return ge::PARAM_INVALID;
    }
    if (erase_index_data->GetAddr() == nullptr) {
      GELOGE(ge::PARAM_INVALID, "Get erase tensorData nullptr.");
      REPORT_INNER_ERR_MSG("E39999", "Get erase tensorData nullptr.");
      return ge::PARAM_INVALID;
    }

    switch (erase_index_type) {
      case ge::DT_INT32:
        index = static_cast<int64_t>(
            *static_cast<int32_t*>(erase_index_data->GetAddr()));
        break;
      case ge::DT_INT64:
        index = *(static_cast<int64_t*>(erase_index_data->GetAddr()));
        break;
      default:
        GELOGE(ge::PARAM_INVALID, "Sequence Erase input index data type should be DT_INT32 "
               "or DT_INT64, [%u] not support.", erase_index_type);
        REPORT_INNER_ERR_MSG("E39999", "Sequence Erase input index data type should be DT_INT32 "
                           "or DT_INT64, [%u] not support.", erase_index_type);
        return ge::PARAM_INVALID;
    }
    output_idx ++;
  }
  GELOGD("erase index is %lld, tensor sequence size is %zu", index, tensor_seq_ptr->Size());
  if (tensor_seq_ptr->Erase(index) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::PARAM_INVALID, "erase value from sequence tensor failed.");
    REPORT_INNER_ERR_MSG("E39999", "erase value from sequence tensor failed.");
    return ge::PARAM_INVALID;
  }

  auto output_tensor = context->MutableInputPointer<Tensor>(output_idx);
  if (output_tensor == nullptr) {
    GELOGE(ge::PARAM_INVALID, "output_tensor is nullptr");
    REPORT_INNER_ERR_MSG("E39999", "output_tensor is nullptr");
    return ge::PARAM_INVALID;
  }

  uint64_t* data_ptr = output_tensor->GetData<uint64_t>();
  if (data_ptr == nullptr) {
    GELOGE(ge::PARAM_INVALID, "output_tensor tensorData is nullptr");
    REPORT_INNER_ERR_MSG("E39999", "output_tensor tensorData is nullptr");
    return ge::PARAM_INVALID;
  }

  *data_ptr = *handle;
  GELOGD("Finish SequenceEraseCompute, tensor sequence size is %zu.",
         tensor_seq_ptr->Size());
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(SequenceEraseCompute).RunFunc(SequenceEraseCompute);
}  // namespace gert