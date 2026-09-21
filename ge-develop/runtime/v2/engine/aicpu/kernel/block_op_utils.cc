/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "block_op_utils.h"
#include "base/err_msg.h"
#include "acl/acl_rt.h"

namespace gert {
ge::Status CheckDeviceSupportBlockingAicpuOpProcess(bool &is_support) {
  int32_t device_id = 0;
  GE_ASSERT_RT_OK(aclrtGetDevice(&device_id));

  int32_t value = 0;
  GE_ASSERT_RT_OK(rtGetDeviceCapability(device_id, FEATURE_TYPE_BLOCKING_OPERATOR, RT_MODULE_TYPE_AICPU, &value));
  if ((value != RT_AICPU_BLOCKING_OP_NOT_SUPPORT) && (value != RT_AICPU_BLOCKING_OP_SUPPORT)) {
    REPORT_INNER_ERR_MSG("E19999", "Value should be %d or %d but %d",
                       RT_AICPU_BLOCKING_OP_NOT_SUPPORT, RT_AICPU_BLOCKING_OP_SUPPORT, value);
    GELOGE(ge::FAILED, "[Check][Value] Value should be %d or %d but %d",
           RT_AICPU_BLOCKING_OP_NOT_SUPPORT, RT_AICPU_BLOCKING_OP_SUPPORT, value);
    return ge::FAILED;
  }

  is_support = (value == RT_AICPU_BLOCKING_OP_SUPPORT);

  return ge::SUCCESS;
}

ge::Status DistributeWaitTaskForAicpuBlockingOp(rtStream_t stream, const AicpuArgsHandler *arg_handler, const char *op_name) {
  const auto rt_event = arg_handler->GetRtEvent();
  GE_ASSERT_NOTNULL(rt_event);
  GE_ASSERT_RT_OK(rtSetTaskTag(op_name));
  uint32_t async_timeout = arg_handler->GetAsyncTimeout();
  GELOGI("Async timeout:0x%x.", async_timeout);
  if (async_timeout != 0xFFFFFFFF) {
    GE_ASSERT_RT_OK(rtStreamWaitEventWithTimeout(stream, rt_event, async_timeout));
  } else {
    GE_ASSERT_RT_OK(rtStreamWaitEvent(stream, rt_event));
  }

  GE_ASSERT_RT_OK(rtSetTaskTag(op_name));
  GE_ASSERT_RT_OK(rtEventReset(rt_event, stream));

  return ge::SUCCESS;
}
}  // namespace gert
