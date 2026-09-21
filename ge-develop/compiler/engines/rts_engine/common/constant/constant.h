/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RTS_ENGINE_COMMON_CONSTANT_CONSTANT_H_
#define RTS_ENGINE_COMMON_CONSTANT_CONSTANT_H_

#include <string>

namespace cce {
namespace runtime {
// engine name
const std::string RTS_ENGINE_NAME = "DNN_VM_RTS";
const std::string RTS_OP_KERNEL_LIB_NAME = "DNN_VM_RTS_OP_STORE";
const std::string RTS_GRAPH_OPTIMIZER_LIB_NAME = "DNN_VM_RTS_GRAPH_OPTIMIZER_STORE";

// rts ffts plus engine name
const std::string RTS_FFTS_PLUS_ENGINE_NAME = "DNN_VM_RTS_FFTS_PLUS";
const std::string RTS_FFTS_PLUS_OP_KERNEL_LIB_NAME = "DNN_VM_RTS_FFTS_PLUS_OP_STORE";
const std::string RTS_FFTS_PLUS_GRAPH_OPTIMIZER_LIB_NAME = "DNN_VM_RTS_FFTS_PLUS_GRAPH_OPTIMIZER_STORE";

// attr name for send op event id
const std::string ATTR_NAME_SEND_ATTR_EVENT_ID = "event_id";
// attr name for recv op event id
const std::string ATTR_NAME_RECV_ATTR_EVENT_ID = "event_id";

// attr name for send op notify id
const std::string ATTR_NAME_SEND_ATTR_NOTIFY_ID = "notify_id";
// attr name for recv op notify id
const std::string ATTR_NAME_RECV_ATTR_NOTIFY_ID = "notify_id";

}  // namespace runtime
}  // namespace cce

#endif  // GE_RTS_ENGINE_COMMON_CONSTANT_CONSTANT_H_
