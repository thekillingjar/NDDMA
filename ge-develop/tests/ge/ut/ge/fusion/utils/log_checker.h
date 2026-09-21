/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CANN_GRAPH_ENGINE_LOG_CHECKER_H
#define CANN_GRAPH_ENGINE_LOG_CHECKER_H

#include "stub/gert_runtime_stub.h"
namespace ge {
namespace ut {
inline bool ErrorLogContain(gert::GertRuntimeStub &runtime_stub, const std::string &expect_str) {
  return runtime_stub.GetSlogStub().FindLogRegex(DLOG_ERROR, expect_str.c_str()) >= 0;
}
inline bool WarnLogContain(gert::GertRuntimeStub &runtime_stub, const std::string &expect_str) {
  return runtime_stub.GetSlogStub().FindLogRegex(2, expect_str.c_str()) >= 0;
}
inline bool DebugLogContain(gert::GertRuntimeStub &runtime_stub, const std::string &expect_str) {
  return runtime_stub.GetSlogStub().FindLogRegex(DLOG_DEBUG, expect_str.c_str()) >= 0;
}
/**
 * 若执行过程中无error log，返回true
 * @param runtime_stub
 * @param expect_str
 * @return
 */
inline bool IsErrorClean(gert::GertRuntimeStub &runtime_stub) {
  return runtime_stub.GetSlogStub().GetLogs(DLOG_ERROR).empty();
}
}  // namespace ut
}  // namespace ge
#endif  // CANN_GRAPH_ENGINE_LOG_CHECKER_H
