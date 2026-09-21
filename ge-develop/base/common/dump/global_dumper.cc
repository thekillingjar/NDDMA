/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/subscriber/global_dumper.h"
#include "common/debug/ge_log.h"
#include "common/global_variables/diagnose_switch.h"
#include "common/dump/exception_dumper.h"

namespace gert {
namespace {
ge::ExceptionDumper g_exception_dumper{false};  // not register to global dumper
}

GlobalDumper *GlobalDumper::GetInstance() {
  static GlobalDumper global_dumper;
  return &global_dumper;
}

void GlobalDumper::OnGlobalDumperSwitch(void *ins, uint64_t enable_flags) {
  auto ess = static_cast<GlobalDumper *>(ins);
  if (ess == nullptr) {
    GELOGW("The instance is nullptr when switch dumper, ignore the event");
    return;
  }
  ess->SetEnableFlags(enable_flags);
}

ge::ExceptionDumper *GlobalDumper::MutableExceptionDumper() {
  const std::lock_guard<std::mutex> lk{mutex_};
  return &g_exception_dumper;
}

GlobalDumper::GlobalDumper() {
  ge::diagnoseSwitch::MutableDumper().RegisterHandler(this, {this, GlobalDumper::OnGlobalDumperSwitch});
}
}  // namespace gert
