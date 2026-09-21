/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "diagnose_switch.h"

namespace ge {
namespace {
SingleDiagnoseSwitch profiling_switch_;
SingleDiagnoseSwitch dumper_switch_;
}  // namespace

namespace diagnoseSwitch {
SingleDiagnoseSwitch &MutableProfiling() {
  return profiling_switch_;
}

const SingleDiagnoseSwitch &GetProfiling() {
  return profiling_switch_;
}

SingleDiagnoseSwitch &MutableDumper() {
  return dumper_switch_;
}
const SingleDiagnoseSwitch &GetDumper() {
  return dumper_switch_;
}

void EnableDataDump() {
  dumper_switch_.SetEnableFlag(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kDataDump}));
}

void EnableOverflowDump() {
  dumper_switch_.SetEnableFlag(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kOverflowDump}));
}

void EnableExceptionDump() {
  dumper_switch_.SetEnableFlag(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kExceptionDump}));
}

void EnableLiteExceptionDump() {
  dumper_switch_.SetEnableFlag(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kLiteExceptionDump}));
}

void EnableHostDump() {
  dumper_switch_.SetEnableFlag(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kHostDump}));
}

void EnableTrainingTrace() {
  profiling_switch_.SetEnableFlag(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>({gert::ProfilingType::kTrainingTrace}));
}

void EnableGeHostProfiling() {
  profiling_switch_.SetEnableFlag(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>({gert::ProfilingType::kGeHost}));
}

void EnableDeviceProfiling() {
  profiling_switch_.SetEnableFlag(gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>(
      {gert::ProfilingType::kDevice, gert::ProfilingType::kTaskTime}));
}

void EnableTaskTimeProfiling() {
  profiling_switch_.SetEnableFlag(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>({gert::ProfilingType::kTaskTime}));
}

void EnableMemoryProfiling() {
  profiling_switch_.SetEnableFlag(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>({gert::ProfilingType::kMemory}));
}

void EnableCannHostProfiling() {
  profiling_switch_.SetEnableFlag(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>({gert::ProfilingType::kCannHost}));
}

void EnableProfiling(const std::vector<gert::ProfilingType> &prof_type) {
  profiling_switch_.SetEnableFlag(gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>(prof_type));
}

void DisableProfiling() {
  profiling_switch_.SetEnableFlag(0UL);
}

void DisableDumper() {
  dumper_switch_.SetEnableFlag(0UL);
}
}
}  // namespace ge
