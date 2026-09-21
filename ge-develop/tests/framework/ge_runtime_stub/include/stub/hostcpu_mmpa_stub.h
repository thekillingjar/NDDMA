/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_UT_GE_RUNTIME_V2_STUB_HOSTCPU_MMPA_STUB_H
#define AIR_CXX_TESTS_UT_GE_RUNTIME_V2_STUB_HOSTCPU_MMPA_STUB_H

#include <string>
#include "depends/mmpa/src/mmpa_stub.h"

namespace gert {
class HostcpuMockMmpa : public ge::MmpaStubApiGe {
 public:
  void *DlSym(void *handle, const char *func_name) override;
  int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override;
  void *DlOpen(const char *file_name, int32_t mode) override;
};
}  // namespace gert

#endif // AIR_CXX_TESTS_UT_GE_RUNTIME_V2_STUB_HOSTCPU_MMPA_STUB_H
