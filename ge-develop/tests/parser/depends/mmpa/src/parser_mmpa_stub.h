/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_TESTS_DEPENDS_MMPA_SRC_MMAP_STUB_H_
#define PARSER_TESTS_DEPENDS_MMPA_SRC_MMAP_STUB_H_

#include "mmpa/mmpa_api.h"
#include <memory>

#include <iostream>

namespace ge {
class MmpaStubApi {
 public:
  MmpaStubApi() = default;
  virtual ~MmpaStubApi() = default;

  virtual INT32 mmDladdr(VOID *addr, mmDlInfo *info)
  {
    return 0;
  }

  virtual VOID *mmDlopen(const CHAR *fileName, INT32 mode)
  {
    return NULL;
  }

  virtual INT32 mmRealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen)
  {
    return 0;
  }
};

class MmpaStub {
 public:
  static MmpaStub& GetInstance() {
    static MmpaStub mmpa_stub;
    return mmpa_stub;
  }

  void SetImpl(const std::shared_ptr<MmpaStubApi> &impl) {
    impl_ = impl;
  }

  MmpaStubApi* GetImpl() {
    return impl_.get();
  }

  void Reset() {
    impl_ = std::make_shared<MmpaStubApi>();
  }

 private:
  MmpaStub(): impl_(std::make_shared<MmpaStubApi>()){}
  std::shared_ptr<MmpaStubApi> impl_;
};

}  // namespace parser
#endif // PARSER_TESTS_DEPENDS_MMPA_SRC_MMAP_STUB_H_
