/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include "mmpa/mmpa_api.h"
#include "generator.h"
#include "common/util/sanitizer_options.h"

int main(int argc, char *argv[]) {
  constexpr int kExpectArgNum = 3;
  if (argc != kExpectArgNum) {
    std::cerr << "Arg format: ascir_ops_header_generator </path/to/ops_so/file> </path/to/header/file>" << std::endl;
    return 1;
  }
  void *const handle = mmDlopen(
      argv[1], static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) | static_cast<uint32_t>(MMPA_RTLD_GLOBAL)));
  if (handle == nullptr) {
    const auto *error = mmDlerror();
    error = (error == nullptr) ? "" : error;
    std::cerr << "dlopen failed, so name:" << argv[1] << ", error info:" << error << std::endl;
    return 1;
  }
  const auto ret = ge::ascir::GenHeaderFile(argv[kExpectArgNum - 1]);
  (void) mmDlclose(handle);
  DT_DO_DETECT_LEAKS();
  return ret;
}