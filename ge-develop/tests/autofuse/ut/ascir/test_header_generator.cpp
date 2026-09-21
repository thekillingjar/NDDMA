/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <sstream>
#include <iostream>
namespace ascir {
void GenHeaderFileToStream(const char *, std::stringstream &ss);

class GeneratorUT : public testing::Test {};
TEST_F(GeneratorUT, GnerateAll_BuiltinOps_Ok) {
  std::stringstream ss;
  GenHeaderFileToStream("/path/to/hello.h", ss);
  std::cout << "SN_DEBUG ===================:" << std::endl;
  std::cout << ss.str() << std::endl;
}
}  // namespace ascir