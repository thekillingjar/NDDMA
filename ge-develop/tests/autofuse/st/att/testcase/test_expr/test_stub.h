/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_TEST_TEST_STUB_H_
#define ATT_TEST_TEST_STUB_H_
#include <iostream>
// #define TEST_F(test_part, test_case) \
//   void Test_##test_part_##test_case()

#define IS_TRUE(condi) \
  do {                 \
    if (condi) {       \
      std::cout << "[pass]: " << #condi << std::endl; \
    }                  \
    else {             \
      std::cout << "[failed]: " << #condi << std::endl; \
    }                 \
  } while (0);

#define DUMPS(info) \
  do {              \
    std::cout << "[info]: " << info << std::endl; \
  } while(0);

#define RUN_TEST_F(test_part, test_case) \
  do {                                   \
    std::cout << "========= " << #test_part << " : " << #test_case << " ==========" << std::endl; \
    Test_##test_part_##test_case();      \
  } while(0);
  
#endif  // ATT_TEST_TEST_STUB_H_