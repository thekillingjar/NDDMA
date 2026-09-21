/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_STUB_GE_LOG_H_
#define INC_STUB_GE_LOG_H_

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GELOGD(fmt, ...)                                                      \
do {                                                                                              \
    printf("[DEBUG][%s:%d] " fmt "\n", __FILE__, __LINE__,  ##__VA_ARGS__); \
  } while (false)

#define GELOGI(fmt, ...)                                                                         \
  do {                                                                                           \
    printf("[INFO][%s:%d] " fmt "\n", __FILE__, __LINE__,  ##__VA_ARGS__); \
  } while (false)

#define GELOGW(fmt, ...)                                                                            \
  do {                                                                                              \
    printf("[WARNING][%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
  } while (false)

#define GELOGE(ERROR_CODE, fmt, ...)                                                                                \
  do {                                                                                                              \
    printf("[ERROR][%s:%d] ErrorNo:%ld " fmt "\n", __FILE__, __LINE__, (long int)ERROR_CODE, \
           ##__VA_ARGS__);                                                                                          \
  } while (false)

#define GEEVENT(fmt, ...)                                                                         \
  do {                                                                                            \
    printf("[EVENT][%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
  } while (false)

#ifdef __cplusplus
}
#endif
#endif  // INC_STUB_GE_LOG_H_
