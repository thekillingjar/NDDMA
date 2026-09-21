/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACL_RT_CONDITION_H
#define ACL_RT_CONDITION_H

typedef enum acltagRtCondition {
  ACl_RT_EQUAL = 0,
  ACl_RT_NOT_EQUAL,
  ACL_RT_GREATER,
  ACL_RT_GREATER_OR_EQUAL,
  ACL_RT_LESS,
  ACL_RT_LESS_OR_EQUAL
} aclrtCondition;

#endif  // ACL_RT_CONDITION_H
