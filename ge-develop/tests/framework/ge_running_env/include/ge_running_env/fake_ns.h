/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H7AEFF0EA_9FDE_487F_8562_2917A2D48EA2
#define H7AEFF0EA_9FDE_487F_8562_2917A2D48EA2

#define FAKE_NS ge
#define FAKE_NS_BEGIN namespace FAKE_NS {
#define FAKE_NS_END }
#define USING_FAKE_NS using namespace FAKE_NS;
#define FWD_DECL_FAKE(type) \
  namespace FAKE_NS {       \
  struct type;              \
  }

#endif
