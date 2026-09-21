/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_E2E_LOAD_SUB_STORE_H
#define AUTOFUSE_E2E_LOAD_SUB_STORE_H
#include "ascendc_ir.h"

void LoadSubStore_BeforeAutofuse(ge::AscGraph &graph);
void LoadSubStore_AfterInferOutput(ge::AscGraph &graph);
void LoadSubStore_AfterGetApiInfo(ge::AscGraph &graph);
void LoadSubStore_AfterScheduler(ge::AscGraph &graph);
void LoadSubStore_AfterQueBufAlloc(ge::AscGraph &graph);
#endif
