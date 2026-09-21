/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __TEST_E2E_LOAD_ABS_STORE_H__
#define __TEST_E2E_LOAD_ABS_STORE_H__

#include "ascendc_ir.h"

void LoadWhere2x3IsUbscalarStore_BeforeAutofuse(ge::AscGraph &graph, ge::DataType data_type);
void LoadWhere2x3IsUbscalarStore_AfterInferOutput(ge::AscGraph &graph, ge::DataType data_type);
void LoadWhere2x3IsUbscalarStore_AfterGetApiInfo(ge::AscGraph &graph);
void LoadWhere2x3IsUbscalarStore_AfterScheduler(ge::AscGraph &graph, bool align_last_axis);
void LoadWhere2x3IsUbscalarStore_AfterQueBufAlloc(ge::AscGraph &graph);
#endif
