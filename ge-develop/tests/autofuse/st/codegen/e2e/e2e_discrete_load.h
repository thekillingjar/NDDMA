/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __TEST_E2E_DISCRETE_LOAD_H__
#define __TEST_E2E_DISCRETE_LOAD_H__

#include "ascendc_ir.h"

void LoadAbsStore_BeforeAutofuse_DiscreteLoad(ge::AscGraph &graph);
void LoadAbsStore_AfterInferOutput_DiscreteLoad(ge::AscGraph &graph);
void LoadAbsStore_AfterGetApiInfo_DiscreteLoad(ge::AscGraph &graph);
void LoadAbsStore_AfterScheduler_z0_SplitTo_z0TBz0Tb(ge::AscGraph &graph);
void LoadAbsStore_AfterQueBufAlloc_DiscreteLoad(ge::AscGraph &graph);
#endif
