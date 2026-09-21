/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __E2E_BROADCAST_FORCE_MERGE_H__
#define __E2E_BROADCAST_FORCE_MERGE_H__

#include "ascendc_ir.h"

void BroadcastLoadStore_BeforeAutofuse(ge::AscGraph &graph);
void BroadcastLoadStore_AfterInferOutput(ge::AscGraph &graph);
void BroadcastLoadStore_AfterGetApiInfo(ge::AscGraph &graph);
void BroadcastLoadStore_AfterQueBufAlloc(ge::AscGraph &graph);

void BroadcastLoadStore_AfterScheduler_z1_SplitTo_z0z1TBz0z1Tbz1t(ge::AscGraph &graph);

#endif
