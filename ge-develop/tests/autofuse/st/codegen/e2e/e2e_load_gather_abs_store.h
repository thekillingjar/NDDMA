/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __E2E_LOAD_GATHER_ABS_STORE_H__
#define __E2E_LOAD_GATHER_ABS_STORE_H__

#include "ascendc_ir.h"

void LoadGatherAbsStore_BeforeAutofuse(ge::AscGraph &graph, int64_t gather_axis, ge::DataType data_type);
void LoadGather_BT_T_AbsStore_AfterAutofuse(ge::AscGraph &graph, ge::DataType data_type);
void LoadGather_T_BT_AbsStore_AfterAutofuse(ge::AscGraph &graph, ge::DataType data_type);
void LoadGather_B_T_AbsStore_AfterAutofuse(ge::AscGraph &graph, ge::DataType data_type);
void LoadGather_BT_AbsStore_AfterAutofuse(ge::AscGraph &graph, ge::DataType data_type);
void LoadGather_FirstAxis_B_T_AbsStore_BeforeAutofuse(ge::AscGraph &graph, int64_t gather_axis, ge::DataType data_type);
void LoadGather_FirstAxis_B_T_AbsStore_AfterAutofuse(ge::AscGraph& graph, ge::DataType data_type);

#endif
