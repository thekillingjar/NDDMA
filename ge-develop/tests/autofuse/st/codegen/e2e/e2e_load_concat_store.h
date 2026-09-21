/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_E2E_LOAD_CONCAT_STORE_H
#define AUTOFUSE_E2E_LOAD_CONCAT_STORE_H
#include "ascendc_ir.h"

void LoadConcatStore_BeforeAutofuse(ge::AscGraph &graph);
void LoadConcatStore_AfterInferOutput(ge::AscGraph &graph);
void LoadConcatStore_AfterGetApiInfo(ge::AscGraph &graph);
void LoadConcatStore_AfterScheduler(ge::AscGraph &graph, int32_t alignment = -1);
void LoadConcatStore_AfterQueBufAlloc(ge::AscGraph &graph);
void LoadConcatStore_BeforeAutofuseConcatInterAxis(ge::AscGraph &graph);
void LoadConcatStore_AfterSchedulerConcatInterAxis(ge::AscGraph &graph);
void LoadConcatStore_BeforeAutofuse3dLastAxis(ge::AscGraph &graph);
void LoadConcatStore_AfterScheduler3dLastAxis(ge::AscGraph &graph);
void LoadConcatStore_BeforeAutofuse7Inputs(ge::AscGraph &graph);
void LoadConcatStore_AfterInferOutput7Inputs(ge::AscGraph &graph);
void LoadConcatStore_AfterGetApiInfo7Inputs(ge::AscGraph &graph);
void LoadConcatStore_AfterScheduler7Inputs(ge::AscGraph &graph);
void LoadConcatStore_AfterQueBufAlloc7Inputs(ge::AscGraph &graph);

void LoadConcatStore_SmallTailBeforeAutofuse(ge::AscGraph &graph,
                                             ge::DataType data_type = ge::DT_FLOAT,
                                             const std::vector<int64_t> &concat_dim_sizes = {1, 2});
void LoadConcatStore_SmallTailAfterInferOutput(ge::AscGraph &graph);
void LoadConcatStore_SmallTailAfterGetApiInfo(ge::AscGraph &graph);
void LoadConcatStore_SmallTailAfterScheduler(ge::AscGraph &graph, int32_t alignment = 8);
void LoadConcatStore_SmallTailAfterQueBufAlloc(ge::AscGraph &graph);
#endif
