/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __TEST_E2E_LOAD_SCALAR_DIV_STORE_H__
#define __TEST_E2E_LOAD_SCALAR_DIV_STORE_H__

#include "ascendc_ir.h"

void LoadScalarDivStore_BeforeAutofuse(ge::AscGraph &graph, ge::DataType in_data_type, ge::DataType out_data_type);
void LoadScalarDivStore_AfterInferOutput(ge::AscGraph &graph, ge::DataType in_data_type, ge::DataType out_data_type);
void LoadScalarDivStore_AfterGetApiInfo(ge::AscGraph &graph);
void LoadScalarDivStore_AfterScheduler(ge::AscGraph &graph);
void LoadScalarDivStore_AfterQueBufAlloc(ge::AscGraph &graph);
#endif


