/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __TEST__E2E_POW_INT32_H__
#define __TEST__E2E_POW_INT32_H__

#include "ascendc_ir.h"

void PowInt32_BeforeAutofuse(ge::AscGraph &graph, ge::DataType data_type);
void PowInt32_AfterInferOutput(ge::AscGraph &graph, ge::DataType data_type);
void PowInt32_AfterGetApiInfo(ge::AscGraph &graph);
void PowInt32_AfterScheduler(ge::AscGraph &graph);
void PowInt32_AfterQueBufAlloc(ge::AscGraph &graph);

void PowInt32_AfterAutofuse(ge::AscGraph &graph, std::vector<ge::AscGraph> &impl_graphs,
                                            ge::DataType data_type);

#endif
