/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_PLUS_QOS_UPDATE_H_
#define FFTS_PLUS_QOS_UPDATE_H_

#include "runtime/rt_ffts_plus_define.h"
#include "graph/utils/node_utils.h"
namespace ffts {

bool UpdateAicAivCtxQos(rtFftsPlusAicAivCtx_t *ctx, int label, int device_id);
bool UpdateMixAicAivCtxQos(rtFftsPlusMixAicAivCtx_t *ctx, int label, int device_id);
bool UpdateDataCtxQos(rtFftsPlusDataCtx_t *ctx, int device_id);

}
#endif
