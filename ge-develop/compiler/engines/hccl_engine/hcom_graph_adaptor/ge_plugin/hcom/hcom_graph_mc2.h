/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM_GRAPH_MC2
#define HCOM_GRAPH_MC2

#include "graph/op_desc.h"
#include "hccl/hcom.h"

namespace hccl {
bool HcomGetGroupsByOpDesc(const ge::OpDescPtr &opdesc, std::vector<std::string> &groups);
u32 HcomGetTilingVersionByOpDesc(const ge::OpDescPtr &opdesc, std::string &tilingData);
rtStream_t HcomGetStreamByOpDesc(const ge::OpDescPtr &opdesc);
void *HcomGetContext(const rtStream_t stream, const void *tilingData, const char *groupName);
HcclResult GetCountFromOpDesc(const ge::OpDescPtr &op, const std::string &sCollectiveType, HcclDataType dataType,
                              u64 &count, u32 rankSize);
ge::graphStatus HcomCreateComResourceMC2(const ge::OpDescPtr &opdesc, std::vector<void *> &contexts);
ge::graphStatus HcomCreateComResource(const ge::OpDescPtr &opdesc, std::vector<void *> &contexts);
#ifndef OPEN_BUILD_PROJECT
void *HcomGetContextV2(const rtStream_t stream, const void *tilingData, const char *groupName);
#endif
}  // namespace hccl

#endif /* HCOM_GRAPH_MC2 */