/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM_GRAPH_SUPERKERNEL
#define HCOM_GRAPH_SUPERKERNEL

#include "graph/op_desc.h"

namespace hccl {
bool GetAddrFromDesc(const ge::OpDescPtr &op, int64_t &inputAddr, int64_t &outputAddr, int64_t &workSpaceAddr);
bool GetGraphIdFromDesc(const ge::OpDescPtr &op, int32_t &graphId);
ge::graphStatus HcomGetSuperKernelHiddenInputs(const ge::OpDescPtr &opdesc, std::vector<void *> &contexts);
ge::graphStatus HcomCreateSuperkernelResource(const ge::OpDescPtr &opdesc, std::vector<void *> &contexts);
}  // namespace hccl

#endif /* HCOM_GRAPH_SUPERKERNEL */