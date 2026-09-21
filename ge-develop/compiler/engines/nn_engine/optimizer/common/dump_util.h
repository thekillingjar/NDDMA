/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_COMMON_DUMP_UTIL_H_
#define FUSION_ENGINE_OPTIMIZER_COMMON_DUMP_UTIL_H_

#include <string>
#include <vector>
#include "graph/node.h"
#include "tensor_engine/tbe_op_info.h"

namespace fe {
/*
 *  @ingroup fe
 *  @brief   get input and output's name
 *  @param   [in]  op                 op desc
 *  @param   [in]  info_store          ops info store pointer
 *  @param   [in/out] op_kernel_info_ptr infos of op
 *  @param   [in/out] input_map        map to recorde inputs's name
 *  @param   [in/out] output_map       map to recorde outputs's name
 *  @return  SUCCESS or FAILED
 */
void DumpOpInfoTensor(const std::vector<te::TbeOpParam> &op_params, std::string &debug_str);

void DumpOpInfo(const te::TbeOpInfo &op_info);

void DumpL1Attr(const ge::Node *node);

void DumpL2Attr(const ge::Node *node);
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_COMMON_DUMP_UTIL_H_
