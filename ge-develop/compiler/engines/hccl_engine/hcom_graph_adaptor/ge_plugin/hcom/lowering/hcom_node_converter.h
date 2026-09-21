/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCOM_NODE_CONVERTER_H
#define HCOM_NODE_CONVERTER_H
#include "register/node_converter_registry.h"
#include "hccl/base.h"
#include "hcom_log.h"

namespace hccl {
gert::LowerResult LoweringHcomNode(const ge::NodePtr &node, const gert::LowerInput &lowerInput);
gert::LowerResult LoweringAlltoAllNode(const ge::NodePtr &node, const gert::LowerInput &lowerInput);
gert::LowerResult LoweringRecvNode(const ge::NodePtr &node, const gert::LowerInput &lowerInput);

gert::bg::DevMemValueHolderPtr MakeSureInputAtHost(const ge::NodePtr &node, const gert::LowerInput &lowerInput,
                                                   u32 index);
std::vector<gert::bg::DevMemValueHolderPtr> MakeSureCommAlltoAllInput(const ge::NodePtr &node,
                                                                      const gert::LowerInput &lower_input);
}  // namespace hccl
#endif  // HCOM_NODE_CONVERTER_H