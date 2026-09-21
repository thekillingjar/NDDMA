/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCFDD0816_CC46_4264_9363_9E8C6934F43E
#define HCFDD0816_CC46_4264_9363_9E8C6934F43E

#include "easy_graph/eg.h"
#include "easy_graph/graph/node_id.h"
#include "graph/op_desc.h"
#include "ge_graph_dsl/ge.h"
#include "ge_graph_dsl/op_desc/op_box.h"

GE_NS_BEGIN

struct OpDescPtrBox : OpBox {
  OpDescPtrBox(const OpDescPtr &op) : op_(op) {}

 private:
  OpDescPtr Build(const ::EG_NS::NodeId &id) const override;
  const OpDescPtr op_;
};

GE_NS_END

#endif /* HCFDD0816_CC46_4264_9363_9E8C6934F43E */
