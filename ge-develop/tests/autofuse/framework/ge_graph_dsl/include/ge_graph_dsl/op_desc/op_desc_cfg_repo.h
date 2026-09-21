/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H600DEDD4_D5B9_4803_AF48_262B2C4FBA94d
#define H600DEDD4_D5B9_4803_AF48_262B2C4FBA94c

#include "easy_graph/infra/singleton.h"
#include "ge_graph_dsl/ge.h"
#include "ge_graph_dsl/op_desc/op_type.h"

GE_NS_BEGIN

struct OpDescCfg;

SINGLETON(OpDescCfgRepo) {
  const OpDescCfg *FindBy(const OpType &);
};

GE_NS_END

#endif /* H600DEDD4_D5B9_4803_AF48_262B2C4FBA94 */
