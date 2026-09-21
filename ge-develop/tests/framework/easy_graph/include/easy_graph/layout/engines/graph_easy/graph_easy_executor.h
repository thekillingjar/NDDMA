/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HAC96EB3A_2169_4BB0_A8EB_7B966C262B2F
#define HAC96EB3A_2169_4BB0_A8EB_7B966C262B2F

#include "easy_graph/layout/layout_executor.h"

EG_NS_BEGIN

struct GraphEasyExecutor : LayoutExecutor {
 private:
  Status Layout(const Graph &, const LayoutOption *) override;
};

EG_NS_END

#endif
