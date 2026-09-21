/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H550E4ACB_BEC7_4E71_8C6F_CD7FA53662A9
#define H550E4ACB_BEC7_4E71_8C6F_CD7FA53662A9

#include "easy_graph/infra/status.h"
#include "easy_graph/infra/singleton.h"

EG_NS_BEGIN

struct LayoutExecutor;
struct LayoutOption;
struct Graph;

SINGLETON(GraphLayout) {
  void Config(LayoutExecutor &, const LayoutOption * = nullptr);
  Status Layout(const Graph &, const LayoutOption * = nullptr);

 private:
  LayoutExecutor *executor_{nullptr};
  const LayoutOption *options_{nullptr};
};

EG_NS_END

#endif
