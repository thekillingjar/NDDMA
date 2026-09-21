/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H13960AED_B5B1_45F9_A664_6CB6C15CA3C1
#define H13960AED_B5B1_45F9_A664_6CB6C15CA3C1

#include "easy_graph/infra/keywords.h"
#include "easy_graph/infra/status.h"

EG_NS_BEGIN

struct Graph;
struct Node;
struct Edge;

INTERFACE(GraphVisitor) {
  DEFAULT(Status, Visit(const Graph &));
  DEFAULT(Status, Visit(const Node &));
  DEFAULT(Status, Visit(const Edge &));
};

EG_NS_END

#endif
