/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge_graph_default_checker.h"

GE_NS_BEGIN

GeGraphDefaultChecker::GeGraphDefaultChecker(const std::string &phase_id, const GraphCheckFun &check_fun)
    : phase_id_(phase_id), check_fun_(check_fun) {}

const std::string &GeGraphDefaultChecker::PhaseId() const { return phase_id_; }

void GeGraphDefaultChecker::Check(const ge::ComputeGraphPtr &graph) const { return check_fun_(graph); }

GE_NS_END
