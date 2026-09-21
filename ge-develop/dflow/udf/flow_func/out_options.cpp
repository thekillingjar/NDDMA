/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func/out_options.h"
#include "flow_func/out_options_impl.h"
#include "common/util.h"

namespace FlowFunc {
namespace {
OutOptionsImpl g_default_options;
}

OutOptions::OutOptions() {
    impl_ = MakeShared<OutOptionsImpl>();
}

OutOptions::~OutOptions() {
}

BalanceConfig *OutOptions::MutableBalanceConfig() {
    if (impl_ != nullptr) {
        return impl_->MutableBalanceConfig();
    } else {
        return g_default_options.MutableBalanceConfig();
    }
}

const BalanceConfig *OutOptions::GetBalanceConfig() const {
    if (impl_ != nullptr) {
        return impl_->GetBalanceConfig();
    } else {
        return g_default_options.GetBalanceConfig();
    }
}
}