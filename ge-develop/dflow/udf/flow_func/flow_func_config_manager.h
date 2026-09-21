/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_FLOW_FUNC_CONFIG_MANAGER_H
#define UDF_FLOW_FUNC_CONFIG_MANAGER_H

#include <memory>
#include "flow_func_config.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY FlowFuncConfigManager {
public:
    static void SetConfig(const std::shared_ptr<FlowFuncConfig> &config);

    static const std::shared_ptr<FlowFuncConfig> &GetConfig();

private:
    static std::shared_ptr<FlowFuncConfig> config_;
};
}
#endif // UDF_FLOW_FUNC_CONFIG_MANAGER_H
