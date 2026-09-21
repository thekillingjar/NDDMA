/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_HELPER_MOBILE_MODEL_HELPER_H
#define INC_FRAMEWORK_COMMON_HELPER_MOBILE_MODEL_HELPER_H

#include <memory>
#include <string>

#include "framework/common/helper/model_save_helper.h"
#include "framework/common/types.h"
#include "graph/model.h"
#include "platform/platform_info.h"
#include "common/op_so_store/op_so_store.h"
#include "common/host_resource_center/host_resource_serializer.h"

namespace ge {

class GE_FUNC_VISIBILITY MobileModelHelper : public ModelSaveHelper {
public:
    MobileModelHelper() noexcept = default;

    ~MobileModelHelper() override = default;

    Status SaveToOmRootModel(
        const GeRootModelPtr& ge_root_model,
        const std::string& output_file,
        ModelBufferData& model,
        const bool is_unknown_shape) override;

    Status SaveToOmModel(
        const GeModelPtr& ge_model,
        const std::string& output_file,
        ModelBufferData& model,
        const GeRootModelPtr& ge_root_model = nullptr) override;

    void SetSaveMode(const bool val) override;

private:
    bool is_offline_ {true};
};

}  // namespace ge

#endif  // INC_FRAMEWORK_COMMON_HELPER_MOBILE_MODEL_HELPER_H
