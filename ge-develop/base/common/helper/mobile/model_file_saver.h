/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_COMMON_HELPER_MOBILE_MODEL_FILE_SAVER_H
#define BASE_COMMON_HELPER_MOBILE_MODEL_FILE_SAVER_H

#include <map>
#include <string>

#include "base_buffer.h"
#include "compiled_model.h"
#include "ge_common/ge_api_error_codes.h"

namespace ge {

class ModelFileSaver {
public:
    ModelFileSaver() = default;

    ~ModelFileSaver() = default;

public:
    Status SaveCompiledModelToFile(
        std::shared_ptr<CompiledModel> model,
        const char* output_model_file,
        const char* output_weight_dir,
        bool save_weights_as_external_data = false) const;

    Status SaveWeightBufferToFile(
        const char* output_weight_dir,
        const std::map<std::string, ge::BaseBuffer>& weights_list_external) const;
};

} // namespace ge

#endif // BASE_COMMON_HELPER_MOBILE_MODEL_FILE_SAVER_H
