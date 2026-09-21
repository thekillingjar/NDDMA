/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_COMMON_HELPER_MOBILE_MODEL_BUFFER_HELPER_H
#define BASE_COMMON_HELPER_MOBILE_MODEL_BUFFER_HELPER_H

#include <vector>

#include "base_buffer.h"
#include "ge_common/ge_api_error_codes.h"
#include "graph/compute_graph.h"
#include "common/model/ge_model.h"
#include "proto/ge_ir_mobile.pb.h"

namespace ge {

class ModelBufferSaver {
public:
    Status SaveCompiledModelToBuffer(
        const ge::GeModelPtr& ge_model,
        const ge::mobile::proto::ModelDef& mobile_model_def,
        const std::vector<ge::BaseBuffer>& weights_buffer,
        const std::vector<ge::BaseBuffer>& all_compiled_targets,
        const ge::BaseBuffer& weights_info_buffer,
        ge::BaseBuffer& dst_buffer);

private:
    std::vector<uint8_t> compiled_model_buffer_; 
};

} // namespace ge

#endif // BASE_COMMON_HELPER_MOBILE_MODEL_BUFFER_HELPER_H
