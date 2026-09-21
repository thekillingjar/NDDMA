/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_COMMON_HELPER_MOBILE_COMPILED_MODEL_H
#define BASE_COMMON_HELPER_MOBILE_COMPILED_MODEL_H

#include <vector>
#include <memory>
#include <string>
#include <set>
#include <algorithm>

#include "base_buffer.h"
#include "compiled_target.h"
#include "model_buffer_helper.h"
#include "ge_common/ge_api_error_codes.h"
#include "graph/compute_graph.h"
#include "common/model/ge_model.h"
#include "proto/ge_ir_mobile.pb.h"

namespace ge {

class CompiledModel;
using CompiledModelPtr = std::shared_ptr<CompiledModel>;

class CompiledModel {
public:
    CompiledModel() = default;
    ~CompiledModel() = default;

    CompiledModel(CompiledModel const&) = delete;
    CompiledModel& operator=(CompiledModel const&) = delete;

    Status SaveToBuffer(
        ge::BaseBuffer& buffer,
        bool save_weights_as_external_data = false,
        std::map<std::string, ge::BaseBuffer>* weights_list_external = nullptr);

    inline void SetGeModel(ge::GeModelPtr ge_model)
    {
        ge_model_ = ge_model;
    }

    inline void SetWeightsList(const std::vector<ge::BaseBuffer>& weights_list)
    {
        weights_list_ = weights_list;
    }

    inline void AddWeight(const ge::BaseBuffer &weight_buffer)
    {
        weights_list_.push_back(weight_buffer);
    }

    inline void SetCompiledTargets(const std::vector<ge::CompiledTargetPtr>& compiled_targets)
    {
        compiled_targets_ = compiled_targets;
    }

    inline void AddCompiledTarget(ge::CompiledTargetPtr compiled_targets)
    {
        compiled_targets_.push_back(compiled_targets);
    }

private:
    Status GetCompiledTargetsBuffer(std::vector<ge::BaseBuffer>& all_targets_buffer);

private:
    ge::GeModelPtr ge_model_;
    std::vector<ge::BaseBuffer> weights_list_;
    std::vector<ge::CompiledTargetPtr> compiled_targets_;
    ge::mobile::proto::ModelDef mobile_model_def_;
    std::vector<std::vector<uint8_t>> compiled_targets_buffer_;
    ge::ModelBufferSaver saver_;
};

} // namespace ge

#endif // BASE_COMMON_HELPER_MOBILE_COMPILED_MODEL_H
