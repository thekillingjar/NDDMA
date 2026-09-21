/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_COMMON_HELPER_MOBILE_COMPILED_TARGET_H
#define BASE_COMMON_HELPER_MOBILE_COMPILED_TARGET_H

#include <memory>
#include <cstddef>
#include <vector>

#include "proto/task_mobile.pb.h"
#include "base_buffer.h"
#include "ge_common/ge_api_error_codes.h"
#include "common/model/ge_model.h"
#include "proto/ge_ir_mobile.pb.h"

namespace ge {

enum class SectionType : std::uint8_t {
    SECTION_TYPE_INVAILD = 0,
    SECTION_TYPE_MODELDEF = 1,
    SECTION_TYPE_BIN = 2,
    SECTION_TYPE_COMPATIBLE = 3,
    SECTION_TYPE_ENCYPTION = 4,
    SECTION_TYPE_COMPATIBLEV2 = 5,
    SECTION_TYPE_EFFICIENT_MODELDEF = 6,
    SECTION_TYPE_GENERIC_MODELDEF = 7,
    SECTION_TYPE_PLACEHOLDER = 8,
    SECTION_TYPE_MAX = 255,
};

struct SectionHolder {
    uint32_t type;
    uint32_t size;
    std::unique_ptr<uint8_t[]> data;
};

class CompiledTarget;
using CompiledTargetPtr = std::shared_ptr<CompiledTarget>;

class CompiledTarget {
public:
    CompiledTarget() = default;
    ~CompiledTarget() = default;

    CompiledTarget(const CompiledTarget&) = delete;
    CompiledTarget& operator=(const CompiledTarget&) = delete;

    ge::Status Serialize(ge::BaseBuffer& buffer);

    std::size_t GetSize() const;

    void AddSection(SectionHolder& section);

    void AddModelTaskDef(std::shared_ptr<mobile::proto::ModelTaskDef> task_def);

    uint32_t SerializeModelTask(uint8_t* addr, uint32_t space_size, uint32_t section_type);

    ge::Status UpdateModelModelTaskDef(ge::mobile::proto::ModelDef& mobile_model_def) const;

private:
    void UpdateGraphOpTaskSize(ge::mobile::proto::ModelDef& mobile_model_def) const;

private:
    std::vector<SectionHolder> sections_;
    std::shared_ptr<mobile::proto::ModelTaskDef> model_task_;
};

} // namespace ge

#endif // BASE_COMMON_HELPER_MOBILE_COMPILED_TARGET_H
