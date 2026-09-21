/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_GE_MODEL_INOUT_TYPES_H_
#define INC_FRAMEWORK_COMMON_GE_MODEL_INOUT_TYPES_H_
#include <cstdint>
#include <string>
#include <vector>
#include <map>

#include "graph/types.h"

namespace ge {
    struct ModelInOutTensorDesc {
    std::string name;
    size_t size;
    Format format;
    DataType dataType;
    std::vector<int64_t> dims;
    std::vector<int64_t> dimsV2;  // supported for static aipp scene
    std::vector<std::pair<int64_t, int64_t>> shape_ranges;
    };

    struct ModelInOutInfo {
    std::vector<ModelInOutTensorDesc> input_desc;
    std::vector<ModelInOutTensorDesc> output_desc;
    std::vector<uint64_t> dynamic_batch;
    std::vector<std::vector<uint64_t>> dynamic_hw;
    std::vector<std::vector<uint64_t>> dynamic_dims;
    std::vector<std::string> dynamic_output_shape;
    std::vector<std::string> data_name_order;
    };

    enum class ModelDescType : uint32_t {
    MODEL_INPUT_DESC,
    MODEL_OUTPUT_DESC,
    MODEL_DYNAMIC_BATCH,
    MODEL_DYNAMIC_HW,
    MODEL_DYNAMIC_DIMS,
    MODEL_DYNAMIC_OUTPUT_SHAPE,
    MODEL_DESIGNATE_SHAPE_ORDER
    };

    struct ModelDescTlvConfig {
    int32_t type = 0;
    uint32_t length = 0U;
    const uint8_t *value = nullptr;
    };

    struct ModelTensorDescBaseInfo {
    size_t size = 0U;
    Format format;
    DataType dt;
    uint32_t name_len = 0U;
    uint32_t dims_len = 0U;
    uint32_t dimsV2_len = 0U;
    uint32_t shape_range_len = 0U;
    };
}  // namespace ge
#endif
