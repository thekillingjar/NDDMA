/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDEX_GENERATOR_FACTORY_H
#define INDEX_GENERATOR_FACTORY_H

#include <memory>
#include "compress.h"
#include "mode_a_index_generator.h"
#include "mode_b_index_generator.h"
#include "mode_warehouse.h"

namespace amctcmp {
class IndexGeneratorFactory {
public:
    std::shared_ptr<IndexGenerator> GetIndexGenerator(CompressMode mode,
                                                      char* dataBuffer,
                                                      char* indexBuffer,
                                                      const CompressConfig& compressConfig)
    {
        if (mode == MODE_A) {
            return std::make_shared<ModeAIndexGenerator>(dataBuffer,
                                                         indexBuffer,
                                                         compressConfig.isTight,
                                                         compressConfig.fractalSize,
                                                         compressConfig.init_offset);
        } else if (mode == MODE_B) {
            return std::make_shared<ModeBIndexGenerator>(dataBuffer,
                                                         indexBuffer,
                                                         compressConfig.isTight,
                                                         compressConfig.fractalSize,
                                                         compressConfig.init_offset);
        } else {
            return nullptr;
        }
    }
};
} // amctcmp
#endif // INDEX_GENERATOR_FACTORY_H
