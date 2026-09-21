/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_COMMON_GRAPH_TUNER_ERRORCODE_H_
#define INC_COMMON_GRAPH_TUNER_ERRORCODE_H_

#include <cstdint>

namespace tune {
    using Status = uint32_t;

    constexpr uint8_t SYSID_TUNE = 10U;

    constexpr uint8_t TUNE_MODID_LX = 30U;
} // namespace tune

#define TUNE_DEF_ERRORNO(sysid, modid, name, value, desc)                                                      \
    constexpr tune::Status name = ((((static_cast<uint32_t>(0xFFU & (static_cast<uint8_t>(sysid)))) << 24U) |  \
        ((static_cast<uint32_t>(0xFFU & (static_cast<uint8_t>(modid)))) << 16U)) |                              \
        (0xFFFFU & (static_cast<uint16_t>(value))))

#define TUNE_DEF_ERRORNO_LX(name, value, desc) TUNE_DEF_ERRORNO(SYSID_TUNE, TUNE_MODID_LX, name, value, desc)

namespace tune {
  TUNE_DEF_ERRORNO(0U,    0U,     SUCCESS,  0U,      "Success");
  TUNE_DEF_ERRORNO(0xFFU, 0xFFU,  FAILED,   0xFFFFU, "Failed");

  TUNE_DEF_ERRORNO_LX(NO_FUSION_STRATEGY,  10U, "Lx fusion strategy is invalid");
  TUNE_DEF_ERRORNO_LX(UNTUNED,             20U, "Untuned");
  TUNE_DEF_ERRORNO_LX(NO_UPDATE_BANK,      30U, "Model Bank is not updated!");
  TUNE_DEF_ERRORNO_LX(HIT_FUSION_STRATEGY, 40U, "Hit lx fusion strategy.");
} // namespace tune
#endif  // INC_COMMON_GRAPH_TUNER_ERRORCODE_H_
 