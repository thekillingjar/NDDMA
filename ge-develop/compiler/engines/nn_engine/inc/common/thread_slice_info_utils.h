/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FE_THREAD_SLICE_INFO_UTILS_H
#define FE_THREAD_SLICE_INFO_UTILS_H
#include <nlohmann/json.hpp>
#include "common/sgt_slice_type.h"
namespace ffts {
void to_json(nlohmann::json &json_value, const DimRange &struct_value);
void to_json(nlohmann::json &j, const OpCut &d);
void to_json(nlohmann::json &json_value, const ThreadSliceMap &struct_value);
void from_json(const nlohmann::json &j, OpCut &d);
void from_json(const nlohmann::json &json_value, DimRange &struct_value);
void from_json(const nlohmann::json &json_value, ThreadSliceMap &struct_value);
void GetSliceJsonInfoStr(const ThreadSliceMap &struct_value, std::string &string_str);
void GetSliceInfoFromJson(ThreadSliceMap &value, const std::string &json_str);
}  // namespace ffts

#endif
