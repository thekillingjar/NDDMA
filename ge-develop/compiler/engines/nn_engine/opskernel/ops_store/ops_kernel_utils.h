/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPSKERNEL_OPS_KERNEL_STORE_OPS_KERNEL_UTILS_H_
#define FUSION_ENGINE_OPSKERNEL_OPS_KERNEL_STORE_OPS_KERNEL_UTILS_H_

#include <string>
#include <nlohmann/json.hpp>
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "ops_store/op_kernel_info.h"

namespace fe {
using std::vector;
using IndexNameMap = std::map<uint32_t, std::string>;

Status ReadJsonObject(const std::string& file, nlohmann::json& json_obj);
std::string GetJsonObjectType(const nlohmann::json &json_object);

/*
 *  @ingroup fe
 *  @brief   compare inputs
 *  @param   [in]  input std::string
 *  @param   [in]  input std::string
 *  @return  true or false
 */
bool CmpInputsNum(std::string input1, std::string input2);

/*
 *  @ingroup fe
 *  @brief   compare outputs
 *  @param   [in]  output std::string
 *  @param   [in]  output std::string
 *  @return  true or false
 */
bool CmpOutputsNum(std::string output1, std::string output2);

bool CheckInputSubStr(const std::string& op_desc_input_name, const std::string& info_input_name);

Status GenerateUnionFormatAndDtype(const vector<ge::Format>& old_formats, const vector<ge::DataType>& old_data_types,
                                   vector<ge::Format>& new_formats, vector<ge::DataType>& new_data_types);

std::string StringVecToString(const std::vector<std::string> &str_vec);
}  // namespace fe

#endif  // FUSION_ENGINE_OPSKERNEL_OPS_KERNEL_STORE_OPS_KERNEL_UTILS_H_
