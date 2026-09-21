/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_TEST_ENGINES_NNENG_GRAPH_CONSTRUCTOR_FE_LLT_UTILS_H_
#define AIR_TEST_ENGINES_NNENG_GRAPH_CONSTRUCTOR_FE_LLT_UTILS_H_

#include <string>
#include <nlohmann/json.hpp>
#include "graph/compute_graph.h"
#include "common/aicore_util_types.h"

namespace fe {
std::string GetCodeDir();
std::string GetGraphPath(const std::string &graph_name);
uint32_t InitPlatformInfo(const std::string &soc_version, const bool is_force = false);
void SetPlatformSocVersion(const std::string &soc_version);
void SetPrecisionMode(const std::string &precision_mode);
void SetContextOption(const std::string &key, const std::string &value);
void InitWithSocVersion(const std::string &soc_version, const std::string &precision_mode);
void InitWithOptions(const std::map<std::string, std::string> &options);
void FillWeightValue(const ge::ComputeGraphPtr &graph);
void FillGraphNodeParaType(const ge::ComputeGraphPtr &graph, fe::OpParamType type = fe::OpParamType::REQUIRED);
void FillNodeParaType(const ge::NodePtr &node, fe::OpParamType type = fe::OpParamType::REQUIRED);
void CreateDir(const std::string &path);
void CreateFileAndFillContent(const std::string fileName,
                              nlohmann::json json_obj = nlohmann::json::object(), const bool flag = false);
void CreateAndCopyJsonFile();
void DelJsonFile();
}
#endif  // AIR_TEST_ENGINES_NNENG_GRAPH_CONSTRUCTOR_FE_LLT_UTILS_H_
