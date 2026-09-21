/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pass/pass_mgr.h"
#include "nlohmann/json.hpp"

namespace att {
const std::string kLoadToL1 = "loadTcsm";
const uint32_t kNd2NzAlign = 128U;

struct MatmulConfigItem {
  std::map<std::string, uint32_t> align_true;
  std::map<std::string, uint32_t> align_false;
};

using MatmulConfig = std::map<std::string, MatmulConfigItem>;

void ToJson(nlohmann::json &j, const MatmulConfigItem &p) {
  j = nlohmann::json{
    {"0", p.align_false},
    {"1", p.align_true}
  };
}

bool GetMatmulAlignConfig(const TuningSpacePtr &tuning_space, std::map<std::string, std::string> &matmul_config) {
  for (const auto &node : tuning_space->node_infos) {
    if ((node.node_type != kLoadToL1) && node.trans_config.empty()) {
      continue;
    }
    GE_ASSERT_TRUE(!node.outputs.empty(), "Node [%s] is LoadL1 but has no output.", node.name.c_str());
    MatmulConfigItem config;
    const auto &dims = node.outputs[0]->dim_info;
    if (dims.empty()) {
      GELOGW("Node [%s] has no dims.", node.name.c_str());
      nlohmann::json j;
      ToJson(j, config);
      matmul_config.emplace(node.trans_config, j.dump());
      continue;
    }
    size_t dim_size = node.outputs[0]->dim_info.size();
    if (dim_size > 0U) {
      config.align_false.emplace(node.outputs[0]->dim_info[dim_size - 1U]->name, kNd2NzAlign);
    }
    if (dim_size > 1U) {
      config.align_true.emplace(node.outputs[0]->dim_info[dim_size - 2U]->name, kNd2NzAlign);
    }
    nlohmann::json j;
    ToJson(j, config);
    matmul_config.emplace(node.trans_config, nlohmann::to_string(j));
  }

  return true;
}

static std::string kmatmul_align_pass = "matmul_align_pass";
REGISTER_GTC_PASS(kmatmul_align_pass, GetMatmulAlignConfig);
}  // namespace att