/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base/err_msg.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "api/aclgrph/attr_options/attr_options.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/util.h"
#include "base/err_msg.h"

namespace ge {
graphStatus WeightCompressFunc(const ComputeGraphPtr &graph, const std::string &cfg_path) {
  GE_CHECK_NOTNULL(graph);
  if (cfg_path.empty()) {
    return GRAPH_SUCCESS;
  }
  std::string real_path = RealPath(cfg_path.c_str());
  if (real_path.empty()) {
    GELOGE(GRAPH_PARAM_INVALID, "[Get][Path]Can not get real path for %s.", cfg_path.c_str());
    REPORT_PREDEFINED_ERR_MSG("E10410", std::vector<const char_t *>({"cfgpath"}), std::vector<const char_t *>({cfg_path.c_str()}));
    return GRAPH_PARAM_INVALID;
  }
  std::ifstream ifs(real_path);
  if (!ifs.is_open()) {
    GELOGE(GRAPH_FAILED, "[Open][File] %s failed", cfg_path.c_str());
    REPORT_PREDEFINED_ERR_MSG("E13001", std::vector<const char_t *>({"file", "errmsg"}),
                       std::vector<const char_t *>({cfg_path.c_str(), "Open file failed"}));
    return GRAPH_FAILED;
  }

  std::string compress_nodes;
  ifs >> compress_nodes;
  ifs.close();
  GELOGI("Compress weight of nodes: %s", compress_nodes.c_str());

  std::vector<std::string> compress_node_vec = StringUtils::Split(compress_nodes, ';');
  std::string node_names;
  for (size_t i = 0; i < compress_node_vec.size(); ++i) {
    bool is_find = false;
    for (auto &node_ptr : graph->GetDirectNode()) {
      GE_CHECK_NOTNULL(node_ptr);
      auto op_desc = node_ptr->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);

      if ((op_desc->GetName() == compress_node_vec[i]) || IsOriginalOpFind(op_desc, compress_node_vec[i])) {
        is_find = true;
        if (!ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_COMPRESS_WEIGHT, true)) {
          GELOGE(GRAPH_FAILED, "[Set][Bool] failed, node:%s.", compress_node_vec[i].c_str());
          REPORT_INNER_ERR_MSG("E19999", "SetBool failed, node:%s.", compress_node_vec[i].c_str());
          return GRAPH_FAILED;
        }
      }
    }
    if (!is_find) {
      node_names = node_names + "[" + compress_node_vec[i].c_str() + "] ";
      GELOGW("node %s is not in graph", compress_node_vec[i].c_str());
    }
  }
  if (!node_names.empty()) {
    REPORT_PREDEFINED_ERR_MSG(
        "W11002", std::vector<const char *>({"filename", "opnames"}),
        std::vector<const char *>({real_path.c_str(), node_names.c_str()}));
    GELOGW("In the compression weight configuration file [%s], some nodes do not exist in graph: %s.",
           real_path.c_str(), node_names.c_str());
  }
  return GRAPH_SUCCESS;
}
}  // namespace ge
