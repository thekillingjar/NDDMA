/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <regex>
#include "api/aclgrph/attr_options/attr_options.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/util.h"
#include "base/err_msg.h"

namespace ge {
namespace {
const size_t kMaxOpsNum = 10;
const std::regex kPascalCaseRegex(R"(^[A-Z][a-zA-Z0-9]*$)");
}  // namespace

inline bool IsValidOpType(const std::string &name) {
  return std::regex_match(name, kPascalCaseRegex);
}

static bool KeepDtypeReportError(const std::vector<std::string> &invalid_list, const std::string &cfg_path) {
  std::stringstream warn_msg;
  size_t list_size = invalid_list.size();
  warn_msg << "config file contains " << list_size;
  if (list_size == 1) {
    warn_msg << " operator not in the graph, ";
  } else {
    warn_msg << " operators not in the graph, ";
  }
  std::string cft_type;
  for (size_t i = 0; i < list_size; i++) {
    if (i == kMaxOpsNum) {
      warn_msg << "..";
      break;
    }
    bool istype = IsContainOpType(invalid_list[i], cft_type);
    if (!istype) {
      warn_msg << "op name:[";
    } else if (cft_type.empty() || !IsValidOpType(cft_type)) {
      std::stringstream err_msg;
      err_msg << "The operator type [" << cft_type << "] in the config file is invalid. It must be in PascalCase.";
      REPORT_PREDEFINED_ERR_MSG("E10003", std::vector<const char_t *>({"parameter", "value", "reason"}),
                                std::vector<const char_t *>({"keep_dtype", cfg_path.c_str(), err_msg.str().c_str()}));
      GELOGE(FAILED, "%s", err_msg.str().c_str());
      return true;
    } else {
      warn_msg << "op type:[";
    }
    warn_msg << cft_type << "]";
    if (i != (list_size - 1)) {
      warn_msg << " ";
    }
  }
  GELOGW("%s", warn_msg.str().c_str());
  return false;
}

graphStatus KeepDtypeFunc(const ComputeGraphPtr &graph, const std::string &cfg_path) {
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
    GELOGE(GRAPH_FAILED, "[Open][File] %s failed.", cfg_path.c_str());
    REPORT_PREDEFINED_ERR_MSG("E13001", std::vector<const char_t *>({"file", "errmsg"}),
                       std::vector<const char_t *>({cfg_path.c_str(), "Open file failed"}));
    return GRAPH_FAILED;
  }

  std::string op_name, op_type;
  std::vector<std::string> invalid_list;
  while (std::getline(ifs, op_name)) {
    if (op_name.empty()) {
      continue;
    }
    op_name = StringUtils::Trim(op_name);
    bool is_find = false;
    bool is_type = IsContainOpType(op_name, op_type);
    for (auto &node_ptr : graph->GetAllNodes()) {
      auto op_desc = node_ptr->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      if (is_type) {
        if (IsOpTypeEqual(node_ptr, op_type)) {
          is_find = true;
          (void)AttrUtils::SetInt(op_desc, ATTR_NAME_KEEP_DTYPE, 1);
        }
      } else {
        if (op_desc->GetName() == op_name || IsOriginalOpFind(op_desc, op_name)) {
          is_find = true;
          (void)AttrUtils::SetInt(op_desc, ATTR_NAME_KEEP_DTYPE, 1);
        }
      }
    }
    if (!is_find) {
      invalid_list.push_back(op_name);
    }
  }
  ifs.close();

  if (!invalid_list.empty() && KeepDtypeReportError(invalid_list, cfg_path)) {
    return GRAPH_PARAM_INVALID;
  }

  return GRAPH_SUCCESS;
}
}  // namespace ge
