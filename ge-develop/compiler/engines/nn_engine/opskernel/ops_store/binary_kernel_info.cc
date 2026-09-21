/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_store/binary_kernel_info.h"
#include <nlohmann/json.hpp>
#include "common/string_utils.h"
#include "common/fe_log.h"
#include "ops_store/ops_kernel_utils.h"
#include "ops_store/ops_kernel_error_codes.h"
#include "ops_store/ops_kernel_constants.h"

namespace fe {
namespace {
  const std::string kTrueStr = "true";
}

BinaryKernelInfo &BinaryKernelInfo::Instance() {
  static BinaryKernelInfo bin_kernel_info;
  return bin_kernel_info;
}

Status BinaryKernelInfo::Initialize(const std::string &config_file_path) {
  if (is_init_) {
    return SUCCESS;
  }
  FE_LOGD("[InitBinKernelInfoStore] Begin to initialize binary kernel info by path [%s].", config_file_path.c_str());
  nlohmann::json op_json_file;
  if (ReadJsonObject(config_file_path, op_json_file) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init][LoadOpJsonFile] Failed to read JSON object from %s.", config_file_path.c_str());
    return FAILED;
  }
  try {
    if (!op_json_file.is_object()) {
      REPORT_FE_ERROR("[GraphOpt][Init][LoadOpJsonFile] The top level of the JSON file should be an object, but it is actually %s.",
                      GetJsonObjectType(op_json_file).c_str());
      return OP_SUB_STORE_ILLEGAL_JSON;
    }
    std::vector<string> op_type_vec;
    for (auto &elem : op_json_file.items()) {
      std::string op_type_temp = elem.key();
      op_type_vec.push_back(StringUtils::Trim(op_type_temp));
    }
    for (auto &op_type : op_type_vec) {
      if (!op_json_file[op_type].is_object()) {
        REPORT_FE_ERROR("[GraphOpt][Init][LoadOpJsonFile] The second level of the JSON file should be an object, but it is actually %s.",
                        GetJsonObjectType(op_json_file[op_type]).c_str());
        return OP_SUB_STORE_ILLEGAL_JSON;
      }
      const nlohmann::json &op_info = op_json_file[op_type];
      const auto &iter = op_info.find(kStrDynamicRankSupport);
      if (iter != op_info.end()) {
        const std::string &val = iter.value().get<std::string>();
        if (val == kTrueStr) {
            support_dynamic_rank_ops_.emplace(op_type);
        }
      }
    }
  } catch (const std::exception &e) {
    REPORT_FE_ERROR("[GraphOpt][Init][LoadBinaryOpJsonFile] Failed to convert file [%s] to JSON. Error message: %s",
                    config_file_path.c_str(), e.what());
    return OP_SUB_STORE_ILLEGAL_JSON;
  }
  FE_LOGD("[InitBinKernelInfoStore] Finish to initialize binary kernel info.");
  is_init_ = true;
  return SUCCESS;
}

bool BinaryKernelInfo::IsBinSupportDynamicRank(const std::string &op_type) const {
  if (support_dynamic_rank_ops_.find(op_type) != support_dynamic_rank_ops_.end()) {
    return true;
  }
  return false;
}

void BinaryKernelInfo::Finalize() {
  support_dynamic_rank_ops_.clear();
  is_init_ = false;
}
}
