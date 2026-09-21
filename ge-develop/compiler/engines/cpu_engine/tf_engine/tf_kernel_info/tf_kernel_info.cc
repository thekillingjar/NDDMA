/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tf_kernel_info.h"

#include <mutex>
#include "config/ops_json_file.h"
#include "util/constant.h"
#include "util/log.h"
#include "util/tf_util.h"
#include "util/util.h"
#include "error_code/error_code.h"
#include "ir2tf/ir2tf_parser_factory.h"

#include "base/err_msg.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"

using domi::tensorflow::NodeDef;

namespace aicpu {
KernelInfoPtr TfKernelInfo::instance_ = nullptr;

inline KernelInfoPtr TfKernelInfo::Instance() {
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    instance_.reset(new (std::nothrow) TfKernelInfo);
  });
  return instance_;
}

ge::Status TfKernelInfo::Initialize(const std::map<std::string, std::string> &options) {
  ge::Status status = KernelInfo::Initialize(options);
  if (status != ge::SUCCESS) {
    return status;
  }
  LoadSupportedOps();
  std::shared_ptr<Ir2tfBaseParser> parser = Ir2tfBaseParser::Instance();
  if (parser == nullptr) {
    AICPU_REPORT_INNER_ERR_MSG("Create Ir2tfBaseParser object failed.");
    return ErrorCode::GET_IR2TF_PARSER_FAILED;
  }
  return parser->LoadMappingConfig();
}

bool TfKernelInfo::ReadOpInfoFromJsonFile() {
  std::string real_tf_ops_file_path;
  const char* path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, path_env);
  std::string path;
  std::string env_path;
  if (path_env != nullptr) {
    env_path = std::string(path_env);
    path = env_path + kTfOpsFileBasedOnEnvPath;
    if (IsPathExist(path)) {
      real_tf_ops_file_path = path;
    } else {
      real_tf_ops_file_path = env_path + kTfOpsFileBasedOnEnvPathOld;
    }
  } else {
    std::string file_path = GetOpsPath(reinterpret_cast<void*>(&TfKernelInfo::Instance));
    path = file_path + kTfOpsFileRelativePath;
    if (IsPathExist(path)) {
      real_tf_ops_file_path = path;
    } else {
      real_tf_ops_file_path = file_path + kTfOpsFileRelativePathOld;
    }
  }
  SetJsonPath(real_tf_ops_file_path);
  AICPUE_LOGI("TfKernelInfo real_tf_ops_file_path is [%s].", real_tf_ops_file_path.c_str());
  aicpu::State ret = OpsJsonFile::Instance().ParseUnderPath(real_tf_ops_file_path, op_info_json_file_);
  if (ret.state != ge::SUCCESS) {
    std::map<std::string, std::string> err_map;
    err_map["filename"] = real_tf_ops_file_path;
    err_map["reason"] = ret.msg;
    REPORT_PREDEFINED_ERR_MSG(GetViewErrorCodeStr(ViewErrorCode::LOAD_AICPU_KERNEL_INFO_ERR).c_str()),
                              std::vector<const char *>({"filename", "reason"}),
                              std::vector<const char *>({real_tf_ops_file_path.c_str(), ret.msg.c_str()});
    AICPUE_LOGE("load tf kernel info file[%s] failed, %s",
        real_tf_ops_file_path.c_str(), ret.msg.c_str());
  }
  return ret.state == ge::SUCCESS;
}

/**
 * For ops in AIcore, Call CompileOp before Generate task in AICPU
 * @param node Node information
 * @return status whether operation successful
 */
ge::Status TfKernelInfo::CompileOp(ge::NodePtr &node) {
  AICPU_CHECK_NOTNULL_ERRCODE(node, INPUT_PARAM_NULL)

  map<string, OpFullInfo> all_op_info;
  AICPU_CHECK_RES_WITH_LOG(GetOpInfos(all_op_info), "Get op infos failed, op[%s].",
                           node->GetName().c_str())
  AICPU_CHECK_RES_WITH_LOG(UpdataOpInfo(*node, all_op_info),
                           "Updata function attr failed, op[%s].",
                           node->GetName().c_str())

  // create nodedef
  ge::Status status = CreateNodeDef(*node);
  if (status != ge::SUCCESS) {
    AICPU_REPORT_INNER_ERR_MSG("Call CreateNodeDef failed, op[%s]", node->GetName().c_str());
    return status;
  }

  // calc Op run para
  status = CalcTfOpRunningParam(*node);
  if (status != ge::SUCCESS) {
    AICPU_REPORT_INNER_ERR_MSG("Call CalcOpRunningParam failed, op[%s]", node->GetName().c_str());
    return status;
  }

  // set shape type
  status = CheckAndSetUnknowType(node);
  if (status != ge::SUCCESS) {
    AICPU_REPORT_INNER_ERR_MSG("Call CheckAndSetUnknowType failed, op[%s]", node->GetName().c_str());
    return status;
  }
  return ge::SUCCESS;
}

void TfKernelInfo::LoadSupportedOps() {
  std::string supported_ops_json;
  const char* path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, path_env);
  std::string path;
  std::string env_path;
  if (path_env != nullptr) {
    env_path = std::string(path_env);
    path = env_path + kOpsInfoJson;
    if (IsPathExist(path)) {
      supported_ops_json = path;
    } else {
      supported_ops_json = env_path + kOpsInfoJsonOld;
    }
  } else {
    std::string file_path = GetOpsPath(reinterpret_cast<void*>(&TfKernelInfo::Instance));
    path = file_path + kOpsJsonRelativePath;
    if (IsPathExist(path)) {
      supported_ops_json = path;
    } else {
      supported_ops_json = file_path + kOpsJsonRelativePathOld;
    }
  }

  std::ifstream fs(supported_ops_json, std::ifstream::in);
  if (!fs.is_open()) {
    AICPUE_LOGE("Failed open config file %s", supported_ops_json.c_str());
    return;
  }

  nlohmann::json root;
  try {
    fs >> root;
  } catch (nlohmann::json::exception &e) {
    fs.close();
    AICPUE_LOGE("tfdebug parse json from %s failed! Error message: %s", supported_ops_json.c_str(), e.what());
    return;
  }
  for (auto iter = root.begin(); iter != root.end(); iter++) {
    supported_ops_.insert(iter.key());
  }
  fs.close();
  AICPUE_LOGI("tfdebug parse json from %s, op num is %lu", supported_ops_json.c_str(), supported_ops_.size());
  return;
}

bool TfKernelInfo::IsSupportedOps(const std::string &op) const {
  return supported_ops_.count(op) > 0;
}

FACTORY_KERNELINFO_CLASS_KEY(TfKernelInfo, kTfKernelInfoChoice)
} // namespace aicpu
