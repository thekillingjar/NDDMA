/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow/compiler/data_flow_graph/function_compile.h"
#include <regex>
#include <sys/file.h>
#include <mutex>
#include "framework/common/scope_guard.h"
#include "dflow/compiler/data_flow_graph/compile_config_json.h"
#include "mmpa/mmpa_api.h"
#include "graph_metadef/common/ge_common/util.h"
#include "dflow/base/utils/data_flow_utils.h"

namespace ge {
FunctionCompile::FunctionCompile(const std::string &pp_name, const CompileConfigJson::FunctionPpConfig &func_pp_cfg) :
                                 function_pp_name_(pp_name), func_pp_cfg_{func_pp_cfg}, compile_result_{} {}

Status FunctionCompile::GetPathToSource(std::string &src_path) const {
  std::regex pattern(R"([A-Za-z0-9./+\-_]+)");
  std::smatch match_result;
  std::string temp_path;
  if (!func_pp_cfg_.cmakelist.empty()) {
    GE_CHK_BOOL_RET_STATUS(std::regex_match(func_pp_cfg_.cmakelist, match_result, pattern), PARAM_INVALID,
                           "Invalid cmakelist: %s", func_pp_cfg_.cmakelist.c_str());
    temp_path = func_pp_cfg_.workspace + "/" + func_pp_cfg_.cmakelist;
  } else {
    temp_path = func_pp_cfg_.workspace + "/CMakeLists.txt";
  }

  src_path = RealPath(temp_path.c_str());
  GE_CHK_BOOL_RET_STATUS(!src_path.empty(), FAILED, "Invalid path:%s.", temp_path.c_str());
  const auto index = src_path.rfind("/");
  if (index != std::string::npos) {
    src_path = src_path.substr(0, index);
  }
  return SUCCESS;
}

Status FunctionCompile::CompileSingleResourceType(const std::string &toolchain, const std::string &resource_type,
                                                  const std::string &target_bin) {
  std::regex pattern(R"([A-Za-z0-9./+\-_]+)");
  std::smatch match_result;
  GE_CHK_BOOL_RET_STATUS(std::regex_match(toolchain, match_result, pattern), PARAM_INVALID, "Invalid toolchain: %s",
                         toolchain.c_str());
  GE_CHK_BOOL_RET_STATUS(std::regex_match(resource_type, match_result, pattern), PARAM_INVALID,
                         "Invalid resource_type: %s", resource_type.c_str());

  const std::string &workspace = func_pp_cfg_.workspace;
  std::string target_dir = workspace + "/build/_" + target_bin;
  std::string release_dir = resource_type + "/release";
  std::string inner_build_dir = resource_type + +"/build";

  std::string cmd = "cd " + target_dir + "; mkdir -p " + release_dir + "; mkdir -p " + inner_build_dir;
  GE_CHK_BOOL_RET_STATUS(system(cmd.c_str()) == 0, FAILED, "Failed to execute cmd[%s] for pp[%s].", cmd.c_str(),
                         function_pp_name_.c_str());
  GELOGI("Execute cmd[%s]", cmd.c_str());

  const std::string compile_lock = target_dir + "/" + inner_build_dir + "/compile_lock";
  int32_t fd = -1;
  {
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    fd = mmOpen2(compile_lock.c_str(), M_CREAT | M_WRONLY, M_IRUSR | M_IWUSR);
    if (fd == -1) {
      const int32_t error_code = mmGetErrorCode();
      GELOGE(FAILED, "Failed to open file[%s], error msg[%s].", compile_lock.c_str(),
             GetErrorNumStr(error_code).c_str());
      return FAILED;
    }
  }

  bool is_locked = false;
  ScopeGuard auto_close([&fd, &is_locked] {
    if (is_locked) {
      (void)flock(fd, LOCK_UN);
    }
    (void)close(fd);
    fd = -1;
  });
  // lock multiprocessing compile
  if (flock(fd, LOCK_EX) != 0) {
    const int32_t error_code = mmGetErrorCode();
    GELOGE(FAILED, "Failed to lock file[%s], error msg[%s].", compile_lock.c_str(), GetErrorNumStr(error_code).c_str());
    return FAILED;
  }
  is_locked = true;

  const std::string build_dir = target_dir + "/" + inner_build_dir;
  // cmake maybe hang when sh link to dash, so must specify bash to run
  cmd = "bash -c \"cd " + build_dir + "; cmake -S " + src_path_ + " -DTOOLCHAIN=" + toolchain +
        " -DUDF_TARGET_LIB=" + target_bin + " -DRELEASE_DIR=../release -DRESOURCE_TYPE=" + resource_type +
        " >output.log 2>&1\"";
  auto sys_ret = system(cmd.c_str());
  if (sys_ret != 0) {
    GELOGW("Failed to execute cmd[%s] for pp[%s], ret=%d, see also %s/output.log.", cmd.c_str(),
           function_pp_name_.c_str(), sys_ret, build_dir.c_str());
    return FAILED;
  }
  GELOGI("Execute cmd[%s].", cmd.c_str());

  cmd = "cd " + build_dir + "; make 2>> output.log";
  sys_ret = system(cmd.c_str());
  GE_CHK_BOOL_RET_STATUS(sys_ret == 0, FAILED,
                         "Failed to execute cmd[%s] for pp[%s], ret=%d, see also %s/output.log.", cmd.c_str(),
                         function_pp_name_.c_str(), sys_ret, build_dir.c_str());
  GELOGI("Execute cmd[%s]", cmd.c_str());
  return SUCCESS;
}

void FunctionCompile::SetRunningResourceInfoToCompileResult() {
  if (func_pp_cfg_.running_resources_info.empty()) {
    // set default running resource info
    constexpr uint32_t kDefaultMemorySize = 100U;
    CompileConfigJson::RunningResourceInfo resource_info;
    resource_info.resource_type = "cpu";
    resource_info.resource_num = 1;
    compile_result_.running_resources_info.emplace_back(resource_info);
    resource_info.resource_type = "memory";
    resource_info.resource_num = kDefaultMemorySize;
    compile_result_.running_resources_info.emplace_back(resource_info);
  } else {
    for (const auto &resource_info : func_pp_cfg_.running_resources_info) {
      compile_result_.running_resources_info.emplace_back(resource_info);
    }
  }
  return;
}

std::string FunctionCompile::GetAscendLatestInstallPath() {
  return DataFlowUtils::GetAscendHomePath();
}

std::string FunctionCompile::GetAscendToolchainPath() {
  static std::once_flag flag;
  static std::string toolchain_path;
  std::call_once(flag, []() {
    toolchain_path = GetAscendLatestInstallPath() + "/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++";
  });
  return toolchain_path;
}

Status FunctionCompile::GetToolchainByResourceType(const std::string &resource_type, std::string &toolchain) {
  if (func_pp_cfg_.toolchain_map[resource_type].empty()) {
    if (resource_type == kResourceTypeAscend) {
      toolchain = GetAscendToolchainPath();
      INT32 mm_ret = mmAccess(toolchain.c_str());
      if (mm_ret != EN_OK) {
        GELOGW("Default ascend toolchain file[%s] does not exist.", toolchain.c_str());
        return FAILED;
      }
    } else {
      toolchain = "g++";
    }
  } else {
    toolchain = func_pp_cfg_.toolchain_map[resource_type];
  }
  return SUCCESS;
}

Status FunctionCompile::CompileAllResourceType() {
  std::set<std::string> resource_types;
  GE_CHK_STATUS_RET(CompileConfigJson::GetResourceTypeFromNumaConfig(resource_types),
                    "Failed to get resource types from numa config.");

  const std::string &workspace = func_pp_cfg_.workspace;
  std::regex workspace_pattern(R"([A-Za-z0-9./+\-_]+)");
  std::regex lib_pattern(R"(^lib[A-Za-z0-9.\-_]+\.so$)");
  std::smatch match_result;
  GE_CHK_BOOL_RET_STATUS(std::regex_match(workspace, match_result, workspace_pattern), PARAM_INVALID,
                         "Invalid workspace: %s", workspace.c_str());
  std::string target_bin = func_pp_cfg_.target_bin;
  GE_CHK_BOOL_RET_STATUS(std::regex_match(target_bin, match_result, lib_pattern), PARAM_INVALID,
                         "Invalid target_bin: %s", target_bin.c_str());
  auto index = target_bin.rfind(".");
  if (index != std::string::npos) {
    target_bin = target_bin.substr(0, index);
    target_bin = target_bin.substr(strlen("lib"));
  }

  std::string cmd = "cd " + workspace + "; mkdir -p build/_" + target_bin;
  GELOGD("Execute cmd[%s]", cmd.c_str());
  auto ret = system(cmd.c_str());
  if (ret != 0) {
    GELOGE(FAILED, "Failed to execute cmd[%s] for pp[%s]", cmd.c_str(), function_pp_name_.c_str());
    return FAILED;
  }
  const auto ge_ret = GetPathToSource(src_path_);
  if (ge_ret != SUCCESS) {
    GELOGE(ge_ret, "Failed to get path to source for pp[%s].", function_pp_name_.c_str());
    return ge_ret;
  }
  for (const auto &resource_type : resource_types) {
    std::string toolchain;
    if (GetToolchainByResourceType(resource_type, toolchain) != SUCCESS) {
      GELOGI("Toolchain file[%s] for resource type[%s] does not exist.", toolchain.c_str(), resource_type.c_str());
      continue;
    }
    auto compile_ret = CompileSingleResourceType(toolchain, resource_type, target_bin);
    if (compile_ret != SUCCESS) {
      GELOGW("compile function pp[%s]'s so failed, resource type[%s], compile_ret=%d.", function_pp_name_.c_str(),
             resource_type.c_str(), compile_ret);
    } else {
      compile_result_.compile_bin_info[resource_type] =
          func_pp_cfg_.workspace + "/build/_" + target_bin + "/" + resource_type + "/release/";
    }
  }

  if (compile_result_.compile_bin_info.empty()) {
    GELOGE(FAILED, "Compile all resource type[%s] for function pp[%s] failed.",
           ToString(std::vector<std::string>(resource_types.cbegin(), resource_types.cend())).c_str(),
           function_pp_name_.c_str());
    return FAILED;
  }

  // running_resource_info
  SetRunningResourceInfoToCompileResult();
  return SUCCESS;
}

Status FunctionCompile::GetBuiltInFuncCompileResult(CompileResult &compile_result) {
  static std::mutex mt;
  static CompileResult built_in_func_compile_result;
  std::lock_guard<std::mutex> guard(mt);
  if (!built_in_func_compile_result.compile_bin_info.empty()) {
    compile_result = built_in_func_compile_result;
    return SUCCESS;
  }
  // set default running resource info
  constexpr uint32_t kDefaultMemorySize = 100U;
  CompileConfigJson::RunningResourceInfo resource_info;
  resource_info.resource_type = "cpu";
  resource_info.resource_num = 1;
  built_in_func_compile_result.running_resources_info.emplace_back(resource_info);
  resource_info.resource_type = "memory";
  resource_info.resource_num = kDefaultMemorySize;
  built_in_func_compile_result.running_resources_info.emplace_back(resource_info);
  // set default runnable resources info
  std::set<std::string> resource_types;
  GE_CHK_STATUS_RET(CompileConfigJson::GetResourceTypeFromNumaConfig(resource_types),
                    "Failed to get resource types from numa config.");
  for (const auto &resource_type : resource_types) {
    // the internal udf supports all resource types, and not need to set bin
    built_in_func_compile_result.compile_bin_info[resource_type] = "";
  }
  compile_result = built_in_func_compile_result;
  return SUCCESS;
}
} // namespace ge
