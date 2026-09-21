/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_PNE_DATA_FLOW_GRAPH_FUNCTION_COMPILE_H
#define AIR_COMPILER_PNE_DATA_FLOW_GRAPH_FUNCTION_COMPILE_H

#include "framework/common/debug/log.h"
#include "dflow/compiler/data_flow_graph/compile_config_json.h"
#include "proto/dflow.pb.h"

namespace ge {
class FunctionCompile {
 public:
  struct CompileResult {
    std::map<std::string, std::string> compile_bin_info; // key:runnable_res_type, value: compile_release_path;
    std::vector<CompileConfigJson::RunningResourceInfo> running_resources_info;
  };

  FunctionCompile(const std::string &pp_name, const CompileConfigJson::FunctionPpConfig &func_pp_cfg);
  ~FunctionCompile() = default;
  Status CompileAllResourceType();
  const CompileResult &GetCompileResult() const {
    return compile_result_;
  }
  static Status GetBuiltInFuncCompileResult(CompileResult &compile_result);

  static std::string GetAscendLatestInstallPath();
 private:
  static std::string GetAscendToolchainPath();
  Status CompileSingleResourceType(const std::string &toolchain, const std::string &resource_type,
                                   const std::string &target_bin);
  Status GetToolchainByResourceType(const std::string &resource_type, std::string &toolchain);
  void SetRunningResourceInfoToCompileResult();
  Status GetPathToSource(std::string &src_path) const;
  std::string function_pp_name_;
  CompileConfigJson::FunctionPpConfig func_pp_cfg_;
  CompileResult compile_result_;
  std::string src_path_;
};
}  // namespace ge
#endif  // AIR_COMPILER_PNE_DATA_FLOW_GRAPH_FUNCTION_COMPILE_H
