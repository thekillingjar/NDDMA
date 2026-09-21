/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow/compiler/data_flow_graph/compile_config_json.h"
#include <fstream>
#include "graph/utils/type_utils.h"
#include "mmpa/mmpa_api.h"
#include "graph/ge_global_options.h"
#include "graph/ge_context.h"
#include "framework/common/ge_types.h"

namespace {
// FunctionPpConfig json key
constexpr char const *kFunctionPpConfigOptionWorkspace = "workspace";
constexpr char const *kFunctionPpConfigOptionTargetBin = "target_bin";
constexpr char const *kFunctionPpConfigOptionInputNum = "input_num";
constexpr char const *kFunctionPpConfigOptionOutputNum = "output_num";
constexpr char const *kFunctionPpConfigOptionFuncList = "func_list";
constexpr char const *kFunctionPpConfigOptionFuncListFuncName = "func_name";
constexpr char const *kFunctionPpConfigOptionFuncListInputsIndex = "inputs_index";
constexpr char const *kFunctionPpConfigOptionFuncListOutputsIndex = "outputs_index";
constexpr char const *kFunctionPpConfigOptionFuncListStreamInput = "stream_input";
constexpr char const *kFunctionPpConfigOptionMakefile = "cmakelist_path";
constexpr char const *kFunctionPpConfigOptionCompiler = "compiler";
constexpr char const *kFunctionPpConfigOptionRunningResourcesInfo = "running_resources_info";
constexpr char const *kFunctionPpConfigOptionRunningResourcesInfoResType = "type";
constexpr char const *kFunctionPpConfigOptionRunningResourcesInfoResNum = "num";
constexpr char const *kFunctionPpConfigOptionCompilerResourceType = "resource_type";
constexpr char const *kFunctionPpConfigOptionCompilerToolchain = "toolchain";
constexpr char const *kFunctionPpConfigOptionHeavyLoad = "heavy_load";
constexpr char const *kFunctionPpConfigOptionVisibleDeviceEnable = "visible_device_enable";
constexpr char const *kFunctionPpConfigOptionBuiltInFlowFunc = "built_in_flow_func";
constexpr char const *kFunctionPpConfigOptionBufCfg = "buf_cfg";
constexpr char const *kFunctionPpConfigOptionBufCfgTotalSize = "total_size";
constexpr char const *kFunctionPpConfigOptionBufCfgBlkSize = "blk_size";
constexpr char const *kFunctionPpConfigOptionBufCfgMaxBufSize = "max_buf_size";
constexpr char const *kFunctionPpConfigOptionBufCfgPageType = "page_type";

// GraphPpConfig json key
constexpr char const *kGraphPpConfigOptionBuildOptions = "build_options";
constexpr char const *kModelPpFusionInputs = "invoke_model_fusion_inputs";
constexpr char const *kGraphPpConfigInputsTensorDesc = "inputs_tensor_desc";
constexpr char const *kGraphPpConfigInputsTensorDescDatatype = "data_type";
constexpr char const *kGraphPpConfigInputsTensorDescShape = "shape";
constexpr char const *kGraphPpConfigInputsTensorDescFormat = "format";
}  // namespace
namespace ge {
Status CompileConfigJson::ReadFunctionPpConfigFromJsonFile(const std::string &file_path,
                                                           FunctionPpConfig &function_pp_config) {
  nlohmann::json json_buff;
  GE_CHK_STATUS_RET_NOLOG(ReadCompileConfigJsonFile(file_path, json_buff));
  GE_CHK_STATUS_RET_NOLOG(CheckFunctionPpConfigJson(json_buff));
  try {
    function_pp_config = json_buff;
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to read function pp config from json[%s].", json_buff.dump().c_str());
    return FAILED;
  }
  if (!function_pp_config.built_in_flow_func) {
    GE_CHK_BOOL_RET_STATUS(!function_pp_config.workspace.empty(), FAILED, "Failed to get config option[%s].",
                           kFunctionPpConfigOptionWorkspace);
    GE_CHK_BOOL_RET_STATUS(!function_pp_config.target_bin.empty(), FAILED, "Failed to get config option[%s].",
                           kFunctionPpConfigOptionTargetBin);
  }
  return SUCCESS;
}

Status CompileConfigJson::ReadModelPpConfigFromJsonFile(const std::string &file_path, ModelPpConfig &model_pp_config) {
  if (file_path.empty()) {
    GELOGI("Not set graph pp config json.");
    return SUCCESS;
  }
  nlohmann::json json_buff;
  GE_CHK_STATUS_RET_NOLOG(ReadCompileConfigJsonFile(file_path, json_buff));
  try {
    model_pp_config = json_buff;
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to read graph pp config from json[%s].", json_buff.dump().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status CompileConfigJson::ReadGraphPpConfigFromJsonFile(const std::string &file_path, GraphPpConfig &graph_pp_config) {
  if (file_path.empty()) {
    GELOGI("Not set graph pp config json.");
    return SUCCESS;
  }
  nlohmann::json json_buff;
  GE_CHK_STATUS_RET_NOLOG(ReadCompileConfigJsonFile(file_path, json_buff));
  try {
    graph_pp_config = json_buff;
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to read graph pp config from json[%s].", json_buff.dump().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status CompileConfigJson::ReadToolchainFromJsonFile(const std::string &file_path,
                                                    std::map<std::string, std::string> &toolchain_map) {
  nlohmann::json json_buff;
  GE_CHK_STATUS_RET(ReadCompileConfigJsonFile(file_path, json_buff), "Read toolchain config[%s] from json failed",
                    file_path.c_str());
  GE_CHK_STATUS_RET(CheckFunctionPpCompiler(json_buff), "Check toolchain config[%s] failed.", file_path.c_str());
  try {
    std::map<std::string, std::vector<std::map<std::string, std::string>>> temp_map = json_buff;
    for (auto &compiler_info : temp_map[kFunctionPpConfigOptionCompiler]) {
      toolchain_map[compiler_info[kFunctionPpConfigOptionCompilerResourceType]] =
        compiler_info[kFunctionPpConfigOptionCompilerToolchain];
    }
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to read toolchain config from json[%s], err msg:%s", json_buff.dump().c_str(), e.what());
    return FAILED;
  }
  return SUCCESS;
}

Status CompileConfigJson::CheckFunctionPpConfigJsonOptional(const nlohmann::json &json_buff) {
  if (json_buff.contains(kFunctionPpConfigOptionWorkspace)) {
    GE_CHK_BOOL_RET_STATUS(json_buff[kFunctionPpConfigOptionWorkspace].is_string(), FAILED,
                           "The config option[%s] value type should be string.", kFunctionPpConfigOptionWorkspace);
  }
  if (json_buff.contains(kFunctionPpConfigOptionTargetBin)) {
    GE_CHK_BOOL_RET_STATUS(json_buff[kFunctionPpConfigOptionTargetBin].is_string(), FAILED,
                           "The config option[%s] value type should be string.", kFunctionPpConfigOptionTargetBin);
  }
  if (json_buff.contains(kFunctionPpConfigOptionMakefile)) {
    GE_CHK_BOOL_RET_STATUS(json_buff[kFunctionPpConfigOptionMakefile].is_string(), FAILED,
                           "The config option[%s] value type should be string.", kFunctionPpConfigOptionMakefile);
  }
  if (json_buff.contains(kFunctionPpConfigOptionHeavyLoad)) {
    GE_CHK_BOOL_RET_STATUS(json_buff[kFunctionPpConfigOptionHeavyLoad].is_boolean(), FAILED,
                           "The config option[%s] value type should be boolean.", kFunctionPpConfigOptionHeavyLoad);
  }
  if (json_buff.contains(kFunctionPpConfigOptionVisibleDeviceEnable)) {
    GE_CHK_BOOL_RET_STATUS(json_buff[kFunctionPpConfigOptionVisibleDeviceEnable].is_boolean(), FAILED,
                           "The config option[%s] value type should be boolean.",
                           kFunctionPpConfigOptionVisibleDeviceEnable);
  }
  if (json_buff.contains(kFunctionPpConfigOptionBuiltInFlowFunc)) {
    GE_CHK_BOOL_RET_STATUS(json_buff[kFunctionPpConfigOptionBuiltInFlowFunc].is_boolean(), FAILED,
                           "The config option[%s] value type should be boolean.",
                           kFunctionPpConfigOptionBuiltInFlowFunc);
  }
  if (json_buff.contains(kFunctionPpConfigOptionCompiler)) {
    GE_CHK_BOOL_RET_STATUS(json_buff[kFunctionPpConfigOptionCompiler].is_string(), FAILED,
                           "The config option[%s] value type should be string.", kFunctionPpConfigOptionCompiler);
  }
  if (json_buff.contains(kFunctionPpConfigOptionRunningResourcesInfo)) {
    GE_CHK_BOOL_RET_STATUS(json_buff[kFunctionPpConfigOptionRunningResourcesInfo].is_array(), FAILED,
        "The config option[%s] value type should be list.", kFunctionPpConfigOptionRunningResourcesInfo);
    GE_CHK_BOOL_RET_STATUS(json_buff[kFunctionPpConfigOptionRunningResourcesInfo].size() > 0, FAILED,
        "The config option[%s] value should be at least one info.", kFunctionPpConfigOptionRunningResourcesInfo);
    GE_CHK_STATUS_RET(CheckFunctionPpRunningResourcesInfo(json_buff[kFunctionPpConfigOptionRunningResourcesInfo]),
                      "Check function pp running resource info failed.");
  }
  if (json_buff.contains(kFunctionPpConfigOptionBufCfg)) {
    GE_CHK_BOOL_RET_STATUS(json_buff[kFunctionPpConfigOptionBufCfg].is_array(), FAILED,
        "The config option[%s] value type should be list.", kFunctionPpConfigOptionBufCfg);
    GE_CHK_BOOL_RET_STATUS(json_buff[kFunctionPpConfigOptionBufCfg].size() > 0, FAILED,
        "The config option[%s] value should be at least one info.", kFunctionPpConfigOptionBufCfg);
    GE_CHK_STATUS_RET(CheckFunctionPpBufCfg(json_buff[kFunctionPpConfigOptionBufCfg]),
        "Check function pp buf cfg failed.");
  }
  return SUCCESS;
}

Status CompileConfigJson::CheckFunctionPpConfigJson(const nlohmann::json &json_buff) {
  GE_CHK_BOOL_RET_STATUS(json_buff.contains(kFunctionPpConfigOptionInputNum), FAILED,
                         "Failed to get config option[%s].", kFunctionPpConfigOptionInputNum);
  GE_CHK_BOOL_RET_STATUS(json_buff[kFunctionPpConfigOptionInputNum].is_number_unsigned(), FAILED,
                         "The config option[%s] value type should be unsigned num.", kFunctionPpConfigOptionInputNum);
  GE_CHK_BOOL_RET_STATUS(json_buff.contains(kFunctionPpConfigOptionOutputNum), FAILED,
                         "Failed to get config option[%s].", kFunctionPpConfigOptionOutputNum);
  GE_CHK_BOOL_RET_STATUS(json_buff[kFunctionPpConfigOptionOutputNum].is_number_unsigned(), FAILED,
                         "The config option[%s] value type should be unsigned num.", kFunctionPpConfigOptionOutputNum);
  GE_CHK_BOOL_RET_STATUS(json_buff.contains(kFunctionPpConfigOptionFuncList), FAILED,
                         "Failed to get config option[%s].", kFunctionPpConfigOptionFuncList);
  GE_CHK_BOOL_RET_STATUS(json_buff[kFunctionPpConfigOptionFuncList].is_array(), FAILED,
                         "The config option[%s] value type should be list.", kFunctionPpConfigOptionFuncList);
  GE_CHK_BOOL_RET_STATUS((json_buff[kFunctionPpConfigOptionFuncList].size() > 0), FAILED,
                         "The config option[%s] value should be at least one func.", kFunctionPpConfigOptionFuncList);
  GE_CHK_STATUS_RET_NOLOG(CheckFunctionPpConfigFuncListJson(json_buff[kFunctionPpConfigOptionFuncList]));
  return CheckFunctionPpConfigJsonOptional(json_buff);
}

Status CompileConfigJson::CheckFunctionPpRunningResourcesInfo(const nlohmann::json &json_buff) {
  for (size_t i = 0; i < json_buff.size(); ++i) {
    GE_CHK_BOOL_RET_STATUS(json_buff[i].contains(kFunctionPpConfigOptionRunningResourcesInfoResType), FAILED,
                           "Failed to get config option[%s][%zu][%s].", kFunctionPpConfigOptionRunningResourcesInfo, i,
                           kFunctionPpConfigOptionRunningResourcesInfoResType);
    GE_CHK_BOOL_RET_STATUS(json_buff[i][kFunctionPpConfigOptionRunningResourcesInfoResType].is_string(), FAILED,
                           "The config option[%s][%zu][%s] value type should be string.",
                           kFunctionPpConfigOptionRunningResourcesInfo, i,
                           kFunctionPpConfigOptionRunningResourcesInfoResType);
    GE_CHK_BOOL_RET_STATUS(json_buff[i].contains(kFunctionPpConfigOptionRunningResourcesInfoResNum), FAILED,
                           "Failed to get config option[%s][%zu][%s].", kFunctionPpConfigOptionRunningResourcesInfo, i,
                           kFunctionPpConfigOptionRunningResourcesInfoResNum);
    GE_CHK_BOOL_RET_STATUS(json_buff[i][kFunctionPpConfigOptionRunningResourcesInfoResNum].is_number_unsigned(), FAILED,
                           "The config option[%s][%zu][%s] value type should be unsigned num.",
                           kFunctionPpConfigOptionRunningResourcesInfo, i,
                           kFunctionPpConfigOptionRunningResourcesInfoResNum);
  }
  return SUCCESS;
}

Status CompileConfigJson::CheckFunctionPpBufCfg(const nlohmann::json &json_buff) {
  for (size_t i = 0; i < json_buff.size(); ++i) {
    GE_CHK_BOOL_RET_STATUS(json_buff[i].contains(kFunctionPpConfigOptionBufCfgTotalSize), FAILED, "Failed to get "
        "config option[%s][%zu][%s].", kFunctionPpConfigOptionBufCfg, i, kFunctionPpConfigOptionBufCfgTotalSize);
    GE_CHK_BOOL_RET_STATUS(json_buff[i][kFunctionPpConfigOptionBufCfgTotalSize].is_number_unsigned(), FAILED,
        "The config option[%s][%zu][%s] value type should be unsigned int.",
        kFunctionPpConfigOptionBufCfg, i, kFunctionPpConfigOptionBufCfgTotalSize);
    GE_CHK_BOOL_RET_STATUS(json_buff[i].contains(kFunctionPpConfigOptionBufCfgBlkSize), FAILED, "Failed to get "
        "config option[%s][%zu][%s].", kFunctionPpConfigOptionBufCfg, i, kFunctionPpConfigOptionBufCfgBlkSize);
    GE_CHK_BOOL_RET_STATUS(json_buff[i][kFunctionPpConfigOptionBufCfgBlkSize].is_number_unsigned(), FAILED,
        "The config option[%s][%zu][%s] value type should be unsigned int.",
        kFunctionPpConfigOptionBufCfg, i, kFunctionPpConfigOptionBufCfgBlkSize);
    GE_CHK_BOOL_RET_STATUS(json_buff[i].contains(kFunctionPpConfigOptionBufCfgMaxBufSize), FAILED, "Failed to get "
        "config option[%s][%zu][%s].", kFunctionPpConfigOptionBufCfg, i, kFunctionPpConfigOptionBufCfgMaxBufSize);
    GE_CHK_BOOL_RET_STATUS(json_buff[i][kFunctionPpConfigOptionBufCfgMaxBufSize].is_number_unsigned(), FAILED,
        "The config option[%s][%zu][%s] value type should be unsigned int.",
        kFunctionPpConfigOptionBufCfg, i, kFunctionPpConfigOptionBufCfgMaxBufSize);
    GE_CHK_BOOL_RET_STATUS(json_buff[i].contains(kFunctionPpConfigOptionBufCfgPageType), FAILED, "Failed to get "
        "config option[%s][%zu][%s].", kFunctionPpConfigOptionBufCfg, i, kFunctionPpConfigOptionBufCfgPageType);
    GE_CHK_BOOL_RET_STATUS(json_buff[i][kFunctionPpConfigOptionBufCfgPageType].is_string(), FAILED,
        "The config option[%s][%zu][%s] value type should be string.",
        kFunctionPpConfigOptionBufCfg, i, kFunctionPpConfigOptionBufCfgPageType);
  }
  return SUCCESS;
}

Status CompileConfigJson::CheckFunctionPpCompiler(const nlohmann::json &json_buff) {
  GE_CHK_BOOL_RET_STATUS(json_buff.contains(kFunctionPpConfigOptionCompiler), FAILED,
                         "Failed to get config option[%s].", kFunctionPpConfigOptionCompiler);
  nlohmann::json json_buff_list = json_buff[kFunctionPpConfigOptionCompiler];
  for (size_t i = 0; i < json_buff_list.size(); ++i) {
    GE_CHK_BOOL_RET_STATUS(json_buff_list[i].contains(kFunctionPpConfigOptionCompilerResourceType), FAILED,
                           "Failed to get config option[%s][%zu][%s].", kFunctionPpConfigOptionCompiler, i,
                           kFunctionPpConfigOptionCompilerResourceType);
    GE_CHK_BOOL_RET_STATUS(json_buff_list[i][kFunctionPpConfigOptionCompilerResourceType].is_string(), FAILED,
                           "The config option[%s][%zu][%s] value type should be string.",
                           kFunctionPpConfigOptionCompiler, i, kFunctionPpConfigOptionCompilerResourceType);
    GE_CHK_BOOL_RET_STATUS(json_buff_list[i].contains(kFunctionPpConfigOptionCompilerToolchain), FAILED,
                           "Failed to get config option[%s][%zu][%s].", kFunctionPpConfigOptionCompiler, i,
                           kFunctionPpConfigOptionCompilerToolchain);
    GE_CHK_BOOL_RET_STATUS(json_buff_list[i][kFunctionPpConfigOptionCompilerToolchain].is_string(), FAILED,
                           "The config option[%s][%zu][%s] value type should be string.",
                           kFunctionPpConfigOptionCompiler, i, kFunctionPpConfigOptionCompilerToolchain);
  }
  return SUCCESS;
}

Status CompileConfigJson::CheckFunctionPpConfigFuncListJson(const nlohmann::json &json_buff) {
  for (size_t i = 0; i < json_buff.size(); ++i) {
    GE_CHK_BOOL_RET_STATUS(json_buff[i].contains(kFunctionPpConfigOptionFuncListFuncName), FAILED,
                           "Failed to get config option[%s][%zu][%s].", kFunctionPpConfigOptionFuncList, i,
                           kFunctionPpConfigOptionFuncListFuncName);
    GE_CHK_BOOL_RET_STATUS(json_buff[i][kFunctionPpConfigOptionFuncListFuncName].is_string(), FAILED,
                           "The config option[%s][%zu][%s] value type should be string.",
                           kFunctionPpConfigOptionFuncList, i, kFunctionPpConfigOptionFuncListFuncName);
    if (json_buff[i].contains(kFunctionPpConfigOptionFuncListInputsIndex)) {
      GE_CHK_BOOL_RET_STATUS(json_buff[i][kFunctionPpConfigOptionFuncListInputsIndex].is_array(), FAILED,
                             "The config option[%s][%zu][%s] value type should be vector<uint32_t>.",
                             kFunctionPpConfigOptionFuncList, i, kFunctionPpConfigOptionFuncListInputsIndex);
      GE_CHK_BOOL_RET_STATUS((json_buff[i][kFunctionPpConfigOptionFuncListInputsIndex].size() > 0), FAILED,
                             "The config option[%s][%zu][%s] value should be at least one func.",
                             kFunctionPpConfigOptionFuncList, i, kFunctionPpConfigOptionFuncListInputsIndex);
    }
    if (json_buff[i].contains(kFunctionPpConfigOptionFuncListOutputsIndex)) {
      GE_CHK_BOOL_RET_STATUS(json_buff[i][kFunctionPpConfigOptionFuncListOutputsIndex].is_array(), FAILED,
                             "The config option[%s][%zu][%s] value type should be vector<uint32_t>.",
                             kFunctionPpConfigOptionFuncList, i, kFunctionPpConfigOptionFuncListOutputsIndex);
      GE_CHK_BOOL_RET_STATUS((json_buff[i][kFunctionPpConfigOptionFuncListOutputsIndex].size() > 0), FAILED,
                             "The config option[%s][%zu][%s] value should be at least one func.",
                             kFunctionPpConfigOptionFuncList, i, kFunctionPpConfigOptionFuncListOutputsIndex);
    }
    if (json_buff[i].contains(kFunctionPpConfigOptionFuncListStreamInput)) {
      GE_CHK_BOOL_RET_STATUS(json_buff[i][kFunctionPpConfigOptionFuncListStreamInput].is_boolean(), FAILED,
                             "The config option[%s][%zu][%s] value type should be boolean.",
                             kFunctionPpConfigOptionFuncList, i, kFunctionPpConfigOptionFuncListStreamInput);
    }
  }
  return SUCCESS;
}

Status CompileConfigJson::ReadCompileConfigJsonFile(const std::string &file_path, nlohmann::json &json_buff) {
  std::ifstream file_stream(file_path);
  GE_CHK_BOOL_RET_STATUS(file_stream.is_open(), FAILED, "Failed to open config file[%s].", file_path.c_str());
  try {
    file_stream >> json_buff;
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to read config file[%s], err msg: %s.", file_path.c_str(), e.what());
    return FAILED;
  }
  GELOGD("Read json file[%s] successfully, content is: %s.", file_path.c_str(), json_buff.dump().c_str());
  return SUCCESS;
}

Status CompileConfigJson::ReadDeployInfoFromJsonFile(const std::string &file_path,
                                                     DeployConfigInfo &deploy_conf) {
  std::ifstream file_stream(file_path);
  GE_CHK_BOOL_RET_STATUS(file_stream.is_open(), FAILED, "Failed to open deploy config file[%s].", file_path.c_str());
  nlohmann::json json_buff;
  try {
    file_stream >> json_buff;
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to read config file[%s], err msg: %s.", file_path.c_str(), e.what());
    return FAILED;
  }
  GELOGD("Read deploy config file[%s] to json, content is: %s.", file_path.c_str(), json_buff.dump().c_str());
  try {
    bool with_deploy_info = false;
    bool with_batch_deploy_info = false;
    if (json_buff.contains("deploy_info")) {
      deploy_conf.deploy_info_list = json_buff["deploy_info"].get<std::vector<FlowNodeDeployInfo>>();
      with_deploy_info = true;
    }
    if (json_buff.contains("batch_deploy_info")) {
      deploy_conf.batch_deploy_info_list = json_buff["batch_deploy_info"].get<std::vector<FlowNodeBatchDeployInfo>>();
      with_batch_deploy_info = true;
    }
    if (json_buff.contains("dynamic_schedule_enable")) {
      deploy_conf.dynamic_schedule_enable = json_buff["dynamic_schedule_enable"].get<bool>();
      GEEVENT("read json file[%s], dynamic_schedule_enable = %u.", file_path.c_str(),
              deploy_conf.dynamic_schedule_enable ? 1 : 0);
    }
    if (json_buff.contains("keep_logic_device_order")) {
      deploy_conf.keep_logic_device_order = json_buff["keep_logic_device_order"].get<bool>();
      GEEVENT("read json file[%s], keep_logic_device_order = %u.", file_path.c_str(),
              deploy_conf.keep_logic_device_order ? 1 : 0);
    }
    if (!with_deploy_info && !with_batch_deploy_info) {
      GELOGE(FAILED, "deploy config file[%s] must contian \"deploy_info\" or \"batch_deploy_info\" node.",
             file_path.c_str());
      return FAILED;
    }
    if (json_buff.contains("deploy_mem_info")) {
      deploy_conf.mem_size_cfg = json_buff["deploy_mem_info"].get<std::vector<FlowNodeBatchMemCfg>>();
    }
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to read json file[%s] to vector, err msg: %s.", file_path.c_str(), e.what());
    return FAILED;
  }
  GELOGD("read json file[%s] to vector success, size=%zu.", file_path.c_str(), deploy_conf.deploy_info_list.size());
  return SUCCESS;
}

template <typename T>
inline void Assign(T &variable, const std::string &key, const nlohmann::json &json_buff) {
  const auto iter = json_buff.find(key);
  if (iter != json_buff.cend()) {
    try {
      variable = iter.value().get<T>();
    } catch (const nlohmann::json::exception &e) {
      GELOGE(FAILED, "Failed to read config[%s] from json[%s], err msg: %s.", key.c_str(), iter.value().dump().c_str(),
             e.what());
      throw e;
    }
  }
}

void from_json(const nlohmann::json &json_buff, CompileConfigJson::FunctionPpConfig &func_pp_config) {
  Assign(func_pp_config.built_in_flow_func, kFunctionPpConfigOptionBuiltInFlowFunc, json_buff);
  if (!func_pp_config.built_in_flow_func) {
    Assign(func_pp_config.workspace, kFunctionPpConfigOptionWorkspace, json_buff);
    Assign(func_pp_config.target_bin, kFunctionPpConfigOptionTargetBin, json_buff);
    Assign(func_pp_config.cmakelist, kFunctionPpConfigOptionMakefile, json_buff);
  }
  Assign(func_pp_config.input_num, kFunctionPpConfigOptionInputNum, json_buff);
  Assign(func_pp_config.output_num, kFunctionPpConfigOptionOutputNum, json_buff);
  Assign(func_pp_config.funcs, kFunctionPpConfigOptionFuncList, json_buff);
  Assign(func_pp_config.running_resources_info, kFunctionPpConfigOptionRunningResourcesInfo, json_buff);
  Assign(func_pp_config.heavy_load, kFunctionPpConfigOptionHeavyLoad, json_buff);
  Assign(func_pp_config.visible_device_enable, kFunctionPpConfigOptionVisibleDeviceEnable, json_buff);
  Assign(func_pp_config.user_buf_cfg, kFunctionPpConfigOptionBufCfg, json_buff);

  std::string compiler;
  Assign(compiler, kFunctionPpConfigOptionCompiler, json_buff);
  if (!compiler.empty() &&
      CompileConfigJson::ReadToolchainFromJsonFile(compiler, func_pp_config.toolchain_map) != SUCCESS) {
    GELOGE(FAILED, "Failed to read compiler config json file[%s]", compiler.c_str());
  }
}

void from_json(const nlohmann::json &json_buff, CompileConfigJson::FunctionDesc &func_desc) {
  Assign(func_desc.func_name, kFunctionPpConfigOptionFuncListFuncName, json_buff);
  Assign(func_desc.inputs_index, kFunctionPpConfigOptionFuncListInputsIndex, json_buff);
  Assign(func_desc.outputs_index, kFunctionPpConfigOptionFuncListOutputsIndex, json_buff);
  Assign(func_desc.stream_input, kFunctionPpConfigOptionFuncListStreamInput, json_buff);
}

void from_json(const nlohmann::json &json_buff, CompileConfigJson::RunningResourceInfo &running_resource_info) {
  Assign(running_resource_info.resource_type, kFunctionPpConfigOptionRunningResourcesInfoResType, json_buff);
  Assign(running_resource_info.resource_num, kFunctionPpConfigOptionRunningResourcesInfoResNum, json_buff);
}

void from_json(const nlohmann::json &json_buff, CompileConfigJson::BufCfg &user_buf_config) {
  Assign(user_buf_config.total_size, kFunctionPpConfigOptionBufCfgTotalSize, json_buff);
  Assign(user_buf_config.blk_size, kFunctionPpConfigOptionBufCfgBlkSize, json_buff);
  Assign(user_buf_config.max_buf_size, kFunctionPpConfigOptionBufCfgMaxBufSize, json_buff);
  Assign(user_buf_config.page_type, kFunctionPpConfigOptionBufCfgPageType, json_buff);
}

void from_json(const nlohmann::json &json_buff, CompileConfigJson::GraphPpConfig &graph_pp_config) {
  Assign(graph_pp_config.build_options, kGraphPpConfigOptionBuildOptions, json_buff);
  Assign(graph_pp_config.inputs_tensor_desc_config, kGraphPpConfigInputsTensorDesc, json_buff);
}

void from_json(const nlohmann::json &json_buff, CompileConfigJson::ModelPpConfig &model_pp_config) {
  Assign(model_pp_config.invoke_model_fusion_inputs, kModelPpFusionInputs, json_buff);
}

void from_json(const nlohmann::json &json_buff, CompileConfigJson::TensorDescConfig &tensor_desc_config) {
  std::vector<int64_t> shape;
  tensor_desc_config.shape_config = {false, shape};
  if (json_buff.contains(kGraphPpConfigInputsTensorDescShape)) {
    Assign(shape, kGraphPpConfigInputsTensorDescShape, json_buff);
    tensor_desc_config.shape_config = {true, shape};
  }
  DataType dtype = DataType::DT_FLOAT;
  tensor_desc_config.dtype_config = {false, dtype};
  if (json_buff.contains(kGraphPpConfigInputsTensorDescDatatype)) {
    std::string data_type_str;
    Assign(data_type_str, kGraphPpConfigInputsTensorDescDatatype, json_buff);
    dtype = TypeUtils::SerialStringToDataType(data_type_str);
    tensor_desc_config.dtype_config = {true, dtype};
  }
  Format format = Format::FORMAT_ND;
  tensor_desc_config.format_config = {false, format};
  if (json_buff.contains(kGraphPpConfigInputsTensorDescFormat)) {
    std::string format_str;
    Assign(format_str, kGraphPpConfigInputsTensorDescFormat, json_buff);
    format = TypeUtils::SerialStringToFormat(format_str);
    tensor_desc_config.format_config = {true, format};
  }
}

void from_json(const nlohmann::json &json_buff, CompileConfigJson::FlowNodeDeployInfo &deploy_info) {
  json_buff.at("flow_node_name").get_to(deploy_info.flow_node_name);
  json_buff.at("logic_device_id").get_to(deploy_info.logic_device_id);
}

void from_json(const nlohmann::json &json_buff, CompileConfigJson::InvokeDeployInfo &invoke_deploy_info) {
  json_buff.at("invoke_name").get_to(invoke_deploy_info.invoke_name);
  if (json_buff.contains("deploy_info_file")) {
    json_buff.at("deploy_info_file").get_to(invoke_deploy_info.deploy_info_file);
  }
  if (json_buff.contains("logic_device_list")) {
    json_buff.at("logic_device_list").get_to(invoke_deploy_info.logic_device_list);
  }
  if (json_buff.contains("redundant_logic_device_list")) {
    json_buff.at("redundant_logic_device_list").get_to(invoke_deploy_info.redundant_logic_device_list);
  }
}

void from_json(const nlohmann::json &json_buff, CompileConfigJson::FlowNodeBatchDeployInfo &batch_deploy_info) {
  json_buff.at("flow_node_list").get_to(batch_deploy_info.flow_node_list);
  json_buff.at("logic_device_list").get_to(batch_deploy_info.logic_device_list);
  if (json_buff.contains("redundant_logic_device_list")) {
    json_buff.at("redundant_logic_device_list").get_to(batch_deploy_info.redundant_logic_device_list);
  }
  if (json_buff.contains("invoke_list")) {
    Assign(batch_deploy_info.invoke_deploy_infos, "invoke_list", json_buff);
  }
}

void from_json(const nlohmann::json &json_buff, CompileConfigJson::FlowNodeBatchMemCfg &flow_node_batch_mem_cfg) {
  json_buff.at("std_mem_size").get_to(flow_node_batch_mem_cfg.std_mem_size);
  json_buff.at("shared_mem_size").get_to(flow_node_batch_mem_cfg.shared_mem_size);
  json_buff.at("logic_device_id").get_to(flow_node_batch_mem_cfg.logic_device_list);
}

Status CompileConfigJson::GetResourceTypeFromNumaConfig(std::set<std::string> &resource_types) {
  auto &global_options_mutex = GetGlobalOptionsMutex();
  const std::lock_guard<std::mutex> lock(global_options_mutex);
  auto &global_options = GetMutableGlobalOptions();
  const auto &numa_config_json_string = global_options[OPTION_NUMA_CONFIG];
  GELOGD("Numa config json:%s", numa_config_json_string.c_str());
  try {
    const nlohmann::json numa_config_json = nlohmann::json::parse(numa_config_json_string.c_str());
    GE_CHK_BOOL_RET_STATUS(numa_config_json.contains("node_def"), FAILED,
                           "Failed to get node_def, numa config json:%s.", numa_config_json_string.c_str());
    for (auto &node_def : numa_config_json["node_def"]) {
      GE_CHK_BOOL_RET_STATUS(node_def.contains("resource_type"), FAILED,
                             "Failed to get resource_type, numa config json:%s.", numa_config_json_string.c_str());
      std::string resource_type;
      Assign(resource_type, "resource_type", node_def);
      GE_CHK_BOOL_RET_STATUS(!resource_type.empty(), FAILED,
                             "Failed to get resource_type from node_def, numa config json:%s.",
                             numa_config_json_string.c_str());
      resource_types.insert(resource_type);
    }
    GE_CHK_BOOL_RET_STATUS(numa_config_json.contains("item_def"), FAILED,
                           "Failed to get item_def, numa config json:%s.", numa_config_json_string.c_str());
    for (auto &item_def : numa_config_json["item_def"]) {
      GE_CHK_BOOL_RET_STATUS(item_def.contains("resource_type"), FAILED,
                             "Failed to get resource_type, numa config json:%s.", numa_config_json_string.c_str());
      std::string resource_type;
      Assign(resource_type, "resource_type", item_def);
      GE_CHK_BOOL_RET_STATUS(!resource_type.empty(), FAILED,
                             "Failed to get resource_type from item_def, numa config json:%s.",
                             numa_config_json_string.c_str());
      resource_types.insert(resource_type);
    }
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Invalid numa config json string[%s], err msg: %s.", numa_config_json_string.c_str(), e.what());
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace ge
