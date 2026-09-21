/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include "tlv_parse.h"
#include "framework/executor_c/ge_log.h"
#include "ge/ge_error_codes.h"
#include "framework/executor_c/ge_executor.h"
#include "json_parser.h"
#include "parse_json_file.h"
const char *DBG_DUMP_MODE_INPUT = "input";
const char *DBG_DUMP_MODE_OUTPUT = "output";
const char *DBG_DUMP_MODE_ALL = "all";
const char *DBG_DUMP_STATUS_SWITCH_ON = "on";
const char *DBG_DUMP_STATUS_SWITCH_OFF = "off";
const char *DBG_DUMP_MODE_NAME = "model_name";
const char *DBG_DUMP_LAYER = "layer";
const char *DBG_DUMP_PATH = "dump_path";
const char *DBG_DUMP_OP_SWITCH = "dump_op_switch";
const char *DBG_DUMP_LIST = "dump_list";
const char *DBG_DUMP_MODE = "dump_mode";
const char *DBG_DUMP_DATA = "dump_data";
const char *DBG_DUMP_DEBUG = "dump_debug";
const char *DBG_DUMP = "dump";
const char *DBG_DUMP_DATA_TENSOR = "tensor";
const char *DBG_DUMP_DATA_STAT = "stats";
const char *DBG_DUMP_SCENE = "dump_scene";
const char *DBG_DUMP_SCENE_NORMAL = "normal";
const char *DBG_DUMP_SCENE_DEBUG = "debug";
const char *DBG_DUMP_SCENE_EXCEPTION = "exception";
static bool dumpEnableFlag = false;

void SetDumpEnableFlag(bool flag) {
  dumpEnableFlag = flag;
}

bool GetDumpEnableFlag(void) {
  return dumpEnableFlag;
}

static void LayersDestroy(void *base) {
  char *config = *(char **)base;
  if (config != NULL) {
    (void)mmFree(config);
    config = NULL;
  }
}

static bool CheckDumpPath(char *dumpPath) {
  char trustPath[PATH_MAX] = {};
  if (mmRealPath(dumpPath, trustPath, PATH_MAX) != EN_OK) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "realPath is invalid.");
    return false;
  }
  const uint32_t accessMode = (uint32_t)MM_R_OK | (uint32_t)MM_W_OK;
  if (mmAccess(trustPath, (int32_t)accessMode) != EN_OK) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "accessMode is invalid.");
    return false;
  }
  return true;
}

static bool DumpPathCheck(DumpConfig *config) {
  if (!CheckDumpPath(config->dumpPath)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[check][dumpPath] field in dump config is invalid");
    return false;
  }
  return true;
}

static bool DumpDataCheck(DumpConfig *config) {
  if (config->dumpData != DUMP_DATA_STAT &&
      config->dumpData != DUMP_DATA_TENSOR) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[check][dumpData] value[%u] error in config, only supports "
         "stats/tensor", config->dumpData);
    return false;
  }
  return true;
}

static bool DumpOtherSceneCheck(DumpConfig *config) {
  if (config->dumpOpSwitch == DUMP_OP_SWITCH_ON) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "dump_op_switch is on when overflow or exception dump is unsupported.");
    return false;
  }
  return true;
}

static bool DumpModeCheck(DumpConfig *config) {
  if (config->dumpMode != DUMP_MODE_INPUT &&
      config->dumpMode != DUMP_MODE_OUTPUT &&
      config->dumpMode != DUMP_MODE_ALL) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[check][dumpMode] value[%u] error in config, only supports "
         "input/output/all", config->dumpMode);
    return false;
  }
  return true;
}

static bool DumpSceneCheck(DumpConfig *config) {
  if (config->dumpScene != DUMP_SCENE_NORMAL &&
      config->dumpScene != DUMP_SCENE_DEBUG &&
      config->dumpScene != DUMP_SCENE_EXCEPTION &&
      config->dumpScene != DUMP_SCENE_DEFAULT_NORMAL) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[check][dumpScene] value[%u] error in config, only supports "
         "normal/debug/exception", config->dumpScene);
    return false;
  }
  return true;
}

static bool DumpListCheck(DumpConfig *config) {
  if ((config->dumpOpSwitch != DUMP_OP_SWITCH_ON) &&
      (config->dumpOpSwitch != DUMP_OP_SWITCH_OFF)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[check][dumpOpSwitch] dump_op_switch value[%u] is invalid in config, only supports "
         "on/off", config->dumpOpSwitch);
    return false;
  }
  size_t dumpListSize = VectorSize(&config->dumpList);
  if ((dumpListSize == 0) && (config->dumpOpSwitch == DUMP_OP_SWITCH_OFF)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[check][dumpConfig] dump_list field is null and dump_op_switch is off in config, "
         "dump config is invalid.");
    return false;
  }

  if (config->dumpOpSwitch == DUMP_OP_SWITCH_OFF) {
    bool isValidList = false;
    for (size_t index = 0UL; index < dumpListSize; ++index) {
      ModelDumpConfig *modelDumpConfig = (ModelDumpConfig *)VectorAt(&config->dumpList, index);
      size_t layerSize = VectorSize(&modelDumpConfig->layers);
      if (!(modelDumpConfig->is_layer_null && (layerSize == 0))) {
          GELOGI("layer field is valid.");
          isValidList = true;
          break;
      }
    }
    if (!isValidList) {
        GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[check][ValidDumpList] dump_list is not null ,but dump_list filed invalid, "
             "dump config is invalid.");
        return false;
    }
    GELOGI("dump_list is valid, dump_op_switch is only dump model.");
  }
  return true;
}

static bool IsValidDumpConfig(DumpConfig *config) {
  if (!DumpSceneCheck(config)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "dump scene is invalid.");
    return false;
  }
  // if dump_path is not exist, can't send dump config
  if (!DumpPathCheck(config)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "dump path is invalid.");
    return false;
  }
  if (!DumpModeCheck(config)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "dump mode is invalid.");
    return false;
  }

  if (!DumpDataCheck(config)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "dump data is invalid.");
    return false;
  }
  if ((config->dumpStatus == DUMP_EXCEPTION_ENABLE) || (config->dumpStatus == DUMP_DEBUG_ENABLE)) {
    if (!DumpOtherSceneCheck(config)) {
      return false;
    }
    return true;
  }
  if (!DumpListCheck(config)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "dump list is invalid.");
    return false;
  }
  return true;
}

static void ConvertDumpModeCfg(CJsonObj *json, DumpConfig *dumpCfg) {
  uint32_t dumpMode = DUMP_MODE_INVALID;
  const char *dumpModeStr = "";
  CJsonObj *dumpModeJson = GetCJsonSubObj(json, DBG_DUMP_MODE);
  if (dumpModeJson != NULL) {
    if (CJsonIsString(dumpModeJson)) {
      dumpModeStr = GetCJsonString(dumpModeJson);
    }
  } else {
    dumpCfg->dumpMode = DUMP_MODE_OUTPUT; // not config dumpMode default value
    return;
  }

  if (strcmp(dumpModeStr, DBG_DUMP_MODE_INPUT) == 0) {
    dumpMode = DUMP_MODE_INPUT;
  } else if (strcmp(dumpModeStr, DBG_DUMP_MODE_OUTPUT) == 0){
    dumpMode = DUMP_MODE_OUTPUT;
  } else if (strcmp(dumpModeStr, DBG_DUMP_MODE_ALL) == 0) {
    dumpMode = DUMP_MODE_ALL;
  }
  dumpCfg->dumpMode = dumpMode;
  GELOGI("dumpMode is %u", dumpCfg->dumpMode);
  return;
}

static void ConvertDumpDebugCfg(CJsonObj *json, DumpConfig *dumpCfg) {
  uint32_t dumpDebug = DUMP_DEBUG_SWITCH_INVALID;
  const char *dumpDebugStr = "";
  CJsonObj *dumpDebugJson = GetCJsonSubObj(json, DBG_DUMP_DEBUG);
  if (dumpDebugJson != NULL) {
    if (CJsonIsString(dumpDebugJson)) {
      dumpDebugStr = GetCJsonString(dumpDebugJson);
    }
  } else {
    dumpCfg->dumpDebug = DUMP_DEBUG_SWITCH_OFF; // not config dumpDebug default value
    return;
  }

  if (strcmp(dumpDebugStr, DBG_DUMP_STATUS_SWITCH_OFF) == 0) {
    dumpDebug = DUMP_DEBUG_SWITCH_OFF;
  } else if (strcmp(dumpDebugStr, DBG_DUMP_STATUS_SWITCH_ON) == 0) {
    dumpDebug = DUMP_DEBUG_SWITCH_ON;
  }
  dumpCfg->dumpDebug = dumpDebug;
  GELOGI("dumpDebug is %u", dumpCfg->dumpDebug);
  return;
}

static void ConvertDumpSceneCfg(CJsonObj *json, DumpConfig *dumpCfg) {
  uint32_t dumpScene = DUMP_SCENE_INVALID;
  const char *dumpSceneStr = "";
  CJsonObj *dumpSceneJson = GetCJsonSubObj(json, DBG_DUMP_SCENE);
  if (dumpSceneJson != NULL) {
    if (CJsonIsString(dumpSceneJson)) {
      dumpSceneStr = GetCJsonString(dumpSceneJson);
    }
  } else {
    dumpCfg->dumpScene = DUMP_SCENE_DEFAULT_NORMAL; // not config dumpScene default value
    return;
  }

  if (strcmp(dumpSceneStr, DBG_DUMP_SCENE_NORMAL) == 0) {
    dumpScene = DUMP_SCENE_NORMAL;
  } else if (strcmp(dumpSceneStr, DBG_DUMP_SCENE_DEBUG) == 0) {
    dumpScene = DUMP_SCENE_DEBUG;
  } else if (strcmp(dumpSceneStr, DBG_DUMP_SCENE_EXCEPTION) == 0) {
    dumpScene = DUMP_SCENE_EXCEPTION;
  }
  dumpCfg->dumpScene = dumpScene;
  GELOGI("dumpScene is %u", dumpCfg->dumpScene);
  return;
}

static void ConvertDumpDataCfg(CJsonObj *json, DumpConfig *dumpCfg) {
  uint32_t dumpData = DUMP_DATA_INVALID;
  const char *dumpDataStr = "";
  CJsonObj *dumpDataJson = GetCJsonSubObj(json, DBG_DUMP_DATA);
  if (dumpDataJson != NULL) {
    if (CJsonIsString(dumpDataJson)) {
      dumpDataStr = GetCJsonString(dumpDataJson);
    }
  } else {
    dumpCfg->dumpData = DUMP_DATA_STAT; // not config dumpMode default value
    return;
  }

  if (strcmp(dumpDataStr, DBG_DUMP_DATA_TENSOR) == 0) {
    dumpData = DUMP_DATA_TENSOR;
  } else if (strcmp(dumpDataStr, DBG_DUMP_DATA_STAT) == 0){
    dumpData = DUMP_DATA_STAT;
  }
  dumpCfg->dumpData = dumpData;
  GELOGI("dumpData is %u", dumpCfg->dumpData);
  return;
}

static Status ConvertDumpPathCfg(CJsonObj *json, DumpConfig *dumpCfg) {
  const char *dumpPathStr = "";
  CJsonObj *dumpPathJson = GetCJsonSubObj(json, DBG_DUMP_PATH);
  if (dumpPathJson != NULL) {
    if (CJsonIsString(dumpPathJson)) {
      dumpPathStr = GetCJsonString(dumpPathJson);
    }
  }
  dumpCfg->dumpPath = StringDepthCpy((uint32_t)strlen(dumpPathStr), dumpPathStr);
  if (dumpCfg->dumpPath == NULL) {
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }
  GELOGI("dumpPathStr = %s dumpPath = %s, is valid", dumpPathStr, dumpCfg->dumpPath);
  return SUCCESS;
}

static void ConvertDumpOpSwtichCfg(CJsonObj *json, DumpConfig *dumpCfg) {
  uint32_t dumpOpSwitch = DUMP_OP_SWITCH_INVALID;
  const char *dumpOpSwitchStr = "";
  CJsonObj *dumpOpSwitchJson = GetCJsonSubObj(json, DBG_DUMP_OP_SWITCH);
  if (dumpOpSwitchJson != NULL) {
    if (CJsonIsString(dumpOpSwitchJson)) {
      dumpOpSwitchStr = GetCJsonString(dumpOpSwitchJson);
    }
  } else {
    dumpCfg->dumpOpSwitch = DUMP_OP_SWITCH_OFF; // not config dumpOpSwitch default value
    return;
  }

  if (strcmp(dumpOpSwitchStr, DBG_DUMP_STATUS_SWITCH_ON) == 0) {
    dumpOpSwitch = DUMP_OP_SWITCH_ON;
  } else if(strcmp(dumpOpSwitchStr, DBG_DUMP_STATUS_SWITCH_OFF) == 0) {
    dumpOpSwitch = DUMP_OP_SWITCH_OFF;
  }
  dumpCfg->dumpOpSwitch = dumpOpSwitch;
  GELOGI("dumpOpSwitch is %u", dumpCfg->dumpOpSwitch);
}

static Status GetModelNameDumpConfig(CJsonObj *jsonObjIt, ModelDumpConfig *modelDumpConfig,
                                     size_t index) {
  CJsonObj *modelNameJson = GetCJsonSubObj(jsonObjIt, DBG_DUMP_MODE_NAME);
  const char *modelNameStr = "";
  if (modelNameJson != NULL) {
    if (CJsonIsString(modelNameJson)){
      modelNameStr = GetCJsonString(modelNameJson);
    }
  }
  if (strlen(modelNameStr) == 0UL) {
    modelDumpConfig->mdlName = NULL;
    return SUCCESS;
  }
  char *modelName = StringDepthCpy((uint32_t)strlen(modelNameStr), modelNameStr);
  if (modelName == NULL) {
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }
  modelDumpConfig->mdlName = modelName;
  GELOGI("dump list index %zu mdlName = %s.", index, modelName);
  return SUCCESS;
}

static Status GetModelLayerDumpConfig(CJsonObj *jsonObjIt, ModelDumpConfig *modelDumpConfig,
                                      size_t index) {
  InitVector(&modelDumpConfig->layers, sizeof(char *));
  SetVectorDestroyItem(&modelDumpConfig->layers, LayersDestroy);
  CJsonObj *layerJson = GetCJsonSubObj(jsonObjIt, DBG_DUMP_LAYER);
  if (layerJson == NULL) {
    modelDumpConfig->is_layer_null = false;
    return SUCCESS;
  }
  modelDumpConfig->is_layer_null = true;
  size_t layerSize = GetCJsonArraySize(layerJson);
  for (size_t i = 0; i < layerSize; ++i) {
    const char *layerStr = "";
    CJsonObj *jsonObjIter = CJsonArrayAt(layerJson, i);
    if (jsonObjIter != NULL) {
      if (CJsonIsString(jsonObjIter)){
        layerStr = GetCJsonString(jsonObjIter);
      }
    }
    char *layerName = StringDepthCpy((uint32_t)strlen(layerStr), layerStr);
    if (layerName == NULL) {
      return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
    }
    GELOGI("dump list index:%zu  i:%zu layerName= %s.", index, i, layerName);
    EmplaceBackVector(&modelDumpConfig->layers, &layerName);
  }
  return SUCCESS;
}

static Status ConvertDumpListCfg(CJsonObj *json, DumpConfig *dumpCfg) {
  CJsonObj *dumpListJson = GetCJsonSubObj(json, DBG_DUMP_LIST);
  if (dumpListJson == NULL) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  if (!CJsonIsArray(dumpListJson)) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  size_t dumpListSize = GetCJsonArraySize(dumpListJson);
  for (size_t index = 0UL; index < dumpListSize; ++index) {
    ModelDumpConfig modelDumpConfig;
    CJsonObj *jsonObjIt = CJsonArrayAt(dumpListJson, index);
    Status ret = GetModelNameDumpConfig(jsonObjIt, &modelDumpConfig, index);
    if (ret != SUCCESS) {
      return ret;
    }
    if (modelDumpConfig.mdlName == NULL) {
      continue;
    }
    ret = GetModelLayerDumpConfig(jsonObjIt, &modelDumpConfig, index);
    if (ret != SUCCESS) {
      mmFree(modelDumpConfig.mdlName);
      return ret;
    }
    EmplaceBackVector(&dumpCfg->dumpList, &modelDumpConfig);
  }
  return SUCCESS;
}

static Status ConvertDumpListCfgOtherScene(CJsonObj *json) {
  CJsonObj *dumpListJson = GetCJsonSubObj(json, DBG_DUMP_LIST);
  if (dumpListJson == NULL) {
    return SUCCESS;
  }
  if (!CJsonIsArray(dumpListJson)) {
    return ACL_GE_INVALID_DUMP_CONFIG;
  }
  size_t dumpListSize = GetCJsonArraySize(dumpListJson);
  if (dumpListSize != 0UL) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "unsupported when open overflow or exception dump.");
    return ACL_GE_INVALID_DUMP_CONFIG;
  }
  return SUCCESS;
}


static Status ConvertDumpCfg(CJsonObj *json, DumpConfig *dumpCfg) {
  // get config info from json first
  ConvertDumpSceneCfg(json, dumpCfg);
  Status ret = ConvertDumpPathCfg(json, dumpCfg);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[convert][dumpConfig] dump path convert failed ret[%u].", ret);
    return ret;
  }
  ConvertDumpModeCfg(json, dumpCfg);
  ConvertDumpDataCfg(json, dumpCfg);
  ConvertDumpOpSwtichCfg(json, dumpCfg);
  ConvertDumpDebugCfg(json, dumpCfg);
  if ((dumpCfg->dumpScene == DUMP_SCENE_NORMAL) || ((dumpCfg->dumpScene == DUMP_SCENE_DEFAULT_NORMAL)
      && (dumpCfg->dumpDebug == DUMP_DEBUG_SWITCH_OFF))) {
    dumpCfg->dumpStatus = DUMP_NORMAL_ENABLE;
    ret = ConvertDumpListCfg(json, dumpCfg);
  } else if (dumpCfg->dumpScene == DUMP_SCENE_EXCEPTION) {
    dumpCfg->dumpStatus = DUMP_EXCEPTION_ENABLE;
    ret = ConvertDumpListCfgOtherScene(json);
  } else if ((dumpCfg->dumpScene == DUMP_SCENE_DEBUG) || (dumpCfg->dumpDebug == DUMP_DEBUG_SWITCH_ON)) {
    dumpCfg->dumpStatus = DUMP_DEBUG_ENABLE;
    ret = ConvertDumpListCfgOtherScene(json);
  }
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[convert][dumpConfig] dump list convert failed ret[%u].", ret);
    return ret;
  }
  GELOGI("dump_status is %u.", dumpCfg->dumpStatus);
  // check config info
  if (!IsValidDumpConfig(dumpCfg)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[check][dumpConfig] dump config is invalid.");
    return ACL_GE_INVALID_DUMP_CONFIG;
  }
  GELOGI("convert dump config success dump_Scene[%u], dump_mode[%u], dump_path[%s], dump_data[%u],"
         "dump_debug[%u], dump_op_switch[%u], dump_status[%u]",
         dumpCfg->dumpScene, dumpCfg->dumpMode, dumpCfg->dumpPath, dumpCfg->dumpData,
         dumpCfg->dumpDebug, dumpCfg->dumpOpSwitch, dumpCfg->dumpStatus);
  return SUCCESS;
}

Status HandleDumpConfig(const char *configPath, DumpConfig *dumpCfg) {
  CJsonObj *obj = CJsonFileParse(configPath);
  if (obj == NULL) {
    GELOGW("json config file is error.");
    return ACL_GE_INVALID_DUMP_CONFIG;
  }
  CJsonObj *dump = GetCJsonSubObj(obj, DBG_DUMP);
  if (dump != NULL) {
    Status ret = ConvertDumpCfg(dump, dumpCfg);
    if (ret != SUCCESS) {
      FreeCJsonObj(obj);
      GELOGW("convert dump config failed ret[%u].", ret);
      return ret;
    }
    SetDumpEnableFlag(true);
  } else {
    FreeCJsonObj(obj);
    GELOGW("dump item not config, no need to dump.");
    return ACL_GE_INVALID_DUMP_CONFIG;
  }
  FreeCJsonObj(obj);
  return SUCCESS;
}
