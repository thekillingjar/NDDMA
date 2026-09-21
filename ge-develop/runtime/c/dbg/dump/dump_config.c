/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string.h>
#include "tlv_parse.h"
#include "parse_json_file.h"
#include "framework/executor_c/ge_log.h"
#include "ge/ge_error_codes.h"
#include "vector.h"
#include "dump_config.h"

typedef struct {
  uint32_t len;
  char value[0];
} CString;

static DumpConfig g_dumpCfgList;

static void DumpConfigDestroy(void *base) {
  ModelDumpConfig *config = (ModelDumpConfig *)base;
  if (config->mdlName != NULL) {
    (void)mmFree(config->mdlName);
    config->mdlName = NULL;
  }
  DeInitVector(&config->layers);
}

static void InitDumpConfig(void) {
  g_dumpCfgList.dumpPath = NULL;
  InitVector(&g_dumpCfgList.dumpList, sizeof(ModelDumpConfig));
  SetVectorDestroyItem(&g_dumpCfgList.dumpList, DumpConfigDestroy);
}

void FreeDumpConfigRes(void) {
  if (g_dumpCfgList.dumpPath != NULL) {
    (void)mmFree(g_dumpCfgList.dumpPath);
    g_dumpCfgList.dumpPath = NULL;
  }
  DeInitVector(&g_dumpCfgList.dumpList);
  GELOGI("mmFree dumpconfig res success.");
}

Status DumpConfigInit(const char *cfg) {
  InitDumpConfig();
  Status ret = HandleDumpConfig(cfg, &g_dumpCfgList);
  if (ret != SUCCESS) {
    FreeDumpConfigRes();
    return ret;
  }
  return SUCCESS;
}

static ModelDumpConfig *GetModelDumpConfig(const char *mdlName) {
  size_t size = VectorSize(&g_dumpCfgList.dumpList);
  for (size_t index = 0UL; index < size; ++index) {
    ModelDumpConfig *config = (ModelDumpConfig *)VectorAt(&g_dumpCfgList.dumpList, index);
    if ((config != NULL) && (config->mdlName != NULL) && (strcmp(config->mdlName, mdlName) == 0)) {
      GELOGI("modelName[%s] is matched.", mdlName);
      return config;
    }
  }
  return NULL;
}

static bool IsLayerNameMatched(ModelDumpConfig * config, char *name) {
  uint32_t layerSize = (uint32_t)VectorSize(&config->layers);
  for (size_t i = 0UL; i < layerSize; ++i) {
    char *layers = *(char **)VectorAt(&config->layers, i);
    if ((layers != NULL) && (strcmp(name, layers) == 0)) {
      GELOGI("tlv op name[%s] mathces json config layers[%s].", name, layers);
      return true;
    }
  }
  return false;
}

bool IsOriOpNameMatch(uint8_t *opName, uint16_t opNameLen, const char *mdlName) {
  if (opName == NULL) {
    return false;
  }
  ModelDumpConfig *config = GetModelDumpConfig(mdlName);
  if (config == NULL) {
    return false;
  }
  if (config->is_layer_null == false) {
    return true;
  }
  uint32_t totalLen = 0U;
  uint8_t *opOriNameTmp = opName;
  while (totalLen + sizeof(CString) < opNameLen) {
    CString *opOriName = (CString *)opOriNameTmp;
    if (opOriName->len == 0U) {
      return false;
    }
    if ((totalLen + opOriName->len) < opOriName->len) {
      return false;
    }
    char *name = StringDepthCpy(opOriName->len, (void *)(opOriNameTmp + sizeof(CString)));
    if (name == NULL) {
      return false;
    }
    GELOGI("IsOriOpNameMatch name[%s] len[%zu].", name, strlen(name));
    if (IsLayerNameMatched(config, name)) {
      mmFree(name);
      name = NULL;
      return true;
    }
    totalLen += opOriName->len + sizeof(CString);
    opOriNameTmp = opName + totalLen;
    mmFree(name);
    name = NULL;
  }
  GELOGI("IsOriOpNameMatch is not mathched.");
  return false;
}

bool IsOpNameMatch(uint8_t *opName, uint16_t opNameLen, const char *mdlName) {
  if (opName == NULL) {
    return false;
  }
  ModelDumpConfig *config = GetModelDumpConfig(mdlName);
  if (config == NULL) {
    return false;
  }
  if (config->is_layer_null == false) {
    return true;
  }
  char *name = StringDepthCpy(opNameLen, (void *)opName);
  if (name == NULL) {
    return false;
  }

  GELOGI("IsOpNameMatch name[%s] len[%zu].", name, strlen(name));
  if (IsLayerNameMatched(config, name)) {
    mmFree(name);
    name = NULL;
    return true;
  }
  mmFree(name);
  name = NULL;
  return false;
}

char *GetDumpPath(void) {
  return g_dumpCfgList.dumpPath;
}

uint32_t GetDumpData(void) {
  return g_dumpCfgList.dumpData;
}

uint32_t GetDumpMode(void) {
  return g_dumpCfgList.dumpMode;
}

uint32_t GetDumpStatus(void) {
  return g_dumpCfgList.dumpStatus;
}