/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_EXECUTOR_C_DBG_PARSE_JSON_FILE_H_
#define GE_EXECUTOR_C_DBG_PARSE_JSON_FILE_H_
#include <stdbool.h>
#include "json_parser.h"
#include "framework/executor_c/ge_executor_types.h"
#ifdef __cplusplus
extern "C" {
#endif
#define ACL_GE_INVALID_DUMP_CONFIG 100009
typedef struct {
  char *mdlName;
  Vector layers;
  bool is_layer_null;
} ModelDumpConfig;

typedef struct {
  char *dumpPath;
  uint32_t dumpScene;
  uint32_t dumpMode;
  uint32_t dumpStatus;
  uint32_t dumpOpSwitch;
  uint32_t dumpDebug;
  uint32_t dumpData;
  Vector dumpList;
} DumpConfig;

enum DumpModeType {
  DUMP_MODE_INPUT,
  DUMP_MODE_OUTPUT,
  DUMP_MODE_ALL,
  DUMP_MODE_INVALID
};

enum DumpStatusType {
  DUMP_NORMAL_ENABLE,
  DUMP_DEBUG_ENABLE,
  DUMP_EXCEPTION_ENABLE
};

enum DumpOpSwitchType {
  DUMP_OP_SWITCH_ON,
  DUMP_OP_SWITCH_OFF,
  DUMP_OP_SWITCH_INVALID
};

enum DumpDebugType {
  DUMP_DEBUG_SWITCH_ON,
  DUMP_DEBUG_SWITCH_OFF,
  DUMP_DEBUG_SWITCH_INVALID
};

enum DumpDataType {
  DUMP_DATA_STAT,
  DUMP_DATA_TENSOR,
  DUMP_DATA_INVALID
};

enum DumpSceneType {
  DUMP_SCENE_NORMAL,
  DUMP_SCENE_DEBUG,
  DUMP_SCENE_EXCEPTION,
  DUMP_SCENE_DEFAULT_NORMAL,
  DUMP_SCENE_INVALID
};

bool GetDumpEnableFlag(void);
void SetDumpEnableFlag(bool flag);
Status HandleDumpConfig(const char *configPath, DumpConfig *dumpCfg);
#ifdef __cplusplus
}
#endif
#endif  // GE_EXECUTOR_C_DBG_PARSE_JSON_FILE_H_
