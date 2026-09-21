/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dlog_pub.h"
#include <stdio.h>
#include <fstream>
#include <string.h>
#include <stdarg.h>

#define MSG_LENGTH_STUB   (1024)
#define SET_MOUDLE_ID_MAP_NAME(x) { #x, x}

#ifdef __cplusplus
extern "C" {
#endif

#ifndef LINUX
#define LINUX 0
#endif

#ifndef OS_TYPE
#define OS_TYPE 0
#endif

#if (OS_TYPE == LINUX)
#define PATH_SLOG "/usr/slog/slog"
#else
#define PATH_SLOG "C:\\Program Files\\Huawei\\HiAI Foundation\\log"
#endif

typedef struct _dcode_stub {
  const char *c_name;
  int c_val;
} DCODE_STUB;
//#if (OS_TYPE == LINUX)
//extern void dlog_init(void);
//void DlogErrorInner(int module_id, const char *fmt, ...);
//void DlogWarnInner(int module_id, const char *fmt, ...);
//void DlogInfoInner(int module_id, const char *fmt, ...);
//void DlogDebugInner(int module_id, const char *fmt, ...);
//void DlogEventInner(int module_id, const char *fmt, ...);
//#endif
}

int CheckLogLevel(int moduleId, int level)
{
    return level < 2;
}

void DlogErrorInner(int module_id, const char *fmt, ...)
{
    if(module_id < 0 || module_id >= INVLID_MOUDLE_ID){
        return;
    }

    if (!CheckLogLevel(module_id, 0)) {
        return;
    }

    int len;
    char msg[MSG_LENGTH_STUB] = {0};
    snprintf(msg,MSG_LENGTH_STUB,"[FE] [ERROR] ");
    va_list ap;

    va_start(ap,fmt);
    len = strlen(msg);
    vsnprintf(msg + len, MSG_LENGTH_STUB- len, fmt, ap);
    va_end(ap);

    printf("%s",msg);
    return;
}

void DlogWarnInner(int module_id, const char *fmt, ...)
{
    if(module_id < 0 || module_id >= INVLID_MOUDLE_ID){
        return;
    }

    if (!CheckLogLevel(module_id, 1)) {
        return;
    }

    int len;
    char msg[MSG_LENGTH_STUB] = {0};
    snprintf(msg,MSG_LENGTH_STUB,"[FE] [WARNING] ");
    va_list ap;

    va_start(ap,fmt);
    len = strlen(msg);
    vsnprintf(msg + len, MSG_LENGTH_STUB- len, fmt, ap);
    va_end(ap);

    printf("%s",msg);
    return;
}

void DlogInfoInner(int module_id, const char *fmt, ...)
{
    if(module_id < 0 || module_id >= INVLID_MOUDLE_ID){
        return;
    }

    if (!CheckLogLevel(module_id, 2)) {
        return;
    }

    int len;
    char msg[MSG_LENGTH_STUB] = {0};
    snprintf(msg,MSG_LENGTH_STUB,"[FE] [INFO] ");
    va_list ap;

    va_start(ap,fmt);
    len = strlen(msg);
    vsnprintf(msg + len, MSG_LENGTH_STUB- len, fmt, ap);
    va_end(ap);

    printf("%s",msg);
    return;
}

void DlogDebugInner(int module_id, const char *fmt, ...)
{
    if(module_id < 0 || module_id >= INVLID_MOUDLE_ID){
        return;
    }

    if (!CheckLogLevel(module_id, 3)) {
        return;
    }

    int len;
    char msg[MSG_LENGTH_STUB] = {0};
    snprintf(msg,MSG_LENGTH_STUB,"[FE] [DEBUG] ");
    va_list ap;

    va_start(ap,fmt);
    len = strlen(msg);
    vsnprintf(msg + len, MSG_LENGTH_STUB- len, fmt, ap);
    va_end(ap);

    printf("%s",msg);
    return;
}

void DlogEventInner(int module_id, const char *fmt, ...)
{
    if(module_id < 0 || module_id >= INVLID_MOUDLE_ID){
        return;
    }

    if (!CheckLogLevel(module_id, 4)) {
        return;
    }

    int len;
    char msg[MSG_LENGTH_STUB] = {0};
    snprintf(msg,MSG_LENGTH_STUB,"[FE] [EVENT] ");
    va_list ap;

    va_start(ap,fmt);
    len = strlen(msg);
    vsnprintf(msg + len, MSG_LENGTH_STUB- len, fmt, ap);
    va_end(ap);

    printf("%s",msg);
    return;
}

