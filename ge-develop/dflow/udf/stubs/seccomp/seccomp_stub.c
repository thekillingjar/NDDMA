/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "seccomp.h"

scmp_filter_ctx seccomp_init(uint32_t def_action) {
    // just stub for compile, do nothing
    (void)def_action;
    static uint8_t ctx_buf[128] = {0};
    return (scmp_filter_ctx)ctx_buf;
}

int seccomp_rule_add(scmp_filter_ctx ctx, uint32_t action, int syscall, unsigned int arg_cnt, ...) {
    // just stub for compile, do nothing
    (void)ctx;
    (void)action;
    (void)syscall;
    (void)arg_cnt;
    return 0;
}

int seccomp_load(const scmp_filter_ctx ctx) {
    // just stub for compile, do nothing
    (void)ctx;
    return 0;
}

int seccomp_attr_set(scmp_filter_ctx ctx, enum scmp_filter_attr attr, uint32_t value) {
    // just stub for compile, do nothing
    (void)ctx;
    (void)attr;
    (void)value;
    return 0;
}