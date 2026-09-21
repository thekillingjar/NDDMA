/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_GRAPH_UTILS_GE_DUMP_GRAPH_WHILELIST_H
#define COMMON_GRAPH_UTILS_GE_DUMP_GRAPH_WHILELIST_H

namespace ge {
// kGeDumpWhitelistFullName + kGeDumpWhitelistKeyName size need small than 100
const std::set<std::string> kGeDumpWhitelistFullName = {
    "PreRunBegin",                          // 用户原始图
    "PreRunAfterNormalizeGraph",            // 图标准化出口图
    "AfterInfershape",                      // infershape出口图
    "PreRunAfterPrepare",                   // 图准备阶段之后的图
    "PreRunAfterOptimizeOriginalGraph",     // 原图优化之后的图
    "PreRunAfterOptimizeAfterStage1",       // 各算子信息库优化处理之后的图
    "PreRunAfterOptimizeSubgraph",          // 子图优化之后的图
    "PreRunAfterOptimizeGraphBeforeBuild",  // 模型编译入口图
    "Build",                                // 模型编译出口图
    "ComputeGraphBeforeLowering",           // lowering前的计算图
    "ExeGraphBeforeOptimize",               // lowering后，执行图优化前的执行图
    "ExecuteGraphAfterSplit"                // 动态shape最终的执行图
};
const std::set<std::string> kGeDumpWhitelistKeyName = {
    "AutoFuseBeforeOptimize",  // 自动融合优化之前的图
    "AutoFuseAfterOptimize",   // 自动融合优化之后的图
    "RunCustomPass"            // 用户自定义pass优化之后的图
};
} // namespace ge

#endif  // COMMON_GRAPH_UTILS__GE_DUMP_GRAPH_WHILELIST_H
