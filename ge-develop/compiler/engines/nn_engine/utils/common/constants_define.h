/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_UTILS_COMMON_CONSTANTS_DEFINE_H_
#define FUSION_ENGINE_UTILS_COMMON_CONSTANTS_DEFINE_H_

#include <map>
#include <string>
#include <vector>
#include "common/aicore_util_attr_define.h"
#include "common/aicore_util_types.h"
#include "common/aicore_util_constants.h"

namespace fe {
// config key in fe.ini
const std::string CONFIG_KEY_ROOT = "rootdir";
const std::string CONFIG_KEY_CUSTOM_PASS_FILE = "fusionpassmgr.aicore.custompasspath";
const std::string CONFIG_KEY_BUILTIN_PASS_FILE = "fusionpassmgr.aicore.graphpasspath";
const std::string CONFIG_KEY_COMPILER_PASS_FILE = "fusionpassmgr.aicore.compiler.graphpasspath";
const std::string CONFIG_KEY_GRAPH_FILE = "fusionrulemgr.aicore.graphfilepath";
const std::string CONFIG_KEY_COMPILER_GRAPH_FILE = "fusionrulemgr.aicore.compiler.graphfilepath";
const std::string CONFIG_KEY_CUSTOM_FILE = "fusionrulemgr.aicore.customfilepath";
const std::string CONFIG_KEY_ORIGINAL_NODES_ENABLE = "dump.originalnodes.enable";
const std::string CONFIG_KEY_MIX_L2_ENABLE = "mix_l2.enable";
const std::string CONFIG_KEY_SUPERKERNEL_PLUS_ENABLE = "superkernel_plus.enable";
const std::string CONFIG_KEY_MAY_DUPLICATE_IN_AUTO_FUSION = "may.duplicate.in.autofusion";
const std::string CONFIG_KEY_SGT_SLICE = "sgtslice";
const std::string VECTOR_CORE_CONFIG_KEY_CUSTOM_PASS_FILE = "fusionpassmgr.vectorcore.custompasspath";
const std::string VECTOR_CORE_CONFIG_KEY_BUILTIN_PASS_FILE = "fusionpassmgr.vectorcore.graphpasspath";
const std::string VECTOR_CORE_CONFIG_KEY_COMPILER_PASS_FILE = "fusionpassmgr.vectorcore.compiler.graphpasspath";
const std::string VECTOR_CORE_CONFIG_KEY_GRAPH_FILE = "fusionrulemgr.vectorcore.graphfilepath";
const std::string VECTOR_CORE_CONFIG_KEY_COMPILER_GRAPH_FILE = "fusionrulemgr.vectorcore.compiler.graphfilepath";
const std::string VECTOR_CORE_CONFIG_KEY_CUSTOM_FILE = "fusionrulemgr.vectorcore.customfilepath";
const std::string DSA_CORE_CONFIG_KEY_CUSTOM_PASS_FILE = "fusionpassmgr.dsacore.custompasspath";
const std::string DSA_CORE_CONFIG_KEY_BUILTIN_PASS_FILE = "fusionpassmgr.dsacore.graphpasspath";
const std::string DSA_CORE_CONFIG_KEY_GRAPH_FILE = "fusionrulemgr.dsacore.graphfilepath";
const std::string DSA_CORE_CONFIG_KEY_CUSTOM_FILE = "fusionrulemgr.dsacore.customfilepath";
const std::string FUSION_CONFIG_BUILT_IN_FILE = "fusion.config.built-in.file";
const std::string FUSION_CONFIG_COMPILER_FILE = "fusion.config.compiler.file";
const std::string SUPPORT_FUSSION_PASS_FILE = "built-in.support.pass.file";

// constants value
const int32_t PRIORITY_START_NUM = 0;
const int32_t OP_STORE_FORMAT_MAX_SIZE = 7;  // The size of opstore config items in fe.ini file

const int32_t DATA_VISIT_DIST_THRESHOLD = 5; // data distance threshlod for rc cache optimization
const int32_t MEM_REUSE_DIST_THRESHOLD = 2; // mem reuse distance threshold

const std::map<L2CacheReadMode, std::string> L2CACHE_READ_MODE_STRING_MAP{
        {L2CacheReadMode::RM_NONE,            "None"},
        {L2CacheReadMode::READ_LAST,          "Read Last"},
        {L2CacheReadMode::READ_INVALID,       "Read Invalid"},
        {L2CacheReadMode::NOT_NEED_WRITEBACK, "Not Need Writeback"}
};

const std::string BUFFER_OPTIMIZE_UNKNOWN = "unknown-buffer-optimize";
const std::string BUFFER_FUSION_MODE_UNKNOWN = "unknown-buffer-fusion-mode";

const std::map<BufferOptimize, std::string> BUFFER_OPTIMIZE_STRING_MAP{
        {EN_UNKNOWN_OPTIMIZE, "unknown_optimize"},
        {EN_OFF_OPTIMIZE,     "off_optimize"},
        {EN_L1_OPTIMIZE,      "l1_optimize"},
        {EN_L2_OPTIMIZE,      "l2_optimize"}
};

const std::map<OpPattern, std::string> OP_PATTERN_STRING_MAP{{OP_PATTERN_OP_KERNEL,          "kernel"},
                                                             {OP_PATTERN_FORMAT_AGNOSTIC,    "formatAgnostic"},
                                                             {OP_PATTERN_BROADCAST,          "broadcast"},
                                                             {OP_PATTERN_BROADCAST_ENHANCED, "broadcastEnhanced"},
                                                             {OP_PATTERN_REDUCE,             "reduce"},
                                                             {OP_PATTERN_OP_CUSTOMIZE,       "dynamic"},
                                                             {OP_PATTERN_RANGE_AGNOSTIC,     "rangeAgnostic"}};

const std::map<PrecisionPolicy, std::string> PRECISION_POLICY_STRING_MAP{
        {WHITE, "white-list"},
        {BLACK, "black-list"},
        {GRAY,  "gray-list"}
};
}
#endif  // FUSION_ENGINE_UTILS_COMMON_CONSTANTS_DEFINE_H_
