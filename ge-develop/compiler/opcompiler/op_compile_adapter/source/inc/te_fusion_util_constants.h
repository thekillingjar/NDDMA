/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TE_FUSION_UTIL_CONSTANTS_H
#define TE_FUSION_UTIL_CONSTANTS_H

#include <string>

#include "tensor_engine/fusion_types.h"
#include "inc/te_fusion_types.h"

namespace te {
namespace fusion {
constexpr const char *AI_CORE = "AiCore";
constexpr const char *VECTOR_CORE = "VectorCore";
constexpr const char *MIX_VECTOR_CORE = "MIX_VECTOR_CORE";
constexpr const char *FE_IMPLY_TYPE = "_fe_imply_type";
constexpr const char *INPUTS = "inputs";
constexpr const char *OUTPUTS = "outputs";
constexpr const char *INT64_MODE = "int64Mode";
constexpr const char *DETERMINISTIC = "deterministic";
constexpr const char *CONST_VALUE = "const_value";
constexpr const char *CONST_VALUE_NULL_DESC = "const_value_null_desc";
constexpr const char *CONST_VALUE_RANGE = "const_value_range";
constexpr const char *L1_FUSION = "l1_fusion";
constexpr const char *IMPL_MODE = "implMode";
constexpr const char *RANGE = "range";
constexpr const char *ORI_RANGE = "ori_range";
constexpr const char *ORI_FORMAT = "ori_format";
constexpr const char *ORI_SHAPE = "ori_shape";
constexpr const char *ADDR_TYPE = "addr_type";
constexpr const char *USE_L1_WORKSPACE = "use_L1_workspace";
constexpr const char *L1_ADDR_FLAG = "L1_addr_flag";
constexpr const char *L1_FUSION_TYPE = "L1_fusion_type";
constexpr const char *SPLIT_INDEX = "split_index";
constexpr const char *L1_WORKSPACE_SIZE = "L1_workspace_size";
constexpr const char *L1_ADDR_OFFSET = "L1_addr_offset";
constexpr const char *L1_VALID_SIZE = "L1_valid_size";
constexpr const char *IS_FIRST_LAYER = "is_first_layer";
constexpr const char *SLICE_OFFSET = "slice_offset";
constexpr const char *VALID_SHAPE = "valid_shape";
constexpr const char *TOTAL_SHAPE = "total_shape";
constexpr const char *STATIC_KEY = "staticKey";
constexpr const char *ATTRS = "attrs";
constexpr const char *GRAPH_OP_PARAMS = "graphOpParams";
constexpr const char *FLOAT = "float";
constexpr const char *ATTR_LIST_FLOAT = "list_float";
constexpr const char *VALUE_RANGE = "value_range";
constexpr const char *FORMAT = "format";
constexpr const char *SHAPE = "shape";
constexpr const char *VALUE = "value";
constexpr const char *DTYPE = "dtype";
constexpr const char *ATTR_NAME_L1_FUSION_SCOPE = "_l1_fusion_scope";
constexpr const char *ATTR_NAME_IS_OP_DYNAMIC_IMPL = "_is_op_dynamic_impl";
constexpr const char *ATTR_NAME_SUPPORT_DYNAMIC_SHAPE = "support_dynamicshape";
constexpr const char *ATTR_NAME_GRAPH_NAME = "_graph_name";
constexpr const char *COMPILE_STRATEGY_KEEP_COMPILING = "set by fe: keep compiling in op tune";
constexpr const char *COMPILE_STRATEGY_NO_TUNE = "NoTune";
constexpr const char *DATA = "Data";
constexpr const char *UNSUPPORTED = "UNSUPPORTED";
constexpr const char *RANGE_UPPER_LIMIT = "upper_limit";
constexpr const char *RANGE_LOWER_LIMIT = "lower_limit";
constexpr const char *TE_OPAQUE_PATTERN = "Opaque";
constexpr const char *FUZZILY_BUILD_TYPE = "fuzzily_build";
constexpr const char *BINARY_BUILD_TYPE = "binary_build";
constexpr const char *ATTR_NAME_IS_LIMITED_GRAPH = "_is_limited_graph";
constexpr const char *FUSION_OP_BUILD_OPTIONS = "fusion_op_build_options";
constexpr const char *OPS_PATH_NAME_PREFIX = "ops_path_name_prefix";
constexpr const char *ATTR_NAME_CAN_NOT_REUSE_OM = "_can_not_reuse_om";
constexpr const char *ATTR_NAME_OP_PATTERN = "_pattern";
constexpr const char *STATUS_CHECK = "status_check";
constexpr const char *SOFTSYNC_DYNAMIC_IMPL = "_softsync_dynamic_impl";
constexpr const char *OPTIONAL_INPUT_MODE = "optional_input_mode";
constexpr const char *DYNAMIC_PARAM_MODE = "dynamic_param_mode";
constexpr const char *STR_UNDEFINDED = "undefined";
constexpr const char *AUTO_FUSION_PATTERN = "_auto_fusion_pattern";
constexpr const char *ONLY_FUSION_CHECK = "_only_fusion_check";
constexpr const char *STR_TRUE = "true";
constexpr const char *STR_FALSE = "false";
constexpr const char *kEnableVectorCore = "enable_vector_core";
constexpr const char *KERNEL_NAME = "kernel_name";
constexpr const char *SUPERKERNEL_PREFIX = "te_superkernel_";
constexpr const char *ENABLE_SPK = "enable_super_kernel";
constexpr const char *SPK_OPTIONS = "_super_kernel_options";
constexpr const char *ASCENDC_SPK_OPTIONS = "super_kernel_options";
constexpr const char *SPK_KERNELNAME = "superkernel_kernelname";
constexpr const char *SPK_CNT = "super_kernel_count";
constexpr const char *SPK_SUB_ID = "super_kernel_sub_id";
constexpr const char *SPK_REUSED_BINARY = "super_kernel_reuse_binary";
constexpr const char *ASCENDC_SPK_SUB_ID = "_ascendc_sp_sub_id";
constexpr const char *ASCENDC_SPK_CNT = "_ascendc_sp_cnt";
constexpr const char *OP_LIST = "op_list";
constexpr const char *BIN_PATH = "bin_path";
constexpr const char *JSON_PATH = "json_path";
constexpr const char *kAiCoreCnt = "ai_core_cnt";
constexpr const char *kCubeCoreCnt = "cube_core_cnt";
constexpr const char *kVectorCoreCnt = "vector_core_cnt";
constexpr const char *kAicCntKeyOp = "_op_aicore_num";
constexpr const char *kAivCntKeyOp = "_op_vectorcore_num";

// python api function name
const std::string FUNC_GET_SPECIFIC_INFO = "get_op_specific_info";
const std::string FUNC_CHECK_SUPPORTED = "check_supported";
const std::string FUNC_OP_SELECT_FORMAT = "op_select_format";
const std::string FUNC_GET_OP_SUPPORT_INFO = "get_op_support_info";
const std::string FUNC_GENERATE_OP_UNIQUE_KEYS = "generate_op_unique_keys";
const std::string OPT_INPUT_MODE = "optionalInputMode";
const std::string DYN_PARAM_MODE = "dynamicParamMode";
const std::string GEN_PLACEHOLDER = "gen_placeholder";
const std::string NO_PLACEHOLDER = "no_placeholder";

const std::string OPT_MODULE_OP_TUNE = "opt_module.op_tune";
const std::string OPT_MODULE_RL_TUNE = "opt_module.rl_tune";
const std::string OPT_MODULE_PASS = "opt_module.pass";
const std::string OPT_SWITCH_OP_TUNE = "enable_ga_bank_query";
const std::string OPT_KEY_LIST_OP_TUNE = "query_ga_bank_for_key_list";
const std::string OPT_SWITCH_RL_TUNE = "enable_rl_bank_query";
const std::string OPT_KEY_LIST_RL_TUNE = "query_rl_bank_for_key_list";
const std::string SWITCH_ON = "True";
const std::string SWITCH_OFF = "False";
const std::string COMPILER_PROCESS_DIED = "compiler process died";

// options path key
constexpr const char *OM_BUILTIN_BINARY_KEY = "om.binary.builtin";
constexpr const char *OP_BUILTIN_BINARY_KEY = "op.binary.builtin";
constexpr const char *OM_CUSTOM_BINARY_KEY = "om.binary.custom";
constexpr const char *OP_CUSTOM_BINARY_KEY = "op.binary.custom";
constexpr const char *OPP_LATEST_PATH = "/opp_latest/";

// minimum value for float and double
const float FLOAT_ESP = 1e-6f;
const double DOUBLE_ESP = 1e-15;
const int SERIALIZATION_JSON_FORMAT = 4;
const uint64_t INVALID_SGT_INDEX = 0xFFFFFFFF;
constexpr int TIMESTAMP_LEN = 100;
const int64_t RANGE_UNLIMITED_LOW_ZERO = 0;
const int64_t RANGE_UNLIMITED_LOW_ONE = 1;
const int64_t RANGE_UNLIMITED_UP = -1;
const int64_t DYNAMIC_SHAPE_DIM = -2;
const int64_t UNKNOWN_DIM = -1;
const double TIME_INTERVAL = 2.999999;    // interval 3s to record log for suppressing massive Logs
const double PRINT_INTERVAL = 9.999999;    // print to screen interval, 10s

// fusion check result initial value
const int CHECK_RES_INITIAL_VALUE = -2;
constexpr int OP_TASK_TYPE_FUSION = 2;

// cache cfg
const int64_t CACHE_AGING_FUCNTION_SWITCH = -1;
const int64_t OP_CACHE_CFG_DEFAULT_VALUE = INT64_MIN;

// file authority
const int FILE_AUTHORITY = 0640;

// ge format
constexpr const char *FORMAT_AGNOSTIC = "formatAgnostic";
constexpr const char *FORMAT_ND = "ND";

// op debug config
constexpr const char *kDumpUbFusion = "dump_UBfusion";
constexpr const char *kDumpBin = "dump_bin";
constexpr const char *kDumpCce = "dump_cce";

// ge OppVersion enum
const int64_t OPP_LATEST_ENUM = 1;
constexpr const char *OPP_PATH = "_opp_path";
const int64_t AICORE_IMPL_TYPE_NUM = 6;

const std::vector<std::pair<int64_t, int64_t>> DIM_N = {
  std::make_pair(1, 1), std::make_pair(2, 3), std::make_pair(4, 7),
  std::make_pair(8, 15), std::make_pair(16, 31), std::make_pair(32, -1)
};
const int64_t DIM_N_MAX = 32;

const std::vector<std::pair<int64_t, int64_t>> DIM_DHW = {
  std::make_pair(1, 3), std::make_pair(4, 15), std::make_pair(16, 31),
  std::make_pair(32, 63), std::make_pair(64, 127), std::make_pair(128, 191),
  std::make_pair(192, 255), std::make_pair(256, 511), std::make_pair(512, 767),
  std::make_pair(768, 1023), std::make_pair(1024, -1)
};
const int64_t DIM_DHW_MAX = 1024;

const std::vector<int64_t> unknownRankShape = {-2};

constexpr const char *KEY_NAN = "nan";
constexpr const char *KEY_INF = "inf";
constexpr const char *KEY_NEGTIVE_INF = "-inf";
constexpr const char *KEY_NULL = "null";

const std::map<std::string, uint32_t> kDataTypeStrToLength = {
    {"uint1", 1},
    {"bool", 1},
    {"int8", 8},
    {"uint8", 8},
};

const std::map<std::string, int64_t> kBoolStrMap {
    {"true", static_cast<int64_t>(true)},
    {"false", static_cast<int64_t>(false)}
};

const std::map<std::string, int64_t> kSwitchStrMap {
    {"1", static_cast<int64_t>(true)},
    {"0", static_cast<int64_t>(false)}
};

const std::string COMPILE_CACHE_MODE_ENABLE = "enable";
const std::string COMPILE_CACHE_MODE_DISABLE = "disable";
const std::string COMPILE_CACHE_MODE_FORCE = "force";
const std::map<std::string, int64_t> kCompileCacheModeStrMap {
    {COMPILE_CACHE_MODE_DISABLE, static_cast<int64_t>(CompileCacheMode::Disable)},
    {COMPILE_CACHE_MODE_ENABLE, static_cast<int64_t>(CompileCacheMode::Enable)},
    {COMPILE_CACHE_MODE_FORCE, static_cast<int64_t>(CompileCacheMode::Force)}
};

const std::map<std::string, int64_t> kOpDebugLevelStrMap {
    {"0", static_cast<int64_t>(OpDebugLevel::Disable)},
    {"1", static_cast<int64_t>(OpDebugLevel::Level1)},
    {"2", static_cast<int64_t>(OpDebugLevel::Level2)},
    {"3", static_cast<int64_t>(OpDebugLevel::Level3)},
    {"4", static_cast<int64_t>(OpDebugLevel::Level4)}
};
} // namespace fusion
} // namespace te
#endif  // TE_FUSION_UTIL_CONSTANTS_H_
