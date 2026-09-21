/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/configuration.h"
#include <algorithm>
#include <sstream>
#include <iosfwd>
#include "common/comm_error_codes.h"
#include "common/fe_log.h"
#include "common/constants_define.h"
#include "common/fe_error_code.h"
#include "common/fe_type_utils.h"
#include "common/string_utils.h"
#include "common/platform_utils.h"
#include "common/fe_context_utils.h"
#include "common/fe_report_error.h"
#include "common/aicore_util_constants.h"
#include "register/optimization_option_registry.h"
#include "common/config_parser/op_cust_dtypes_config_parser.h"
#include "common/config_parser/op_impl_mode_config_parser.h"
#include "common/config_parser/modify_mixlist_config_parser.h"
#include "common/config_parser/op_debug_config_parser.h"
#include "mmpa/mmpa_api.h"
#include "ge_common/ge_api_types.h"
#include "graph/ge_context.h"
#include "graph/tuning_utils.h"
#include "graph/ge_local_context.h"
#include "framework/common/ge_types.h"
#include "register/register_base.h"
namespace fe {
namespace {
enum OpStoreInfoIndex {
    IDX_PRIORITY = 0,     // priority index
    IDX_OPIMPL_TYPE,      // op_impl_type index
    IDX_CFG_FILEPATH,     // cfg_file_path index
    IDX_OPIMPL_FILEPATH,  // op_impl_file_path index
    IDX_NEED_PRECOMPILE,  // need precompile
    IDX_NEED_COMPILE,     // need compile
    ID_IS_FULL_PATH,      // ascend_custom_opp_path
    IDX_BOTTOM
};

std::map<ge::OoLevel, std::string> opt_values = {
    {ge::OoLevel::kO0, kStrFalse},
    {ge::OoLevel::kO1, kStrFalse},
    {ge::OoLevel::kO2, kStrTrue},
    {ge::OoLevel::kO3, kStrTrue}
};
REG_OPTION(kComLevelO1Opt)
    .LEVELS(ge::OoLevel::kO0)
    .DEFAULT_VALUES(opt_values);

std::map<ge::OoLevel, std::string> o3_opt_values = {
    {ge::OoLevel::kO0, kStrFalse},
    {ge::OoLevel::kO1, kStrFalse},
    {ge::OoLevel::kO2, kStrFalse},
    {ge::OoLevel::kO3, kStrTrue}
};
REG_OPTION(kComLevelO3Opt)
.LEVELS(ge::OoLevel::kO0)
.DEFAULT_VALUES(o3_opt_values);

const size_t kHardwareInfoLen = 2;
const float kDefaultCompressRatioThreshold = 0.8f;
constexpr char const *kShortSocVersionAscend035 = "Ascend035";
constexpr char const *OP_STOREINFO_PREFIX = "op.store.";
constexpr char const *NEED_PRECOMPILE_TRUE = "true";
constexpr char const *NEED_COMPILE_TRUE = "true";
constexpr char const *kOpBunitinBinaryKey = "op.binary.builtin";
constexpr char const *kOpCustomBinaryKey = "op.binary.custom";
constexpr char const *kOmCustomBinaryKey = "om.binary.custom";
constexpr char const *kTbeCustomStoreKey = "op.store.tbe-custom";
constexpr char const *kEnableFirstQuantKey = "enable_first_layer_quantization";
constexpr char const *kOldBuiltInPassPath = "fusion_pass/built_in";
constexpr char const *kFp16OpType = "fp16.op_type";
constexpr char const *kOppConfigFile = "vendors/config.ini";
constexpr char const *CONFIG_FILE_RELATIVE_PATH = "plugin/opskernel/fe_config/fe.ini";
constexpr char const *CONFIG_OPS_RELATIVE_PATH = "/opp/";
constexpr char const *CONFIG_OPS_RELATIVE_PATH_BACK = "../../opp/";
constexpr char const *CONFIG_LIB64_RELATIVE_PATH = "/lib64/";
constexpr char const *CONFIG_OPP_LATEST_PATH = "/opp_latest/";
constexpr char const *kStrOpCompile = "op_compile";
constexpr char const *kStrUbFusion = "ub_fusion";
constexpr char const *kQuantBiasOptimze = "ge.experiment.quant_bias_optimize";
constexpr char const *kExportCompileStat = "ge.exportCompileStat";
constexpr char const *kCompileTaskTraceTimeInterval = "compile_task_trace.time_interval";
constexpr char const *kCompileTaskTraceTimeConstThreshold = "compile_task_trace.time_interval";
const uint64_t kDefaultTraceTimeInterval = 300;
const uint64_t kDefaultTraceTimeConstThreshold = 300;
const std::map<int64_t, string> kCustomPathMap{
        {EN_IMPL_CUSTOM_TBE, "ai_core"},
        {EN_IMPL_VECTOR_CORE_CUSTOM_TBE, "vector_core"}
};

constexpr char const *ASCEND_OPP_PATH = "ASCEND_OPP_PATH";
const std::map<ENV_STR_PARAM, std::tuple<mmEnvId, std::string>> kEnvParamMap {
        {ENV_STR_PARAM::AscendOppPath, {MM_ENV_ASCEND_OPP_PATH, "ASCEND_OPP_PATH"}},
        {ENV_STR_PARAM::NetworkAnalysis, {MM_ENV_ENABLE_NETWORK_ANALYSIS_DEBUG, "ENABLE_NETWORK_ANALYSIS_DEBUG"}},
        {ENV_STR_PARAM::DynamicImplFirst, {MM_ENV_OP_DYNAMIC_COMPILE_STATIC, "OP_DYNAMIC_COMPILE_STATIC"}},
        {ENV_STR_PARAM::AscendCustomOppPath, {MM_ENV_ASCEND_CUSTOM_OPP_PATH, "ASCEND_CUSTOM_OPP_PATH"}},
        {ENV_STR_PARAM::DumpGeGraph, {MM_ENV_DUMP_GE_GRAPH, "DUMP_GE_GRAPH"}},
        {ENV_STR_PARAM::DumpGraphLevel, {MM_ENV_DUMP_GRAPH_LEVEL, "DUMP_GRAPH_LEVEL"}},
        {ENV_STR_PARAM::EnableAclnn, {MM_ENV_ENABLE_ACLNN, "ENABLE_ACLNN"}},
        {ENV_STR_PARAM::NpuCollectPath, {MM_ENV_NPU_COLLECT_PATH, "NPU_COLLECT_PATH"}},
        {ENV_STR_PARAM::AscendWorkPath, {MM_ENV_ASCEND_WORK_PATH, "ASCEND_WORK_PATH"}},
        {ENV_STR_PARAM::MinCompileResourceUsageCtrl,
          {MM_ENV_MIN_COMPILE_RESOURCE_USAGE_CTRL, "MIN_COMPILE_RESOURCE_USAGE_CTRL"}},
        {ENV_STR_PARAM::EnableRt2, {MM_ENV_ENABLE_RUNTIME_V2, "ENABLE_RUNTIME_V2"}},
        {ENV_STR_PARAM::AscendHomePath, {MM_ENV_ASCEND_HOME_PATH, "ASCEND_HOME_PATH"}}
};

const std::map<string, int64_t> kSwitchMap {{"1", 1}, {"0", 0}};
const std::map<string, int64_t> kJitCompileMap {{"1", 1}, {"0", 0}, {"2", 2}};
const std::map<string, int64_t> kDisableReuseMemoryMap {{"1", 0}, {"0", 1}};

const std::map<std::string, int64_t> kFormatModeConfigMap {
  {"0", static_cast<int64_t>(FormatModeType::FORMAT_MODE_NZNZ)},
  {"1", static_cast<int64_t>(FormatModeType::FORMAT_MODE_NDND)},
  {"2", static_cast<int64_t>(FormatModeType::FORMAT_MODE_NDNZ)},
};

const std::map<string, int64_t> kBufferOptimizeMap {
        {OFF_OPTIMIZE, static_cast<int64_t>(EN_OFF_OPTIMIZE)},
        {L1_OPTIMIZE, static_cast<int64_t>(EN_L1_OPTIMIZE)},
        {L2_OPTIMIZE, static_cast<int64_t>(EN_L2_OPTIMIZE)}
};

const std::map<string, int64_t> kExportCompileStatMap {
        {"0", static_cast<int64_t>(ExportCompileStatType::NONE)},
        {"1", static_cast<int64_t>(ExportCompileStatType::AFTER_EXEC_COMPLITE)},
        {"2", static_cast<int64_t>(ExportCompileStatType::AFTER_COMPILE_COMPLITE)}
};

const std::map<CONFIG_PARAM, std::tuple<int64_t, string, std::map<string, int64_t>>> kConfigParamMap {
    {CONFIG_PARAM::SmallChannel, {0, ge::ENABLE_SMALL_CHANNEL, kSwitchMap}},
    {CONFIG_PARAM::JitCompile, {1, ge::JIT_COMPILE, kJitCompileMap}},
    {CONFIG_PARAM::VirtualType, {0, ge::VIRTUAL_TYPE, kSwitchMap}},
    {CONFIG_PARAM::CompressWeight, {0, ge::ENABLE_COMPRESS_WEIGHT, kSwitchMap}},
    {CONFIG_PARAM::SparseMatrixWeight, {0, ge::ENABLE_SPARSE_MATRIX_WEIGHT, kSwitchMap}},
    {CONFIG_PARAM::ReuseMemory, {1, ge::OPTION_EXEC_DISABLE_REUSED_MEMORY, kDisableReuseMemoryMap}},
    {CONFIG_PARAM::BufferOptimize, {static_cast<int64_t>(EN_UNKNOWN_OPTIMIZE),
                                    ge::BUFFER_OPTIMIZE, kBufferOptimizeMap}},
    {CONFIG_PARAM::FormatMode, {0, ge::OPTION_EXEC_FORMAT_MODEL, kFormatModeConfigMap}},
    {CONFIG_PARAM::QuantDumpable, {0, ge::QUANT_DUMPABLE, kSwitchMap}},
    {CONFIG_PARAM::QuantBiasOptimize, {1, kQuantBiasOptimze, kSwitchMap}},
    {CONFIG_PARAM::ExportCompileStat, {1, kExportCompileStat, kExportCompileStatMap}}
};

constexpr char const *kHardwareInfo = "ge.hardwareInfo";
constexpr char const *kFusionLicense = "opt_module.fe";

const std::map<CONFIG_STR_PARAM, string> kConfigStrParamMap {
    {CONFIG_STR_PARAM::HardwareInfo, kHardwareInfo},
    {CONFIG_STR_PARAM::FusionLicense, kFusionLicense}
};

const std::vector<std::string> kImplModeParamVec = {
  ge::OP_PRECISION_MODE,
  ge::OP_SELECT_IMPL_MODE,
  ge::OPTYPELIST_FOR_IMPLMODE,
  ge::ALLOW_HF32
};

const std::map<CONFIG_PARSER_PARAM, std::vector<std::string>> kConfigParserParamMap {
      {CONFIG_PARSER_PARAM::ImplMode, kImplModeParamVec},
      {CONFIG_PARSER_PARAM::CustDtypes, {ge::CUSTOMIZE_DTYPES}},
      {CONFIG_PARSER_PARAM::ModifyMixlist, {ge::MODIFY_MIXLIST}}
};

const std::vector<string> kForceClosePassVec = {"MatMulBiasAddFusionPass", "AutomaticUbFusion",
                                                "ReshapeTransposeFusionPass", "SquareSumV1",
                                                "TbeEltwiseCastFusionPass"};

const std::map<std::string, std::string> kLicensePassNameMap {
        {"3", "MatMulBiasAddFusionPass"},
        {"4", "OneHotFusionPass"},
        {"6", "MulAddNL2LossFusionPass"},
        {"7", "AutomaticUbFusion"},
        {"9", "TransposeReshapeFusionPass"},
        {"17", "TbeConv2DBackpropElemwiseFusionPass"},
        {"18", "TbeConvBnreduceFusionPass"},
        {"19", "BatchMatmulFusionPass"},
        {"20", "ConstToAttrStridedSliceFusion"},
        {"21", "ExtremumGradFusionPass"},
        {"22", "LayerNormGradFusionPass"},
        {"23", "LayerNormGradFusionPassBetaGammaV2"},
        {"25", "MatmulCastFusionPass"},
        {"26", "ReshapeTransposeFusionPass"},
        {"27", "SquareSumV1"},
        {"28", "StridedSliceGradFusionPass"},
        {"29", "ZUnsortedSegmentSumUpdateFusionPass"},
        {"30", "ATbeMatmulElemwiseFusionPass"},
        {"31", "BatchMatmulConfusiontransposeUbFusion"},
        {"32", "MatmulConfusiontransposeUbFusion"},
        {"33", "TbeBatchMatmulFusedMulAddFusionPass"},
        {"34", "TbeEltwiseFusionPass"},
        {"35", "TbeFullyconnectionElemwiseDequantFusionPass"},
        {"36", "TbeMultiOutputFusionPass"},
        {"37", "MulAddFusionPass"},
        {"39", "clip_by_norm_nodivsquaresum"},
        {"40", "PadConv2dFusionPass"},
        {"41", "SparseSoftMaxFusionPass"},
        {"42", "MulAddNPass"},
        {"43", "Resnet50DbnDwFusionPass"},
        {"44", "BatchMatmulDropOutDoMaskV3DFusionPass"},
        {"45", "MatmulConfusiontransposeUbFusion"},
        {"46", "MatmulDropOutDoMaskV3DFusionPass"},
        {"47", "TbeBatchMatmulElementWiseFusionPass"},
        {"48", "SoftmaxWithDropOutDoMaskFusion"}
};

bool ComparePriority(const FEOpsStoreInfo &left, const FEOpsStoreInfo &right) { return left.priority < right.priority; }
}

Configuration::Configuration(const string &engine_name)
    : is_init_(false),
      engine_name_(engine_name),
      is_dynamic_impl_first_(true),
      enable_network_analysis_(false),
      enable_op_impl_strategy_(true),
      enable_ub_fusion_(true),
      enable_aclnn_(false),
      enable_first_layer_quantization_(false),
      op_store_priority_count_(0),
      mem_reuse_dist_threshold_(MEM_REUSE_DIST_THRESHOLD),
      ai_core_compress_ratio_(0),
      cust_dtypes_parser_(nullptr),
      impl_mode_parser_(nullptr),
      mix_list_parser_(nullptr),
      op_debug_config_parse_(nullptr) {}

Configuration::~Configuration() {}

Configuration &Configuration::Instance(const string &engine_name) {
  static std::map<string, Configuration &> config_map;
  if (config_map.empty()) {
    static Configuration ai_core_config(AI_CORE_NAME);
    static Configuration vector_core_config(VECTOR_CORE_NAME);
    static Configuration dsa_core_config(kDsaCoreName);
    config_map.insert({AI_CORE_NAME, ai_core_config});
    config_map.insert({VECTOR_CORE_NAME, vector_core_config});
    config_map.insert({kDsaCoreName, dsa_core_config});
  }
  std::map<string, Configuration &>::const_iterator iter = config_map.find(engine_name);
  if (iter != config_map.cend()) {
    return iter->second;
  }
  FE_LOGD("Engine name [%s] is not found, using the default instance.", engine_name.c_str());
  /* If engine_name is invalid, we just return the first element of map
   * config_map */
  return config_map.begin()->second;
}

Status Configuration::Initialize(const std::map<string, string> &options) {
  if (is_init_) {
    return SUCCESS;
  }
  FE_LOGD("Begin to initialize Configuration.");

  content_map_.clear();
  op_binary_path_map_.clear();
  op_store_priority_count_ = 0;

  // initialize environment variables
  InitParamFromEnv();

  // initialize parameters based on the options from ge
  Status status = InitConfigParamFromOptions(options);
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init] Failed to initialize parameters based on options from ge.");
    return FAILED;
  }

  status = InitLibPath();
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init] Failed to initialize the real path of libfe.");
    return FAILED;
  }

  status = InitAscendOpsPath();
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init] Failed to load ascend ops path.");
    return status;
  }

  InitCustomOpStore();
  status = LoadOppConfigFile();
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init] Failed to load custom config file.");
    return status;
  }

  status = LoadConfigFile();
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init] Failed to load the data from configuration file.");
    return status;
  }

  InitBinaryConfigFilePath();

  status = AssembleOpsStoreInfoVector();
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init] Failed to build OpsStoreInfo Vector.");
    return status;
  }

  // init other parameters based on soc version and items of ini file
  InitParametersOfConfigFile();
  return InitializeConfigParser(options);
}

Status Configuration::InitializeConfigParser(const std::map<string, string> &options) {
  FE_MAKE_SHARED(op_debug_config_parse_ = std::make_shared<OpDebugConfigParser>(
                     env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::NpuCollectPath)]),
                 return FAILED);
  FE_CHECK_NOTNULL(op_debug_config_parse_);
  Status op_debug_config_status = op_debug_config_parse_->InitializeFromOptions(options);
  if (op_debug_config_status != SUCCESS) {
    FE_LOGD("[OpdebugConfig][Init] Init unsuccessful.");
    return op_debug_config_status;
  }
  OpDebugConfigParserPtr op_debug_config_parser = std::dynamic_pointer_cast<OpDebugConfigParser>
                                                  (op_debug_config_parse_);
  op_debug_config_parser->SetOpDebugConfigEnv(env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::NpuCollectPath)]);

  Status status = InitializeExtend(options);
  if (status != SUCCESS) {
    return status;
  }

  FE_MAKE_SHARED(impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(ascend_ops_path_),
                 return FAILED);
  FE_CHECK_NOTNULL(impl_mode_parser_);
  Status op_precision_mode_status = impl_mode_parser_->InitializeFromOptions(options);
  if (op_precision_mode_status != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init] Failed to init op precision mode.");
    return op_precision_mode_status;
  }

  FE_MAKE_SHARED(cust_dtypes_parser_ = std::make_shared<OpCustDtypesConfigParser>(), return FAILED);
  FE_CHECK_NOTNULL(cust_dtypes_parser_);
  Status cust_dtypes_status = cust_dtypes_parser_->InitializeFromOptions(options);
  if (cust_dtypes_status != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init] Failed to init cust dtypes.");
    return cust_dtypes_status;
  }
  std::string combined_params_key = CombinedParamsKeyFromOptions(CONFIG_PARSER_PARAM::CustDtypes, options);
  config_parser_map_vec_[static_cast<size_t>(CONFIG_PARSER_PARAM::CustDtypes)]
      .emplace(combined_params_key, cust_dtypes_parser_);

  FE_MAKE_SHARED(mix_list_parser_ = std::make_shared<ModifyMixlistConfigParser>(), return FAILED);
  FE_CHECK_NOTNULL(mix_list_parser_);
  Status mix_list_status = mix_list_parser_->InitializeFromOptions(options);
  if (mix_list_status != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init] Failed to init mix list.");
    return mix_list_status;
  }
  combined_params_key = CombinedParamsKeyFromOptions(CONFIG_PARSER_PARAM::ModifyMixlist, options);
  config_parser_map_vec_[static_cast<size_t>(CONFIG_PARSER_PARAM::ModifyMixlist)]
      .emplace(combined_params_key, mix_list_parser_);

  is_init_ = true;
  FE_LOGI("Initialization of Configuration end.");
  return SUCCESS;
}

void Configuration::InitParamFromEnv() {
  for (const std::pair<const ENV_STR_PARAM, std::tuple<mmEnvId, std::string>> &item : kEnvParamMap) {
    const char_t *env_value = nullptr;
    MM_SYS_GET_ENV(std::get<0>(item.second), env_value);
    if ((env_value != nullptr) && (env_value[0U] != '\0')) {
      FE_LOGD("The value of env[%s] is [%s].", (std::get<1>(item.second)).c_str(), env_value);
      string env_str_value = string(env_value);
      (void)StringUtils::Trim(env_str_value);
      env_str_param_vec_[static_cast<size_t>(item.first)] = env_str_value;
    } else {
      FE_LOGD("Env[%s] is not found.", (std::get<1>(item.second)).c_str());
    }
  }

  // parse NetworkAnalysis
  if (!env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::NetworkAnalysis)].empty()) {
    enable_network_analysis_ = atoi(env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::NetworkAnalysis)].c_str());
  }
  FE_LOGD("The enable_network_analysis is [%d].", enable_network_analysis_);

  // parse DynamicImplFirst
  is_dynamic_impl_first_ = true;
  string dynamic_impl_first_str = env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::DynamicImplFirst)];
  if (!dynamic_impl_first_str.empty()) {
    (void)StringUtils::Trim(dynamic_impl_first_str);
    StringUtils::ToLowerString(dynamic_impl_first_str);
    is_dynamic_impl_first_ = dynamic_impl_first_str != kStrFalse;
  }
  FE_LOGD("Is dynamic implementation first: [%d].", is_dynamic_impl_first_);

  // parse MinCompileResourceUsageCtrl
  std::string &op_compile_func_ctrl =
          env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::MinCompileResourceUsageCtrl)];
  if (!op_compile_func_ctrl.empty()) {
    auto params = StringUtils::Split(op_compile_func_ctrl, ',');
    auto iter = std::find(params.begin(), params.end(), kStrOpCompile);
    enable_op_impl_strategy_ = (iter == params.end());
    iter = std::find(params.begin(), params.end(), kStrUbFusion);
    enable_ub_fusion_ = (iter == params.end());
  }
  FE_LOGD("Enable op_impl strategy is [%d], enable ub_fusion is [%d].", enable_op_impl_strategy_, enable_ub_fusion_);

  // parse EnableAclnn
  enable_aclnn_ = false;
  string enable_aclnn_str = env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::EnableAclnn)];
  if (!enable_aclnn_str.empty()) {
    (void)StringUtils::Trim(enable_aclnn_str);
    StringUtils::ToLowerString(enable_aclnn_str);
    enable_aclnn_ = (enable_aclnn_str == kStrTrue);
  }
  FE_LOGD("The value of the enable_aclnn flag is [%d].", enable_aclnn_);

  std::string rt2_str = env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::EnableRt2)];
  enable_rt2_ = rt2_str == "0" ? false : true;
}

const std::string& Configuration::GetDumpGeGraph() const {
  return env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::DumpGeGraph)];
}

const std::string& Configuration::GetDumpGraphLevel() const {
  return env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::DumpGraphLevel)];
}

const std::string& Configuration::GetAscendWorkPath() const {
  return env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::AscendWorkPath)];
}

Status Configuration::InitConfigParamFromOptions(const std::map<string, string> &options) {
  for (const auto &item : kConfigParamMap) {
    config_param_vec_[static_cast<size_t>(item.first)] = std::get<0>(item.second);
    string param_key = std::get<1>(item.second);
    const auto iter = options.find(param_key);
    if (iter == options.cend() || iter->second.empty()) {
      FE_LOGD("The value for [%s] is either not found or it is empty.", param_key.c_str());
      continue;
    }
    std::map<string, int64_t> param_str_map = std::get<2>(item.second);
    const auto iter_value = param_str_map.find(iter->second);
    if (iter_value != param_str_map.cend()) {
      config_param_vec_[static_cast<size_t>(item.first)] = iter_value->second;
      FE_LOGD("The value of param[%s] is [%s].", param_key.c_str(), iter->second.c_str());
    }
  }
  for (const auto &item : kConfigStrParamMap) {
    const auto iter = options.find(item.second);
    if (iter != options.cend() && !iter->second.empty()) {
      config_str_param_vec_[static_cast<size_t>(item.first)] = iter->second;
      FE_LOGD("The value of param[%s] is [%s].", item.second.c_str(), iter->second.c_str());
    }
  }

  ParseHardwareInfo();
  bool is_config = options.find(kFusionLicense) != options.end();
  ParseFusionLicense(is_config);
  if (ParseVirtualType() != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status Configuration::InitConfigParamFromContext() {
  std::lock_guard<std::mutex> lock_guard(config_param_mutex_);
  for (const auto &item : kConfigParamMap) {
    string param_key = std::get<1>(item.second);
    string param_value;
    (void)ge::GetContext().GetOption(param_key, param_value);
    if (param_value.empty()) {
      FE_LOGD("The value for [%s] is either not found or it is empty.", param_key.c_str());
      continue;
    }
    std::map<string, int64_t> param_str_map = std::get<2>(item.second);
    const auto iter_value = param_str_map.find(param_value);
    if (iter_value == param_str_map.cend()) {
      std::string reason = "The Valid value is " + fe::StringUtils::MapKeyToString(param_str_map);
      ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID, {param_value, param_key, reason});
      ReportErrorMessage(err_msg);
      FE_LOGE("The value[%s] of param[%s] is invalid.", param_value.c_str(), param_key.c_str());
      return FAILED;
    }
    config_param_vec_[static_cast<size_t>(item.first)] = iter_value->second;
  }
  for (const auto &item : kConfigStrParamMap) {
    string param_value;
    ge::graphStatus ret = ge::GetContext().GetOption(item.second, param_value);
    if (ret == ge::GRAPH_SUCCESS) {
      config_str_param_vec_[static_cast<size_t>(item.first)] = param_value;
      FE_LOGD("The value of param[%s] is [%s].", item.second.c_str(), param_value.c_str());
    }
  }

  ParseHardwareInfo();
  string fusion_license;
  bool is_config = ge::GetContext().GetOption(kFusionLicense, fusion_license) == ge::GRAPH_SUCCESS;
  ParseFusionLicense(is_config);
  if (ParseVirtualType() != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

void Configuration::ParseHardwareInfo() {
  string hardware_info_str = config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::HardwareInfo)];
  if (hardware_info_str.empty()) {
    return;
  }
  std::map<string, string> hardware_info_map_tmp;
  std::vector<string> hardware_infos = StringUtils::Split(hardware_info_str, ';');
  for (const auto &hardware_info : hardware_infos) {
    std::vector<string> hardware_size = StringUtils::Split(hardware_info, ':');
    if (hardware_size.size() != kHardwareInfoLen) {
      continue;
    }
    if (hardware_size[1].empty() || hardware_size[1] == "0") {
      continue;
    }
    hardware_info_map_tmp.insert(std::pair<string, string>(hardware_size[0], hardware_size[1]));
  }
  std::lock_guard<std::mutex> lock_guard(hardware_info_map_mutex_);
  hardware_info_map_ = std::move(hardware_info_map_tmp);
}

void Configuration::ParseFusionLicense(const bool &is_config) {
  std::string fusion_license_str = config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)];
  if (!is_config && fusion_license_str.empty()) {
    fusion_license_str = "ALL";
    config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = fusion_license_str;
  }
  license_fusion_detail_value_.clear();
  std::vector<string> license_vec = StringUtils::Split(StringUtils::Trim(fusion_license_str), ':');
  for (size_t i = 0; i < license_vec.size(); ++i) {
    auto iter = kLicensePassNameMap.find(license_vec[i].c_str());
    if (iter != kLicensePassNameMap.end() && !iter->second.empty()) {
      FE_LOGD("ParseFusionLicense %s-%s.", iter->first.c_str(), iter->second.c_str());
      license_fusion_detail_value_.insert(iter->second.c_str());
    }
  }
}

Status Configuration::ParseVirtualType() {
  FE_LOGD("ParseVirtualType begin.");
  const std::vector<uint32_t> &vir_type_list = PlatformUtils::Instance().GetVirTypeList();
  if (IsEnableVirtualType() && vir_type_list.empty()) {
    std::string reason_str = "The SoC does not support virtual_type, or it supports virtual_type but vir_type_list is not configured in file_path";
    ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID, {"1", ge::VIRTUAL_TYPE, reason_str});
    ReportErrorMessage(err_msg);
    FE_LOGE("The value at index [1] of parameter [%s] is invalid: %s.", ge::VIRTUAL_TYPE.c_str(), reason_str.c_str());
    return FAILED;
  }
  return SUCCESS;
}

void Configuration::InitParametersOfConfigFile() {
  InitMemReuseDistThreshold();

  InitCompressRatio();

  InitFp16OpType();
}

Status Configuration::Finalize() {
  FE_LOGD("Finalizing the Configuration.");
  if (!is_init_) {
    FE_LOGD("Configuration has not been initialized, Finalize is not allowed.");
    return SUCCESS;
  }

  content_map_.clear();
  op_binary_path_map_.clear();
  ops_store_info_vector_.clear();
  is_init_ = false;
  FE_LOGI("Configuration finalize successfully.");
  return SUCCESS;
}

Status Configuration::InitLibPath() {
  mmDlInfo dl_info;
  Configuration &(*instance_ptr)(const string &) = &Configuration::Instance;
  if (mmDladdr(reinterpret_cast<void *>(instance_ptr), &dl_info) != 0) {
    REPORT_FE_ERROR("[GraphOpt][Init][InitLibPath] Failed to get so file path.");
    return FAILED;
  } else {
    string so_path = dl_info.dli_fname;
    FE_LOGD("Libfe so file path is %s.", so_path.c_str());

    if (so_path.empty()) {
      REPORT_FE_ERROR("[GraphOpt][Init][InitLibPath] So file path is empty.");
      return FAILED;
    }

    lib_path_ = GetRealPath(so_path);
    int32_t pos = lib_path_.rfind('/');
    if (pos < 0) {
      REPORT_FE_ERROR("[GraphOpt][Init][InitLibPath] The current .so file path does not contain /.");
      return FAILED;
    }

    lib_path_ = lib_path_.substr(0, pos + 1);
  }
  FE_LOGD("The real path of libfe is %s.", lib_path_.c_str());
  return SUCCESS;
}

void Configuration::InitCustomOpStore() {
  const std::string custom_opp_path = aclGetCustomOpLibPath();
  if (custom_opp_path.empty()) {
    FE_LOGD("ascend custom opp path is empty.");
    return;
  }

  FE_LOGD("The path of ascend custom opp is %s.", custom_opp_path.c_str());
  std::vector<std::string> custom_opp_path_vec;
  StringUtils::SplitWithTrim(custom_opp_path, ':', custom_opp_path_vec);
  if (custom_opp_path_vec.empty()) {
    FE_LOGD("Custom opp path is empty.");
    return;
  }

  for (const auto &it : kCustomPathMap) {
    for (const std::string &custom_path : custom_opp_path_vec) {
      if (!AddCustomOpStoreContent(custom_path, it.second, it.first, true)) {
        FE_LOGD("Resolve op impl path(%s) unsuccess, do next.", custom_path.c_str());
        continue;
      }
      op_store_priority_count_++;
    }
  }
}

Status Configuration::InitAscendOpsPath() {
  const string &ascend_ops_path = env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::AscendOppPath)];
  if (!ascend_ops_path.empty()) {
    ascend_ops_path_ = GetRealPath(ascend_ops_path);
  } else {
    const char_t *env_value = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, env_value);
    if (env_value == nullptr) {
      ascend_ops_path_ = GetRealPath(lib_path_ + CONFIG_OPS_RELATIVE_PATH_BACK);
    } else {
      std::string path_env = string(env_value);
      ascend_ops_path_ = GetRealPath(path_env + CONFIG_OPS_RELATIVE_PATH);
    }
  }

  if (ascend_ops_path_.empty()) {
    ErrorMessageDetail err_msg(EM_ENVIRONMENT_VARIABLE_FAILED,
                               {ascend_ops_path, ASCEND_OPP_PATH,
                                "ASCEND_OPP_PATH does not exist or access permission is denied during FE initialization"});
    ReportErrorMessage(err_msg);
    FE_LOGE("[GraphOpt][Init][InitOpsPath] Ascend opp path is invalid.");
    return INVALID_FILE_PATH;
  }

  ascend_ops_path_ = ascend_ops_path_ + "/";
  FE_LOGD("The path of Ascend opp is %s.", ascend_ops_path_.c_str());
  return SUCCESS;
}

Status Configuration::GetOppLatestPath(std::string &opp_path) const {
  FE_LOGD("Enter the get opp latest path function.");
  const char_t *env_value = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, env_value);
  if (env_value == nullptr) {
    FE_LOGI("Env: MM_ENV_ASCEND_HOME_PATH is not found.");
    return FAILED;
  }
  std::string path_env = string(env_value);
  (void)StringUtils::Trim(path_env);
  if (path_env.empty()) {
    FE_LOGI("Ascend home path is empty.");
    return FAILED;
  }
  opp_path = path_env + CONFIG_OPP_LATEST_PATH;
  opp_path = GetRealPath(opp_path);
  FE_LOGD("The path of opp_latest is %s.", opp_path.c_str());
  if (opp_path.empty()) {
    FE_LOGI("Opp latest real path[%s] is empty.", opp_path.c_str());
    return FAILED;
  }
  return SUCCESS;
}

bool Configuration::IsPathExistedInOpp(const std::string &path, bool is_full_path) const
{
  std::string real_path = path;
  if (!is_full_path) {
    real_path = ascend_ops_path_ + path;
  }
  real_path = GetRealPath(real_path);
  return !real_path.empty();
}

bool Configuration::CheckIsValidAbstractPath(const std::string &path) const {
  if (path.empty() || path[0] != '/') {
    return false;
  }
  return IsPathExistedInOpp(path, true);
}

// i|(impl_type|(i<<32))|vendors/xxx/op_impl/ai_core/tbe/config|vendors/xxx/op_impl/ai_core/tbe/xxx_impl|true|true
bool Configuration::AddCustomOpStoreContent(const std::string &full_or_sub_path, const std::string &path_type,
                                            const int64_t main_impl_type, const bool is_full_path) {
  FE_LOGD("Begin to add [%s] custom op store from path[%s], impl type[%ld], is_full_path[%d].",
          path_type.c_str(), full_or_sub_path.c_str(), main_impl_type, is_full_path);
  std::string impl_path;
  std::stringstream ss;
  if (is_full_path) {
    if (!CheckIsValidAbstractPath(full_or_sub_path)) {
      REPORT_FE_WARN("[Init][LoadAscendCustomOppPath] ASCEND_CUSTOM_OPP_PATH[%s] does not exist or is not an abstract path.",
                     full_or_sub_path.c_str());
      return false;
    }
    ss << full_or_sub_path;
    if (full_or_sub_path.back() != '/') {
      ss << "/";
    }
    ss << "op_impl/" << path_type;
    std::vector<std::string> dir_name;
    StringUtils::SplitWithTrim(full_or_sub_path, '/', dir_name);
    std::string &sub_path = dir_name.back();
    impl_path = ss.str()  + "/tbe/" + sub_path + "_impl";
  } else {
    ss << "vendors/" << full_or_sub_path << "/op_impl/" << path_type;
    impl_path = ss.str()  + "/tbe/" + full_or_sub_path + "_impl";
  }
  std::string cfg_path = ss.str() + "/tbe/config";
  if (!IsPathExistedInOpp(cfg_path, is_full_path) || !IsPathExistedInOpp(impl_path, is_full_path)) {
    return false;
  }

  int64_t impl_type = main_impl_type | (op_store_priority_count_ << 32);
  std::stringstream custom_op_store_ss;
  custom_op_store_ss << std::to_string(op_store_priority_count_) << "|" << std::to_string(impl_type);
  custom_op_store_ss << "|" << cfg_path << "|" << impl_path;
  custom_op_store_ss << "|true|true";
  if (is_full_path) {
    custom_op_store_ss << "|true";
  }

  std::string custom_op_store_value = custom_op_store_ss.str();
  std::string custom_op_store_key = kTbeCustomStoreKey;
  if (op_store_priority_count_ != 0) {
    custom_op_store_key += std::to_string(op_store_priority_count_);
  }
  custom_op_store_value = StringUtils::Trim(custom_op_store_value);
  if (content_map_.count(custom_op_store_key) != 0) {
    FE_LOGD("Op custom store [%s] already exists.", custom_op_store_key.c_str());
    return false;
  }
  content_map_.emplace(custom_op_store_key, custom_op_store_value);
  FE_LOGD("Insert ascend custom %s impl: key=%s, value=%s.",
          path_type.c_str(), custom_op_store_key.c_str(), custom_op_store_value.c_str());
  return true;
}

// "op.binary.tbe-custom = impl_type1|path1,impl_type1|path1"
void Configuration::ResolveBinaryPath(const string &sub_path, const string &path_type,
                                      const int64_t main_impl_type, bool isOm, const std::string &binaryKey) {
  std::stringstream ss;
  ss << "/vendors/";
  ss << sub_path;
  ss << "/op_impl/" << path_type << "/tbe/";
  string sub_file = isOm ?  "model/" : "kernel/";
  ss << sub_file;
  string file_path = ss.str();
  int64_t impl_type = main_impl_type | (op_store_priority_count_ << 32);
  std::string pri_str = std::to_string(op_store_priority_count_);
  if (!IsPathExistedInOpp(file_path, false)) {
    return;
  }
  ss << "config";
  if (!IsPathExistedInOpp(ss.str(), false)) {
    return;
  }
  ss.str("");
  ss << std::to_string(impl_type) << "|" << file_path;
  file_path = ss.str();
  file_path = StringUtils::Trim(file_path);
  auto it = op_binary_path_map_.find(binaryKey);
  if (it == op_binary_path_map_.end()) {
    FE_LOGD("Insert binary: key=%s, value=%s.", binaryKey.c_str(), file_path.c_str());
    op_binary_path_map_.emplace(binaryKey, file_path);
  } else {
    it->second = it->second + "," + file_path;
    FE_LOGD("Append binary: key=%s, value=%s.", binaryKey.c_str(), it->second.c_str());
  }
}

Status Configuration::LoadOppConfigFile() {
  std::vector<string> custom_opp_path_vec;
  Status status = GetCustomOppPathFromOppConfigFile(custom_opp_path_vec);
  if (status != SUCCESS) {
    return status;
  }
  if (custom_opp_path_vec.empty()) {
    return SUCCESS;
  }
  for (const auto &it : kCustomPathMap) {
    for (const std::string &custom_path : custom_opp_path_vec) {
      if (!AddCustomOpStoreContent(custom_path, it.second, it.first, false)) {
        FE_LOGD("Resolve op impl path(%s) unsuccess, do next.", custom_path.c_str());
        continue;
      }
      ResolveBinaryPath(custom_path, it.second, it.first, false, kOpCustomBinaryKey);
      ResolveBinaryPath(custom_path, it.second, it.first, true, kOmCustomBinaryKey);
      op_store_priority_count_++;
    }
  }
  FE_LOGD("End loading opp custom config.");
  return SUCCESS;
}

Status Configuration::GetCustomOppPathFromOppConfigFile(std::vector<string> &custom_opp_path_vec) const {
  std::string opp_config_file = ascend_ops_path_ + kOppConfigFile;
  FE_LOGD("Begin to load opp config file[%s].", opp_config_file.c_str());
  std::string real_opp_config_file_path = GetRealPath(opp_config_file);
  if (real_opp_config_file_path.empty()) {
    FE_LOGI("The opp config file path [%s] does not exist.", opp_config_file.c_str());
    return SUCCESS;
  }
  std::ifstream ifs(real_opp_config_file_path);
  if (!ifs.is_open()) {
    FE_LOGD("Custom config.ini does not exist.");
    return SUCCESS;
  }
  string line;
  while (std::getline(ifs, line)) {
    line = StringUtils::Trim(line);
    if (line.empty() || line.find('#') == 0) {
      continue;
    }
    if (line.find("load_priority") == string::npos) {
      continue;
    }
    size_t pos_of_equal = line.find('=');
    if (pos_of_equal == string::npos) {
      ifs.close();
      ErrorMessageDetail err_msg(EM_INVALID_CONTENT,
        {"load_priority", real_opp_config_file_path, "Line:\"" + line + "\" not contain \"=\"."});
      ReportErrorMessage(err_msg);
      REPORT_FE_ERROR("[GraphOpt][LoadOppConfigFile] Config format is error.");
      return INVALID_FILE_PATH;
    }
    string value = line.substr(pos_of_equal + 1);
    StringUtils::SplitWithTrim(value, ',', custom_opp_path_vec);
    FE_LOGD("Get number %zu custom paths.", custom_opp_path_vec.size());
    break;
  }
  ifs.close();
  return SUCCESS;
}

void Configuration::InitBinaryConfigFilePath() {
  const auto &iter = op_binary_path_map_.find(kOpBunitinBinaryKey);
  if (iter == op_binary_path_map_.end()) {
      FE_LOGD("Not found builtin binary key:op.binary.builtin in map.");
      return;
  }
  const size_t &pos = iter->second.find('|');
  if (pos == string::npos) {
      FE_LOGD("Config format is wrong.");
      return;
  }

  string kernel_path = ascend_ops_path_ + iter->second.substr(pos + 1);
  std::stringstream bin_cfg_file_path;
  bin_cfg_file_path << kernel_path;
  bin_cfg_file_path << "config/" << PlatformUtils::Instance().GetSocVersion() << "/op_info_config.json";
  bin_cfg_file_ = GetRealPath(bin_cfg_file_path.str());
  if (bin_cfg_file_.empty()) {
    FE_LOGD("[GraphOpt][InitOpsPath] Cannot find binary config file %s.", bin_cfg_file_path.str().c_str());
    return;
  }
  FE_LOGD("The real file path of Ascend opp kernel is [%s].", bin_cfg_file_.c_str());
}

const std::string& Configuration::GetBinaryConfigFilePath() const {
  FE_LOGD("The binary configuration file path is %s.", bin_cfg_file_.c_str());
  return bin_cfg_file_;
}

Status Configuration::LoadConfigFile() {
  FE_LOGD("Begin to LoadConfigFile.");
  string config_file_real_path = GetRealPath(lib_path_ + CONFIG_FILE_RELATIVE_PATH);
  FE_LOGI("The real path of fe.ini is %s.", config_file_real_path.c_str());

  if (config_file_real_path.empty()) {
    REPORT_FE_ERROR("[GraphOpt][InitOpsPath] The FE configuration file path is invalid.");
    return INVALID_FILE_PATH;
  }

  std::ifstream ifs(config_file_real_path);
  if (!ifs.is_open()) {
    REPORT_FE_ERROR("[GraphOpt][InitOpsPath] Open configuration file failed.");
    return INVALID_FILE_PATH;
  }
  string line;
  while (std::getline(ifs, line)) {
    if (line.empty()) {
      continue;
    }
    line = StringUtils::Trim(line);
    if (line.empty() || line.find('#') == 0) {
      continue;
    }

    size_t pos_of_equal = line.find('=');
    if (pos_of_equal == string::npos) {
      continue;
    }
    string key = line.substr(0, pos_of_equal);
    string value = line.substr(pos_of_equal + 1);
    key = StringUtils::Trim(key);
    value = StringUtils::Trim(value);
    if (!key.empty() && (content_map_.count(key) == 0)) {
      content_map_.emplace(key, value);
      FE_LOGD("Context map insert key:%s, value:%s.", key.c_str(), value.c_str());
      if (key.find("binary.builtin") != string::npos) {
        op_binary_path_map_.emplace(key, value);
      }
    }
  }
  ifs.close();
  FE_LOGD("End of loadConfigFile.");
  return SUCCESS;
}

bool Configuration::IsIgnoreOpStore(const FEOpsStoreInfo &ops_store_info) const {
  OpImplType main_impl = GetMainImplType<OpImplType>(ops_store_info.op_impl_type);
  bool is_vevtor_core_type = main_impl == EN_IMPL_VECTOR_CORE_HW_TBE ||
                             main_impl == EN_IMPL_VECTOR_CORE_CUSTOM_TBE;
  bool is_dsa_core_type = main_impl == EN_IMPL_HW_DSA;
  return (engine_name_ == AI_CORE_NAME && (is_vevtor_core_type || is_dsa_core_type)) ||
         (engine_name_ == VECTOR_CORE_NAME && !is_vevtor_core_type) ||
         (engine_name_ == kDsaCoreName && !is_dsa_core_type);
}

Status Configuration::AssembleOpsStoreInfoVector() {
  FE_LOGD("AssembleOpsStoreInfoVector of engine %s begin.", engine_name_.c_str());

  if (content_map_.empty()) {
    REPORT_FE_ERROR("[GraphOpt][Init][AssmOpsStoreInfoVec] There is no OP store item in fe.ini.");
    return OPSTORE_EMPTY;
  }
  ops_store_info_vector_.clear();

  std::map<string, string>::iterator iter;
  for (iter = content_map_.begin(); iter != content_map_.end(); ++iter) {
    if (iter->first.find(OP_STOREINFO_PREFIX) == string::npos) {
      continue;
    }

    // Format : op.storeinfo.name=priority|op_impl_type|cfg_file_path|op_impl_file_path
    string op_store_name = iter->first.substr(std::strlen(OP_STOREINFO_PREFIX));
    if (op_store_name.empty()) {
      REPORT_FE_ERROR("[GraphOpt][Init][AssmOpsStoreInfoVec] The name of the op store info config item is empty.");
      return OPSTORE_NAME_EMPTY;
    }
    if (iter->second.empty()) {
      REPORT_FE_ERROR("[GraphOpt][Init][AssmOpsStoreInfoVec] The value of [%s] in the fe.ini file is empty.",
                      op_store_name.c_str());
      return OPSTORE_VALUE_EMPTY;
    }

    std::vector<string> op_store_vector = StringUtils::Split(iter->second, '|');
    Status verify_status = VerifyOpStoreVector(op_store_vector, op_store_name);
    if (verify_status != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][Init][AssmOpsStoreInfoVec] Failed to verify op store [%s].", op_store_name.c_str());
      return verify_status;
    }

    FEOpsStoreInfo ops_store_info;
    Status status = AssembleEachOpsStoreInfo(op_store_name, op_store_vector, ops_store_info);
    if (status != SUCCESS) {
      return status;
    }
    if (!IsIgnoreOpStore(ops_store_info)) {
      ops_store_info_vector_.push_back(ops_store_info);
      FE_LOGI("OP store [%s] has been added to the ops_store_info_vector_ of %s.", ops_store_info.fe_ops_store_name.c_str(),
              engine_name_.c_str());
    }
  }

  if (ops_store_info_vector_.empty()) {
    REPORT_FE_ERROR("[GraphOpt][Init][AssmOpsStoreInfoVec] There is no OP store item in fe.ini of engine %s.",
                    engine_name_.c_str());
    return OPSTORE_EMPTY;
  }
  std::sort(ops_store_info_vector_.begin(), ops_store_info_vector_.end(), ComparePriority);
  FE_LOGD("Finish assemble OpsStoreInfoVector whose size is %zu of engine %s.", ops_store_info_vector_.size(),
          engine_name_.c_str());
  return SUCCESS;
}

Status Configuration::AssembleEachOpsStoreInfo(string &op_store_name, std::vector<string> &op_store_vector,
                                               FEOpsStoreInfo &ops_store_info) {
  if (op_store_vector.size() > IDX_BOTTOM) {
    REPORT_FE_ERROR("[GraphOpt][Init] The opp store size: %zu has overflowed.", op_store_vector.size());
    return OPSTORE_CONFIG_NOT_INTEGRAL;
  }
  ops_store_info.fe_ops_store_name = op_store_name;
  if (op_store_name.find("tbe-custom") == std::string::npos) {
    ops_store_info.is_custom_store = false;
  } else {
    ops_store_info.is_custom_store = true;
  }
  std::istringstream(op_store_vector.at(IDX_PRIORITY)) >> ops_store_info.priority;
  string str_op_impl_type = op_store_vector.at(IDX_OPIMPL_TYPE);
  if (!StringUtils::IsInteger(str_op_impl_type)) {
    REPORT_FE_ERROR("[GraphOpt][Init][AssemEachOpsStoreInfo] The opimpltype of opstore[%s] should be a positive integer.",
                    op_store_name.c_str());
    return OPSTORE_OPIMPLTYPE_INVALID;
  }
  int64_t impl_type;
  std::istringstream(str_op_impl_type) >> impl_type;
  ops_store_info.op_impl_type = static_cast<OpImplType>(impl_type);
  FE_LOGD("The file:%s , impl_type:%ld.", op_store_vector.at(IDX_OPIMPL_FILEPATH).c_str(), ops_store_info.op_impl_type);
  if (ops_store_info.op_impl_type == EN_IMPL_HW_TBE || ops_store_info.op_impl_type == EN_IMPL_VECTOR_CORE_HW_TBE ||
      ops_store_info.op_impl_type == EN_IMPL_HW_DSA) {
    ops_store_info.priority += op_store_priority_count_;
  }
  Status status = CheckOpStoreInfo(ops_store_info);
  if (status != SUCCESS) {
    return status;
  }

  if (ops_store_info.op_impl_type != EN_IMPL_PLUGIN_TBE &&
      (op_store_vector.at(IDX_CFG_FILEPATH).empty() || op_store_vector.at(IDX_OPIMPL_FILEPATH).empty())) {
    REPORT_FE_ERROR("[GraphOpt][Init][AssemEachOpsStoreInfo] At least one columns value in [%s] of fe.ini is empty.",
                    op_store_name.c_str());
    return OPSTORE_VALUE_ITEM_EMPTY;
  }

  std::string str_cfg_file_path = ascend_ops_path_ + op_store_vector.at(IDX_CFG_FILEPATH);
  std::string str_op_imply_file_path = ascend_ops_path_ + op_store_vector.at(IDX_OPIMPL_FILEPATH);
  if (op_store_vector.size() == OP_STORE_FORMAT_MAX_SIZE && op_store_vector[ID_IS_FULL_PATH] == "true") {
    str_cfg_file_path = op_store_vector.at(IDX_CFG_FILEPATH);
    str_op_imply_file_path = op_store_vector.at(IDX_OPIMPL_FILEPATH);
  }
  OpImplType impl = GetMainImplType<OpImplType>(ops_store_info.op_impl_type);
  if (impl == EN_IMPL_HW_TBE || impl== EN_IMPL_CUSTOM_TBE || impl == EN_IMPL_HW_DSA) {
    // default value of sub path is the lower case of soc version
    string sub_path = PlatformUtils::Instance().GetShortSocVersion();
    std::transform(sub_path.begin(), sub_path.end(), sub_path.begin(), ::tolower);
    str_cfg_file_path = str_cfg_file_path + "/" + sub_path;
  }
  FE_LOGD("The configuration file path of op store[%s] is %s.", op_store_name.c_str(), str_cfg_file_path.c_str());
  FE_LOGD("The implementation path of op store[%s] is %s.", op_store_name.c_str(), str_op_imply_file_path.c_str());

  if (impl == EN_IMPL_HW_TBE && GetRealPath(str_cfg_file_path).empty()) {
    const std::map<string, string> error_key_map = {
        {EM_VALUE, ascend_ops_path_}, {EM_ENV, ASCEND_OPP_PATH}, {EM_REASON, "ASCEND_OPP_PATH does not exist or access permission is denied during FE initialization"}};
    (void)REPORT_PREDEFINED_ERR_MSG(
        EM_ENVIRONMENT_VARIABLE_FAILED.c_str(),
        std::vector<const char *>({EM_VALUE.c_str(), EM_ENV.c_str(), EM_REASON.c_str()}),
        std::vector<const char *>({ascend_ops_path_.c_str(), ASCEND_OPP_PATH, "ASCEND_OPP_PATH does not exist or access permission is denied during FE initialization"}));
    FE_LOGE("Builtin op store[%s] with path [%s] is invalid.", op_store_name.c_str(), str_cfg_file_path.c_str());
    ErrorMessageDetail err_msg(EM_ENVIRONMENT_VARIABLE_FAILED,
        {ascend_ops_path_, ASCEND_OPP_PATH, "ASCEND_OPP_PATH does not exist or access permission is denied during FE initialization"});
    ReportErrorMessage(err_msg);
    FE_LOGE("Builtin op store[%s] with path [%s] is invalid.", op_store_name.c_str(), str_cfg_file_path.c_str());
    return OPSTORE_EMPTY;
  }

  ops_store_info.cfg_file_path = str_cfg_file_path;
  ops_store_info.op_impl_file_path = str_op_imply_file_path;

  string str_need_pre_compile = op_store_vector.at(IDX_NEED_PRECOMPILE);
  string str_need_compile = op_store_vector.at(IDX_NEED_COMPILE);
  std::transform(str_need_pre_compile.begin(), str_need_pre_compile.end(), str_need_pre_compile.begin(), ::tolower);
  std::transform(str_need_compile.begin(), str_need_compile.end(), str_need_compile.begin(), ::tolower);
  ops_store_info.need_pre_compile = str_need_pre_compile == NEED_PRECOMPILE_TRUE;
  ops_store_info.need_compile = str_need_compile == NEED_COMPILE_TRUE;
  return SUCCESS;
}

Status Configuration::VerifyOpStoreVector(std::vector<std::string> &op_store_vector,
                                          const std::string &op_store_name) const {
  if (op_store_vector.size() != OP_STORE_FORMAT_MAX_SIZE && op_store_vector.size() != OP_STORE_FORMAT_MAX_SIZE - 1) {
    REPORT_FE_ERROR("[GraphOpt][Init][VerifyOpStoreVec] The size of the columns in %s is %zu, which should be either %d or %d.",
                    op_store_name.c_str(), op_store_vector.size(), OP_STORE_FORMAT_MAX_SIZE,
                    OP_STORE_FORMAT_MAX_SIZE - 1);
    return OPSTORE_VALUE_ITEM_SIZE_INCORRECT;
  }

  if (op_store_vector.at(IDX_PRIORITY).empty() || op_store_vector.at(IDX_OPIMPL_TYPE).empty() ||
      op_store_vector.at(IDX_NEED_PRECOMPILE).empty() || op_store_vector.at(IDX_NEED_COMPILE).empty()) {
    REPORT_FE_ERROR("[GraphOpt][Init][VerifyOpStoreVec] At least one of the columns in the [%s] value in fe.ini is empty.",
                    op_store_name.c_str());
    return OPSTORE_VALUE_ITEM_EMPTY;
  }

  if (!StringUtils::IsInteger(op_store_vector.at(IDX_PRIORITY))) {
    REPORT_FE_ERROR("[GraphOpt][Init][VerifyOpStoreVec] The priority of opstore[%s] should be non-negative integer",
                    op_store_name.c_str());
    return OPSTORE_PRIORITY_INVALID;
  }
  return SUCCESS;
}

const string &Configuration::GetSocVersion() const { return PlatformUtils::Instance().GetSocVersion(); }

void Configuration::SetOpsStoreInfo(const FEOpsStoreInfo &fe_ops_store_info) {
  std::lock_guard<std::mutex> lock_guard(ops_store_info_vector_mutex_);
  ops_store_info_vector_.push_back(fe_ops_store_info);
  std::sort(ops_store_info_vector_.begin(), ops_store_info_vector_.end(), ComparePriority);
}

const std::vector<FEOpsStoreInfo> &Configuration::GetOpsStoreInfo() const {
  std::lock_guard<std::mutex> lock_guard(ops_store_info_vector_mutex_);
  return ops_store_info_vector_;
}

Status Configuration::GetOpStoreInfoByImplType(OpImplType op_impl_type, FEOpsStoreInfo &op_store_info) const {
  FE_LOGD("Begin to get op store info by impl_type. op_impl_type=%ld.", op_impl_type);
  Status return_status = FAILED;
  const std::vector<FEOpsStoreInfo> &ops_store_info_vec = GetOpsStoreInfo();
  for (auto &fe_op_store_info : ops_store_info_vec) {
    if (op_impl_type == fe_op_store_info.op_impl_type) {
      op_store_info = fe_op_store_info;
      return_status = SUCCESS;
      break;
    }
  }
  return return_status;
}

Status Configuration::CheckOpStoreInfo(const FEOpsStoreInfo &op_store_info) const {
  if (op_store_info.priority < PRIORITY_START_NUM) {
    REPORT_FE_ERROR("[GraphOpt][Init][ChkOpStoreInfo] The priority[%d] is out of range.", op_store_info.priority);
    return OPSTORE_PRIORITY_INVALID;
  }

  if (op_store_info.op_impl_type >= EN_SUBTYPE_RESERVED || op_store_info.op_impl_type < EN_IMPL_CUSTOM_CONSTANT_CCE) {
    REPORT_FE_ERROR("[GraphOpt][Init][ChkOpStoreInfo] The op impl type of op store [%s] is invalid.",
                    op_store_info.fe_ops_store_name.c_str());
    return OPSTORE_OPIMPLTYPE_INVALID;
  }

  for (const FEOpsStoreInfo &op_store : ops_store_info_vector_) {
    if (op_store.priority == op_store_info.priority) {
      REPORT_FE_ERROR("[GraphOpt][Init][ChkOpStoreInfo] Priority [%d] already exists in %s.",
                      op_store_info.priority, op_store.fe_ops_store_name.c_str());
      return OPSTORE_PRIORITY_INVALID;
    }
    if (op_store.op_impl_type == op_store_info.op_impl_type) {
      REPORT_FE_ERROR("[GraphOpt][Init][ChkOpStoreInfo] The op impl type of op stores can not be repeated.");
      return OPSTORE_OPIMPLTYPE_REPEAT;
    }
  }
  return SUCCESS;
}

bool Configuration::ContainKey(const string &key) const {
  auto iter = content_map_.find(key);
  return iter != content_map_.end();
}

Status Configuration::GetStringValue(const string &key, string &return_value) const {
  auto iter = content_map_.find(key);
  if (iter == content_map_.end()) {
    FE_LOGW("Cannot find the value for key [%s].", key.c_str());
    return FAILED;
  }

  return_value = iter->second;
  return SUCCESS;
}

bool Configuration::GetBoolValue(const string &key, bool default_value) const {
  auto iter = content_map_.find(key);
  if (iter == content_map_.end()) {
    return default_value;
  }

  bool value = false;
  std::istringstream(iter->second) >> std::boolalpha >> value;
  return value;
}

bool Configuration::IsInLicenseControlMap(const string &key) const {
  for (auto iter = kLicensePassNameMap.begin(); iter != kLicensePassNameMap.end(); ++iter) {
    if (iter->second == key) {
      return true;
    }
  }
  return false;
}

bool Configuration::GetOpImplMode(const std::string &op_name, const std::string &op_type,
                                  std::string &op_impl_mode) const {
  OpImplModeConfigParserPtr impl_mode_ptr = std::dynamic_pointer_cast<OpImplModeConfigParser>(impl_mode_parser_);
  if (impl_mode_ptr == nullptr) {
    return false;
  }
  return impl_mode_ptr->GetOpImplMode(op_name, op_type, op_impl_mode);
}

bool Configuration::GetCustomizeDtypeByOpType(const string &op_type, OpCustomizeDtype &custom_dtype) {
  std::string cust_dtypes_path = CombinedParamsKeyFromContext(CONFIG_PARSER_PARAM::CustDtypes);
  FE_LOGD("The cust_dtypes_path is %s.", cust_dtypes_path.c_str());
  BaseConfigParserPtr cust_dtypes_parser =
      GetConfigParserFromContext(CONFIG_PARSER_PARAM::CustDtypes, cust_dtypes_path);
  if (cust_dtypes_parser == nullptr) {
    cust_dtypes_parser = cust_dtypes_parser_;
  }
  OpCustDtypesConfigParserPtr cust_dtypes_ptr = std::dynamic_pointer_cast<OpCustDtypesConfigParser>(cust_dtypes_parser);
  if (cust_dtypes_ptr == nullptr) {
    return false;
  }
  return cust_dtypes_ptr->GetCustomizeDtypeByOpType(op_type, custom_dtype);
}

bool Configuration::GetCustomizeDtypeByOpName(const string &op_name, OpCustomizeDtype &custom_dtype) {
  std::string cust_dtypes_path = CombinedParamsKeyFromContext(CONFIG_PARSER_PARAM::CustDtypes);
  FE_LOGD("The cust_dtypes_path is %s.", cust_dtypes_path.c_str());
  BaseConfigParserPtr cust_dtypes_parser =
      GetConfigParserFromContext(CONFIG_PARSER_PARAM::CustDtypes, cust_dtypes_path);
  if (cust_dtypes_parser == nullptr) {
    cust_dtypes_parser = cust_dtypes_parser_;
  }
  OpCustDtypesConfigParserPtr cust_dtypes_ptr = std::dynamic_pointer_cast<OpCustDtypesConfigParser>(cust_dtypes_parser);
  if (cust_dtypes_ptr == nullptr) {
    return false;
  }
  return cust_dtypes_ptr->GetCustomizeDtypeByOpName(op_name, custom_dtype);
}


PrecisionPolicy Configuration::GetPrecisionPolicy(const std::string &op_type,
                                                  const PrecisionPolicy &op_kernel_policy) {
  std::string mix_list_path = CombinedParamsKeyFromContext(CONFIG_PARSER_PARAM::ModifyMixlist);
  FE_LOGD("The mix_list_path is set to %s.", mix_list_path.empty() ? "null" : mix_list_path.c_str());
  BaseConfigParserPtr mix_list_parser =
      GetConfigParserFromContext(CONFIG_PARSER_PARAM::ModifyMixlist, mix_list_path);
  if (mix_list_parser == nullptr) {
    mix_list_parser = mix_list_parser_;
  }
  ModifyMixlistConfigParserPtr mix_list_ptr = std::dynamic_pointer_cast<ModifyMixlistConfigParser>(mix_list_parser);
  if (mix_list_ptr == nullptr) {
    return op_kernel_policy;
  }
  return mix_list_ptr->GetPrecisionPolicy(op_type, op_kernel_policy);
}

void Configuration::GetCompilerGraphFilePath(string &graph_file_path) {
  Status ret = SUCCESS;
  if (engine_name_ == VECTOR_CORE_NAME) {
    ret = GetStringValue(VECTOR_CORE_CONFIG_KEY_COMPILER_GRAPH_FILE, graph_file_path);
  }
  if (engine_name_ == AI_CORE_NAME) {
    ret = GetStringValue(CONFIG_KEY_COMPILER_GRAPH_FILE, graph_file_path);
  }
  if (ret == SUCCESS) {
    if (graph_file_path.empty()) {
      FE_LOGW("The path of input compiler graph fusion rule json file is empty.");
      return;
    }
    const char_t *env_value = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, env_value);
    if (env_value == nullptr) {
      graph_file_path = lib_path_ + graph_file_path;
    } else {
      std::string path_env = string(env_value);
      graph_file_path = path_env + CONFIG_LIB64_RELATIVE_PATH + graph_file_path;
    }
  }
  return;
}

Status Configuration::GetGraphFilePath(string &graph_file_path) {
  GetCompilerGraphFilePath(graph_file_path);
  std::string real_path = GetRealPath(graph_file_path);
  if (!real_path.empty()) {
    FE_LOGD("The real path of graph fusion rule json is %s.", graph_file_path.c_str());
    return SUCCESS;
  }

  Status ret;
  if (engine_name_ == VECTOR_CORE_NAME) {
    ret = GetStringValue(VECTOR_CORE_CONFIG_KEY_GRAPH_FILE, graph_file_path);
  } else if (engine_name_ == kDsaCoreName) {
    ret = GetStringValue(DSA_CORE_CONFIG_KEY_GRAPH_FILE, graph_file_path);
  } else {
    ret = GetStringValue(CONFIG_KEY_GRAPH_FILE, graph_file_path);
  }
  if (ret == SUCCESS) {
    if (graph_file_path.empty()) {
      FE_LOGW("The path for the input built-in graph fusion rule JSON file is empty.");
      return SUCCESS;
    }
    graph_file_path = ascend_ops_path_ + graph_file_path;
  }
  FE_LOGD("The real path of graph fusion rule json is %s.", graph_file_path.c_str());
  return ret;
}

Status Configuration::GetCustomFilePath(string &custom_file_path) {
  Status ret;
  if (engine_name_ == VECTOR_CORE_NAME) {
    ret = GetStringValue(VECTOR_CORE_CONFIG_KEY_CUSTOM_FILE, custom_file_path);
  } else if (engine_name_ == kDsaCoreName) {
    ret = GetStringValue(DSA_CORE_CONFIG_KEY_CUSTOM_FILE, custom_file_path);
  } else {
    ret = GetStringValue(CONFIG_KEY_CUSTOM_FILE, custom_file_path);
  }
  if (ret == SUCCESS) {
    if (custom_file_path.empty()) {
      FE_LOGW("The path for the input custom graph fusion rule JSON file is empty.");
      return SUCCESS;
    }
    custom_file_path = ascend_ops_path_ + custom_file_path;
  }
  return ret;
}

Status Configuration::GetCustomPassFilePath(string &custom_pass_file_path) {
  Status ret;
  if (engine_name_ == VECTOR_CORE_NAME) {
    ret = GetStringValue(VECTOR_CORE_CONFIG_KEY_CUSTOM_PASS_FILE, custom_pass_file_path);
  } else if (engine_name_ == kDsaCoreName) {
    ret = GetStringValue(DSA_CORE_CONFIG_KEY_CUSTOM_PASS_FILE, custom_pass_file_path);
  } else {
    ret = GetStringValue(CONFIG_KEY_CUSTOM_PASS_FILE, custom_pass_file_path);
  }
  if (ret == SUCCESS) {
    if (custom_pass_file_path.empty()) {
      FE_LOGW("Input custom graph fusion pass path is empty.");
      return SUCCESS;
    }
    custom_pass_file_path = ascend_ops_path_ + custom_pass_file_path;
  }
  return ret;
}

/* If built in pass file path does not exist, use the old version path. */
Status Configuration::GetCompatibleBuiltInPath(string &builtin_pass_file_path) const {
  Status ret;

  ret = GetStringValue(CONFIG_KEY_BUILTIN_PASS_FILE, builtin_pass_file_path);
  if (ret != SUCCESS) {
    return ret;
  }

  if (GetRealPath(ascend_ops_path_ + builtin_pass_file_path).empty()) {
    FE_LOGI("File path [%s] is invalid; using old path fusion_pass/built_in.", builtin_pass_file_path.c_str());
    builtin_pass_file_path = kOldBuiltInPassPath;
    return SUCCESS;
  }

  return SUCCESS;
}

void Configuration::GetCompilerPassFilePath(string &compiler_pass_file_path) {
  Status ret = SUCCESS;
  if (engine_name_ == VECTOR_CORE_NAME) {
    ret = GetStringValue(VECTOR_CORE_CONFIG_KEY_COMPILER_PASS_FILE, compiler_pass_file_path);
  }
  if (engine_name_ == AI_CORE_NAME) {
    ret = GetStringValue(CONFIG_KEY_COMPILER_PASS_FILE, compiler_pass_file_path);
  }

  if (ret == SUCCESS) {
    if (compiler_pass_file_path.empty()) {
      FE_LOGW("Input graph fusion so path is empty.");
      return;
    }
    const char_t *env_value = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, env_value);
    if (env_value == nullptr) {
      compiler_pass_file_path = lib_path_ + compiler_pass_file_path;
    } else {
      std::string path_env = string(env_value);
      compiler_pass_file_path = path_env + CONFIG_LIB64_RELATIVE_PATH + compiler_pass_file_path;
    }
  }
  return;
}

Status Configuration::GetBuiltinPassFilePath(string &builtin_pass_file_path) {
  GetCompilerPassFilePath(builtin_pass_file_path);
  std::string real_path = GetRealPath(builtin_pass_file_path);
  if (!real_path.empty()) {
    FE_LOGD("Input graph fusion so path is %s.", real_path.c_str());
    return SUCCESS;
  }

  Status ret;
  if (engine_name_ == VECTOR_CORE_NAME) {
    ret = GetStringValue(VECTOR_CORE_CONFIG_KEY_BUILTIN_PASS_FILE, builtin_pass_file_path);
  } else if (engine_name_ == kDsaCoreName) {
    ret = GetStringValue(DSA_CORE_CONFIG_KEY_BUILTIN_PASS_FILE, builtin_pass_file_path);
  } else {
    ret = GetCompatibleBuiltInPath(builtin_pass_file_path);
  }
  if (ret == SUCCESS) {
    if (builtin_pass_file_path.empty()) {
      FE_LOGW("Input built-in graph fusion pass path is empty.");
      return SUCCESS;
    }
    builtin_pass_file_path = ascend_ops_path_ + builtin_pass_file_path;
  }
  return ret;
}

string Configuration::GetRootPath() {
  string root_path;
  Status ret = GetStringValue(CONFIG_KEY_ROOT, root_path);
  if (ret != SUCCESS) {
    root_path = lib_path_;
  }
  return root_path;
}

string Configuration::GetPrecisionModeStr() {
  std::string precision_mode;
  (void)FEContextUtils::GetPrecisionMode(precision_mode);
  return precision_mode;
}

BufferOptimize Configuration::GetBufferOptimize() const {
  return static_cast<BufferOptimize>(config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)]);
}

bool Configuration::IsEnableSmallChannel() const {
  return static_cast<bool>(config_param_vec_[static_cast<size_t>(CONFIG_PARAM::SmallChannel)]);
}

void Configuration::SetEnableSmallChannel(const bool &small_channel) {
  config_param_vec_[static_cast<size_t>(CONFIG_PARAM::SmallChannel)] = static_cast<int64_t>(small_channel);
}

/*
  jitCompile: 0 means false
              1 means true
              2 means auto, unknownShape op use false, staticShape op use true
*/
JitCompileCfg Configuration::GetJitCompileCfg() const {
  return static_cast<JitCompileCfg>(config_param_vec_[static_cast<size_t>(CONFIG_PARAM::JitCompile)]);
}

bool Configuration::IsEnableVirtualType() const {
  return static_cast<bool>(config_param_vec_[static_cast<size_t>(CONFIG_PARAM::VirtualType)]);
}

bool Configuration::IsEnableReuseMemory() const {
  return static_cast<bool>(GetConfigParamValueFromContext(CONFIG_PARAM::ReuseMemory));
}

bool Configuration::IsEnableCompressWeight() const {
  return static_cast<bool>(config_param_vec_[static_cast<size_t>(CONFIG_PARAM::CompressWeight)]);
}

bool Configuration::IsEnableSparseMatrixWeight() const {
  return static_cast<bool>(config_param_vec_[static_cast<size_t>(CONFIG_PARAM::SparseMatrixWeight)]);
}

bool Configuration::IsQuantDumpable() const {
  return static_cast<bool>(config_param_vec_[static_cast<size_t>(CONFIG_PARAM::QuantDumpable)]);
}

const string& Configuration::GetLicenseFusionStr() const {
  std::lock_guard<std::mutex> lock_guard(config_param_mutex_);
  return config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)];
}

const std::set<string>& Configuration::GetLicenseFusionDetailInfo() const {
  return license_fusion_detail_value_;
}

FormatModeType Configuration::GetFormatModeCfg() const {
  return static_cast<FormatModeType>(GetConfigParamValueFromContext(CONFIG_PARAM::FormatMode));
}

ExportCompileStatType Configuration::GetExportCompileStat() const {
  return static_cast<ExportCompileStatType>(GetConfigParamValueFromContext(CONFIG_PARAM::ExportCompileStat));
}

bool Configuration::IsEnableQuantBiasOptimize() const {
  return static_cast<bool>(config_param_vec_[static_cast<size_t>(CONFIG_PARAM::QuantBiasOptimize)]);
}

const std::map<int32_t, float>& Configuration::GetCompressRatios() const { return compress_ratios_; }

const float& Configuration::GetAICoreCompressRatio() const { return ai_core_compress_ratio_; }

int32_t Configuration::GetMemReuseDistThreshold() const { return mem_reuse_dist_threshold_; }

const std::unordered_set<string>& Configuration::GetFp16OpTypeList() const { return fp16_op_type_list_; }

const std::map<string, string>& Configuration::GetBinaryPathMap() const {
  return op_binary_path_map_;
}

std::string Configuration::GetAllOpsImplPath() const {
  std::stringstream all_path;
  for (const auto &op_store_info : ops_store_info_vector_) {
    all_path << op_store_info.op_impl_file_path;
    all_path << "|";
    FE_LOGD("Insert op impl path:%s, ops store name:%s.",
            op_store_info.op_impl_file_path.c_str(), op_store_info.fe_ops_store_name.c_str());
  }
  return all_path.str();
}

const std::string Configuration::GetOpDebugConfig() const {
  OpDebugConfigParserPtr op_debug_config_parser = std::dynamic_pointer_cast<OpDebugConfigParser>
                                                  (op_debug_config_parse_);
  return op_debug_config_parser->GetOpDebugConfig();
}

bool Configuration::GetMemoryCheckSwitch() const {
  OpDebugConfigParserPtr op_debug_config_parser = std::dynamic_pointer_cast<OpDebugConfigParser>
                                                  (op_debug_config_parse_);
  return op_debug_config_parser->IsNeedMemoryCheck();
}
std::map<string, string> Configuration::GetHardwareInfo() const {
  std::lock_guard<std::mutex> lock_guard(hardware_info_map_mutex_);
  return hardware_info_map_;
}

bool Configuration::IsDynamicImplFirst() const {
  return is_dynamic_impl_first_;
}

Status Configuration::RefreshParameters() {
  Status ret = InitConfigParamFromContext();
  if (ret != SUCCESS) {
    FE_LOGE("Failed to refresh config parameters from context.");
    return ret;
  }

  ret = InitDebugConfigParam();
  if (ret != SUCCESS) {
    FE_LOGE("Failed to refresh option parameters from context.");
    return ret;
  }

  ret = RefreshImplMode();
  if (ret != SUCCESS) {
    FE_LOGE("Failed to refresh implementation mode from context.");
    return ret;
  }

  ret = RefreshCustDtypes();
  if (ret != SUCCESS) {
    FE_LOGE("Failed to refresh customer data types from context.");
    return ret;
  }

  ret = RefreshMixList();
  if (ret != SUCCESS) {
    FE_LOGE("Failed to refresh mixlist from context.");
    return ret;
  }
  RefreshSmallChannel();
  return SUCCESS;
}

Status Configuration::InitDebugConfigParam() {
  if (op_debug_config_parse_->InitializeFromContext() != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status Configuration::RefreshImplMode() {
  FE_CHECK_NOTNULL(impl_mode_parser_);
  return impl_mode_parser_->InitializeFromContext();
}

void Configuration::RefreshSmallChannel() {
  std::string opt_val;
  (void)ge::GetThreadLocalContext().GetOo().GetValue(kComLevelO1Opt, opt_val);
  if (opt_val == kStrFalse) {
    FE_LOGI("Force-close small channel at current compile level.");
    SetEnableSmallChannel(false);
  }
  return;
}

Status Configuration::RefreshCustDtypes() {
  Status ret;
  std::string combined_params_key = CombinedParamsKeyFromContext(CONFIG_PARSER_PARAM::CustDtypes);
  if (combined_params_key.empty()) {
    FE_LOGD("The combined_params_key for cust_dtypes is empty in the context.");
    return SUCCESS;
  }
  FE_LOGD("The combined_params_key of cust_dtypes is %s.", combined_params_key.c_str());
  BaseConfigParserPtr cust_dtypes_parser =
      GetConfigParserFromContext(CONFIG_PARSER_PARAM::CustDtypes, combined_params_key);
  if (cust_dtypes_parser == nullptr) {
    BaseConfigParserPtr new_cust_dtypes_parser = nullptr;
    FE_MAKE_SHARED(new_cust_dtypes_parser = std::make_shared<OpCustDtypesConfigParser>(), return FAILED);
    FE_CHECK_NOTNULL(new_cust_dtypes_parser);
    ret = new_cust_dtypes_parser->InitializeFromContext(combined_params_key);
    if (ret != SUCCESS) {
      FE_LOGE("Failed to initialize new_cust_dtypes_parser from context.");
      return ret;
    }
    std::lock_guard<std::mutex>
        lock_guard(config_parser_map_mutex_vec_[static_cast<size_t>(CONFIG_PARSER_PARAM::CustDtypes)]);
    config_parser_map_vec_[static_cast<size_t>(CONFIG_PARSER_PARAM::CustDtypes)]
        .emplace(combined_params_key, new_cust_dtypes_parser);
    return SUCCESS;
  }
  return SUCCESS;
}

Status Configuration::RefreshMixList() {
  Status ret;
  std::string combined_params_key = CombinedParamsKeyFromContext(CONFIG_PARSER_PARAM::ModifyMixlist);
  if (combined_params_key.empty()) {
    FE_LOGD("The combined_params_key of modify_mixlist is empty from context.");
    return SUCCESS;
  }
  FE_LOGD("The combined_params_key of modify_mixlist is %s.", combined_params_key.c_str());

  BaseConfigParserPtr mix_list_parser =
      GetConfigParserFromContext(CONFIG_PARSER_PARAM::ModifyMixlist, combined_params_key);
  if (mix_list_parser == nullptr) {
    BaseConfigParserPtr new_mix_list_parser = nullptr;
    FE_MAKE_SHARED(new_mix_list_parser = std::make_shared<ModifyMixlistConfigParser>(), return FAILED);
    FE_CHECK_NOTNULL(new_mix_list_parser);
    ret = new_mix_list_parser->InitializeFromContext(combined_params_key);
    if (ret != SUCCESS) {
      FE_LOGE("Failed to initialize new_mix_list_parser from context.");
      return ret;
    }
    std::lock_guard<std::mutex>
        lock_guard(config_parser_map_mutex_vec_[static_cast<size_t>(CONFIG_PARSER_PARAM::ModifyMixlist)]);
    config_parser_map_vec_[static_cast<size_t>(CONFIG_PARSER_PARAM::ModifyMixlist)]
        .emplace(combined_params_key, new_mix_list_parser);
    return SUCCESS;
  }
  return SUCCESS;
}

int32_t Configuration::ParseDataVisitDistThreshold() const {
  string threshold;
  Status status = GetStringValue("l2cache.datavisitdist.threshold", threshold);
  if (status != SUCCESS || threshold.empty()) {
    return DATA_VISIT_DIST_THRESHOLD;
  }
  try {
    return stoi(threshold);
  } catch (...) {
    FE_LOGE("Failed to convert %s to an integer value.", threshold.c_str());
    return DATA_VISIT_DIST_THRESHOLD;
  }
}

void Configuration::InitMemReuseDistThreshold() {
  string threshold;
  Status status = GetStringValue("mem.reusedist.threshold", threshold);
  if (status != SUCCESS || threshold.empty()) {
    return;
  }
  try {
    mem_reuse_dist_threshold_ = stoi(threshold);
  } catch (...) {
    FE_LOGW("Failed to convert %s to an integer value.", threshold.c_str());
  }
  FE_LOGD("get mem reuse distance:%d.", mem_reuse_dist_threshold_);
  return;
}

void Configuration::InitFp16OpType() {
  std::vector<string> op_types = ParseConfig(kFp16OpType, ',');
  std::unordered_set<string> fp16_op_type_list(op_types.begin(), op_types.end());
  fp16_op_type_list_ = std::move(fp16_op_type_list);
  return;
}

void Configuration::InitCompressRatio() {
  string ratio_str;
  Status status = GetStringValue("multi_core.compress_ratio", ratio_str);
  if (status != SUCCESS || ratio_str.empty()) {
    return;
  }
  ai_core_compress_ratio_ = kDefaultCompressRatioThreshold;
  std::vector<string> multi_core_ratios = StringUtils::Split(ratio_str, '|');
  for (const auto &multi_core_ratio : multi_core_ratios) {
    std::vector<string> core_ratio = StringUtils::Split(multi_core_ratio, ':');
    if (core_ratio.size() != 2) { // 2 means no ':' or configure illegal
      continue;
    }
    int32_t core_num;
    try {
      core_num = std::stoi(core_ratio[0]);
    } catch (...) {
      FE_LOGW("Failed to convert %s to an integer value.", core_ratio[0].c_str());
      continue;
    }
    float ratio;
    try {
      ratio = std::stof(core_ratio[1]);
    } catch (...) {
      FE_LOGW("Convert %s to float value failed.", core_ratio[1].c_str());
      continue;
    }
    compress_ratios_[core_num] = ratio;
    if (core_num == static_cast<int32_t>(PlatformUtils::Instance().GetAICoreNum())) {
      ai_core_compress_ratio_ = ratio;
    }
  }
  FE_LOGD("The ai_core compression ratio threshold is %f.", ai_core_compress_ratio_);
}

std::vector<string> Configuration::ParseConfig(const string &key, char pattern) const {
  string value_str;
  Status get_status = GetStringValue(key, value_str);
  FE_LOGD("ParseConfig: key=[%s], value=[%s].", key.c_str(), value_str.c_str());
  std::vector<string> result;
  if (get_status == SUCCESS) {
    return StringUtils::Split(StringUtils::Trim(value_str), pattern);
  }
  return result;
}

void Configuration::GetCompilerFusionConfigFilePath(std::string &fusion_config_file_path) const {
  auto iter = content_map_.find(FUSION_CONFIG_COMPILER_FILE);
  if (iter != content_map_.end()) {
    fusion_config_file_path = iter->second;
  } else {
    FE_LOGD("Cannot find %s in content_map_", FUSION_CONFIG_COMPILER_FILE.c_str());
    return;
  }
  const char_t *env_value = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, env_value);
  if (env_value == nullptr) {
    fusion_config_file_path = lib_path_ + fusion_config_file_path;
  } else {
    std::string path_env = string(env_value);
    fusion_config_file_path = path_env + CONFIG_LIB64_RELATIVE_PATH + fusion_config_file_path;
  }
  FE_LOGD("The actual path to the compiler fusion configuration file is %s.", fusion_config_file_path.c_str());
  return;
}

string Configuration::GetBuiltInFusionConfigFilePath() const {
  std::string fusion_config_file_path = "";
  GetCompilerFusionConfigFilePath(fusion_config_file_path);
  std::string real_path = GetRealPath(fusion_config_file_path);
  if (!real_path.empty()) {
    FE_LOGD("The real path of fusion config file is %s", fusion_config_file_path.c_str());
    return fusion_config_file_path;
  }

  auto iter = content_map_.find(FUSION_CONFIG_BUILT_IN_FILE);
  if (iter != content_map_.end()) {
    fusion_config_file_path = iter->second;
  }
  string real_file_path = ascend_ops_path_ + fusion_config_file_path;
  FE_LOGD("The real path of the built-in fusion switch file is %s.", real_file_path.c_str());
  return real_file_path;
}

string Configuration::GetSupportFusionPassFilePath() const {
  std::string support_fusion_pass_file_path = "";
  auto iter = content_map_.find(SUPPORT_FUSSION_PASS_FILE);
  if (iter != content_map_.end()) {
    support_fusion_pass_file_path = iter->second;
  }
  const char_t *env_value = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, env_value);
  if (env_value == nullptr) {
    support_fusion_pass_file_path = lib_path_ + support_fusion_pass_file_path;
  } else {
    std::string path_env = string(env_value);
    support_fusion_pass_file_path = path_env + CONFIG_LIB64_RELATIVE_PATH + support_fusion_pass_file_path;
  }
  return support_fusion_pass_file_path;
}

bool Configuration::EnableL1Fusion() const {
  return GetBufferOptimize() == EN_L1_OPTIMIZE;
}

bool Configuration::EnableL2Fusion() const {
  std::string build_mode_value = FEContextUtils::GetBuildMode();
  if (build_mode_value == ge::BUILD_MODE_BASELINE || build_mode_value == ge::BUILD_MODE_TUNING) {
    return true;
  }
  return GetBufferOptimize() != EN_OFF_OPTIMIZE;
}

bool Configuration::GetDumpOriginalNodesEnable() const {
  return GetBoolValue(CONFIG_KEY_ORIGINAL_NODES_ENABLE, false);
}

bool Configuration::GetMixL2Enable() const {
  bool enable = false;
  enable = GetBoolValue(CONFIG_KEY_MIX_L2_ENABLE, true);
  FE_LOGD("Get MiXl2 switch[%d].", enable);
  return enable;
}

bool Configuration::IsEnableSuperkernelPlus() const {
  return PlatformUtils::Instance().GetShortSocVersion() == kShortSocVersionAscend035 &&
         GetBoolValue(CONFIG_KEY_SUPERKERNEL_PLUS_ENABLE, false);
}

bool Configuration::GetDuplicationSwitch() const {
  return GetBoolValue(CONFIG_KEY_MAY_DUPLICATE_IN_AUTO_FUSION, false);
}

bool Configuration::IsEnableNetworkAnalysis() const { return enable_network_analysis_; }

bool Configuration::IsEnableOpImplStrategy() const { return enable_op_impl_strategy_; }

bool Configuration::IsEnableUbFusion() const { return enable_ub_fusion_; }

bool Configuration::IsEnableAclnn() const { return enable_aclnn_; };

bool Configuration::IsEnableRt2() const { return enable_rt2_; };

bool Configuration::IsEnableFirstLayerQuantization() const {
  return IsEnableSmallChannel() ? enable_first_layer_quantization_ : false;
}

string Configuration::GetFeLibPath() const { return lib_path_ + "plugin/opskernel/"; }

string Configuration::GetLibPath() const { return lib_path_; }

bool Configuration::GetConfigValueByKey(const std::map<string, string> &options, const string &file_key,
                                        const string &cfg_key, string &value, string &file_path) const {
  std::map<string, string>::const_iterator iter = options.find(file_key);
  if (iter == options.end() || iter->second.empty()) {
    FE_LOGD("Not find key[%s] in options.", file_key.c_str());
    return true;
  }
  file_path = iter->second;
  string real_path = GetRealPath(file_path);
  FE_LOGI("The real path for the config is %s.", real_path.c_str());
  std::ifstream ifs(real_path);
  if (!ifs.is_open()) {
    ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID, {file_path, file_key,
                               "The file does not exist or its access permission is denied"});
    ReportErrorMessage(err_msg);
    FE_LOGE("[Configuration][GetConfigValueByKey] Config file: [%s] did not exist.", file_path.c_str());
    return false;
  }
  string line;
  while (std::getline(ifs, line)) {
    line = StringUtils::Trim(line);
    if (line.empty() || line.find('#') == 0) {
      continue;
    }
    if (line.find(cfg_key) == string::npos) {
      continue;
    }
    value = line;
    FE_LOGD("Getting config line value: %s.", value.c_str());
    break;
  }
  ifs.close();
  return true;
}

bool Configuration::IsConfigDebugListOp(const ge::OpDescPtr &op_desc_ptr) const {
  OpDebugConfigParserPtr op_debug_config_parser = std::dynamic_pointer_cast<OpDebugConfigParser>
                                                  (op_debug_config_parse_);
  return op_debug_config_parser->IsOpDebugListOp(op_desc_ptr);
}

Status Configuration::InitializeExtend(const std::map<string, string> &options) {
  if (!InitFirstLayerQuantization(options)) {
    return FAILED;
  }
  return SUCCESS;
}
bool Configuration::InitFirstLayerQuantization(const std::map<string, string> &options) {
  string line;
  string file_path;
  if (!GetConfigValueByKey(options, ge::COMPRESSION_OPTIMIZE_CONF, kEnableFirstQuantKey, line, file_path)) {
    return false;
  }
  if (line.empty()) {
    return true;
  }
  size_t pos_of_equal = line.find(':');
  if (pos_of_equal == string::npos) {
    ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID,
        {file_path, ge::COMPRESSION_OPTIMIZE_CONF, "Line:\"" + line + "\" not contain \":\""});
    ReportErrorMessage(err_msg);
    REPORT_FE_ERROR("[Configuration][InitFirstLayerQuantization]Config [%s] format is error.", line.c_str());
    return false;
  }
  string value = line.substr(pos_of_equal + 1);
  value = StringUtils::Trim(value);
  if (value == "true") {
    enable_first_layer_quantization_ = true;
  } else if (value == "false") {
    enable_first_layer_quantization_ = false;
  } else {
    ErrorMessageDetail err_msg(
        EM_INPUT_OPTION_INVALID,
        {file_path, ge::COMPRESSION_OPTIMIZE_CONF,
         "In file:\"" + file_path + "\", enable_first_layer_quantization only support true or false"});
    ReportErrorMessage(err_msg);
    REPORT_FE_ERROR(
        "[Configuration][InitFirstLayerQuantization]Config value [%s] is unsupported, it should be"
        "true or false.",
        value.c_str());
    return false;
  }
  FE_LOGD("Get enable first layer flag is %s.", value.c_str());
  return true;
}

bool Configuration::IsEnableCustomImplMode() const {
  OpImplModeConfigParserPtr impl_mode_ptr = std::dynamic_pointer_cast<OpImplModeConfigParser>(impl_mode_parser_);
  return impl_mode_ptr->IsEnableCustomImplMode();
}

std::string Configuration::EmplaceHf32ModeForAclnn(const std::string &hf32_mode) const {
  OpImplModeConfigParserPtr impl_mode_ptr = std::dynamic_pointer_cast<OpImplModeConfigParser>(impl_mode_parser_);
  if (impl_mode_ptr == nullptr) {
    return hf32_mode;
  }
  return impl_mode_ptr->EmplaceHf32ModeForAclnn(hf32_mode);
}

bool Configuration::IsEnableL2Buffer() const {
  return GetBufferOptimize() != EN_OFF_OPTIMIZE && PlatformUtils::Instance().GetL2Type() == L2Type::Buff;
}

std::string Configuration::GetConfigStrParamValueFromContext(CONFIG_STR_PARAM config_str_param_enum_type) const {
  const auto iter = kConfigStrParamMap.find(config_str_param_enum_type);
  if (iter != kConfigStrParamMap.cend()) {
    string param_value;
    (void)ge::GetContext().GetOption(iter->second, param_value);
    if (!param_value.empty()) {
      return param_value;
    } else {
      FE_LOGD("The value for [%s] is either not found or it is empty in the context.", iter->second.c_str());
      return config_str_param_vec_[static_cast<size_t>(config_str_param_enum_type)];
    }
  }
  return config_str_param_vec_[static_cast<size_t>(config_str_param_enum_type)];
}

int64_t Configuration::GetConfigParamValueFromContext(CONFIG_PARAM config_param_enum_type) const {
  const auto item = kConfigParamMap.find(config_param_enum_type);
  if (item != kConfigParamMap.cend()) {
    string param_key = std::get<1>(item->second);
    string param_value;
    (void)ge::GetContext().GetOption(param_key, param_value);
    if (!param_value.empty()) {
      std::map<string, int64_t> param_map = std::get<2>(item->second);
      const auto iter_value = param_map.find(param_value);
      if (iter_value != param_map.cend()) {
        FE_LOGD("The value of param[%s] after conversion is %ld from context", param_key.c_str(), iter_value->second);
        return iter_value->second;
      } else {
        FE_LOGW("The value [%s] for parameter [%s] is invalid in this context.", param_value.c_str(), param_key.c_str());
        return config_param_vec_[static_cast<size_t>(config_param_enum_type)];
      }
    } else {
      FE_LOGD("The value for [%s] is either not found or it is empty. from context.", param_key.c_str());
      return config_param_vec_[static_cast<size_t>(config_param_enum_type)];
    }
  }
  return config_param_vec_[static_cast<size_t>(config_param_enum_type)];
}

std::string Configuration::CombinedParamsKeyFromOptions(CONFIG_PARSER_PARAM config_parser_param_enum_type,
                                                        const std::map<string, string> &options) const {
  std::string combined_params_key = "";
  const auto item = kConfigParserParamMap.find(config_parser_param_enum_type);
  if (item != kConfigParserParamMap.cend()) {
    for (auto param_key : item->second) {
      std::string param_value = "";
      std::map<std::string, std::string>::const_iterator iter = options.find(param_key);
      if (iter != options.cend() && !iter->second.empty()) {
        combined_params_key += iter->second;
        if (param_key != item->second.back()) {
          combined_params_key += ",";
        }
      }
    }
    return combined_params_key;
  }
  return combined_params_key;
}

std::string Configuration::CombinedParamsKeyFromContext(CONFIG_PARSER_PARAM config_parser_param_enum_type) const {
  std::string combined_params_key = "";
  const auto item = kConfigParserParamMap.find(config_parser_param_enum_type);
  if (item != kConfigParserParamMap.cend()) {
    for (auto param_key : item->second) {
      std::string param_value = "";
      (void)ge::GetContext().GetOption(param_key, param_value);
      combined_params_key += param_value;
      if (param_key != item->second.back()) {
        combined_params_key += ",";
      }
    }
    return combined_params_key;
  }
  return combined_params_key;
}

BaseConfigParserPtr Configuration::GetConfigParserFromContext(CONFIG_PARSER_PARAM config_parser_param_enum_type,
    const std::string &combined_params_key) {
  std::lock_guard<std::mutex>
      lock_guard(config_parser_map_mutex_vec_[static_cast<size_t>(config_parser_param_enum_type)]);
  if (!combined_params_key.empty()) {
    const auto &config_parser_map = config_parser_map_vec_[static_cast<size_t>(config_parser_param_enum_type)];
    const auto &iter_value = config_parser_map.find(combined_params_key);
    if (iter_value != config_parser_map.cend()) {
      return iter_value->second;
    } else {
      return nullptr;
    }
  }
  return nullptr;
}

uint64_t Configuration::GetCompileTaskTraceTimeInterval() const {
  std::string time_interval_str;
  GetStringValue(kCompileTaskTraceTimeInterval, time_interval_str);
  try {
    return std::stoull(time_interval_str);
  } catch (const std::invalid_argument &e) {
    return kDefaultTraceTimeInterval;
  }
}

uint64_t Configuration::GetCompileTaskTraceTimeConstThreshold() const {
  std::string time_interval_str;
  GetStringValue(kCompileTaskTraceTimeConstThreshold, time_interval_str);
  try {
    return std::stoull(time_interval_str);
  } catch (const std::invalid_argument &e) {
    return kDefaultTraceTimeConstThreshold;
  }
}
}  // namespace fe
