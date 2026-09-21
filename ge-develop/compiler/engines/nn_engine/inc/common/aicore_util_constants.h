/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_INC_COMMON_AICORE_UTIL_CONSTANTS_H_
#define FUSION_ENGINE_INC_COMMON_AICORE_UTIL_CONSTANTS_H_

#include <string>
#include <map>
#include <vector>
#include <unordered_set>
#include "graph/types.h"

namespace fe {
/* engine name of AI core and vector core */
const std::string AI_CORE_NAME = "AIcoreEngine";
const std::string VECTOR_CORE_NAME = "VectorEngine";
const std::string kDsaCoreName = "DSAEngine";

const std::string L1_OPTIMIZE = "l1_optimize";
const std::string L2_OPTIMIZE = "l2_optimize";
const std::string OFF_OPTIMIZE = "off_optimize";

/* allow auto mix precision */
const std::string ALLOW_MIX_PRECISION = "allow_mix_precision";
const std::string ALLOW_MIX_PRECISION_FP16 = "allow_mix_precision_fp16";
const std::string V2_MIX_FP16 = "mixed_float16";
const std::string ALLOW_MIX_PRECISION_BF16 = "allow_mix_precision_bf16";
const std::string V2_MIX_BF16 = "mixed_bfloat16";
const std::string CUBE_FP16IN_FP32OUT = "cube_fp16in_fp32out";

/* force float16  */
const std::string FORCE_FP16 = "force_fp16";
const std::string V2_FP16 = "fp16";
/* force lowerprecison  */
const std::string FORCE_LOWERPRECISION = "force_lowerprecision";

/* force float32  */
const std::string FORCE_FP32 = "force_fp32";

/* allow fp32 to fp16 */
const std::string ALLOW_FP32_TO_FP16 = "allow_fp32_to_fp16";
/* allow fp32 to bf16 */
const std::string ALLOW_FP32_TO_BF16 = "allow_fp32_to_bf16";
/* allow fp32 to lowerprecision */
const std::string ALLOW_FP32_TO_LOWPRECISION = "allow_fp32_to_lowprecision";
/* need to update dtype when op checksupport*/
const std::string NEED_UPDATE_DTYPE_WHEN_OP_CHECKSUPPORT = "need_update_dtype_when_op_checksupport";

/* must remain origin dtype */
const std::string MUST_KEEP_ORIGIN_DTYPE = "must_keep_origin_dtype";

const std::string V2_ORIGIN = "origin";

/* hif8 */
const std::string kCubeHif8 = "cube_hif8";
const std::string kMixedHif8 = "mixed_hif8";

const std::string SHAPE_GENERALIZED = "shape_generalized";

const uint32_t DEFAULT_SUB_FORMAT = 1;

const uint32_t SUPPORT_ALL_SUB_FORMAT = 0;

const int64_t IS_UNKNOWN_SHAPE_VALUE = 1;

const int64_t SHAPE_UNKNOWN_DIM = -1;

const int64_t SHAPE_UNKNOWN_DIM_NUM = -2;

const int64_t SHAPE_NUMBER_0 = 0;

const int64_t SHAPE_NUMBER_2 = 2;

const int64_t SHAPE_NUMBER_8 = 8;

const int64_t SHAPE_NUMBER_16 = 16;

const int64_t SHAPE_NUMBER_32 = 32;

const int64_t SHAPE_NUMBER_64 = 64;

const int64_t SHAPE_NUMBER_128 = 128;

const int64_t SHAPE_NUMBER_256 = 256;

const uint32_t INVALID_DTYPE_AND_FORMAT_SIZE = 0xffffffff;

constexpr uint32_t kInvalidTaskRatio = 0xFFFFFFFFU;

constexpr uint64_t kTaskPlaceHolderAddr = 0xFFFFFFFFFFFFFFFFU;

const size_t kMinThreadNum = 2;

const int32_t kAnchorMaxIdxNum = 64;

const size_t kMaxOpNmaLen = 512;

const size_t kMinPromoteSize = 2;

const uint32_t kManualMode = 0U;

const uint32_t kAutoMode = 1U;

const size_t kMixNotifyIdNum = 2;

const int32_t kDataVisitDistThreshold = 5; // data distance threshlod for rc cache optimization

const int32_t kRuntimeTypeHeterogeneous = 1; // Helper scene

// mode == 1 indicates we need reserve 8 Bytes for the args beginning
constexpr const int64_t IS_MIX_FIRST_FIELD_MODE = 1;

constexpr char const *kCoreTypeAIC = "AIC";

constexpr char const *kCoreTypeAIV = "AIV";

constexpr char const *kCoreTypeMixAIC = "MIX_AIC";

constexpr char const *kCoreTypeMixAIV = "MIX_AIV";

constexpr char const *kCoreTypeMixVectorCore = "MIX_VECTOR_CORE";

constexpr char const *kCoreTypeMixAICore = "MIX_AICORE";

const std::string kCoreTypeMixEnhance = "MIX";

const std::string kTbeMixCubePrefix = "_mix_aic";

const std::string kTbeMixVectorPrefix = "_mix_aiv";

const std::string kKernelNamesPrefix = "_kernel_names_prefix";

const std::string kTbeMixEnhancedPrefix = "_mix_enhanced";

const std::string kDynRatioTilingKey = "tilingKey";

const std::string kMixDynamicRatio = "_mix_dynamic_ratio";

const std::string kDynRatioAttr = "mix_tiling_with_ratio_attr";

const std::string kDynRatioTiling = "mix_tiling_key";

const std::string kDynRatioCRatio = "mix_tiling_c_ratio";

const std::string kDynRatioVRatio = "mix_tiling_v_ratio";

const std::string kThreadKernelName = "_thread_kernelname";

const std::string kFFTSPlusMode = "ffts-plus";

const std::string kFFTSMode = "ffts";

const std::string kAICoreMode = "ffts_mode";

const std::string kDsaCoreEngineName = "DSAEngine";

const std::string kAttrDsaCtxDef = "_dsa_ctx_def";

const std::string kAttrAICpuCtxDef = "_ffts_plus_aicpu_ctx_def";

const std::string kRuntimeContentx = "_ffts_runtime_context";

const std::string kAttrAICoreCtxDef = "_aicore_ctx_def";

const std::string kAttrAICoreCtxType = "_aicore_ctx_type";

const std::string kCubeVecCombine = "cube_vector_combine";
const std::string kCubeVecFuse = "fuse";
const std::string kCubeVecSplit = "split";
const std::string kHardwareSyncBetweenCore = "hardware_sync_betweencore";
const std::string kSupportFixPipeAbility = "support_fixpipe_ability";
const std::string kCfgSocInfo = "SoCInfo";
const std::string kCfgAiCoreCnt = "ai_core_cnt";
const std::string kCfgVectorCoreCnt = "vector_core_cnt";
const std::string kCfgVirTypeList = "vir_type_list";
const std::string kAttrOptionalInputMode = "optionalInputMode";
const std::string kAttrOptionalOutputMode = "optionalOutputMode";
constexpr const char* kInputParaTypeList = "_input_para_type_list";
constexpr const char* kOutputParaTypeList = "_output_para_type_list";
constexpr const char* kInputNameList = "_input_name_list";
constexpr const char* kOutputNameList = "_output_name_list";
constexpr const char* kInputInsertOptPosList = "_input_insert_opt_pos_list";
const std::string kModelBinFileSuffix = ".om";
const std::string kDefaultTrueStr = "1";
constexpr const char* kOpDebugConfig = "op_debug_config";
constexpr const char* kOpDebugList = "op_debug_list";
constexpr const char* kOpDebugCompile = "_op_debug_compile";
constexpr const char* kAttrDynamicParamMode = "dynamicParamMode";
constexpr const char* kAttrArgsSizeByFormat = "args_size_by_format";
constexpr const char* kGenPlaceholder = "gen_placeholder";
constexpr const char* kNoPlaceholder = "no_placeholder";
constexpr const char* kFoldedWithDesc = "folded_with_desc";
constexpr const char* kUnFolded = "unfolded";
constexpr const char* kAttrOutputInplaceAbility = "_output_inplace_ability";

constexpr const char* kInvalidPattern = "Opaque";
constexpr const char* kMemoryCheck = "_memcheck";
constexpr const char* kAippConfigPath = "aipp_config_path";
constexpr const char* kInfNan = "INF_NAN";
const std::string kStrTrue = "true";
const std::string kStrFalse = "false";

// no usage for fe, keep it for op pass

constexpr const char* kFESingleOpScene = "_fe_single_op_scene";
constexpr const char* kStrRangeLimit = "rangeLimit";
constexpr const char* kStrPrebuildPattern = "prebuildPattern";
constexpr const char* kStrPromoteType = "promoteType";
constexpr char const *kStrOutputInplaceAbility = "outputInplaceAbility";

constexpr const char* ATTR_NAME_DISABLE_MIX_VECTOR_CORE = "_disable_mix_vector_core";
constexpr const char* ATTR_NAME_MIX_CORE_NUM_VEC = "_mix_core_num_vec";
const std::map<std::string, bool> STR_BOOL_MAP{{"1", true}, {"0", false}};

const std::vector<ge::Format> FE_ORIGIN_FORMAT_VECTOR = {ge::FORMAT_NCHW,  ge::FORMAT_NHWC,  ge::FORMAT_HWCN,
                                                         ge::FORMAT_CHWN,  ge::FORMAT_NDHWC, ge::FORMAT_NCDHW,
                                                         ge::FORMAT_DHWCN, ge::FORMAT_DHWNC, ge::FORMAT_ND};

const std::unordered_set<ge::Format> FE_ORIGIN_FORMAT_SET =
    {ge::FORMAT_NCHW,  ge::FORMAT_NHWC,  ge::FORMAT_HWCN,
     ge::FORMAT_CHWN,  ge::FORMAT_NDHWC, ge::FORMAT_NCDHW,
     ge::FORMAT_DHWCN, ge::FORMAT_DHWNC, ge::FORMAT_ND};

const std::vector<ge::Format> FE_HEAVY_FORMAT_VECTOR = {
    ge::FORMAT_NC1HWC0_C04, ge::FORMAT_NC1HWC0,  ge::FORMAT_C1HWNCoC0,    ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_NZ,  ge::FORMAT_NDC1HWC0, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_WINO, ge::FORMAT_C1HWC0, ge::FORMAT_FRACTAL_NZ_C0_2,
    ge::FORMAT_FRACTAL_NZ_C0_4, ge::FORMAT_FRACTAL_NZ_C0_8, ge::FORMAT_FRACTAL_NZ_C0_16, ge::FORMAT_FRACTAL_NZ_C0_32};

const std::unordered_set<ge::Format> FE_HEAVY_FORMAT_SET = {
    ge::FORMAT_NC1HWC0_C04, ge::FORMAT_NC1HWC0,  ge::FORMAT_C1HWNCoC0,    ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_NZ,  ge::FORMAT_NDC1HWC0, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_C1HWC0, ge::FORMAT_FRACTAL_NZ_C0_2, ge::FORMAT_FRACTAL_NZ_C0_4,
    ge::FORMAT_FRACTAL_NZ_C0_8, ge::FORMAT_FRACTAL_NZ_C0_16, ge::FORMAT_FRACTAL_NZ_C0_32};

const std::unordered_set<ge::Format> FE_3D_FORMAT_SET = {
    ge::FORMAT_NDHWC, ge::FORMAT_NCDHW, ge::FORMAT_DHWCN, ge::FORMAT_DHWNC,
    ge::FORMAT_NDC1HWC0, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE};

const std::unordered_set<ge::DataType> kDtypeSensitiveOpNotHeavy = {
    ge::DT_INT64, ge::DT_COMPLEX32, ge::DT_COMPLEX64};

const std::map<std::string, int64_t> kOpImplStrToInt{
    {"default",           0x1},
    {"high_performance",  0x2},
    {"high_precision",    0x4},
    {"super_performance", 0x8},
    {"support_out_of_bound_index", 0x10},
    {"enable_float_32_execution", 0x20},
    {"enable_hi_float_32_execution", 0x40}
};
const std::map<int64_t, std::string> kOpImplIntToStr{
    {0x1, "default"},
    {0x2, "high_performance"},
    {0x4, "high_precision"},
    {0x8, "super_performance"},
    {0x10, "support_out_of_bound_index"},
    {0x20, "enable_float_32_execution"},
    {0x40, "enable_hi_float_32_execution"}
};

constexpr const char* kComLevelO1Opt = "fe_compile_o1_opt";
constexpr const char* kComLevelO3Opt = "fe_compile_o3_opt";
constexpr const char* kForbiddenPass = "forbidden_close_pass";

constexpr const char* kDyInputsIndexes = "_dynamic_inputs_indexes";
constexpr const char* kDyOutputsIndexes = "_dynamic_outputs_indexes";
constexpr const char* kDyInputsAddNum = "_dynamic_inputs_add_num";
constexpr const char* kOpKernelAllInputSize = "_op_kernel_all_input_size";
constexpr const char* kOpKernelAllOutputSize = "_op_kernel_all_output_size";

const int32_t NCHW_DIM_N = 0;
const int32_t NCHW_DIM_C = 1;
const int32_t NCHW_DIM_H = 2;
const int32_t NCHW_DIM_W = 3;

const int32_t NHWC_DIM_N = 0;
const int32_t NHWC_DIM_H = 1;
const int32_t NHWC_DIM_W = 2;
const int32_t NHWC_DIM_C = 3;

const int32_t HWCN_DIM_H = 0;
const int32_t HWCN_DIM_W = 1;
const int32_t HWCN_DIM_C = 2;
const int32_t HWCN_DIM_N = 3;

const int32_t CHWN_DIM_C = 0;
const int32_t CHWN_DIM_H = 1;
const int32_t CHWN_DIM_W = 2;
const int32_t CHWN_DIM_N = 3;

const int32_t NDHWC_DIM_N = 0;
const int32_t NDHWC_DIM_D = 1;
const int32_t NDHWC_DIM_H = 2;
const int32_t NDHWC_DIM_W = 3;
const int32_t NDHWC_DIM_C = 4;

const int32_t NCDHW_DIM_N = 0;
const int32_t NCDHW_DIM_C = 1;
const int32_t NCDHW_DIM_D = 2;
const int32_t NCDHW_DIM_H = 3;
const int32_t NCDHW_DIM_W = 4;

const int32_t DHWCN_DIM_D = 0;
const int32_t DHWCN_DIM_H = 1;
const int32_t DHWCN_DIM_W = 2;
const int32_t DHWCN_DIM_C = 3;
const int32_t DHWCN_DIM_N = 4;

const int32_t DHWNC_DIM_D = 0;
const int32_t DHWNC_DIM_H = 1;
const int32_t DHWNC_DIM_W = 2;
const int32_t DHWNC_DIM_N = 3;
const int32_t DHWNC_DIM_C = 4;

const uint32_t NCHW_DIMENSION_NUM = 4;
constexpr const char* kLocalMemorySize = "local_memory_size";
}  // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_AICORE_UTIL_CONSTANTS_H_
