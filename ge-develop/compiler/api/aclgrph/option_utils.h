/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FRAMEWORK_DOMI_ATC_IR_COMMON_H_
#define FRAMEWORK_DOMI_ATC_IR_COMMON_H_

#include <unistd.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <set>

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/preprocess/multi_batch_options.h"
#include "framework/common/string_util.h"

namespace ge {
static std::set<std::string> caffe_support_input_format = {"NCHW", "ND"};
static std::set<std::string> tf_support_input_format = {"NCHW", "NHWC", "ND", "NCDHW", "NDHWC"};
static std::set<std::string> onnx_support_input_format = {"NCHW", "ND", "NCDHW"};

static std::map<std::string, domi::domiTensorFormat_t> input_format_str_to_geformat = {
    {"ND", domi::DOMI_TENSOR_ND},
    {"NCHW", domi::DOMI_TENSOR_NCHW},
    {"NHWC", domi::DOMI_TENSOR_NHWC},
    {"CHWN", domi::DOMI_TENSOR_CHWN},
    {"NC1HWC0", domi::DOMI_TENSOR_NC1HWC0},
    {"NHWC1C0", domi::DOMI_TENSOR_NHWC1C0},
    {"NCDHW", domi::DOMI_TENSOR_NCDHW},
    {"NDHWC", domi::DOMI_TENSOR_NDHWC}
};
static const std::string kEnableCompressWeightTrue = "1";
static const std::string kEnableCompressWeightFalse = "0";

static const std::unordered_set<std::string> kConstLifecycleOptions = {"graph", "session"};
static const std::unordered_set<std::string> kFeatureMapRefreshOptions = {"0", "1"};
static const std::unordered_set<std::string> kBoolOptions = {"", "true", "false"};
static const std::unordered_set<std::string> kStateOptions = {"", "0", "1"};
constexpr char const *kAutoFuseEnableOption = "--enable_autofuse";
constexpr char const *kSliceScheduleOption = "--experimental_enable_jit_executor_v2";

/*
* @brief 获取环境变量AUTOFUSE_FLAGS中对应option的值，比如AUTOFUSE_FLAGS="--enable_autofuse=true", 
* 那么调用GetAutofuseFlagValue("--enable_autofuse"),便会返回true字符串
* @param option 需要解析的option的名字
* @return 返回对应字符串的值
*/
std::string GetAutofuseFlagValue(const std::string &option);

bool CheckDynamicBatchSizeInputShapeValid(std::map<std::string, std::vector<int64_t>> shape_map,
                                          std::string &dynamic_batch_size);

bool CheckDynamicImagesizeInputShapeValid(std::map<std::string, std::vector<int64_t>> shape_map,
                                          const std::string &input_format, std::string &dynamic_image_size);

bool CheckDynamicDimsInputShapeValid(const std::map<std::string, std::vector<int64_t>> &shape_map,
                                     std::string &dynamic_dims);

bool CheckAndParseDynamicDims(int32_t dynamic_dim_num, std::string &dynamic_dims);

Status CheckInputShapeValueValid(const std::string &input_shape_value);

Status CheckHintShapeConflictWithDynamicParam(std::string &hint_shape, std::string &dynamic_batch_size,
                                              std::string &dynamic_image_size, std::string &dynamic_dims);

Status ParserShapeRangeByIndex(std::string &input_shape, std::string &input_shape_range);

Status CheckAndTransferInputShapeToRange(std::string &input_shape, std::string &input_shape_range,
                                         std::string &dynamic_batch_size, std::string &dynamic_image_size,
                                         std::string &dynamic_dims);
/*
* @brief 获取ge.inputHintShape中对应option的值, 并将其转化为GeShape
* @out_param option_shape 从option中解析的字符串转成成的Shape
* @return 成功返回GRAPH_SUCCESS, 失败返回FAILED
*/
Status ParseHintInputShape(std::vector<GeShape> &option_shape);

Status ParserShapeRangeByName(std::string &input_shape, std::string &input_shape_range);

Status CheckDynamicInputParamValid(std::string &dynamic_batch_size, std::string &dynamic_image_size,
                                   std::string &dynamic_dims, const std::string &input_shape,
                                   const std::string &input_shape_range, const std::string &input_format,
                                   bool &is_dynamic_input);

bool ParseInputShape(const std::string &input_shape, std::map<std::string, std::vector<int64_t>> &shape_map,
                     std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map,
                     bool is_dynamic_input = false);
Status ParseInputShapeRange(const std::string &shape_range,
                            std::map<std::string, std::vector<std::pair<int64_t, int64_t>>> &shape_range_map);
Status ParseInputShapeRange(const std::string &shape_range,
                            std::vector<std::vector<std::pair<int64_t, int64_t>>> &range);

Status CheckOutputTypeParamValid(const std::string &output_type);
Status CheckBufferOptimizeParamValid(const std::string &buffer_optimize);
Status CheckCompressWeightParamValid(const std::string &enable_compress_weight,
                                     const std::string &compress_weight_conf);
Status CheckSparseParamValid(const std::string &sparsity);
int32_t CheckLogParamValidAndSetLogLevel(const std::string &log);
Status CheckInsertOpConfParamValid(const std::string &insert_op_conf);
Status CheckDisableReuseMemoryParamValid(const std::string &disable_reuse_memory);
Status CheckPrecisionModeParamValid(const std::map<std::string, std::string> &options);
Status CheckPrecisionModeParamValid(const std::string &precision_mode);
Status CheckPrecisionModeV2ParamValid(const std::string &precision_mode_v2);
Status CheckPrecisionModeV2Conflict(const std::string &precision_mode, const std::string &precision_mode_v2);
Status CheckEnableSingleStreamParamValid(const std::string &enable_single_stream);
Status CheckExternalWeightParamValid(const std::string &enable_external_weight);
Status CheckIsWeightClipParamValid(const std::string &is_weight_clip);
Status CheckAcParallelEnableParamValid(const std::string &ac_parallel_enable);
Status CheckTilingScheduleOptimizeParamValid(const std::string &tiling_schedule_optimize);
Status CheckQuantDumpableParamValid(const std::string &quant_dumpable);
Status CheckAttrCompressionParamValid(const std::string &enable_attr_compression);
Status CheckImplmodeParamValid(const std::string &optypelist_for_implmode, std::string &op_select_implmode);
Status CheckModifyMixlistParamValid(const std::map<std::string, std::string> &options);
Status CheckModifyMixlistParamValid(const std::string &precision_mode, const std::string &precision_mode_v2,
                                    const std::string &modify_mixlist);
Status CheckAllowHF32ParamValid(const std::string &allow_hf32);
Status CheckInputFormat(const std::string &input_format);
Status CheckKeepTypeParamValid(const std::string &keep_dtype);
void PrintOptionMap(std::map<std::string, std::string> &options, std::string tips);
void EraseEndSemicolon(std::string &param);
Status UpdateDataOpShape(const OpDescPtr &op, std::map<std::string, std::vector<int64_t>> &shape_map);
Status UpdateDataOpShapeRange(
    const OpDescPtr &op, const std::map<std::string, std::vector<std::pair<int64_t, int64_t>>> &name_shape_range_map);
Status UpdateDataOpShapeRange(const OpDescPtr &op,
                              const std::vector<std::vector<std::pair<int64_t, int64_t>>> &index_shape_range_map);
Status UpdateDynamicInputShapeRange(const ge::ComputeGraphPtr &compute_graph, const std::string &input_shape_range);
void UpdateDataOpFormat(const OpDescPtr &op, const std::string &format);
Status CheckHostEnvOsAndHostEnvCpuValid(const std::string &host_env_os, const std::string &host_env_cpu);
void SetDefaultHostEnvOsAndHostEnvCpu(std::string &host_env_os, std::string &host_env_cpu);
Status CheckOptionValidValues(const std::map<std::string, std::string> &options, const std::string &key,
                              const std::unordered_set<std::string> &valid_values);
Status CheckOptionValidThreshold(const std::map<std::string, std::string> &options, const std::string &str);
Status CheckValidValueRange(const std::string &key, const std::string &value, const int64_t min, const int64_t max);
Status CheckIoReuseMemIndexesOption(const ComputeGraphPtr &compute_graph,
                                    const std::map<std::string, std::string> &options);
Status CheckScreenPrinterOption(const std::map<std::string, std::string> &options);
Status CheckOptimizationOptionValid(const std::map<std::string, std::string> &options);
Status CheckOutputReuseMemIndexesOption(const std::map<std::string, std::string> &options,
                                        bool &has_output_set_reuse_mem);
Status CheckInputReuseMemIndexesOption(const std::map<std::string, std::string> &options,
                                       bool &has_input_set_reuse_mem);
Status CheckOutputReuseInputMemIndexesOption(const ComputeGraphPtr &compute_graph,
                                             const std::map<std::string, std::string> &options);
bool ParseSingleShapeRange(std::string &shape_range, std::vector<std::pair<int64_t, int64_t>> &shape_range_vec);
std::string SupportedHostEnvCpuList(std::unordered_set<std::string> &supported_os_cpu);
std::string SupportedHostEnvOsList(std::unordered_map<std::string, std::unordered_set<std::string>>
                                   &opp_supported_os_cpu);
bool EnableSliceSchedule();
}
#endif  // FRAMEWORK_DOMI_ATC_IR_COMMON_H_
