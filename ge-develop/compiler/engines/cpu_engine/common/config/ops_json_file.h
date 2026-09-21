/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_OPS_JSON_FILE_H_
#define AICPU_OPS_JSON_FILE_H_

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "error_code/error_code.h"
#include "aicpu_ops_kernel_info_store/op_struct.h"

namespace {
// op kernel json file configuration item: op name
const std::string kKernelConfigOpName = "opName";

// op kernel json file configuration item: TFKernel
const std::string kKernelConfigTfKernel = "TFKernel";

// op kernel json file configuration item: engine information that the op
const std::string kKernelConfigOpInfo = "opInfo";

// op kernel json file configuration item: kernel lib name
const std::string kKernelConfigLibName = "libName";

// op kernel json file configuration item: which engin
const std::string kKernelConfigEngine = "engine";

// op kernel json file configuration item: which opsKernelStore
const std::string kKernelConfigKernelLib = "opKernelLib";

// op kernel json file configuration item: compute cost
const std::string kKernelConfigComputeCost = "computeCost";

// op kernel json file configuration item: ops flag
const std::string kKernelConfigOpsFlag = "opsFlag";

// op kernel json file configuration item: ops format agnostic
const std::string kKernelConfigFormatAgnostic = "formatAgnostic";

// op kernel json file configuration item: shape type
const std::string kKernelConfigShapeType = "subTypeOfInferShape";

// op kernel json file configuration item: workspace size
const std::string kKernelConfigWorkspaceSize = "workspaceSize";

// op kernel json file configuration item: whether to support is related to
// shape
const std::string kKernelConfigFlagPartial = "flagPartial";

// op kernel json file configuration item: whether to support asynchronous
const std::string kKernelConfigFlagAsync = "flagAsync";

// op kernel json file configuration item: whether to support no tiling
const std::string kKernelConfigNoTiling = "noTiling";

// op kernel json file configuration item: whether user defined
const std::string kKernelConfigUserDefined = "userDefined";

// op kernel json file configuration item: kernel so name
const std::string kKernelConfigKernelSo = "kernelSo";

// op kernel json file configuration item: kernel slice pattern
const std::string kKernelConfigSlicePattern = "slicePattern";

// op kernel json file configuration item: function name
const std::string kKernelConfigFunctionName = "functionName";

// op kernel json file configuration item: input output format
const std::string kKernelConfigFormat = "format";

// op kernel json file configuration item: input output DataType
const std::string kKernelConfigDataType = "type";

// op kernel json file configuration item: input output src DataType for AutoCast
const std::string kKernelConfigSrcType = "srcAutoCastType";

// op kernel json file configuration item: input output dst DataType for AutoCast
const std::string kKernelConfigDstType = "dstAutoCastType";

// op kernel json file configuration item: input output name
const std::string kKernelConfigName = "name";

// op kernel json file configuration item: all op infos
const std::string kKernelConfigOpInfos = "opInfos";

// ops_flag config
const std::string kOpsFlagOpen = "1";

// ops_flag config
const std::string kOpsFlagClose = "0";

// ops_flag config
const std::string kOpsFlagEmpty = "";

// op kernel json file configuration item: topic type
const std::string kKernelConfigTopicType = "topicType";

const std::string kTopicTypeDeviceOnly = "DEVICE_ONLY";

const std::string kTopicTypeDeviceFirst = "DEVICE_FIRST";

const std::string kTopicTypeHostOnly = "HOST_ONLY";

const std::string kTopicTypeHostFirst = "HOST_FIRST";

// op kernel json file configuration item: resource
const std::string kKernelConfigResource = "resource";

const std::string kResourceQueue = "RES_QUEUE";

const std::string kResourceChannel = "RES_CHANNEL";

const std::string kResourceVdecChannel = "RES_VDEC_CHANNEL";

// op kernel json file configuration item: flag SupportBlockDim
const std::string kKernelConfigSupportBlockDim = "flagSupportBlockDim";

// op kernel json file configuration item: blockDimByIndex
const std::string kKernelConfigBlockDimByIndex = "blockDimByIndex";

// op kernel json file configuration item: implementType
const std::string kKernelConfigImplementType = "implementType";
}  // namespace

namespace aicpu {
struct InOutInfo {
  nlohmann::json in_output_format;
  nlohmann::json in_output_type;
  nlohmann::json in_output_real_name;
  nlohmann::json in_output_src_type;
  nlohmann::json in_output_dst_type;
};
class OpsJsonFile {
 public:
  /**
   * Get instance
   * @return OpsJsonFile reference
   */
  static OpsJsonFile &Instance();

  /**
   * Destructor
   */
  virtual ~OpsJsonFile() = default;

  /**
   * Read json file in specified path(based on source file's current path)
   * @param file_path json file path
   * @param json_read read json handle
   * @return whether read file successfully
   */
  aicpu::State ParseUnderPath(const std::string &file_path, nlohmann::json &json_read) const;

  // Copy prohibit
  OpsJsonFile(const OpsJsonFile &) = delete;
  // Copy prohibit
  OpsJsonFile &operator=(const OpsJsonFile &) = delete;
  // Move prohibit
  OpsJsonFile(OpsJsonFile &&) = delete;
  // Move prohibit
  OpsJsonFile &operator=(OpsJsonFile &&) = delete;

 private:
  // Constructor
  OpsJsonFile() = default;

  /**
   * convert json file format
   * @param json_read Original format json
   * @return whether convert format successfully
   */
  bool ConvertJsonFormat(nlohmann::json &json_read) const;

  /**
   * convert topic type from string to enum
   * @param topic_type_str origin topic type
   * @param topic_type output topic type
   */
  void ConvertTopicType(const std::string &topic_type_str,
                        FWKAdapter::FWKExtTopicType &topic_type) const;

  /**
   * parse input output format from json read from tf_kernel.json
   * @param json_read Original format json
   * @param format_json input output format json
   * @param type_json input output type json
   * @param name_json input output name json
   * @return whether read file successfully
   */
  bool ParseInputOutput(const nlohmann::json &json_read, InOutInfo &in_out_info) const;

  /**
   * Check and get compute_cost field from tf_kernel.json
   * @param buff json object
   * @param op_name operator name string
   * @param compute_cost compute cost value
   * @return whether parse compute_cost successfully
   */
  bool CheckAndGetComputeCost(const nlohmann::json &buff,
                              const std::string &op_name, int &compute_cost) const;

  /**
   * Check and get required boolean field from tf_kernel.json
   * @param buff json object
   * @param op_name operator name string
   * @param field_str json field string
   * @param value parsed bool value
   * @param use_default if buff is empty and use_default is true, value = false, if buff is empty and use_default is
   * false, parse bool value failed, if buff is not empty, use_default is unused.
   * @return whether parse bool value successfully
   */
  bool CheckAndGetBoolValue(const nlohmann::json &buff, const std::string &op_name, const std::string &field_str,
                            bool &value, bool use_default = false) const;

  /**
   * Check and get required format agnostic from tf_kernel.json
   * @param buff json object
   * @param op_name operator name string
   * @param field_str json field string
   * @param value parsed bool value
   * @return whether parse bool value successfully
   */
  bool CheckAndGetFormatAgnostic(const nlohmann::json &buff,
                                 const std::string &op_name,
                                 const std::string &field_str, bool &value) const;

  /**
   * Check and get shape type field from kernel info
   * @param buff json object
   * @param op_name operator name string
   * @param shape_type shape type for op
   * @return whether parse shape_type successfully
   */
  bool CheckAndGetShapeType(const nlohmann::json &buff,
                            const std::string &op_name, int &shape_type) const;

  /**
   * Check and get nonessential boolean field from tf_kernel.json
   * @param buff json object
   * @param op_name operator name string
   * @param field_str json field string
   * @param value parsed bool value
   * @return whether parse bool value successfully
   */
  bool CheckAndGetNonessentialBoolValue(const nlohmann::json &buff,
                                        const std::string &op_name,
                                        const std::string &field_str,
                                        bool &value) const;

  /**
   * Check whether read data from tf_kernel.json successfully
   * @param buff json object
   * @param op_name operator name string
   * @param ops_flag op flag
   * @return whether get bool value successfully
   */
  bool CheckAndGetOpsFlag(const nlohmann::json &buff, const std::string &op_name,
                          std::string &ops_flag) const;

  /**
   * Check and get workspace size from kernel info
   * @param buff json object
   * @param op_name operator name string
   * @param workspace_size shape type for op
   * @return whether check workspace_size successfully
   */
  bool CheckAndGetWorkspaceSize(const nlohmann::json &buff,
                                const std::string &op_name, int &workspace_size) const;

  /**
   * Check and get topic type from kernel info
   * @param buff json object
   * @param op_name operator name string
   * @param topic_type output topic type
   * @return whether check topic_type successfully
   */
  bool CheckAndGetTopicType(const nlohmann::json &buff,
                            const std::string &op_name,
                            FWKAdapter::FWKExtTopicType &topic_type) const;

  bool CheckAndGetSlicePattern(const nlohmann::json &buff,
                               const std::string &op_name,
                               std::string &slice_pattern) const;

  /**
   * Check and get resource from kernel info
   * @param buff json object
   * @param op_name operator name string
   * @param resource output resource
   * @return whether check resource successfully
   */
  bool CheckAndGetResource(const nlohmann::json &buff,
                           const std::string &op_name,
                           std::string &resource) const;

  /**
   * Check and get blockIndex from kernel info
   * @param buff json object
   * @param op_name operator name string
   * @param blockIndex output blockIndex
   * @return whether check blockIndex successfully
   */
  bool CheckAndGetBlockDimByIndex(const nlohmann::json &buff,
                                  const std::string &op_name, int &blockIndex) const;

  /**
   * Check and get implement type from kernel info
   * @param buff json object
   * @param op_name operator name string
   * @param implement_type output implement_type
   * @return whether check implement_type successfully
   */
  bool CheckAndGetImplementType(const nlohmann::json &buff,
                                const std::string &op_name, int &implement_type) const;
};
                       
/**
 * OpInfoDescs json to struct object function
 * @param json_read read json handle
 * @param infos all op infos
 * @return whether read file successfully
 */
void from_json(const nlohmann::json &json_read, OpInfoDescs &infos);

void from_json(const nlohmann::json &json_read, OpInfoDesc &desc);

/**
 * OpInfo json to struct object function
 * @param json_read read json handle
 * @param op_info engine information that the op
 * @return whether read file successfully
 */
void from_json(const nlohmann::json &json_read, OpFullInfo &op_info);

}  // namespace aicpu

#endif
