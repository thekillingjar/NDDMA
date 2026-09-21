/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_json_file.h"

#include "graph/types.h"
#include "util/log.h"
#include "util/util.h"

using namespace std;
using namespace nlohmann;
using namespace ::aicpu::FWKAdapter;

namespace {
const int kMaxWorkspaceSize = 1 * 500 * 1024; // for 500KB
const int DEFAULT_WROKSPACE_SIZE = 2048; // 2048 is for 2KB
const int DEFAULT_DEVICE_IMPLEMENT_TYPE = 2;
}

namespace aicpu {
OpsJsonFile &OpsJsonFile::Instance() {
  static OpsJsonFile instance;
  return instance;
}

aicpu::State OpsJsonFile::ParseUnderPath(const string &file_path, json &json_read) const {
  aicpu::State ret = ReadJsonFile(file_path, json_read);
  if (ret.state != ge::SUCCESS) {
    AICPUE_LOGW("Read kernel json file failed, file_path[%s].", file_path.c_str());
    return aicpu::State(ge::SUCCESS);
  }
  // inner error
  return ConvertJsonFormat(json_read) ? aicpu::State(ge::SUCCESS) : aicpu::State(ge::FAILED);
}

bool OpsJsonFile::ConvertJsonFormat(json &json_read) const {
  json op_infos = json::array();
  json json_null;
  for (auto it = json_read.cbegin(); it != json_read.cend(); ++it) {
    json new_json = it.value();
    string op_name = it.key();
    new_json[kKernelConfigOpName] = op_name;

    struct InOutInfo info;
    bool ret = ParseInputOutput(it.value(), info);

    AICPU_IF_BOOL_EXEC((!ret),
                       AICPU_REPORT_INNER_ERR_MSG("Call OpsJsonFile::ParseInputOutput failed, op[%s].", op_name.c_str());
                       return false)
    new_json[kKernelConfigOpInfo][kKernelConfigFormat] = info.in_output_format;
    new_json[kKernelConfigOpInfo][kKernelConfigDataType] = info.in_output_type;
    new_json[kKernelConfigOpInfo][kKernelConfigName] = info.in_output_real_name;
    new_json[kKernelConfigOpInfo][kKernelConfigSrcType] = info.in_output_src_type;
    new_json[kKernelConfigOpInfo][kKernelConfigDstType] = info.in_output_dst_type;

    // compute cost
    int compute_cost = 0;
    auto buff = new_json[kKernelConfigOpInfo][kKernelConfigComputeCost];
    AICPU_IF_BOOL_EXEC((!CheckAndGetComputeCost(buff, op_name, compute_cost)), return false);
    new_json[kKernelConfigOpInfo][kKernelConfigComputeCost] = compute_cost;

    // flag async
    bool flag_async = false;
    buff = new_json[kKernelConfigOpInfo][kKernelConfigFlagAsync];
    AICPU_IF_BOOL_EXEC((!CheckAndGetBoolValue(buff, op_name, "flagAsync", flag_async)), return false);
    new_json[kKernelConfigOpInfo][kKernelConfigFlagAsync] = flag_async;

    // no tiling
    bool no_tiling = false;
    buff = new_json[kKernelConfigOpInfo][kKernelConfigNoTiling];
    AICPU_IF_BOOL_EXEC((!CheckAndGetBoolValue(buff, op_name, "noTiling", no_tiling, true)), return false);
    new_json[kKernelConfigOpInfo][kKernelConfigNoTiling] = no_tiling;

    // flag partial
    bool flag_partial = false;
    buff = new_json[kKernelConfigOpInfo][kKernelConfigFlagPartial];
    AICPU_IF_BOOL_EXEC((!CheckAndGetBoolValue(buff, op_name, "flagPartial", flag_partial)), return false);
    new_json[kKernelConfigOpInfo][kKernelConfigFlagPartial] = flag_partial;

    // format agnostic
    bool format_agnostic = false;
    buff = new_json[kKernelConfigOpInfo][kKernelConfigFormatAgnostic];
    AICPU_IF_BOOL_EXEC((!CheckAndGetFormatAgnostic(buff, op_name, "formatAgnostic", format_agnostic)), return false);
    new_json[kKernelConfigOpInfo][kKernelConfigFormatAgnostic] = format_agnostic;

    // ops flag
    string ops_flag;
    buff = new_json[kKernelConfigOpInfo][kKernelConfigOpsFlag];
    AICPU_IF_BOOL_EXEC((!CheckAndGetOpsFlag(buff, op_name, ops_flag)), return false);
    new_json[kKernelConfigOpInfo][kKernelConfigOpsFlag] = ops_flag;

    // shape type
    int shape_type = 0;
    buff = new_json[kKernelConfigOpInfo][kKernelConfigShapeType];
    AICPU_IF_BOOL_EXEC((!CheckAndGetShapeType(buff, op_name, shape_type)), return false);
    new_json[kKernelConfigOpInfo][kKernelConfigShapeType] = shape_type;

    // workspace
    int workspace_size = 0;
    buff = new_json[kKernelConfigOpInfo][kKernelConfigWorkspaceSize];
    AICPU_IF_BOOL_EXEC((!CheckAndGetWorkspaceSize(buff, op_name, workspace_size)), return false);
    new_json[kKernelConfigOpInfo][kKernelConfigWorkspaceSize] = workspace_size;

    // user defined: AICPUKernel only
    bool user_defined = false;
    buff = new_json[kKernelConfigOpInfo][kKernelConfigUserDefined];
    AICPU_IF_BOOL_EXEC((!CheckAndGetNonessentialBoolValue(buff, op_name, "userDefined", user_defined)), return false);
    new_json[kKernelConfigOpInfo][kKernelConfigUserDefined] = user_defined;

    // function name: AICPUKernel only
    buff = new_json[kKernelConfigOpInfo][kKernelConfigFunctionName];
    AICPU_IF_BOOL_EXEC(
        (buff == json_null),
        new_json[kKernelConfigOpInfo][kKernelConfigFunctionName] = "";);

    // kernel so: AICPUKernel only
    buff = new_json[kKernelConfigOpInfo][kKernelConfigKernelSo];
    AICPU_IF_BOOL_EXEC(
        (buff == json_null),
        new_json[kKernelConfigOpInfo][kKernelConfigKernelSo] = "";);

    // ops topic type
    FWKAdapter::FWKExtTopicType topic_type;
    buff = new_json[kKernelConfigOpInfo][kKernelConfigTopicType];
    AICPU_IF_BOOL_EXEC((!CheckAndGetTopicType(buff, op_name, topic_type)), return false);
    new_json[kKernelConfigOpInfo][kKernelConfigTopicType] = topic_type;

    // slice pattern
    string slicePattern = "";
    buff = new_json[kKernelConfigOpInfo][kKernelConfigSlicePattern];
    AICPU_IF_BOOL_EXEC(!CheckAndGetSlicePattern(buff, op_name, slicePattern), return false);
    new_json[kKernelConfigOpInfo][kKernelConfigSlicePattern] = slicePattern;

    // ops resource
    std::string resource;
    buff = new_json[kKernelConfigOpInfo][kKernelConfigResource];
    AICPU_IF_BOOL_EXEC((!CheckAndGetResource(buff, op_name, resource)), return false);
    new_json[kKernelConfigOpInfo][kKernelConfigResource] = resource;

    // ops support blockdim flag
    bool flag_support_block_dim = false;
    buff = new_json[kKernelConfigOpInfo][kKernelConfigSupportBlockDim];
    AICPU_IF_BOOL_EXEC((!CheckAndGetBoolValue(buff, op_name, "flagSupportBlockDim", flag_support_block_dim, true)),
                       return true);
    new_json[kKernelConfigOpInfo][kKernelConfigSupportBlockDim] = flag_support_block_dim;

    // ops blockDim By Index
    int blockDimByIndex = -1;
    buff = new_json[kKernelConfigOpInfo][kKernelConfigBlockDimByIndex];
    AICPU_IF_BOOL_EXEC((!CheckAndGetBlockDimByIndex(buff, op_name, blockDimByIndex)), return true);
    new_json[kKernelConfigOpInfo][kKernelConfigBlockDimByIndex] = blockDimByIndex;

    // ops implementType default value represents device
    int implement_type = DEFAULT_DEVICE_IMPLEMENT_TYPE;
    buff = new_json[kKernelConfigOpInfo][kKernelConfigImplementType];
    AICPU_IF_BOOL_EXEC((!CheckAndGetImplementType(buff, op_name, implement_type)), return true);
    new_json[kKernelConfigOpInfo][kKernelConfigImplementType] = implement_type;

    op_infos.push_back(new_json);
  }

  json_read = {};
  json_read[kKernelConfigLibName] = kKernelConfigTfKernel;
  AICPU_IF_BOOL_EXEC(!op_infos.empty(),
                     json_read[kKernelConfigLibName] = op_infos[0][kKernelConfigOpInfo][kKernelConfigKernelLib];);
  json_read[kKernelConfigOpInfos] = op_infos;

  std::string kernel_lib = json_read[kKernelConfigLibName];
  AICPUE_LOGI("Convert json success, kernel config is %s.", kernel_lib.c_str());
  return true;
}

bool OpsJsonFile::ParseInputOutput(const json &json_read, InOutInfo &in_out_info) const {
  json format_result;
  json type_result;
  json name_result;
  json src_type_result;
  json dst_type_result;
  const string input_str = "input";
  const string output_str = "output";
  const string dynamic_input_str = "dynamic_input";
  const string dynamic_output_str = "dynamic_output";
  for (json::const_iterator iter = json_read.cbegin(); iter != json_read.cend();
       ++iter) {
    const string key = iter.key();
    // json_read maybe include key inputn or outputn (n is a number satisfied n
    // >=0,such as input0, output2,...) this judgement ensures that the next
    // std::strncmp() can be executed correctly
    if (key.size() < output_str.size()) {
      continue;
    }

    if ((strncmp(key.c_str(), input_str.c_str(), input_str.size()) == 0) ||
        (strncmp(key.c_str(), output_str.c_str(), output_str.size()) == 0) ||
        (strncmp(key.c_str(), dynamic_input_str.c_str(),
                 dynamic_input_str.size()) == 0) ||
        (strncmp(key.c_str(), dynamic_output_str.c_str(),
                 dynamic_output_str.size()) == 0)) {
      json json_key = json_read[key];
      auto iter_format = json_key.find(kKernelConfigFormat);
      if (iter_format != json_key.end()) {
        string format = iter_format.value().get<string>();
        AICPU_IF_BOOL_EXEC(
            (format != "ND") && (format != "NHWC") && (format != "NCHW"),
            AICPU_REPORT_INNER_ERR_MSG(
              "Invalid format[%s], should be ND, NHWC or NCHW.", format.c_str());
            return false)
        format_result[key] = format;
      }
      auto iter_type = json_key.find(kKernelConfigDataType);
      if (iter_type != json_key.end()) {
        string type = iter_type.value().get<string>();
        type_result[key] = type;
      }
      auto iter_name = json_key.find(kKernelConfigName);
      if (iter_name != json_key.end()) {
        string name = iter_name.value().get<string>();
        name_result[key] = name;
      }
      auto iter_src_type = json_key.find(kKernelConfigSrcType);
      if (iter_src_type != json_key.end()) {
        string src_type = iter_src_type.value().get<string>();
        src_type_result[key] = src_type;
      }
      auto iter_dst_type = json_key.find(kKernelConfigDstType);
      if (iter_dst_type != json_key.end()) {
        string dst_type = iter_dst_type.value().get<string>();
        dst_type_result[key] = dst_type;
      }
    }
  }
  in_out_info.in_output_format = format_result;
  in_out_info.in_output_type = type_result;
  in_out_info.in_output_real_name = name_result;
  in_out_info.in_output_src_type = src_type_result;
  in_out_info.in_output_dst_type = dst_type_result;
  return true;
}

bool OpsJsonFile::CheckAndGetComputeCost(const json &buff, const string &op_name,
                                         int &compute_cost) const {
  if (!buff.empty()) {
    aicpu::State state = StringToNum(buff.get<string>(), compute_cost);
    if (state.state != ge::SUCCESS) {
        AICPU_REPORT_INNER_ERR_MSG("Convert %s[%s] to int for op[%s] failed, %s.",
            buff.get<string>().c_str(), kKernelConfigComputeCost.c_str(),
            op_name.c_str(), state.msg.c_str());
        return false;
    }
    return true;
  } else {
    AICPU_REPORT_INNER_ERR_MSG("[%s] is empty, op[%s].",
        kKernelConfigComputeCost.c_str(), op_name.c_str());
    return false;
  }
}

bool OpsJsonFile::CheckAndGetOpsFlag(const json &buff, const string &op_name,
                                     string &ops_flag) const {
  if (!buff.empty()) {
    ops_flag = buff.get<string>();
    int is_open = ops_flag.compare("OPS_FLAG_OPEN");
    int is_close = ops_flag.compare("OPS_FLAG_CLOSE");
    if ((is_open != 0) && (is_close != 0)) {
      AICPU_REPORT_INNER_ERR_MSG("Invalid op_info.opsFlag[%s], should be "
          "OPS_FLAG_OPEN or OPS_FLAG_CLOSE, op[%s].",
          ops_flag.c_str(), op_name.c_str());
      return false;
    }
    if (is_open == 0) {
      ops_flag = kOpsFlagOpen;
    } else {
      ops_flag = kOpsFlagClose;
    }
  } else {
    AICPUE_LOGI("Read kernel json file, op[%s], op_info.opsFlag is empty.",
                op_name.c_str());
    ops_flag = kOpsFlagEmpty;
  }
  return true;
}

void OpsJsonFile::ConvertTopicType(const std::string &topic_type_str,
                                   FWKExtTopicType &topic_type) const {
  topic_type = FWK_ADPT_TOPIC_INVALID;
  if (topic_type_str == kTopicTypeDeviceOnly) {
    topic_type = FWK_ADPT_TOPIC_DEVICE_ONLY;
  } else if (topic_type_str == kTopicTypeDeviceFirst) {
    topic_type = FWK_ADPT_TOPIC_DEVICE_FIRST;
  } else if (topic_type_str == kTopicTypeHostOnly) {
    topic_type = FWK_ADPT_TOPIC_HOST_ONLY;
  } else if (topic_type_str == kTopicTypeHostFirst) {
    topic_type = FWK_ADPT_TOPIC_HOST_FIRST;
  }
}

bool OpsJsonFile::CheckAndGetTopicType(const nlohmann::json &buff,
                                       const std::string &op_name,
                                       FWKExtTopicType &topic_type) const {
  if (!buff.empty()) {
    ConvertTopicType(buff.get<string>(), topic_type);
    AICPUE_LOGI("Read kernel json file, opName:[%s], topicType:[%d].", op_name.c_str(), static_cast<int>(topic_type));
    if (topic_type == FWK_ADPT_TOPIC_INVALID) {
      AICPU_REPORT_INNER_ERR_MSG("Read kernel json file, opName:[%s], topicType:[%d] is invalid, "
          "only support [DEVICE_ONLY/DEVICE_FIRST/HOST_ONLY/HOST_FIRST].",
          op_name.c_str(), static_cast<int>(topic_type));
      return false;
    }
    return true;
  }
  topic_type = FWK_ADPT_TOPIC_DEVICE_ONLY;

  return true;
}

bool OpsJsonFile::CheckAndGetResource(const nlohmann::json &buff,
                                      const std::string &op_name,
                                      std::string &resource) const {
  if (!buff.empty()) {
    resource = buff.get<string>();
    std::vector<std::string> resourceList = SplitString(resource, ",");
    for (size_t i = 0; i < resourceList.size(); i++) {
      if ((resourceList[i] != kResourceQueue) && (resourceList[i] != kResourceChannel) &&
          (resourceList[i] != kResourceVdecChannel)) {
        AICPU_REPORT_INNER_ERR_MSG("Read kernel json file, opName:[%s], resource:[%s] is invalid, "
            "only support [%s/%s/%s].",
            op_name.c_str(), resource.c_str(), kResourceQueue.c_str(),
            kResourceChannel.c_str(), kResourceVdecChannel.c_str());
        return false;
      }
    }
  }
  return true;
}

bool OpsJsonFile::CheckAndGetBoolValue(const json &buff, const string &op_name, const string &field_str, bool &value,
                                       bool use_default) const {
  if (!buff.empty()) {
    aicpu::State state = StringToBool(buff.get<string>(), value);
    if (state.state == ge::SUCCESS) {
      AICPUE_LOGD("op_info.%s is [%d], op[%s].", field_str.c_str(), value, op_name.c_str());
      return true;
    } else {
      AICPU_REPORT_INNER_ERR_MSG("invalid op_info.%s[%s], should be False or True, op[%s].", field_str.c_str(),
                               buff.get<string>().c_str(), op_name.c_str());
      return false;
    }
  } else if (use_default) {
    AICPUE_LOGD("op_info.%s is empty, use default value[%d], op[%s].", field_str.c_str(), value, op_name.c_str());
    return true;
  } else {
    AICPU_REPORT_INNER_ERR_MSG("op_info.%s is empty, op[%s].",
        field_str.c_str(), op_name.c_str());
    return false;
  }
}

// check and get shape type field from kernel info
bool OpsJsonFile::CheckAndGetShapeType(const json &buff, const string &op_name,
                                       int &shape_type) const {
  if (!buff.empty()) {
    aicpu::State state = StringToNum(buff.get<string>(), shape_type);
    if (state.state != ge::SUCCESS) {
        AICPU_REPORT_INNER_ERR_MSG("Convert %s[%s] to int for op[%s] failed, %s.",
            buff.get<string>().c_str(), kKernelConfigShapeType.c_str(),
            op_name.c_str(), state.msg.c_str());
        return false;
    }
    if ((shape_type < static_cast<int>(ge::DEPEND_IN_SHAPE)) ||
        (shape_type > static_cast<int>(ge::DEPEND_COMPUTE))) {
      AICPU_REPORT_INNER_ERR_MSG(
          "invalid shape type[%d], should be in[%d, %d], op[%s].", shape_type,
          static_cast<int>(ge::DEPEND_IN_SHAPE),
          static_cast<int>(ge::DEPEND_COMPUTE), op_name.c_str());
      return false;
    }
  } else {
    AICPUE_LOGD(
        "Read kernel json file, op[%s], op_info.shapeType is empty, use "
        "default.",
        op_name.c_str());
    shape_type = static_cast<int>(ge::DEPEND_IN_SHAPE);
  }
  return true;
}

bool OpsJsonFile::CheckAndGetNonessentialBoolValue(const json &buff,
                                                   const string &op_name,
                                                   const string &field_str,
                                                   bool &value) const {
  if (!buff.empty()) {
    if (StringToBool(buff.get<string>(), value).state != ge::SUCCESS) {
      AICPU_REPORT_INNER_ERR_MSG(
          "invalid op_info.%s[%s], should be False or True, op[%s].",
          field_str.c_str(), buff.get<string>().c_str(), op_name.c_str());
      return false;
    }
  }
  return true;
}

bool OpsJsonFile::CheckAndGetFormatAgnostic(const json &buff,
                                            const string &op_name,
                                            const string &field_str,
                                            bool &value) const {
  if (!buff.empty()) {
    aicpu::State state = StringToBool(buff.get<string>(), value);
    if (state.state == ge::SUCCESS) {
      return true;
    } else {
      AICPU_REPORT_INNER_ERR_MSG("Call StringToBool failed, %s. [%s] must be False"
          " or True in op info store file, op[%s]",
          state.msg.c_str(), field_str.c_str(), op_name.c_str());
      return false;
    }
  } else {
    AICPUE_LOGD("Read kernel json file, op[%s], op_info.%s is empty.",
                op_name.c_str(), field_str.c_str());
    return true;
  }
}

// check and get workspace size from kernel info
bool OpsJsonFile::CheckAndGetWorkspaceSize(const nlohmann::json &buff,
                                           const std::string &op_name,
                                           int &workspace_size) const {
  if (!buff.empty()) {
    aicpu::State state = StringToNum(buff.get<string>(), workspace_size);
    if (state.state != ge::SUCCESS) {
        AICPU_REPORT_INNER_ERR_MSG("Convert %s[%s] to int for op[%s] failed, %s.",
            kKernelConfigWorkspaceSize.c_str(), buff.get<string>().c_str(),
            op_name.c_str(), state.msg.c_str());
        return false;
    }
    workspace_size *= 1024; // 1024 is for kb
    if (workspace_size < 0) {
      AICPU_REPORT_INNER_ERR_MSG("invalid %s[%d] should be in (0, %d). op[%s]",
          kKernelConfigWorkspaceSize.c_str(), workspace_size, kMaxWorkspaceSize,
          op_name.c_str());
      return false;
    } else if (workspace_size > kMaxWorkspaceSize) {
      AICPUE_LOGW("workspaceSize is morn than 500kb and set it to 500kb");
      workspace_size = kMaxWorkspaceSize;
    }
  } else {
    AICPUE_LOGD(
        "Read kernel json file, op[%s], op_info.workspaceSize is empty and use default value[2KB]",
        op_name.c_str());
    workspace_size = DEFAULT_WROKSPACE_SIZE;
  }
  return true;
}

bool OpsJsonFile::CheckAndGetBlockDimByIndex(const json &buff, const string &op_name,
                                             int &blockIndex) const {
  if (!buff.empty()) {
    aicpu::State state = StringToNum(buff.get<string>(), blockIndex);
    if (state.state != ge::SUCCESS) {
        AICPU_REPORT_INNER_ERR_MSG("Convert %s[%s] to int for op[%s] failed, %s.",
            buff.get<string>().c_str(), kKernelConfigBlockDimByIndex.c_str(),
            op_name.c_str(), state.msg.c_str());
        return false;
    }
  } else {
      AICPUE_LOGD("[%s] is empty, op[%s].",
                  kKernelConfigBlockDimByIndex.c_str(), op_name.c_str());
  }
  return true;
}

bool OpsJsonFile::CheckAndGetImplementType(const json &buff, const string &op_name,
                                           int &implement_type) const {
  if (!buff.empty()) {
    aicpu::State state = StringToNum(buff.get<string>(), implement_type);
    if (state.state != ge::SUCCESS) {
        AICPU_REPORT_INNER_ERR_MSG("Convert %s[%s] to int for op[%s] failed, %s.",
            buff.get<string>().c_str(), kKernelConfigImplementType.c_str(),
            op_name.c_str(), state.msg.c_str());
        return false;
    }
  } else {
      AICPUE_LOGD("[%s] is empty, op[%s].", kKernelConfigImplementType.c_str(), op_name.c_str());
  }
  return true;
}

bool OpsJsonFile::CheckAndGetSlicePattern(const nlohmann::json &buff,
                                          const std::string &op_name,
                                          std::string &slice_pattern) const {
  static const unordered_set<string> patternSet = {"elemwise",
                                                   "elemwiseBroadcast",
                                                   "broadcast",
                                                   "slidingWindow",
                                                   "slidingWindowDeconv",
                                                   "cubeMatmul",
                                                   "reduce",
                                                   "resize",
                                                   "scatter",
                                                   "segment"};
  if (!buff.empty()) {
    slice_pattern = buff.get<string>();
    if (patternSet.count(slice_pattern) == 0) {
      AICPU_REPORT_INNER_ERR_MSG(
          "Read kernel json file, opName:[%s], slice_pattern:[%s] is invalid.",
          op_name.c_str(), slice_pattern.c_str());
      return false;
    }
  }

  return true;
}

template <typename T>
inline void Assignment(T &varible, const string &key, const json &json_read) {
  auto iter = json_read.find(key);
  if (iter != json_read.end()) {
    varible = iter.value().get<T>();
  }
}

void from_json(const json &json_read, OpInfoDescs &infos) {
  Assignment(infos.opInfos, kKernelConfigOpInfos, json_read);
  Assignment(infos.libName, kKernelConfigLibName, json_read);
}

void from_json(const json &json_read, OpFullInfo &op_info) {
  Assignment(op_info.engine, kKernelConfigEngine, json_read);

  Assignment(op_info.opKernelLib, kKernelConfigKernelLib, json_read);

  Assignment(op_info.computeCost, kKernelConfigComputeCost, json_read);

  Assignment(op_info.flagPartial, kKernelConfigFlagPartial, json_read);

  Assignment(op_info.flagAsync, kKernelConfigFlagAsync, json_read);

  Assignment(op_info.noTiling, kKernelConfigNoTiling, json_read);

  op_info.workspaceSize = 0;
  Assignment(op_info.workspaceSize, kKernelConfigWorkspaceSize, json_read);

  // param[userDefined] only in aicpu_kernel.json
  op_info.userDefined = false;
  Assignment(op_info.userDefined, kKernelConfigUserDefined, json_read);

  // param[kernelSo] only in aicpu_kernel.json
  op_info.kernelSo = "";
  Assignment(op_info.kernelSo, kKernelConfigKernelSo, json_read);

  // param[functionName] only in aicpu_kernel.json
  op_info.functionName = "";
  Assignment(op_info.functionName, kKernelConfigFunctionName, json_read);

  Assignment(op_info.formatAgnostic, kKernelConfigFormatAgnostic, json_read);

  Assignment(op_info.opsFlag, kKernelConfigOpsFlag, json_read);

  Assignment(op_info.shapeType, kKernelConfigShapeType, json_read);

  Assignment(op_info.topicType, kKernelConfigTopicType, json_read);

  Assignment(op_info.slicePattern, kKernelConfigSlicePattern, json_read);

  Assignment(op_info.resource, kKernelConfigResource, json_read);

  Assignment(op_info.flagSupportBlockDim, kKernelConfigSupportBlockDim, json_read);

  Assignment(op_info.blockDimByIndex, kKernelConfigBlockDimByIndex, json_read);

  Assignment(op_info.implementType, kKernelConfigImplementType, json_read);

  // param[inOutputFormat]
  auto iter = json_read.find(kKernelConfigFormat);
  if (iter != json_read.end()) {
    json format_json = iter.value();
    for (json::iterator it = format_json.begin(); it != format_json.end(); ++it) {
      string in_output_name = it.key();
      string in_output_format = it.value().get<string>();
      op_info.inOutFormat[in_output_name] = in_output_format;
    }
  }

  // param[inOutDataType]
  iter = json_read.find(kKernelConfigDataType);
  if (iter != json_read.end()) {
    json data_type_json = iter.value();
    for (json::iterator it = data_type_json.begin(); it != data_type_json.end();
         ++it) {
      string in_output_name = it.key();
      string in_output_data_type = it.value().get<string>();
      op_info.inOutDataType[in_output_name] = in_output_data_type;
    }
  }

  // param[inOutName]
  iter = json_read.find(kKernelConfigName);
  if (iter != json_read.end()) {
    json name_json = iter.value();
    for (json::iterator it = name_json.begin(); it != name_json.end(); ++it) {
      string in_output_name = it.key();
      string in_output_real_name = it.value().get<string>();
      op_info.inOutRealName[in_output_real_name] = in_output_name;
    }
  }

  // param[castSrcType]
  iter = json_read.find(kKernelConfigSrcType);
  if (iter != json_read.end()) {
    json src_type_json = iter.value();
    for (json::iterator it = src_type_json.begin(); it != src_type_json.end();
         ++it) {
      string input_name = it.key();
      string input_src_type = it.value().get<string>();
      op_info.castSrcType[input_name] = input_src_type;
    }
  }

  // param[castDstType]
  iter = json_read.find(kKernelConfigDstType);
  if (iter != json_read.end()) {
    json dst_type_json = iter.value();
    for (json::iterator it = dst_type_json.begin(); it != dst_type_json.end(); ++it) {
      string input_name = it.key();
      string input_dst_type = it.value().get<string>();
      op_info.castDstType[input_name] = input_dst_type;
    }
  }
}

void from_json(const json &json_read, OpInfoDesc &desc) {
  Assignment(desc.opName, kKernelConfigOpName, json_read);
  Assignment(desc.opInfo, kKernelConfigOpInfo, json_read);
}
}  // namespace aicpu
