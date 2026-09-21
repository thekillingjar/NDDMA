/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "json_file_operator.h"
#include <fstream>
#include "util/dvpp_constexpr.h"
#include "util/dvpp_define.h"
#include "util/dvpp_error_code.h"
#include "util/dvpp_log.h"
#include "util/util.h"

namespace dvpp {
bool JsonFileOperator::ReadJsonFile(const std::string& file_path_new,
    const std::string& file_path, nlohmann::json& json_file) {
  std::ifstream ifs(file_path_new);
  DVPP_CHECK_IF_THEN_DO(!ifs.is_open(),
      DVPP_ENGINE_LOG_WARNING("can not open [%s]", file_path_new.c_str());
      ifs.open(file_path);
      DVPP_CHECK_IF_THEN_DO(!ifs.is_open(),
          DVPP_REPORT_INNER_ERR_MSG("open [%s] failed", file_path.c_str());
          return false));

  try {
    ifs >> json_file;
    ifs.close();
  } catch (const nlohmann::json::exception& e) {
    DVPP_CHECK_IF_THEN_DO(ifs.is_open(), ifs.close());
    DVPP_REPORT_INNER_ERR_MSG("error msg is [%s]", e.what());
    return false;
  }

  return ConvertJsonFormat(json_file);
}

bool JsonFileOperator::ConvertJsonFormat(nlohmann::json& json_file) {
  nlohmann::json dvpp_ops = nlohmann::json::array();
  for (auto iter = json_file.cbegin(); iter != json_file.cend(); ++iter) {
    nlohmann::json dvpp_op = iter.value();
    dvpp_op[kJsonDvppOpName] = iter.key();

    // convert info::engine
    bool ret = ConvertJsonFormatString(dvpp_op, kJsonDvppOpInfoEngine);
    DVPP_CHECK_IF_THEN_DO(!ret, return false);

    // convert info::opKernelLib
    ret = ConvertJsonFormatString(dvpp_op, kJsonDvppOpInfoOpKernelLib);
    DVPP_CHECK_IF_THEN_DO(!ret, return false);

    // convert info::computeCost
    ret = ConvertJsonFormatInt32(dvpp_op, kJsonDvppOpInfoComputeCost);
    DVPP_CHECK_IF_THEN_DO(!ret, return false);

    // convert info::flagPartial
    ret = ConvertJsonFormatBool(dvpp_op, kJsonDvppOpInfoFlagPartial);
    DVPP_CHECK_IF_THEN_DO(!ret, return false);

    // convert info::flagAsync
    ret = ConvertJsonFormatBool(dvpp_op, kJsonDvppOpInfoFlagAsync);
    DVPP_CHECK_IF_THEN_DO(!ret, return false);

    // convert info::kernelSo
    ret = ConvertJsonFormatString(dvpp_op, kJsonDvppOpInfoKernelSo);
    DVPP_CHECK_IF_THEN_DO(!ret, return false);

    // convert info::functionName
    ret = ConvertJsonFormatString(dvpp_op, kJsonDvppOpInfoFunctionName);
    DVPP_CHECK_IF_THEN_DO(!ret, return false);

    // convert info::userDefined
    ret = ConvertJsonFormatBool(dvpp_op, kJsonDvppOpInfoUserDefined);
    DVPP_CHECK_IF_THEN_DO(!ret, return false);

    // convert info::workspaceSize
    ret = ConvertJsonFormatInt32(dvpp_op, kJsonDvppOpInfoWorkspaceSize);
    DVPP_CHECK_IF_THEN_DO(!ret, return false);

    // convert info::resource
    ret = ConvertJsonFormatString(dvpp_op, kJsonDvppOpInfoResource);
    DVPP_CHECK_IF_THEN_DO(!ret, return false);

    // convert info::memoryType
    ret = ConvertJsonFormatString(dvpp_op, kJsonDvppOpInfoMemoryType);
    DVPP_CHECK_IF_THEN_DO(!ret, return false);

    // convert info::widthMin
    ret = ConvertJsonFormatInt64(dvpp_op, kJsonDvppOpInfoWidthMin);
    DVPP_CHECK_IF_THEN_DO(!ret, return false);

    // convert info::widthMax
    ret = ConvertJsonFormatInt64(dvpp_op, kJsonDvppOpInfoWidthMax);
    DVPP_CHECK_IF_THEN_DO(!ret, return false);

    // convert info::widthAlign
    ret = ConvertJsonFormatInt64(dvpp_op, kJsonDvppOpInfoWidthAlign);
    DVPP_CHECK_IF_THEN_DO(!ret, return false);

    // convert info::heightMin
    ret = ConvertJsonFormatInt64(dvpp_op, kJsonDvppOpInfoHeightMin);
    DVPP_CHECK_IF_THEN_DO(!ret, return false);

    // convert info::heightMax
    ret = ConvertJsonFormatInt64(dvpp_op, kJsonDvppOpInfoHeightMax);
    DVPP_CHECK_IF_THEN_DO(!ret, return false);

    // convert info::heightAlign
    ret = ConvertJsonFormatInt64(dvpp_op, kJsonDvppOpInfoHeightAlign);
    DVPP_CHECK_IF_THEN_DO(!ret, return false);

    // convert info::(inputs & outputs)
    ret = ConvertJsonFormatInputOutput(dvpp_op);
    DVPP_CHECK_IF_THEN_DO(!ret, return false);

    // convert info::attrs
    ret = ConvertJsonFormatAttr(dvpp_op);
    DVPP_CHECK_IF_THEN_DO(!ret, return false);

    dvpp_ops.push_back(dvpp_op);
  }

  json_file = {};
  json_file[kJsonDvppOpsInfoLibName] = kDvppOpsKernel;
  json_file[kJsonDvppOpsInfoLibOps] = dvpp_ops;

  PrintJsonFileContext(json_file);
  return true;
}

bool JsonFileOperator::ConvertJsonFormatString(
    nlohmann::json& dvpp_op, const std::string& key) {
  std::string op_name = dvpp_op[kJsonDvppOpName];
  auto json_str = dvpp_op[kJsonDvppOpInfo][key];
  DVPP_CHECK_IF_THEN_DO(json_str.empty(),
      dvpp_op[kJsonDvppOpInfo][key] = "";
      return true); // 内容为空不能报错，可能是没有该属性

  dvpp_op[kJsonDvppOpInfo][key] = json_str.get<std::string>();
  return true;
}

bool JsonFileOperator::ConvertJsonFormatInt32(
    nlohmann::json& dvpp_op, const std::string& key) {
  std::string op_name = dvpp_op[kJsonDvppOpName];
  auto json_str = dvpp_op[kJsonDvppOpInfo][key];
  DVPP_CHECK_IF_THEN_DO(json_str.empty(),
      DVPP_REPORT_INNER_ERR_MSG("op[%s]: %s is empty",
                              op_name.c_str(), key.c_str());
      return false);

  std::string value_str = json_str.get<std::string>();
  int32_t value = 0;
  auto error_code = StringToNum(value_str, value);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("op[%s]: convert %s[%s] to int failed",
          op_name.c_str(), key.c_str(), value_str.c_str());
      return false);

  dvpp_op[kJsonDvppOpInfo][key] = value;
  return true;
}

bool JsonFileOperator::ConvertJsonFormatInt64(
    nlohmann::json& dvpp_op, const std::string& key) {
  std::string op_name = dvpp_op[kJsonDvppOpName];
  auto json_str = dvpp_op[kJsonDvppOpInfo][key];
  DVPP_CHECK_IF_THEN_DO(json_str.empty(),
      DVPP_REPORT_INNER_ERR_MSG("op[%s]: %s is empty",
                              op_name.c_str(), key.c_str());
      return false);

  std::string value_str = json_str.get<std::string>();
  int64_t value = 0;
  auto error_code = StringToNum(value_str, value);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("op[%s]: convert %s[%s] to int failed",
          op_name.c_str(), key.c_str(), value_str.c_str());
      return false);

  dvpp_op[kJsonDvppOpInfo][key] = value;
  return true;
}

bool JsonFileOperator::ConvertJsonFormatBool(
    nlohmann::json& dvpp_op, const std::string& key) {
  std::string op_name = dvpp_op[kJsonDvppOpName];
  auto json_str = dvpp_op[kJsonDvppOpInfo][key];
  DVPP_CHECK_IF_THEN_DO(json_str.empty(),
      DVPP_REPORT_INNER_ERR_MSG("op[%s]: %s is empty",
                            op_name.c_str(), key.c_str());
      return false);

  std::string value_str = json_str.get<std::string>();
  bool value = false;
  auto error_code = StringToBool(value_str, value);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("op[%s]: convert %s[%s] to bool failed",
          op_name.c_str(), key.c_str(), value_str.c_str());
      return false);

  dvpp_op[kJsonDvppOpInfo][key] = value;
  return true;
}

bool JsonFileOperator::ConvertJsonFormatInputOutput(nlohmann::json& dvpp_op) {
  nlohmann::json inputs;
  nlohmann::json outputs;

  for (auto iter = dvpp_op.cbegin(); iter != dvpp_op.cend(); ++iter) {
    std::string key = iter.key();
    if (key.size() < kNum6) {
      // only convert key: inputn and outputn
      // n >= 0, such as input0 output1, so size is 6 at least
      continue;
    }

    if ((strncmp(key.c_str(), kJsonStrInput.c_str(),
                 kJsonStrInput.size()) != 0) &&
        (strncmp(key.c_str(), kJsonStrOutput.c_str(),
                 kJsonStrOutput.size()) != 0)) {
      continue;
    }

    nlohmann::json op_info = dvpp_op[key];

    std::string name = "";
    auto it = op_info.find(kJsonStrName);
    if (it != op_info.end()) {
      name = it.value().get<std::string>();
    }

    std::string type = "";
    it = op_info.find(kJsonStrType);
    if (it != op_info.end()) {
      type = it.value().get<std::string>();
    }

    std::string format = "";
    it = op_info.find(kJsonStrFormat);
    if (it != op_info.end()) {
      format = it.value().get<std::string>();
    }

    nlohmann::json in_out_info;
    in_out_info[kJsonInOutInfoDataName] = name;
    in_out_info[kJsonInOutInfoDataType] = type;
    in_out_info[kJsonInOutInfoDataFormat] = format;

    if (strncmp(key.c_str(), kJsonStrInput.c_str(),
                kJsonStrInput.size()) == 0) {
      inputs[key] = in_out_info;
    } else {
      outputs[key] = in_out_info;
    }
  }

  dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoInputs] = inputs;
  dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoOutputs] = outputs;

  return true;
}

bool JsonFileOperator::ConvertJsonFormatAttr(nlohmann::json& dvpp_op) {
  nlohmann::json attrs;

  for (auto iter = dvpp_op.cbegin(); iter != dvpp_op.cend(); ++iter) {
    std::string key = iter.key();
    if (key != kJsonStrAttr) {
      continue;
    }

    nlohmann::json attrs_info = dvpp_op[key];
    for (auto attr = attrs_info.cbegin(); attr != attrs_info.cend(); ++attr) {
      std::string type = "";
      auto it = attr.value().find(kJsonAttrInfoType);
      if (it != attr.value().end()) {
        type = it.value().get<std::string>();
      }

      std::string value = "";
      it = attr.value().find(kJsonAttrInfoValue);
      if (it != attr.value().end()) {
        value = it.value().get<std::string>();
      }

      nlohmann::json attr_info;
      attr_info[kJsonAttrInfoType] = type;
      attr_info[kJsonAttrInfoValue] = value;
      attrs[attr.key()] = attr_info;
    }
  }

  dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoAttrs] = attrs;

  return true;
}

void JsonFileOperator::PrintJsonFileContext(nlohmann::json& json_file) {
  DVPP_ENGINE_LOG_DEBUG("print json file context start");
  nlohmann::json& dvpp_ops = json_file[kJsonDvppOpsInfoLibOps];
  uint32_t index = 0;
  for (auto iter = dvpp_ops.begin(); iter != dvpp_ops.end(); ++iter) {
    nlohmann::json dvpp_op = iter.value();
    std::string temp_str = dvpp_op[kJsonDvppOpName];
    DVPP_ENGINE_LOG_DEBUG("op[%s]", temp_str.c_str());

    temp_str = dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoEngine];
    DVPP_ENGINE_LOG_DEBUG("engine[%s]", temp_str.c_str());

    temp_str = dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoOpKernelLib];
    DVPP_ENGINE_LOG_DEBUG("opKernelLib[%s]", temp_str.c_str());

    int32_t temp_int32 = dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoComputeCost];
    DVPP_ENGINE_LOG_DEBUG("computeCost[%d]", temp_int32);

    DVPP_CHECK_IF_THEN_DO(
        dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoFlagPartial] == true,
        DVPP_ENGINE_LOG_DEBUG("flagPartial[true]"));
    DVPP_CHECK_IF_THEN_DO(
        dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoFlagPartial] == false,
        DVPP_ENGINE_LOG_DEBUG("flagPartial[false]"));

    DVPP_CHECK_IF_THEN_DO(
        dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoFlagAsync] == true,
        DVPP_ENGINE_LOG_DEBUG("flagAsync[true]"));
    DVPP_CHECK_IF_THEN_DO(
        dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoFlagAsync] == false,
        DVPP_ENGINE_LOG_DEBUG("flagAsync[false]"));

    temp_str = dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoKernelSo];
    DVPP_ENGINE_LOG_DEBUG("kernelSo[%s]", temp_str.c_str());

    temp_str = dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoFunctionName];
    DVPP_ENGINE_LOG_DEBUG("functionName[%s]", temp_str.c_str());

    DVPP_CHECK_IF_THEN_DO(
        dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoUserDefined] == true,
        DVPP_ENGINE_LOG_DEBUG("userDefined[true]"));
    DVPP_CHECK_IF_THEN_DO(
        dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoUserDefined] == false,
        DVPP_ENGINE_LOG_DEBUG("userDefined[false]"));

    temp_int32 = dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoWorkspaceSize];
    DVPP_ENGINE_LOG_DEBUG("workspaceSize[%d]", temp_int32);

    temp_str = dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoResource];
    DVPP_ENGINE_LOG_DEBUG("resource[%s]", temp_str.c_str());

    temp_str = dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoMemoryType];
    DVPP_ENGINE_LOG_DEBUG("memoryType[%s]", temp_str.c_str());

    int64_t temp_int64 = dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoWidthMin];
    DVPP_ENGINE_LOG_DEBUG("widthMin[%ld]", temp_int64);

    temp_int64 = dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoWidthMax];
    DVPP_ENGINE_LOG_DEBUG("widthMax[%ld]", temp_int64);

    temp_int64 = dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoWidthAlign];
    DVPP_ENGINE_LOG_DEBUG("widthAlign[%ld]", temp_int64);

    temp_int64 = dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoHeightMin];
    DVPP_ENGINE_LOG_DEBUG("heightMin[%ld]", temp_int64);

    temp_int64 = dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoHeightMax];
    DVPP_ENGINE_LOG_DEBUG("heightMax[%ld]", temp_int64);

    temp_int64 = dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoHeightAlign];
    DVPP_ENGINE_LOG_DEBUG("heightAlign[%ld]", temp_int64);

    // print info::(inputs & outputs)
    PrintDvppOpInfoInputOutput(dvpp_op);

    // print info::attrs
    PrintDvppOpInfoAttr(dvpp_op);

    ++index;
  }

  DVPP_ENGINE_LOG_DEBUG("ops total num is %u", index);
  DVPP_ENGINE_LOG_DEBUG("print json file context success");
}

void JsonFileOperator::PrintDvppOpInfoInputOutput(nlohmann::json& dvpp_op) {
  nlohmann::json& inputs = dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoInputs];
  for (auto iter = inputs.begin(); iter != inputs.end(); ++iter) {
    DVPP_ENGINE_LOG_DEBUG("input[%s]", iter.key().c_str());
    nlohmann::json input = iter.value();
    std::string data_name = input[kJsonInOutInfoDataName];
    DVPP_ENGINE_LOG_DEBUG("name[%s]", data_name.c_str());
    std::string data_type = input[kJsonInOutInfoDataType];
    DVPP_ENGINE_LOG_DEBUG("type[%s]", data_type.c_str());
    std::string data_format = input[kJsonInOutInfoDataFormat];
    DVPP_ENGINE_LOG_DEBUG("format[%s]", data_format.c_str());
  }

  nlohmann::json& outputs = dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoOutputs];
  for (auto iter = outputs.begin(); iter != outputs.end(); ++iter) {
    DVPP_ENGINE_LOG_DEBUG("output[%s]", iter.key().c_str());
    nlohmann::json output = iter.value();
    std::string data_name = output[kJsonInOutInfoDataName];
    DVPP_ENGINE_LOG_DEBUG("name[%s]", data_name.c_str());
    std::string data_type = output[kJsonInOutInfoDataType];
    DVPP_ENGINE_LOG_DEBUG("type[%s]", data_type.c_str());
    std::string data_format = output[kJsonInOutInfoDataFormat];
    DVPP_ENGINE_LOG_DEBUG("format[%s]", data_format.c_str());
  }
}

void JsonFileOperator::PrintDvppOpInfoAttr(nlohmann::json& dvpp_op) {
  nlohmann::json& attrs = dvpp_op[kJsonDvppOpInfo][kJsonDvppOpInfoAttrs];
  for (auto iter = attrs.begin(); iter != attrs.end(); ++iter) {
    DVPP_ENGINE_LOG_DEBUG("attr[%s]", iter.key().c_str());
    nlohmann::json attr = iter.value();
    std::string type = attr[kJsonAttrInfoType];
    DVPP_ENGINE_LOG_DEBUG("type[%s]", type.c_str());
    std::string value = attr[kJsonAttrInfoValue];
    DVPP_ENGINE_LOG_DEBUG("value[%s]", value.c_str());
  }
}

template <typename T>
inline void Assignment(
    const nlohmann::json& json_file, const std::string& key, T& variable) {
  auto iter = json_file.find(key);
  if (iter != json_file.end()) {
    variable = iter.value().get<T>();
  }
}

void from_json(const nlohmann::json& json_file,
               DvppOpsInfoLib& dvpp_ops_info_lib) {
  Assignment(json_file, kJsonDvppOpsInfoLibName, dvpp_ops_info_lib.name);
  Assignment(json_file, kJsonDvppOpsInfoLibOps, dvpp_ops_info_lib.ops);
}

void from_json(const nlohmann::json& json_file, DvppOp& dvpp_op) {
  Assignment(json_file, kJsonDvppOpName, dvpp_op.name);
  Assignment(json_file, kJsonDvppOpInfo, dvpp_op.info);
}

void from_json(const nlohmann::json& json_file, DvppOpInfo& dvpp_op_info) {
  dvpp_op_info.engine = "";
  Assignment(json_file, kJsonDvppOpInfoEngine, dvpp_op_info.engine);

  dvpp_op_info.opKernelLib = "";
  Assignment(json_file, kJsonDvppOpInfoOpKernelLib, dvpp_op_info.opKernelLib);

  dvpp_op_info.computeCost = 0;
  Assignment(json_file, kJsonDvppOpInfoComputeCost, dvpp_op_info.computeCost);

  dvpp_op_info.flagPartial = false;
  Assignment(json_file, kJsonDvppOpInfoFlagPartial, dvpp_op_info.flagPartial);

  dvpp_op_info.flagAsync = false;
  Assignment(json_file, kJsonDvppOpInfoFlagAsync, dvpp_op_info.flagAsync);

  dvpp_op_info.kernelSo = "";
  Assignment(json_file, kJsonDvppOpInfoKernelSo, dvpp_op_info.kernelSo);

  dvpp_op_info.functionName = "";
  Assignment(json_file, kJsonDvppOpInfoFunctionName, dvpp_op_info.functionName);

  dvpp_op_info.userDefined = false;
  Assignment(json_file, kJsonDvppOpInfoUserDefined, dvpp_op_info.userDefined);

  dvpp_op_info.workspaceSize = 0;
  Assignment(json_file, kJsonDvppOpInfoWorkspaceSize,
             dvpp_op_info.workspaceSize);

  dvpp_op_info.resource = "";
  Assignment(json_file, kJsonDvppOpInfoResource, dvpp_op_info.resource);

  dvpp_op_info.memoryType = "";
  Assignment(json_file, kJsonDvppOpInfoMemoryType, dvpp_op_info.memoryType);

  dvpp_op_info.widthMin = 0;
  Assignment(json_file, kJsonDvppOpInfoWidthMin, dvpp_op_info.widthMin);

  dvpp_op_info.widthMax = 0;
  Assignment(json_file, kJsonDvppOpInfoWidthMax, dvpp_op_info.widthMax);

  dvpp_op_info.widthAlign = 0;
  Assignment(json_file, kJsonDvppOpInfoWidthAlign, dvpp_op_info.widthAlign);

  dvpp_op_info.heightMin = 0;
  Assignment(json_file, kJsonDvppOpInfoHeightMin, dvpp_op_info.heightMin);

  dvpp_op_info.heightMax = 0;
  Assignment(json_file, kJsonDvppOpInfoHeightMax, dvpp_op_info.heightMax);

  dvpp_op_info.heightAlign = 0;
  Assignment(json_file, kJsonDvppOpInfoHeightAlign, dvpp_op_info.heightAlign);

  auto iter = json_file.find(kJsonDvppOpInfoInputs);
  if (iter != json_file.end()) {
    nlohmann::json json_inputs = iter.value();
    for (auto it = json_inputs.begin(); it != json_inputs.end(); ++it) {
      nlohmann::json json_info = it.value();
      struct InOutInfo info;
      info.dataName = "";
      if (json_info.find(kJsonInOutInfoDataName) != json_info.end()) {
        info.dataName = json_info[kJsonInOutInfoDataName];
      }
      info.dataType = "";
      if (json_info.find(kJsonInOutInfoDataType) != json_info.end()) {
        info.dataType = json_info[kJsonInOutInfoDataType];
      }
      info.dataFormat = "";
      if (json_info.find(kJsonInOutInfoDataFormat) != json_info.end()) {
        info.dataFormat = json_info[kJsonInOutInfoDataFormat];
      }
      dvpp_op_info.inputs[it.key()] = info;
    }
  }

  iter = json_file.find(kJsonDvppOpInfoOutputs);
  if (iter != json_file.end()) {
    nlohmann::json json_outputs = iter.value();
    for (auto it = json_outputs.begin(); it != json_outputs.end(); ++it) {
      nlohmann::json json_info = it.value();
      struct InOutInfo info;
      info.dataName = "";
      if (json_info.find(kJsonInOutInfoDataName) != json_info.end()) {
        info.dataName = json_info[kJsonInOutInfoDataName];
      }
      info.dataType = "";
      if (json_info.find(kJsonInOutInfoDataType) != json_info.end()) {
        info.dataType = json_info[kJsonInOutInfoDataType];
      }
      info.dataFormat = "";
      if (json_info.find(kJsonInOutInfoDataFormat) != json_info.end()) {
        info.dataFormat = json_info[kJsonInOutInfoDataFormat];
      }
      dvpp_op_info.outputs[it.key()] = info;
    }
  }

  iter = json_file.find(kJsonDvppOpInfoAttrs);
  if (iter != json_file.end()) {
    nlohmann::json json_attrs = iter.value();
    for (auto it = json_attrs.begin(); it != json_attrs.end(); ++it) {
      nlohmann::json json_info = it.value();
      struct AttrInfo info;
      info.type = "";
      if (json_info.find(kJsonAttrInfoType) != json_info.end()) {
        info.type = json_info[kJsonAttrInfoType];
      }
      info.value = "";
      if (json_info.find(kJsonAttrInfoValue) != json_info.end()) {
        info.value = json_info[kJsonAttrInfoValue];
      }
      dvpp_op_info.attrs[it.key()] = info;
    }
  }
}
} // namespace dvpp
