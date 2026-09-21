/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <securec.h>
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "framework/common/debug/ge_log.h"
#include "register/op_tiling_info.h"
#include "register/op_tiling_registry.h"
#include "op_tiling/op_tiling_utils.h"
#include "op_tiling/op_tiling_constants.h"
#include "common/util/tiling_utils.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "exe_graph/runtime/tiling_context.h"
#include "common/checker.h"
#include "common/util/mem_utils.h"
#include "register/op_check_register.h"
#include "register/tilingdata_base.h"

namespace {
bool DumpResultInfo(const std::string &result_string, char *result_info_char, const size_t result_info_len) {
  if (result_info_char == nullptr) {
    GE_LOGE("run_info buffer is null");
    return false;
  }
  if (result_string.size() >= result_info_len) {
    GE_LOGE("result_info too large. %zu/%zu", result_string.size(), result_info_len);
    return false;
  }
  return memcpy_s(result_info_char, result_string.size() + 1, result_string.c_str(), result_string.size() + 1) == EOK;
}

void CopyConstDataWithFloat16(const nlohmann::json &json_array, std::vector<uint8_t> &value) {
  std::vector<float> const_value = json_array.get<std::vector<float>>();
  std::vector<uint16_t> const_data_vec;
  for (const auto i : const_value) {
    uint16_t const_data_uint16 = optiling::Float32ToFloat16(i);
    const_data_vec.emplace_back(const_data_uint16);
  }
  uint8_t *pv_begin = reinterpret_cast<uint8_t *>(const_data_vec.data());
  uint8_t *pv_end = pv_begin + (const_data_vec.size() * sizeof(uint16_t));
  value = std::vector<uint8_t>(pv_begin, pv_end);
}

template<typename T>
void GetConstDataPointer(const nlohmann::json &json_array, std::vector<uint8_t> &const_value) {
  std::vector<T> value = json_array.get<std::vector<T>>();
  uint8_t *pv_begin = reinterpret_cast<uint8_t *>(value.data());
  uint8_t *pv_end = pv_begin + (value.size() * sizeof(T));
  const_value = std::vector<uint8_t>(pv_begin, pv_end);
}

bool CopyConstData(const std::string &dtype, const nlohmann::json &json_array, std::vector<uint8_t> &value) {
  if (dtype == "int8") {
    GetConstDataPointer<int8_t>(json_array, value);
  } else if (dtype == "uint8") {
    GetConstDataPointer<uint8_t>(json_array, value);
  } else if (dtype == "int16") {
    GetConstDataPointer<int16_t>(json_array, value);
  } else if (dtype == "uint16") {
    GetConstDataPointer<uint16_t>(json_array, value);
  } else if (dtype == "int32") {
    GetConstDataPointer<int32_t>(json_array, value);
  } else if (dtype == "uint32") {
    GetConstDataPointer<uint32_t>(json_array, value);
  } else if (dtype == "int64") {
    GetConstDataPointer<int64_t>(json_array, value);
  } else if (dtype == "uint64") {
    GetConstDataPointer<uint64_t>(json_array, value);
  } else if (dtype == "float32") {
    GetConstDataPointer<float>(json_array, value);
  } else if (dtype == "double") {
    GetConstDataPointer<double>(json_array, value);
  } else if (dtype == "float16") {
    CopyConstDataWithFloat16(json_array, value);
  } else {
    GE_LOGE("Unknown dtype: %s", dtype.c_str());
    return false;
  }
  return true;
}

void ParseConstShapeDescV2(const nlohmann::json &shape_json, ge::Operator &op_para,
                           std::map<std::string, std::vector<uint8_t>> &const_values) {
  std::vector<int64_t> shape;
  std::string format_str;
  std::string dtype_str;

  if (!shape_json.contains("const_value")) {
    GELOGI("Not const tenosr");
    return;
  }
  if (!shape_json.contains("name")) {
    REPORT_INNER_ERR_MSG("E19999", "const tensor has no name");
    return;
  }
  std::string name = shape_json["name"];

  if (shape_json.contains("shape")) {
    shape = shape_json["shape"].get<std::vector<int64_t>>();
  }
  if (shape_json.contains("format")) {
    format_str = shape_json["format"].get<std::string>();
  }
  if (shape_json.contains("dtype")) {
    dtype_str = shape_json["dtype"].get<std::string>();
  }

  std::vector<uint8_t> value;
  const bool bres = CopyConstData(dtype_str, shape_json["const_value"], value);
  if (!bres) {
    REPORT_INNER_ERR_MSG("E19999", "CopyConstData faild.  buffer is null");
    return;
  }
  auto res = const_values.emplace(name, std::move(value));
  if (res.first == const_values.end()) {
    return;  // CodeDEX complains 'CHECK_CONTAINER_EMPTY'
  }

  const ge::GeShape ge_shape(shape);
  ge::DataType ge_dtype = ge::DT_UNDEFINED;
  if (!dtype_str.empty()) {
    std::transform(dtype_str.begin(), dtype_str.end(), dtype_str.begin(), ::toupper);
    dtype_str = "DT_" + dtype_str;
    ge_dtype = ge::TypeUtils::SerialStringToDataType(dtype_str);
  }
  ge::Format ge_format = ge::FORMAT_RESERVED;
  if (!format_str.empty()) {
    std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
    ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
  }
  ge::GeTensorDesc ge_tensor(ge_shape, ge_format, ge_dtype);
  ge_tensor.SetName(name);
  ge::GeTensor const_tensor(ge_tensor, res.first->second);
  ge::GeTensorPtr const_tensor_ptr = ge::MakeShared<ge::GeTensor>(const_tensor);
  ge::OpDescPtr const_op_desc = ge::OpDescUtils::CreateConstOpZeroCopy(const_tensor_ptr);
  ge::Operator const_op = ge::OpDescUtils::CreateOperatorFromOpDesc(const_op_desc);
  (void) op_para.SetInput(name.c_str(), const_op);
}

void ParseShapeDescV2(const nlohmann::json &shape, ge::OpDescPtr &op_desc, const bool is_input) {
  ge::GeTensorDesc tensor;
  if (shape.contains("shape")) {
    tensor.SetShape(ge::GeShape(shape["shape"].get<std::vector<int64_t>>()));
  }
  if (shape.contains("ori_shape")) {
    tensor.SetOriginShape(ge::GeShape(shape["ori_shape"].get<std::vector<int64_t>>()));
  }
  if (shape.contains("format")) {
    std::string format_str = shape["format"].get<std::string>();
    std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
    ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
    tensor.SetFormat(ge_format);
  }
  if (shape.contains("ori_format")) {
    std::string format_str = shape["ori_format"].get<std::string>();
    std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
    ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
    tensor.SetOriginFormat(ge_format);
  }
  if (shape.contains("dtype")) {
    std::string dtype_str = shape["dtype"].get<std::string>();
    std::transform(dtype_str.begin(), dtype_str.end(), dtype_str.begin(), ::toupper);
    dtype_str = "DT_" + dtype_str;
    ge::DataType ge_dtype = ge::TypeUtils::SerialStringToDataType(dtype_str);
    tensor.SetDataType(ge_dtype);
  }
  if (shape.contains("name")) {
    std::string name = shape["name"];
    tensor.SetName(name);
    is_input ? op_desc->AddInputDesc(name, tensor) : op_desc->AddOutputDesc(name, tensor);
  } else {
    is_input ? op_desc->AddInputDesc(tensor) : op_desc->AddOutputDesc(tensor);
  }
}

void ParseShapeDescListV2(const nlohmann::json &shape_list, ge::OpDescPtr &op_desc, const bool is_input) {
  for (const auto &elem : shape_list) {
    if (elem.is_array()) {
      for (const auto &shape : elem) {
        ParseShapeDescV2(shape, op_desc, is_input);
      }
    } else {
      ParseShapeDescV2(elem, op_desc, is_input);
    }
  }
}

void ParseConstTensorListV2(const nlohmann::json &shape_list, ge::Operator &operator_para,
                            std::map<std::string, std::vector<uint8_t>> &const_values) {
  for (const auto &elem : shape_list) {
    if (elem.is_array()) {
      for (const auto &shape : elem) {
        ParseConstShapeDescV2(shape, operator_para, const_values);
      }
    } else {
      ParseConstShapeDescV2(elem, operator_para, const_values);
    }
  }
}

template<typename T>
void ParseAndSetAttrValue(ge::Operator &op, const nlohmann::json &attr, const std::string &attr_name) {
  T attr_value = attr["value"].get<T>();
  (void) op.SetAttr(attr_name.c_str(), attr_value);
}
template<typename T>
void ParseAndSetAttrListValue(ge::Operator &op, const nlohmann::json &attr, const std::string &attr_name) {
  std::vector<T> attr_value = attr["value"].get<std::vector<T>>();
  (void) op.SetAttr(attr_name.c_str(), attr_value);
}

void ParseAndSetAttrListListValue(ge::Operator &op, const nlohmann::json &attr, const std::string &attr_name) {
  std::vector<std::vector<int32_t>> attr_value_int32 = attr["value"].get<std::vector<std::vector<int32_t>>>();
  std::vector<std::vector<int64_t>> attr_value_int64;
  std::vector<int64_t> temp_int64_vec;
  for (const auto &vec_int32 : attr_value_int32) {
    for (const auto &item : vec_int32) {
      int64_t tmp = static_cast<int64_t>(item);
      temp_int64_vec.emplace_back(tmp);
    }
    attr_value_int64.emplace_back(temp_int64_vec);
    temp_int64_vec.clear();
  }

  (void) op.SetAttr(attr_name.c_str(), attr_value_int64);
}

void ParseAndSetAttrListListInt64Value(ge::Operator &op, const nlohmann::json &attr, const std::string &attr_name) {
  const std::vector<std::vector<int64_t>> attr_value_int64 = attr["value"].get<std::vector<std::vector<int64_t>>>();
  (void) op.SetAttr(attr_name.c_str(), attr_value_int64);
}

using ParseAndSetAttrValueFunc = std::function<void(ge::Operator &, const nlohmann::json &, const std::string &)>;
using ParseAndSetAttrValuePtr = std::shared_ptr<ParseAndSetAttrValueFunc>;

const std::map<std::string, ParseAndSetAttrValuePtr> parse_attr_dtype_map = {
    {"bool", ge::MakeShared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<bool>)},
    {"float", ge::MakeShared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<float>)},
    {"float32", ge::MakeShared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<float>)},
    {"int", ge::MakeShared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<int32_t>)},
    {"int32", ge::MakeShared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<int32_t>)},
    {"int64", ge::MakeShared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<int64_t>)},
    {"str", ge::MakeShared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<std::string>)},
    {"list_bool", ge::MakeShared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<bool>)},
    {"list_float", ge::MakeShared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<float>)},
    {"list_float32", ge::MakeShared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<float>)},
    {"list_int", ge::MakeShared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<int32_t>)},
    {"list_int32", ge::MakeShared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<int32_t>)},
    {"list_int64", ge::MakeShared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<int64_t>)},
    {"list_str", ge::MakeShared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<std::string>)},
    {"list_list_int", ge::MakeShared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListListValue)},
    {"list_list_int32", ge::MakeShared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListListValue)},
    {"list_list_int64", ge::MakeShared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListListInt64Value)}};

void ParseAndSetAttr(const nlohmann::json &attr, ge::Operator &op) {
  if ((!attr.contains("name")) || (!attr.contains("dtype")) || (!attr.contains("value"))) {
    REPORT_INNER_ERR_MSG("E19999", "cur attr does not contain name or dtype or value.");
    return;
  }
  std::string attr_name;
  std::string dtype;
  attr_name = attr["name"].get<std::string>();
  dtype = attr["dtype"].get<std::string>();
  auto iter = parse_attr_dtype_map.find(dtype);
  if (iter == parse_attr_dtype_map.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Unknown dtype[%s], which is unsupported.", dtype.c_str());
    return;
  }
  ParseAndSetAttrValuePtr func_ptr = iter->second;
  if (func_ptr == nullptr) {
    GE_LOGE("ParseAndSetAttrValueFunc ptr cannot be null!");
    return;
  }
  (*func_ptr)(op, attr, attr_name);
  GELOGD("Finish to set attr[name: %s] to Operator.", attr_name.c_str());
}

void ParseAndSetAttrsList(const nlohmann::json &attrs_list, ge::Operator &op) {
  for (const auto &attr : attrs_list) {
    ParseAndSetAttr(attr, op);
  }
}

void CheckAndSetAttr(const char *attrs, ge::Operator &operator_param) {
  if (attrs != nullptr) {
    GELOGD("Attrs set from pyAPI is: %s", attrs);
    const nlohmann::json attrs_json = nlohmann::json::parse(attrs);
    ParseAndSetAttrsList(attrs_json, operator_param);
  } else {
    GELOGD("Attrs has not been set.");
  }
  return;
}

void ParseInputsAndOutputs(const char *inputs, const char *outputs, ge::OpDescPtr &op_desc,
                           ge::Operator &operator_param, std::map<std::string, std::vector<uint8_t>> &const_values) {
  const nlohmann::json inputs_json = nlohmann::json::parse(inputs);
  const nlohmann::json outputs_json = nlohmann::json::parse(outputs);
  ParseShapeDescListV2(inputs_json, op_desc, true);
  ParseShapeDescListV2(outputs_json, op_desc, false);
  operator_param = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  ParseConstTensorListV2(inputs_json, operator_param, const_values);
}
}  // Anonymous Namespace

using namespace optiling;

extern "C" int32_t AscendCPyInterfaceCheckOp(const char *check_type, const char *optype, const char *inputs,
                                             const char *outputs, const char *attrs, char *result_info,
                                             const size_t result_info_len) {
  if ((check_type == nullptr) || (optype == nullptr) || (inputs == nullptr) || (outputs == nullptr) ||
      (attrs == nullptr) || (result_info == nullptr)) {
    GELOGE(ge::GRAPH_FAILED, "check_type/optype/inputs/outputs/attrs/result_info is null, %s, %s, %s, %s, %s, %s",
           check_type, optype, inputs, outputs, attrs, result_info);
    return 0;
  }
  ge::AscendString check_type_str = check_type;
  ge::AscendString op_type_str = optype;
  auto check_func = OpCheckFuncRegistry::GetOpCapability(check_type_str, op_type_str);
  if (check_func == nullptr) {
    GELOGW("Failed to GetOpCapability. check_type = %s, optype = %s", check_type, optype);
    return 0;
  }

  ge::OpDescPtr op_desc_ptr = ge::MakeShared<ge::OpDesc>("", op_type_str.GetString());
  std::map<std::string, std::vector<uint8_t>> const_values;
  ge::Operator operator_param;
  try {
    ParseInputsAndOutputs(inputs, outputs, op_desc_ptr, operator_param, const_values);
    CheckAndSetAttr(attrs, operator_param);
  } catch (...) {
    REPORT_INNER_ERR_MSG("E19999",
                         "Failed to parse json in AscendCPyInterfaceCheckOp. inputs = %s, outputs = %s, attrs = %s",
                         inputs, outputs, attrs);
    return 0;
  }

  ge::AscendString result;
  try {
    const ge::graphStatus rc = (check_func)(operator_param, result);
    GELOGI("check cap return rc = %u, check_type = %s, optype = %s.", rc, check_type, optype);
    if (rc != ge::GRAPH_SUCCESS) {
      GELOGE(ge::GRAPH_FAILED, "check cap return failed. check_type = %s, optype = %s", check_type, optype);
      return 0;
    }
  } catch (...) {
    GELOGE(ge::GRAPH_FAILED, "check cap abnormal failed. check_type = %s, optype = %s", check_type, optype);
    return 0;
  }
  std::string std_str = result.GetString();
  bool dump_res = DumpResultInfo(std_str, result_info, result_info_len);
  if (!dump_res) {
    REPORT_INNER_ERR_MSG("E19999", "DumpResultInfo failed. result = %s", std_str.c_str());
    return 0;
  }
  return 1;
}

extern "C" int32_t AscendCPyInterfaceGeneralized(const char *optype, const char *inputs, const char *outputs,
                                                 const char *attrs, const char *generalize_config, char *result_info,
                                                 const size_t result_info_len) {
  if ((optype == nullptr) || (inputs == nullptr) || (outputs == nullptr) || (attrs == nullptr) ||
      (generalize_config == nullptr) || (result_info == nullptr)) {
    GELOGE(ge::GRAPH_FAILED,
           "optype/inputs/outputs/attrs/generalize_config/result_info is null, %s, %s, %s, %s, %s, %s", optype, inputs,
           outputs, attrs, generalize_config, result_info);
    return 0;
  }
  ge::AscendString op_type_str = optype;
  auto generalize_func = OpCheckFuncRegistry::GetParamGeneralize(op_type_str);
  if (generalize_func == nullptr) {
    GELOGW("Failed to GetParamGeneralize. optype = %s", optype);
    return 0;
  }

  ge::OpDescPtr op_desc_ptr = ge::MakeShared<ge::OpDesc>("", op_type_str.GetString());
  std::map<std::string, std::vector<uint8_t>> const_values;
  ge::Operator operator_params;
  try {
    ParseInputsAndOutputs(inputs, outputs, op_desc_ptr, operator_params, const_values);
    CheckAndSetAttr(attrs, operator_params);
  } catch (...) {
    GELOGE(ge::GRAPH_FAILED, "Failed to parse json in AscendCPyInterfaceGeneralized. %s, %s, %s",
           inputs, outputs, attrs);
    return 0;
  }
  ge::AscendString generalize_config_str(generalize_config);
  ge::AscendString result;
  try {
    const ge::graphStatus rc = (generalize_func)(operator_params, generalize_config_str, result);
    GELOGI("generalize_func return rc = %d, optype = %s, generalize_config = %s", rc, optype, generalize_config);
    if (rc != ge::GRAPH_SUCCESS) {
      GELOGE(ge::GRAPH_FAILED,  "call generalize_func failed. optype = %s, generalize_config = %s", optype,
             generalize_config);
      return 0;
    }
  } catch (...) {
    GELOGE(ge::GRAPH_FAILED, "call generalize_func failed. optype = %s, generalize_config = %s", optype,
           generalize_config);
    return 0;
  }
  std::string result_str = result.GetString();
  bool dump_res = DumpResultInfo(result_str, result_info, result_info_len);
  if (!dump_res) {
    REPORT_INNER_ERR_MSG("E19999", "DumpResultInfo failed. result = %s", result_str.c_str());
    return 0;
  }
  return 1;
}

extern "C" int32_t AscendCPyInterfaceGetTilingDefInfo(const char *optype, char *result_info, size_t result_info_len) {
  if ((optype == nullptr) || (result_info == nullptr)) {
    GELOGE(ge::GRAPH_FAILED, "optype/result_info is null, %s, %s", optype, result_info);
    return 0;
  }
  ge::AscendString op_type_str = optype;
  auto tiling_def = CTilingDataClassFactory::GetInstance().CreateTilingDataInstance(optype);
  if (tiling_def == nullptr) {
    GELOGW("Failed to CreateTilingDataInstance. optype = %s", optype);
    return 0;
  }

  nlohmann::json json_obj;
  json_obj["class_name"] = tiling_def->GetTilingClassName();
  json_obj["data_size"] = tiling_def->GetDataSize();
  const auto &field_list = tiling_def->GetFieldInfo();
  nlohmann::json json_field_list;
  for (const auto &field : field_list) {
    nlohmann::json json_field;
    json_field["classType"] = field.classType_;
    json_field["name"] = field.name_;
    json_field["dtype"] = field.dtype_;
    if (json_field["classType"] == "1") {
      json_field["arrSize"] = field.arrSize_;
    } else if (json_field["classType"] == "2") {
      json_field["structType"] = field.structType_;
      json_field["structSize"] = field.structSize_;
    }
    json_field_list.emplace_back(json_field);
  }
  json_obj["fields"] = json_field_list;
  const std::string json_str = json_obj.dump();
  bool dump_res = DumpResultInfo(json_str, result_info, result_info_len);
  if (!dump_res) {
    GELOGE(ge::GRAPH_FAILED, "AscendCPyInterfaceGetTilingDefInfo DumpResultInfo failed. result = %s", json_str.c_str());
    return 0;
  }
  return 1;
}

extern "C" int32_t AscendCPyInterfaceOpReplay(const char *optype, const char *soc_version, const int32_t block_dim,
                                              const char *tiling_data, const char *kernel_name, const char *entry_file,
                                              const char *output_kernel_file, const int32_t core_type,
                                              const int32_t task_ration, const int32_t tiling_key) {
  if ((optype == nullptr) || (soc_version == nullptr) || (tiling_data == nullptr) || (kernel_name == nullptr) ||
      (entry_file == nullptr) || (output_kernel_file == nullptr)) {
    GELOGE(ge::GRAPH_FAILED,
           "optype/soc_version/tiling_data/kernel_name/entry_file/output_kernel_file is null, "
           "%s, %s, %s, %s, %s, %s",
           optype, soc_version, tiling_data, kernel_name, entry_file, output_kernel_file);
    return 0;
  }
  constexpr int32_t CORE_TYPE_BOTH = 0;
  constexpr int32_t CORE_TYPE_CUBE = 1;
  constexpr int32_t CORE_TYPE_VEC = 2;
  if ((core_type != CORE_TYPE_BOTH) && (core_type != CORE_TYPE_CUBE) && (core_type != CORE_TYPE_VEC)) {
    GELOGE(ge::GRAPH_FAILED,
           "core_type is valid, should be one of 0/1/2, but args is "
           "%d",
           core_type);
    return 0;
  }
  constexpr int32_t TASK_RATION_ONE = 1;
  constexpr int32_t TASK_RATION_TWO = 2;
  if ((task_ration != TASK_RATION_ONE) && (task_ration != TASK_RATION_TWO)) {
    GELOGE(ge::GRAPH_FAILED,
           "task_ration is valid, should be one of 1/2, but args is "
           "%d",
           task_ration);
    return 0;
  }
  ge::AscendString op_type_str = optype;
  ge::AscendString soc_version_str = soc_version;
  auto replay_func = OpCheckFuncRegistry::GetReplay(op_type_str, soc_version_str);
  if (replay_func == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "Failed to GetReplay. optype = %s, soc_version = %s", optype, soc_version);
    return 0;
  }

  try {
    ReplayFuncParam replayParam;
    replayParam.block_dim = block_dim;
    replayParam.tiling_data = tiling_data;
    replayParam.kernel_name = kernel_name;
    replayParam.entry_file = entry_file;
    replayParam.gentype = 0;
    replayParam.output_kernel_file = output_kernel_file;
    replayParam.task_ration = task_ration;
    replayParam.tiling_key = tiling_key;
    const int32_t rc = (replay_func)(replayParam, core_type);
    if (rc <= 0) {
      GELOGE(ge::GRAPH_FAILED, "call replay_func return %d. optype = %s, soc_version = %s", rc, optype, soc_version);
      return 0;
    }
    GELOGI("replay_func return rc = %d, optype = %s, soc_version = %s.", rc, optype, soc_version);
  } catch (...) {
    GELOGE(ge::GRAPH_FAILED, "call replay_func segment fault. optype = %s, soc_version = %s", optype, soc_version);
    return 0;
  }
  return 1;
}
