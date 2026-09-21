/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "util.h"
#include "util/dvpp_constexpr.h"
#include "util/dvpp_define.h"
#include "util/dvpp_error_code.h"
#include "util/dvpp_log.h"

namespace dvpp {
const std::map<ge::DataType, int32_t> kDataTypeSizeMap = {
  {ge::DataType::DT_FLOAT16,    kSizeOfFloat16},
  {ge::DataType::DT_FLOAT,      sizeof(float)},
  {ge::DataType::DT_DOUBLE,     sizeof(double)},
  {ge::DataType::DT_INT8,       sizeof(int8_t)},
  {ge::DataType::DT_UINT8,      sizeof(uint8_t)},
  {ge::DataType::DT_INT16,      sizeof(int16_t)},
  {ge::DataType::DT_UINT16,     sizeof(uint16_t)},
  {ge::DataType::DT_INT32,      sizeof(int32_t)},
  {ge::DataType::DT_UINT32,     sizeof(uint32_t)},
  {ge::DataType::DT_INT64,      sizeof(int64_t)},
  {ge::DataType::DT_UINT64,     sizeof(uint64_t)},
  {ge::DataType::DT_BOOL,       sizeof(bool)},
  {ge::DataType::DT_RESOURCE,   kSizeOfResource},
  {ge::DataType::DT_STRING,     kSizeOfString},
  {ge::DataType::DT_STRING_REF, kSizeOfStringRef},
  {ge::DataType::DT_COMPLEX64,  kSizeOfComplex64},
  {ge::DataType::DT_COMPLEX128, kSizeOfComplex128},
  {ge::DataType::DT_QINT16,     sizeof(int16_t)},
  {ge::DataType::DT_QUINT16,    sizeof(uint16_t)},
  {ge::DataType::DT_QINT8,      sizeof(int8_t)},
  {ge::DataType::DT_QUINT8,     sizeof(uint8_t)},
  {ge::DataType::DT_QINT32,     sizeof(int32_t)},
  {ge::DataType::DT_VARIANT,    kSizeOfVariant}
};

const std::map<std::string, ge::DataType> kStringToDataTypeMap = {
  {"DT_FLOAT",          ge::DataType::DT_FLOAT},
  {"DT_FLOAT16",        ge::DataType::DT_FLOAT16},
  {"DT_INT8",           ge::DataType::DT_INT8},
  {"DT_INT32",          ge::DataType::DT_INT32},
  {"DT_UINT8",          ge::DataType::DT_UINT8},
  {"DT_INT16",          ge::DataType::DT_INT16},
  {"DT_UINT16",         ge::DataType::DT_UINT16},
  {"DT_UINT32",         ge::DataType::DT_UINT32},
  {"DT_INT64",          ge::DataType::DT_INT64},
  {"DT_UINT64",         ge::DataType::DT_UINT64},
  {"DT_DOUBLE",         ge::DataType::DT_DOUBLE},
  {"DT_BOOL",           ge::DataType::DT_BOOL},
  {"DT_STRING",         ge::DataType::DT_STRING},
  {"DT_DUAL_SUB_INT8",  ge::DataType::DT_DUAL_SUB_INT8},
  {"DT_DUAL_SUB_UINT8", ge::DataType::DT_DUAL_SUB_UINT8},
  {"DT_COMPLEX64",      ge::DataType::DT_COMPLEX64},
  {"DT_COMPLEX128",     ge::DataType::DT_COMPLEX128},
  {"DT_QINT8",          ge::DataType::DT_QINT8},
  {"DT_QINT16",         ge::DataType::DT_QINT16},
  {"DT_QINT32",         ge::DataType::DT_QINT32},
  {"DT_QUINT8",         ge::DataType::DT_QUINT8},
  {"DT_QUINT16",        ge::DataType::DT_QUINT16},
  {"DT_RESOURCE",       ge::DataType::DT_RESOURCE},
  {"DT_STRING_REF",     ge::DataType::DT_STRING_REF},
  {"DT_DUAL",           ge::DataType::DT_DUAL},
  {"DT_VARIANT",        ge::DataType::DT_VARIANT},
  {"DT_BF16",           ge::DataType::DT_BF16},
  {"DT_UNDEFINED",      ge::DataType::DT_UNDEFINED},
  {"DT_INT4",           ge::DataType::DT_INT4},
  {"DT_UINT1",          ge::DataType::DT_UINT1},
  {"DT_INT2",           ge::DataType::DT_INT2},
  {"DT_UINT2",          ge::DataType::DT_UINT2}
};

const std::map<ge::DataType, std::string> kDataTypeToStringMap = {
  {ge::DataType::DT_FLOAT,          "DT_FLOAT"},
  {ge::DataType::DT_FLOAT16,        "DT_FLOAT16"},
  {ge::DataType::DT_INT8,           "DT_INT8"},
  {ge::DataType::DT_INT32,          "DT_INT32"},
  {ge::DataType::DT_UINT8,          "DT_UINT8"},
  {ge::DataType::DT_INT16,          "DT_INT16"},
  {ge::DataType::DT_UINT16,         "DT_UINT16"},
  {ge::DataType::DT_UINT32,         "DT_UINT32"},
  {ge::DataType::DT_INT64,          "DT_INT64"},
  {ge::DataType::DT_UINT64,         "DT_UINT64"},
  {ge::DataType::DT_DOUBLE,         "DT_DOUBLE"},
  {ge::DataType::DT_BOOL,           "DT_BOOL"},
  {ge::DataType::DT_STRING,         "DT_STRING"},
  {ge::DataType::DT_DUAL_SUB_INT8,  "DT_DUAL_SUB_INT8"},
  {ge::DataType::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"},
  {ge::DataType::DT_COMPLEX64,      "DT_COMPLEX64"},
  {ge::DataType::DT_COMPLEX128,     "DT_COMPLEX128"},
  {ge::DataType::DT_QINT8,          "DT_QINT8"},
  {ge::DataType::DT_QINT16,         "DT_QINT16"},
  {ge::DataType::DT_QINT32,         "DT_QINT32"},
  {ge::DataType::DT_QUINT8,         "DT_QUINT8"},
  {ge::DataType::DT_QUINT16,        "DT_QUINT16"},
  {ge::DataType::DT_RESOURCE,       "DT_RESOURCE"},
  {ge::DataType::DT_STRING_REF,     "DT_STRING_REF"},
  {ge::DataType::DT_DUAL,           "DT_DUAL"},
  {ge::DataType::DT_VARIANT,        "DT_VARIANT"},
  {ge::DataType::DT_BF16,           "DT_BF16"},
  {ge::DataType::DT_UNDEFINED,      "DT_UNDEFINED"},
  {ge::DataType::DT_INT4,           "DT_INT4"},
  {ge::DataType::DT_UINT1,          "DT_UINT1"},
  {ge::DataType::DT_INT2,           "DT_INT2"},
  {ge::DataType::DT_UINT2,          "DT_UINT2"}
};

const std::map<std::string, ge::Format> kStringToDataFormatMap = {
  {"FORMAT_NCHW",                            ge::Format::FORMAT_NCHW},
  {"FORMAT_NHWC",                            ge::Format::FORMAT_NHWC},
  {"FORMAT_ND",                              ge::Format::FORMAT_ND},
  {"FORMAT_NC1HWC0",                         ge::Format::FORMAT_NC1HWC0},
  {"FORMAT_FRACTAL_Z",                       ge::Format::FORMAT_FRACTAL_Z},
  {"FORMAT_NC1C0HWPAD",                      ge::Format::FORMAT_NC1C0HWPAD},
  {"FORMAT_NHWC1C0",                         ge::Format::FORMAT_NHWC1C0},
  {"FORMAT_FSR_NCHW",                        ge::Format::FORMAT_FSR_NCHW},
  {"FORMAT_FRACTAL_DECONV",                  ge::Format::FORMAT_FRACTAL_DECONV},
  {"FORMAT_C1HWNC0",                         ge::Format::FORMAT_C1HWNC0},
  {"FORMAT_FRACTAL_DECONV_TRANSPOSE",        ge::Format::FORMAT_FRACTAL_DECONV_TRANSPOSE},
  {"FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS",  ge::Format::FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS},
  {"FORMAT_NC1HWC0_C04",                     ge::Format::FORMAT_NC1HWC0_C04},
  {"FORMAT_FRACTAL_Z_C04",                   ge::Format::FORMAT_FRACTAL_Z_C04},
  {"FORMAT_CHWN",                            ge::Format::FORMAT_CHWN},
  {"FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS", ge::Format::FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS},
  {"FORMAT_HWCN",                            ge::Format::FORMAT_HWCN},
  {"FORMAT_NC1KHKWHWC0",                     ge::Format::FORMAT_NC1KHKWHWC0},
  {"FORMAT_BN_WEIGHT",                       ge::Format::FORMAT_BN_WEIGHT},
  {"FORMAT_FILTER_HWCK",                     ge::Format::FORMAT_FILTER_HWCK},
  {"FORMAT_HASHTABLE_LOOKUP_LOOKUPS",        ge::Format::FORMAT_HASHTABLE_LOOKUP_LOOKUPS},
  {"FORMAT_HASHTABLE_LOOKUP_KEYS",           ge::Format::FORMAT_HASHTABLE_LOOKUP_KEYS},
  {"FORMAT_HASHTABLE_LOOKUP_VALUE",          ge::Format::FORMAT_HASHTABLE_LOOKUP_VALUE},
  {"FORMAT_HASHTABLE_LOOKUP_OUTPUT",         ge::Format::FORMAT_HASHTABLE_LOOKUP_OUTPUT},
  {"FORMAT_HASHTABLE_LOOKUP_HITS",           ge::Format::FORMAT_HASHTABLE_LOOKUP_HITS},
  {"FORMAT_C1HWNCoC0",                       ge::Format::FORMAT_C1HWNCoC0},
  {"FORMAT_MD",                              ge::Format::FORMAT_MD},
  {"FORMAT_NDHWC",                           ge::Format::FORMAT_NDHWC},
  {"FORMAT_FRACTAL_ZZ",                      ge::Format::FORMAT_FRACTAL_ZZ},
  {"FORMAT_FRACTAL_NZ",                      ge::Format::FORMAT_FRACTAL_NZ},
  {"FORMAT_NCDHW",                           ge::Format::FORMAT_NCDHW},
  {"FORMAT_DHWCN",                           ge::Format::FORMAT_DHWCN},
  {"FORMAT_NDC1HWC0",                        ge::Format::FORMAT_NDC1HWC0},
  {"FORMAT_FRACTAL_Z_3D",                    ge::Format::FORMAT_FRACTAL_Z_3D},
  {"FORMAT_CN",                              ge::Format::FORMAT_CN},
  {"FORMAT_NC",                              ge::Format::FORMAT_NC},
  {"FORMAT_DHWNC",                           ge::Format::FORMAT_DHWNC},
  {"FORMAT_FRACTAL_Z_3D_TRANSPOSE",          ge::Format::FORMAT_FRACTAL_Z_3D_TRANSPOSE},
  {"FORMAT_FRACTAL_ZN_LSTM",                 ge::Format::FORMAT_FRACTAL_ZN_LSTM},
  {"FORMAT_FRACTAL_Z_G",                     ge::Format::FORMAT_FRACTAL_Z_G},
  {"FORMAT_RESERVED",                        ge::Format::FORMAT_RESERVED},
  {"FORMAT_ALL",                             ge::Format::FORMAT_ALL},
  {"FORMAT_NULL",                            ge::Format::FORMAT_NULL},
  {"FORMAT_ND_RNN_BIAS",                     ge::Format::FORMAT_ND_RNN_BIAS},
  {"FORMAT_FRACTAL_ZN_RNN",                  ge::Format::FORMAT_FRACTAL_ZN_RNN},
  {"FORMAT_NYUV",                            ge::Format::FORMAT_NYUV},
  {"FORMAT_NYUV_A",                          ge::Format::FORMAT_NYUV_A}
};

const std::map<ge::Format, std::string> kDataFormatToStringMap = {
  {ge::Format::FORMAT_NCHW,                            "FORMAT_NCHW"},
  {ge::Format::FORMAT_NHWC,                            "FORMAT_NHWC"},
  {ge::Format::FORMAT_ND,                              "FORMAT_ND"},
  {ge::Format::FORMAT_NC1HWC0,                         "FORMAT_NC1HWC0"},
  {ge::Format::FORMAT_FRACTAL_Z,                       "FORMAT_FRACTAL_Z"},
  {ge::Format::FORMAT_NC1C0HWPAD,                      "FORMAT_NC1C0HWPAD"},
  {ge::Format::FORMAT_NHWC1C0,                         "FORMAT_NHWC1C0"},
  {ge::Format::FORMAT_FSR_NCHW,                        "FORMAT_FSR_NCHW"},
  {ge::Format::FORMAT_FRACTAL_DECONV,                  "FORMAT_FRACTAL_DECONV"},
  {ge::Format::FORMAT_C1HWNC0,                         "FORMAT_C1HWNC0"},
  {ge::Format::FORMAT_FRACTAL_DECONV_TRANSPOSE,        "FORMAT_FRACTAL_DECONV_TRANSPOSE"},
  {ge::Format::FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS,  "FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS"},
  {ge::Format::FORMAT_NC1HWC0_C04,                     "FORMAT_NC1HWC0_C04"},
  {ge::Format::FORMAT_FRACTAL_Z_C04,                   "FORMAT_FRACTAL_Z_C04"},
  {ge::Format::FORMAT_CHWN,                            "FORMAT_CHWN"},
  {ge::Format::FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS, "FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS"},
  {ge::Format::FORMAT_HWCN,                            "FORMAT_HWCN"},
  {ge::Format::FORMAT_NC1KHKWHWC0,                     "FORMAT_NC1KHKWHWC0"},
  {ge::Format::FORMAT_BN_WEIGHT,                       "FORMAT_BN_WEIGHT"},
  {ge::Format::FORMAT_FILTER_HWCK,                     "FORMAT_FILTER_HWCK"},
  {ge::Format::FORMAT_HASHTABLE_LOOKUP_LOOKUPS,        "FORMAT_HASHTABLE_LOOKUP_LOOKUPS"},
  {ge::Format::FORMAT_HASHTABLE_LOOKUP_KEYS,           "FORMAT_HASHTABLE_LOOKUP_KEYS"},
  {ge::Format::FORMAT_HASHTABLE_LOOKUP_VALUE,          "FORMAT_HASHTABLE_LOOKUP_VALUE"},
  {ge::Format::FORMAT_HASHTABLE_LOOKUP_OUTPUT,         "FORMAT_HASHTABLE_LOOKUP_OUTPUT"},
  {ge::Format::FORMAT_HASHTABLE_LOOKUP_HITS,           "FORMAT_HASHTABLE_LOOKUP_HITS"},
  {ge::Format::FORMAT_C1HWNCoC0,                       "FORMAT_C1HWNCoC0"},
  {ge::Format::FORMAT_MD,                              "FORMAT_MD"},
  {ge::Format::FORMAT_NDHWC,                           "FORMAT_NDHWC"},
  {ge::Format::FORMAT_FRACTAL_ZZ,                      "FORMAT_FRACTAL_ZZ"},
  {ge::Format::FORMAT_FRACTAL_NZ,                      "FORMAT_FRACTAL_NZ"},
  {ge::Format::FORMAT_NCDHW,                           "FORMAT_NCDHW"},
  {ge::Format::FORMAT_DHWCN,                           "FORMAT_DHWCN"},
  {ge::Format::FORMAT_NDC1HWC0,                        "FORMAT_NDC1HWC0"},
  {ge::Format::FORMAT_FRACTAL_Z_3D,                    "FORMAT_FRACTAL_Z_3D"},
  {ge::Format::FORMAT_CN,                              "FORMAT_CN"},
  {ge::Format::FORMAT_NC,                              "FORMAT_NC"},
  {ge::Format::FORMAT_DHWNC,                           "FORMAT_DHWNC"},
  {ge::Format::FORMAT_FRACTAL_Z_3D_TRANSPOSE,          "FORMAT_FRACTAL_Z_3D_TRANSPOSE"},
  {ge::Format::FORMAT_FRACTAL_ZN_LSTM,                 "FORMAT_FRACTAL_ZN_LSTM"},
  {ge::Format::FORMAT_FRACTAL_Z_G,                     "FORMAT_FRACTAL_Z_G"},
  {ge::Format::FORMAT_RESERVED,                        "FORMAT_RESERVED"},
  {ge::Format::FORMAT_ALL,                             "FORMAT_ALL"},
  {ge::Format::FORMAT_NULL,                            "FORMAT_NULL"},
  {ge::Format::FORMAT_ND_RNN_BIAS,                     "FORMAT_ND_RNN_BIAS"},
  {ge::Format::FORMAT_FRACTAL_ZN_RNN,                  "FORMAT_FRACTAL_ZN_RNN"},
  {ge::Format::FORMAT_NYUV,                            "FORMAT_NYUV"},
  {ge::Format::FORMAT_NYUV_A,                          "FORMAT_NYUV_A"}
};

bool IsSocVersionSupported(const std::string& version) {
  return IsSocVersionMLR1(version);
}

bool IsSocVersionMLR1(const std::string& version) {
  auto iter = std::find(
      kSocVersionMLR1.begin(), kSocVersionMLR1.end(), version);
  DVPP_CHECK_IF_THEN_DO(iter == kSocVersionMLR1.end(), return false);
  return true;
}

DvppErrorCode StringToBool(const std::string& str, bool& result) {
  result = false;
  std::string value = str;
  try {
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    if ((value == "true") || (value == "false")) {
      std::istringstream(value) >> std::boolalpha >> result;
      return DvppErrorCode::kSuccess;
    }
  } catch (std::exception& e) {
    // something else reason
    return DvppErrorCode::kUnknown;
  }
  return DvppErrorCode::kUnknown;
}

void StringToDataType(const std::string& str,
                      std::set<ge::DataType>& data_type) {
  std::set<std::string> data_type_strs;
  SplitSequence(str, kComma, data_type_strs);

  for (auto& data_type_str : data_type_strs) {
    auto iter = kStringToDataTypeMap.find(data_type_str);
    if (iter != kStringToDataTypeMap.end()) {
      data_type.insert(iter->second);
    }
  }
}

bool DataTypeToString(ge::DataType data_type, std::string& str) {
  auto iter = kDataTypeToStringMap.find(data_type);
  DVPP_CHECK_IF_THEN_DO((iter == kDataTypeToStringMap.end()),
      str = "DT_INVALID";
      return false);

  str = iter->second;
  return true;
}

void StringToDataFormat(const std::string& str,
                        std::set<ge::Format>& data_format) {
  std::set<std::string> data_format_strs;
  SplitSequence(str, kComma, data_format_strs);

  for (auto& data_format_str : data_format_strs) {
    auto iter = kStringToDataFormatMap.find(data_format_str);
    if (iter != kStringToDataFormatMap.end()) {
      data_format.insert(iter->second);
    }
  }
}

bool DataFormatToString(ge::Format data_format, std::string& str) {
  auto iter = kDataFormatToStringMap.find(data_format);
  DVPP_CHECK_IF_THEN_DO((iter == kDataFormatToStringMap.end()),
      str = "FORMAT_INVALID";
      return false);

  str = iter->second;
  return true;
}

bool GetDataTypeSize(ge::DataType data_type, int32_t& data_type_size) {
  auto iter = kDataTypeSizeMap.find(data_type);
  DVPP_CHECK_IF_THEN_DO((iter == kDataTypeSizeMap.end()),
      data_type_size = 0;
      return false);

  data_type_size = iter->second;
  return true;
}

void SplitSequence(const std::string& str, const std::string& pattern,
                   std::set<std::string>& result) {
  // Easy to intercept the last piece of data
  std::string temp_str = str + pattern;

  size_t pos = temp_str.find(pattern);
  size_t size = temp_str.size();

  while (pos != std::string::npos) {
    std::string x = temp_str.substr(0, pos);
    if (!x.empty()) {
      (void)result.emplace(x);
    }

    temp_str = temp_str.substr(pos + pattern.length(), size);
    pos = temp_str.find(pattern);
  }
}

void SplitSequence(const std::string& str, const std::string& pattern,
                   std::vector<int64_t>& result) {
  // Easy to intercept the last piece of data
  std::string temp_str = str + pattern;

  size_t pos = temp_str.find(pattern);
  size_t size = temp_str.size();

  while (pos != std::string::npos) {
    std::string x = temp_str.substr(0, pos);
    if (!x.empty()) {
      std::stringstream tmp_ss(x);
      int64_t temp_integer;
      char tmp_char;
      if ((tmp_ss >> temp_integer) && !(tmp_ss >> tmp_char)) { // 判断是否为整数
        (void)result.push_back(temp_integer);
      }
    }

    temp_str = temp_str.substr(pos + pattern.length(), size);
    pos = temp_str.find(pattern);
  }
}

uint64_t GenerateUniqueKernelId() {
  static uint64_t kernel_id = 0;
  // 这里即使回绕也没有安全隐患 因为足够大 业务不会出问题
  return ++kernel_id;
}

uint64_t GenerateUniqueSessionId() {
  static uint64_t session_id = 0;
  // 这里即使回绕也没有安全隐患 因为足够大 业务不会出问题
  return ++session_id;
}

bool GetAttrValueString(const ge::OpDescPtr& op_desc_ptr,
                        const std::string& attr_name,
                        dvppops::AttrValue& attr_value) {
  std::string string_value;
  bool ret = ge::AttrUtils::GetStr(op_desc_ptr, attr_name, string_value);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::GetStr function failed to get op[%s] attr[%s]",
          op_desc_ptr->GetType().c_str(), attr_name.c_str());
      return false);

  attr_value.set_s(string_value);
  return true;
}

bool GetAttrValueBool(const ge::OpDescPtr& op_desc_ptr,
                      const std::string& attr_name,
                      dvppops::AttrValue& attr_value) {
  bool bool_value = false;
  bool ret = ge::AttrUtils::GetBool(op_desc_ptr, attr_name, bool_value);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::GetBool function failed to get op[%s] attr[%s]",
          op_desc_ptr->GetType().c_str(), attr_name.c_str());
      return false);

  attr_value.set_b(bool_value);
  return true;
}

bool GetAttrValueInt(const ge::OpDescPtr& op_desc_ptr,
                     const std::string& attr_name,
                     dvppops::AttrValue& attr_value) {
  int64_t int_value = 0;
  bool ret = ge::AttrUtils::GetInt(op_desc_ptr, attr_name, int_value);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::GetInt function failed to get op[%s] attr[%s]",
          op_desc_ptr->GetType().c_str(), attr_name.c_str());
      return false);

  attr_value.set_i(int_value);
  return true;
}

bool GetAttrValue(const ge::OpDescPtr& op_desc_ptr,
                  const std::string& attr_name,
                  const ge::GeAttrValue::ValueType& ge_value_type,
                  dvppops::AttrValue& attr_value) {
  bool ret = false;
  switch (ge_value_type) {
    case ge::GeAttrValue::ValueType::VT_STRING: { // 1
      ret = GetAttrValueString(op_desc_ptr, attr_name, attr_value);
      break;
    }
    case ge::GeAttrValue::ValueType::VT_BOOL: { // 3
      ret = GetAttrValueBool(op_desc_ptr, attr_name, attr_value);
      break;
    }
    case ge::GeAttrValue::ValueType::VT_INT: { // 4
      ret = GetAttrValueInt(op_desc_ptr, attr_name, attr_value);
      break;
    }
    default: {
      // ge_value_type 类型有很多 按照Dvpp实际需求支持
      DVPP_ENGINE_LOG_WARNING(
          "op[%s] Currently can not support attr value type of [%d]",
          op_desc_ptr->GetType().c_str(), ge_value_type);
      ret = false;
      break;
    }
  }

  return ret;
}

bool CheckAttrValueString(const ge::OpDescPtr& op_desc_ptr,
                          const std::string& attr_name,
                          const std::string& check_value) {
  std::string string_value;
  bool ret = ge::AttrUtils::GetStr(op_desc_ptr, attr_name, string_value);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::GetStr function failed to get op[%s] attr[%s]",
          op_desc_ptr->GetType().c_str(), attr_name.c_str());
      return false);

  // attr可能支持多种值 比如DecodeJpeg的dst_img_format
  std::set<std::string> result;
  SplitSequence(check_value, kComma, result);

  return (result.find(string_value) != result.end());
}

bool CheckAttrValueBool(const ge::OpDescPtr& op_desc_ptr,
                        const std::string& attr_name,
                        const std::string& check_value) {
  bool bool_value = false;
  bool ret = ge::AttrUtils::GetBool(op_desc_ptr, attr_name, bool_value);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::GetBool function failed to get op[%s] attr[%s]",
          op_desc_ptr->GetType().c_str(), attr_name.c_str());
      return false);

  bool check_value_bool = false;
  auto error_code = StringToBool(check_value, check_value_bool);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("string[%s] to bool failed", check_value.c_str());
      return false);

  return (bool_value == check_value_bool);
}

bool CheckAttrValueInt(const ge::OpDescPtr& op_desc_ptr,
                       const std::string& attr_name,
                       const std::string& check_value) {
  int64_t int_value = 0;
  bool ret = ge::AttrUtils::GetInt(op_desc_ptr, attr_name, int_value);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::GetInt function failed to get op[%s] attr[%s]",
          op_desc_ptr->GetType().c_str(), attr_name.c_str());
      return false);

  DVPP_ENGINE_LOG_DEBUG("check_value %s, int_value %ld",
      check_value.c_str(), int_value);
  // 表示取值范围
  size_t pos = check_value.find(kTilde);
  if (pos != std::string::npos) {
    std::vector<int64_t> result_vector;
    SplitSequence(check_value, kTilde, result_vector);
    DVPP_CHECK_IF_THEN_DO((result_vector.size() != kNum2),
        DVPP_REPORT_INNER_ERR_MSG(
            "ListListInt limit value num %zu should be 2,  op[%s] attr[%s]",
            result_vector.size(), op_desc_ptr->GetType().c_str(),
            attr_name.c_str());
        return false);
    return ((int_value >= result_vector[kNum0]) &&
            (int_value <= result_vector[kNum1]));
  }

  // 表示取值列表
  std::set<std::string> result;
  SplitSequence(check_value, kComma, result);
  return (result.find(std::to_string(int_value)) != result.end());
}

bool CheckAttrValueDataType(const ge::OpDescPtr& op_desc_ptr,
                            const std::string& attr_name,
                            const std::string& check_value) {
  ge::DataType data_type_value;
  bool ret = ge::AttrUtils::GetDataType(
      op_desc_ptr, attr_name, data_type_value);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::GetDataType failed to get op[%s] attr[%s]",
          op_desc_ptr->GetType().c_str(), attr_name.c_str());
      return false);

  std::set<ge::DataType> check_data_type;
  StringToDataType(check_value, check_data_type);
  // C++ 20 support contains
  return (check_data_type.find(data_type_value) != check_data_type.end());
}

bool CheckAttrValueListInt(const ge::OpDescPtr& op_desc_ptr,
                           const std::string& attr_name,
                           const std::string& check_value) {
  std::vector<int64_t> list_int_value;
  bool ret = ge::AttrUtils::GetListInt(
      op_desc_ptr, attr_name, list_int_value);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::GetListInt failed to get op[%s] attr[%s]",
          op_desc_ptr->GetType().c_str(), attr_name.c_str());
      return false);
  // 当前只有GaussianBlur算子有ListInt类型，暂且强校验
  DVPP_CHECK_IF_THEN_DO((list_int_value.size() != kNum2),
      DVPP_REPORT_INNER_ERR_MSG(
          "ListInt value size not equal %u, op[%s] attr[%s].", kNum2,
          op_desc_ptr->GetType().c_str(), attr_name.c_str());
      return false);

  std::set<std::string> result;
  SplitSequence(check_value, kComma, result);

  return (result.find(std::to_string(list_int_value[0])) != result.end()) &&
         (result.find(std::to_string(list_int_value[1])) != result.end());
}

bool CheckAttrValue(const ge::OpDescPtr& op_desc_ptr,
                    const std::string& attr_name,
                    const ge::GeAttrValue::ValueType& ge_value_type,
                    const std::string& check_value) {
  bool ret = false;
  switch (ge_value_type) {
    case ge::GeAttrValue::ValueType::VT_STRING: { // 1
      ret = CheckAttrValueString(op_desc_ptr, attr_name, check_value);
      break;
    }
    case ge::GeAttrValue::ValueType::VT_BOOL: { // 3
      ret = CheckAttrValueBool(op_desc_ptr, attr_name, check_value);
      break;
    }
    case ge::GeAttrValue::ValueType::VT_INT: { // 4
      ret = CheckAttrValueInt(op_desc_ptr, attr_name, check_value);
      break;
    }
    case ge::GeAttrValue::ValueType::VT_DATA_TYPE: { // 11
      ret = CheckAttrValueDataType(op_desc_ptr, attr_name, check_value);
      break;
    }
    case ge::GeAttrValue::ValueType::VT_LIST_LIST_INT: { // 10
      ret = true; // 此类型检测不通用，在各个算子内部校验
      break;
    }
    case ge::GeAttrValue::ValueType::VT_LIST_INT: { // 1004
      ret = CheckAttrValueListInt(op_desc_ptr, attr_name, check_value);
      break;
    }
    default: {
      // ge_value_type 类型有很多 按照Dvpp实际需求支持
      DVPP_ENGINE_LOG_WARNING(
          "op[%s] Currently can not support attr value type of [%d]",
          op_desc_ptr->GetType().c_str(), ge_value_type);
      ret = false;
      break;
    }
  }

  return ret;
}

void PrintDvppOpInfoInputOutput(DvppOpInfo& dvpp_op_info) {
  for (auto iter = dvpp_op_info.inputs.begin();
       iter != dvpp_op_info.inputs.end(); ++iter) {
    DVPP_ENGINE_LOG_DEBUG("input[%s]", iter->first.c_str());
    DVPP_ENGINE_LOG_DEBUG("name[%s]", iter->second.dataName.c_str());
    DVPP_ENGINE_LOG_DEBUG("type[%s]", iter->second.dataType.c_str());
    DVPP_ENGINE_LOG_DEBUG("format[%s]", iter->second.dataFormat.c_str());
  }

  for (auto iter = dvpp_op_info.outputs.begin();
       iter != dvpp_op_info.outputs.end(); ++iter) {
    DVPP_ENGINE_LOG_DEBUG("output[%s]", iter->first.c_str());
    DVPP_ENGINE_LOG_DEBUG("name[%s]", iter->second.dataName.c_str());
    DVPP_ENGINE_LOG_DEBUG("type[%s]", iter->second.dataType.c_str());
    DVPP_ENGINE_LOG_DEBUG("format[%s]", iter->second.dataFormat.c_str());
  }
}

void PrintDvppOpInfoAttr(DvppOpInfo& dvpp_op_info) {
  for (auto iter = dvpp_op_info.attrs.begin();
       iter != dvpp_op_info.attrs.end(); ++iter) {
    DVPP_ENGINE_LOG_DEBUG("attr[%s]", iter->first.c_str());
    DVPP_ENGINE_LOG_DEBUG("type[%s]", iter->second.type.c_str());
    DVPP_ENGINE_LOG_DEBUG("value[%s]", iter->second.value.c_str());
  }
}

void PrintDvppOpInfo(std::map<std::string, DvppOpInfo>& dvpp_ops) {
  DVPP_ENGINE_LOG_DEBUG("start print dvpp op info.");
  for (auto iter = dvpp_ops.begin(); iter != dvpp_ops.end(); ++iter) {
    DVPP_ENGINE_LOG_DEBUG("op[%s].", iter->first.c_str());

    DVPP_ENGINE_LOG_DEBUG("engine[%s].", iter->second.engine.c_str());

    DVPP_ENGINE_LOG_DEBUG("opKernelLib[%s].", iter->second.opKernelLib.c_str());

    DVPP_ENGINE_LOG_DEBUG("computeCost[%d].", iter->second.computeCost);

    DVPP_CHECK_IF_THEN_DO(iter->second.flagPartial,
        DVPP_ENGINE_LOG_DEBUG("flagPartial[true]"));
    DVPP_CHECK_IF_THEN_DO(!iter->second.flagPartial,
        DVPP_ENGINE_LOG_DEBUG("flagPartial[false]"));

    DVPP_CHECK_IF_THEN_DO(iter->second.flagAsync,
        DVPP_ENGINE_LOG_DEBUG("flagAsync[true]"));
    DVPP_CHECK_IF_THEN_DO(!iter->second.flagAsync,
        DVPP_ENGINE_LOG_DEBUG("flagAsync[false]"));

    DVPP_ENGINE_LOG_DEBUG("kernelSo[%s]", iter->second.kernelSo.c_str());

    DVPP_ENGINE_LOG_DEBUG("functionName[%s]",
                          iter->second.functionName.c_str());

    DVPP_CHECK_IF_THEN_DO(iter->second.userDefined,
        DVPP_ENGINE_LOG_DEBUG("userDefined[true]"));
    DVPP_CHECK_IF_THEN_DO(!iter->second.userDefined,
        DVPP_ENGINE_LOG_DEBUG("userDefined[false]"));

    DVPP_ENGINE_LOG_DEBUG("workspaceSize[%d]", iter->second.workspaceSize);

    DVPP_ENGINE_LOG_DEBUG("resource[%s]", iter->second.resource.c_str());

    DVPP_ENGINE_LOG_DEBUG("memoryType[%s]", iter->second.memoryType.c_str());

    DVPP_ENGINE_LOG_DEBUG("widthMin[%ld]", iter->second.widthMin);

    DVPP_ENGINE_LOG_DEBUG("widthMax[%ld]", iter->second.widthMax);

    DVPP_ENGINE_LOG_DEBUG("widthAlign[%ld]", iter->second.widthAlign);

    DVPP_ENGINE_LOG_DEBUG("heightMin[%ld]", iter->second.heightMin);

    DVPP_ENGINE_LOG_DEBUG("heightMax[%ld]", iter->second.heightMax);

    DVPP_ENGINE_LOG_DEBUG("heightAlign[%ld]", iter->second.heightAlign);

    // print info::(inputs & outputs)
    PrintDvppOpInfoInputOutput(iter->second);

    // print info::attrs
    PrintDvppOpInfoAttr(iter->second);
  }
  DVPP_ENGINE_LOG_DEBUG("dvpp ops total num is: %zu.", dvpp_ops.size());
  DVPP_ENGINE_LOG_DEBUG("end print dvpp op info.");
}

void GetDimsValueByDataFormat(const std::vector<int64_t>& dims,
    ge::Format data_format, int64_t& n, int64_t& c, int64_t& width,
    int64_t& height) {
  switch (data_format) {
    case ge::Format::FORMAT_NCHW: {
      DVPP_CHECK_IF_THEN_DO(dims.size() == kNum4,
          n = dims[kNum0];
          c = dims[kNum1];
          width = dims[kNum3];
          height = dims[kNum2]);

      // CHW 因为GE未定义CHW类型 所以需要内部区分
      DVPP_CHECK_IF_THEN_DO(dims.size() == kNum3,
          n = kDynamicDim;
          c = dims[kNum0];
          width = dims[kNum2];
          height = dims[kNum1]);
      break;
    }
    case ge::Format::FORMAT_NYUV:
    case ge::Format::FORMAT_NHWC: {
      DVPP_CHECK_IF_THEN_DO(dims.size() == kNum4,
          n = dims[kNum0];
          c = dims[kNum3];
          width = dims[kNum2];
          height = dims[kNum1]);

      // HWC 因为GE未定义HWC类型 所以需要内部区分
      DVPP_CHECK_IF_THEN_DO(dims.size() == kNum3,
          n = kDynamicDim;
          c = dims[kNum2];
          width = dims[kNum1];
          height = dims[kNum0]);
      break;
    }
    default: {
      // data_format 类型有很多 按照Dvpp实际需求支持
      std::string data_format_str;
      (void)DataFormatToString(data_format, data_format_str);
      DVPP_ENGINE_LOG_WARNING("dvpp does not support data format [%s]",
                              data_format_str.c_str());
      break;
    }
  }
}

bool InferDataFormatND(const std::vector<int64_t>& dims,
    ge::Format& data_format, std::string& unsupported_reason) {
  // 仅FORMAT_ND需要推测
  DVPP_CHECK_IF_THEN_DO(data_format != ge::Format::FORMAT_ND, return true);

  DVPP_CHECK_IF_THEN_DO((dims.size() != kNum4) && (dims.size() != kNum3),
      unsupported_reason = "dims size[" + std::to_string(dims.size()) +
                           "] should be 3 or 4.";
      return false);

  // 通过C的值去推测
  // FORMAT_NHWC
  DVPP_CHECK_IF_THEN_DO((dims.size() == kNum4) &&
                        ((dims[kNum3] == kNum1) || (dims[kNum3] == kNum3)),
      data_format = ge::Format::FORMAT_NHWC;
      return true);

  // FORMAT_HWC 因为没有该定义 也使用 FORMAT_NHWC
  DVPP_CHECK_IF_THEN_DO((dims.size() == kNum3) &&
                        ((dims[kNum2] == kNum1) || (dims[kNum2] == kNum3)),
      data_format = ge::Format::FORMAT_NHWC;
      return true);

  // FORMAT_NCHW
  DVPP_CHECK_IF_THEN_DO((dims.size() == kNum4) &&
                        ((dims[kNum1] == kNum1) || (dims[kNum1] == kNum3)),
      data_format = ge::Format::FORMAT_NCHW;
      return true);

  // FORMAT_CHW 因为没有该定义 也使用 FORMAT_NCHW
  DVPP_CHECK_IF_THEN_DO((dims.size() == kNum3) &&
                        ((dims[kNum0] == kNum1) || (dims[kNum0] == kNum3)),
      data_format = ge::Format::FORMAT_NCHW;
      return true);

  // 不支持
  unsupported_reason = "do not support FORMAT_ND infer";
  return false;
}

void GetShapeRangeByDataFormat(std::vector<std::pair<int64_t, int64_t>>& range,
    const std::vector<int64_t>& dims, ge::Format data_format) {
  constexpr int64_t anyValue = -1; // -1:表示最大可以是任意值
  switch (data_format) {
    case ge::Format::FORMAT_NCHW: {
      DVPP_CHECK_IF_THEN_DO(dims.size() == kNum4,
          range.emplace_back(std::make_pair(kNum1, kNum1));
          range.emplace_back(std::make_pair(kNum1, kNum3));
          range.emplace_back(std::make_pair(kNum1, anyValue));
          range.emplace_back(std::make_pair(kNum1, anyValue)));

      // CHW 因为GE未定义CHW类型 所以需要内部区分
      DVPP_CHECK_IF_THEN_DO(dims.size() == kNum3,
          range.emplace_back(std::make_pair(kNum1, kNum3));
          range.emplace_back(std::make_pair(kNum1, anyValue));
          range.emplace_back(std::make_pair(kNum1, anyValue)));
      break;
    }
    case ge::Format::FORMAT_NHWC: {
      DVPP_CHECK_IF_THEN_DO(dims.size() == kNum4,
          range.emplace_back(std::make_pair(kNum1, kNum1));
          range.emplace_back(std::make_pair(kNum1, anyValue));
          range.emplace_back(std::make_pair(kNum1, anyValue));
          range.emplace_back(std::make_pair(kNum1, kNum3)));

      // HWC 因为GE未定义HWC类型 所以需要内部区分
      DVPP_CHECK_IF_THEN_DO(dims.size() == kNum3,
          range.emplace_back(std::make_pair(kNum1, anyValue));
          range.emplace_back(std::make_pair(kNum1, anyValue));
          range.emplace_back(std::make_pair(kNum1, kNum3)));
      break;
    }
    default: {
      // 按照dim size设置对应的range
      for (auto it = dims.begin(); it != dims.end(); ++it) {
        range.emplace_back(std::make_pair(kNum1, anyValue));
      }
      break;
    }
  }
}

std::string RealPath(const std::string &path) {
  if (path.empty()) {
    DVPP_ENGINE_LOG_INFO("path string is nullptr.");
    return "";
  }
  if (path.size() >= PATH_MAX) {
    DVPP_ENGINE_LOG_INFO("file path %s is too long!", path.c_str());
    return "";
  }

  // PATH_MAX is the system marco，indicate the maximum length for file path
  // pclint check，one param in stack can not exceed 1K bytes
  char resoved_path[PATH_MAX] = {0x00};

  // path not exists or not allowed to read，return nullptr
  // path exists and readable, return the resoved path
  std::string res = "";
  if (realpath(path.c_str(), resoved_path) != nullptr) {
    res = resoved_path;
  } else {
    DVPP_ENGINE_LOG_INFO("path %s is not exist.", path.c_str());
  }

  return res;
}
} // namespace dvpp
