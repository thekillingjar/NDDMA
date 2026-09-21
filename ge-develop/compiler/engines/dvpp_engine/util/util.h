/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_ENGINE_UTIL_UTIL_H_
#define DVPP_ENGINE_UTIL_UTIL_H_

#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "common/dvpp_ops.h"
#include "proto/dvpp_node_def.pb.h"
#include "util/dvpp_error_code.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"

namespace dvpp {
bool IsSocVersionSupported(const std::string& version);
bool IsSocVersionMLR1(const std::string& version);

/**
 * convert string to num, NumToString use std::to_string
 * @param str: src string
 * @param num: goal num
 * @return whether convert string to num successfully
 */
template<typename T>
DvppErrorCode StringToNum(const std::string& str, T& num) {
  std::istringstream iss(str);
  iss >> num;
  try {
    if (str == std::to_string(num)) {
      return DvppErrorCode::kSuccess;
    }
  } catch (const std::bad_alloc& e) {
    return DvppErrorCode::kBadAlloc;
  }
  return DvppErrorCode::kUnknown;
}

template <typename _Tp, typename... _Args>
inline std::shared_ptr<_Tp> MakeShared(_Args &&... __args) {
  using _Tp_nc = typename std::remove_const<_Tp>::type;
  const std::shared_ptr<_Tp> ret(
      new (std::nothrow) _Tp_nc(std::forward<_Args>(__args)...));
  return ret;
}

DvppErrorCode StringToBool(const std::string& str, bool& result);

void StringToDataType(const std::string& str,
                      std::set<ge::DataType>& data_type);
bool DataTypeToString(ge::DataType data_type, std::string& str);

void StringToDataFormat(const std::string& str,
                        std::set<ge::Format>& data_format);
bool DataFormatToString(ge::Format data_format, std::string& str);

bool GetDataTypeSize(ge::DataType data_type, int32_t& data_type_size);

void SplitSequence(const std::string& str, const std::string& pattern,
                   std::set<std::string>& result);

void SplitSequence(const std::string& str, const std::string& pattern,
                   std::vector<int64_t>& result);

uint64_t GenerateUniqueKernelId();
uint64_t GenerateUniqueSessionId();

bool GetAttrValue(const ge::OpDescPtr& op_desc_ptr,
                  const std::string& attr_name,
                  const ge::GeAttrValue::ValueType& ge_value_type,
                  dvppops::AttrValue& attr_value);

bool CheckAttrValue(const ge::OpDescPtr& op_desc_ptr,
                    const std::string& attr_name,
                    const ge::GeAttrValue::ValueType& ge_value_type,
                    const std::string& check_value);

template<typename T>
bool IsInRange(const T& value, const T& min, const T& max) {
  return ((value >= min) && (value <= max));
}

void PrintDvppOpInfo(std::map<std::string, DvppOpInfo>& dvpp_ops);

void GetDimsValueByDataFormat(const std::vector<int64_t>& dims,
    ge::Format data_format, int64_t& n, int64_t& c, int64_t& width,
    int64_t& height);

bool InferDataFormatND(const std::vector<int64_t>& dims,
    ge::Format& data_format, std::string& unsupported_reason);

void GetShapeRangeByDataFormat(std::vector<std::pair<int64_t, int64_t>>& range,
    const std::vector<int64_t>& dims, ge::Format data_format);

std::string RealPath(const std::string &path);
} // namespace dvpp

#endif // DVPP_ENGINE_UTIL_UTIL_H_
