/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_UTIL_H
#define AICPU_UTIL_H

#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <string>
#include "securec.h"
#include "error_code/error_code.h"
#include "ge/ge_api_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/types.h"
#include "log.h"
#include "aicpu_ops_kernel_info_store/op_struct.h"
#include "common/sgt_slice_type.h"
#include "mmpa/mmpa_api.h"
namespace aicpu {
/**
 * two input shapes is same or not
 * @param first_shape, first shape
 * @param second_shape, second shape
 * @return whether same
 */
bool IsSameShape(const ge::GeShape &first_shape, const ge::GeShape &second_shape);

/**
 * get .so file path
 * @param instance, src file
 * @return file path
 */
const std::string GetSoPath(const void *instance);

/**
 * convert string to num
 * @param str, src string
 * @param num, goal num
 * @return whether convert string to num successfully
 */
template <typename Type>
aicpu::State StringToNum(const std::string &str, Type &num) {
  std::istringstream iss(str);
  iss >> num;
  try {
    if (str == std::to_string(num)) {
      return aicpu::State(ge::SUCCESS);
    }
  } catch (const std::bad_alloc &e) {
    return aicpu::State(ge::FAILED, e.what());
  }
  return aicpu::State(ge::FAILED, "unknown reason");
}

/**
 * read data from json file
 * @param file_path, src file path
 * @param json_read, Json data read from src file
 * @return whether read json file successfully
 */
aicpu::State ReadJsonFile(const std::string &file_path, nlohmann::json &json_read);

/**
 * Convert string to bool.
 * @param str The source string
 * @param result string to bool
 * @return bool convert success or fail
 */
aicpu::State StringToBool(const std::string &str, bool &result);

/*
 * str cat util function
 * param[in] params need concat to string
 * return concatted string
 */
template <typename T>
std::string Stringcat(const T arg) {
  std::stringstream oss;
  oss << arg;
  return oss.str();
}

/**
 * make two string into one
 * @param arg, string one
 * @param arg_left, other string
 * @return result string
 */
template <typename T, typename... Tstr>
std::string Stringcat(T arg, Tstr... arg_left) {
  std::ostringstream oss;
  oss << arg;
  oss << Stringcat(arg_left...);
  return oss.str();
}

/**
 * Split string sequence contains specific separators
 * @param str Source string to be splitted
 * @param pattern Separators collections
 * @param set<string> result, string sequence removed separator
 * @return
 */
void SplitSequence(const std::string &str, const std::string &pattern,
                   std::set<std::string> &result);

/** Get dataType corresponding to dataTypeName from data_types map
 * @param data_types All dataType info
 * @param data_type_name DataType name
 * @param data_type DataType
 */
void GetDataType(const std::map<std::string, std::string> &data_types,
                 const std::string &data_type_name,
                 std::set<ge::DataType> &data_type);

/** Get dataType corresponding to dataTypeName from data_types map
 * @param data_types All dataType info
 * @param data_type_name DataType name
 * @param data_type DataType
 */
void GetDataType(const std::map<std::string, std::string> &data_types,
                 const std::string &data_type_name,
                 std::vector<ge::DataType> &data_type);

/**
 * Convert DataType to string.
 * @param elem
 * @param data_type
 * @return str
 */
bool ConvertDataType2String(std::string &elem, ge::DataType data_type);

/**
 * Get the ge::DataType's size
 * @param data_type dataType's information
 * @return DataType's size
 */
int32_t GetDataTypeSize(ge::DataType data_type);

/**
 * Check if overflow for int64's mul
 * @param a mul value a
 * @param b mul value b
 * @return true:overflow, false: normal
 */
bool CheckInt64MulOverflow(int64_t a, int64_t b);

/**
 * Check if overflow for uint64's add
 * @param a add value a
 * @param b add value b
 * @return true:overflow, false: normal
 */
bool CheckUint64AddOverflow(uint64_t a, uint64_t b);

/**
 * Check if overflow for uint32's add
 * @param a add value a
 * @param b add value b
 * @return true:overflow, false: normal
 */
bool CheckUint32AddOverflow(uint32_t a, uint32_t b);

/**
 * Check if overflow for int64's add
 * @param a add value a
 * @param b add value b
 * @return true:overflow, false: normal
 */
bool CheckInt64AddOverflow(int64_t a, int64_t b);

/**
 * Generate Ge Node
 * @param name node name
 * @param type op name
 * @param in_count input number
 * @param out_count output number
 * @param format node format
 * @param data_type node dataTypr
 * @param shape node shape
 * @return NodePtr
 */
ge::NodePtr GenGeNode(const std::string &name, const std::string &type,
                      int in_count, int out_count, ge::Format format,
                      ge::DataType data_type, std::vector<int64_t> shape);

/*
 * get debug string of vector
 * param[in] v vector
 * return vector debug string
 */
template <typename T>
std::string DebugString(const std::vector<T> &v) {
  std::ostringstream oss;
  oss << "[";
  if (v.size() > 0) {
    for (size_t i = 0U; i < v.size() - 1; ++i) {
      oss << v[i] << ", ";
    }
    oss << v[v.size() - 1];
  }
  oss << "]";
  return oss.str();
}

/**
 * Get current time
 * @return current time in str
 */
std::string CurrentTimeInStr();

/**
 * Read Bytes from BinaryFile
 * @param file name and file value
 * @return bool success or fail
 */
bool ReadBytesFromBinaryFile(const string &file_name,
                             std::vector<char> &buffer);

/**
 * Read realpath from BinaryFile
 * @param file path
 * @return real path
 */
const std::string RealPath(const std::string &path);

/**
 * Validate string
 * @param str character string
 * @param mode pattern mode
 * @return bool success or fail
 */
bool ValidateStr(const std::string &str, const std::string &mode);

/**
 * Get kernel lib name
 * @param op_type op type string
 * @param all_op_info op info
 * @return string lib name
 */
ge::string GetKernelLibNameByOpType(const string &op_type, const std::map<string, OpFullInfo> &all_op_info);

/**
 * Check is known shape
 * @param op_desc_ptr op desc ptr
 * @return bool is known or not
 */
bool IsUnknowShape(const ge::OpDescPtr &op_desc_ptr);

/**
 * Calculate and set outputs size
 * @param op_desc_ptr Ge op description pointer
 * @return whether handle success
 */
ge::Status SetOutPutsSize(std::shared_ptr<ge::OpDesc> &op_desc_ptr);

/**
 * Set outputs size for unknow type 4 which output is result summary
 * @param op_desc_ptr Ge op description pointer
 * @return whether handle success
 */
ge::Status SetOutSizeForSummary(std::shared_ptr<ge::OpDesc> &op_desc_ptr);

/**
 * Get the total size by datatype and ge shape
 * @param data_type Represent that how many size for each data
 * @param ge_shape Represent the dims information
 * @param total_size The total size
 * @return whether handle success
 */
State GetTotalSizeByShapeAndType(const ge::DataType &data_type,
                                 const ge::GeShape &ge_shape,
                                 int64_t &total_size);

/**
 * Get the total size by datatype and dims
 * @param data_type Represent that how many size for each data
 * @param dims Represent the shape dims information
 * @param total_size The total size
 * @return whether handle success
 */
State GetTotalSizeByDimsAndType(const ge::DataType &data_type,
                                const vector<int64_t> &dims,
                                int64_t &total_size);

/**
 * Calculate output size for unknow type 3 which output shape is range
 * @param data_type Represent that how many size for each data
 * @param shape_range outputs shape range for unknow type 3
 * @param total_size output total size
 * @return whether handle success
 */
ge::Status GetOutSizeByShapeRange(const ge::DataType &data_type,
    const std::vector<std::pair<int64_t, int64_t>> &shape_range,
    int64_t &total_size);

std::vector<std::string> SplitString(const std::string str, const std::string split);

void GetThreadTensorShape(const vector<vector<vector<ffts::DimRange>>> &shape_range,
                          vector<vector<vector<int64_t>>> &shape);

ge::Status GetOriginalType(const ge::OpDescPtr &op_desc_ptr, std::string &type);

bool IsNotiling(const ge::OpDescPtr &op_desc_ptr);

bool IsPathExist(const std::string &path);

std::string Trim(const string &source, const char delims);

void SplitLine(const string &str, const string &pattern, std::vector<string> &result);

bool ReadConfigFile(const string &file_path, std::vector<string> &result);

template <typename _Tp, typename... _Args>
static inline std::shared_ptr<_Tp> MakeShared(_Args &&... args) {
  using TpNc = typename std::remove_const<_Tp>::type;
  return std::make_shared<TpNc>(std::forward<_Args>(args)...);
}
}  // namespace aicpu

// Check the result,if the result is not SUCCESS, return directly
#define AICPU_CHECK_RES(res)  \
  do {                        \
    ge::Status ret = (res);   \
    if (ret != ge::SUCCESS) { \
      return ret;             \
    }                         \
  } while (0);

// Check the result,if the result is not SUCCESS, return directly
#define AICPU_CHECK_RES_WITH_LOG(res, err_desc...) \
  do {                                            \
    ge::Status ret = (res);                       \
    if (ret != ge::SUCCESS) {                     \
      AICPU_REPORT_INNER_ERR_MSG(err_desc);          \
      return ret;                                 \
    }                                             \
  } while (0);

#define AICPU_IF_BOOL_EXEC(expr, exec_expr...) \
  {                                            \
    if (expr) {                                \
      exec_expr;                               \
    }                                          \
  };

#define AICPU_CHECK_NOTNULL_ERRCODE(val, err_code)                 \
  do {                                                             \
    if ((val) == nullptr) {                                        \
      AICPU_REPORT_INNER_ERR_MSG("[%s] is null.", #val);             \
      return (err_code);                                           \
    }                                                              \
  } while (0);

// Validate the input number to ensure the value greater than 0
#define AICPU_CHECK_GREATER_THAN_ZERO(number, err_code, err_desc...) \
  do {                                                               \
    if ((number) <= 0) {                                             \
      AICPU_REPORT_INNER_ERR_MSG(err_desc);                            \
      return (err_code);                                             \
    }                                                                \
  } while (0);

  // Validate the shape_range to ensure the value greater than or equal to 0
#define AICPU_CHECK_SHAPE_RANGE_GREATER_THAN_OR_EQUAL_TO_ZERO(number, err_code, err_desc...) \
  do {                                                               \
    if ((number) < 0) {                                             \
      AICPU_REPORT_INNER_ERR_MSG(err_desc);                            \
      return (err_code);                                             \
    }                                                                \
  } while (0);

// Validate whether the value1 equals value2
#define CHECK_EQUAL(value1, value2, err_code, err_desc...) \
  do {                                                     \
    if ((value1) != (value2)) {                            \
      AICPU_REPORT_INNER_ERR_MSG(err_desc);                  \
      return (err_code);                                   \
    }                                                      \
  } while (0);

// Validate whether the value1 add value2 is overflow
#define CHECK_UINT64_ADD_OVERFLOW(a, b, err_code, err_desc...) \
  do {                                                         \
    if (CheckUint64AddOverflow((a), (b))) {                    \
      AICPU_REPORT_INNER_ERR_MSG(err_desc);                      \
      return (err_code);                                       \
    }                                                          \
  } while (0);

// Validate whether the value1 add value2 is overflow
#define CHECK_INT64_ADD_OVERFLOW(a, b, err_code, err_desc...) \
  do {                                                        \
    if (CheckInt64AddOverflow((a), (b))) {                    \
      AICPU_REPORT_INNER_ERR_MSG(err_desc);                     \
      return (err_code);                                      \
    }                                                         \
  } while (0);

// Validate whether the value1 mul value2 is overflow
#define CHECK_INT64_MUL_OVERFLOW(a, b, err_code, err_desc...) \
  do {                                                        \
    if (CheckInt64MulOverflow((a), (b))) {                    \
      AICPU_REPORT_INNER_ERR_MSG(err_desc);                     \
      return (err_code);                                      \
    }                                                         \
  } while (0);

// Validate whether the value1 add value2 is overflow
#define CHECK_UINT32_ADD_OVERFLOW(a, b, err_code, err_desc...) \
  do {                                                         \
    if (CheckUint32AddOverflow((a), (b))) {                    \
      AICPU_REPORT_INNER_ERR_MSG(err_desc);                      \
      return (err_code);                                       \
    }                                                          \
  } while (0);

// Validate whether the return value is true
#define CHECK_RES_BOOL(res, err_code, exec_expr) \
  do {                                             \
    if (!(res)) {                                  \
      exec_expr;                                   \
      return (err_code);                           \
    }                                              \
  } while (0);

#define AICPU_MAKE_SHARED(execExpr0, exec_expr...) \
  try {                                            \
    execExpr0;                                     \
  } catch (...) {                                  \
    exec_expr;                                     \
  }

#define AICPU_CHECK_NOTNULL(val)                       \
  do {                                                 \
    if ((val) == nullptr) {                            \
      AICPU_REPORT_INNER_ERR_MSG("[%s] is null.", #val); \
      return OBJECT_IS_NULL;                           \
    }                                                  \
  } while (0);

#define AICPU_CHECK_FALSE_EXEC(expr, execExpr...) \
  {                                            \
    bool ret = (expr);                            \
    if (!ret) {                                   \
      execExpr;                                   \
    }                                             \
  };
#endif
