/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_COMMON_PRE_CHECKER_H_
#define PARSER_COMMON_PRE_CHECKER_H_

#include <string>
#include <vector>
#include "framework/omg/parser/parser_types.h"
#include "omg/omg_inner_types.h"

namespace ge {
using std::map;
using std::string;
using std::vector;
using Status = domi::Status;
/**
 * @ingroup domi_omg
 * @brief pre_check
 * @author
 */
class PreChecker {
 public:
  /**
   * @ingroup domi_omg
   * @brief Operator unique identification
   */
  using OpId = const void *;

  /**
   * @ingroup domi_omg
   * @brief error code, 1~99:Error, 100~199:Waring。
   */
  enum class ErrorCode {
    // no error
    OK = 0,

    // type unsupported
    TYPE_UNSUPPORTED = 1,

    // param invalid
    PARAM_INVALID = 2,

    // type ambiguous
    TYPE_AMBIGUOUS = 8,

    // name repeated
    NAME_REPEATED = 9
  };

  /**
   * @ingroup domi_omg
   * @brief Operator error description
   */
  struct Cause {
    // error code
    ErrorCode code;

    // error message
    string message;
  };

 public:
  /**
   * @ingroup domi_omg
   * @brief instance interface
   */
  static PreChecker &Instance();

  /**
   * @ingroup domi_omg
   * @brief set model name
   */
  void SetModelName(const string &name);

  /**
   * @ingroup domi_omg
   * @brief add op information
   */
  Status AddOp(OpId id, const string &name, const string &type);

  /**
   * @ingroup domi_omg
   * @brief Judge whether the operator name is duplicate
   */
  Status CheckName(OpId id);

  /**
   * @ingroup domi_omg
   * @brief check operation type
   *        1、Check whether the operator type supports according to the global frameworktype
   *        2、Check if the operator type is ambiguous
   */
  Status CheckType(OpId id, bool is_tensorflow = false);

  void RefreshErrorMessageByName(const string &op_name, ErrorCode code, const string& msg);

  /**
   * @ingroup domi_omg
   * @brief Add custom error description
   */
  Status AddCause(OpId id, ErrorCode code, const string &msg);

  /**
   * @ingroup domi_omg
   * @brief Add custom error description
   */
  Status AddCause(OpId id, const Cause &cause);

  /**
   * @ingroup domi_omg
   * @brief Clear all operator information
   */
  void Clear();

  /**
   * @ingroup domi_omg
   * @brief Clear the error information of the specified operator
   */
  Status Clear(OpId id, const string &message = "");

  /**
   * @ingroup domi_omg
   * @brief Determine if an error has been detected
   */
  bool HasError();

  /**
   * @ingroup domi_omg
   * @brief Save inspection results（JSON）
   */
  Status Save(const string &file);

 private:
  /**
   * @ingroup domi_omg
   * @brief operation information
   */
  struct Info {
    // Operator identifier
    OpId id;

    // Operator name
    string name;

    // Operator type
    string type;

    // Error description, which may contain multiple (for example, both name and type are illegal)
    vector<Cause> causes;
  };

  PreChecker();
  ~PreChecker();
  PreChecker(const PreChecker &);
  PreChecker &operator=(const PreChecker &);

  // Initialize internal data
  void Init();

  // Judge whether the type is supported
  Status CheckTypeSupported(OpId id, const string &type, const string &name, bool is_tensorflow);

  // Determine if an error has been detected
  bool HasError(OpId id);

 private:
  // model name
  string model_name_;

  // Save operator check results
  map<OpId, Info> op_map_;

  // Save operator list in original order
  vector<OpId> ops_;

  // save frame related operator types
  map<string, string> *fmk_op_types_;
};
}  // namespace ge
#endif  // PARSER_COMMON_PRE_CHECKER_H_
