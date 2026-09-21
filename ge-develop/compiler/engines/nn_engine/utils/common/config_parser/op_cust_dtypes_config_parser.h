/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_UTILS_COMMON_OP_CUST_DTYPES_CONFIG_PARSER_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_UTILS_COMMON_OP_CUST_DTYPES_CONFIG_PARSER_

#include <vector>
#include <mutex>
#include "common/aicore_util_types.h"
#include "graph/types.h"
#include "common/base_config_parser.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
using std::string;
class OpCustDtypesConfigParser : public BaseConfigParser {
 public:
  OpCustDtypesConfigParser();
  ~OpCustDtypesConfigParser() override;

  Status InitializeFromOptions(const std::map<string, string> &options) override;
  using BaseConfigParser::InitializeFromContext;
  Status InitializeFromContext(const std::string &combined_params_key) override;
  bool GetCustomizeDtypeByOpType(const string &op_type, OpCustomizeDtype &custom_dtype) const;
  bool GetCustomizeDtypeByOpName(const string &op_name, OpCustomizeDtype &custom_dtype) const;

 private:
  Status ParseCustomDtypeContent();
  bool ParseFileContent(const string &custom_file_path);
  bool ParseDTypeFromLine(const string &line_str, const bool &is_type);
  bool SplitNameOrType(const string &line_str, string &name_or_type) const;
  bool SplitInoutDtype(const string &line_str, std::vector<string> &input_dtype,
                       std::vector<string> &output_dtype) const;
  bool FeedOpCustomizeDtype(const string &op_type, const bool &is_type,
                            const std::vector<string> &input_dtype,
                            const std::vector<string> &output_dtype);
  bool CheckValidAndTrans(string &dtype_str, ge::DataType &ge_dtype) const;
  void CheckAndSetCustomizeDtype(const bool &is_type, const string &op_type,
                                 const OpCustomizeDtype &custom_dtype_vec);

  string cust_dtypes_path_;
  std::map<string, OpCustomizeDtype> op_name_cust_dtypes_;
  std::map<string, OpCustomizeDtype> op_type_cust_dtypes_;
};
using OpCustDtypesConfigParserPtr = std::shared_ptr<OpCustDtypesConfigParser>;
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_UTILS_COMMON_OP_CUST_DTYPES_CONFIG_PARSER_
