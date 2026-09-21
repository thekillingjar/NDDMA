/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_SYMBOLIC_CODEGEN_CODE_PRINTER_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_SYMBOLIC_CODEGEN_CODE_PRINTER_H_
#include <string>
#include <sstream>
#include "ge_common/ge_api_error_codes.h"

namespace ge {
class CodePrinter {
public:
  // 默认无函数构造
  CodePrinter() = default;
  ~CodePrinter() = default;
  /**
   * 将拼接好的字符串输出
  */
  std::string GetOutputStr() const;
  /**
   * 添加一行，自动换行
   */
  void AddLine(const std::string& input_string);

  void AddInclude(const std::string& include_name);
  void AddNamespaceBegin(const std::string& namespace_name);
  void AddNamespaceEnd(const std::string& namespace_name);
  void DefineFuncBegin(const std::string& return_type, const std::string& func_name,
                       const std::string& param_name);
  void DefineFuncEnd();
private:
  std::stringstream output_;
  std::string output_file_path_;
};
} //namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_SYMBOLIC_CODEGEN_CODE_PRINTER_H_