/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include "framework/common/debug/ge_log.h"
#include "code_printer.h"

namespace ge {
void CodePrinter::AddInclude(const std::string &include_name) {
  output_ << "#include \"" << include_name << "\"" << std::endl;
}

void CodePrinter::AddNamespaceBegin(const std::string &namespace_name) {
  output_ << "namespace " << namespace_name << " {" << std::endl;
}

void CodePrinter::AddNamespaceEnd(const std::string &namespace_name) {
  output_ << "} // namespace " << namespace_name << std::endl;
}

void CodePrinter::DefineFuncBegin(const std::string &return_type, const std::string &func_name,
                                const std::string &param_name) {
  output_ << return_type << " " << func_name << "(" << param_name << ")" << std::endl << "{" << std::endl;
}

void CodePrinter::DefineFuncEnd() {
  output_ << "}\n" << std::endl;
}

void CodePrinter::AddLine(const std::string &input_string) {
  output_ << input_string << std::endl;
}

std::string CodePrinter::GetOutputStr() const {
  return output_.str();
}
}  // namespace ge