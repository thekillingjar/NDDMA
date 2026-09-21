/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef METADEF_CXX_TESTS_UT_ASCENDC_IR_CODE_EXTRACTOR_H_
#define METADEF_CXX_TESTS_UT_ASCENDC_IR_CODE_EXTRACTOR_H_
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

class CodeExtractor {
 public:
  struct FunctionCode {
    std::string signature;
    std::string body;
  };

  static FunctionCode ExtractFunction(const std::string &header_path,
                                      const std::string &class_name,
                                      const std::string &target_func) {
    std::ifstream file(header_path);
    std::vector<std::string> lines;
    std::string line;

    while (std::getline(file, line)) {
      lines.push_back(line);
    }

    return AnalyzeLines(lines, class_name, target_func);
  }

 private:
  static FunctionCode AnalyzeLines(const std::vector<std::string> &lines,
                                   const std::string &class_name,
                                   const std::string &target_func) {
    enum State { SEARCHING, IN_CLASS, IN_FUNCTION };
    State state = SEARCHING;
    int brace_level = 0;
    FunctionCode result;
    size_t start_line = 0;

    for (size_t i = 0; i < lines.size(); ++i) {
      std::string trimmed = Trim(lines[i]);

      if (state == SEARCHING) {
        if (class_name.empty() || IsClassStart(trimmed, class_name)) {
          state = IN_CLASS;
          continue;
        }
      } else if (state == IN_CLASS) {
        if (IsFunctionStart(trimmed, target_func)) {
          state = IN_FUNCTION;
          start_line = i;
          result.signature = trimmed;
          brace_level += CountBraces(trimmed);
        }
      } else {
        brace_level += CountBraces(lines[i]);
        if (brace_level == 0) {
          // 收集函数体代码
          for (size_t j = start_line; j <= i; ++j) {
            result.body += (lines[j] + "\n");
          }
          break;
        }
      }
    }
    return result;
  }

  static std::string Trim(const std::string &s) {
    size_t start = s.find_first_not_of(" \t");
    size_t end = s.find_last_not_of(" \t");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
  }

  static int CountBraces(const std::string &s) {
    return std::count(s.begin(), s.end(), '{') -
        std::count(s.begin(), s.end(), '}');
  }

  static bool IsFunctionStart(const std::string &line,
                              const std::string &func_name) {
    return line.find(func_name + "(") != std::string::npos &&
        (line.find('{') != std::string::npos ||
            line.find(';') == std::string::npos);
  }

  static bool IsClassStart(const std::string &line,
                           const std::string &class_name) {
    return line.find(class_name + " :") != std::string::npos &&
        (line.find('{') != std::string::npos ||
            line.find(';') == std::string::npos);
  }
};
#endif //METADEF_CXX_TESTS_UT_ASCENDC_IR_CODE_EXTRACTOR_H_
