/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_PRETTY_TABLE_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_PRETTY_TABLE_H_
#include <utility>
#include <vector>
#include <string>
#include <ostream>
namespace gert {
struct TableRow {
  TableRow(std::initializer_list<std::string> columns, std::string color_code)
      : columns(columns), color_code(std::move(color_code)) {}
  TableRow(std::vector<std::string> columns, std::string color_code)
      : columns(std::move(columns)), color_code(std::move(color_code)) {}
  std::vector<std::string> columns;
  std::string color_code;
};
class PrettyTable {
 public:
  PrettyTable &SetHeader(std::initializer_list<std::string> header) {
    header_ = header;
    return *this;
  }
  PrettyTable &AddRow(std::initializer_list<std::string> row) {
    rows_.emplace_back(row, "");
    return *this;
  }
  PrettyTable &AddColorRow(std::initializer_list<std::string> row) {
    // 1: bold, 31: red, see escape code for details
    rows_.emplace_back(row, "1;31");
    return *this;
  }
  void Print(std::ostream &s) const {
    if (!ColNumValid(s)) {
      return;
    }
    auto column_widths = CalcColWidth();
    PrintHeader(column_widths, s);
    PrintBody(column_widths, s);
    PrintTail(column_widths, s);
  }

 private:
  void PrintHeader(const std::vector<size_t> &width, std::ostream &s) const {
    auto line_width = CalcLineWidth(width);
    PrintLine(line_width, s);
    PrintRow({header_, ""}, width, s);
    PrintLine(line_width, s);
  }
  void PrintBody(const std::vector<size_t> &width, std::ostream &s) const {
    for (const auto &row : rows_) {
      PrintRow(row, width, s);
    }
  }
  static void PrintTail(const std::vector<size_t> &width, std::ostream &s) {
    auto line_width = CalcLineWidth(width);
    PrintLine(line_width, s);
  }
  static void PrintRow(const TableRow &row, const std::vector<size_t> &width, std::ostream &s) {
    if (!row.color_code.empty()) {
      s << "\033[" << row.color_code << "m";
    }
    auto &row_content = row.columns;
    s << '|';
    for (size_t i = 0; i < row_content.size(); ++i) {
      s << ' ';  // 每列开头打个空格
      s << row_content[i];
      auto padding_size = width[i] - row_content[i].size();
      if (padding_size > 0) {
        s << std::string(padding_size, ' ');
      }
      s << " |";  // 每列结尾打个空格
    }
    if (!row.color_code.empty()) {
      s << "\033[0m";
    }
    s << std::endl;
  }
  static void PrintLine(size_t line_width, std::ostream &s) {
    s << std::string(line_width, '-') << std::endl;
  }
  static size_t CalcLineWidth(const std::vector<size_t> &column_width) {
    size_t line_width = column_width.size() + 1;  // 竖线数量
    for (auto col_width : column_width) {
      line_width += col_width;
      line_width += 2;  // 为每列预留前后的空格
    }
    return line_width;
  }
  [[nodiscard]] std::vector<size_t> CalcColWidth() const {
    std::vector<size_t> col_width(header_.size(), 0);
    for (size_t i = 0; i < header_.size(); ++i) {
      col_width[i] = std::max(col_width[i], header_[i].size());
      for (const auto &row : rows_) {
        col_width[i] = std::max(col_width[i], row.columns[i].size());
      }
    }
    return col_width;
  }
  bool ColNumValid(std::ostream &s) const {
    auto expect_col = header_.size();
    for (const auto &row : rows_) {
      if (row.columns.size() != expect_col) {
        s << "Exists row has unmatched colum num: " << std::endl;

        s << "Row(Column num " << row.columns.size() << "): |";
        for (const auto &element : row.columns) {
          s << ' ' << element << " |";
        }
        s << std::endl;

        s << "Head(Column num " << header_.size() << "): |";
        for (const auto &element : header_) {
          s << ' ' << element << " |";
        }
        s << std::endl;

        return false;
      }
    }
    return true;
  }

 private:
  std::vector<std::string> header_;
  std::vector<TableRow> rows_;
};
}  // namespace gert
#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_PRETTY_TABLE_H_
