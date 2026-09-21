/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_INC_COMMON_STRING_UTILS_H_
#define FUSION_ENGINE_INC_COMMON_STRING_UTILS_H_

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include "common/fe_log.h"
#include "graph/types.h"
#include "graph/utils/type_utils.h"

const size_t INPLACE_SIZE = 2;
const int64_t NUM_TEN = 10;
namespace fe {
using std::string;

class StringUtils {
 public:
  static void TransStringToListListInt(const string &outputInplaceAbility, std::vector<std::vector<int64_t>> &res) {
    bool num_flag = false;
    std::vector<int64_t> output_inplace;
    size_t i = 0;
    int64_t data = 0;
    uint32_t left_brace_num = 0;
    while (i < outputInplaceAbility.size()) {
      if (outputInplaceAbility[i] == '{') {
        left_brace_num++;
      }
      // trans digit to num
      while (i < outputInplaceAbility.size() && isdigit(outputInplaceAbility[i])) {
        int64_t tmp_data = outputInplaceAbility[i++] - '0';
        data = data * NUM_TEN + tmp_data;
        num_flag = true;
      }
      // get num push to inplace
      if (num_flag) {
        output_inplace.emplace_back(data);
        num_flag = false;
        data = 0;
      }
      // when inplace size is 2 push to res to assember listlistint
      char right_brace = outputInplaceAbility[i];
      if (!IsValidOutputInplace(right_brace, output_inplace, res, left_brace_num)) {
        res.clear();
        FE_LOGW("The value of outputInplaceAbility %s is invalid, and valid value is {{a,b},{c,d}....}",
                outputInplaceAbility.c_str());
        return;
      }
      i++;
    }
    IsOutputPair(left_brace_num, res, outputInplaceAbility);
  }

  static void IsOutputPair(const uint32_t &left_brace_num, std::vector<std::vector<int64_t>> &res,
                           const string &outputInplaceAbility) {
    if (left_brace_num != 0) {
      res.clear();
      FE_LOGW("The pair value of outputInplaceAbility %s is invalid, and valid value is {{a,b},{c,d}....}",
              outputInplaceAbility.c_str());
    }
  }

  static bool IsValidOutputInplace(const char &ch, std::vector<int64_t> &output_inplace,
                                   std::vector<std::vector<int64_t>> &res, uint32_t &left_brace_num) {                      
    if (ch != '}') {
      return true;
    }
    if (left_brace_num > 0) {
      left_brace_num--;
      if (output_inplace.size() == INPLACE_SIZE) {
        res.emplace_back(output_inplace);
        output_inplace.clear();
        return true;
      }
      if (left_brace_num == 0) {
        return true;
      }
      return false;
    }
    return false;
  }

  /** @ingroup domi_common
   *  @brief split string
   *  @link http://stackoverflow.com/questions/236129/how-to-split-a-string-in-c
   *  @param [in] str   : string need to be split
   *  @param [in] pattern : separator
   *  @return string vector after split
   */
  static std::vector<string> Split(const string &str, char pattern) {
    std::vector<string> res_vec;
    string str_pattern;
    str_pattern += pattern;
    res_vec = Split(str, str_pattern);
    return res_vec;
  }

  static std::vector<string> Split(const string &str, const string &pattern) {
    std::vector<string> res_vec;
    if (str.empty()) {
      return res_vec;
    }
    string str_and_pattern = str + pattern;
    size_t pos = str_and_pattern.find(pattern);
    size_t size = str_and_pattern.size();
    while (pos != string::npos) {
      string sub_str = str_and_pattern.substr(0, pos);
      res_vec.push_back(sub_str);
      str_and_pattern = str_and_pattern.substr(pos + pattern.size(), size);
      pos = str_and_pattern.find(pattern);
    }
    return res_vec;
  }

  static void SplitWithTrim(const string &str, char pattern, std::vector<string> &res_vec) {
    if (str.empty() || pattern == ' ') {
      return;
    }
    string str_and_pattern = str + pattern;
    size_t pos = str_and_pattern.find(pattern);
    size_t size = str_and_pattern.size();
    while (pos != string::npos) {
      string sub_str = str_and_pattern.substr(0, pos);
      (void)Trim(sub_str);
      if (!sub_str.empty()) {
        res_vec.push_back(sub_str);
      }
      str_and_pattern = str_and_pattern.substr(pos + 1, size);
      pos = str_and_pattern.find(pattern);
    }
  }

  static string &Ltrim(string &s) {
#if __cplusplus >= 201103L
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int c) { return !std::isspace(c); }));
#else
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
#endif
    return s;
  }

  static string &Rtrim(string &s) {
#if __cplusplus >= 201103L
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int c) { return !std::isspace(c); }).base(), s.end());
#else
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
#endif
    return s;
  }

  /** @ingroup domi_common
   *  @brief remove the white space at the begining and end position of string
   *  @param [in] s : the string need to be trim
   *  @return the string after trim
   */
  static string &Trim(string &s) { return Ltrim(Rtrim(s)); }

  /** @ingroup domi_common
  *  @brief whether a string is start with a specific string（prefix）
  *  @param [in] str    : a string to be compared
  *  @param [in] prefix : a string prefix
  *  @return true else false
  */
  static bool StartWith(const string &str, const string prefix) {
    return ((str.size() >= prefix.size()) && (str.compare(0, prefix.size(), prefix) == 0));
  }

  static bool EndWith(const string &str, const string &suffix) {
    if (str.length() < suffix.length()) {
      return false;
    }
    string sub_str = str.substr(str.length() - suffix.length());
    return sub_str == suffix;
  }

  /** @ingroup fe
  *  @brief transfer a string to upper string
  *  @param [in] str : a string to be transfer
  *  @return None
  */
  static void ToUpperString(string &str) { transform(str.begin(), str.end(), str.begin(), (int (*)(int))toupper); }

  /** @ingroup fe
  *  @brief transfer a string to lower string
  *  @param [in] str : a string to be transfer
  *  @return None
  */
  static void ToLowerString(string &str) { transform(str.begin(), str.end(), str.begin(), (int (*)(int))tolower); }

  /** @ingroup fe
   *  @brief whether a string match another string, after triming space
   *  @param [in] left    : a string to be matched
   *  @param [in] right   : a string to be matched
   *  @return true else false
   */
  static bool MatchString(string left, string right) {
    if (Trim(left) == Trim(right)) {
      return true;
    }
    return false;
  }

  static bool MatchWithoutCase(string left, string right) {
    StringUtils::ToLowerString(left);
    StringUtils::ToLowerString(right);
    if (Trim(left) == Trim(right)) {
      return true;
    }
    return false;
  }

  /**
 * check whether the input str is integer
 * @param[in] str input string
 * @param[out] true/false
 */
  static bool IsInteger(const string &str) {
    if (str.empty()) {
      return false;
    }
    for (auto c : str) {
      if (!isdigit(c)) {
        return false;
      }
    }
    return true;
  }

  template <typename CONTAINER>
  static string IntegerVecToString(const CONTAINER &integer_vec) {
    string result = "{";
    for (auto ele : integer_vec) {
      string ele_str;
      using DT = typename std::remove_cv<decltype(ele)>::type;
      if (std::is_enum<DT>::value) {
        if (std::is_same<ge::Format, DT>::value) {
          ele_str = ge::TypeUtils::FormatToSerialString(static_cast<ge::Format>(ele));
        } else if (std::is_same<ge::DataType, DT>::value) {
          ele_str = ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType >(ele));
        } else {
          ele_str = std::to_string(ele);
        }
      } else {
        ele_str = std::to_string(ele);
      }
      result += ele_str;
      result += ",";
    }
    if (result.empty()) {
      return "";
    }
    /* Get rid of the last comma. */
    if (result.back() == ',') {
      result = result.substr(0, result.length() - 1);
    }
    result += "}";
    return result;
  }

  static std::string StrVecToString(const std::vector<std::string> &str_vec) {
    std::string result = "[";
    bool is_first = true;
    for (const std::string &str : str_vec) {
      if (is_first) {
        is_first = false;
      } else {
        result += ",";
      }
      result += str;
    }

    result += "]";
    return result;
  }

  template <typename T>
  static string MapKeyToString(const std::map<string, T> &map) {
    string result = "[";
    bool is_first = true;
    for (const auto& pair : map) {
      if (is_first) {
        is_first = false;
      } else {
        result += ",";
      }
      result += pair.first;
    }
    result += "]";
    return result;
  }
};
}  // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_STRING_UTILS_H_
