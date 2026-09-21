/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_UTILS_GENERATOR_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_UTILS_GENERATOR_H_

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include "generator_interface.h"

namespace ge {
namespace es {

/**
 * ES算子的Utils函数生成器
 * 目前仅实现es_elog.h报错日至功能
 */
class UtilsGenerator {
 public:
  // 生成Utils的代码
  void GenUtils(std::string &util_name) {
    std::stringstream h_stream("");
    GenUtilsBody(util_name, h_stream);
    utils_to_hss[util_name] = std::move(h_stream);
  }

  // 获取生成器名称（用于日志和错误信息）
  std::string GetGeneratorName() const {
    return "Utils generator";
  }

  // 获取Utils名称
  std::vector<std::string> GetUtilFileNames() const {
    std::vector<std::string> util_names;
    util_names.reserve(utils_to_hss.size());
    for (const auto &util_map: utils_to_hss) {
      util_names.emplace_back("es_" + util_map.first + ".h");
    }
    return util_names;
  }

  // 获取Utils名称
  std::string GetPerUtilFileName(const std::string &op_type) const {
    return PerUtilHeaderFileName(op_type);
  }

  const std::unordered_map<std::string, std::stringstream> &GetPerUtilContents() const {
    return utils_to_hss;
  }

  void GenPerUtilFiles(const std::string &output_dir, const std::vector<string> &util_names) const {
    // 生成utils相关hpp
    WritePerUtilFiles(output_dir, util_names, GetPerUtilContents(),
                    [this](const std::string &util_name) { return GetPerUtilFileName(util_name); });
  }

 private:
  static std::string PerUtilHeaderFileName(const std::string &util_name) {
    return std::string("es_") + util_name + ".h";
  }

  static void GenUtilsBody(const std::string& utils_name, std::stringstream &hss) {
    if (utils_name == "log") {
      GenELog(hss);
    }
  }

  static void GenELog(std::stringstream &hss) {
    GenCopyright(hss);
    std::string elog_contents = R"(
#ifndef CANN_GRAPH_ENGINE_ES_ELOG_H
#define CANN_GRAPH_ENGINE_ES_ELOG_H
#include "graph/ge_error_codes.h"
#include <securec.h>
#include <memory>
#include <vector>
namespace ge {
namespace es {
struct EsErrorResult {
  operator bool() const {
    return false;
  }
  operator ge::graphStatus() const {
    return ge::GRAPH_PARAM_INVALID;
  }
  template<typename T>
  operator std::unique_ptr<T>() const {
    return nullptr;
  }
  template<typename T>
  operator std::shared_ptr<T>() const {
    return nullptr;
  }
  template<typename T>
  operator T *() const {
    return nullptr;
  }
  template<typename T>
  operator std::vector<std::shared_ptr<T>>() const {
    return {};
  }
  template<typename T>
  operator std::vector<T>() const {
    return {};
  }
  operator std::string() const {
    return "";
  }
  template<typename T>
  operator T() const {
    return T();
  }
};

inline std::vector<char> CreateErrorMsg(const char *format, ...) {
  va_list args;
  va_start(args, format);
  va_list args_copy;
  va_copy(args_copy, args);
  const size_t len = static_cast<size_t>(vsnprintf(nullptr, 0, format, args_copy));
  va_end(args_copy);
  std::vector<char> msg(len + 1U, '\0');
  const auto ret = vsnprintf_s(msg.data(), len + 1U, len, format, args);
  va_end(args);
  return (ret > 0) ? msg : std::vector<char>{};
}

inline std::vector<char> CreateErrorMsg() {
  return {};
}

// 为了支持 C++11 版本，自定义EsMakeUnique封装 make_unique
template<typename T, typename... Args>
std::unique_ptr<T> EsMakeUnique(Args&&... args) {
  return std::unique_ptr<T>(new (std::nothrow) T(std::forward<Args>(args)...));
}

#define ES_ASSERT(exp, v, ...)                               \
  do {                                                       \
    if (!(exp)) {                                            \
      auto msg = ge::es::CreateErrorMsg(__VA_ARGS__);                \
      if (msg.empty()) {                                     \
        std::cerr << __FILE__ << ": " << __LINE__ << " [" << __FUNCTION__ << "]: Assert failed: " << #exp << std::endl; \
      } else {                                               \
        std::cerr << __FILE__ << ": "<< __LINE__ << " ["<< __FUNCTION__ << "]: "<< msg.data() << std::endl;             \
      }                                                      \
      return ge::es::EsErrorResult();                              \
    }                                                        \
  } while (false);
#define ES_ASSERT_NOTNULL(v, ...) ES_ASSERT(((v) != nullptr), v, __VA_ARGS__)
#define ES_ASSERT_SUCCESS(v, ...) ES_ASSERT(((v) == ge::SUCCESS), __VA_ARGS__)
#define ES_ASSERT_GRAPH_SUCCESS(v, ...) ES_ASSERT(((v) == ge::GRAPH_SUCCESS), __VA_ARGS__)
#define ES_ASSERT_RT_OK(v, ...) ES_ASSERT(((v) == 0), __VA_ARGS__)
#define ES_ASSERT_EOK(v, ...) ES_ASSERT(((v) == EOK), __VA_ARGS__)
#define ES_ASSERT_TRUE(v, ...) ES_ASSERT((v), __VA_ARGS__)
#define ES_ASSERT_HYPER_SUCCESS(v, ...) ES_ASSERT(((v).IsSuccess()), __VA_ARGS__)
}  // namespace es
}  // namespace ge

#endif  // CANN_GRAPH_ENGINE_ES_ELOG_H
)";
    hss << elog_contents;
  }

  std::unordered_map<std::string, std::stringstream> utils_to_hss;
};

}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_UTILS_GENERATOR_H_