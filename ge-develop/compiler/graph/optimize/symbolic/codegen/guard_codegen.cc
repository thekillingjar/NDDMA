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
#include <regex>
#include <securec.h>
#include <unordered_set>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <vector>
#include "common/checker.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "attribute_group/attr_group_shape_env.h"
#include "code_printer.h"
#include "mmpa/mmpa_api.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "framework/common/types.h"
#include "mmpa/mmpa_api.h"
#include "guard_codegen.h"
#include "common/compile_profiling/ge_call_wrapper.h"

namespace ge {
namespace {
const std::string kSymbolCheckSoName = "guard_check.so";
const std::string kSymbolCheckFileName = "guard_check.cc";
const std::string k2Space = "  ";
constexpr uint32_t kMaxFileNameLen = 128U;
constexpr char_t const *kGuardCheckSoDataResult = "_guard_check_so_data";
constexpr size_t kConstLogHeaderLen = 110U;
const std::regex kSafePathRegex(R"(^(?!.*\.{2})[A-Za-z0-9./+\-_]+$)"); // 允许字母、数字、_ / . -, 但禁止..

Status GenCheckInfos(CodePrinter &printer, const std::vector<SymbolCheckInfo> &symbol_check_infos) {
  for (const auto &iter : symbol_check_infos) {
    printer.AddLine(k2Space + "check_result = " + std::string(iter.expr.Serialize().get()) + ";");
    printer.AddLine(k2Space + "check_results.emplace_back(check_result);");
  }
  GELOGI("GenCheckInfos, symbol_check_infos size: %zu", symbol_check_infos.size());
  return SUCCESS;
}

std::string GetFileHeader() {
  std::string file_header = R"(
#include <cmath>
#include <unordered_map>
#include <type_traits>
#include <string>
#include <iostream>
#include <limits>
#include <algorithm>
#include <initializer_list>
#include "exe_graph/runtime/tensor.h"
#include "securec.h"

using char_t = char;
namespace {
// integer comparison
template <typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type
CompareEqual(T a, T b) {
  return a == b;
}

// float comparison (absolute tolerance  + relative tolerance)
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type
CompareEqual(T a, T b) {
  const T abs_diff = std::abs(a - b);
  if (abs_diff <= std::numeric_limits<T>::epsilon()) {
    return true;
  }
  const T rel_diff = abs_diff / std::max(std::abs(a), std::abs(b));
  return rel_diff <= static_cast<T>(1e-6); // adjustable threshold
}

template <typename TA, typename TB,
    typename std::enable_if<(std::is_floating_point<TA>::value || std::is_integral<TA>::value) &&
    (std::is_floating_point<TB>::value || std::is_integral<TB>::value), bool>::type = 0>
bool CompareEqual(TA a, TB b) {
  if (std::is_same<TA, TB>::value) {
    return CompareEqual(a, b);
  }
  if ((std::is_floating_point<TA>::value) && (std::is_integral<TB>::value)) {
    return CompareEqual(a, static_cast<TA>(b));
  }
  return CompareEqual(static_cast<TB>(a), b);
}

#define Max(a, b) ((a) > (b) ? (a) : (b))
#define Min(a, b) ((a) < (b) ? (a) : (b))
#define Pow(a, b) (std::pow(a, b))
#define Exp(a) (std::exp(a))
#define Log(a) (std::log(a))
#define Sqrt(a) (std::sqrt(a))
#define Ceiling(a) (std::ceil(a))
#define Floor(a) (std::floor(a))
#define Abs(a) (std::abs(a))
#define Rational(a, b) (static_cast<double>(a) / static_cast<double>(b))
#define ExpectEq(a, b) (CompareEqual(a, b))
#define ExpectNe(a, b) (!CompareEqual(a, b))
#define ExpectLe(a, b) (a <= b)
#define ExpectLt(a, b) (a < b)
#define True true
#define False false
#define LogicAnd(...) \
    [&] { \
        const bool values[] = {__VA_ARGS__}; \
        return std::all_of(std::begin(values), std::end(values), [](bool x) { return x; }); \
    }()

#define LogicOr(...) \
    [&] { \
        const bool values[] = {__VA_ARGS__}; \
        return std::any_of(std::begin(values), std::end(values), [](bool x) { return x; }); \
    }()

template <typename Ta, typename Tb>
auto Mod(Ta left, Tb right) -> decltype(left % right) {
  return left % right;
}

// pecialized version for float (using std::fmod)
template <typename Ta, typename Tb>
auto Mod(Ta left, Tb right) ->
    typename std::enable_if<std::is_floating_point<Ta>::value || std::is_floating_point<Tb>::value,
                            decltype(std::fmod(left, right))>::type {
  return std::fmod(left, right);
}

struct GraphInputsContext {
  GraphInputsContext(gert::Tensor **tensors, size_t num_tensors)
      : input_tensors_(tensors), num_inputs_(num_tensors) {}
  const gert::Tensor *GetGraphInputTensor(size_t data_index) const {
    if (data_index > num_inputs_) {
      return nullptr;
    }
    auto *tensor = reinterpret_cast<const gert::Tensor *>(input_tensors_[data_index]);
    if (tensor == nullptr) {
      return nullptr;
    }
    return tensor;
  }
private:
  gert::Tensor **input_tensors_{nullptr};
  size_t num_inputs_{0U};
};
} // namespace
)";
  return file_header;
}

Status GenErrorInfos(CodePrinter &printer, const std::vector<SymbolCheckInfo> &symbol_check_infos) {
  for (const auto &iter : symbol_check_infos) {
    printer.AddLine(k2Space + "err_msgs.emplace_back(\"[" + iter.file + ":" + std::to_string(iter.line) +
                    "] Check Symbol Check Expression: " + std::string(iter.expr.Serialize().get()) + 
                    " Context Info: " + iter.dfx_info + " missed.\");");
  }
  return SUCCESS;
}

Status GenGetErrorMsg(CodePrinter &printer, const ge::ShapeEnvAttr *shape_env_attr) {
  GE_ASSERT_NOTNULL(shape_env_attr);
  printer.AddLine(k2Space + "std::vector<std::string> err_msgs;");
  auto err_size = shape_env_attr->GetAllSymbolCheckInfos().size() + shape_env_attr->GetAllSymbolAssertInfos().size();
  printer.AddLine(k2Space + "err_msgs.reserve(" + std::to_string(err_size) + ");");
  GE_ASSERT_SUCCESS(GenErrorInfos(printer, shape_env_attr->GetAllSymbolCheckInfos()));
  GE_ASSERT_SUCCESS(GenErrorInfos(printer, shape_env_attr->GetAllSymbolAssertInfos()));
  printer.AddLine(k2Space + "return err_msgs;");
  return SUCCESS;
}

Status GenGetCheckResults(CodePrinter &printer, ge::ShapeEnvAttr *shape_env_attr) {
  GE_ASSERT_NOTNULL(shape_env_attr);
  for (const auto &sym_to_source: shape_env_attr->GetAllSym2Src()) {
    printer.AddLine(
      k2Space + "const auto " + std::string(sym_to_source.first.Serialize().get()) + " = " + sym_to_source.second->
       GetSourceStr() + ";");
  }
  printer.AddLine(k2Space + "std::vector<uint8_t> check_results;");
  auto guard_size = shape_env_attr->GetAllSymbolCheckInfos().size() + shape_env_attr->GetAllSymbolAssertInfos().size();
  printer.AddLine(k2Space + "check_results.reserve(" + std::to_string(guard_size) + ");");
  printer.AddLine(k2Space + "uint8_t check_result;");
  GE_ASSERT_SUCCESS(GenCheckInfos(printer, shape_env_attr->GetAllSymbolCheckInfos()));
  GE_ASSERT_SUCCESS(GenCheckInfos(printer, shape_env_attr->GetAllSymbolAssertInfos()));
  printer.AddLine(k2Space + "return check_results;");
  return SUCCESS;
}

Status GenGuardCheckFunc(CodePrinter &printer) {
  std::string check_code = R"(  auto context = std::make_unique<GraphInputsContext>(tensors, num_tensors);
  if (context == nullptr) {
    return false;
  }
  auto check_results = GetCheckResults(context);
  for (size_t i = 0UL; i < check_results.size(); i++) {
    if (check_results[i] == 0U) {
      const char *log_level = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");
      if ((log_level != nullptr) && (std::atoi(log_level) <= 1)) {
        auto err_msgs = GetErrorMsg();
        snprintf_s(reason, reason_size, reason_size - 1, err_msgs[i].c_str());
        reason[reason_size - 1] = '\0';
      }
      return false;
    }
  }
  return true;)";
  printer.AddLine(check_code);
  return SUCCESS;
}

Status GetOppPath(std::string &opp_path) {
  GELOGI("Enter get opp path schedule");
  const char_t *path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, path_env);
  GE_ASSERT_TRUE(path_env != nullptr, "[Call][MM_SYS_GET_ENV] Failed to get env ASCEND_OPP_PATH.");
  opp_path = path_env;
  std::string file_path = RealPath(opp_path.c_str());
  GE_ASSERT_TRUE(!file_path.empty(), "[Call][RealPath] File path %s is invalid.", opp_path.c_str());
  if (opp_path.back() != '/') {
    opp_path += '/';
  }
  GELOGI("Get opp path from env: %s", opp_path.c_str());
  return SUCCESS;
}

void DebugCodeGenResult(const CodePrinter &printer) {
  GELOGI("Guard Func Start:");
  auto msg = printer.GetOutputStr();
  size_t max_log_len = MSG_LENGTH - kConstLogHeaderLen;
  size_t current_index = 0U;
  while (current_index < msg.size()) {
    auto sub_str = msg.substr(current_index, max_log_len);
    GELOGI("\n%s", sub_str.c_str());
    current_index += max_log_len;
  }
  GELOGI("Guard Func End");
}

bool CheckPathInCmdIsValid(const std::string &so_path, const std::string &file_path,
                           const std::string &ascend_include_path) {
  return std::regex_match(so_path, kSafePathRegex) && std::regex_match(file_path, kSafePathRegex) &&
         std::regex_match(ascend_include_path, kSafePathRegex);
}

graphStatus CompileGuardCheckFunc(const CodePrinter &printer, const ComputeGraphPtr &graph) {
  // codegen
  // 1. dump code to file
  char_t file_path[kMaxFileNameLen] = {"\0"};
  auto src_fd = static_cast<int32_t>(syscall(__NR_memfd_create, kSymbolCheckFileName.c_str(), 0));
  std::string output_str = printer.GetOutputStr();
  mmWrite(src_fd, &output_str[0], output_str.size());
  (void)lseek(static_cast<int32_t>(src_fd), 0, SEEK_SET);
  auto ret = snprintf_s(file_path, kMaxFileNameLen, kMaxFileNameLen - 1U, "/proc/self/fd/%d", src_fd);
  GE_ASSERT_TRUE(ret >= 0, "snprintf_s failed, ret: %d", ret);
  GE_MAKE_GUARD(close_src_fd, [&src_fd]() {
    mmClose(src_fd);
  });

  // 2. compile code to so
  char_t so_path[kMaxFileNameLen] = {"\0"};
  auto so_fd = static_cast<int32_t>(syscall(__NR_memfd_create, kSymbolCheckSoName.c_str(), 0));
  ret = snprintf_s(so_path, kMaxFileNameLen, kMaxFileNameLen - 1U, "/proc/self/fd/%d", so_fd);
  GE_ASSERT_TRUE(ret >= 0, "snprintf_s failed, ret: %d", ret);
  GE_MAKE_GUARD(close_so_fd, [&so_fd]() {
    mmClose(so_fd);
  });

  std::string opp_path;
  GE_ASSERT_SUCCESS(GetOppPath(opp_path), "Failed to get environment variable ASCEND_OPP_PATH, please set it.");

  std::string ascend_include = RealPath((opp_path + "/../include/").c_str());
  GE_ASSERT_TRUE(!ascend_include.empty(), "[Call][RealPath] ascend include path %s is invalid.",
                 ascend_include.c_str());
  const std::string so_path_dir(so_path);
  const std::string file_path_dir(file_path);
  std::string command = "g++ -O2 -fstack-protector-all -shared -fPIC -Wl,-z,now -Wl,-z,noexecstack -s -o " +
                        so_path_dir + " -I " + ascend_include + " -L c_sec" + " -x c++ " + file_path_dir;
  GELOGI("To compile guard_check.so, command: %s", command.c_str());

  GE_ASSERT_TRUE(CheckPathInCmdIsValid(so_path_dir, file_path_dir, ascend_include),
                 "CheckPathInCmdIsValid failed, so_path_dir = %s, file_path_dir = %s, ascend_include = %s.",
                 so_path_dir.c_str(), file_path_dir.c_str(), ascend_include.c_str());
  GE_TIMESTAMP_START(GuardCheckCompile);
  GE_ASSERT_TRUE(system(command.c_str()) == 0, "Failed to compile guard_check.so");
  GE_TIMESTAMP_END(GuardCheckCompile, "GuardCheckCompile");

  // 3. read so to graph
  std::ifstream file(so_path, std::ios::binary | std::ios::ate);
  GE_ASSERT_TRUE(file.is_open());
  GE_MAKE_GUARD(close_file, [&file]() {
    file.close();
  });
  std::streamsize file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char_t> buffer(file_size);
  file.read(buffer.data(), file_size);
  GE_ASSERT_TRUE(AttrUtils::SetStr(graph, kGuardCheckSoDataResult, std::string(buffer.data(), buffer.size())));
  return GRAPH_SUCCESS;
}
}  // namespace

Status GuardCodegen::GuardFuncCodegenAndCompile(const ComputeGraphPtr &graph) const {
  GE_ASSERT_NOTNULL(graph);
  auto shape_env_attr = graph->GetAttrsGroup<ShapeEnvAttr>();
  GE_ASSERT_NOTNULL(shape_env_attr);
  CodePrinter printer;
  printer.AddLine(GetFileHeader());
  printer.AddLine("using char_t = char;");
  printer.AddNamespaceBegin("");
  // Add Func GetErrorMsg
  printer.DefineFuncBegin("std::vector<std::string>", "GetErrorMsg", "");
  GE_ASSERT_SUCCESS(GenGetErrorMsg(printer, shape_env_attr));
  printer.DefineFuncEnd();
  // Add Func GetCheckResults
  printer.DefineFuncBegin("std::vector<uint8_t>", "GetCheckResults",
                          "const std::unique_ptr<GraphInputsContext> &context");
  GE_ASSERT_SUCCESS(GenGetCheckResults(printer, shape_env_attr));
  printer.DefineFuncEnd();
  printer.AddNamespaceEnd("");
  printer.AddLine("extern \"C\"");
  printer.DefineFuncBegin("bool", "GuardCheckFunc",
                          "gert::Tensor **tensors, size_t num_tensors, char_t *reason, size_t reason_size");
  GE_ASSERT_SUCCESS(GenGuardCheckFunc(printer));
  printer.DefineFuncEnd();
  if (dlog_getlevel(GE_MODULE_NAME, nullptr) <= DLOG_INFO) {
    DebugCodeGenResult(printer);
  }
  GE_ASSERT_SUCCESS(CompileGuardCheckFunc(printer, graph));
  return GRAPH_SUCCESS;
}
}  // namespace ge