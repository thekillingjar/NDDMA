/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_REPORT_ERROR_H
#define FUSION_ENGINE_REPORT_ERROR_H
#include <string>
#include <map>
#include <vector>
namespace fe {
struct ErrorMessageDetail {
 public:
  ErrorMessageDetail(const std::string &err_code, const std::vector<std::string> &args)
      : error_code(err_code), arg_list(args) {}
  ~ErrorMessageDetail() = default;
  void ToParamMap(std::map<std::string, std::string> &args_map);
  void ModifyArgsByErrorCode();
  const std::string &GetErrorCode() {
    return error_code;
  }

 private:
  std::string error_code;
  std::vector<std::string> arg_list;
  ErrorMessageDetail() = default;
};
void ReportErrorMessage(ErrorMessageDetail &error_detail);
}  // namespace fe
#endif
