/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CODEGEN_ASCENDC_API_REGISTRY_H
#define CODEGEN_ASCENDC_API_REGISTRY_H

#include <string>
#include <unordered_map>
#include <functional>

namespace codegen {
class AscendCApiRegistry {
 public:
  static AscendCApiRegistry &GetInstance();

  const std::string &GetFileContent(const std::string &api_name);

  void RegisterApi(const std::unordered_map<std::string, std::string> &api_to_file_content);

 private:
  AscendCApiRegistry() = default;
  ~AscendCApiRegistry() = default;
  std::unordered_map<std::string, std::string> api_to_file_content_;
};
}  // namespace codegen

#endif  // CODEGEN_ASCENDC_API_REGISTRY_H
