/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef AUTOFUSE_API_TILING_REGISTER_H
#define AUTOFUSE_API_TILING_REGISTER_H
#include <string>
#include "common/checker.h"
#include "base/base_types.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
namespace att {
using AutofuseApiTilingDataGenerator = std::function<ge::Status(
    const std::string &tiling_data_type, const ge::AscGraph &graph, const ge::AscNodePtr &node, std::string &code_string, uint32_t tiling_case_id)>;
struct ApiTilingGenerator {
  AutofuseApiTilingDataGenerator func_invoke_generator;
  AutofuseApiTilingDataGenerator func_impl_generator;
  AutofuseApiTilingDataGenerator head_files_generator;
};
class ApiTilingGenRegistry {
 public:
  static ApiTilingGenRegistry &Instance();
  bool IsApiTilingRegistered(const std::string &op_type);
  void RegFunc(const std::string &op_type, const AutofuseApiTilingDataGenerator &gen_func_define,
               const AutofuseApiTilingDataGenerator &gen_func_call,
               const AutofuseApiTilingDataGenerator &gen_head_files);
 AutofuseApiTilingDataGenerator GetApiTilingGenImplFunc(const std::string &op_type);
 AutofuseApiTilingDataGenerator GetApiTilingGenInvokeFunc(const std::string &op_type);
 AutofuseApiTilingDataGenerator GetApiTilingGenHeadFiles(const std::string &op_type);

 private:
  ApiTilingGenRegistry() = default;
  ~ApiTilingGenRegistry() = default;
  std::map<std::string, ApiTilingGenerator> api_tiling_generators_;
};

class ApiTilingGenRegister {
 public:
  ApiTilingGenRegister(const std::string &op_type, const AutofuseApiTilingDataGenerator &gen_func_define,
                       const AutofuseApiTilingDataGenerator &gen_func_call,
                       const AutofuseApiTilingDataGenerator &gen_head_files) {
    ApiTilingGenRegistry::Instance().RegFunc(op_type, gen_func_define, gen_func_call, gen_head_files);
  }
  ~ApiTilingGenRegister() = default;
};

#define REGISTER_API_TILING_FUNC(op_type, gen_func_define, gen_func_call, gen_head_files) \
  ApiTilingGenRegister reg_##op_type(op_type, gen_func_define, gen_func_call, gen_head_files)
}  // namespace att

#endif  // AUTOFUSE_API_TILING_REGISTER_H
