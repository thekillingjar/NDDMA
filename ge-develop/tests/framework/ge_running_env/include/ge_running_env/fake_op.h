/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H737AD661_27C0_400F_8B08_29701308C5D0
#define H737AD661_27C0_400F_8B08_29701308C5D0

#include <string>
#include <set>
#include <variant>
#include "fake_ns.h"
#include "ge_running_env/env_installer.h"
#include "graph/operator_factory.h"
#include "graph/op_kernel_bin.h"

FAKE_NS_BEGIN

struct FakeOp : EnvInstaller {
  FakeOp(const std::string& op_type);

  FakeOp& Inputs(const std::vector<std::string>&);
  FakeOp& Outputs(const std::vector<std::string>&);
  FakeOp& InferShape(InferShapeFunc);
  FakeOp& InfoStoreAndBuilder(const std::string&);
  template<typename  T>
  FakeOp& AttrsDef(const std::string& name, const T value) {
    attrs_.emplace(name, value);
    return *this;
  }
  template<class T>
  FakeOp& ExtAttrsDef(const std::string &name, const T &value) {
    ext_attrs_.emplace(name, value);
    return *this;
  }

 private:
  void Install() const override;
  void InstallTo(std::map<string, OpsKernelInfoStorePtr>&) const override;

 private:
  const std::string op_type_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  InferShapeFunc info_fun_;
  std::set<std::string> info_store_names_;
  std::map<std::string, std::variant<int64_t, std::string>> attrs_;
  std::map<std::string, std::variant<OpKernelBinPtr>> ext_attrs_;
};

FAKE_NS_END

#endif /* H737AD661_27C0_400F_8B08_29701308C5D0 */
