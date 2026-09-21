/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H1D9F4FDE_BB21_4DE4_AC7E_751920B45039
#define H1D9F4FDE_BB21_4DE4_AC7E_751920B45039

#include "fake_ns.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "register/ops_kernel_builder_registry.h"
#include "engines/manager/engine/engine_manager.h"

FAKE_NS_BEGIN

struct EnvInstaller {
  virtual void InstallTo(std::map<string, OpsKernelInfoStorePtr>&) const {}
  virtual void InstallTo(std::map<string, GraphOptimizerPtr>&, std::vector<std::pair<std::string, GraphOptimizerPtr>>&) const {}
  virtual void InstallTo(std::map<string, OpsKernelBuilderPtr>&) const {}
  virtual void InstallTo(std::map<string, std::set<std::string>>&) const {}
  virtual void InstallTo(std::map<string, std::string>&) const {}
  virtual void InstallTo(std::map<string, DNNEnginePtr>&) const {}
  virtual void Install() const {}
};

FAKE_NS_END

#endif
