/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_lib_register_impl.h"
#include "graph_metadef/graph/debug/ge_util.h"

#include "mmpa/mmpa_api.h"
#include "framework/common/debug/ge_log.h"
#include "common/ge_common/string_util.h"
#include "common/checker.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "graph_metadef/graph/utils/file_utils.h"

extern "C" __attribute__((weak)) void SetMetadefPluginCustomOpLibPathForC(const char* custom_op_Lib_path);
namespace {
  const std::string custom_so_name = "libcust_opapi.so";
}

namespace ge {
OpLibRegister::OpLibRegister(const char_t *vendor_name) : impl_(ComGraphMakeUnique<OpLibRegisterImpl>()) {
  if (impl_ != nullptr) {
    impl_->SetVendorName(vendor_name);
  }
}

OpLibRegister::OpLibRegister(const OpLibRegister &other) {
  if (other.impl_ != nullptr) {
    OpLibRegistry::GetInstance().RegisterInitFunc(*other.impl_);
  }
}

OpLibRegister::OpLibRegister(OpLibRegister &&other) noexcept {
  if (other.impl_ != nullptr) {
    OpLibRegistry::GetInstance().RegisterInitFunc(*other.impl_);
  }
}

OpLibRegister::~OpLibRegister() = default;

OpLibRegister &OpLibRegister::RegOpLibInit(OpLibRegister::OpLibInitFunc func) {
  if (impl_ != nullptr) {
    impl_->SetInitFunc(func);
  }
  return *this;
}

OpLibRegistry &OpLibRegistry::GetInstance() {
  static OpLibRegistry instance;
  return instance;
}

const char_t* OpLibRegistry::InitAndGetCustomOpLibPath() {
  // in tfa scene, GEInitialize is called after aclGetCustomOpLibPath.
  // so if is_init_ is false, call PreProcessForCustomOp
  if (!is_init_) {
    GE_ASSERT_GRAPH_SUCCESS(PreProcessForCustomOp());
  }
  GELOGI("get op lib path is %s", op_lib_paths_.c_str());
  return op_lib_paths_.c_str();
}

void OpLibRegistry::RegisterInitFunc(OpLibRegisterImpl &register_impl) {
  const std::string vendor_name = register_impl.GetVendorName();
  auto func = register_impl.GetInitFunc();
  const std::lock_guard<std::mutex> lk(mu_);
  const auto it = vendor_names_set_.insert(vendor_name);
  // ignore same vendor_name op lib when register secondly
  if (it.second) {
    if (func != nullptr) {
      (void) vendor_funcs_.emplace_back(vendor_name, func);
    }
    GELOGI("%s op lib register successfully", vendor_name.c_str());
  } else {
    GELOGW("%s op lib has already registered", vendor_name.c_str());
  }
}

/**
 * @brief 对环境变量下ASCEND_CUSTOM_OPP_PATH新so交付的自定义算子目录作预处理，需要保证在获取自定义算子目录前调用，
 * @brief 当前提供metadef接口, air仓各个流程初始化靠前的位置调用
 *
 * 当前最新的自定义算子工程交付分为run包交付和so交付（新做的）两种形式：
 * 新的so交付的形式下：export ASCEND_CUSTOM_OPP_PATH=/path/to/customize:/path/to/mdc:/path/to/lhisi
 * 三个目录下都只有一个libcust_opapi.so
 *
 * 老的run包交付的形式下：export ASCEND_CUSTOM_OPP_PATH=/path/to/customize:/path/to/mdc:/path/to/lhisi
 * 三个目录下都有完整的算子子目录，如op_proto,op_impl子目录等
 *
 * 当前支持两种方式混用。混用优先级以新的so交付方式优先。
 * 例如export ASCEND_CUSTOM_OPP_PATH=/home/a:/home/b:/home/c,其中只有/home/b是新so交付的方式
 * 则最终优先级别顺序为b,a,c
 * @return
 */
graphStatus OpLibRegistry::PreProcessForCustomOp() {
  if (is_init_) {
    GELOGD("pre process for custom op has already been called");
    return GRAPH_SUCCESS;
  }
  std::string custom_opp_path;
  const char_t *custom_opp_path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_CUSTOM_OPP_PATH, custom_opp_path_env);
  if (custom_opp_path_env != nullptr) {
    custom_opp_path = custom_opp_path_env;
  }
  std::vector<std::string> so_real_paths;
  GE_ASSERT_GRAPH_SUCCESS(GetAllCustomOpApiSoPaths(custom_opp_path, so_real_paths));
  GE_ASSERT_GRAPH_SUCCESS(CallInitFunc(custom_opp_path, so_real_paths));
  is_init_ = true;
  return GRAPH_SUCCESS;
}

graphStatus OpLibRegistry::GetAllCustomOpApiSoPaths(const std::string &custom_opp_path,
                                                    std::vector<std::string> &so_real_paths) const {
  if (custom_opp_path.empty()) {
    GELOGI("custom_opp_path is empty, no need to get custom op so");
    return GRAPH_SUCCESS;
  }
  GELOGI("value of env ASCEND_CUSTOM_OPP_PATH is %s.", custom_opp_path.c_str());
  std::vector<std::string> current_custom_opp_path = StringUtils::Split(custom_opp_path, ':');

  if (current_custom_opp_path.empty()) {
    GELOGI("find no custom opp path, just return");
    return GRAPH_SUCCESS;
  }

  for (const auto &path : current_custom_opp_path) {
    if (path.empty()) {
      continue;
    }
    const std::string so_path = path + "/" + custom_so_name;
    std::string so_real_path = RealPath(so_path.c_str());
    if (!so_real_path.empty()) {
      GELOGI("find so_real_path %s", so_real_path.c_str());
      (void) so_real_paths.emplace_back(so_real_path);
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus OpLibRegistry::CallInitFunc(const std::string &custom_opp_path,
                                        const std::vector<std::string> &so_real_paths) {
  // dlopen so orderly
  for (const auto &so_path : so_real_paths) {
    GELOGI("begin dlopen %s", so_path.c_str());
    void* const handle = mmDlopen(so_path.c_str(), static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW)));
    GE_ASSERT_NOTNULL(handle, "Failed to dlopen %s! errmsg:%s", so_path.c_str(), mmDlerror());
    (void) handles_.emplace_back(handle);
  }

  // call init func orderly
  const std::lock_guard<std::mutex> lk(mu_);
  for (auto &vendor_func : vendor_funcs_) {
    GELOGI("begin to call %s init func", vendor_func.first.c_str());
    AscendString tmp_dir("");
    GE_ASSERT_GRAPH_SUCCESS(vendor_func.second(tmp_dir));
    GELOGI("end to call %s init func, tmp_dir is %s", vendor_func.first.c_str(), tmp_dir.GetString());
    op_lib_paths_ += (std::string(tmp_dir.GetString()) + ":");
  }
  if (custom_opp_path.empty()) { // ignore the end :
    op_lib_paths_ = op_lib_paths_.substr(0, op_lib_paths_.find_last_of(':'));
  } else {
    op_lib_paths_ += custom_opp_path; // add origin env path to ensure priority(so mode first, runbag mode second)
  }
  PluginManager::SetCustomOpLibPath(op_lib_paths_);
  if (SetMetadefPluginCustomOpLibPathForC != nullptr) {
    SetMetadefPluginCustomOpLibPathForC(op_lib_paths_.c_str());
  }
  GELOGI("CallInitFunc %zu successfully, op_lib_paths_ is %s", vendor_funcs_.size(), op_lib_paths_.c_str());
  return GRAPH_SUCCESS;
}

void OpLibRegistry::ClearHandles() {
  for (auto handle : handles_) {
    (void)mmDlclose(handle);
  }
  handles_.clear();
}

OpLibRegistry::~OpLibRegistry() {
  ClearHandles();
}
} // namespace ge
