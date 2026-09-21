/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PASS_PASS_MGR_H_
#define PASS_PASS_MGR_H_

#include <unordered_map>
#include <string>
#include <vector>
#include <functional>
#include "parser/tuning_space.h"

namespace att {
using PassFunc = std::function<bool(const TuningSpacePtr&, std::map<std::string, std::string>&)>;

class ATTPassMgr {
public:
  static ATTPassMgr& Instance()
  {
    static ATTPassMgr pass_mgr;
    return pass_mgr;
  }

  void RegistePass(const std::string &pass_name, const PassFunc &func)
  {
    pass_name_list_.emplace_back(pass_name);
    pass_func_map_[pass_name] = func;
    GELOGI("Register pass name[%s].", pass_name.c_str());
  }

  PassFunc GetPass(const std::string &pass_name)
  {
    if (pass_func_map_.find(pass_name) == pass_func_map_.end()) {
      return nullptr;
    }
    return pass_func_map_[pass_name];
  }

  void GetPassList(std::vector<PassFunc> &res)
  {
    for (const auto &pass_name : pass_name_list_) {
      res.emplace_back(pass_func_map_[pass_name]);
    }
  }

private:
  ATTPassMgr() = default;
  ~ATTPassMgr() = default;

private:
  std::vector<std::string> pass_name_list_;
  std::unordered_map<std::string, PassFunc> pass_func_map_;
};

class PassRegister {
public:
  PassRegister(const std::string &pass_name, const PassFunc &func)
  {
    ATTPassMgr::Instance().RegistePass(pass_name, func);
  }
  ~PassRegister() =default;
};

#define REGISTER_GTC_PASS(pass_name, func_name) \
  static PassRegister g_Reg##pass_name(pass_name, func_name)
} // namespace att

#endif  // PASS_PASS_MGR_H_