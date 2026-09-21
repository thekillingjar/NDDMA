/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "op_def_impl.h"

namespace ops {
OpMC2Def::OpMC2Def() : impl_(new(std::nothrow) OpMC2DefImpl) {}

OpMC2Def::OpMC2Def(const OpMC2Def &mc2_def) : impl_(new(std::nothrow) OpMC2DefImpl) {
  this->impl_->group_list = mc2_def.impl_->group_list;
}

OpMC2Def::~OpMC2Def() = default;

OpMC2Def &OpMC2Def::operator=(const OpMC2Def &mc2_def) {
  if (this != &mc2_def) {
    *this->impl_ = *mc2_def.impl_;
  }
  return *this;
}

OpMC2Def &OpMC2Def::HcclGroup(const char *value) {
  if (std::find(this->impl_->group_list.begin(), this->impl_->group_list.end(), value) ==
      this->impl_->group_list.end()) {
    this->impl_->group_list.emplace_back(value);
  }
  return *this;
}

OpMC2Def &OpMC2Def::HcclGroup(std::vector<const char *> value) {
  for (const char *val : value) {
    if (std::find(this->impl_->group_list.begin(), this->impl_->group_list.end(), val) ==
        this->impl_->group_list.end()) {
      this->impl_->group_list.emplace_back(val);
    }
  }
  return *this;
}

std::vector<ge::AscendString> &OpMC2Def::GetHcclGroups(void) const {
  return this->impl_->group_list;
}

void OpMC2Def::HcclServerType(enum HcclServerType type, const char* soc) {
  ge::AscendString soc_version;
  if (soc == nullptr || strlen(soc) == 0) {
    soc_version = "";
  } else {
    soc_version = soc;
  }
  this->impl_->server_type_[soc_version] = type;
}

/**
 * @brief get hccl server type by soc version
 * @param soc_version "" means checking if any hccl server type has been set
 * @return hccl server type corresponding to soc version.
           For scenarios where soc version is empty, return MAX if not set, AICPU if set.
 */
enum HcclServerType OpMC2Def::GetHcclServerType(const ge::AscendString &soc_version) const {
  if (this->impl_->server_type_.empty()) {
    return HcclServerType::MAX;
  }
  if (soc_version.GetLength() == 0) {
    return HcclServerType::AICPU;
  }
  if (this->impl_->server_type_.find(soc_version) != this->impl_->server_type_.end()) {
    return this->impl_->server_type_[soc_version];
  }
  if (this->impl_->server_type_.find("") != this->impl_->server_type_.end()) {
    return this->impl_->server_type_[""];
  }
  return HcclServerType::MAX;
}

}  // namespace ops
