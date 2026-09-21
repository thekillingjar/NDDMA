/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/cmo_id_gen_strategy.h"
#include "common/fe_log.h"

namespace fe {
CMOIdGenStrategy::CMOIdGenStrategy() {}
CMOIdGenStrategy::~CMOIdGenStrategy() {}

CMOIdGenStrategy& CMOIdGenStrategy::Instance() {
  static CMOIdGenStrategy cmoIdGenStrategy;
  return cmoIdGenStrategy;
}

Status CMOIdGenStrategy::Finalize() {
  std::lock_guard<std::mutex> lock_guard(cmo_id_mutex_);
  reuse_cmo_id_map_.clear();
  return SUCCESS;
}

uint16_t CMOIdGenStrategy::GenerateCMOId(const ge::Node &node) {
  std::lock_guard<std::mutex> lock_guard(cmo_id_mutex_);
  int64_t topo_id = node.GetOpDesc()->GetId();
  FE_LOGD("Begin to generate cmo id for node[name:%s, type:%s].", node.GetName().c_str(), node.GetType().c_str());
  auto iter = reuse_cmo_id_map_.begin();
  while (iter != reuse_cmo_id_map_.end()) {
    if (topo_id > iter->second) {
      uint16_t cmo_id = iter->first;
      (void)reuse_cmo_id_map_.erase(iter++);
      return cmo_id;
    } else {
      iter++;
    }
  }
  uint32_t cmo_id = GetAtomicId();
  if (cmo_id > UINT16_MAX) {
    return 0;
  }
  return static_cast<uint16_t>(cmo_id);
}

void CMOIdGenStrategy::UpdateReuseMap(const int64_t &topo_id, const uint16_t &cmo_id) {
  std::lock_guard<std::mutex> lock_guard(cmo_id_mutex_);
  reuse_cmo_id_map_[cmo_id] = topo_id;
}

uint32_t CMOIdGenStrategy::GetAtomicId() const {
  static std::atomic<uint32_t> global_cmo_id(1);
  return global_cmo_id.fetch_add(1, std::memory_order_relaxed);
}
}
