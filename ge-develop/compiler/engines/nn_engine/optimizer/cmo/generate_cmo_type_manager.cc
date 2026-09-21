/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "generate_cmo_type_manager.h"
#include "generate_cmo_type_invalid.h"
#include "generate_cmo_type_prefetch.h"
#include "generate_cmo_type_writeback.h"
#include "common/fe_log.h"

namespace fe {
Status GenerateCMOTypeManager::Register(CmoType type) {
  GenerateCMOTypeBasePtr cmo_type_ptr = nullptr;
  switch (type) {
    case CmoType::CMO_TYPE_PREFETCH:
      FE_MAKE_SHARED(cmo_type_ptr = std::make_shared<GenerateCMOTypePrefetch>(), return FAILED);
      break;
    case CmoType::CMO_TYPE_INVALID:
      FE_MAKE_SHARED(cmo_type_ptr = std::make_shared<GenerateCMOTypeInvalid>(), return FAILED);
      break;
    case CmoType::CMO_TYPE_WRITEBACK:
      FE_MAKE_SHARED(cmo_type_ptr = std::make_shared<GenerateCMOTypeWriteback>(), return FAILED);
      break;
    default:
      break;
  }
  if (cmo_type_ptr != nullptr) {
    cmo_generate_map_.insert(std::make_pair(type, cmo_type_ptr));
  }
  return SUCCESS;
}

Status GenerateCMOTypeManager::Initialize() {
  if (init_flag_) {
    return SUCCESS;
  }
  init_flag_ = true;
  if (!cmo_generate_map_.empty()) {
    return SUCCESS;
  }
  for (int i = 0; i < static_cast<int>(CmoType::CMO_TYPE_BUTT); i++) {
    if (Register(static_cast<CmoType>(i)) != SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

Status GenerateCMOTypeManager::Finalize() {
  cmo_generate_map_.clear();
  init_flag_ = false;
  return SUCCESS;
}

void GenerateCMOTypeManager::GenerateType(const ge::NodePtr &node,
                                          const StreamCtrlMap &stream_ctrls,
                                          std::unordered_map<ge::NodePtr, ge::NodePtr> &prefetch_cache_map,
                                          std::map<uint32_t, std::map<int64_t, ge::NodePtr>> &stream_node_map) const {
  FE_LOGD("begin to generate cmo type for node:[name=%s, type=%s]",
          node->GetOpDesc()->GetName().c_str(), node->GetOpDesc()->GetType().c_str());
  for (int i = 0; i < static_cast<int>(CmoType::CMO_TYPE_BUTT); i++) {
    auto cmo_type = cmo_generate_map_.find(static_cast<CmoType>(i));
    if (cmo_type != cmo_generate_map_.end()) {
      cmo_type->second->GenerateType(node, stream_ctrls, prefetch_cache_map, stream_node_map);
    }
  }
  FE_LOGD("Ending generation of CMO type for node: [name=%s, type=%s]",
          node->GetOpDesc()->GetName().c_str(), node->GetOpDesc()->GetType().c_str());
}
}
