/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dfx_extend_info.h"

#include <unordered_map>
#include "common/checker.h"
#include "utils/extern_math_util.h"

namespace gert {
// 各属性在POD内存中的偏移量，属性排列为：
// op_impl_mode\task_type\ctx_type\ctx_id_size\ctx_ids\cmo_ids_size\cmo_id_type\cmo_ids
// 单个uint32_t字段的固定偏移量
constexpr uint32_t kStepOffset = 1U;
constexpr uint32_t kCtxTypeAddrOffset = 2U;
// 每个CMO属性的固定偏移量
constexpr uint32_t kCmoStepOffset = 2U;
constexpr uint32_t kCtxSizeAddrOffset = 3U;
constexpr uint32_t kCtxIdAddrOffset = 4U;
// 分别为cmo各属性在vector中的index
constexpr uint32_t pre_cmo_index = 0U;
constexpr uint32_t inval_cmo_index = 1U;
constexpr uint32_t wback_cmo_index = 2U;
ge::graphStatus DfxExtendInfo::CalcSize(const size_t ctx_id_num,
                                        std::vector<std::vector<uint32_t>> &cmo_context_ids,
                                        size_t &total_size) {
  size_t ctx_id_size;
  size_t cmo_context_size = 0U;
  GE_ASSERT_TRUE(!ge::MulOverflow(sizeof(uint32_t), ctx_id_num, ctx_id_size));
  for (auto iter = cmo_context_ids.begin(); iter != cmo_context_ids.end(); ++iter) {
    size_t cmo_context_size_tmp;
    GE_ASSERT_TRUE(!ge::MulOverflow(sizeof(uint32_t), iter->size(), cmo_context_size_tmp));
    GE_ASSERT_TRUE(!ge::AddOverflow(cmo_context_size, cmo_context_size_tmp, cmo_context_size));
    GE_ASSERT_TRUE(!ge::AddOverflow(cmo_context_size, kCmoStepOffset * sizeof(uint32_t), cmo_context_size));
  }

  total_size = sizeof(DfxExtendInfo);
  GE_ASSERT_TRUE(!ge::AddOverflow(total_size, (kCtxIdAddrOffset) * sizeof(uint32_t), total_size));
  GE_ASSERT_TRUE(!ge::AddOverflow(total_size, ctx_id_size, total_size));
  GE_ASSERT_TRUE(!ge::AddOverflow(total_size, cmo_context_size, total_size));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DfxExtendInfo::Init(const ge::char_t *node_name, std::vector<std::string> &cmo_context_names,
                                    std::vector<std::vector<uint32_t>> &cmo_context_ids) {
  node_name_ = node_name;
  for (uint32_t i = 0U; i < CMO_IDX_KEY.size(); i++) {
    cmo_context_num_[i] = cmo_context_ids.at(i).size();
    auto cmo_ctx_name = cmo_context_names.at(i);
    cmo_context_name_[i] = const_cast<ge::char_t *>(cmo_ctx_name.c_str());
  }
  auto ret = memset_s(reserved_, sizeof(reserved_), 0, sizeof(reserved_));
  if (ret != EOK) {
    GELOGE(ge::GRAPH_FAILED, "MemSet DfxExtendInfo failed!");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

// 设置CMO相关信息
void DfxExtendInfo::SetCmoInfo(std::vector<uint32_t> &cmo_context_types,
                                     std::vector<std::vector<uint32_t>> &cmo_context_ids) {
    // Set cmo ctx size
  const auto mutable_pre_cmo_ctx_size = GetPreCmoCtxSizeAddr();
  if (mutable_pre_cmo_ctx_size != nullptr) {
    *(const_cast<uint32_t *>(mutable_pre_cmo_ctx_size)) = cmo_context_num_[pre_cmo_index];
  }
   
  if (cmo_context_num_[pre_cmo_index] > 0U) {
    // SetPreCmoCtxType
    const auto mutable_pre_cmo_ctx_type = GetPreCmoCtxTypeAddr();
    if (mutable_pre_cmo_ctx_type != nullptr) {
      *(const_cast<uint32_t *>(mutable_pre_cmo_ctx_type)) = cmo_context_types.at(pre_cmo_index);
    }
        
    // set cmo ctx ids
    for (size_t i = 0U; i < cmo_context_num_[pre_cmo_index]; i++) {
      const auto mutable_pre_ctx_id = GetPreCmoCtxIdAddr(i);
      if (mutable_pre_ctx_id != nullptr) {
        *(const_cast<uint32_t *>(mutable_pre_ctx_id)) = cmo_context_ids.at(pre_cmo_index).at(i);
      }
    }
  }

  const auto mutable_inval_cmo_ctx_size = GetInvalCmoCtxSizeAddr();
  if (mutable_inval_cmo_ctx_size != nullptr) {
    *(const_cast<uint32_t *>(mutable_inval_cmo_ctx_size)) = cmo_context_num_[inval_cmo_index];
  }
    
  if (cmo_context_num_[inval_cmo_index] > 0U) {
    // SetPreCmoCtxType
    const auto mutable_inval_cmo_ctx_type = GetInvalCmoCtxTypeAddr();
    if (mutable_inval_cmo_ctx_type != nullptr) {
      *(const_cast<uint32_t *>(mutable_inval_cmo_ctx_type)) = cmo_context_types.at(1);
    }
        
    // set cmo ctx ids
    for (size_t i = 0U; i < cmo_context_num_[inval_cmo_index]; i++) {
      const auto mutable_inval_ctx_id = GetInvalCmoCtxIdAddr(i);
      if (mutable_inval_ctx_id != nullptr) {
        *(const_cast<uint32_t *>(mutable_inval_ctx_id)) = cmo_context_ids.at(inval_cmo_index).at(i);
      }
    }
  }

  const auto mutable_wback_cmo_ctx_size = GetWbackCmoCtxSizeAddr();
  if (mutable_wback_cmo_ctx_size != nullptr) {
    *(const_cast<uint32_t *>(mutable_wback_cmo_ctx_size)) = cmo_context_num_[wback_cmo_index];
  }

  if (cmo_context_num_[wback_cmo_index] > 0U) {
    // SetPreCmoCtxType
    const auto mutable_wback_cmo_ctx_type = GetWbackCmoCtxTypeAddr();
    if (mutable_wback_cmo_ctx_type != nullptr) {
      *(const_cast<uint32_t *>(mutable_wback_cmo_ctx_type)) = cmo_context_types.at(wback_cmo_index);
    }
        
    // set cmo ctx ids
    for (size_t i = 0U; i < cmo_context_num_[wback_cmo_index]; i++) {
      const auto mutable_wback_ctx_id = GetWbackCmoCtxIdAddr(i);
      if (mutable_wback_ctx_id != nullptr) {
        *(const_cast<uint32_t *>(mutable_wback_ctx_id)) = cmo_context_ids.at(wback_cmo_index).at(i);
      }
    }
  }
}

// 设置CTX相关信息
void DfxExtendInfo::SetCtxInfo(const size_t task_type, const size_t ctx_type, const size_t op_impl_mode,
                               const std::vector<uint32_t> ctx_id_vec) {
    // set op impl mode
    const auto mutable_op_impl_mode = GetOpImplModeAddr();
    *(const_cast<uint32_t *>(mutable_op_impl_mode)) = op_impl_mode;

    // set task type
    const auto mutable_task_type = GetTaskTypeAddr();
    *(const_cast<uint32_t *>(mutable_task_type)) = task_type;

    // SetCtxType
    const auto mutable_ctx_type = GetCtxTypeAddr();
    *(const_cast<uint32_t *>(mutable_ctx_type)) = ctx_type;

    // Set ctx size
    ctx_id_num_ += ctx_id_vec.size();
    const auto mutable_ctx_size = GetCtxSizeAddr();
    *(const_cast<uint32_t *>(mutable_ctx_size)) = ctx_id_num_;

    // SetCtxIds
    for (size_t i = 0U; i < ctx_id_vec.size(); i++) {
        const auto mutable_ctx_id = GetCtxIdAddr(i);
        if (mutable_ctx_id != nullptr) {
            *(const_cast<uint32_t *>(mutable_ctx_id)) = ctx_id_vec.at(i);
        }
    }
}

// 依次获取各字段的地址
const uint32_t *DfxExtendInfo::GetOpImplModeAddr() const {
  return reinterpret_cast<const uint32_t *>(&place_holder_);
}

const uint32_t *DfxExtendInfo::GetTaskTypeAddr() const {
  return reinterpret_cast<const uint32_t *>(&place_holder_ + sizeof(uint32_t));
}

const uint32_t *DfxExtendInfo::GetCtxTypeAddr() {
    return reinterpret_cast<const uint32_t *>(&place_holder_ + kCtxTypeAddrOffset * sizeof(uint32_t));
}

const uint32_t *DfxExtendInfo::GetCtxSizeAddr() {
    return reinterpret_cast<const uint32_t *>(&place_holder_ + kCtxSizeAddrOffset * sizeof(uint32_t));
}

const uint32_t *DfxExtendInfo::GetCtxIdAddr(size_t index) {
    if (*(GetCtxSizeAddr()) == 0U) {
        return nullptr;
    }
    return reinterpret_cast<const uint32_t *>(&place_holder_ + (kCtxIdAddrOffset + index)*sizeof(uint32_t));
}

const uint32_t *DfxExtendInfo::GetCmoBaseAddr() {
    return reinterpret_cast<const uint32_t *>(&place_holder_ + (kCtxIdAddrOffset + ctx_id_num_)*sizeof(uint32_t));
}

// 使用字段时，先对地址指针判空，如果返回空指针，则该属性不存在
const uint32_t *DfxExtendInfo::GetPreCmoCtxSizeAddr() {
    if (cmo_context_num_[pre_cmo_index] == 0U) {
        return nullptr;
    } else {
        return GetCmoBaseAddr();
    }
}

const uint32_t *DfxExtendInfo::GetPreCmoCtxTypeAddr() {
    if (cmo_context_num_[pre_cmo_index] == 0U) {
        return nullptr;
    } else {
        return GetPreCmoCtxSizeAddr() + kStepOffset;
    }
}

const uint32_t *DfxExtendInfo::GetPreCmoCtxIdAddr(size_t index) {
    if (GetPreCmoCtxSizeAddr() == nullptr) {
        return nullptr;
    } else {
        return GetPreCmoCtxTypeAddr() + kStepOffset + index;
    }
}

const uint32_t *DfxExtendInfo::GetInvalCmoCtxSizeAddr() {
    if (cmo_context_num_[inval_cmo_index] == 0U) {
        return nullptr;
    }

    if (cmo_context_num_[pre_cmo_index] == 0U) {
        return GetCmoBaseAddr();
    } else {
        return GetCmoBaseAddr() + kCmoStepOffset + cmo_context_num_[pre_cmo_index];
    }
}

const uint32_t *DfxExtendInfo::GetInvalCmoCtxTypeAddr() {
    if (cmo_context_num_[inval_cmo_index] == 0U) {
        return nullptr;
    } else {
        return GetInvalCmoCtxSizeAddr() + kStepOffset;
    }
}

const uint32_t *DfxExtendInfo::GetInvalCmoCtxIdAddr(size_t index) {
    if (GetInvalCmoCtxSizeAddr() == nullptr) {
        return nullptr;
    } else {
        return GetInvalCmoCtxTypeAddr() + kStepOffset + index;
    }
}

const uint32_t *DfxExtendInfo::GetWbackCmoCtxSizeAddr() {
    if (cmo_context_num_[wback_cmo_index] == 0U) {
        return nullptr;
    }
    auto cmo_base_addr = GetCmoBaseAddr();
    if (cmo_context_num_[pre_cmo_index] != 0U) {
        cmo_base_addr += kCmoStepOffset + cmo_context_num_[pre_cmo_index];
    }

    if (cmo_context_num_[inval_cmo_index] != 0U) {
        cmo_base_addr += kCmoStepOffset + cmo_context_num_[inval_cmo_index];
    }

    return cmo_base_addr;
}

const uint32_t *DfxExtendInfo::GetWbackCmoCtxTypeAddr() {
    if (cmo_context_num_[wback_cmo_index] == 0U) {
        return nullptr;
    } else {
        return GetWbackCmoCtxSizeAddr() + kStepOffset;
    }
}

const uint32_t *DfxExtendInfo::GetWbackCmoCtxIdAddr(size_t index) {
    if (GetInvalCmoCtxSizeAddr() == nullptr) {
        return nullptr;
    } else {
        return GetWbackCmoCtxTypeAddr() + kStepOffset + index;
    }
}

//  获取集合的尺寸，后续根据尺寸获取属性值, 对空集合，返回0
size_t DfxExtendInfo::GetCtxIdsNum() {
    if (GetCtxSizeAddr() == nullptr) {
        return 0U;
    }
    return *(GetCtxSizeAddr());
}

size_t DfxExtendInfo::GetPreCmoCtxNum() {
    if (GetPreCmoCtxSizeAddr() == nullptr) {
        return 0U;
    }
    return *(GetPreCmoCtxSizeAddr());
}

size_t DfxExtendInfo::GetInvalCmoCtxNum() {
    if (GetInvalCmoCtxSizeAddr() == nullptr) {
        return 0U;
    }
    return *(GetInvalCmoCtxSizeAddr());
}

size_t DfxExtendInfo::GetWbackCmoCtxNum() {
    if (GetWbackCmoCtxSizeAddr() == nullptr) {
        return 0U;
    }
    return *(GetWbackCmoCtxSizeAddr());
}
}
