/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_SUBSCRIBER_PROFILER_DFX_EXTEND_INFO_H_
#define AIR_CXX_RUNTIME_V2_SUBSCRIBER_PROFILER_DFX_EXTEND_INFO_H_
#include <type_traits>
#include <cstdint>
#include <cstddef>
#include <unordered_map>
#include "graph/types.h"
#include "engine/node_converter_utils.h"
#include "engine/aicore/fe_rt2_common.h"
namespace gert {
  class DfxExtendInfo {
    public:
      const ge::char_t *GetNodeName() const {
        return node_name_;
      }

      DfxExtendInfo() = delete;
      DfxExtendInfo(const DfxExtendInfo &) = delete;
      DfxExtendInfo(DfxExtendInfo &&) = delete;
      DfxExtendInfo &operator=(const DfxExtendInfo &) = delete;
      DfxExtendInfo &operator=(DfxExtendInfo &&) = delete;
      static ge::graphStatus CalcSize(const size_t ctx_id_num,
                                      std::vector<std::vector<uint32_t>> &cmo_context_ids,
                                      size_t &total_size);
      ge::graphStatus Init(const ge::char_t *node_name, std::vector<std::string> &cmo_context_names,
                        std::vector<std::vector<uint32_t>> &cmo_context_ids);
      void SetCmoInfo(std::vector<uint32_t> &cmo_context_types,
                                    std::vector<std::vector<uint32_t>> &cmo_context_ids);
      void SetCtxInfo(const size_t task_type, const size_t ctx_type, const size_t op_impl_mode,
                              const std::vector<uint32_t> ctx_id_vec);
      const uint32_t *GetOpImplModeAddr() const;
      const uint32_t *GetTaskTypeAddr() const;
      const uint32_t *GetCtxTypeAddr();
      const uint32_t *GetCtxSizeAddr();
      const uint32_t *GetCtxIdAddr(size_t index);
      const uint32_t *GetPreCmoCtxSizeAddr();
      const uint32_t *GetPreCmoCtxTypeAddr();
      const uint32_t *GetPreCmoCtxIdAddr(size_t index);
      const uint32_t *GetInvalCmoCtxSizeAddr();
      const uint32_t *GetInvalCmoCtxTypeAddr();
      const uint32_t *GetInvalCmoCtxIdAddr(size_t index);
      const uint32_t *GetWbackCmoCtxSizeAddr();
      const uint32_t *GetWbackCmoCtxTypeAddr();
      const uint32_t *GetWbackCmoCtxIdAddr(size_t index);
      size_t GetCtxIdsNum();
      size_t GetPreCmoCtxNum();
      size_t GetInvalCmoCtxNum();
      size_t GetWbackCmoCtxNum();
    private:
      const uint32_t *GetCmoBaseAddr();
      size_t ctx_id_num_ = 0U;
      size_t cmo_context_num_[3];
      const ge::char_t *node_name_;
      const ge::char_t *cmo_context_name_[3];
      uint8_t reserved_[8];
      uint8_t place_holder_;
  };
  static_assert(std::is_standard_layout<DfxExtendInfo>::value, "The class DfxExtendInfo must be a POD");
 }
#endif  // AIR_CXX_RUNTIME_V2_SUBSCRIBER_PROFILER_DFX_EXTEND_INFO_H_
