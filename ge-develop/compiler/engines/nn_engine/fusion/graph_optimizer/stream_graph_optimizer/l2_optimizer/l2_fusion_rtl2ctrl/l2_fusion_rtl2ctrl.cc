/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_rtl2ctrl/l2_fusion_rtl2ctrl.h"
#include <vector>
#include "common/fe_log.h"

namespace fe {
void L2FusionRtl2ctrl::InitRtl2ctrl(rtL2Ctrl_t &l2ctrl) {
  l2ctrl.size = 0;
  l2ctrl.l2_in_main = 0;

  for (uint32_t i = 0; i < L2_CTRL_REMAP_SIZE; ++i) {
    l2ctrl.remap[i] = 0;
  }

  for (uint32_t i = 0; i < L2_CTRL_DATA_SIZE; ++i) {
    l2ctrl.data[i].L2_mirror_addr = 0;
    l2ctrl.data[i].L2_data_section_size = 0;
    l2ctrl.data[i].L2_preload = 0;
    l2ctrl.data[i].modified = 0;
    l2ctrl.data[i].priority = 0;
    l2ctrl.data[i].prev_L2_page_offset_base = -1;
    l2ctrl.data[i].L2_page_offset_base = 0;
    l2ctrl.data[i].L2_load_to_ddr = 0;
  }
}

Status L2FusionRtl2ctrl::UpdateInputDatal2ctrlId(TensorL2AllocMap &input, TensorL2AllocMap &standing_data) {
  for (TensorL2AllocMap::iterator it = input.begin(); it != input.end(); ++it) {
    FE_CHECK(standing_data.count(it->first) == 0,
             REPORT_FE_ERROR("[StreamOpt][L2Opt][UpdInDataL2CtrlId] Standing Data value is zero."), return fe::FAILED);
    it->second.data_in_l2_id = standing_data[it->first].data_in_l2_id;
  }
  return fe::SUCCESS;
}

void L2FusionRtl2ctrl::SetDataL2ctrl(uint32_t max_data_num, TensorL2AllocMap &data, rtL2Ctrl_t &l2ctrl) {
  for (auto it = data.cbegin(); it != data.cend(); ++it) {
    uint32_t data_id = it->second.data_in_l2_id;
    if (data_id >= max_data_num) {
      FE_LOGD("Data ID is %u, which is greater than or equal to the maximum data number (%u).", data_id, max_data_num);
      return;
    }

    l2ctrl.data[data_id].L2_mirror_addr = it->second.data.ddr_addr;
    l2ctrl.data[data_id].L2_data_section_size = it->second.data.data_size;
    l2ctrl.data[data_id].L2_preload = it->second.L2_preload;
    l2ctrl.data[data_id].modified = it->second.modified;
    l2ctrl.data[data_id].priority = it->second.priority;
    l2ctrl.data[data_id].prev_L2_page_offset_base = it->second.pre_L2_page_offset_base;
    l2ctrl.data[data_id].L2_page_offset_base = it->second.L2_page_offset_base;
    l2ctrl.data[data_id].L2_load_to_ddr = it->second.L2_load_to_ddr;
  }
}

void L2FusionRtl2ctrl::SetOutputDataL2ctrl(uint32_t max_data_num, TensorL2AllocMap &standing,
                                           TensorL2AllocMap &data, rtL2Ctrl_t &out_l2ctrl) {
  for (TensorL2AllocMap::iterator it = data.begin(); it != data.end(); ++it) {
    uint32_t data_id = it->second.data_in_l2_id;
    if (data_id >= max_data_num) {
      FE_LOGD("Data Id is %u, which is greater than or equal to the maximum allowed data number: %u", data_id, max_data_num);
      return;
    }
    if (standing.find(it->first) == standing.end() && it->second.occupy_data_id < 0) {
      out_l2ctrl.data[data_id].L2_mirror_addr = it->second.data.ddr_addr;
      out_l2ctrl.data[data_id].L2_data_section_size = it->second.data.data_size;
      out_l2ctrl.data[data_id].L2_preload = it->second.L2_preload;
      out_l2ctrl.data[data_id].modified = it->second.modified;
      out_l2ctrl.data[data_id].priority = it->second.priority;
      out_l2ctrl.data[data_id].prev_L2_page_offset_base = it->second.pre_L2_page_offset_base;
      out_l2ctrl.data[data_id].L2_page_offset_base = it->second.L2_page_offset_base;
      out_l2ctrl.data[data_id].L2_load_to_ddr = it->second.L2_load_to_ddr;
    } else {
      out_l2ctrl.data[data_id].L2_mirror_addr = it->second.data.ddr_addr;
    }
  }
}

// create the l2ctrl in l2
Status L2FusionRtl2ctrl::UpdateRtL2Ctrl(OpL2AllocMap &l2_alloc_map, uint64_t max_page_num, uint32_t max_data_num) {
  for (OpL2AllocMap::iterator it = l2_alloc_map.begin(); it != l2_alloc_map.end(); ++it) {
    InitRtl2ctrl(it->second.l2ctrl);
    it->second.l2ctrl.size = max_page_num;

    SetDataL2ctrl(max_data_num, it->second.standing_data, it->second.l2ctrl);
    SetOutputDataL2ctrl(max_data_num, it->second.standing_data, it->second.output, it->second.l2ctrl);
    SetOutputDataL2ctrl(max_data_num, it->second.standing_data, it->second.converge, it->second.l2ctrl);
    SetDataL2ctrl(max_data_num, it->second.filter_preload, it->second.l2ctrl);
    SetDataL2ctrl(max_data_num, it->second.input_preload, it->second.l2ctrl);

    FE_CHECK(UpdateInputDatal2ctrlId(it->second.input, it->second.standing_data) != fe::SUCCESS,
             REPORT_FE_ERROR("[StreamOpt][L2Opt][UpdRtL2Ctrl] UpdateInputDatal2ctrlId failed!"), return fe::FAILED);
  }

  return fe::SUCCESS;
}
}  // namespace fe
