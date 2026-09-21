/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_handler/l2_fusion_handler.h"
#include <map>
#include <string>
#include <vector>
#include "common/math_util.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_allocation/l2_fusion_allocation.h"
#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_parser/l2_fusion_parser.h"
#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_rtl2ctrl/l2_fusion_rtl2ctrl.h"

namespace fe {
Status L2FusionHandler::GetL2DataAlloc(const uint64_t &mem_base, const ge::ComputeGraph &graph,
                                       TaskL2InfoMap &l2_info_map) {
  FE_LOGD("We use normal l2 buffer here.");
  FE_CHECK(GetNormalL2DataAlloc(mem_base, graph, l2_info_map) != SUCCESS,
           REPORT_FE_ERROR("[StreamOpt][L2Opt][UpdL2FusIn] GetNormalL2DataAlloc operation failed!"),
           return FAILED);
  return SUCCESS;
}

Status L2FusionHandler::GetNormalL2DataAlloc(const uint64_t &mem_base, const ge::ComputeGraph &graph,
                                             TaskL2InfoMap &l2_info_map) {
  // get l2 buffer info
  L2BufferInfo l2_buffer_info;
  L2FusionComm::GetL2HardwareSet(l2_buffer_info);

  // get dependect map for alloc
  std::map<uint64_t, size_t> depend_count_map;
  FE_CHECK(L2FusionParser::GetDataDependentCountMap(graph, depend_count_map) != SUCCESS,
           REPORT_FE_ERROR("[StreamOpt][L2Opt][GetNormalL2DataAlloc] GetDataDependentCountMap failed!"),
           return FAILED);

  OpReferTensorIndexMap refer_tensor_index_map;
  L2FusionParser::GetOpReferTensorIndexMap(graph, refer_tensor_index_map);

  // get data which can be allocated in l2
  std::vector<OpL2DataInfo> op_l2_data_vec;
  FE_CHECK(L2FusionParser::GetDataFromGraph(graph, refer_tensor_index_map, op_l2_data_vec) != SUCCESS,
           REPORT_FE_ERROR("[StreamOpt][L2Opt][GetNormalL2DataAlloc] GetDataFromGraph failed!"), return FAILED);
  L2FusionComm::DisplayOpL2DataInfo(op_l2_data_vec);

  uint64_t max_page_num = 0;
  OpL2AllocMap l2_alloc_map;
  FE_CHECK(L2FusionAllocation::AllocateData(l2_buffer_info, op_l2_data_vec, depend_count_map,
                                            l2_alloc_map, max_page_num) != SUCCESS,
           REPORT_FE_ERROR("[StreamOpt][L2Opt][GetNormalL2DataAlloc] Allocate Data failed!"), return FAILED);
  FE_CHECK(L2FusionRtl2ctrl::UpdateRtL2Ctrl(l2_alloc_map, max_page_num, l2_buffer_info.max_data_num) != SUCCESS,
           REPORT_FE_ERROR("[StreamOpt][L2Opt][GetNormalL2DataAlloc] UpdateRtL2Ctrl failed!"), return FAILED);
  L2FusionComm::DisplayOpL2AllocInfo(l2_alloc_map);
  FE_CHECK(GenerateFinalL2Info(mem_base, graph, l2_alloc_map, l2_info_map) != SUCCESS,
           REPORT_FE_ERROR("[StreamOpt][L2Opt][GetNormalL2DataAlloc] GenerateFinalL2Info failed!"), return FAILED);
  DisplayL2Info(l2_info_map);
  return SUCCESS;
}

void L2FusionHandler::GenerateL2Data(const uint64_t &mem_base, const TensorL2AllocMap &src_data,
                                     L2DataMap &dst_data) {
  dst_data.clear();
  for (TensorL2AllocMap::const_iterator data_it = src_data.begin(); data_it != src_data.end(); ++data_it) {
    if (CheckUint64AddOverflow(mem_base, data_it->second.data.ddr_addr) != SUCCESS) {
      REPORT_FE_ERROR("[StreamOpt][L2Opt][GenerL2Data] Addition of UINT64 values %lu and %lu can result in overflow!",
                      mem_base, data_it->second.data.ddr_addr);
      return;
    }
    uint64_t ddr_addr = mem_base + data_it->second.data.ddr_addr;
    L2Data data_tmp;
    data_tmp.l2Index = data_it->second.data_in_l2_id;
    data_tmp.l2Addr = data_it->second.data_in_l2_addr;
    data_tmp.l2PageNum = data_it->second.l2PageNum;
    dst_data.insert(L2DataPair(ddr_addr, data_tmp));
  }
}

Status L2FusionHandler::GenerateFinalL2Info(const uint64_t &mem_base, const ge::ComputeGraph &graph,
                                            const OpL2AllocMap &l2_alloc_map, TaskL2InfoMap &l2_info_map) {
  l2_info_map.clear();
  ge::ComputeGraph::Vistor<ge::NodePtr> nodes = graph.GetDirectNode();
  for (const ge::NodePtr &node : nodes) {
    if (L2FusionComm::IsNonValidOp(node)) {
      continue;
    }

    FE_LOGD("Now we generate final L2 info for node: %s.", node.get()->GetName().c_str());
    TaskL2Info l2_info;
    l2_info.nodeName = node->GetName();
    OpL2AllocMap::const_iterator alloc_iterator = l2_alloc_map.find(node->GetName());
    int64_t node_id = node->GetOpDesc()->GetId();
    FE_LOGD("nodeId is %ld.", node_id);
    if (alloc_iterator != l2_alloc_map.end()) {
      l2_info.l2ctrl = alloc_iterator->second.l2ctrl;
      for (uint32_t i = 0; i < L2_CTRL_DATA_SIZE; ++i) {
        if (l2_info.l2ctrl.data[i].L2_data_section_size > 0) {
          FE_UINT64_ADDCHECK(l2_info.l2ctrl.data[i].L2_mirror_addr, mem_base);
          l2_info.l2ctrl.data[i].L2_mirror_addr += mem_base;
          // load l2 data to ddr if we need dump l2 data.(1 - need load out, 0 - no need)
          l2_info.l2ctrl.data[i].L2_load_to_ddr = 0;
        }
      }
      GenerateL2Data(mem_base, alloc_iterator->second.input, l2_info.input);
      GenerateL2Data(mem_base, alloc_iterator->second.output, l2_info.output);
    } else {
      FE_LOGD("node %s, id = %ld, does not have l2ctrl.", node->GetName().c_str(), node_id);
      L2FusionRtl2ctrl::InitRtl2ctrl(l2_info.l2ctrl);
      l2_info.input.clear();
      l2_info.output.clear();
    }
    l2_info_map.insert(TaskL2InfoPair(node->GetName(), l2_info));
  }
  return SUCCESS;
}

void L2FusionHandler::DisplayL2DataInfo(const string &title, const L2DataMap &input, rtL2Ctrl_t &l2ctrl,
                                        L2DataMap data) {
  L2BufferInfo l2;
  L2FusionComm::GetL2HardwareSet(l2);

  int64_t page_size = 0;
  if (l2.page_num == 0) {
    FE_LOGW("L2 page number is zero.");
  } else {
    page_size = l2.l2_buffer_size / l2.page_num;
  }
  for (auto it = data.cbegin(); it != data.cend(); ++it) {
    FE_LOGD("%s: data_in_l2_id = %u, if_input = %u, data_size = %7u, l2_page_n = %lu", title.c_str(),
            it->second.l2Index, static_cast<uint32_t>(input.count(it->first)),
            l2ctrl.data[it->second.l2Index].L2_data_section_size, it->second.l2PageNum);
    FE_LOGD("l2AddrO = %lu, ddr_addr_key = %lu, ddr_addr = %lu, offset = %2u, l2_load_to_ddr = %u.",
            it->second.l2Addr - (l2ctrl.data[it->second.l2Index].L2_page_offset_base) * page_size, it->first,
            l2ctrl.data[it->second.l2Index].L2_mirror_addr,
            static_cast<uint32_t>(l2ctrl.data[it->second.l2Index].L2_page_offset_base),
            static_cast<uint32_t>(l2ctrl.data[it->second.l2Index].L2_load_to_ddr));
    FE_LOGD("prev_offset = %2d.", static_cast<int32_t>(l2ctrl.data[it->second.l2Index].prev_L2_page_offset_base));
  }
}

void L2FusionHandler::DisplayL2Info(const TaskL2InfoMap &l2_info_map) {
  for (TaskL2InfoMap::const_iterator iter = l2_info_map.begin(); iter != l2_info_map.end(); ++iter) {
    const TaskL2Info &l2_info = iter->second;
    rtL2Ctrl_t l2ctrl = iter->second.l2ctrl;
    FE_LOGD("Node key is %s, node name = %s, l2ctrl.size = %lu.",
            iter->first.c_str(), l2_info.nodeName.c_str(), l2ctrl.size);
    FE_LOGD("inputNum = %lu, outputNum = %lu.", l2_info.input.size(), l2_info.output.size());
    DisplayL2DataInfo("input", l2_info.input, l2ctrl, l2_info.input);
    DisplayL2DataInfo("output", l2_info.input, l2ctrl, l2_info.output);
  }
}
}  // namespace fe
