/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter/tbe_adapter/kernel_launch/l2_cache_kernel_launch.h"
#include <memory>
#include <vector>
#include "common/fe_log.h"
#include "common/configuration.h"
#include "common/fe_type_utils.h"
#include "graph/utils/anchor_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/node_utils.h"

namespace fe {
namespace {
const std::string ATTR_NAME_L2CACHE_GRAPH_READ_MODE = "_fe_l2cache_graph_read_mode";
const std::set<std::string> LIFECYCLE_IS_END_OPS = {DATA, AIPPDATA, ANN_DATA,
                                                    CONSTANT, CONSTANTOP};
const std::set<std::string> LIFECYCLE_IS_NOT_END_OPS = {VARIABLE};
}
size_t L2CacheKernelLaunch::GetAppendArgsSizeOf() {
  return sizeof(uint64_t);  // uinit64_t: 8
}

size_t L2CacheKernelLaunch::GetAppendArgsNum() { return input_num_; }

Status L2CacheKernelLaunch::AddAppendArgs(const ge::Node &node, void *all_args_buff, const uint32_t &args_size) {
  auto op_desc_ptr = node.GetOpDesc();
  auto op_name = node.GetName();
  auto op_type = node.GetType();

  // 1. gerenate read mode
  vector<uint64_t> read_modes;
  if (GenerateReadModes(node, read_modes) != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][AddAppendArgs][Op %s,type %s] failed to generate the read mode.",
                    op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  if (read_modes.size() != GetAppendArgsNum()) {
    REPORT_FE_ERROR("[GenTask][AddAppendArgs] Node[%s, %s]: append_args_num %zu is not equal to read_modes_size %zu.",
                    op_name.c_str(), op_type.c_str(), GetAppendArgsNum(), read_modes.size());
    return FAILED;
  }

  // 2. add append args
  size_t each_append_arg_size = GetAppendArgsSizeOf();
  size_t left_append_arg_size = each_append_arg_size * GetAppendArgsNum();
  uint64_t cur_ptr = ge::PtrToValue(all_args_buff) + args_size;
  for (uint64_t &read_mode : read_modes) {
    errno_t ret = memcpy_s(reinterpret_cast<void *>(cur_ptr), left_append_arg_size,
                           reinterpret_cast<void *>(&read_mode), each_append_arg_size);
    if (ret != EOK) {
      return FAILED;
    }
    left_append_arg_size -= each_append_arg_size;
    cur_ptr += each_append_arg_size;
  }
  return SUCCESS;
}

Status L2CacheKernelLaunch::GenerateReadModes(const ge::Node &node, vector<uint64_t> &read_modes) const {
  auto op_desc_ptr = node.GetOpDesc();
  auto op_name = op_desc_ptr->GetName();
  auto op_type = op_desc_ptr->GetType();
  bool is_enable_reuse_mem = Configuration::Instance(AI_CORE_NAME).IsEnableReuseMemory();

  for (const auto &in_data_anchor : node.GetAllInDataAnchors()) {
    if (in_data_anchor == nullptr) {
      continue;
    }
    auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }

    auto idx = in_data_anchor->GetIdx();
    auto input_desc = op_desc_ptr->MutableInputDesc(idx);
    if (input_desc == nullptr) {
      continue;
    }
    // 1. get the src node of the input
    auto src_node = peer_out_anchor->GetOwnerNode();
    FE_CHECK_NOTNULL(src_node);
    auto read_mode = GenRmForSpecialInputOps(src_node, is_enable_reuse_mem);
    if (read_mode == L2CacheReadMode::RM_NONE) {
      // 2. get the life cycle of the input desc
      auto is_life_cycle_end = IsLifeCycleEnd(node, input_desc, idx);
      // 3. genarate rm by life cycle and read distance
      read_mode = GenerateReadMode(node, input_desc, idx, is_life_cycle_end);
    }
    // 4. set the attr
    (void)ge::AttrUtils::SetInt(input_desc, ATTR_NAME_L2CACHE_GRAPH_READ_MODE, static_cast<int64_t>(read_mode));

    read_modes.emplace_back(static_cast<uint64_t>(read_mode));
    FE_LOGD("Op[name=%s,type=%s,input=%d]: the graph read_mode=[%s].", op_name.c_str(), op_type.c_str(), idx,
            L2CacheReadMode2Str(read_mode).c_str());
  }

  return SUCCESS;
}

L2CacheReadMode L2CacheKernelLaunch::GenRmForSpecialInputOps(const ge::NodePtr &src_node,
                                                             bool is_enable_reuse_mem) const {
  auto src_node_type = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(src_node);
  // Const/Data
  if (LIFECYCLE_IS_END_OPS.count(src_node_type) != 0) {
    return is_enable_reuse_mem ? L2CacheReadMode::NOT_NEED_WRITEBACK : L2CacheReadMode::READ_LAST;
  }

  // Variable
  if (LIFECYCLE_IS_NOT_END_OPS.count(src_node_type) != 0) {
    return L2CacheReadMode::READ_LAST;
  }
  return L2CacheReadMode::RM_NONE;
}

bool L2CacheKernelLaunch::IsLifeCycleEnd(const ge::Node &node,
                                         const ge::GeTensorDescPtr &input_desc,
                                         int input_idx) const {
  auto op_desc = node.GetOpDesc();
  auto op_name = op_desc->GetName();
  auto op_type = op_desc->GetType();

  bool is_life_cycle_end = false;
  if (ge::AttrUtils::HasAttr(input_desc, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE)) {
    (void)ge::AttrUtils::GetBool(input_desc, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, is_life_cycle_end);
    FE_LOGD("Op[name=%s,type=%s,input=%d]: has attr %s, the life_cycle is %s.", op_name.c_str(), op_type.c_str(),
            input_idx, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE.c_str(), is_life_cycle_end ? "end" : "not end");
    return is_life_cycle_end;
  }
  return is_life_cycle_end;
}

L2CacheReadMode L2CacheKernelLaunch::GenerateReadMode(const ge::Node &node, const ge::GeTensorDescPtr &input_desc,
                                                      int input_idx, bool is_life_cycle_end) const {
  auto op_desc = node.GetOpDesc();
  auto op_name = op_desc->GetName();
  auto op_type = op_desc->GetType();

  // 1. no read distance on the input desc
  if (!ge::AttrUtils::HasAttr(input_desc, ge::ATTR_NAME_DATA_VISIT_DISTANCE)) {
    FE_LOGD("Op[name=%s,type=%s,input=%d]: no attr %s.", op_name.c_str(), op_type.c_str(), input_idx,
            ge::ATTR_NAME_DATA_VISIT_DISTANCE.c_str());
    return is_life_cycle_end ? L2CacheReadMode::NOT_NEED_WRITEBACK : L2CacheReadMode::READ_LAST;
  }

  // 2. there is the read distance on the input desc
  vector<int32_t> data_visit_dist_vec;
  (void)ge::AttrUtils::GetListInt(input_desc, ge::ATTR_NAME_DATA_VISIT_DISTANCE, data_visit_dist_vec);
  auto data_visit_dist_size = data_visit_dist_vec.size();

  // 3. life cycle is end
  if (is_life_cycle_end) {
    int32_t data_visit_dist_from_pre_node = 0;
    if (data_visit_dist_size == 0) {
      FE_LOGD(
          "Op[name=%s,type=%s,input=%d]: no read distance from previous node, set data_visit_dist_from_pre_node to be "
          "-1.",
          op_name.c_str(), op_type.c_str(), input_idx);
      data_visit_dist_from_pre_node = -1;
    } else {
      data_visit_dist_from_pre_node = data_visit_dist_vec[0];
    }

    FE_LOGD("Op[name=%s,type=%s,input=%d]: data_visit_dist_from_pre_node=[%d], data_visit_dist_threshold=[%d].",
            op_name.c_str(), op_type.c_str(), input_idx, data_visit_dist_from_pre_node, kDataVisitDistThreshold);
    return L2CacheReadMode::READ_INVALID;
  }

  // 4. life cycle is not end
  int32_t data_visit_dist_to_next_node = 0;
  if (data_visit_dist_size < 2) {
    FE_LOGW("Op[name=%s,type=%s,input=%d]: no read distance to next node, set data_visit_dist_to_next_node to be -1.",
            op_name.c_str(), op_type.c_str(), input_idx);
    data_visit_dist_to_next_node = -1;
  } else {
    data_visit_dist_to_next_node = data_visit_dist_vec[1];
  }
  FE_LOGD("Op[name=%s,type=%s,input=%d]: data_visit_dist_to_next_node=[%d], data_visit_dist_threshold=[%d].",
          op_name.c_str(), op_type.c_str(), input_idx, data_visit_dist_to_next_node, kDataVisitDistThreshold);
  return data_visit_dist_to_next_node <= kDataVisitDistThreshold ?
         L2CacheReadMode::READ_LAST : L2CacheReadMode::READ_INVALID;
}
}  // namespace fe
