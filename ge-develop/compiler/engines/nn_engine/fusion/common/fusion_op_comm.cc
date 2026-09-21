/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/fusion_op_comm.h"
#include <memory>
#include <set>
#include "common/configuration.h"
#include "common/fe_context_utils.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/sgt_slice_type.h"
#include "common/ffts_plus_type.h"
#include "common/aicore_util_constants.h"
#include "common/aicore_util_attr_define.h"
#include "common/util/constants.h"
#include "common/util/op_info_util.h"
#include "common/op_info_common.h"
#include "common/fusion_statistic/fusion_statistic_writer.h"
#include "google/protobuf/text_format.h"
#include "framework/common/ge_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/detail/attributes_holder.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph/model_serialize.h"
#include "graph/op_kernel_bin.h"
#include "graph/tuning_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/type_utils.h"
#include "proto/ge_ir.pb.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_constant.h"
#include "register/graph_optimizer/fusion_common/graph_pass_util.h"
#include "ops_store/ops_kernel_manager.h"
#include "ops_store/ops_kernel_common.h"
using ge::AnchorPtr;
using ge::AttrUtils;
using ge::ComputeGraph;
using ge::DataAnchor;
using ge::GRAPH_FAILED;
using ge::GRAPH_SUCCESS;
using ge::graphStatus;
using ge::GraphUtils;
using ge::NodePtr;
using ge::OpDesc;

namespace fe {
const string ORIGINAL_NODES = "original_nodes";
const std::string  MIX_AIC_PREFIX = "_mix_aic";
const std::string  MIX_AIV_PREFIX = "_mix_aiv";
const std::string kStageSetFusionOp = "[SubGraphOpt][MergeFusionNode][SetFusionOp]";
inline int64_t GetImplyTypeToFusionOp(vector<ge::NodePtr> &fus_nodelist) {
  int64_t imply_type = -1;
  for (ge::NodePtr node : fus_nodelist) {
    if (AttrUtils::GetInt(node->GetOpDesc(), FE_IMPLY_TYPE, imply_type)) {
      return imply_type;
    }
  }
  return imply_type;
}

namespace {
inline void SetGroupId(const ge::OpDescPtr &fus_opdef, const uint32_t &parallel_group_id) {
  uint32_t tmp_parallel_group_id = 0;
  if (!ge::AttrUtils::GetInt(fus_opdef, ge::ATTR_NAME_PARALLEL_GROUP_ID, tmp_parallel_group_id)) {
      (void)ge::AttrUtils::SetInt(fus_opdef, ge::ATTR_NAME_PARALLEL_GROUP_ID, parallel_group_id);
  }
}

inline void GetGroupId(const ge::NodePtr &node, uint32_t &parallel_group_id) {
  if (ge::AttrUtils::GetInt(node->GetOpDesc(), ge::ATTR_NAME_PARALLEL_GROUP_ID, parallel_group_id)) {
    FE_LOGD("Node[%s, %s] has _parallel_group_id which is %u.", node->GetName().c_str(),
        node->GetType().c_str(), parallel_group_id);
  } else {
    FE_LOGD("Node[%s, %s] without parallel_group_id attr.", node->GetName().c_str(),
        node->GetType().c_str());
  }
}

void DfsFindOuterInput(const int64_t &indx, const ge::NodePtr &node, const std::set<ge::NodePtr> &fus_node_set,
                       vector<ge::ConstGeTensorDescPtr> &inplace_tensor,
                       vector<vector<ge::ConstGeTensorDescPtr>> &output_place);

void DfsFindSingleOpOuterInput(const ge::InDataAnchorPtr &anchor, const std::set<ge::NodePtr> &fus_node_set,
                               const ge::NodePtr &node, vector<ge::ConstGeTensorDescPtr> &inplace_tensor,
                               vector<vector<ge::ConstGeTensorDescPtr>> &output_place) {
  // Get peer out anchor
  auto peer_anchor = anchor->GetPeerOutAnchor();
  if (peer_anchor == nullptr) {
    FE_LOGD("The anchor has no peer. node: %s", node->GetName().c_str());
    return;
  }
  // Judge peer out node is outer input
  ge::NodePtr src_node = peer_anchor->GetOwnerNode();
  uint32_t src_idx = peer_anchor->GetIdx();
  if (fus_node_set.count(src_node) == 0) {
    FE_LOGD("Found outer input node [%s].", src_node->GetName().c_str());
    ge::ConstGeTensorDescPtr src_tenor_ptr = node->GetOpDesc()->GetInputDescPtr(anchor->GetIdx());
    inplace_tensor.push_back(src_tenor_ptr);
    output_place.push_back(inplace_tensor);
    inplace_tensor.pop_back();
    return;
  }
  // Continue to find outer input
  vector<vector<int64_t>> idx_list;
  ge::AttrUtils::GetListListInt(src_node->GetOpDesc(), kAttrOutputInplaceAbility, idx_list);
  if (idx_list.size() == 0) {
    FE_LOGD("The src node: %s without %s", src_node->GetName().c_str(), kAttrOutputInplaceAbility);
    return;
  }
  ge::OpDescPtr op_desc_ptr = src_node->GetOpDesc();
  PrintOutputInplace(op_desc_ptr, idx_list);
  for (auto idx : idx_list) {
    if (static_cast<uint32_t>(idx[0]) == src_idx) {
      DfsFindOuterInput(idx[1], src_node, fus_node_set, inplace_tensor, output_place);
    }
  }
}

void DfsFindOuterInput(const int64_t &indx, const ge::NodePtr &node, const std::set<ge::NodePtr> &fus_node_set,
                       vector<ge::ConstGeTensorDescPtr> &inplace_tensor,
                       vector<vector<ge::ConstGeTensorDescPtr>> &output_place) {
  FE_CHECK(node->GetInDataAnchor(indx) == nullptr, FE_LOGW("Node [%s] without input index[%lld]",
          node->GetName().c_str(), indx), return);
  auto peer_out_anchor = node->GetInDataAnchor(indx)->GetPeerOutAnchor();
  FE_CHECK(peer_out_anchor == nullptr, FE_LOGW("Node [%s] anchor[%lld] without peerOut anchor",
          node->GetName().c_str(), indx), return);
  auto src_node = peer_out_anchor->GetOwnerNode();
  uint32_t src_idx = peer_out_anchor->GetIdx();
  // First Finded the outer input
  if (fus_node_set.count(src_node) == 0) {
    FE_LOGD("Found outer input node [%s].", src_node->GetName().c_str());
    ge::ConstGeTensorDescPtr src_tenor_ptr = node->GetOpDesc()->GetInputDescPtr(indx);
    inplace_tensor.push_back(src_tenor_ptr);
    output_place.push_back(inplace_tensor);
    inplace_tensor.pop_back();
    return;
  }
  // judge tbe src_node with outputInplace
  vector<vector<int64_t>> inplace;
  ge::AttrUtils::GetListListInt(src_node->GetOpDesc(), kAttrOutputInplaceAbility, inplace);
  FE_CHECK(inplace.size() == 0, FE_LOGD("The src node: %s without %s.",
           src_node->GetName().c_str(), kAttrOutputInplaceAbility), return);
  ge::OpDescPtr op_desc_ptr = src_node->GetOpDesc();
  PrintOutputInplace(op_desc_ptr, inplace);
  // Traverse src node inplace Info
  for (auto indx_list : inplace) {
    if (static_cast<uint32_t>(indx_list[0]) == src_idx) {
      auto inAnchor = src_node->GetInDataAnchor(indx_list[1]);
      // anchor
      if (inAnchor == nullptr) {
        FE_LOGW("Node[%s] inAnchor is nullptr, input indx[%lld] is invalid.", src_node->GetNamePtr(), indx_list[1]);
        continue;
      }
      DfsFindSingleOpOuterInput(inAnchor, fus_node_set, src_node, inplace_tensor, output_place);
    }
  }
}

void FindFusionNodesOuterOutput(const ge::OutDataAnchorPtr &anchor, const ge::NodePtr &node,
                                       const std::set<ge::NodePtr> &fus_node_set,
                                       vector<vector<ge::ConstGeTensorDescPtr>> &output_place) {
  vector<vector<int64_t>> inplace;
  ge::AttrUtils::GetListListInt(node->GetOpDesc(), kAttrOutputInplaceAbility, inplace);
  // To judge the current node with output inplace ability
  if (inplace.size() == 0) {
    return;
  }
  auto peer_in_anchors = anchor->GetPeerInDataAnchors();
  for (auto peer_in_anchor : peer_in_anchors) {
    ge::NodePtr peer_in_node = peer_in_anchor->GetOwnerNode();
    // Finded  outer output
    if (fus_node_set.count(peer_in_node) == 0) {
      // Dfx to print the iplace info
      ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
      PrintOutputInplace(op_desc_ptr, inplace);
      uint32_t idx = anchor->GetIdx();
      // to traverse all inplace info of current node
      for (auto indx_list : inplace) {
        // current output anchor satisfies tbe outputInplace output
        if (static_cast<uint32_t>(indx_list[0]) == idx) {
          // Save the outer ouput tensorptr, for fusionOp index map
          ge::ConstGeTensorDescPtr dst_tenor_ptr = node->GetOpDesc()->GetOutputDescPtr(idx);
          vector<ge::ConstGeTensorDescPtr> inplace_tensor = {dst_tenor_ptr};
          // To find outer input from curren node
          DfsFindOuterInput(indx_list[1], node, fus_node_set, inplace_tensor, output_place);
        }
      }
      break;
    }
  }
}

inline void SetGroupIdToFusionOp(const vector<ge::NodePtr> &fus_nodelist, const ge::OpDescPtr &fus_opdef,
                                 const string &engine_name) {
  (void)engine_name;
  uint32_t parallel_group_id = DefaultGroupID;
  uint32_t general_parall_group_id = DefaultGroupID;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (IsHeavyOp(node)) {
      GetGroupId(node, parallel_group_id);
    } else {
      GetGroupId(node, general_parall_group_id);
    }
  }

  if (parallel_group_id != static_cast<uint32_t>(DefaultGroupID)) {
    SetGroupId(fus_opdef, parallel_group_id);
  } else if (general_parall_group_id != static_cast<uint32_t>(DefaultGroupID)) {
    SetGroupId(fus_opdef, general_parall_group_id);
  }
}

inline void SetListStrAttrToFusionOp(const vector<ge::NodePtr> &fus_nodelist, const ge::OpDescPtr &fus_opdef,
                                     const string &attr_name) {
  vector<string> str_val;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (ge::AttrUtils::GetListStr(node->GetOpDesc(), attr_name, str_val)) {
      (void)ge::AttrUtils::SetListStr(fus_opdef, attr_name, str_val);
      break;
    }
  }
}

inline void SetStrAttrToFusionOp(const vector<ge::NodePtr> &fus_nodelist, const ge::OpDescPtr &fus_opdef,
                                 const string &attr_name) {
  string str_val;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (ge::AttrUtils::GetStr(node->GetOpDesc(), attr_name, str_val)) {
      (void)ge::AttrUtils::SetStr(fus_opdef, attr_name, str_val);
      break;
    }
  }
}

inline void SetStrVecAttrToFusionOp(const vector<ge::NodePtr> &fus_nodelist, const ge::OpDescPtr &fus_opdef,
                                    const string &attr_name) {
  vector<string> str_vec_val;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (ge::AttrUtils::GetListStr(node->GetOpDesc(), attr_name, str_vec_val)) {
      (void)ge::AttrUtils::SetListStr(fus_opdef, attr_name, str_vec_val);
      break;
    }
  }
}

inline void SetNewStrVecAttrToFusionOp(const vector<ge::NodePtr> &fus_nodelist, const ge::OpDescPtr &fus_opdef,
                                       const string &attr_name) {
  vector<string> str_vec_val;
  for (const ge::NodePtr &node : fus_nodelist) {
    string node_str;
    if (ge::AttrUtils::GetStr(node->GetOpDesc(), attr_name, node_str)) {
      FE_LOGD("Get fusion op %s attr %s from node %s, value is %s.", fus_opdef->GetName().c_str(),
              attr_name.c_str(), node->GetOpDesc()->GetName().c_str(), node_str.c_str());
      str_vec_val.emplace_back(node_str);
    }
  }
  (void)ge::AttrUtils::SetListStr(fus_opdef, attr_name, str_vec_val);
  FE_LOGD("Set fusion op %s attr %s with vector size %zu.", fus_opdef->GetName().c_str(),
          attr_name.c_str(), str_vec_val.size());
}

inline void SetListListInt64AttrToFusionOp(const vector<ge::NodePtr> &fus_nodelist,
                                           const ge::OpDescPtr &fus_opdef,
                                           const string &attr_name) {
  vector<vector<int64_t>> int_val;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (ge::AttrUtils::GetListListInt(node->GetOpDesc(), attr_name, int_val)) {
      (void)ge::AttrUtils::SetListListInt(fus_opdef, attr_name, int_val);
      break;
    }
  }
}

inline void SetListInt64AttrToFusionOp(const vector<ge::NodePtr> &fus_nodelist,
                                       const ge::OpDescPtr &fus_opdef,
                                       const string &attr_name) {
  vector<int64_t> int_val;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (ge::AttrUtils::GetListInt(node->GetOpDesc(), attr_name, int_val)) {
      (void)ge::AttrUtils::SetListInt(fus_opdef, attr_name, int_val);
      break;
    }
  }
}

inline void SetListInt32AttrToFusionOp(const vector<ge::NodePtr> &fus_nodelist,
                                       const ge::OpDescPtr &fus_opdef,
                                       const string &attr_name) {
  vector<int32_t> int_val;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (ge::AttrUtils::GetListInt(node->GetOpDesc(), attr_name, int_val)) {
      (void)ge::AttrUtils::SetListInt(fus_opdef, attr_name, int_val);
      break;
    }
  }
}

inline void SetListFloatAttrToFusionOp(const vector<ge::NodePtr> &fus_nodelist,
                                       const ge::OpDescPtr &fus_opdef,
                                       const string &attr_name) {
  vector<float> int_val;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (ge::AttrUtils::GetListFloat(node->GetOpDesc(), attr_name, int_val)) {
      (void)ge::AttrUtils::SetListFloat(fus_opdef, attr_name, int_val);
      break;
    }
  }
}

inline void SetInt64AttrToFusionOp(const vector<ge::NodePtr> &fus_nodelist, const ge::OpDescPtr &fus_opdef,
                                   const string &attr_name) {
  int64_t int_val;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (ge::AttrUtils::GetInt(node->GetOpDesc(), attr_name, int_val)) {
      (void)ge::AttrUtils::SetInt(fus_opdef, attr_name, int_val);
      break;
    }
  }
}

inline void SetListBytesAttrToFusionOp(const vector<ge::NodePtr> &fus_nodelist,
                                       const ge::OpDescPtr &fus_opdef,
                                       const string &attr_name) {
  vector<ge::Buffer> byte_val;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (ge::AttrUtils::GetListBytes(node->GetOpDesc(), attr_name, byte_val)) {
      (void)ge::AttrUtils::SetListBytes(fus_opdef, attr_name, byte_val);
      break;
    }
  }
}

inline void SetBytesAttrToFusionOp(const vector<ge::NodePtr> &fus_nodelist,
                                   const ge::OpDescPtr &fus_opdef,
                                   const string &attr_name) {
  ge::Buffer int_val;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (ge::AttrUtils::GetBytes(node->GetOpDesc(), attr_name, int_val)) {
      (void)ge::AttrUtils::SetBytes(fus_opdef, attr_name, int_val);
      FE_LOGD("Tbe tbe_kernel_buffer_ size:%lu.", int_val.GetSize());
      break;
    }
  }
}

inline void SetListBoolAttrToFusionOp(const vector<ge::NodePtr> &fus_nodelist,
                                      const ge::OpDescPtr &fus_opdef,
                                      const string &attr_name) {
  vector<bool> bool_val;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (ge::AttrUtils::GetListBool(node->GetOpDesc(), attr_name, bool_val)) {
      (void)ge::AttrUtils::SetListBool(fus_opdef, attr_name, bool_val);
      break;
    }
  }
}

inline void SetBoolAttrToFusionOp(const vector<ge::NodePtr> &fus_nodelist,
                                  const ge::OpDescPtr &fus_opdef,
                                  const string &attr_name) {
  bool bool_val = false;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (ge::AttrUtils::GetBool(node->GetOpDesc(), attr_name, bool_val)) {
      (void)ge::AttrUtils::SetBool(fus_opdef, attr_name, bool_val);
      break;
    }
  }
}

inline void SetStrPathToFusionOp(const vector<ge::NodePtr> &fus_nodelist,
                                 const ge::OpDescPtr &fus_opdef,
                                 const string &path_name) {
    string path_str;
    for (const ge::NodePtr &node : fus_nodelist) {
        if (node->GetOpDesc() == nullptr) {
          break;
        }
        path_str = node->GetOpDesc()->TryGetExtAttr(path_name, string(""));
        if (path_str.empty()) {
            fus_opdef->SetExtAttr(path_name, path_str);
            break;
        }
    }
}

inline void SetMemAttrToFusionOp(const vector<ge::NodePtr> &fus_nodelist,
                                 const ge::OpDescPtr &fus_opdef, const string &attr_name) {
  bool bool_val = false;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (ge::AttrUtils::GetBool(node->GetOpDesc(), attr_name, bool_val) && bool_val) {
      (void)ge::AttrUtils::SetBool(fus_opdef, attr_name, bool_val);
      break;
    }
  }
}

inline void SetKernelNameToFusionOp(const vector<ge::NodePtr> &fus_nodelist,
                                    const ge::OpDescPtr &fus_opdef) {
  string kernel_name;
  for (const ge::NodePtr &node : fus_nodelist) {
    bool is_mix = false;
    if (ge::AttrUtils::GetStr(node->GetOpDesc(), MIX_AIC_PREFIX + kKernelName, kernel_name)) {
      (void)ge::AttrUtils::SetStr(fus_opdef, MIX_AIC_PREFIX + fus_opdef->GetName() + kKernelName, kernel_name);
      (void)ge::AttrUtils::SetStr(fus_opdef, MIX_AIC_PREFIX + kKernelName, kernel_name);
      is_mix = true;
    }
    if (ge::AttrUtils::GetStr(node->GetOpDesc(), MIX_AIV_PREFIX + kKernelName, kernel_name)) {
      (void)ge::AttrUtils::SetStr(fus_opdef, MIX_AIV_PREFIX + fus_opdef->GetName() + kKernelName, kernel_name);
      (void)ge::AttrUtils::SetStr(fus_opdef, MIX_AIV_PREFIX + kKernelName, kernel_name);
      is_mix = true;
    }
    if (ge::AttrUtils::GetStr(node->GetOpDesc(), kTbeMixEnhancedPrefix + kKernelName, kernel_name)) {
      (void)ge::AttrUtils::SetStr(fus_opdef, kTbeMixEnhancedPrefix + fus_opdef->GetName() + kKernelName, kernel_name);
      (void)ge::AttrUtils::SetStr(fus_opdef, kTbeMixEnhancedPrefix + kKernelName, kernel_name);
      is_mix = true;
    }
    if (is_mix) {
      break;
    }
    if (ge::AttrUtils::GetStr(node->GetOpDesc(), kKernelName, kernel_name)) {
      (void)ge::AttrUtils::SetStr(fus_opdef, fus_opdef->GetName() + kKernelName, kernel_name);
      (void)ge::AttrUtils::SetStr(fus_opdef, kKernelName, kernel_name);
      break;
    }
  }
}

inline void SetMultiKernelThreadKernelNameToFusionOp(const vector<ge::NodePtr> &fus_nodelist,
                                                     const ge::OpDescPtr &fus_opdef) {
  vector<string> kernel_names;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (ge::AttrUtils::GetListStr(node->GetOpDesc(), kThreadKernelName, kernel_names)) {
      (void)ge::AttrUtils::SetListStr(fus_opdef, kThreadKernelName, kernel_names);
      break;
    }
  }
}

inline void SetL1FusionSubGraphNoToFusionOp(const vector<ge::NodePtr> &fus_nodelist,
                                            const ge::OpDescPtr &fus_opdef) {
  string attr_name_l1_fusion_sub_graph_no = "_L1_fusion_sub_graph_no";
  string str_value;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (ge::AttrUtils::GetStr(node->GetOpDesc(), attr_name_l1_fusion_sub_graph_no, str_value)) {
      (void)ge::AttrUtils::SetStr(fus_opdef, attr_name_l1_fusion_sub_graph_no, str_value);
      break;
    }
  }
}

inline void SetNoTaskAndDumpNeeded(const vector<ge::NodePtr> &fus_nodelist, const ge::OpDescPtr &fus_opdef) {
  bool str_value = false;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NO_TASK_AND_DUMP_NEEDED, str_value)) {
      (void)ge::AttrUtils::SetBool(fus_opdef, ge::ATTR_NO_TASK_AND_DUMP_NEEDED, str_value);
      break;
    }
  }
}

inline void SetSqrtmodeToFusionOp(const vector<ge::NodePtr> &fus_nodelist, const ge::OpDescPtr &fus_opdef) {
  bool sqrt_mode = false;
  string attr_name = "sqrt_mode";
  string attr_name_quant = "quant_sqrt_mode";
  string attr_name_dequant = "dequant_sqrt_mode";
  string quant_op = "AscendQuant";
  string dequant_op = "AscendDequant";

  for (const ge::NodePtr &node : fus_nodelist) {
    if (ge::AttrUtils::GetBool(node->GetOpDesc(), attr_name, sqrt_mode)) {
      if (node->GetType() == quant_op || node->GetType() == dequant_op) {
        string attr_name_tmp = node->GetType() == quant_op ? attr_name_quant : attr_name_dequant;
        (void)ge::AttrUtils::SetBool(fus_opdef, attr_name_tmp, sqrt_mode);
      }
    }
  }

  return;
}

inline void SetBoolAttrToFusionOp(const ge::NodePtr &node_ptr, const ge::OpDescPtr &fus_opdef,
                                  const string &attr_name) {
  bool bool_val = false;
  if (ge::AttrUtils::GetBool(node_ptr->GetOpDesc(), attr_name, bool_val)) {
    (void)ge::AttrUtils::SetBool(fus_opdef, attr_name, bool_val);
  }
}

inline void SetStrAttrToFusionOp(const ge::NodePtr &node_ptr, const ge::OpDescPtr &fus_opdef,
                                 const string &attr_name) {
  string str_val;
  if (ge::AttrUtils::GetStr(node_ptr->GetOpDesc(), attr_name, str_val)) {
    (void)ge::AttrUtils::SetStr(fus_opdef, attr_name, str_val);
  }
}

inline void SetL1InfoToFusionOp(const vector<ge::NodePtr> &fus_nodelist, const ge::OpDescPtr &fus_opdef) {
  if (fus_nodelist.empty()) {
    FE_LOGD("fusNodelist is empty");
    return;
  }
  ge::NodePtr node = fus_nodelist[0];
  FE_CHECK(node == nullptr, FE_LOGD("node is nullptr"), return);

  int64_t groupid;
  if (ge::AttrUtils::GetInt(node->GetOpDesc(), ge::ATTR_NAME_L1_FUSION_GROUP_ID, groupid)) {
    (void)ge::AttrUtils::SetInt(fus_opdef, ge::ATTR_NAME_L1_FUSION_GROUP_ID, groupid);
  }
  SetBoolAttrToFusionOp(node, fus_opdef, NEED_RE_PRECOMPILE);
  SetStrAttrToFusionOp(node, fus_opdef, ge::ATTR_NAME_FUSION_GROUP_TYPE);
  SetStrAttrToFusionOp(node, fus_opdef, ge::ATTR_NAME_CONTINUOUS_STREAM_LABEL);

  std::vector<int32_t> tvm_workspace_types;
  if (ge::AttrUtils::GetListInt(node->GetOpDesc(), ge::TVM_ATTR_NAME_WORKSPACE_TYPE, tvm_workspace_types)) {
    (void)ge::AttrUtils::SetListInt(fus_opdef, ge::TVM_ATTR_NAME_WORKSPACE_TYPE, tvm_workspace_types);
  }

  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, kLxNoReuseMemFlag);
  int32_t output_real_calc_flag = 0;
  for (const ge::NodePtr &node_to_fuse : fus_nodelist) {
    if (ge::AttrUtils::GetInt(node_to_fuse->GetOpDesc(), ge::ATTR_NAME_GET_TENSOR_ACTUAL_SIZE, output_real_calc_flag)) {
      (void)ge::AttrUtils::SetInt(fus_opdef, ge::ATTR_NAME_GET_TENSOR_ACTUAL_SIZE, output_real_calc_flag);
      FE_LOGI("L1Fusion -- AddFusionConcatEdge : [%d]", output_real_calc_flag);
      break;
    }
  }
  return;
}

inline void SetL2InfoToFusionOp(const vector<ge::NodePtr> &fus_nodelist, const ge::OpDescPtr &fus_opdef) {
  if (fus_nodelist.empty()) {
    FE_LOGD("fusNodelist is empty");
    return;
  }
  ge::NodePtr node = fus_nodelist[0];
  FE_CHECK(node == nullptr, FE_LOGD("node is nullptr."), return);

  // ATTR_NAME_L2_FUSION_GROUP_ID,
  // L2_OPTIMIZED,ATTR_NAME_FUSION_GROUP_TYPE,ATTR_NAME_CONTINUOUS_STREAM_LABEL
  int64_t groupid;
  string continuous_stream_label;
  if (ge::AttrUtils::GetInt(node->GetOpDesc(), ge::ATTR_NAME_L2_FUSION_GROUP_ID, groupid)) {
    (void)ge::AttrUtils::SetInt(fus_opdef, ge::ATTR_NAME_L2_FUSION_GROUP_ID, groupid);
  }

  SetBoolAttrToFusionOp(node, fus_opdef, NEED_RE_PRECOMPILE);
  SetStrAttrToFusionOp(node, fus_opdef, ge::ATTR_NAME_FUSION_GROUP_TYPE);
  SetStrAttrToFusionOp(node, fus_opdef, ge::ATTR_NAME_CONTINUOUS_STREAM_LABEL);

  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, kLxNoReuseMemFlag);
  int32_t output_real_calc_flag = 0;
  for (const ge::NodePtr &node_tmp : fus_nodelist) {
    if (ge::AttrUtils::GetInt(node_tmp->GetOpDesc(), ge::ATTR_NAME_GET_TENSOR_ACTUAL_SIZE, output_real_calc_flag)) {
      (void)ge::AttrUtils::SetInt(fus_opdef, ge::ATTR_NAME_GET_TENSOR_ACTUAL_SIZE, output_real_calc_flag);
      break;
    }
  }
  return;
}

inline void RefreshFusionOpType(ge::OpDescPtr &fus_opdef, const ge::NodePtr &first_node) {
  if (first_node == nullptr || first_node->GetOpDesc() == nullptr) {
    return;
  }
  std::string fusion_op_type;
  (void)ge::AttrUtils::GetStr(first_node->GetOpDesc(), UB_FUSION_OP_TYPE, fusion_op_type);
  if (!fusion_op_type.empty()) {
    ge::OpDescUtilsEx::SetType(fus_opdef, fusion_op_type);
  }
}

inline void SetSgtCoreTypeToFusionOp(const vector<ge::NodePtr> &fus_nodelist, const ge::OpDescPtr &fus_opdef) {
  string core_type;
  for (const ge::NodePtr &node : fus_nodelist) {
    string cur_type;
    if (ge::AttrUtils::GetStr(node->GetOpDesc(), kSgtCubeVectorCoreType, cur_type)) {
      if (!cur_type.empty()) {
        core_type = cur_type;
      }
      if (core_type == AI_CORE_TYPE) {
        break;
      }
    }
  }
  (void)ge::AttrUtils::SetStr(fus_opdef, kSgtCubeVectorCoreType, core_type);
}

Status SetTbeKernelBin(const ge::OpDescPtr &fus_opdef, const ge::NodePtr node, const string &attr_name,
                       const string &key_name)
{
  FE_LOGD("Set kernel bin of node:%s, attr:%s, key_name:%s.", node->GetName().c_str(), attr_name.c_str(),
          key_name.c_str());
  ge::OpKernelBinPtr tbe_kernel_ptr_old =
      node->GetOpDesc()->TryGetExtAttr(attr_name, ge::OpKernelBinPtr());
  FE_CHECK(tbe_kernel_ptr_old == nullptr,
           REPORT_FE_ERROR("[MergeFusionNode][SetTbeKernelBin] Op[%s]: the tbe_kernel_ptr_old is nullptr.",
                           node->GetName().c_str()), return FAILED);
  auto buffer_size = tbe_kernel_ptr_old->GetBinDataSize();
  std::vector<char> buffer_vec(tbe_kernel_ptr_old->GetBinData(), tbe_kernel_ptr_old->GetBinData() + buffer_size);
  ge::OpKernelBinPtr tbe_kernel_ptr_fused = nullptr;
  FE_MAKE_SHARED(tbe_kernel_ptr_fused =
                     std::make_shared<ge::OpKernelBin>(key_name, std::move(buffer_vec)), return FAILED);
  if (tbe_kernel_ptr_fused) {
    fus_opdef->SetExtAttr(attr_name, tbe_kernel_ptr_fused);
  }
  return SUCCESS;
}

Status SetNormalFusionOpAttr(const vector<ge::NodePtr> &fus_nodelist, ge::OpDescPtr &fus_opdef,
                             const ge::NodePtr first_node)
{
  FE_LOGD("Set normal fusion node attribute.");
  std::string kernel_name;
  ge::AttrUtils::GetStr(first_node->GetOpDesc(), kKernelName, kernel_name);
  if (SetTbeKernelBin(fus_opdef, first_node, ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_name) != SUCCESS) {
    return FAILED;
  }
  (void)ge::AttrUtils::SetStr(fus_opdef, ge::ATTR_NAME_TBE_KERNEL_NAME, kernel_name);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, ATTR_NAME_TBE_KERNEL_SIZE);
  SetBytesAttrToFusionOp(fus_nodelist, fus_opdef, ge::ATTR_NAME_TBE_KERNEL_BUFFER);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, ge::TVM_ATTR_NAME_METADATA);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, ATTR_NAME_KERNEL_LIST_FIRST_NAME);
  std::string core_type;
  (void)ge::AttrUtils::GetStr(fus_opdef, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
  FE_LOGD("Fused node: %s, core type: %s", fus_opdef->GetName().c_str(), core_type.c_str());
  if ((core_type == "MIX")) {
    SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, kTaskRadio);
  }
  return SUCCESS;
}

Status SetMixFusionOpAttr(const vector<ge::NodePtr> &fus_nodelist, ge::OpDescPtr &fus_opdef,
                          const ge::NodePtr first_node)
{
  FE_LOGD("Set mix fusion node attribute.");
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, MIX_AIC_PREFIX + ATTR_NAME_TBE_KERNEL_SIZE);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, MIX_AIV_PREFIX + ATTR_NAME_TBE_KERNEL_SIZE);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, kTbeMixEnhancedPrefix + ATTR_NAME_TBE_KERNEL_SIZE);
  SetBytesAttrToFusionOp(fus_nodelist, fus_opdef, MIX_AIC_PREFIX + ge::ATTR_NAME_TBE_KERNEL_BUFFER);
  SetBytesAttrToFusionOp(fus_nodelist, fus_opdef, MIX_AIV_PREFIX + ge::ATTR_NAME_TBE_KERNEL_BUFFER);
  SetBytesAttrToFusionOp(fus_nodelist, fus_opdef, kTbeMixEnhancedPrefix + ge::ATTR_NAME_TBE_KERNEL_BUFFER);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, MIX_AIC_PREFIX + ge::TVM_ATTR_NAME_METADATA);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, MIX_AIV_PREFIX + ge::TVM_ATTR_NAME_METADATA);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, kTbeMixEnhancedPrefix + ge::TVM_ATTR_NAME_METADATA);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, MIX_AIC_PREFIX + ge::ATTR_NAME_TBE_KERNEL_NAME);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, MIX_AIV_PREFIX + ge::ATTR_NAME_TBE_KERNEL_NAME);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, kTbeMixEnhancedPrefix + ge::ATTR_NAME_TBE_KERNEL_NAME);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, MIX_AIC_PREFIX + ATTR_NAME_KERNEL_LIST_FIRST_NAME);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, MIX_AIV_PREFIX + ATTR_NAME_KERNEL_LIST_FIRST_NAME);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, kTbeMixEnhancedPrefix + ATTR_NAME_KERNEL_LIST_FIRST_NAME);
  SetListStrAttrToFusionOp(fus_nodelist, fus_opdef, kKernelNamesPrefix);
  std::string kernel_name;
  if (ge::AttrUtils::GetStr(first_node->GetOpDesc(), MIX_AIC_PREFIX + kKernelName, kernel_name)) {
    if (SetTbeKernelBin(fus_opdef, first_node, MIX_AIC_PREFIX + ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_name) !=
        SUCCESS) {
      return FAILED;
    }
  }
  if (ge::AttrUtils::GetStr(first_node->GetOpDesc(), MIX_AIV_PREFIX + kKernelName, kernel_name)) {
    if (SetTbeKernelBin(fus_opdef, first_node, MIX_AIV_PREFIX + ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_name) !=
        SUCCESS) {
      return FAILED;
    }
  }
  if (ge::AttrUtils::GetStr(first_node->GetOpDesc(), kTbeMixEnhancedPrefix + kKernelName, kernel_name)) {
    if (SetTbeKernelBin(fus_opdef, first_node, kTbeMixEnhancedPrefix + ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_name) !=
        SUCCESS) {
      return FAILED;
    }
  }
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, kTaskRadio);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, kModeInArgsFirstField);
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, kAttrIntercoreSync);
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, kMixEnhancedKernel);
  std::string mix_l2;
  if (ge::AttrUtils::GetStr(first_node->GetOpDesc(), ATTR_NAME_ALIAS_ENGINE_NAME, mix_l2)) {
    (void)ge::AttrUtils::SetStr(fus_opdef, ATTR_NAME_ALIAS_ENGINE_NAME, mix_l2);
  }
  return SUCCESS;
}
} // namespace

ge::OpDescPtr FusionOpComm::CreateFusionOp(vector<ge::NodePtr> &fus_nodelist, const string &engine_name) {
  if (fus_nodelist.empty()) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][CrtFusOp] fusNodelist is empty.");
    return nullptr;
  }

  ge::NodePtr first_node = fus_nodelist[0];
  FE_CHECK(first_node == nullptr,
           REPORT_FE_ERROR("[SubGraphOpt][PostProcess][CrtFusOp] CreateFusionOp returned nullptr pointer."), return nullptr);
  ge::OpDescPtr fus_opdef = std::shared_ptr<OpDesc>(new (std::nothrow) OpDesc());
  FE_CHECK(fus_opdef == nullptr,
           REPORT_FE_ERROR("[SubGraphOpt][PostProcess][CrtFusOp] CreateFusionOp returned nullptr pointer."), return nullptr);

  std::string fusion_node_name;
  fusion_node_name.clear();

  for (ge::NodePtr &node : fus_nodelist) {
    FE_CHECK(node == nullptr, REPORT_FE_ERROR("[SubGraphOpt][PostProcess][CrtFusOp] pScopeAllocator is nullptr."),
             return nullptr);
    fusion_node_name += node->GetOpDesc()->GetName();
    if (fusion_node_name.size() > kMaxOpNmaLen) {
      fusion_node_name = first_node->GetOpDesc()->GetName() + "_ub_fusion";
      break;
    }
  }
  fus_opdef->SetName(fusion_node_name);

  ge::OpDescPtr first_op_desc = first_node->GetOpDesc();
  if (CopyFusionScopeAttr(first_op_desc, fus_opdef) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][CrtFusOp] Failed to set fusion scope attr for Op[%s, optype[%s]].",
                    fus_opdef->GetName().c_str(), fus_opdef->GetType().c_str());
    return nullptr;
  }

  // copy pass_name
  string pass_name;
  if (ge::AttrUtils::GetStr(first_op_desc, kPassNameUbAttr, pass_name)) {
    bool set_failed = (GraphPassUtil::StoreAndUpdataOriginFusionPassName(fus_opdef, fus_nodelist, pass_name) !=
        SUCCESS) || (!ge::AttrUtils::SetStr(fus_opdef, kPassNameUbAttr, pass_name));
    if (set_failed) {
      REPORT_FE_ERROR("[SubGraphOpt][PostProcess][CrtFusOp] Op[%s, optype[%s]]: Failed to set the attribute %s.",
                      first_op_desc->GetName().c_str(), first_op_desc->GetType().c_str(), kPassNameUbAttr.c_str());
      return nullptr;
    }
  }

  // copy session graph id
  string session_graph_id;
  if (ge::AttrUtils::GetStr(first_op_desc, ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
    if (!ge::AttrUtils::SetStr(fus_opdef, ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
      REPORT_FE_ERROR("[SubGraphOpt][PostProcess][CrtFusOp] Op[%s]: Failed to set the attribute %s.",
                      fusion_node_name.c_str(), ge::ATTR_NAME_SESSION_GRAPH_ID.c_str());
      return nullptr;
    }
  }

  int64_t imply_type = GetImplyTypeToFusionOp(fus_nodelist);
  FE_CHECK(imply_type == -1,
           REPORT_FE_ERROR("[SubGraphOpt][PostProcess][CrtFusOp] The imply_type -1 is invalid, it must be in [0,11]."),
           return nullptr);

  if (!AttrUtils::SetInt(fus_opdef, FE_IMPLY_TYPE, imply_type)) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][CrtFusOp] Op[%s]: set imply_type failed, the imply_type is %ld.",
                    fusion_node_name.c_str(), imply_type);
    return nullptr;
  }

  FE_LOGD("Op[%s]: the implytype is %ld.", fusion_node_name.c_str(), imply_type);

  ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
  slice_info_ptr = first_node->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  if (!OpIsAutoThread(slice_info_ptr)) {
    return SetTBEFusionOp(fus_nodelist, fus_opdef, engine_name, pass_name);
  } else {
    return SetMultiKernelTBEFusionOp(fus_nodelist, fus_opdef, engine_name, pass_name);
  }
}

/*
data0   data1
  \       /
   \     /
    \0  /1
     conv data2
      \0    / 1
     dequant data3
        \0   /1
       dequant1
          \0
        output
info:
    1)conv/dequant/dequant1 is fusion_nodes
    2)conv inplace {{0, 0}}
    3) dequat inplace{{0, 0}, {0, 1}}
    4)dequant inplace{{0, 0}, {0, 1}}
how to get the fusion op inplace info:
    1) traverse fusion_node_list, to find outer output node [dequant1]
    2) traverse dequant1's all data anchor and to find satisfies its'  inplace output idx
    3) to find the dequant1, inplace input idex (0, 1), whose src node is {dequant, data3}
    4) to judge {dequant, data3} node is outer input node?
    5) dequant is not outer input node, continue to find outer input node if dequant with inplace[travese]
       data3 is outer input node, save the tensor ptr
    repeat 2) -> 5) to find all out input [data1, data2, data3]
    and final the fusion op inplace is {{0,1},{0,2},{0,3}}
*/

bool FusionOpComm::GetOutputInplaceAbilityAttrs(const vector<ge::NodePtr> &fus_nodelist,
                                                vector<vector<ge::ConstGeTensorDescPtr>> &output_place) {
  // find outer output
  FE_LOGD("Start to Get fusion op OutputInplaceAbility.");
  std::set<ge::NodePtr> fus_node_set;
  for (auto node : fus_nodelist) {
    fus_node_set.emplace(node);
  }
  for (auto node : fus_nodelist) {
    auto all_output_anchor = node->GetAllOutDataAnchors();
    for (auto anchor : all_output_anchor) {
      FindFusionNodesOuterOutput(anchor, node, fus_node_set, output_place);
    }
  }
  // only output_place.size > 0, the fusion op with outputPlaceAbility
  return output_place.size() > 0;
}

ge::OpDescPtr FusionOpComm::SetMultiKernelTBEFusionOp(const vector<ge::NodePtr> &fus_nodelist,
                                                      ge::OpDescPtr &fus_opdef,
                                                      const string &engine_name,
                                                      const string &pass_name) {
  if (fus_nodelist.empty()) {
    FE_LOGE("fusNodelist is empty.");
    return nullptr;
  }
  ge::NodePtr first_node = fus_nodelist[0];
  for (const auto &node : fus_nodelist) {
    if (node->GetOpDesc()->HasAttr(ge::TVM_ATTR_NAME_THREAD_MAGIC)) {
      FE_LOGD("Op[%s, %s] has attr TVM_ATTR_NAME_THREAD_MAGIC.",
              node->GetOpDesc()->GetName().c_str(),  node->GetOpDesc()->GetType().c_str());
      first_node = node;
      break;
    }
  }

  // AddTvmNode
  (void)AddTvmNode(fus_opdef);

  // 1 type same with fusion node_list
  ge::OpDescUtilsEx::SetType(fus_opdef, first_node->GetOpDesc()->GetType());
  (void)ge::AttrUtils::SetBool(fus_opdef, ATTR_NAME_IS_COMPIED_FUSION_OP, true);

  // set json infos to fusion op
  SetStrPathToFusionOp(fus_nodelist, fus_opdef, ge::ATTR_NAME_OP_FILE_PATH);
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, AIPP_CONV_FLAG);
  SetListStrAttrToFusionOp(fus_nodelist, fus_opdef, ge::TVM_ATTR_NAME_THREAD_MAGIC);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, FUSION_OP_SLICE_INFO);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, ge::ATTR_NAME_ENGINE_NAME_FOR_LX);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, ge::ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX);
  SetListInt64AttrToFusionOp(fus_nodelist, fus_opdef, ATTR_NAME_THREAD_TBE_KERNEL_SIZE);
  SetListBytesAttrToFusionOp(fus_nodelist, fus_opdef, ge::ATTR_NAME_THREAD_TBE_KERNEL_BUFFER);
  SetListInt64AttrToFusionOp(fus_nodelist, fus_opdef, ge::TVM_ATTR_NAME_THREAD_BLOCKDIM);
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, NEED_RE_PRECOMPILE);
  SetListBoolAttrToFusionOp(fus_nodelist, fus_opdef, ge::TVM_ATTR_NAME_THREAD_N_BATCH_SPLIT);
  SetMultiKernelThreadKernelNameToFusionOp(fus_nodelist, fus_opdef);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, FE_IMPLY_TYPE);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, kGlobalworkspaceBytes);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, kGlobalworkspaceType);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, kGlobalworkspaceSize);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, kAttrScheduleMode);

  SetListStrAttrToFusionOp(fus_nodelist, fus_opdef, ge::TVM_ATTR_NAME_THREAD_METADATA);
  SetListStrAttrToFusionOp(fus_nodelist, fus_opdef, kOpDfxOptions);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, kOpDfxBufferSize);
  SetListListInt64AttrToFusionOp(fus_nodelist, fus_opdef, TBE_OP_THREAD_ATOMIC_OUTPUT_INDEX);
  SetListInt64AttrToFusionOp(fus_nodelist, fus_opdef, TBE_OP_THREAD_ATOMIC_WORKSPACE_FLAG);
  SetListInt64AttrToFusionOp(fus_nodelist, fus_opdef, ATTR_NAME_THREAD_TVM_CACHE_READ_MODE);
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, ATTR_NAME_FE_WEIGHT_COMPRESS);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, kAttrKernelBinId);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, kAttrMemsetKernelBinId);

  SetMemAttrToFusionOp(fus_nodelist, fus_opdef, ge::ATTR_NAME_CONTINUOUS_INPUT);
  SetMemAttrToFusionOp(fus_nodelist, fus_opdef, ge::ATTR_NAME_CONTINUOUS_OUTPUT);
  SetGroupIdToFusionOp(fus_nodelist, fus_opdef, engine_name);
  SetSqrtmodeToFusionOp(fus_nodelist, fus_opdef);
  SetDataDumpAttrToFusionOp(fus_nodelist, fus_opdef, pass_name);

  // set attrs related to unknown shape
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, ATTR_NAME_SUPPORT_DYNAMIC_SHAPE);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, COMPILE_INFO_JSON);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, COMPILE_INFO_KEY);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, kAttrOptionalInputMode);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, kAttrOptionalOutputMode);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, kAttrDynamicParamMode);
  SetListInt64AttrToFusionOp(fus_nodelist, fus_opdef, OP_THREAD_PARA_SIZE);
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, ATTR_NAME_LX_FUSION_PASS);
  SetStrVecAttrToFusionOp(fus_nodelist, fus_opdef, ATTR_NAME_OP_INFER_DEPENDS);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, ATTR_NAME_KERNEL_LIST_FIRST_NAME);
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, ge::ATTR_NAME_REFERENCE);

  // for ffts_plus
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, kTypeFFTSPlus);
  SetListStrAttrToFusionOp(fus_nodelist, fus_opdef, ATTR_NAME_THREAD_CUBE_VECTOR_CORE_TYPE);

  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, kThreadId);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, kSliceInstanceNum);
  SetNewStrVecAttrToFusionOp(fus_nodelist, fus_opdef, kOriginalNode);

  if (Configuration::Instance(engine_name).EnableL1Fusion()) {
    SetL1InfoToFusionOp(fus_nodelist, fus_opdef);
    SetL1FusionSubGraphNoToFusionOp(fus_nodelist, fus_opdef);
    SetNoTaskAndDumpNeeded(fus_nodelist, fus_opdef);
  }
  std::string build_mode_value = FEContextUtils::GetBuildMode();
  if (Configuration::Instance(engine_name).EnableL2Fusion()) {
    SetL2InfoToFusionOp(fus_nodelist, fus_opdef);
    SetNoTaskAndDumpNeeded(fus_nodelist, fus_opdef);
  }
  if (Configuration::Instance(engine_name).GetDumpOriginalNodesEnable()) {
    SetOriginalNodesOpDescToFusionOp(fus_nodelist, fus_opdef);
  }

  vector<int64_t> work_space_bytes = first_node->GetOpDesc()->GetWorkspaceBytes();
  fus_opdef->SetWorkspaceBytes(work_space_bytes);

  vector<int64_t> work_space = first_node->GetOpDesc()->GetWorkspace();
  fus_opdef->SetWorkspace(work_space);

  std::vector<ge::OpKernelBinPtr> tbe_kernel_ptr_vec =
      first_node->GetOpDesc()->TryGetExtAttr(ge::OP_EXTATTR_NAME_THREAD_TBE_KERNEL, vector<ge::OpKernelBinPtr>());
  if (tbe_kernel_ptr_vec.empty()) {
    REPORT_FE_ERROR("%s Node[%s, %s]: the tbe_kernel_ptr_old is nullptr.", kStageSetFusionOp.c_str(),
                    first_node->GetOpDesc()->GetName().c_str(), first_node->GetOpDesc()->GetType().c_str());
    return nullptr;
  }

  vector<std::string> kernel_name_vec;
  (void)ge::AttrUtils::GetListStr(first_node->GetOpDesc(), kThreadKernelName, kernel_name_vec);
  if (kernel_name_vec.size() < tbe_kernel_ptr_vec.size()) {
    REPORT_FE_ERROR("%s The size %zu of kernel_name_vec is less than the size %zu of tbe_kernel_ptr_vec.",
                    kStageSetFusionOp.c_str(), kernel_name_vec.size(), tbe_kernel_ptr_vec.size());
    return nullptr;
  }

  std::vector<ge::OpKernelBinPtr> list_buffer_vec;
  for (size_t i = 0; i < tbe_kernel_ptr_vec.size(); i++) {
    auto buffer_size = tbe_kernel_ptr_vec[i]->GetBinDataSize();
    std::vector<char> buffer_vec(tbe_kernel_ptr_vec[i]->GetBinData(),
                                 tbe_kernel_ptr_vec[i]->GetBinData() + buffer_size);
    ge::OpKernelBinPtr tbe_kernel_ptr_fused = nullptr;
    FE_MAKE_SHARED(tbe_kernel_ptr_fused = std::make_shared<ge::OpKernelBin>(kernel_name_vec[i], std::move(buffer_vec)),
                  return nullptr);
    list_buffer_vec.emplace_back(tbe_kernel_ptr_fused);
  }
  if (!list_buffer_vec.empty()) {
    fus_opdef->SetExtAttr(ge::OP_EXTATTR_NAME_THREAD_TBE_KERNEL, list_buffer_vec);
  } else {
    FE_LOGW("List buffer vector of op %s is empty.", fus_opdef->GetName().c_str());
  }

  (void)ge::AttrUtils::SetListStr(fus_opdef, ge::ATTR_NAME_THREAD_TBE_KERNEL_NAME, kernel_name_vec);
  fus_opdef->SetOpEngineName(first_node->GetOpDesc()->GetOpEngineName());
  fus_opdef->SetOpKernelLibName(first_node->GetOpDesc()->GetOpKernelLibName());
  return fus_opdef;
}

Status FusionOpComm::CopyFusionScopeAttr(const ge::OpDescPtr &src_op_desc, ge::OpDescPtr &dst_op_desc) {
  if (src_op_desc == nullptr || dst_op_desc == nullptr) {
    return FAILED;
  }
  int64_t scope_id = 0;
  bool is_l1_fusion = false;
  if (!GetFusionScopeAttr(src_op_desc, scope_id, is_l1_fusion)) {
    FE_LOGE("Failed to get scope attr from node[%s, %s].",
            src_op_desc->GetName().c_str(), src_op_desc->GetType().c_str());
    return FAILED;
  }
  if (is_l1_fusion) {
    if (!ge::AttrUtils::SetInt(dst_op_desc, L1_SCOPE_ID_ATTR, scope_id)) {
      FE_LOGE("Failed to set fusion scope attr[%s] for Op[%s, %s].",
              L1_SCOPE_ID_ATTR.c_str(), dst_op_desc->GetName().c_str(), dst_op_desc->GetType().c_str());
      return FAILED;
    }
  } else {
    if (!ge::AttrUtils::SetInt(dst_op_desc, SCOPE_ID_ATTR, scope_id)) {
      FE_LOGE("Failed to set fusion scope attr[%s] for Op[%s, %s].",
              SCOPE_ID_ATTR.c_str(), dst_op_desc->GetName().c_str(), dst_op_desc->GetType().c_str());
      return FAILED;
    }
  }
  if (ge::AttrUtils::GetInt(src_op_desc, FAILED_L1_SCOPE_ID_ATTR, scope_id)) {
    (void)ge::AttrUtils::SetInt(dst_op_desc, FAILED_L1_SCOPE_ID_ATTR, scope_id);
  }
  return SUCCESS;
}

ge::OpDescPtr FusionOpComm::SetTBEFusionOp(const vector<ge::NodePtr> &fus_nodelist,
                                           ge::OpDescPtr &fus_opdef,
                                           const string &engine_name,
                                           const string &pass_name) {
  if (fus_nodelist.empty()) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][SetTBEFusOp] fusNodelist is empty.");
    return nullptr;
  }
  ge::NodePtr first_node = fus_nodelist[0];
  for (const auto &node : fus_nodelist) {
    if (node->GetOpDesc()->HasAttr(ge::TVM_ATTR_NAME_MAGIC)) {
      FE_LOGD("Op[%s, %s] has attr TVM_ATTR_NAME_MAGIC.",
              node->GetOpDesc()->GetName().c_str(),  node->GetOpDesc()->GetType().c_str());
      first_node = node;
      break;
    }
  }

  // AddTvmNode
  (void)AddTvmNode(fus_opdef);

  // 1 type same with fusion node_list
  ge::OpDescUtilsEx::SetType(fus_opdef, first_node->GetOpDesc()->GetType());
  RefreshFusionOpType(fus_opdef, first_node);
  (void)ge::AttrUtils::SetBool(fus_opdef, ATTR_NAME_IS_COMPIED_FUSION_OP, true);

  // set json infos to fusion op
  SetStrPathToFusionOp(fus_nodelist, fus_opdef, ge::ATTR_NAME_OP_FILE_PATH);
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, AIPP_CONV_FLAG);
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, kMixIsAiv);
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, kFftsplusTask);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, ge::TVM_ATTR_NAME_MAGIC);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, FUSION_OP_SLICE_INFO);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, ge::ATTR_NAME_ENGINE_NAME_FOR_LX);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, ge::ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, ge::ATTR_NAME_OM_BINARY_PATH);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, kAttrKernelBinId);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, kAttrScheduleMode);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, kLocalMemorySize);
  SetGroupIdToFusionOp(fus_nodelist, fus_opdef, engine_name);
  std::string core_type;
  (void)ge::AttrUtils::GetStr(fus_opdef, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
  FE_LOGD("Fused node: %s, core type: %s.", fus_opdef->GetName().c_str(), core_type.c_str());
  if (!fus_opdef->HasAttr(ge::ATTR_NAME_OM_BINARY_PATH)) {
    Status ret = FAILED;
    if ((core_type == "MIX_AIC") || (core_type == "MIX_AIV")) {
      ret = SetMixFusionOpAttr(fus_nodelist, fus_opdef, first_node);
    } else {
      ret = SetNormalFusionOpAttr(fus_nodelist, fus_opdef, first_node);
    }
    if (ret == FAILED) {
      return nullptr;
    }
  }
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, kModeInArgsFirstField);
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, kAttrIntercoreSync);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, ge::TVM_ATTR_NAME_BLOCKDIM);
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, NEED_RE_PRECOMPILE);
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, ge::ATTR_N_BATCH_SPILT);
  SetKernelNameToFusionOp(fus_nodelist, fus_opdef);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, FE_IMPLY_TYPE);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, kGlobalworkspaceBytes);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, kGlobalworkspaceType);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, kGlobalworkspaceSize);

  SetListInt64AttrToFusionOp(fus_nodelist, fus_opdef, TBE_OP_ATOMIC_OUTPUT_INDEX);
  SetListInt64AttrToFusionOp(fus_nodelist, fus_opdef, TBE_OP_ATOMIC_WORKSPACE_INDEX);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, TBE_OP_ATOMIC_WORKSPACE_FLAG);
  SetListInt32AttrToFusionOp(fus_nodelist, fus_opdef, TBE_OP_ATOMIC_DTYPES);
  SetListInt64AttrToFusionOp(fus_nodelist, fus_opdef, TBE_OP_ATOMIC_INT64_VALUES);
  SetListFloatAttrToFusionOp(fus_nodelist, fus_opdef, TBE_OP_ATOMIC_FLOAT_VALUES);
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, ATTR_NAME_FE_WEIGHT_COMPRESS);

  SetMemAttrToFusionOp(fus_nodelist, fus_opdef, ge::ATTR_NAME_CONTINUOUS_INPUT);
  SetMemAttrToFusionOp(fus_nodelist, fus_opdef, ge::ATTR_NAME_CONTINUOUS_OUTPUT);
  SetSqrtmodeToFusionOp(fus_nodelist, fus_opdef);
  SetDataDumpAttrToFusionOp(fus_nodelist, fus_opdef, pass_name);

  // set attrs related to unknown shape
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, ATTR_NAME_SUPPORT_DYNAMIC_SHAPE);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, COMPILE_INFO_JSON);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, COMPILE_INFO_KEY);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, kAttrOptionalInputMode);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, kAttrOptionalOutputMode);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, kAttrDynamicParamMode);
  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, kAttrHeadFilePath);

  SetStrAttrToFusionOp(fus_nodelist, fus_opdef, OP_JSON_FILE_PATH);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, OP_PARA_SIZE);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, ORI_OP_PARA_SIZE);
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, kMemoryCheck);
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, ATTR_NAME_LX_FUSION_PASS);
  SetStrVecAttrToFusionOp(fus_nodelist, fus_opdef, ATTR_NAME_OP_INFER_DEPENDS);
  SetBoolAttrToFusionOp(fus_nodelist, fus_opdef, ge::ATTR_NAME_REFERENCE);

  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, kThreadId);
  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, kSliceInstanceNum);
  SetNewStrVecAttrToFusionOp(fus_nodelist, fus_opdef, kOriginalNode);
  SetSgtCoreTypeToFusionOp(fus_nodelist, fus_opdef);

  // weight prefetch attr
  SetStrVecAttrToFusionOp(fus_nodelist, fus_opdef, kAttrWeightPrefetchType);
  SetListInt64AttrToFusionOp(fus_nodelist, fus_opdef, kAttrWeightPrefetchDstOffset);
  SetListInt64AttrToFusionOp(fus_nodelist, fus_opdef, kAttrWeightPrefetchSrcOffset);
  SetListInt64AttrToFusionOp(fus_nodelist, fus_opdef, kAttrWeightPrefetchDataSize);

  SetInt64AttrToFusionOp(fus_nodelist, fus_opdef, OP_IMPL_MODE_ENUM);
  if (Configuration::Instance(engine_name).EnableL1Fusion()) {
    SetL1InfoToFusionOp(fus_nodelist, fus_opdef);
    SetL1FusionSubGraphNoToFusionOp(fus_nodelist, fus_opdef);
    SetNoTaskAndDumpNeeded(fus_nodelist, fus_opdef);
  }
  std::string build_mode_value = FEContextUtils::GetBuildMode();
  if (Configuration::Instance(engine_name).EnableL2Fusion()) {
    SetL2InfoToFusionOp(fus_nodelist, fus_opdef);
    SetNoTaskAndDumpNeeded(fus_nodelist, fus_opdef);
  }
  if (Configuration::Instance(engine_name).GetDumpOriginalNodesEnable()) {
    SetOriginalNodesOpDescToFusionOp(fus_nodelist, fus_opdef);
  }
  vector<int64_t> work_space_bytes = first_node->GetOpDesc()->GetWorkspaceBytes();
  fus_opdef->SetWorkspaceBytes(work_space_bytes);

  vector<int64_t> work_space = first_node->GetOpDesc()->GetWorkspace();
  fus_opdef->SetWorkspace(work_space);
  fus_opdef->SetOpEngineName(first_node->GetOpDesc()->GetOpEngineName());
  fus_opdef->SetOpKernelLibName(first_node->GetOpDesc()->GetOpKernelLibName());
  return fus_opdef;
}

void FusionOpComm::SetDataDumpAttrToFusionOp(const vector<ge::NodePtr> &fus_nodelist,
                                             ge::OpDescPtr &fus_opdef, const string& pass_name) {
  std::vector<ge::NodePtr> original_nodes;
  for (ge::NodePtr node : fus_nodelist) {
    original_nodes.push_back(node);
  }
  (void)RecordOriginalNames(original_nodes, fus_opdef);
  GraphPassUtil::RecordOriginalOpAttrs(original_nodes, fus_opdef, pass_name);

  bool is_multi_op = false;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_IS_MULTIOP, is_multi_op)) {
      (void)ge::AttrUtils::SetBool(fus_opdef, ge::ATTR_NAME_DATA_DUMP_IS_MULTIOP, is_multi_op);
      break;
    }
  }
}

void FusionOpComm::SetOriginalNodesOpDescToFusionOp(const vector<ge::NodePtr> &fus_nodelist,
                                                    const ge::OpDescPtr &fus_opdef) {
  ge::ModelSerializeImp model_serialize_imp;
  ge::proto::OpDef op_def_proto;
  std::string op_def_str;
  std::vector<std::string> original_nodes;
  for (const ge::NodePtr &node : fus_nodelist) {
    if (model_serialize_imp.SerializeNode(node, &op_def_proto)) {
      try {
        google::protobuf::TextFormat::PrintToString(op_def_proto, &op_def_str);
      } catch (const std::exception &e) {
        FE_LOGW("Failed to PrintToString. Message is %s", e.what());
      }
      original_nodes.push_back(op_def_str);
    } else {
      FE_LOGW("Serialize node[name:%s] not successfully.", node->GetName().c_str());
    }
  }
  if (!ge::AttrUtils::SetListStr(fus_opdef, ORIGINAL_NODES, original_nodes)) {
    FE_LOGW("Set original_nodes not successfully to node [%s].", fus_opdef->GetName().c_str());
  }
}

void FusionOpComm::RecordOriginalNames(std::vector<ge::NodePtr> original_nodes, ge::OpDescPtr &op_desc) {
  std::vector<std::string> original_names;
  std::vector<std::string> original_types;
  for (ge::NodePtr node_tmp : original_nodes) {
    std::vector<std::string> names_tmp;
    std::vector<std::string> types_tmp;
    ge::OpDescPtr opdesc_tmp = node_tmp->GetOpDesc();
    FE_CHECK(opdesc_tmp == nullptr, FE_LOGW("opdescTmp is null."), return);
    bool is_has_attr = ge::AttrUtils::GetListStr(opdesc_tmp, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, names_tmp);
    ge::AttrUtils::GetListStr(opdesc_tmp, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, types_tmp);
    if (is_has_attr) {
      if (!names_tmp.empty()) {
        for (auto &node_name : names_tmp) {
          if (!node_name.empty()) {
            original_names.push_back(node_name);
          }
        }
      }
      if (!types_tmp.empty()) {
        for (auto &node_type : types_tmp) {
          if (!node_type.empty()) {
            original_types.push_back(node_type);
          }
        }
      }
    } else {
      original_names.push_back(opdesc_tmp->GetName());
      original_types.push_back(opdesc_tmp->GetType());
    }
  }

  if (!ge::AttrUtils::SetListStr(op_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names)) {
    FE_LOGW("Failed to set ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES.");
    return;
  }
  if (!ge::AttrUtils::SetListStr(op_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, original_types)) {
    FE_LOGW("Failed to set ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES.");
    return;
  }
}

Status FusionOpComm::AddTvmNode(ge::OpDescPtr op_desc) {
  if (!ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][AddTvmNode] Op[%s]: Failed to set the attribute %s.",
                    op_desc->GetName().c_str(), ge::ATTR_NAME_IMPLY_TYPE.c_str());
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace fe
