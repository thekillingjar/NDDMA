/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_flow_attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/debug/ge_log.h"
#include "common/checker.h"
#include "graph/utils/anchor_utils.h"

namespace ge {
namespace {
constexpr const char *kDefaultEnqueuePolicy = "FIFO";
constexpr const char *kOverwritePolicy = "OVERWRITE";
constexpr int32_t kDefaultFifoDepth = 128;
constexpr int32_t kInvalidDepth = -1;

template <typename T>
static Status CheckFlowAttr(const T &obj) {
  int32_t depth = kDefaultFifoDepth;
  std::string policy(kDefaultEnqueuePolicy);
  (void)ge::AttrUtils::GetInt(obj, ATTR_NAME_FLOW_ATTR_DEPTH, depth);
  (void)ge::AttrUtils::GetStr(obj, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, policy);
  GE_ASSERT_TRUE(depth > 0, "[Check][FlowAttr] failed, depth=%d is invalid.", depth);
  GE_ASSERT_TRUE((policy == kDefaultEnqueuePolicy) || (policy == kOverwritePolicy),
                 "[Check][FlowAttr] failed, policy must be OVERWRITE or FIFO, but is %s.", policy.c_str());
  return SUCCESS;
}

template <typename T>
static Status GetFlowAttrRaw(const T &obj, NamedAttrs &attrs, bool &has_attr) {
  (void)ge::AttrUtils::GetBool(obj, ATTR_NAME_FLOW_ATTR, has_attr);
  if (has_attr) {
    GE_ASSERT_SUCCESS(CheckFlowAttr(obj), "[Get][FlowAttr] failed.");
    int32_t depth = kInvalidDepth;
    std::string policy;
    bool has_depth = ge::AttrUtils::GetInt(obj, ATTR_NAME_FLOW_ATTR_DEPTH, depth);
    bool has_policy = ge::AttrUtils::GetStr(obj, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, policy);
    // value has check in CheckFlowAttr
    if (has_depth) {
      (void)ge::AttrUtils::SetInt(attrs, ATTR_NAME_FLOW_ATTR_DEPTH, depth);
    }
    if (has_policy) {
      (void)ge::AttrUtils::SetStr(attrs, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, policy);
    }
  }
  return SUCCESS;
}

template <typename T>
static Status GetFlowAttr(const T &obj, NamedAttrs &attrs, bool &has_attr, const int32_t def_fifo_depth,
                          const std::string &def_enqueue_policy) {
  (void)ge::AttrUtils::GetBool(obj, ATTR_NAME_FLOW_ATTR, has_attr);
  if (has_attr) {
    GE_ASSERT_SUCCESS(CheckFlowAttr(obj), "[Get][FlowAttr] failed.");
    int32_t depth = def_fifo_depth;
    std::string policy(def_enqueue_policy);
    (void)ge::AttrUtils::GetInt(obj, ATTR_NAME_FLOW_ATTR_DEPTH, depth);
    (void)ge::AttrUtils::GetStr(obj, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, policy);
    if (depth > 0) {
      GE_ASSERT_TRUE(ge::AttrUtils::SetInt(attrs, ATTR_NAME_FLOW_ATTR_DEPTH, depth),
                     "[Set][Int] attr = %s, val = %d failed", ATTR_NAME_FLOW_ATTR_DEPTH.c_str(), depth);
    }
    if (!policy.empty()) {
      GE_ASSERT_TRUE(ge::AttrUtils::SetStr(attrs, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, policy),
                     "[Set][Str] attr = %s, val = %s failed", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(),
                     policy.c_str());
    }
  }
  return SUCCESS;
}

template <typename T>
static Status SetFlowAttrRaw(const NamedAttrs &attrs, T &obj) {
  bool has_attr = false;
  (void)ge::AttrUtils::GetBool(obj, ATTR_NAME_FLOW_ATTR, has_attr);
  if (!has_attr) {
    (void)ge::AttrUtils::SetBool(obj, ATTR_NAME_FLOW_ATTR, true);
    int32_t depth = kInvalidDepth;
    std::string policy;
    bool has_depth = ge::AttrUtils::GetInt(attrs, ATTR_NAME_FLOW_ATTR_DEPTH, depth);
    bool has_policy = ge::AttrUtils::GetStr(attrs, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, policy);
    if (has_depth && (depth > 0)) {
      GE_ASSERT_TRUE(ge::AttrUtils::SetInt(obj, ATTR_NAME_FLOW_ATTR_DEPTH, depth),
                     "[Set][Int] attr = %s, val = %d failed", ATTR_NAME_FLOW_ATTR_DEPTH.c_str(), depth);
    }
    if (has_policy && !policy.empty()) {
      GE_ASSERT_TRUE(ge::AttrUtils::SetStr(obj, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, policy),
                     "[Set][Str] attr = %s, val = %s failed", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(),
                     policy.c_str());
    }
  }
  return SUCCESS;
}

template <typename T>
static Status SetFlowAttr(const NamedAttrs &attrs, T &obj, const int32_t def_fifo_depth,
                          const std::string &def_enqueue_policy) {
  bool has_attr = false;
  (void)ge::AttrUtils::GetBool(obj, ATTR_NAME_FLOW_ATTR, has_attr);
  if (!has_attr) {
    (void)ge::AttrUtils::SetBool(obj, ATTR_NAME_FLOW_ATTR, true);
    int32_t depth = def_fifo_depth;
    std::string policy(def_enqueue_policy);
    (void)ge::AttrUtils::GetInt(attrs, ATTR_NAME_FLOW_ATTR_DEPTH, depth);
    (void)ge::AttrUtils::GetStr(attrs, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, policy);
    GE_ASSERT_TRUE(ge::AttrUtils::SetBool(obj, ATTR_NAME_FLOW_ATTR, true), "SetBool attribute of %s failed.",
                   ATTR_NAME_FLOW_ATTR.c_str());
    GE_ASSERT_TRUE(ge::AttrUtils::SetInt(obj, ATTR_NAME_FLOW_ATTR_DEPTH, depth), "SetInt attribute of %s failed.",
                   ATTR_NAME_FLOW_ATTR_DEPTH.c_str());
    GE_ASSERT_TRUE(ge::AttrUtils::SetStr(obj, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, policy),
                   "SetStr attribute of %s failed.", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str());
    GELOGD("Set flow attr, depth = %d, policy = %s.", depth, policy.c_str());
  }
  return SUCCESS;
}
}  // namespace
Status DataFlowAttrUtils::SupplementFlowAttr(const ComputeGraphPtr &root_graph) {
  // if do not have graph attr, use global default attr
  int32_t default_queue_depth = kDefaultFifoDepth;
  std::string default_queue_policy = kDefaultEnqueuePolicy;
  (void)ge::AttrUtils::GetInt(root_graph, ATTR_NAME_FLOW_ATTR_DEPTH, default_queue_depth);
  (void)ge::AttrUtils::GetStr(root_graph, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, default_queue_policy);
  GELOGI("Get default queue in supplement flow attr, default depth = %d, enqueue policy = %s, root graph = %s.",
         default_queue_depth, default_queue_policy.c_str(), root_graph->GetName().c_str());
  for (const auto &node : root_graph->GetDirectNode()) {
    // if op has been set flow attr, all input/output should been set flow attr.
    GE_ASSERT_SUCCESS(SupplementNodeFlowAttr(node), "[Call][PrepareNodeFlowAttr] of node:%s failed.",
                      node->GetName().c_str());
  }
  for (const auto &node : root_graph->GetDirectNode()) {
    for (auto &in_anchor : node->GetAllInDataAnchors()) {
      auto in_tensor = in_anchor->GetOwnerNode()->GetOpDesc()->MutableInputDesc(AnchorUtils::GetIdx(in_anchor));
      // if op input/output has been set flow attr, the peer tensor should been set flow attr as well.
      const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
      if (peer_out_anchor == nullptr) {
        continue;
      }
      bool has_node_attr = false;
      NamedAttrs node_attr;
      int32_t node_queue_depth = kInvalidDepth;
      std::string node_queue_policy;
      GE_ASSERT_SUCCESS(GetFlowAttrRaw(node->GetOpDesc(), node_attr, has_node_attr),
                        "[Get][FlowAttr] failed, node = %s.", node->GetName().c_str());
      (void)ge::AttrUtils::GetInt(node_attr, ATTR_NAME_FLOW_ATTR_DEPTH, node_queue_depth);
      (void)ge::AttrUtils::GetStr(node_attr, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, node_queue_policy);
      if (has_node_attr) {
        if (node_queue_depth > 0) {
          default_queue_depth = node_queue_depth;
        }
        if (!node_queue_policy.empty()) {
          default_queue_policy = node_queue_policy;
        }
      }
      GE_ASSERT_SUCCESS(
          SupplementMismatchEdge(peer_out_anchor, node, in_tensor, default_queue_depth, default_queue_policy),
          "[Call][PrepareMismatchEdge] failed, node:%s, default depth = %d, policy = %s", node->GetName().c_str(),
          default_queue_depth, default_queue_policy.c_str());
    }
  }
  return SUCCESS;
}
Status DataFlowAttrUtils::SupplementNodeFlowAttr(const NodePtr &node) {
  NamedAttrs node_attr;
  bool has_node_flow_attr = false;
  GE_ASSERT_SUCCESS(GetFlowAttrRaw(node->GetOpDesc(), node_attr, has_node_flow_attr),
                    "[Get][FlowAttr] failed, node:%s.", node->GetName().c_str());
  // prepare for node input/output
  for (uint32_t idx = 0U; idx < node->GetOpDesc()->GetAllInputsSize(); idx++) {
    const auto &in_tensor = node->GetOpDesc()->MutableInputDesc(idx);
    bool has_edge_flow_attr = false;
    (void)ge::AttrUtils::GetBool(in_tensor, ATTR_NAME_FLOW_ATTR, has_edge_flow_attr);
    if (has_edge_flow_attr) {
      GE_ASSERT_SUCCESS(CheckFlowAttr(in_tensor), "[Check][FlowAttr] failed of node:%s in tensor:%u.",
                        node->GetName().c_str(), idx);
    }
    if (!has_edge_flow_attr && has_node_flow_attr) {
      GE_ASSERT_SUCCESS(SetFlowAttrRaw(node_attr, in_tensor), "[Set][FlowAttr] of data:%s in tensor(in) %u failed.",
                        node->GetName().c_str(), idx);
      GELOGD("Set node:%s _flow_attr in tensor:%u successfully.", node->GetName().c_str(), idx);
    }
  }
  for (uint32_t idx = 0U; idx < node->GetOpDesc()->GetAllOutputsDescSize(); idx++) {
    const auto &out_tensor = node->GetOpDesc()->MutableOutputDesc(idx);
    bool has_edge_flow_attr = false;
    (void)ge::AttrUtils::GetBool(out_tensor, ATTR_NAME_FLOW_ATTR, has_edge_flow_attr);
    if (has_edge_flow_attr) {
      GE_ASSERT_SUCCESS(CheckFlowAttr(out_tensor), "[Check][FlowAttr] failed of node:%s in tensor(out):%u.",
                        node->GetName().c_str(), idx);
    }
    if (!has_edge_flow_attr && has_node_flow_attr) {
      GE_ASSERT_SUCCESS(SetFlowAttrRaw(node_attr, out_tensor), "[Set][FlowAttr] of data:%s in tensor %u failed.",
                        node->GetName().c_str(), idx);
      GELOGD("Set flow attr of node[%s] output index:%u successfully.", node->GetName().c_str(), idx);
    }
  }
  return SUCCESS;
}

Status DataFlowAttrUtils::SupplementMismatchEdge(const DataAnchorPtr &peer_out_anchor, const NodePtr &node,
                                                 GeTensorDescPtr &in_tensor, const int32_t def_fifo_depth,
                                                 const std::string &def_enqueue_policy) {
  bool has_in_tensor_attr = false;
  bool has_out_tensor_attr = false;
  const bool has_in_attr = AttrUtils::GetBool(in_tensor, ATTR_NAME_FLOW_ATTR, has_in_tensor_attr);
  NamedAttrs in_tensor_attr;
  if (has_in_attr) {
    GE_ASSERT_SUCCESS(GetFlowAttrRaw(in_tensor, in_tensor_attr, has_in_tensor_attr),
                      "[Get][FlowAttr] failed, node = %s.", node->GetName().c_str());
  }
  auto out_node = peer_out_anchor->GetOwnerNode();
  auto out_tensor = out_node->GetOpDesc()->MutableOutputDesc(AnchorUtils::GetIdx(peer_out_anchor));
  if (out_tensor == nullptr) {
    GELOGD("Get out control anchor of node:%s is null.", node->GetName().c_str());
    return SUCCESS;
  }
  const bool has_out_attr = AttrUtils::GetBool(out_tensor, ATTR_NAME_FLOW_ATTR, has_out_tensor_attr);
  NamedAttrs out_tensor_attr;
  if (has_out_attr) {
    GE_ASSERT_SUCCESS(GetFlowAttrRaw(out_tensor, out_tensor_attr, has_out_tensor_attr),
                      "[Get][FlowAttr] failed, node = %s.", node->GetName().c_str());
  }
  // 1) check set input/output valid
  if (has_in_attr && has_out_attr) {
    GE_ASSERT_TRUE(has_out_tensor_attr == has_in_tensor_attr,
                   "[Check][Input] out_node:%s out:%d should be set same flow attr flag:%d to node:%s flow flag:%d",
                   out_node->GetName().c_str(), AnchorUtils::GetIdx(peer_out_anchor), has_out_tensor_attr,
                   node->GetName().c_str(), has_in_tensor_attr);
  }
  // 2) if op input/output has been set flow attr, the peer tensor should been set flow attr
  if (has_in_tensor_attr && !has_out_tensor_attr) {
    GELOGD("node = %s has input and do not has output attr, set input attr -> output", node->GetName().c_str());
    GE_ASSERT_SUCCESS(SetFlowAttr(in_tensor_attr, out_tensor, def_fifo_depth, def_enqueue_policy),
                      "[Set][Attr] of node:%s out:%u failed.", node->GetName().c_str(),
                      AnchorUtils::GetIdx(peer_out_anchor));
  }
  if (!has_in_tensor_attr && has_out_tensor_attr) {
    GELOGD("node = %s has output and do not has input attr, set output attr -> input", node->GetName().c_str());
    GE_ASSERT_SUCCESS(SetFlowAttr(out_tensor_attr, in_tensor, def_fifo_depth, def_enqueue_policy),
                      "[Set][Attr] of node:%s of in tensor failed.", node->GetName().c_str());
  }
  return SUCCESS;
}
}  // namespace ge