/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/value_holder.h"
#include "value_holder_inner.h"

#include <deque>
#include <stack>

#include <securec.h>
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_op_types.h"
#include "graph/utils/graph_utils.h"
#include "common/checker.h"

#include "exe_graph/lowering/exe_graph_attrs.h"
#include "exe_graph/lowering/extend_exe_graph.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/def_types.h"
#include "graph/fast_graph/execute_graph.h"
namespace gert {
namespace bg {
namespace {
constexpr const ge::char_t *kInnerDataNodes = "_inner_data_nodes";
thread_local std::deque<std::unique_ptr<GraphFrame>> graph_frames;
thread_local GraphFrame *current_frame;
bool IsGraphOutType(const char *node_type) {
  return strcmp(kNetOutput, node_type) == 0 || strcmp(kInnerNetOutput, node_type) == 0;
}
ge::OpDescPtr CreateOpDesc(const std::string &node_name, const char *node_type, size_t in_count, size_t out_count) {
  auto op_desc = ge::MakeShared<ge::OpDesc>(node_name, node_type);
  GE_ASSERT_NOTNULL(op_desc);
  for (size_t i = 0; i < in_count; ++i) {
    if (op_desc->AddInputDesc(ge::GeTensorDesc()) != ge::GRAPH_SUCCESS) {
      GE_LOGE("Failed to create OpDesc for node %s, io-count %zu/%zu, add input desc %zu failed ", node_name.c_str(),
              in_count, out_count, i);
      return nullptr;
    }
  }
  for (size_t i = 0; i < out_count; ++i) {
    if (op_desc->AddOutputDesc(ge::GeTensorDesc()) != ge::GRAPH_SUCCESS) {
      GE_LOGE("Failed to create OpDesc for node %s, io-count %zu/%zu, add output desc %zu failed ", node_name.c_str(),
              in_count, out_count, i);
      return nullptr;
    }
  }
  return op_desc;
}

struct ConnectionPathPoint {
  ge::FastNode *node;
  ge::ExecuteGraph *graph;
};

ge::EdgeDstEndpoint EnsureHasDataEdge(ge::FastNode *src, int32_t src_index, ge::FastNode *cur_node) {
  GE_ASSERT_NOTNULL(cur_node);
  for (const auto edge : src->GetOutEdgesRefByIndex(src_index)) {
    if (edge == nullptr) {
      continue;
    }
    auto dst_node = edge->dst;
    GE_ASSERT_NOTNULL(dst_node);
    if (dst_node == cur_node) {
      return {cur_node, edge->dst_input};
    }
  }

  const auto index = cur_node->GetDataInNum();
  GE_ASSERT_SUCCESS(ge::FastNodeUtils::AppendInputEdgeInfo(cur_node, index + 1));

  auto graph = cur_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);

  GE_ASSERT_NOTNULL(graph->AddEdge(src, src_index, cur_node, static_cast<int32_t>(index)));
  return {cur_node, static_cast<int32_t>(index)};
}

ge::Status GetTempFrameByGraph(ge::ExecuteGraph *graph, std::unique_ptr<GraphFrame> &frame) {
  GE_ASSERT_TRUE(!GetGraphFrames().empty(), "Failed to do frame selection, there is no root-frame exists");
  const auto root_frame = GetGraphFrames().begin()->get();
  GE_ASSERT_NOTNULL(root_frame, "Failed to find the root frame");

  GE_ASSERT_NOTNULL(graph);
  frame = ge::ComGraphMakeUnique<GraphFrame>(graph->shared_from_this(), *root_frame);
  GE_ASSERT_NOTNULL(frame);
  const auto current_graph_frame = ValueHolder::GetCurrentFrame();
  GE_ASSERT_NOTNULL(current_graph_frame);
  frame->SetCurrentComputeNode(current_graph_frame->GetCurrentComputeNode());

  return ge::SUCCESS;
}

ge::FastNode *EnsureHasData(const ConnectionPathPoint &point, int32_t index, bool &new_created) {
  std::unique_ptr<GraphFrame> frame;
  GE_ASSERT_SUCCESS(GetTempFrameByGraph(point.graph, frame));
  ge::FastNode *data = nullptr;
  if (!FindValFromMapExtAttr<int32_t, ge::FastNode *>(frame->GetExecuteGraph().get(), kInnerDataNodes, index, data)) {
    data = ValueHolder::AddNode(kInnerData, 0, 1, *frame);
    GE_ASSERT_NOTNULL(data);
    GE_ASSERT_TRUE(ge::AttrUtils::SetInt(data->GetOpDescBarePtr(), ge::ATTR_NAME_INDEX, index));
    AddKVToMapExtAttr<int32_t, ge::FastNode *>(frame->GetExecuteGraph().get(), kInnerDataNodes, index, data);
    new_created = true;
  }
  return data;
}

ge::Status GetOutsideGuarderType(const ge::FastNode *node, const ge::FastNode *src_node_from_parent_graph,
                                 std::string &guarder_type) {
  const auto inside_guarder_type = ge::AttrUtils::GetStr(node->GetOpDescBarePtr(), kGuarderNodeType);
  // 因为透传祖先图的valueholer而产生的InnerData都是ValueHolder类自行产生的，此时产生InnerData的时候不会追加guarder，
  // 所以理论上InnerData是不会带有guarder的, 此处只校验只有outside guarder场景，子图内部Innerdata有guarder属于异常场景
  GE_ASSERT_TRUE(inside_guarder_type == nullptr);

  const auto op_desc = src_node_from_parent_graph->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto outside_guarder_type = ge::AttrUtils::GetStr(op_desc, kNodeWithGuarderOutside);
  if (outside_guarder_type != nullptr) {
    guarder_type = *outside_guarder_type;
    return ge::SUCCESS;
  }
  const auto guarder_node_type = ge::AttrUtils::GetStr(op_desc, kGuarderNodeType);
  if (guarder_node_type != nullptr) {
    guarder_type = *guarder_node_type;
  }
  return ge::SUCCESS;
}

ge::EdgeSrcEndpoint ConnectFromParents(ge::FastNode *src, int32_t src_index, const ge::FastNode *dst) {
  auto next_graph = dst->GetExtendInfo()->GetOwnerGraphBarePtr();
  const auto src_graph = src->GetExtendInfo()->GetOwnerGraphBarePtr();
  if (src_graph != next_graph) {
    std::stack<ConnectionPathPoint> connect_path;

    bool full_path = false;
    while (next_graph != nullptr) {
      const auto parent_node = next_graph->GetParentNodeBarePtr();
      if (parent_node == nullptr) {
        // log out of loop scope
        break;
      }
      connect_path.push({parent_node, next_graph});
      next_graph = parent_node->GetExtendInfo()->GetOwnerGraphBarePtr();
      if (next_graph == src_graph) {
        full_path = true;
        break;
      }
    }

    if (!full_path) {
      GE_LOGE(
          "Failed to connect from %s index %d to node %s, the src node does not on the graph or on its parent graphs",
          src->GetNamePtr(), src_index, dst->GetNamePtr());
      return {nullptr, ge::kInvalidEdgeIndex};
    }

    while (!connect_path.empty()) {
      auto point = std::move(connect_path.top());
      connect_path.pop();

      const auto dst_endpoint = EnsureHasDataEdge(src, src_index, point.node);
      GE_ASSERT_NOTNULL(dst_endpoint.node);

      bool new_created = false;
      const auto data_node = EnsureHasData(point, dst_endpoint.index, new_created);
      GE_ASSERT_NOTNULL(data_node);

      std::string guarder_type;
      GE_ASSERT_SUCCESS(GetOutsideGuarderType(data_node, src, guarder_type));
      if ((new_created) && (!guarder_type.empty())) {
        (void)ge::AttrUtils::SetStr(data_node->GetOpDescBarePtr(), kNodeWithGuarderOutside, guarder_type);
      }

      src = data_node;
      src_index = 0;
    }
  }
  return {src, src_index};
}

ge::graphStatus AddDataEdge(ge::FastNode *src, int32_t src_index, ge::FastNode *dst, int32_t dst_index) {
  auto src_endpoint = ConnectFromParents(src, src_index, dst);
  if (src_endpoint.node == nullptr) {
    GE_LOGE("Failed to connect from %s(%d) to %s(%d), connect from parents failed", src->GetNamePtr(), src_index,
            dst->GetNamePtr(), dst_index);
    return ge::GRAPH_FAILED;
  }
  auto graph = dst->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);
  auto edge = graph->AddEdge(src_endpoint.node, src_endpoint.index, dst, dst_index);
  if (edge == nullptr) {
    GE_LOGE("Failed to connect edge from %s:%d to %s:%d", src->GetNamePtr(), src_index,
            dst->GetNamePtr(), dst_index);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

HyperStatus AddDependencyBetweenNodes(ge::FastNode *src, ge::FastNode *dst) {
  auto src_graph = src->GetExtendInfo()->GetOwnerGraphBarePtr();
  auto dst_graph = dst->GetExtendInfo()->GetOwnerGraphBarePtr();
  if (src_graph != dst_graph) {
    return HyperStatus::ErrorStatus("The source node %s(%s) and dst node %s(%s) does not on the same graph",
                                    src->GetNamePtr(), src->GetTypePtr(), dst->GetNamePtr(),
                                    dst->GetTypePtr());
  }
  if (src_graph == nullptr) {
    return HyperStatus::ErrorStatus("The source node %s(%s) and dst node %s(%s) does not on the graph",
                                    src->GetNamePtr(), src->GetTypePtr(), dst->GetNamePtr(),
                                    dst->GetTypePtr());
  }
  if (src_graph->AddEdge(src, ge::kControlEdgeIndex,
                         dst, ge::kControlEdgeIndex) == nullptr) {
    return HyperStatus::ErrorStatus("Failed to add control edge from %s to %s", src->GetNamePtr(),
                                    dst->GetNamePtr());
  }
  return HyperStatus::Success();
}

ge::graphStatus AddDependencyToGuarder(ge::FastNode *src, ge::FastNode *guarder) {
  auto guarder_graph = guarder->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(guarder_graph);
  ge::FastNode *current_node = src;
  while (current_node->GetExtendInfo()->GetOwnerGraphBarePtr() != guarder_graph) {
    auto owner_graph = current_node->GetExtendInfo()->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(owner_graph);
    auto parent_node = owner_graph->GetParentNodeBarePtr();
    GE_ASSERT_NOTNULL(parent_node,
                      "Failed to add dependency from node %s(%s) to guarder %s(%s), the guarder node does not on the "
                      "same graph or the parent graphs of the source node",
                      src->GetNamePtr(), src->GetTypePtr(), guarder->GetNamePtr(), guarder->GetTypePtr());
    current_node = const_cast<ge::FastNode *>(parent_node);
  }
  GE_ASSERT_HYPER_SUCCESS(AddDependencyBetweenNodes(current_node, guarder));
  return ge::GRAPH_SUCCESS;
}

ge::NodePtr GetComputeNodeByIndex(const GraphFrame &frame, size_t index) {
  auto &indexes_to_node = frame.GetIndexesToNode();
  GE_ASSERT_TRUE(indexes_to_node.size() > index, "The current compute node index %zu out of range", index);
  return indexes_to_node[index];
}
}  // namespace
std::atomic<int64_t> ValueHolder::id_generator_{0};
ValueHolder::~ValueHolder() = default;

ValueHolder::ValueHolder()
    : id_(id_generator_++), type_(ValueHolderType::kValueHolderTypeEnd),
      fast_node_(nullptr), index_(0), placement_(0) {}

bool ValueHolder::IsOk() const noexcept {
  return error_msg_ == nullptr;
}
ValueHolder::ValueHolderType ValueHolder::GetType() const noexcept {
  return type_;
}

ge::FastNode *ValueHolder::GetFastNode() const noexcept {
  return fast_node_;
}

int32_t ValueHolder::GetOutIndex() const noexcept {
  return index_;
}
int64_t ValueHolder::GetId() const noexcept {
  return id_;
}

ge::ExecuteGraph *ValueHolder::GetExecuteGraph() const noexcept {
  return fast_node_->GetExtendInfo()->GetOwnerGraphBarePtr();
}

ValueHolderPtr ValueHolder::CreateError(const char *fmt, va_list arg) {
  auto value_holder = std::shared_ptr<ValueHolder>(new (std::nothrow) ValueHolder());
  GE_ASSERT_NOTNULL(value_holder);
  value_holder->error_msg_ = std::unique_ptr<char[]>(CreateMessage(fmt, arg));
  return value_holder;
}
ValueHolderPtr ValueHolder::CreateError(const char *fmt, ...) {
  va_list arg;
  va_start(arg, fmt);
  auto holder = CreateError(fmt, arg);
  va_end(arg);
  return holder;
}
std::string ValueHolder::GenerateNodeName(const char *node_type, const GraphFrame &frame) {
  std::string node_name(node_type);
  const auto &current_compute_node = frame.GetCurrentComputeNode();
  if (current_compute_node != nullptr) {
    node_name.append("_").append(current_compute_node->GetName());
  }
  node_name.append("_").append(std::to_string(id_generator_));
  ++id_generator_;
  return node_name;
}

ge::FastNode *ValueHolder::AddNode(const char *node_type, size_t input_count, size_t output_count,
                                   const GraphFrame &frame) {
  auto graph = frame.GetExecuteGraph();
  GE_ASSERT_NOTNULL(graph);

  auto node = graph->AddNode(CreateOpDesc(GenerateNodeName(node_type, frame), node_type, input_count, output_count));
  GE_ASSERT_NOTNULL(node);

  // add compute node info index
  size_t index;
  if (frame.GetCurrentNodeIndex(index)) {
    if (!ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), kComputeNodeIndex, static_cast<int64_t>(index))) {
      GE_LOGE("Failed to add node %s, add ComputeNodeIndex failed", node_type);
      return nullptr;
    }
  }

  return node;
}

ge::FastNode *ValueHolder::CreateNode(const char *node_type, const std::vector<ValueHolderPtr> &inputs,
                                      size_t out_count) {
  auto frame = GetCurrentFrame();
  if (frame == nullptr) {
    GE_LOGE("The current frame does not exist, "
            "the function ValueHolder::PushGraphFrame should be called before construct the graph");
    return nullptr;
  }
  auto node = ValueHolder::AddNode(node_type, inputs.size(), out_count, *frame);
  GE_ASSERT_NOTNULL(node);

  /*
   * todo 检查是否有子图向父图连接的场景，这种场景需要报错
   *      父图向子图连接的场景，为父图节点创建一个InnerData
   */
  for (size_t i = 0U; i < inputs.size(); ++i) {
    GE_ASSERT_NOTNULL(inputs[i]);
    GE_ASSERT_NOTNULL(inputs[i]->fast_node_);
    GE_ASSERT_SUCCESS(AddDataEdge(inputs[i]->fast_node_, inputs[i]->index_, node, static_cast<int32_t>(i)));
    if (inputs[i]->guarder_ != nullptr && !IsGraphOutType(node_type)) {
      GE_ASSERT_SUCCESS(AddDependencyToGuarder(node, inputs[i]->guarder_->GetFastNode()));
    }
  }
  return node;
}

ValueHolderPtr ValueHolder::CreateMateFromNode(ge::FastNode *node, int32_t index, ValueHolderType type) {
  auto holder = std::shared_ptr<ValueHolder>(new (std::nothrow) ValueHolder());
  GE_ASSERT_NOTNULL(holder);

  holder->type_ = type;
  holder->fast_node_ = node;
  holder->index_ = index;
  holder->op_desc_ = holder->fast_node_->GetOpDescPtr();
  return holder;
}

void ValueHolder::SetErrorMsg(const char *fmt, va_list arg) {
  error_msg_ = std::unique_ptr<char[]>(CreateMessage(fmt, arg));
}

std::vector<ValueHolderPtr> ValueHolder::CreateDataOutput(const char *node_type,
                                                          const std::vector<ValueHolderPtr> &inputs,
                                                          size_t out_count) {
  auto node = CreateNode(node_type, inputs, out_count);
  if (node == nullptr) {
    return {out_count, nullptr};
  }
  return CreateFromNodeStart<ValueHolder>(node, out_count);
}

/**
 * @param data const数据
 * @param size const数据的长度
 * @param is_string 此const是否是个字符串, todo: 当前对string支持的不好
 * @return
 */
ValueHolderPtr ValueHolder::CreateConst(const void *data, size_t size, bool is_string) {
  GE_ASSERT_NOTNULL(data);
  auto node = ValueHolder::CreateNode(kConst, {}, 1U);
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_SUCCESS(op_desc->SetAttr("is_string", ge::AnyValue::CreateFrom(is_string)));
  GE_ASSERT_TRUE(ge::AttrUtils::SetZeroCopyBytes(op_desc, kConstValue,
                                                 ge::Buffer::CopyFrom(ge::PtrToPtr<void, uint8_t>(data), size)));
  return CreateFromNode<ValueHolder>(node, 0, ValueHolderType::kConst);
}

ValueHolderPtr ValueHolder::CreateFeed(int64_t index) {
  auto node = ValueHolder::CreateNode(kData, {}, 1U);
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE(ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), kFeedIndex, index));
  return CreateFromNode<ValueHolder>(node, 0, ValueHolderType::kFeed);
}

ValueHolderPtr ValueHolder::CreateConstData(int64_t index) {
  auto node = ValueHolder::CreateNode(kConstData, {}, 1U);
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE(ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), kFeedIndex, index));
  return CreateFromNode<ValueHolder>(node, 0, ValueHolderType::kConstData);
}

HyperStatus ValueHolder::AddInnerDataToKVMap(int32_t index) const noexcept {
  if (fast_node_->GetType() != kInnerData) {
    return HyperStatus::ErrorStatus("Failed to add node to KVMap, because node type is not InnerData.");
  }
  GELOGI("Set inner data: %s to kv map, index: %d", fast_node_->GetNamePtr(), index);
  AddKVToMapExtAttr<int32_t, ge::FastNode *>(fast_node_->GetExtendInfo()->GetOwnerGraphBarePtr(),
                                             kInnerDataNodes, index, fast_node_);
  return HyperStatus::Success();
}

ValueHolderPtr ValueHolder::CreateSingleDataOutput(const char *node_type, const std::vector<ValueHolderPtr> &inputs) {
  auto holders = CreateDataOutput(node_type, inputs, 1U);
  if (holders.empty()) {
    return nullptr;
  }
  return holders[0];
}

HyperStatus ValueHolder::AddDependency(const ValueHolderPtr &src, const ValueHolderPtr &dst) {
  if (src == nullptr || src->GetFastNode() == nullptr) {
    return HyperStatus::ErrorStatus("Failed to add control ege, because the src does not have a node.");
  }
  if (dst == nullptr || dst->GetFastNode() == nullptr) {
    return HyperStatus::ErrorStatus("Failed to add control ege, because the dst does not have a node.");
  }
  if (src->GetFastNode() == dst->GetFastNode()) {
    GELOGW("Add dependency between the same node %s, skip", src->GetFastNode()->GetNamePtr());
    return HyperStatus::Success();
  }
  return AddDependencyBetweenNodes(src->GetFastNode(), dst->GetFastNode());
}

ge::graphStatus ValueHolder::AppendInputs(const std::vector<ValueHolderPtr> &src) {
  const uint32_t start_index = fast_node_->GetDataInNum();
  GE_ASSERT_SUCCESS(
      ge::FastNodeUtils::AppendInputEdgeInfo(fast_node_, start_index + static_cast<uint32_t>(src.size())));
  const auto &dst = CreateFromNode<ValueHolder>(fast_node_, static_cast<size_t>(start_index), src.size());
  GE_ASSERT_TRUE(dst.size() == src.size());

  for (size_t i = 0U; i < src.size(); ++i) {
    GE_ASSERT_NOTNULL(src[i]);
    GE_ASSERT_NOTNULL(dst[i]);
    GE_ASSERT_SUCCESS(
        AddDataEdge(src[i]->fast_node_, src[i]->GetOutIndex(), dst[i]->fast_node_, dst[i]->GetOutIndex()));
  }

  return ge::SUCCESS;
}

GraphFrame *ValueHolder::PushGraphFrame() {
  if (!graph_frames.empty()) {
    GELOGE(ge::INTERNAL_ERROR,
           "Failed to push root graph frame, if you want to push a non-root graph frame, specify which ValueHolder the "
           "graph frame belongs and the ir name.");
    return nullptr;
  }
  auto graph = ge::MakeShared<ge::ExecuteGraph>("ROOT");
  GE_ASSERT_NOTNULL(graph);
  auto frame = new (std::nothrow) GraphFrame(graph);
  GE_ASSERT_NOTNULL(frame);
  return ValueHolder::PushGraphFrame(frame);
}

GraphFrame *ValueHolder::PushGraphFrame(const ValueHolderPtr &belongs, const char *graph_name) {
  GE_ASSERT_NOTNULL(belongs);
  GE_ASSERT_NOTNULL(belongs->GetFastNode());
  GE_ASSERT_NOTNULL(graph_name);
  if (graph_frames.empty()) {
    GELOGE(ge::INTERNAL_ERROR, "Failed to push a non-root graph frame, there is no root graph frames exists");
    return nullptr;
  }
  auto &parent_frame = *graph_frames.back();
  auto instance_name = GenerateNodeName(graph_name, parent_frame);
  auto graph = ge::MakeShared<ge::ExecuteGraph>(instance_name);
  GE_ASSERT_NOTNULL(graph);

  auto frame_holder = ge::ComGraphMakeUnique<GraphFrame>(graph, parent_frame);
  GE_ASSERT_NOTNULL(frame_holder);

  int64_t compute_node_index;
  if (ge::AttrUtils::GetInt(belongs->GetFastNode()->GetOpDescBarePtr(), kComputeNodeIndex, compute_node_index)) {
    auto compute_node = GetComputeNodeByIndex(*frame_holder.get(), static_cast<size_t>(compute_node_index));
    if (compute_node != nullptr) {
      frame_holder->SetCurrentComputeNode(compute_node);
    }
  }

  GE_ASSERT_SUCCESS(ge::FastNodeUtils::AppendSubgraphToNode(belongs->GetFastNode(), graph_name, graph));
  return ValueHolder::PushGraphFrame(frame_holder.release());
}

GraphFrame *ValueHolder::PushGraphFrame(GraphFrame *graph_frame) {
  GE_ASSERT_NOTNULL(graph_frame);
  if (!graph_frames.empty()) {
    GE_ASSERT_TRUE((graph_frames.back()->GetExecuteGraph().get() ==
                   graph_frame->GetExecuteGraph()->GetParentGraphBarePtr()),
                   "Last graph frame in stack %s is not parent graph frame of %s.",
                   graph_frames.back()->GetExecuteGraph()->GetName().c_str(),
                   graph_frame->GetExecuteGraph()->GetName().c_str());
  }
  graph_frames.emplace_back(graph_frame);
  return graph_frames.back().get();
}

std::unique_ptr<GraphFrame> ValueHolder::PopGraphFrame() {
  if (graph_frames.empty()) {
    return nullptr;
  }
  auto ret = std::move(graph_frames.back());
  graph_frames.pop_back();
  return ret;
}
GraphFrame *ValueHolder::GetCurrentFrame() {
  if (current_frame != nullptr) {
    return current_frame;
  }
  if (graph_frames.empty()) {
    return nullptr;
  }
  return graph_frames.back().get();
}
void ValueHolder::ClearGraphFrameResource() {
  graph_frames.clear();
  current_frame = nullptr;
}
void ValueHolder::SetCurrentComputeNode(const ge::NodePtr &node) {
  auto frame = GetCurrentFrame();
  if (frame == nullptr) {
    GELOGW("Ignore to add current compute node, the current frame is nullptr");
    return;
  }
  frame->SetCurrentComputeNode(node);
}
void ValueHolder::AddRelevantInputNode(const ge::NodePtr &node) {
  auto frame = GetCurrentFrame();
  if (frame == nullptr) {
    GELOGW("Ignore to add relevant input node, the current frame is nullptr");
  } else {
    frame->AddRelevantInputNode(node);
  }
}
std::unique_ptr<ValueHolder::CurrentComputeNodeGuarder> ValueHolder::SetScopedCurrentComputeNode(
    const ge::NodePtr &node) {
  auto frame = GetCurrentFrame();
  GE_ASSERT_NOTNULL(frame);

  auto guarder = ge::ComGraphMakeUnique<CurrentComputeNodeGuarder>(frame->GetCurrentComputeNode());
  GE_ASSERT_NOTNULL(guarder);
  frame->SetCurrentComputeNode(node);
  return guarder;
}

ge::ExecuteGraph *ValueHolder::GetCurrentExecuteGraph() {
  auto frame = GetCurrentFrame();
  GE_ASSERT_NOTNULL(frame);
  return frame->GetExecuteGraph().get();
}

ge::graphStatus ValueHolder::RefFrom(const ValueHolderPtr &other) {
  GE_ASSERT_NOTNULL(fast_node_);
  GE_ASSERT_NOTNULL(other);
  GE_ASSERT_NOTNULL(other->fast_node_);

  if (index_ < 0 || other->index_ < 0) {
    GELOGE(ge::PARAM_INVALID, "Invalid index to ref %d -> %d", index_, other->index_);
    return ge::PARAM_INVALID;
  }

  const auto op_desc = fast_node_->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto &td = op_desc->MutableOutputDesc(index_);
  GE_ASSERT_NOTNULL(td);

  GE_ASSERT_TRUE(ge::AttrUtils::SetStr(td, kRefFromNode, other->GetFastNode()->GetName()));
  GE_ASSERT_TRUE(ge::AttrUtils::SetInt(td, kRefFromIndex, other->index_));
  return ge::GRAPH_SUCCESS;
}

ValueHolderPtr ValueHolder::CreateVoidGuarder(const char *node_type, const ValueHolderPtr &resource,
                                              const std::vector<ValueHolderPtr> &args) {
  GE_ASSERT_NOTNULL(resource);
  std::vector<ValueHolderPtr> inputs;
  inputs.reserve(args.size() + 1);
  inputs.emplace_back(resource);
  inputs.insert(inputs.cend(), args.cbegin(), args.cend());
  auto ret = CreateVoid<ValueHolder>(node_type, inputs);
  GE_ASSERT_NOTNULL(ret);
  GE_ASSERT_NOTNULL(ret->GetFastNode());
  GE_ASSERT_TRUE(ge::AttrUtils::SetInt(ret->GetFastNode()->GetOpDescBarePtr(), kReleaseResourceIndex, 0));
  const auto resource_node = resource->GetFastNode();
  GE_ASSERT_NOTNULL(resource_node);
  GE_ASSERT_TRUE(ge::AttrUtils::SetStr(resource_node->GetOpDescBarePtr(), kGuarderNodeType, node_type));
  resource->SetGuarder(ret);
  return ret;
}

const int32_t &ValueHolder::GetPlacement() const {
  return placement_;
}
void ValueHolder::SetPlacement(const int32_t &placement) {
  placement_ = placement;
}
void ValueHolder::ReleaseAfter(const ValueHolderPtr &other) {
  if (guarder_ == nullptr) {
    GELOGW("Current holder from node %s index %d does not has a guarder", fast_node_->GetNamePtr(), index_);
    return;
  }
  AddDependency(other, guarder_);
}
std::unique_ptr<GraphFrame> ValueHolder::PopGraphFrame(const std::vector<ValueHolderPtr> &outputs,
                                                       const std::vector<ValueHolderPtr> &targets) {
  const char *node_type = kNetOutput;
  if (graph_frames.size() > 1U) {
    // The NetOutput type means "Network outputs", subgraph use InnerNetOutput as output type
    node_type = kInnerNetOutput;
  }
  return PopGraphFrame(outputs, targets, node_type);
}

std::unique_ptr<GraphFrame> ValueHolder::PopGraphFrame(const std::vector<ValueHolderPtr> &outputs,
                                                       const std::vector<ValueHolderPtr> &targets,
                                                       const char *out_node_type) {
  GE_ASSERT_NOTNULL(out_node_type);
  auto out_holder = CreateVoid<ValueHolder>(out_node_type, outputs);
  GE_ASSERT_NOTNULL(out_holder);
  if (strcmp(ge::NETOUTPUT, out_node_type) == 0) {
    // the name of NetOutput node must be `NetOutput`
    GE_ASSERT_NOTNULL(out_holder->GetFastNode());
    const auto op_desc = out_holder->GetFastNode()->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    op_desc->SetName(out_node_type);
  }

  for (const auto &target : targets) {
    AddDependency(target, out_holder);
  }
  return PopGraphFrame();
}

ValueHolderPtr ValueHolder::GetGuarder() const noexcept {
  return guarder_;
}

void ValueHolder::SetGuarder(const bg::ValueHolderPtr &guarder) noexcept {
  guarder_ = guarder;
}

void SetCurrentFrame(GraphFrame *frame) {
  current_frame = frame;
}
GraphFrame *GetCurrentFrame() {
  return current_frame;
}

std::vector<ValueHolderPtr> ValueHolder::GetLastExecNodes() {
  if (graph_frames.empty()) {
    return {};
  }
  auto frame = graph_frames.cbegin()->get();
  if (graph_frames.size() > 1U) {
    frame = (graph_frames.begin() + 1)->get();
  }
  return frame->GetLastExecNodes();
}
std::deque<std::unique_ptr<GraphFrame>> &GetGraphFrames() {
  return graph_frames;
}
}  // namespace bg
}  // namespace gert
