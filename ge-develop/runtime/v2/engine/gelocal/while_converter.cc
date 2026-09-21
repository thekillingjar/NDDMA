/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/ge_inner_attrs.h"
#include "engine/node_converter_utils.h"
#include "exe_graph/lowering/lowering_definitions.h"
#include "exe_graph/lowering/value_holder_utils.h"
#include "framework/common/types.h"
#include "graph_builder/bg_identity.h"
#include "graph_builder/converter_checker.h"
#include "graph_builder/multi_stream/bg_event.h"
#include "graph/utils/graph_dump_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "lowering/graph_converter.h"
#include "lowering/placement/placed_lowering_result.h"
#include "register/node_converter_registry.h"

namespace gert {
namespace {
constexpr const char *kLowerWhileBodyOutput = "_lower_while_body_output";
constexpr const char *kLowerWhileCondOutput = "_lower_while_cond_output";

/// @brief Get compute node lower inputs placed on specific placement by arg 'placement'
/// @param node Target compute node
/// @param placement Specific placement for collecting lower inputs
/// @param global_data Global data for lowering graph
/// @return LowerResult with lower-shape and type or LowerResult with error status if any error occurs
LowerResult GetNodeLowerInputPlacedOn(const ge::NodePtr &node, TensorPlacement placement,
                                      LoweringGlobalData &global_data) {
  LowerResult ret;
  for (const auto &node_and_anchor : node->GetInDataNodesAndAnchors()) {
    auto &in_node = node_and_anchor.first;
    auto output_index = node_and_anchor.second->GetIdx();
    auto placed_result = in_node->GetOpDescBarePtr()->GetExtAttr<PlacedLoweringResult>(kLoweringResult);
    LOWER_REQUIRE_NOTNULL(placed_result, "The in node %s of net-output %s have no placed lowering result",
                          in_node->GetNamePtr(), node->GetNamePtr());

    const auto *result = placed_result->GetOutputResult(global_data, output_index, {placement, bg::kMainStream}, false);
    LOWER_REQUIRE_NOTNULL(result);
    LOWER_REQUIRE(result->has_init);

    ret.out_addrs.emplace_back(result->address);
    ret.out_shapes.emplace_back(result->shape);
  }
  return ret;
}

template <typename T>
bool IsNotAliasOrOrdered(const std::vector<T> &targets, const std::vector<T> &ordered, size_t index) {
  for (const auto &target : targets) {
    for (size_t i = 0U; i < ordered.size(); i++) {
      if ((target != nullptr) && (target == ordered[i]) && (index != i)) {
        GELOGI("While node input %zu flow to mismatch-ordered %zu", index, i);
        return false;
      }
    }
  }
  return true;
}

bool IsWhileBodyCanInplace(const ge::NodePtr &net_output, const LowerInput &lower_input) {
  auto body = net_output->GetOwnerComputeGraphBarePtr();
  GE_ASSERT_NOTNULL(body);

  for (const auto &node : body->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    if (node->GetType() != ge::DATA) {
      continue;
    }

    auto data_node = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(data_node);
    size_t index = 0U;
    GE_ASSERT(ge::AttrUtils::GetInt(data_node, ge::ATTR_NAME_PARENT_NODE_INDEX, index));
    auto placed_lower_result = data_node->GetExtAttr<PlacedLoweringResult>(kLoweringResult);
    GE_ASSERT_NOTNULL(placed_lower_result);
    auto lower_result = placed_lower_result->GetResult();
    GE_ASSERT_NOTNULL(lower_result);
    if (!IsNotAliasOrOrdered(lower_result->out_addrs, lower_input.input_addrs, index)) {
      return false;
    }
    if (!IsNotAliasOrOrdered(lower_result->out_shapes, lower_input.input_shapes, index)) {
      return false;
    }
  }

  return true;
}

// Lower func for net-output node of while body graph.
// This function guarantee the lower-results placed on DEVICE
// add a 'NoOp' as a barrier for update inputs for next loop, the barrier is needed as output shape may
// update some-body-op-infer-shape-inputs accidentally as output shared memory with them
LowerResult LoweringWhileBodyOutput(const ge::NodePtr &node, const LowerInput &lower_input) {
  GELOGI("Start lowering while body output %s", node->GetNamePtr());
  auto device_inputs = GetNodeLowerInputPlacedOn(node, kOnDeviceHbm, *lower_input.global_data);
  LOWER_RETURN_IF_ERROR(device_inputs);

  LOWER_REQUIRE_HYPER_SUCCESS(LoweringAccessMemCrossStream(node, device_inputs.out_addrs));

  auto &out_shapes = device_inputs.out_shapes;
  auto &out_addrs = device_inputs.out_addrs;
  if (!IsWhileBodyCanInplace(node, lower_input)) {
    GELOGI("Add identity as barrier for update inputs for next loop as %s body can not inplace", node->GetNamePtr());
    out_shapes = bg::IdentityShape(out_shapes);
    out_addrs = bg::IdentityAddr(out_addrs, node->GetOpDescBarePtr()->GetStreamId());
    for (auto &out_addr : out_addrs) {
      bg::ValueHolder::CreateVoidGuarder("FreeMemory", out_addr, {});
    }
  }

  auto shape_and_addrs = bg::IdentityShapeAndAddr(out_shapes, out_addrs, node->GetOpDescBarePtr()->GetStreamId());
  LOWER_REQUIRE_VALID_HOLDER(shape_and_addrs);
  size_t num_shapes = shape_and_addrs.size() >> 1U;

  LowerResult ret;
  ret.out_shapes.insert(ret.out_shapes.end(), shape_and_addrs.begin(), shape_and_addrs.begin() + num_shapes);
  ret.out_addrs.insert(ret.out_addrs.end(), shape_and_addrs.begin() + num_shapes, shape_and_addrs.end());
  ret.order_holders.push_back(shape_and_addrs.front());

  return ret;
}

// Lower func for net-output node of while cond graph.
// This function guarantee the lower-results placed on HOST
LowerResult LoweringWhileCondOutput(const ge::NodePtr &node, const LowerInput &lower_input) {
  GELOGI("Start lowering while cond output %s", node->GetNamePtr());
  return GetNodeLowerInputPlacedOn(node, kOnHost, *lower_input.global_data);
}

REGISTER_NODE_CONVERTER(kLowerWhileBodyOutput, LoweringWhileBodyOutput);
REGISTER_NODE_CONVERTER(kLowerWhileCondOutput, LoweringWhileCondOutput);

/// @brief Get instance graph name of graph-type node attr
/// @param node Target node
/// @param attr Graph-type attr name
/// @return Instance graph name or empty string if any error occurs
std::string GetGraphTypeAttrInstanceGraphName(const ge::NodePtr &node, const char *attr) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto &graph_attr_index = op_desc->GetSubgraphNameIndexes();
  const std::map<std::string, uint32_t>::const_iterator iter = graph_attr_index.find(attr);
  if (iter != graph_attr_index.cend()) {
    return op_desc->GetSubgraphInstanceName(iter->second);
  }
  GELOGW("Node %s has no graph attr named %s", node->GetNamePtr(), attr);
  return op_desc->GetSubgraphInstanceName((std::string(attr) == "cond") ? 0U : 1U);
}

/// @brief Get instance graph of graph-type node attr
/// @param node Target node
/// @param attr Graph-type attr name
/// @return Instance graph or nullptr if any error occurs
ge::ComputeGraphPtr GetAttrSubgraph(const ge::NodePtr &node, const char *attr) {
  // Lookup instance graph by graph-type attr value
  auto attr_graph_name = GetGraphTypeAttrInstanceGraphName(node, attr);
  GE_ASSERT(!attr_graph_name.empty(), "Empty instance graph name for attr %s", attr);
  auto root_compute_graph = ge::GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
  GE_ASSERT_NOTNULL(root_compute_graph);
  auto attr_graph = root_compute_graph->GetSubgraph(attr_graph_name);
  GE_ASSERT_NOTNULL(attr_graph, "Root graph %s has no subgraph named %s", root_compute_graph->GetName().c_str(),
                    attr_graph_name.c_str());
  return attr_graph;
}

/// @brief Create a single lower while node
/// @param node Compute while node to be lowering
/// @param lower_input Lowering inputs
/// @param lower_node Created lower while node
/// @return Lower while node outputs or error status if any error occurs
LowerResult CreateLowerWhileNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  std::vector<bg::ValueHolderPtr> inputs;
  size_t num_tensors = lower_input.input_addrs.size();
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  if (!lower_input.input_addrs.empty()) {
    // Identity shape and addr as while input maybe also inputs of other ops
    auto identified_shapes = bg::IdentityShape(lower_input.input_shapes);
    LOWER_REQUIRE_VALID_HOLDER(identified_shapes);
    const auto &identified_addrs = bg::IdentityAddr(lower_input.input_addrs, op_desc->GetStreamId());
    LOWER_REQUIRE_VALID_HOLDER(identified_addrs);
    inputs.insert(inputs.cend(), identified_shapes.cbegin(), identified_shapes.cend());
    inputs.insert(inputs.cend(), identified_addrs.cbegin(), identified_addrs.cend());
  }

  LowerResult ret;
  if (node->GetAllOutDataAnchorsSize() == 0U) {  // Stateful while node which has no outputs
    ret.order_holders.push_back(bg::ValueHolder::CreateVoid<bg::ValueHolder>(node->GetTypePtr(), inputs));
    LOWER_REQUIRE_VALID_HOLDER(ret.order_holders);
    return ret;
  }
  size_t num_outputs = node->GetAllOutDataAnchorsSize() << 1U;
  auto holders = bg::DevMemValueHolder::CreateDataOutput(node->GetTypePtr(), inputs, num_outputs,
                                                         op_desc->GetStreamId());
  LOWER_REQUIRE_VALID_HOLDER(holders);
  ret.order_holders.push_back(holders.front());
  ret.out_shapes.insert(ret.out_shapes.cend(), holders.cbegin(), holders.cbegin() + node->GetAllOutDataAnchorsSize());
  ret.out_addrs.insert(ret.out_addrs.cend(), holders.cbegin() + node->GetAllOutDataAnchorsSize(), holders.cend());

  for (size_t i = 0U; i < ret.out_shapes.size(); i++) {
    LOWER_REQUIRE(i + num_tensors < inputs.size());
    ret.out_shapes[i]->RefFrom(inputs[i]);
    ret.out_addrs[i]->RefFrom(inputs[i + num_tensors]);
    bg::ValueHolder::CreateVoidGuarder("FreeMemory", ret.out_addrs[i], {});
  }

  return ret;
}

/// @brief Build control frame for subgraph-call node
/// @param subgraph_call target subgraph-call node to build control frame
/// @return Built control frame graph or nullptr if any error occurs
bool BuildGraphCallControlFrame(const bg::ValueHolderPtr &subgraph_call) {
  GE_ASSERT_NOTNULL(bg::ValueHolder::PushGraphFrame(subgraph_call, "control_frame"));
  auto pivot = bg::ValueHolder::CreateVoid<bg::ValueHolder>("BranchPivot", {});
  GE_ASSERT(IsValidHolder(pivot));
  auto done = bg::ValueHolder::CreateVoid<bg::ValueHolder>("BranchDone", {});
  GE_ASSERT(IsValidHolder(done));
  bg::ValueHolder::AddDependency(pivot, done);
  auto frame = bg::ValueHolder::PopGraphFrame();
  GE_ASSERT_NOTNULL(frame);
  return frame->GetExecuteGraph() != nullptr;
}

/// @brief Build a subgraph-call node for given graph
/// @param graph Callee graph of the subgraph-call node
/// @param inputs Inputs of the subgraph-call node in building
/// @param global_data Global data for lowering graph
/// @param outputs Output value-holders of the built subgraph-call node
/// @return GRAPH_SUCCESS or GRAPH_FAILED if any error occurs
ge::graphStatus BuildGraphCall(const ge::ComputeGraphPtr &graph, const std::vector<bg::ValueHolderPtr> &inputs,
                               LoweringGlobalData &global_data, std::vector<bg::ValueHolderPtr> &outputs) {
  auto net_output = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  GE_ASSERT_NOTNULL(net_output, "Failed build graph call for %s as has no net-output node", graph->GetName().c_str());
  size_t num_compute_tensors = net_output->GetAllInDataAnchorsSize();

  outputs = bg::ValueHolder::CreateDataOutput("SubgraphCall", inputs, num_compute_tensors << 1U);
  GE_ASSERT(IsValidHolder(outputs));
  GE_ASSERT_TRUE(!outputs.empty());
  GE_ASSERT_TRUE(BuildGraphCallControlFrame(outputs.front()));

  GE_ASSERT_NOTNULL(bg::ValueHolder::PushGraphFrame(outputs.front(), "f"));

  auto lower_result = ConvertComputeSubgraphToExecuteGraph(graph, global_data, 0);
  GE_ASSERT_NOTNULL(lower_result, "Failed lower cond graph %s", graph->GetName().c_str());
  GE_ASSERT(lower_result->result.IsSuccess());

  GE_ASSERT_EQ(lower_result->out_addrs.size(), 1U);
  GE_ASSERT_NOTNULL(net_output->GetOpDescBarePtr());
  const auto &input_desc = net_output->GetOpDescBarePtr()->GetInputDescPtr(0U);
  GE_ASSERT_NOTNULL(input_desc);
  auto dt = input_desc->GetDataType();
  auto data_type = bg::ValueHolder::CreateConst(&dt, sizeof(dt));
  GE_ASSERT(IsValidHolder(data_type));
  auto cond_output = bg::ValueHolder::CreateSingleDataOutput(
      "GenCondForWhile", {lower_result->out_addrs[0U], lower_result->out_shapes[0U], data_type});
  auto graph_frame = bg::ValueHolder::PopGraphFrame({cond_output}, {});
  GE_ASSERT_NOTNULL(graph_frame);
  ge::DumpGraph(graph_frame->GetExecuteGraph().get(), "LowerWhile_CondGraph");
  return ge::GRAPH_SUCCESS;
}

/// @brief Build an subgraph-call node for subgraph 'cond' of the while node
/// @param while_node Target while node
/// @param inputs Inputs of the subgraph-call node in building
/// @param global_data Global data for lowering graph
/// @param outputs Output value-holders of the built subgraph-call node
/// @return GRAPH_SUCCESS or GRAPH_FAILED if any error occurs
ge::graphStatus BuildCondGraphCall(const ge::NodePtr &while_node, const std::vector<bg::ValueHolderPtr> &inputs,
                                   LoweringGlobalData &global_data, std::vector<bg::ValueHolderPtr> &outputs) {
  auto graph = GetAttrSubgraph(while_node, "cond");
  GE_ASSERT_NOTNULL(graph);

  auto net_output = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  GE_ASSERT_NOTNULL(net_output, "Cond graph %s has no net-output node", graph->GetName().c_str());
  ge::AttrUtils::SetStr(net_output->GetOpDescBarePtr(), "_ge_attr_lowering_func", kLowerWhileCondOutput);

  return BuildGraphCall(graph, inputs, global_data, outputs);
}

/// @brief Lower the 'body' subgraph of given while node. This func guarantee the lower outputs placed on kDeviceHbm
/// @param while_node Target while node
/// @param lower_while Lower-ed while node which is the parent of the lower-ed 'body' graph
/// @param global_data Global data for lowering graph
/// @return GRAPH_SUCCESS or GRAPH_FAILED if any error occurs
ge::graphStatus LowerBodyGraph(const ge::NodePtr &while_node, const bg::ValueHolderPtr &lower_while,
                               LoweringGlobalData &global_data) {
  GE_ASSERT_NOTNULL(bg::ValueHolder::PushGraphFrame(lower_while, "body"));

  auto body_graph = GetAttrSubgraph(while_node, "body");
  GE_ASSERT_NOTNULL(body_graph);
  auto net_output = body_graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  GE_ASSERT_NOTNULL(net_output, "Body graph %s has no net-output node", body_graph->GetName().c_str());
  (void)ge::AttrUtils::SetStr(net_output->GetOpDescBarePtr(), "_ge_attr_lowering_func", kLowerWhileBodyOutput);

  auto lower_body_result = ConvertComputeSubgraphToExecuteGraph(body_graph, global_data, 0);
  GE_ASSERT_NOTNULL(lower_body_result, "Failed lower body graph %s", body_graph->GetName().c_str());
  GE_ASSERT(lower_body_result->result.IsSuccess());

  std::vector<bg::ValueHolderPtr> body_outputs = lower_body_result->out_shapes;
  body_outputs.insert(body_outputs.cend(), lower_body_result->out_addrs.cbegin(), lower_body_result->out_addrs.cend());
  auto body_graph_frame = bg::ValueHolder::PopGraphFrame(body_outputs, {});
  GE_ASSERT_NOTNULL(body_graph_frame);

  ge::DumpGraph(body_graph_frame->GetExecuteGraph().get(), "LowerWhile_BodyGraph");
  return ge::GRAPH_SUCCESS;
}

/// @brief Construct the control frame graph, control frame graph is the lowering represent of control logic of while op
///                 +------------------------------------------------------------------------------------+
///                 v                                                                                    |
/// +-------+     +------------+     +--------------+     +------------------+     +-------------+     +------------+
/// | Enter | --> | WaitAnyOne | --> | SubgraphCall | --> | CondSwitchNotify | --> | BranchPivot | --> | BranchDone |
/// +-------+     +------------+     +--------------+     +------------------+     +-------------+     +------------+
///                                                         |
///                                                         |
///                                                         v
///                                                       +--------------+
///                                                       |     Exit     |
///                                                       +--------------+
/// @param while_node The compute while node in lowering
/// @param lower_node The in-complete lower node of compute while node, with no subgraph added
/// @param global_data Lowering global data
/// @return The constructed control frame graph or nullptr if any error occurs
ge::graphStatus BuildControlFrame(const ge::NodePtr &while_node, const bg::ValueHolderPtr &lower_while,
                                  LoweringGlobalData &global_data) {
  GE_ASSERT_NOTNULL(bg::ValueHolder::PushGraphFrame(lower_while, "control_frame"));
  // frame graph inputs x2 as compute tensor split to addr + shape in lower representation
  size_t num_inputs = (while_node->GetAllInDataAnchorsSize() << 1U);
  std::vector<bg::ValueHolderPtr> inputs;
  inputs.reserve(num_inputs);
  for (size_t i = 0U; i < num_inputs; i++) {
    inputs.emplace_back(bg::ValueHolder::CreateSingleDataOutput("InnerData", {}));
    GE_ASSERT(IsValidHolder(inputs.back()));
    GE_ASSERT(ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(inputs.back()), "index", i));
  }

  auto enter = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Enter", {});
  auto wait_anyone = bg::ValueHolder::CreateVoid<bg::ValueHolder>("WaitAnyone", {});
  bg::ValueHolder::AddDependency(enter, wait_anyone);

  // Construct SubgraphCall node which is caller of cond graph
  std::vector<bg::ValueHolderPtr> cond_outputs;
  GE_ASSERT_GRAPH_SUCCESS(BuildCondGraphCall(while_node, inputs, global_data, cond_outputs));
  bg::ValueHolder::AddDependency(wait_anyone, cond_outputs.front());

  const size_t kNotifyOutputNum = 3U;  // 0:placeholder 1:body 2:exit
  auto switch_notifies = bg::ValueHolder::CreateDataOutput("CondSwitchNotify", cond_outputs, kNotifyOutputNum);
  GE_ASSERT(IsValidHolder(switch_notifies));
  // Node has attr ge::kRequestWatcher can get watcher when create or init outputs
  GE_ASSERT(ge::AttrUtils::SetBool(bg::ValueHolderUtils::GetNodeOpDescBarePtr(switch_notifies[0]), ge::kRequestWatcher,
                                   true));
  auto watcher_placeholder = bg::ValueHolder::CreateVoid<bg::ValueHolder>("WatcherPlaceholder", {switch_notifies[0]});
  GE_ASSERT(IsValidHolder(watcher_placeholder));

  // No-op is required for keep body graph call a data output of switch notify
  auto body_pivot = bg::ValueHolder::CreateVoid<bg::ValueHolder>("BranchPivot", {switch_notifies[1]});
  auto body_done = bg::ValueHolder::CreateVoid<bg::ValueHolder>("BranchDone", {});
  GE_ASSERT_HYPER_SUCCESS(bg::ValueHolder::AddDependency(body_pivot, body_done));

  auto exit = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Exit", {switch_notifies[2]});
  GE_ASSERT(IsValidHolder(exit));

  bg::ValueHolder::PopGraphFrame();
  return ge::GRAPH_SUCCESS;
}
}  // namespace

/// @brief Lower function for while node
/// The 'identity' before 'lower-while' is needed as 'Inputs' may be also inputs of other node, and
/// 'lower-while' may update 'Inputs' accidentally before other node consume it as
/// 'lower-while' share inputs memory with outputs
///                                                     +-------------+
///                                        ............ | Body graph  |
///                                        :            +-------------+
/// +--------+     +----------+     +-------------+     +---------------------+
/// | Inputs | --> | Identity | --> | Lower while | ... | Control frame graph |
/// +--------+     +----------+     +-------------+     +---------------------+
/// @param node The compute while node to be lowering
/// @param lower_input Lowering input
/// @return LowerResult contain output holders or error status if any error occurs
LowerResult LoweringWhile(const ge::NodePtr &node, const LowerInput &lower_input) {
  // Keep lower while a v2 style single node
  LowerResult ret = CreateLowerWhileNode(node, lower_input);
  LOWER_RETURN_IF_ERROR(ret);

  auto &lower_while = ret.order_holders.empty() ? ret.out_shapes.front() : ret.order_holders.front();
  // Construct control frame graph
  LOWER_REQUIRE_GRAPH_SUCCESS(BuildControlFrame(node, lower_while, *lower_input.global_data));
  // Construct body graph
  LOWER_REQUIRE_GRAPH_SUCCESS(LowerBodyGraph(node, lower_while, *lower_input.global_data));

  return ret;
}

// todo: 这里的处理不太合理，while不应该声明自己的placement，此处声明了placement位于DeviceHbm，是不对的
//   正式方案的关键点：
//   1. while不声明输入的placement
//   2. 在lowering while的子图时，Data产生的placement应该为Unknown；
//   3. lowering if子图时，Data产生的placement与if时机输入的placement相同  -- 当前有bug，在if的输入为host时，可能漏了h2d操作
//   4. 创建ValueHolder时，默认的placement应该位于HostDDR
//   5. 添加消除pass，在while的body输出与body输入、cond输入的MakeSureAt的placement相同时，可以将MakeSureAt外提到While外部，
//      进一步如果While外部输入的placement与MakeSureAt的placement相同时，可以直接删除MakeSureAt
REGISTER_NODE_CONVERTER_PLACEMENT("While", 0, LoweringWhile);
REGISTER_NODE_CONVERTER_PLACEMENT("StatelessWhile", 0, LoweringWhile);
}  // namespace gert
