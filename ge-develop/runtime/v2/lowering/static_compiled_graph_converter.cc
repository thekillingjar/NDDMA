/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <queue>
#include "graph_converter.h"
#include "node_converter_utils.h"
#include "register/node_converter_registry.h"
#include "graph_builder/assign_device_mem.h"
#include "graph/utils/node_utils.h"
#include "common/checker.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/graph_utils.h"
#include "common/types.h"
#include "graph_builder/converter_checker.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/model_manager/davinci_model.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/model/ge_model.h"
#include "common/memory/mem_type_utils.h"
#include "framework/common/ge_types.h"
#include "graph_builder/bg_memory.h"
#include "graph_builder/bg_rt_session.h"
#include "graph_builder/bg_davinci_model_finalizer.h"
#include "graph_builder/bg_reusable_stream_allocator.h"
#include "lowering/placement/placed_lowering_result.h"
#include "exe_graph/lowering/lowering_definitions.h"
#include "exe_graph/lowering/value_holder_utils.h"
#include "static_model_output_allocator.h"
#include "kernel/memory/sink_weight_assigner.h"
#include "engine/aicpu/converter/aicpu_callback.h"
#include "graph_builder/bg_model_desc.h"
#include "graph/unfold/graph_unfolder.h"
#include "graph/ge_context.h"
#include "common/model/ge_root_model.h"
#include "graph/manager/graph_var_manager.h"
#include "graph_builder/bg_condition.h"

namespace gert {
namespace {
constexpr size_t kLogLengthLimit = 800U;
constexpr uint64_t kSessionScopeMemory = 0x100000000U;
constexpr uint32_t kFlattenOffsetKey = 0U;
constexpr uint32_t kFlattenSizeKey = 1U;
constexpr const char_t *kStaticModelAddrFixed = "ge.exec.static_model_addr_fixed";
constexpr const char_t *kRootGraphFrozenInput = "ge.exec.frozenInputIndexes";
constexpr size_t kHbmFixedAddrIndex = 0UL;
constexpr size_t kHbmFixedSizeIndex = 1UL;
constexpr size_t kHbmFixedTensorDataIndex = 2UL;
constexpr size_t kP2pFixedAddrIndex = 3UL;
constexpr size_t kP2pFixedSizeIndex = 4UL;
constexpr size_t kP2pFixedTensorDataIndex = 5UL;

const std::unordered_map<uint64_t, TensorPlacement> kRtMemTypeToPlacement = {{RT_MEMORY_HBM, kOnDeviceHbm},
                                                                             {RT_MEMORY_P2P_DDR, kOnDeviceP2p},
                                                                             {RT_MEMORY_DEFAULT, kOnDeviceHbm}};

TensorPlacement GetPlacement(uint64_t memory_type) {
  if ((memory_type & kSessionScopeMemory) == kSessionScopeMemory) {
    GELOGI("transform rt memory type[0x%lx] with session scope", memory_type);
    memory_type &= ~kSessionScopeMemory;
  }
  auto iter = kRtMemTypeToPlacement.find(memory_type);
  if (iter != kRtMemTypeToPlacement.cend()) {
    GELOGI("transform rt memory type[0x%lx] to memory placement[%u]", memory_type, iter->second);
    return iter->second;
  }
  GELOGE(ge::FAILED, "transform rt memory type[0x%lx] to memory placement failed", memory_type);
  return kTensorPlacementEnd;
}

void GetFixedFeatureMemory(LoweringGlobalData &global_data, const rtMemType_t mem_type, const void *&addr,
                           size_t &size) {
  addr = nullptr;
  size = 0UL;
  const auto &all_fixed_feature_mem = global_data.GetAllTypeFixedFeatureMemoryBase();
  const auto &iter = all_fixed_feature_mem.find(mem_type);
  if (iter != all_fixed_feature_mem.end()) {
    addr = iter->second.first;
    size = iter->second.second;
  }
}

bool IsUseHbmFixedFeatureMemory(LoweringGlobalData &global_data) {
  const void *user_set_fixed_addr = nullptr;
  size_t fixed_size = 0U;
  GetFixedFeatureMemory(global_data, RT_MEMORY_HBM, user_set_fixed_addr, fixed_size);
  return (user_set_fixed_addr != nullptr)
         || kernel::IsNeedMallocFixedMemoryOnInitGraph(user_set_fixed_addr, fixed_size);
}

/*
 * 1 如果用户设置了正常的fix地址，Ge不再申请
 * 2 如果用户将fix地址设置为nullptr，size也设置为0，Ge不申请fix内存
 * 3 如果用户没有设置fix地址，Ge申请fix地址
 */
ge::Status MallocFixedFeatureMemOnInitRootIfNeed(LoweringGlobalData &global_data,
                                                 const rtMemType_t mem_type,
                                                 bg::ValueHolderPtr &fixed_mem,
                                                 bg::ValueHolderPtr &fixed_mem_size,
                                                 bg::ValueHolderPtr &fixed_tensor_data) {
  const void *user_set_fixed_addr = nullptr;
  size_t fixed_size = 0U;
  GetFixedFeatureMemory(global_data, mem_type, user_set_fixed_addr, fixed_size);
  fixed_mem = bg::ValueHolder::CreateConst(&user_set_fixed_addr, sizeof(uintptr_t));
  fixed_mem_size = bg::ValueHolder::CreateConst(&fixed_size, sizeof(size_t));
  GE_ASSERT_NOTNULL(fixed_mem);
  GE_ASSERT_NOTNULL(fixed_mem_size);
  const auto placement = GetPlacement(mem_type);

  // 没配置动静态图复用，存在这个类型的fixed feature memory，用户没有设置
  auto builder = [&global_data, &fixed_mem_size, &placement]() -> std::vector<bg::ValueHolderPtr> {
    auto init_outputs = bg::FrameSelector::OnInitRoot(
        [&global_data, &fixed_mem_size, &placement]() -> std::vector<bg::ValueHolderPtr> {
          const auto memory_holder = bg::AllocMem(placement, fixed_mem_size, global_data, bg::kMainStream);
          return {memory_holder};
        });
    return init_outputs;
  };

  // 配置动静态图复用，使用session级allocator申请fix优先内存，如果用户注册了外置allocator，使用用户注册的allocator申请
  const auto session_id_holder = bg::HolderOnInit(bg::GetSessionId(global_data));
  auto session_mem_builder = [&session_id_holder, &global_data, &fixed_mem_size, &placement]()
      -> std::vector<bg::ValueHolderPtr> {
    const auto memory_holder = bg::AllocFixedFeatureMemory(session_id_holder, placement, fixed_mem_size, global_data);
    return {memory_holder};
  };

  // 只创建一个TensorData，不申请内存
  auto empty_builder = []() -> std::vector<bg::ValueHolderPtr> {
    auto init_outputs = bg::FrameSelector::OnInitRoot(
        []() -> std::vector<bg::ValueHolderPtr> {
          const auto memory_holder = bg::ValueHolder::CreateSingleDataOutput("EmptyTensorData", {});
          return {memory_holder};
        });
    return init_outputs;
  };
  if (kernel::IsNeedMallocFixedMemoryOnInitGraph(user_set_fixed_addr, fixed_size)) {
    if (ge::VarManager::IsGeUseExtendSizeMemoryFull()) {
      auto init_outputs = global_data.GetOrCreateUniqueValueHolder("SessionFixedFeatureMemory_"
                                                                   + std::to_string(mem_type),
                                                                   session_mem_builder);
      GE_ASSERT_TRUE(!init_outputs.empty());
      fixed_tensor_data = init_outputs[0];
      GELOGI("need to malloc fixed_feature_memory base by user allocator or session fixed base allocator. type: %s,"
          " addr: %p, size: %zu", ge::MemTypeUtils::ToString(mem_type).c_str(), user_set_fixed_addr, fixed_size);
    } else {
      auto init_outputs = global_data.GetOrCreateUniqueValueHolder("FixedFeatureMemory_" + std::to_string(mem_type),
                                                                   builder);
      GE_ASSERT_TRUE(!init_outputs.empty());
      fixed_tensor_data = init_outputs[0];
      GELOGI("need to malloc fixed_feature_memory base by user allocator or inner allocator. type: %s, addr: %p,"
          " size: %zu", ge::MemTypeUtils::ToString(mem_type).c_str(), user_set_fixed_addr, fixed_size);
    }
  } else {
    auto init_outputs = global_data.GetOrCreateUniqueValueHolder("EmptyFixedFeatureMemory_"
                                                                     + std::to_string(mem_type), empty_builder);
    GE_ASSERT_TRUE(!init_outputs.empty());
    fixed_tensor_data = init_outputs[0];
    GELOGI("no need to malloc fixed_feature_memory base. type: %s, addr: %p, size: %zu",
           ge::MemTypeUtils::ToString(mem_type).c_str(), user_set_fixed_addr, fixed_size);
  }
  return ge::SUCCESS;
}

ge::Status MallocFixedFeatureMemOnInitRootIfNeed(LoweringGlobalData &global_data,
                                       std::vector<bg::ValueHolderPtr> &all_fixed_mem_holder) {
  all_fixed_mem_holder.resize(kP2pFixedTensorDataIndex + 1UL, nullptr);
  // RT_MEMORY_HBM fixed feature memory
  GE_ASSERT_SUCCESS(MallocFixedFeatureMemOnInitRootIfNeed(global_data, RT_MEMORY_HBM,
                                                          all_fixed_mem_holder[kHbmFixedAddrIndex],
                                                          all_fixed_mem_holder[kHbmFixedSizeIndex],
                                                          all_fixed_mem_holder[kHbmFixedTensorDataIndex]));

  // RT_MEMORY_P2P_DDR fixed feature memory
  GE_ASSERT_SUCCESS(MallocFixedFeatureMemOnInitRootIfNeed(global_data, RT_MEMORY_P2P_DDR,
                                                          all_fixed_mem_holder[kP2pFixedAddrIndex],
                                                          all_fixed_mem_holder[kP2pFixedSizeIndex],
                                                          all_fixed_mem_holder[kP2pFixedTensorDataIndex]));
  return ge::SUCCESS;
}

bg::ValueHolderPtr FileConstantMemToContinuousVecHolder(LoweringGlobalData &global_data) {
  std::vector<gert::kernel::FileConstantNameAndMem> name_and_mems;
  for (const auto &item : global_data.GetAllFileConstantMems()) {
    gert::kernel::FileConstantNameAndMem name_and_mem{};
    const auto ret = strcpy_s(name_and_mem.name, sizeof(name_and_mem.name), item.first.c_str());
    GE_ASSERT_TRUE(ret == EOK, "strcpy_s failed, copy length: %zu, ret: %d", item.first.length(), ret);
    name_and_mem.mem = item.second.device_mem;
    name_and_mem.size = item.second.mem_size;
    name_and_mems.emplace_back(std::move(name_and_mem));
  }
  GELOGI("FileConstant memory vector size: %zu", name_and_mems.size());
  return bg::CreateContVecHolder(name_and_mems);
}

std::vector<bg::ValueHolderPtr> CreateDavinciModelOnInitRoot(const ge::ComputeGraphPtr &graph, ge::GeModel *ge_model,
                                                             LoweringGlobalData &global_data) {
  // construct DavinciModel Init weight info
  std::vector<int64_t> flatten_weight = GetFlattenOffsetInfo(ge_model);
  GE_ASSERT_TRUE(flatten_weight.size() == kFlattenKeySize);
  TensorData weight_tensor;
  weight_tensor.SetAddr(ge::ValueToPtr(flatten_weight[kFlattenOffsetKey]), nullptr);
  weight_tensor.SetSize(flatten_weight[kFlattenSizeKey]);
  auto weight_info_holder = bg::ValueHolder::CreateConst(&weight_tensor, sizeof(TensorData));
  int64_t logic_stream_id = 0;
  auto stream_id_holder = bg::ValueHolder::CreateConst(&logic_stream_id, sizeof(logic_stream_id));
  GE_ASSERT_TRUE(IsValidHolder(stream_id_holder));
  auto assign_mem_holder = bg::ValueHolder::CreateSingleDataOutput(
      kernel::kAssignWeightMemory,
      {weight_info_holder, bg::HolderOnInit(AssignDeviceMem::GetOrCreateMemAssigner(global_data)), stream_id_holder});

  auto model_holder = bg::ValueHolder::CreateConst(&ge_model, sizeof(uintptr_t));
  GE_ASSERT(IsValidHolder(model_holder));
  // todo rt_session不作为init的输出，而是提供方法获取init图里的节点。这样子DavinciModelCreate可以在init图中
  const auto &session_id = bg::HolderOnInit(bg::GetSessionId(global_data));
  const auto &step_id = bg::HolderOnInit(GetStepId(global_data));
  const auto &root_graph = ge::GraphUtils::FindRootGraph(graph);
  GE_ASSERT_NOTNULL(root_graph);
  const uint32_t root_graph_id = root_graph->GetGraphID();
  const auto root_graph_id_holder = bg::ValueHolder::CreateConst(&root_graph_id, sizeof(uint32_t));
  std::string is_addr_fixed_opt;
  (void)ge::GetContext().GetOption(kStaticModelAddrFixed, is_addr_fixed_opt);
  std::string frozen_input;
  (void)ge::GetContext().GetOption(kRootGraphFrozenInput, frozen_input);
  const auto frozen_holder = bg::ValueHolder::CreateConst(frozen_input.c_str(), frozen_input.size() + 1UL, true);
  std::vector<bg::ValueHolderPtr> fixed_holder;
  GE_ASSERT_SUCCESS(MallocFixedFeatureMemOnInitRootIfNeed(global_data, fixed_holder));
  const auto file_constant_mem_holder = FileConstantMemToContinuousVecHolder(global_data);
  if (is_addr_fixed_opt.empty()) {
    return bg::ValueHolder::CreateDataOutput(
        "DavinciModelCreate",
        {model_holder, session_id, assign_mem_holder, step_id, root_graph_id_holder,
         bg::HolderOnInit(bg::GetSpaceRegistries(global_data)),
         bg::HolderOnInit(bg::GetFileConstantWeightDir(global_data)),
         bg::HolderOnInit(bg::ReusableStreamAllocator(global_data)), fixed_holder[kHbmFixedAddrIndex],
         fixed_holder[kHbmFixedSizeIndex], bg::HolderOnInit(fixed_holder[kHbmFixedTensorDataIndex]),
         fixed_holder[kP2pFixedAddrIndex], fixed_holder[kP2pFixedSizeIndex],
         bg::HolderOnInit(fixed_holder[kP2pFixedTensorDataIndex]), frozen_holder, file_constant_mem_holder},
        1U);
  }

  // 临时方案，解决HCCL算子二级地址拷贝性能问题，待HCCL 1230正式方案上库后删除
  // 新增DavinciModelCreateV2 kernel,静态子图申请workspace内存时不再单独申请
  /**
   * 规避方案：ge在下发hccl task时传递一级地址，地址不再需要刷新，省去二级地址拷贝的开销，
   * 但是需要保证地址固定不变。
   * 方案详述：动态图中的静态子图申请workspace内存时，按照该动态图中静态子图的最大size申请，
   * 所有静态子图共用这部分内存，保证每次执行时workspace地址固定，同时申请后的地址传递给davinciModel
   * 实例，由davinciModel实例在init阶段下发传给hccl。
   *
   * 正式方案由HCCL模块适配上库。
  */
  auto builder = [&global_data]() -> std::vector<bg::ValueHolderPtr> {
    auto init_outputs = bg::FrameSelector::OnInitRoot(
        [&global_data]() -> std::vector<bg::ValueHolderPtr> {
        int64_t size = global_data.GetStaticModelWsSize();
        const auto memory_size_holder = bg::ValueHolder::CreateConst(&size, sizeof(size));
        const auto memory_holder = bg::AllocMem(kOnDeviceHbm, memory_size_holder, global_data, bg::kMainStream);
        return {memory_holder};
      });
    return init_outputs;
  };
  auto init_outputs = global_data.GetOrCreateUniqueValueHolder("WorkspaceMemFromInit", builder);
  GE_ASSERT_TRUE(init_outputs.size() == 1U);
  return bg::ValueHolder::CreateDataOutput(
          "DavinciModelCreateV2",
          {model_holder, session_id, assign_mem_holder, step_id, root_graph_id_holder,
           bg::HolderOnInit(bg::GetSpaceRegistries(global_data)),
           bg::HolderOnInit(bg::GetFileConstantWeightDir(global_data)),
           bg::HolderOnInit(bg::ReusableStreamAllocator(global_data)), bg::HolderOnInit(init_outputs[0U])},
          1U);
}

bg::ValueHolderPtr CreateDavinciModel(const ge::ComputeGraphPtr &graph, ge::GeModel *ge_model,
                                      LoweringGlobalData &global_data) {
  auto builder = [&graph, &ge_model, &global_data]() -> std::vector<bg::ValueHolderPtr> {
    return CreateDavinciModelOnInitRoot(graph, ge_model, global_data);
  };

  const auto &davinci_model_create = bg::FrameSelector::OnInitRoot(builder);
  GE_ASSERT_TRUE(!davinci_model_create.empty());

  const auto &davinci_model_finalizer = bg::DavinciModelFinalizer(davinci_model_create[0], global_data);
  GE_ASSERT_NOTNULL(davinci_model_finalizer);

  return davinci_model_create[0];
}

ge::Status MallocWorkspaceMem(ge::GeModel *ge_model, LoweringGlobalData &global_data,
                              std::vector<bg::ValueHolderPtr> &workspace_holders,
                              std::vector<bg::ValueHolderPtr> &owned_memories) {
  std::ostringstream oss;
  const std::vector<ge::MemInfo> mem_infos = ge::ModelUtils::GetAllMemoryTypeSize(ge_model->shared_from_this());
  const bool is_use_hbm_fixed_feature_mem = IsUseHbmFixedFeatureMemory(global_data);
  for (const auto &mem_info : mem_infos) {
    if (mem_info.memory_size <= 0) {
      oss << "[type: " << std::hex << mem_info.memory_type << ", size: 0], ";
      continue;
    }
    /* fixed mem is set not alloc */
    if (is_use_hbm_fixed_feature_mem && (mem_info.memory_type == RT_MEMORY_HBM) &&
        mem_info.is_fixed_addr_prior) {
      continue;
    }
    /*
     * p2p 都作为fixed feature memory
     * 只要size不为0，在init图中申请；如果用户将地址和size都设置为0，init图中不申请，那么在davinci model init的时候申请
     */
    if (mem_info.memory_type == RT_MEMORY_P2P_DDR) {
      oss << "[type: " << ge::MemTypeUtils::ToString(mem_info.memory_type)
          << ", size: " << mem_info.memory_size << ", as fixed mem, skip alloc here], ";
      continue;
    }
    const TensorPlacement placement = GetPlacement(mem_info.memory_type);
    GE_ASSERT(placement != kTensorPlacementEnd, "Invalid memory placement[0x%lx]", mem_info.memory_type);
    const auto memory_type_holder = bg::ValueHolder::CreateConst(&(mem_info.memory_type), sizeof(mem_info.memory_type));
    workspace_holders.emplace_back(memory_type_holder);
    const auto memory_size_holder = bg::ValueHolder::CreateConst(&(mem_info.memory_size), sizeof(mem_info.memory_size));
    const auto memory_holder = bg::AllocMemWithoutGuarder(placement, memory_size_holder, global_data, bg::kMainStream);
    workspace_holders.emplace_back(memory_holder);
    owned_memories.emplace_back(memory_holder);
    oss << "[type: " << ge::MemTypeUtils::ToString(mem_info.memory_type)
        << std::dec << ", placement: " << placement
        << ", size: " << mem_info.memory_size << "], ";
  }
  GELOGI("total workspace_holders num: %zu, %s", workspace_holders.size(), oss.str().c_str());
  return ge::SUCCESS;
}
/*
              lowering阶段将这些输出地址组成vector返回

             /                 |                              \

    GetRunAddress         AllocMemory(output)                  AllocMemory(output)

      |                    /      \                             /      \               …… 多个
 Const(offset)  CreateAllocator  Const(memory size)  CreateL1Allocator  Const(memory size)
*/

bg::ValueHolderPtr Execute(const bg::ValueHolderPtr &davinci_model,
                           const std::vector<bg::DevMemValueHolderPtr> &input_addrs, LoweringGlobalData &global_data,
                           std::vector<bg::DevMemValueHolderPtr> &outputs) {
  std::vector<bg::ValueHolderPtr> inputs;
  const size_t input_num = input_addrs.size();
  const size_t output_num = outputs.size();
  inputs.emplace_back(davinci_model);
  inputs.emplace_back(global_data.GetStream());
  inputs.emplace_back(bg::ValueHolder::CreateConst(&input_num, sizeof(size_t)));
  inputs.emplace_back(bg::ValueHolder::CreateConst(&output_num, sizeof(size_t)));
  inputs.insert(inputs.cend(), input_addrs.cbegin(), input_addrs.cend());
  inputs.insert(inputs.cend(), outputs.cbegin(), outputs.cend());
  return bg::ValueHolder::CreateSingleDataOutput("DavinciModelExecute", inputs);
}

bg::ValueHolderPtr UpdateWorkspaces(const bg::ValueHolderPtr &davinci_model,
                                    const std::vector<bg::ValueHolderPtr> &workspace_holders) {
  std::vector<bg::ValueHolderPtr> inputs;
  const size_t workspace_num = workspace_holders.size();
  inputs.emplace_back(davinci_model);
  inputs.emplace_back(bg::ValueHolder::CreateConst(&workspace_num, sizeof(size_t)));
  inputs.insert(inputs.cend(), workspace_holders.cbegin(), workspace_holders.cend());
  return bg::ValueHolder::CreateSingleDataOutput("DavinciModelUpdateWorkspaces", inputs);
}

LowerResult ConstructStaticModelInputs(const std::vector<ge::NodePtr> &data_nodes, LoweringGlobalData &global_data) {
  LowerResult lower_result;
  LowerInput input;
  input.global_data = &global_data;
  for (const auto &node : data_nodes) {
    auto lower_data_result = LoweringNode(node, input, {});
    LOWER_REQUIRE_NOTNULL(lower_data_result);
    LOWER_REQUIRE(lower_data_result->out_shapes.size() == 1U);
    LOWER_REQUIRE(lower_data_result->out_addrs.size() == 1U);
    const auto placed_result = node->GetOpDescBarePtr()->GetExtAttr<PlacedLoweringResult>(kLoweringResult);
    LOWER_REQUIRE_NOTNULL(placed_result, "Model data node %s have no placed lowering result", node->GetNamePtr());

    // 动态shape静态子图输入一定是零拷贝地址，一定是默认的HBM内存，不会是特殊类型的HBM内存
    auto result =
        placed_result->GetOutputResult(global_data, 0, {TensorPlacement::kOnDeviceHbm, bg::kMainStream}, false);
    LOWER_REQUIRE_NOTNULL(result);
    LOWER_REQUIRE(result->has_init);

    GELOGD("Model data %s device addr from %s", node->GetNamePtr(),
           bg::ValueHolderUtils::GetNodeTypeBarePtr(result->address));
    lower_result.out_shapes.push_back(result->shape);
    lower_result.out_addrs.push_back(result->address);
    lower_result.order_holders.insert(lower_result.order_holders.cend(), result->order_holders.cbegin(),
                                      result->order_holders.cend());
  }
  return lower_result;
}

const LowerResult *BuildKnownSubgraphOutputs(const ge::NodePtr &node, LowerInput &inputs,
                                             const std::vector<bg::ValueHolderPtr> &ordered_inputs) {
  std::map<ge::NodePtr, LowerResult> node_and_lower_results;
  for (const auto in_anchor : node->GetAllInDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(in_anchor);
    GE_ASSERT_NOTNULL(in_anchor->GetPeerOutAnchor());
    GE_ASSERT_NOTNULL(in_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr());
    auto in_node = in_anchor->GetPeerOutAnchor()->GetOwnerNode();
    auto src_index = in_anchor->GetPeerOutAnchor()->GetIdx();
    auto dst_index = in_anchor->GetIdx();
    GE_ASSERT_TRUE(static_cast<size_t>(dst_index) <= inputs.input_shapes.size());
    GE_ASSERT_TRUE(static_cast<size_t>(dst_index) <= inputs.input_addrs.size());
    GELOGI("Graph output %d ref from node %s output %d", dst_index, in_node->GetNamePtr(), src_index);
    auto &lower_result = node_and_lower_results[in_node];
    if (static_cast<size_t>(src_index) >= lower_result.out_shapes.size()) {
      lower_result.out_shapes.resize(src_index + 1U);
      lower_result.out_addrs.resize(src_index + 1U);
    }
    // get Netoutput input shape as graph out_shape, in case static graph has shape_uncontinousely optimize
    const auto &input_desc = node->GetOpDesc()->GetInputDescPtr(in_anchor->GetIdx());
    auto input_shape = NodeConverterUtils::CreateOutputShape(input_desc);
    lower_result.out_shapes[src_index] = input_shape;
    lower_result.out_addrs[src_index] = inputs.input_addrs[dst_index];
    lower_result.order_holders = ordered_inputs;
  }

  for (auto node_and_lower_result : node_and_lower_results) {
    auto &in_node = node_and_lower_result.first;
    GELOGI("Start assemble lower result of node %s type %s", in_node->GetNamePtr(), in_node->GetTypePtr());
    GE_ASSERT(in_node->GetOpDescBarePtr()->SetExtAttr(
        kLoweringResult, PlacedLoweringResult(in_node, std::move(node_and_lower_result.second))));
  }
  GELOGI("Start lowering static graph net-output %s", node->GetNamePtr());
  auto lower_result = LoweringNode(node, inputs, ordered_inputs);
  GE_ASSERT_NOTNULL(lower_result, "Failed lower static graph net-output node %s", node->GetNamePtr());
  return lower_result;
}

ge::Status CollectGraphInputsAndNetOutput(const ge::ComputeGraphPtr &graph, std::vector<ge::NodePtr> &inputs,
                                          ge::NodePtr &net_output) {
  std::map<uint32_t, ge::NodePtr> index_to_data_nodes;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == ge::DATA) {
      uint32_t index = 0U;
      const auto op_desc = node->GetOpDescBarePtr();
      GE_ASSERT_NOTNULL(op_desc);
      if (!ge::AttrUtils::GetInt(op_desc, ge::ATTR_NAME_PARENT_NODE_INDEX, index)) {
        GELOGW("Data %s seems not subgraph input, maybe %s is a static compiled root graph?", node->GetNamePtr(),
               graph->GetName().c_str());
        GE_ASSERT(ge::AttrUtils::GetInt(op_desc, ge::ATTR_NAME_INDEX, index), "%s get index failed",
                  node->GetNamePtr());
      }
      index_to_data_nodes[index] = node;
    } else if (node->GetType() == ge::NETOUTPUT) {
      net_output = node;
    }
  }
  GE_ASSERT_NOTNULL(net_output);
  for (auto &index_to_data_node : index_to_data_nodes) {
    inputs.emplace_back(index_to_data_node.second);
  }
  return ge::SUCCESS;
}

using ReuseMap = std::map<size_t, size_t>;
using ReuseIdx2OffsetMap = std::map<size_t, kernel::MemoryBaseTypeOffset>;
struct OutputsReuseDebugInfo {
  ReuseIdx2OffsetMap ref_outputs;  // outputs_idex to offset
  ReuseMap reuse_inputs;           // outputs_idex to inputs_idx
  ReuseMap reuse_outputs;          // outputs_idex to outputs_idex
  vector<size_t> no_reuse_outputs;
};

OutputsReuseDebugInfo GetReuseMap(const std::vector<OutputReuseInfo> &output_reuse_infos) {
  OutputsReuseDebugInfo debug;
  for (size_t i = 0U; i < output_reuse_infos.size(); ++i) {
    const auto &output_info = output_reuse_infos[i];
    if (!output_info.is_reuse) {
      debug.no_reuse_outputs.emplace_back(i);
      continue;
    }
    switch (output_info.reuse_type) {
      case OutputReuseType::kRefOutput:
        debug.ref_outputs.insert({i, output_info.mem_base_type_offset});
        break;
      case OutputReuseType::kReuseOutput:
        debug.reuse_outputs.insert({i, output_info.reuse_index});
        break;
      case OutputReuseType::kReuseInput:
        debug.reuse_inputs.insert({i, output_info.reuse_index});
        break;
      case OutputReuseType::kNoReuse:
      case OutputReuseType::kEnd:
      case OutputReuseType::kRefVariable:
        break;
      default:
        break;
    }
  }
  return debug;
}

std::string DebugString(const std::vector<OutputReuseInfo> &output_reuse_infos) {
  auto debug = GetReuseMap(output_reuse_infos);
  std::stringstream ss;
  ss << "outputs num: " << output_reuse_infos.size() << ", no reuse outputs: [";
  for (const auto &idx : debug.no_reuse_outputs) {
    ss << idx << ", ";
  }
  ss << "], reuse outputs: [";
  for (const auto &idx_pair : debug.reuse_outputs) {
    ss << "(" << idx_pair.first << "->" << idx_pair.second << "), ";
  }
  ss << "], reuse inputs: [";
  for (const auto &idx_pair : debug.reuse_inputs) {
    ss << "(" << idx_pair.first << "->" << idx_pair.second << "), ";
  }
  ss << "], reference outputs: [";
  for (const auto &idx_pair : debug.ref_outputs) {
    ss << "(" << idx_pair.first << "->" << static_cast<int32_t>(idx_pair.second.base_type) << "/"
       << idx_pair.second.offset << "), ";
  }
  ss << "]";
  auto str = ss.str();
  return str.length() > kLogLengthLimit ? str.substr(0, kLogLengthLimit) + "..." : str;
}
}  // namespace

bool IsGraphStaticCompiled(const ge::ComputeGraphPtr &graph) {
  return !GraphUnfolder::IsGraphDynamicCompiled(graph);
}

bool IsStaticCompiledGraphHasTaskToLaunch(ge::GeModel *static_model) {
  return (!static_model->GetTBEKernelStore().IsEmpty() || !static_model->GetCustAICPUKernelStore().IsEmpty() ||
          ((static_model->GetModelTaskDefPtr() != nullptr) && (static_model->GetModelTaskDefPtr()->task_size() > 1)));
}

bool IsStaticCompiledGraphHasTaskToLaunch(const ge::ComputeGraphPtr &graph, LoweringGlobalData &global_data) {
  auto model = global_data.GetGraphStaticCompiledModel(graph->GetName());
  if (model == nullptr) {
    return false;
  }
  auto static_model = static_cast<ge::GeModel *>(model);
  GELOGI("graph: %s, tbe kernel empty: %d, aicpu is empty: %d, task size is:%d",
         graph->GetName().c_str(), static_model->GetTBEKernelStore().IsEmpty(),
         static_model->GetCustAICPUKernelStore().IsEmpty(),
         static_model->GetModelTaskDefPtr()->task_size());
  // There is task generated result of netoutput node
  return IsStaticCompiledGraphHasTaskToLaunch(static_model);
}

bool IsNeedLoweringAsStaticCompiledGraph(const ge::ComputeGraphPtr &graph, LoweringGlobalData &global_data) {
  if (IsGraphStaticCompiled(graph) && IsStaticCompiledGraphHasTaskToLaunch(graph, global_data)) {
    GELOGI("lowering as static compiled graph is needed for %s.", graph->GetName().c_str());
    return true;
  }
  GELOGI("lowering as static compiled graph is not needed for %s.", graph->GetName().c_str());
  return false;
}

LowerResult LoweringStaticModel(const ge::ComputeGraphPtr &graph, ge::GeModel *static_model,
                                const std::vector<bg::DevMemValueHolderPtr> &input_addrs,
                                LoweringGlobalData &global_data) {
  GELOGI("Start lowering static model of graph %s", graph->GetName().c_str());
  GELOGI("Assemble static model of graph %s", graph->GetName().c_str());
  // Create and init gert static model from static compile model result
  auto davinci_model_holder = CreateDavinciModel(graph, static_model, global_data);
  LOWER_REQUIRE(IsValidHolder(davinci_model_holder));

  // Malloc workspace memory
  std::vector<bg::ValueHolderPtr> workspace_holders;
  std::vector<bg::ValueHolderPtr> owned_memories;
  std::string is_addr_fixed_opt;
  (void)ge::GetContext().GetOption(kStaticModelAddrFixed, is_addr_fixed_opt);
  if (is_addr_fixed_opt.empty()) {
    LOWER_REQUIRE_SUCCESS(MallocWorkspaceMem(static_model, global_data, workspace_holders, owned_memories));
  }

  std::vector<OutputReuseInfo> output_reuse_infos;
  LOWER_REQUIRE_GRAPH_SUCCESS(StaticModelOutputAllocator::GenerateOutputsReuseInfos(graph, output_reuse_infos));
  GELOGI("Reuse info of static graph %s: %s", graph->GetName().c_str(), DebugString(output_reuse_infos).c_str());

  LowerResult model_output;
  bg::ValueHolderPtr update_workspaces_holder = nullptr;
  if (workspace_holders.empty()) {
    model_output =
        StaticModelOutputAllocator(davinci_model_holder, input_addrs).AllocAllOutputs(output_reuse_infos, global_data);
  } else {
    update_workspaces_holder = UpdateWorkspaces(davinci_model_holder, workspace_holders);
    LOWER_REQUIRE_VALID_HOLDER(update_workspaces_holder);
    model_output = StaticModelOutputAllocator(davinci_model_holder, input_addrs, update_workspaces_holder)
                       .AllocAllOutputs(output_reuse_infos, global_data);
  }
  LOWER_RETURN_IF_ERROR(model_output);
  // Newly malloc memory stored in ordered_holders
  owned_memories.insert(owned_memories.cend(), model_output.order_holders.cbegin(), model_output.order_holders.cend());

  // Build model call for model
  auto model_execute_holder = Execute(davinci_model_holder, input_addrs, global_data, model_output.out_addrs);
  if (update_workspaces_holder != nullptr) {
    LOWER_REQUIRE_HYPER_SUCCESS(bg::ValueHolder::AddDependency(update_workspaces_holder, model_execute_holder));
  }

  // Free owned memory
  for (auto &memory_block : owned_memories) {
    auto free_holder = bg::ValueHolder::CreateVoidGuarder("FreeMemory", memory_block, {});
    LOWER_REQUIRE_HYPER_SUCCESS(bg::ValueHolder::AddDependency(model_execute_holder, free_holder));
  }

  return {HyperStatus::Success(),
          {model_execute_holder},
          std::move(model_output.out_shapes),
          std::move(model_output.out_addrs)};
}

const LowerResult *LoweringStaticCompiledComputeGraph(const ge::ComputeGraphPtr &graph,
                                                      LoweringGlobalData &global_data) {
  GELOGI("Start lowering static compiled graph %s", graph->GetName().c_str());
  auto parent_compute_node = bg::ValueHolder::GetCurrentFrame()->GetCurrentComputeNode();

  std::vector<ge::NodePtr> data_nodes;
  ge::NodePtr net_output = nullptr;
  GE_ASSERT_SUCCESS(CollectGraphInputsAndNetOutput(graph, data_nodes, net_output));
  GELOGI("Graph %s has %zu inputs and %u outputs", graph->GetName().c_str(), data_nodes.size(),
         net_output->GetAllInDataAnchorsSize());

  auto model_inputs = ConstructStaticModelInputs(data_nodes, global_data);
  GE_ASSERT_HYPER_SUCCESS(model_inputs.result);
  GELOGI("Graph model %s num lower inputs %zu", graph->GetName().c_str(), model_inputs.out_addrs.size());

  auto model = global_data.GetGraphStaticCompiledModel(graph->GetName());
  GE_ASSERT_NOTNULL(model, "Compile result of static graph %s not found", graph->GetName().c_str());

  bg::ValueHolder::SetCurrentComputeNode(parent_compute_node);
  auto static_model = static_cast<ge::GeModel *>(model);
  auto model_lower_result = LoweringStaticModel(graph, static_model, model_inputs.out_addrs, global_data);
  GE_ASSERT(model_lower_result.result.IsSuccess());

  GELOGI("Graph model %s num lower outputs %zu", graph->GetName().c_str(), model_lower_result.out_addrs.size());
  LowerInput model_outputs;
  model_outputs.input_shapes = model_lower_result.out_shapes;
  model_outputs.input_addrs = model_lower_result.out_addrs;
  model_outputs.global_data = &global_data;
  GELOGI("Start build static compiled graph %s outputs", graph->GetName().c_str());
  const auto result = BuildKnownSubgraphOutputs(net_output, model_outputs, model_lower_result.order_holders);
  bg::ValueHolder::SetCurrentComputeNode(parent_compute_node);
  return result;
}
}  // namespace gert
