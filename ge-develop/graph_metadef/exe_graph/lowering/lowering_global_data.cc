/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/lowering_global_data.h"
#include <memory>
#include "common/checker.h"
#include "exe_graph/lowering/frame_selector.h"
#include "graph/fast_graph/execute_graph.h"

namespace gert {
namespace {
constexpr const ge::char_t *kGlobalDataL2AllocatorsPrefix = "L2_Allocators_";
const std::set<gert::TensorPlacement> kCurrentAllocatorSupportPlacement = {
    TensorPlacement::kOnDeviceHbm, TensorPlacement::kOnHost, TensorPlacement::kFollowing};
const std::set<gert::AllocatorUsage> kCurrentAllocatorSupportUsage = {
    AllocatorUsage::kAllocNodeOutput, AllocatorUsage::kAllocNodeWorkspace, AllocatorUsage::kAllocNodeShapeBuffer};

inline bool IsPlacementSupportExternalAllocator(const TensorPlacement placement) {
  return kCurrentAllocatorSupportPlacement.find(placement) != kCurrentAllocatorSupportPlacement.cend();
}

inline bool IsUsageSupportExternalAllocator(const AllocatorUsage &usage) {
  return kCurrentAllocatorSupportUsage.find(usage) != kCurrentAllocatorSupportUsage.cend();
}

bool CurrentOnInitGraph() {
  const ge::FastNode *subgraph_node = nullptr;
  auto current_graph = bg::ValueHolder::GetCurrentExecuteGraph();
  while ((current_graph != nullptr) && (current_graph->GetParentNodeBarePtr() != nullptr)) {
    subgraph_node = current_graph->GetParentNodeBarePtr();
    current_graph = subgraph_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  }

  if (subgraph_node != nullptr) {
    return strcmp(subgraph_node->GetTypePtr(), GetExecuteGraphTypeStr(ExecuteGraphType::kInit)) == 0;
  } else {
    return false;
  }
}
/*
 * 此处用于判断是否能使用init图里的allocator:
 * 1.用户设置了always_external_allocator选项
 * 2.AllocatorDesc在我们当前支持的allocator范围内
 *
 * 为了兼容性考虑，当前只能支持现有的allocator，否则后续我们新增placement/useage时则会出错，用户老的版本加上我们新的软件会出错
 * */
bool CanUseInitAllocator(const bool always_external_allocator, const AllocatorDesc &desc) {
  if (!always_external_allocator) {
    return false;
  }
  if (IsPlacementSupportExternalAllocator(desc.placement) && IsUsageSupportExternalAllocator(desc.usage)) {
    return true;
  } else {
    GELOGW("We don't support placement[%d] or usage[%d] current while always_external_allocator is true",
           static_cast<int32_t>(desc.placement), static_cast<int32_t>(desc.usage));
    return false;
  }
}

std::string GenL2AllocatorKey(int64_t logic_stream_id, const AllocatorDesc desc) {
  if (TensorPlacementUtils::IsOnDevice(desc.placement)) {
    return kGlobalDataL2AllocatorsPrefix + std::to_string(desc.placement) + std::to_string(logic_stream_id);
  } else {
    return kGlobalDataL2AllocatorsPrefix + std::to_string(desc.placement);
  }
}

std::string GenAllL2AllocatorsKey(const AllocatorDesc desc) {
  return kGlobalDataL2AllocatorsPrefix + std::to_string(desc.placement);
}

std::string GenInitL2AllocatorsKey(const AllocatorDesc desc) {
  return kGlobalDataL2AllocatorsPrefix + std::to_string(desc.placement) + "-Init";
}

bg::ValueHolderPtr GetOrCreateL2Allocators(const AllocatorDesc desc, LoweringGlobalData &global_data) {
  auto all_l2_allocators_key = GenAllL2AllocatorsKey(desc);
  return global_data.GetOrCreateUniqueValueHolder(all_l2_allocators_key, [&desc, &global_data]() -> bg::ValueHolderPtr {
    auto init_out = bg::FrameSelector::OnInitRoot([&desc, &global_data]() -> std::vector<bg::ValueHolderPtr> {
      auto placement_holder = bg::ValueHolder::CreateConst(&desc.placement, sizeof(desc.placement));
      auto init_l1_allocator = global_data.GetOrCreateL1Allocator(desc);
      auto stream_num = global_data.GetUniqueValueHolder(kGlobalDataModelStreamNum);
      GE_ASSERT_NOTNULL(stream_num);
      // CreateL2Allocators第一个输出是AllL2Allocators，表示所有二级内存池；
      // 第二个输出是AllL2MemPools，表示所有二级内存池里实际申请内存的allocator；
      auto l2_allocators_on_init = bg::ValueHolder::CreateDataOutput("CreateL2Allocators",
                                                                     {stream_num, placement_holder}, 2)[0];
      bg::ValueHolder::AddDependency(init_l1_allocator, l2_allocators_on_init);
      return {l2_allocators_on_init};
    });
    GE_ASSERT_EQ(init_out.size(), 1U);
    GE_ASSERT_NOTNULL(init_out[0]);
    return init_out[0];
  });
}

bg::ValueHolderPtr GetOrCreateDeviceL2Allocator(int64_t logic_stream_id, AllocatorDesc desc,
                                                const std::string &l2_allocator_key, LoweringGlobalData &global_data) {
  auto builder = [&logic_stream_id, &desc, &global_data]() -> bg::ValueHolderPtr {
    bg::ValueHolderPtr l2_allocators = GetOrCreateL2Allocators(desc, global_data);
    GE_ASSERT_NOTNULL(l2_allocators);
    auto logic_stream_id_holder = bg::ValueHolder::CreateConst(&logic_stream_id, sizeof(logic_stream_id));
    auto rt_stream_holder = global_data.GetStreamById(logic_stream_id);
    GE_ASSERT_NOTNULL(rt_stream_holder);
    auto l1_allocator = global_data.GetOrCreateL1Allocator(desc);
    return bg::ValueHolder::CreateSingleDataOutput(
        "SelectL2Allocator", {logic_stream_id_holder, rt_stream_holder, l1_allocator, l2_allocators});
  };
  return global_data.GetOrCreateUniqueValueHolder(l2_allocator_key, builder);
}

bg::ValueHolderPtr GetOrCreateHostL2Allocator(AllocatorDesc desc, const std::string &l2_allocator_key,
                                              LoweringGlobalData &global_data) {
  auto builder = [&desc, &global_data]() -> bg::ValueHolderPtr {
    return bg::ValueHolder::CreateSingleDataOutput("CreateHostL2Allocator", {global_data.GetOrCreateL1Allocator(desc)});
  };
  return global_data.GetOrCreateUniqueValueHolder(l2_allocator_key, builder);
}
}  // namespace

const LoweringGlobalData::NodeCompileResult *LoweringGlobalData::FindCompiledResult(const ge::NodePtr &node) const {
  const auto iter = node_name_to_compile_result_holders_.find(node->GetName());
  if (iter == node_name_to_compile_result_holders_.cend()) {
    return nullptr;
  }
  return &iter->second;
}
LoweringGlobalData &LoweringGlobalData::AddCompiledResult(const ge::NodePtr &node,
                                                          LoweringGlobalData::NodeCompileResult compile_result) {
  node_name_to_compile_result_holders_[node->GetName()] = std::move(compile_result);
  return *this;
}

void *LoweringGlobalData::GetGraphStaticCompiledModel(const std::string &graph_name) const {
  const auto iter = graph_to_static_models_.find(graph_name);
  if (iter == graph_to_static_models_.cend()) {
    return nullptr;
  }
  return iter->second;
}

LoweringGlobalData &LoweringGlobalData::AddStaticCompiledGraphModel(const std::string &graph_name, void *const model) {
  graph_to_static_models_[graph_name] = model;
  return *this;
}

bg::ValueHolderPtr LoweringGlobalData::GetL1Allocator(const AllocatorDesc &desc) const {
  if (CurrentOnInitGraph()) {
    return GetUniqueValueHolder(desc.GetKey() + "-Init");
  } else {
    return GetUniqueValueHolder(desc.GetKey());
  }
}
LoweringGlobalData &LoweringGlobalData::SetExternalAllocator(bg::ValueHolderPtr &&allocator) {
  return SetExternalAllocator(std::move(allocator), ExecuteGraphType::kMain);
}
LoweringGlobalData &LoweringGlobalData::SetExternalAllocator(bg::ValueHolderPtr &&allocator,
                                                             const ExecuteGraphType graph_type) {
  if (graph_type >= ExecuteGraphType::kNum) {
    return *this;
  }
  external_allocators_.holders[static_cast<size_t>(graph_type)] = std::move(allocator);
  return *this;
}

bool LoweringGlobalData::CanUseExternalAllocator(const ExecuteGraphType &graph_type,
                                                 const TensorPlacement placement) const {
  return IsPlacementSupportExternalAllocator(placement)
      && (external_allocators_.holders[static_cast<size_t>(graph_type)] != nullptr);
}

bg::ValueHolderPtr LoweringGlobalData::GetExternalAllocator(const bool from_init, const string &key,
                                                            const AllocatorDesc &desc) {
  bg::ValueHolderPtr init_selected_allocator = nullptr;
  auto init_out =
      bg::FrameSelector::OnInitRoot([&desc, &init_selected_allocator, this]() -> std::vector<bg::ValueHolderPtr> {
        auto placement_holder = bg::ValueHolder::CreateConst(&desc.placement, sizeof(desc.placement));
        init_selected_allocator = nullptr;
        if (CanUseExternalAllocator(ExecuteGraphType::kInit, desc.placement)) {
          init_selected_allocator = bg::ValueHolder::CreateSingleDataOutput(
              "GetExternalL1Allocator",
              {placement_holder,
               external_allocators_.holders[static_cast<size_t>(ExecuteGraphType::kInit)]});
        } else {
          GELOGE(ge::PARAM_INVALID, "always_external_allocator option is true but external_allocators is nullptr!");
        }
        return {init_selected_allocator};
      });
  GE_ASSERT_EQ(init_out.size(), 1U);
  GE_ASSERT_NOTNULL(init_out[0]);

  auto allocator = bg::FrameSelector::OnMainRoot([&desc, &init_out, this]() -> std::vector<bg::ValueHolderPtr> {
    auto main_selected_allocator = init_out[0];
    auto placement_holder = bg::ValueHolder::CreateConst(&desc.placement, sizeof(desc.placement));
    if (CanUseExternalAllocator(ExecuteGraphType::kMain, desc.placement)) {
      main_selected_allocator = bg::ValueHolder::CreateSingleDataOutput(
          "SelectL1Allocator",
          {placement_holder, external_allocators_.holders[static_cast<size_t>(ExecuteGraphType::kMain)], init_out[0],
           GetStreamById(kDefaultMainStreamId)});
    }
    return {main_selected_allocator};
  });
  GE_ASSERT_EQ(allocator.size(), 1U);

  SetUniqueValueHolder(key + "-Init", init_selected_allocator);
  SetUniqueValueHolder(key, allocator[0]);
  if (from_init) {
    return init_selected_allocator;
  } else {
    return allocator[0];
  }
}

/* CanUseInitAllocator is true
 * +------------------------------------------------------------------+
 * |Main Graph                                                        |
 * |                 AllocMemory                                      |
 * |                     |                                            |
 * |                 (allocator)                                      |
 * |                     |                                            |
 * |                 InnerData                                        |
 * +------------------------------------------------------------------+
 * +------------------------------------------------------------------+
 * |Init Graph                                                        |
 * |                                                                  |
 * |                   InnerNetOutput                                 |
 * |                        ^                                         |
 * |                        |                                         |
 * |                    GetExternalL1Allocator                        |
 * |                   /     |      \                                 |
 * |  Const(placement) Const(usage) Data(Allocator)(-2) |             |
 * +------------------------------------------------------------------+
 */

/* CanUseInitAllocator is false
 * +------------------------------------------------------------------+
 * |Main Graph                                                        |
 * |                 (allocator)                                      |
 * |                     |                                            |
 * |     +------>  SelectL1Allocator  <-----+                         |
 * |     |           /       \              |                         |
 * | InnerData  InnerData   InnerData   Data(-2)                      |
 * +------------------------------------------------------------------+
 * +------------------------------------------------------------------+
 * |Init Graph                                                        |
 * |                                                                  |
 * |   +------+--->  InnerNetOutput    (allocator)                    |
 * |   |      |              ^            |                           |
 * |   |      |              |     SelectL1Allocator                  |
 * |   |      |              |    /         ^     \                   |
 * |   |      |     CreateL1Allocator       |   Data(Allocator)(-2)   |
 * |   |      |          /  \               |                         |
 * |   |  Const(placement)  Const(usage)    |                         |
 * |   |                         |          |                         |
 * |   +-------------------------+----------+                         |
 * +------------------------------------------------------------------+
 */
bg::ValueHolderPtr LoweringGlobalData::GetOrCreateL1Allocator(const AllocatorDesc desc) {
  const auto key = desc.GetKey();
  const auto init_key = key + "-Init";
  const auto from_init = CurrentOnInitGraph();

  bg::ValueHolderPtr allocator_holder;
  if (from_init) {
    allocator_holder = GetUniqueValueHolder(init_key);
  } else {
    allocator_holder = GetUniqueValueHolder(key);
  }

  if (allocator_holder != nullptr) {
    return allocator_holder;
  }
  /*
   * 用户设置always_external_allocator场景下，同时external_allocators_不为空的情况下，一定认为所有类型的allocator都创建好了，原因：
   * 1.不能考虑外置的external_allocators_中存在某些类型的allocator没有创建，之前为了保证正确性，必须在构图时根据placement跟usage
   * 创建一个CreateAllocator节点，在执行时创建兜底的allocator对象，但是allocator对象是需要浪费host内存资源，对单算子场景下，
   * 频繁创建导致host内存上升，因此设置了always_external_allocator的场景下不考虑某些类型的allocator没有创建
   *
   * 2.为什么这个地方不能判断满足当前placement+usage的allocator是否已经创建好了？这个地方还在构图，此时还是valueholder，还没有到初始化
   * 执行，因此无法感知用户是否完整创建了所有allocator，只有初始化图执行时才知道。
   *
   * 3.因此对于此场景，考虑在初始化图执行时做一个校验，用户设置了always_external_allocator的场景下，确保所有类型的allocator都创建好了
   *  因此, 在单算子场景下，需要无脑校验
   *
   * 4.为了兼容性考虑，当前只能支持现有的allocator，否则后续我们新增placement/useage时则会出错，用户老的版本加上我们新的软件会出错
   *
   * 5.always_external_allocator可以后续整改为always_use_init_allocator
   * */
  if (CanUseInitAllocator(GetLoweringOption().always_external_allocator, desc)) {
    return GetExternalAllocator(from_init, key, desc);
  } else {
    bg::ValueHolderPtr init_selected_allocator = nullptr;
    auto init_out =
        bg::FrameSelector::OnInitRoot([&desc, &init_selected_allocator, this]() -> std::vector<bg::ValueHolderPtr> {
          auto placement_holder = bg::ValueHolder::CreateConst(&desc.placement, sizeof(desc.placement));
          auto created_allocator = bg::ValueHolder::CreateSingleDataOutput("CreateL1Allocator", {placement_holder});
          if (CanUseExternalAllocator(ExecuteGraphType::kInit, desc.placement)) {
            const auto init_external_allocator =
                external_allocators_.holders[static_cast<size_t>(ExecuteGraphType::kInit)];
            init_selected_allocator = bg::ValueHolder::CreateSingleDataOutput(
                "SelectL1Allocator",
                {placement_holder, init_external_allocator, created_allocator, GetStreamById(kDefaultMainStreamId)});
            // here init_selected_allocator to init_node_out just for deconstruct sequence.
            // To make sure memblock alloced from init graph, which deconstruction relies on allocator alive.
            return {created_allocator, placement_holder, init_selected_allocator};
          } else {
            init_selected_allocator = created_allocator;
            return {created_allocator, placement_holder};
          }
        });
    GE_ASSERT_TRUE(init_out.size() >= 2U);

    auto allocator = bg::FrameSelector::OnMainRoot([&init_out, &desc, this]() -> std::vector<bg::ValueHolderPtr> {
      auto main_selected_allocator = init_out[0];
      if (CanUseExternalAllocator(ExecuteGraphType::kMain, desc.placement)) {
        const auto main_external_allocator = external_allocators_.holders[static_cast<size_t>(ExecuteGraphType::kMain)];
        main_selected_allocator = bg::ValueHolder::CreateSingleDataOutput(
            "SelectL1Allocator", {init_out[1], main_external_allocator, init_out[0],
                                  GetStreamById(kDefaultMainStreamId)});
      }
      return {main_selected_allocator};
    });
    GE_ASSERT_EQ(allocator.size(), 1U);

    SetUniqueValueHolder(key + "-Init", init_selected_allocator);
    SetUniqueValueHolder(key, allocator[0]);

    if (from_init) {
      return init_selected_allocator;
    } else {
      return allocator[0];
    }
  }
}

bg::ValueHolderPtr LoweringGlobalData::GetOrCreateUniqueValueHolder(
    const std::string &name, const std::function<bg::ValueHolderPtr()> &builder) {
  return GetOrCreateUniqueValueHolder(name, [&builder]() -> std::vector<bg::ValueHolderPtr> { return {builder()}; })[0];
}

std::vector<bg::ValueHolderPtr> LoweringGlobalData::GetOrCreateUniqueValueHolder(
    const std::string &name, const std::function<std::vector<bg::ValueHolderPtr>()> &builder) {
  const decltype(unique_name_to_value_holders_)::const_iterator &iter = unique_name_to_value_holders_.find(name);
  if (iter == unique_name_to_value_holders_.cend()) {
    auto holder = builder();
    return unique_name_to_value_holders_.emplace(name, holder).first->second;
  }
  return iter->second;
}
void LoweringGlobalData::SetUniqueValueHolder(const string &name, const bg::ValueHolderPtr &holder) {
  if (!unique_name_to_value_holders_.emplace(name, std::vector<bg::ValueHolderPtr>{holder}).second) {
    unique_name_to_value_holders_.erase(name);
    unique_name_to_value_holders_.emplace(name, std::vector<bg::ValueHolderPtr>{holder});
  }
}
bg::ValueHolderPtr LoweringGlobalData::GetUniqueValueHolder(const string &name) const {
  const auto &iter = unique_name_to_value_holders_.find(name);
  if (iter == unique_name_to_value_holders_.cend()) {
    return nullptr;
  }
  return iter->second[0];
}

void LoweringGlobalData::SetValueHolders(const string &name, const bg::ValueHolderPtr &holder) {
  unique_name_to_value_holders_[name].emplace_back(holder);
}

size_t LoweringGlobalData::GetValueHoldersSize(const string &name) {
  const auto &iter = unique_name_to_value_holders_.find(name);
  if (iter == unique_name_to_value_holders_.cend()) {
    return 0U;
  }
  return iter->second.size();
}

void LoweringGlobalData::SetModelWeightSize(const size_t require_weight_size) {
  model_weight_size_ = require_weight_size;
}
size_t LoweringGlobalData::GetModelWeightSize() const {
  return model_weight_size_;
}

const LoweringOption &LoweringGlobalData::GetLoweringOption() const {
  return lowering_option_;
}
void LoweringGlobalData::SetLoweringOption(const LoweringOption &lowering_option) {
  lowering_option_ = lowering_option;
}

/*
* init_graph
*                                                +--------------------------+
*                                                |          |               |
*  Const(placement)  Const(stream_num)    Const(placement)  Const(usage)    |
*          \          /                            \        /               |       Data(externel_allocator)
*           CreateL2Allocators                 CreateL1Allocator ----------+-----+      |       Data(external_stream)
*                             \                     /                       |     \    |       /
*                              \                   /                        |    SelectAllocator
*                                  InnerNetOutput <-------------------------+         /      \
*                                                                    CreateHostL2Allocator   CreateInitL2Allocator
*
*
* main_graph
*                                              SelectL1Allocator
*                  Data(rt_streams)            /                \
*                        |                    /  InnerData     CreateHostL2Allocator
*                 SplitRtStreams             /(l2 allocators)
*                              \            /   /
*                              SelectL2Allocator
*
*   有外置allocator场景下的L2 allocator
*/
bg::ValueHolderPtr LoweringGlobalData::GetOrCreateL2Allocator(int64_t logic_stream_id, AllocatorDesc desc) {
  if (CurrentOnInitGraph()) {
    // 在init图也可能有申请内存的行为，也需要使用l2 allocator来申请。
    // init图中的l2 allocator要绑定到init图的主流上，因此使用新kernel，与main图中l2 allocator构图不同
    return GetOrCreateInitL2Allocator(desc);
  }

  auto l2_allocator_key = GenL2AllocatorKey(logic_stream_id, desc);
  auto l2_allocator = GetUniqueValueHolder(l2_allocator_key);
  if (l2_allocator != nullptr) {
    return l2_allocator;
  }
  // To make sure l2 allocator lowering at main graph. In case l2 allocator lowering at subgraph cause a link
  // from inside subgraph to main root.
  auto tmp_l2_allocators = bg::FrameSelector::OnMainRoot(
      [&desc, &logic_stream_id, &l2_allocator_key, this]() -> std::vector<bg::ValueHolderPtr> {
        if (TensorPlacementUtils::IsOnDevice(desc.placement)) {
          return {GetOrCreateDeviceL2Allocator(logic_stream_id, desc, l2_allocator_key, *this)};
        } else {
          return {GetOrCreateHostL2Allocator(desc, l2_allocator_key, *this)};
        }
      });
  GE_ASSERT_TRUE(tmp_l2_allocators.size() == 1U);

  // main图中创建host allocator时确保init图中也创建相应的host allocator,为CEM做准备
  if (TensorPlacementUtils::IsOnHost(desc.placement)) {
    auto init_l2_host_allocator =
      bg::FrameSelector::OnInitRoot([&desc, this]() -> std::vector<bg::ValueHolderPtr> {
        return {GetOrCreateInitL2Allocator(desc)};
    });
    GE_ASSERT_TRUE(init_l2_host_allocator.size() == 1U);
  }
  return tmp_l2_allocators[0];
}

bg::ValueHolderPtr LoweringGlobalData::GetInitL2Allocator(AllocatorDesc desc) const {
  auto init_l2_allocator_key = GenInitL2AllocatorsKey(desc);
  return GetUniqueValueHolder(init_l2_allocator_key);
}

bg::ValueHolderPtr LoweringGlobalData::GetMainL2Allocator(int64_t logic_stream_id, AllocatorDesc desc) const {
  auto main_l2_allocator_key = GenL2AllocatorKey(logic_stream_id, desc);
  return GetUniqueValueHolder(main_l2_allocator_key);
}

/**
 * This interface can only use on init graph
 * @param logic_stream_id
 * @param desc
 * @param global_data
 * @return
 */
bg::ValueHolderPtr LoweringGlobalData::GetOrCreateInitL2Allocator(const AllocatorDesc desc) {
  if (!CurrentOnInitGraph()) {
    return nullptr;
  }
  auto init_l2_allocator_key = GenInitL2AllocatorsKey(desc);
  bg::ValueHolderPtr init_l2_allocator = nullptr;
  if (TensorPlacementUtils::IsOnHost(desc.placement)) {
    auto builder = [&]() -> bg::ValueHolderPtr {
      return bg::ValueHolder::CreateSingleDataOutput("CreateHostL2Allocator",
                                                     {GetOrCreateL1Allocator(desc)});
    };
    init_l2_allocator = GetOrCreateUniqueValueHolder(init_l2_allocator_key, builder);
  } else if (TensorPlacementUtils::IsOnDevice(desc.placement)) {
    bg::ValueHolderPtr l2_allocators = GetOrCreateL2Allocators(desc, *this);
    GE_ASSERT_NOTNULL(l2_allocators);

    auto builder = [&]() -> bg::ValueHolderPtr {
      return bg::ValueHolder::CreateDataOutput(
          "CreateInitL2Allocator",
          {GetOrCreateL1Allocator(desc), bg::HolderOnInit(l2_allocators), GetStreamById(kDefaultMainStreamId)}, 2)[0];
    };
    init_l2_allocator = GetOrCreateUniqueValueHolder(init_l2_allocator_key, builder);
  } else {
    GELOGE(ge::PARAM_INVALID, "Unsupported placement %s.", desc.GetKey().c_str());
    return nullptr;
  }
  GE_ASSERT_NOTNULL(init_l2_allocator);

  // 将InitL2Allocator放到init的输出上，保证其在根图析构时 析构
  bg::FrameSelector::OnInitRoot([&init_l2_allocator]()->std::vector<bg::ValueHolderPtr> {
    return {init_l2_allocator};
  });
  return init_l2_allocator;
}

bg::ValueHolderPtr LoweringGlobalData::GetStreamById(int64_t logic_stream_id) const {
  ExecuteGraphType graph_type = ExecuteGraphType::kMain;
  if (CurrentOnInitGraph()) {
    graph_type = ExecuteGraphType::kInit;
    GE_ASSERT_TRUE(logic_stream_id == kDefaultMainStreamId);
  }
  const auto split_rt_streams = streams_.holders[static_cast<size_t>(graph_type)];
  GE_ASSERT_TRUE(static_cast<int64_t>(split_rt_streams.size()) > logic_stream_id);
  return split_rt_streams[logic_stream_id];
}

bg::ValueHolderPtr LoweringGlobalData::GetNotifyById(int64_t logic_notify_id) const {
  ExecuteGraphType graph_type = ExecuteGraphType::kMain;
  if (CurrentOnInitGraph()) {
    graph_type = ExecuteGraphType::kInit;
  }
  const auto rt_notifies = notifies_.holders[static_cast<size_t>(graph_type)];
  GE_ASSERT_TRUE(static_cast<int64_t>(rt_notifies.size()) > logic_notify_id,
                 "notify id [%ld] is invalid, total usable notify nums:[%zu].", logic_notify_id, rt_notifies.size());
  return rt_notifies[logic_notify_id];
}

/*
 * Init
 * +-------------------------------------------------+
 * |      Data(-1)   Const(stream num = 1)           |
 * |            \      /                             |
 * |          SplitRtStreams    Const(stream_num)    |
 * |               \               /                 |
 * |                 InnerNetoutput                  |
 * +-------------------------------------------------+
 * Main
 * +-------------------------------------------------+
 * |      Data(-1)   InnerData(stream_num)           |
 * |            \      /                             |
 * |          SplitRtStreams                         |
 * |               |                                 |
 * |                                                 |
 * +-------------------------------------------------+
 */
std::vector<bg::ValueHolderPtr> LoweringGlobalData::LoweringAndSplitRtStreams(int64_t stream_num) {
  ExecuteGraphType graph_type = ExecuteGraphType::kMain;
  if (CurrentOnInitGraph()) {
    graph_type = ExecuteGraphType::kInit;
    GE_ASSERT_TRUE(stream_num == 1);
  }

  if (graph_type == ExecuteGraphType::kMain) {
    is_single_stream_scene_ = (stream_num <= 1);
  }
  const auto stream_num_holder = bg::ValueHolder::CreateConst(&stream_num, sizeof(stream_num));
  GE_ASSERT_NOTNULL(stream_num_holder);
  auto execute_arg_streams = bg::ValueHolder::CreateFeed(-1);
  GE_ASSERT_NOTNULL(execute_arg_streams);
  auto streams =
      bg::ValueHolder::CreateDataOutput("SplitRtStreams", {execute_arg_streams, stream_num_holder}, stream_num);
  streams_.holders[static_cast<size_t>(graph_type)] = streams;
  return streams_.holders[static_cast<size_t>(graph_type)];
}

void LoweringGlobalData::SetRtNotifies(const std::vector<bg::ValueHolderPtr> &notify_holders) {
  ExecuteGraphType graph_type = ExecuteGraphType::kMain;
  if (CurrentOnInitGraph()) {
    graph_type = ExecuteGraphType::kInit;
  }
  notifies_.holders[static_cast<size_t>(graph_type)] = notify_holders;
}

bg::ValueHolderPtr LoweringGlobalData::GetOrCreateAllL2Allocators() {
  return GetOrCreateL2Allocators({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput}, *this);
}
}  // namespace gert
