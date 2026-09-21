/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_event.h"

#include "common/checker.h"
#include "exe_graph/lowering/frame_selector.h"
#include "graph/debug/ge_attr_define.h"
#include "kernel/common_kernel_impl/event.h"
namespace gert {
namespace bg {
namespace {
constexpr const ge::char_t *kGlobalDataGertEvents = "GertEvents";

ge::Status CollectFirstSyncEventsInfo(const ge::ComputeGraphPtr &compute_graph, std::vector<EventInfo> &event_infos) {
  for (const auto node : compute_graph->GetDirectNodePtr()) {
    const auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    if (op_desc->GetStreamId() <= kDefaultMainStreamId) {
      continue;
    }
    if (node->GetInAllNodes().empty()) {
      std::vector<int64_t> recive_id_list;
      (void)ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_RECV_EVENT_IDS, recive_id_list);
      if (recive_id_list.empty()) {
        EventInfo event_info(kDefaultMainStreamId, op_desc->GetStreamId());
        event_infos.emplace_back(event_info);
      }
    }
  }
  return ge::SUCCESS;
}
ge::Status CollectLastSyncEventsInfo(const ge::ComputeGraphPtr &compute_graph, std::vector<EventInfo> &event_infos) {
  for (const auto node : compute_graph->GetDirectNodePtr()) {
    const auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    if (op_desc->GetStreamId() <= kDefaultMainStreamId) {
      continue;
    }
    if (node->GetOutAllNodes().empty()) {
      std::vector<int64_t> send_id_list;
      (void)ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_SEND_EVENT_IDS, send_id_list);
      if (send_id_list.empty()) {
        EventInfo event_info(op_desc->GetStreamId(), kDefaultMainStreamId);
        event_infos.emplace_back(event_info);
      }
    }
  }
  return ge::SUCCESS;
}

void CollectLastResourceCleanEventsInfo(int64_t stream_num, std::vector<EventInfo> &event_infos) {
  if (stream_num <= 1) {
    return;
  }
  for (int64_t i = 1; i < stream_num; ++i) {
    EventInfo event_info(i, kDefaultMainStreamId);
    event_infos.emplace_back(event_info);
  }
}

ge::graphStatus CollectAllEventInfos(const ge::ComputeGraphPtr &compute_graph, int64_t event_num,
                                     std::vector<EventInfo> &event_infos) {
  event_infos.resize(event_num); // todo safe
  std::unordered_set<int64_t> event_ids;
  std::unordered_set<int64_t> send_event_ids;
  std::unordered_set<int64_t> recive_event_ids;
  for (const auto node : compute_graph->GetAllNodesPtr()) {
    if (!node->GetOwnerComputeGraphBarePtr()->GetGraphUnknownFlag()) {
      continue;
    }
    const auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    std::vector<int64_t> send_id_list;
    std::vector<int64_t> recive_id_list;
    (void)ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_SEND_EVENT_IDS, send_id_list);
    (void)ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_RECV_EVENT_IDS, recive_id_list);
    const int64_t stream_id = op_desc->GetStreamId();
    for (const auto send_event_id : send_id_list) {
      event_ids.insert(send_event_id);
      GE_ASSERT_TRUE(send_event_ids.insert(send_event_id).second, "Found duplicated send event id %ld on node %s.",
                     send_event_id, node->GetNamePtr());

      GE_ASSERT_TRUE(send_event_id < event_num, "Found send event id %ld out of range %ld.", send_event_id, event_num);
      event_infos[send_event_id].src_logic_stream_id = stream_id;
    }
    for (const auto recive_event_id : recive_id_list) {
      event_ids.insert(recive_event_id);
      GE_ASSERT_TRUE(recive_event_ids.insert(recive_event_id).second, "Found duplicated send event id %ld on node %s.",
                     recive_event_id, node->GetNamePtr());

      GE_ASSERT_TRUE(recive_event_id < event_num, "Found recive event id %ld out of range %ld.", recive_event_id,
                     event_num);
      event_infos[recive_event_id].dst_logic_stream_id = stream_id;
    }
  }
  GE_ASSERT_TRUE(event_ids.size() == static_cast<size_t>(event_num),
                 "EvenId in nodes is not equal with event num %lld on model.", event_num);
  for (size_t i = 0U; i < event_infos.size(); ++i) {
    GE_ASSERT_TRUE(event_infos[i].HasInitialized(), "Found event %zu no in pair.", i);
  }
  return ge::GRAPH_SUCCESS;
}
void AppendEventInfos(EventInfo *event_info_addrs, const std::vector<EventInfo> event_infos, size_t start_index) {
  for (size_t i = 0U; i < event_infos.size(); ++i) {
    event_info_addrs[start_index + i] = event_infos[i];
  }
}

ValueHolderPtr CreateConstOfAllEventInfos(const std::vector<EventInfo> &model_event_info,
                                          const std::vector<std::vector<EventInfo>> &stage_2_sync_events) {
  GE_ASSERT_EQ(stage_2_sync_events.size(), static_cast<size_t>(SyncEventStage::kStageEnd));
  auto &first_sync_event_info = stage_2_sync_events[static_cast<size_t>(SyncEventStage::kFirstSyncStage)];
  auto &last_sync_event_info = stage_2_sync_events[static_cast<size_t>(SyncEventStage::kLastSyncStage)];
  auto &last_resource_clean_events_info =
      stage_2_sync_events[static_cast<size_t>(SyncEventStage::kLastResourceCleanStage)];

  const auto all_event_num = model_event_info.size() + first_sync_event_info.size() + last_sync_event_info.size() +
                             last_resource_clean_events_info.size();
  size_t total_size = 0U;
  auto all_event_infos = ContinuousVector::Create<EventInfo>(all_event_num, total_size);
  auto all_event_infos_vec = reinterpret_cast<ContinuousVector *>(all_event_infos.get());
  GE_ASSERT_NOTNULL(all_event_infos_vec);
  all_event_infos_vec->SetSize(all_event_num);
  auto event_info_addrs = reinterpret_cast<EventInfo *>(all_event_infos_vec->MutableData());
  AppendEventInfos(event_info_addrs, model_event_info, 0U);
  AppendEventInfos(event_info_addrs, first_sync_event_info, model_event_info.size());
  AppendEventInfos(event_info_addrs, last_sync_event_info, model_event_info.size() + first_sync_event_info.size());
  AppendEventInfos(event_info_addrs, last_resource_clean_events_info,
                   model_event_info.size() + first_sync_event_info.size() + last_sync_event_info.size());
  return ValueHolder::CreateConst(all_event_infos.get(), total_size);
}
} // namespace

ValueHolderPtr CollectAndCreateGertEvents(const ge::ComputeGraphPtr &compute_graph, const ModelDesc &model_desc,
                                          LoweringGlobalData &global_data,
                                          std::vector<std::vector<EventInfo>> &stage_2_sync_events) {
  std::vector<EventInfo> model_event_info;
  const int64_t event_num = model_desc.GetReusableEventNum();
  GE_ASSERT_GRAPH_SUCCESS(CollectAllEventInfos(compute_graph, event_num, model_event_info));
  GE_ASSERT_EQ(stage_2_sync_events.size(), static_cast<size_t>(SyncEventStage::kStageEnd));
  auto &first_sync_events = stage_2_sync_events[static_cast<size_t>(SyncEventStage::kFirstSyncStage)];
  auto &last_sync_events = stage_2_sync_events[static_cast<size_t>(SyncEventStage::kLastSyncStage)];
  auto &last_resource_clean_events = stage_2_sync_events[static_cast<size_t>(SyncEventStage::kLastResourceCleanStage)];
  GE_ASSERT_SUCCESS(CollectFirstSyncEventsInfo(compute_graph, first_sync_events));
  GE_ASSERT_SUCCESS(CollectLastSyncEventsInfo(compute_graph, last_sync_events));
  CollectLastResourceCleanEventsInfo(model_desc.GetReusableStreamNum(), last_resource_clean_events);

  if (model_event_info.empty() && first_sync_events.empty() && last_sync_events.empty() &&
      last_resource_clean_events.empty()) {
    return nullptr;
  }

  auto init_out = FrameSelector::OnInitRoot(
      [&model_event_info, &stage_2_sync_events]() -> std::vector<ValueHolderPtr> {
        auto all_events_info = CreateConstOfAllEventInfos(model_event_info, stage_2_sync_events);
        GE_ASSERT_NOTNULL(all_events_info);
        return {ValueHolder::CreateSingleDataOutput("CreateGertEvents", {all_events_info})};
      });
  GE_ASSERT_TRUE(!init_out.empty());
  auto builder = [&init_out]() -> ValueHolderPtr {
    return init_out[0];
  };
  return global_data.GetOrCreateUniqueValueHolder(bg::kGlobalDataGertEvents, builder);
}

void LoweringFirstSyncEvents(const std::vector<EventInfo> &first_sync_events, int64_t current_logic_event_num,
                             LoweringGlobalData &global_data) {
  if (first_sync_events.empty()) {
    return;
  }
  std::vector<int64_t> logic_event_ids(first_sync_events.size());
  for (size_t i = 0U; i < first_sync_events.size(); ++i) {
    logic_event_ids[i] = current_logic_event_num + i;
  }

  auto first_event_sync_builder = [&]() -> std::vector<bg::ValueHolderPtr> {
    auto send_holder = bg::SendEvents(kDefaultMainStreamId, logic_event_ids, global_data);
    std::vector<bg::ValueHolderPtr> wait_holders;
    for (size_t i = 0U; i < logic_event_ids.size(); ++i) {
      auto wait_holder = bg::WaitEvents(first_sync_events[i].dst_logic_stream_id, {logic_event_ids[i]}, global_data);
      bg::ValueHolder::AddDependency(send_holder, wait_holder);
      wait_holders.emplace_back(wait_holder);
    }
    return wait_holders;
  };

  bg::FrameSelector::OnMainRootFirst(first_event_sync_builder);
  return;
}

void LoweringLastSyncEvents(const std::vector<EventInfo> &last_sync_events, int64_t current_logic_event_num,
                            LoweringGlobalData &global_data) {
  if (last_sync_events.empty()) {
    return;
  }
  std::vector<int64_t> wait_logic_event_ids(last_sync_events.size());
  for (size_t i = 0U; i < last_sync_events.size(); ++i) {
    wait_logic_event_ids[i] = current_logic_event_num + i;
  }
  auto last_event_sync_builder = [&]() -> std::vector<bg::ValueHolderPtr> {
    auto wait_holder = bg::WaitEvents(kDefaultMainStreamId, wait_logic_event_ids, global_data);

    std::vector<bg::ValueHolderPtr> send_holders;
    for (size_t i = 0U; i < last_sync_events.size(); ++i) {
      auto event_info = last_sync_events[i];
      std::vector<int64_t> send_logic_event_ids = {current_logic_event_num + static_cast<int64_t>(i)};
      auto send_holder = bg::SendEvents(event_info.src_logic_stream_id, send_logic_event_ids, global_data);
      bg::ValueHolder::AddDependency(send_holder, wait_holder);
      send_holders.emplace_back(send_holder);
    }
    return {wait_holder};
  };
  bg::FrameSelector::OnMainRootLastEventSync(last_event_sync_builder);
}

std::vector<ValueHolderPtr> PrepareEventInputs(int64_t logic_stream_id, const std::vector<int64_t> &logic_event_ids,
                                               LoweringGlobalData &global_data) {
  std::vector<ValueHolderPtr> inputs;

  auto src_stream_holder = global_data.GetStreamById(logic_stream_id);
  inputs.emplace_back(src_stream_holder);

  size_t total_size = 0U;
  auto event_id_vec_holder = ContinuousVector::Create<int64_t>(logic_event_ids.size(), total_size);
  auto evend_id_vec = reinterpret_cast<ContinuousVector *>(event_id_vec_holder.get());
  GE_ASSERT_NOTNULL(evend_id_vec);
  evend_id_vec->SetSize(logic_event_ids.size());
  for (size_t i = 0U; i < logic_event_ids.size(); ++i) {
    *(reinterpret_cast<int64_t *>(evend_id_vec->MutableData()) + i) = logic_event_ids[i];
  }
  auto logic_event_ids_holder = ValueHolder::CreateConst(event_id_vec_holder.get(), total_size);
  inputs.emplace_back(logic_event_ids_holder);

  auto all_gert_events_holder = global_data.GetUniqueValueHolder(bg::kGlobalDataGertEvents);
  GE_ASSERT_NOTNULL(all_gert_events_holder);
  inputs.emplace_back(all_gert_events_holder);

  auto all_rt_events_holder = global_data.GetUniqueValueHolder(bg::kGlobalDataRtEvents);
  GE_ASSERT_NOTNULL(all_rt_events_holder);
  inputs.emplace_back(all_rt_events_holder);

  auto l2_allocator =
      global_data.GetOrCreateL2Allocator(logic_stream_id, {kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  GE_ASSERT_NOTNULL(l2_allocator);
  inputs.emplace_back(l2_allocator);
  return inputs;
}

void LoweringLastResourceCleanEvents(const std::vector<EventInfo> &last_sync_events, int64_t current_logic_event_num,
                                     LoweringGlobalData &global_data) {
  if (last_sync_events.empty()) {
    return;
  }
  std::vector<int64_t> logic_event_ids(last_sync_events.size());
  for (size_t i = 0U; i < last_sync_events.size(); ++i) {
    logic_event_ids[i] = current_logic_event_num + i;
  }
  auto last_event_sync_builder = [&]() -> std::vector<bg::ValueHolderPtr> {
    auto inputs = PrepareEventInputs(kDefaultMainStreamId, logic_event_ids, global_data);
    // last wait events requiered all l2 allocators
    inputs[static_cast<size_t>(SendEventsInput::kAllocator)] = global_data.GetOrCreateAllL2Allocators();
    auto wait_holder = ValueHolder::CreateVoid<bg::ValueHolder>(
        "LastWaitEvents", inputs);

    std::vector<bg::ValueHolderPtr> send_holders;
    for (size_t i = 0U; i < last_sync_events.size(); ++i) {
      auto event_info = last_sync_events[i];
      std::vector<int64_t> logical_event_ids = {current_logic_event_num + static_cast<int64_t>(i)};
      auto send_holder = ValueHolder::CreateVoid<bg::ValueHolder>(
          "LastSendEvents", PrepareEventInputs(event_info.src_logic_stream_id, logical_event_ids, global_data));
      bg::ValueHolder::AddDependency(send_holder, wait_holder);
      send_holders.emplace_back(send_holder);
    }
    return {wait_holder};
  };
  bg::FrameSelector::OnMainRootLastResourceClean(last_event_sync_builder);
}

ValueHolderPtr SendEvents(int64_t logic_stream_id, const std::vector<int64_t> &logic_event_ids,
                          LoweringGlobalData &global_data) {
  return ValueHolder::CreateVoid<bg::ValueHolder>("SendEvents",
                                                  PrepareEventInputs(logic_stream_id, logic_event_ids, global_data));
}

ValueHolderPtr WaitEvents(int64_t logic_stream_id, const std::vector<int64_t> &logic_event_ids,
                          LoweringGlobalData &global_data) {
  return ValueHolder::CreateVoid<bg::ValueHolder>("WaitEvents",
                                                  PrepareEventInputs(logic_stream_id, logic_event_ids, global_data));
}

HyperStatus LoweringAccessMemCrossStream(const ge::NodePtr &curr_compute_node,
                                         std::vector<bg::DevMemValueHolderPtr> &input_addrs) {
  const auto curr_stream_id = curr_compute_node->GetOpDescBarePtr()->GetStreamId();
  // lowering accessMemCrossStream
  bg::ValueHolderPtr curr_stream_id_holder = nullptr;
  for (size_t i = 0U; i < input_addrs.size(); ++i) {
    auto addr_holder = input_addrs[i];
    if (addr_holder->GetLogicStream() == curr_stream_id) {
      continue;
    }
    if (curr_stream_id_holder == nullptr) {
      curr_stream_id_holder = bg::ValueHolder::CreateConst(&curr_stream_id, sizeof(curr_stream_id));
    }
    GELOGD("LoweringAccessMemCrossStream, node name: %s, input index: %u, curr_stream_id: %lld, logic stream id: %lld",
           curr_compute_node->GetNamePtr(), i, curr_stream_id, addr_holder->GetLogicStream());
    auto access_holder = bg::DevMemValueHolder::CreateSingleDataOutput(
        "AccessMemCrossStream", {addr_holder, curr_stream_id_holder}, curr_stream_id);
    access_holder->SetPlacement(addr_holder->GetPlacement());
    bg::ValueHolder::CreateVoidGuarder("FreeMemory", access_holder, {});
    input_addrs[i] = access_holder;
  }
  return HyperStatus::Success();
}
}  // namespace bg
}  // namespace gert

