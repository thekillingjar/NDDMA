/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_RT_EVENT_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_RT_EVENT_H_

#include <framework/runtime/model_desc.h>
#include "exe_graph/lowering/dev_mem_value_holder.h"
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/lowering/lowering_global_data.h"
namespace gert {
namespace bg {
constexpr const ge::char_t *kGlobalDataRtEvents = "ExecuteArgRtEvents";
const int64_t kInValidStreamId = -1;
enum class SyncEventStage { kFirstSyncStage = 0, kLastSyncStage, kLastResourceCleanStage, kStageEnd };
struct EventInfo {
  int64_t src_logic_stream_id;
  int64_t dst_logic_stream_id;
  EventInfo() : src_logic_stream_id(kInValidStreamId), dst_logic_stream_id(kInValidStreamId) {}
  EventInfo(int64_t src_stream, int64_t dst_stream)
      : src_logic_stream_id(src_stream), dst_logic_stream_id(dst_stream) {}
  bool HasInitialized() const {
    return (src_logic_stream_id != kInValidStreamId) && (dst_logic_stream_id != kInValidStreamId);
  }
};
/*
 *    Const(EventInfos)
 *       |
 *    CreateGertEvents
 *
 * 收集模型所需所有event，包括模型开始和结束同步的event。创建对应GertEvent。
 */
ValueHolderPtr CollectAndCreateGertEvents(const ge::ComputeGraphPtr &compute_graph, const ModelDesc &model_desc,
                                          LoweringGlobalData &global_data,
                                          std::vector<std::vector<EventInfo>> &stage_2_sync_events);
void LoweringFirstSyncEvents(const std::vector<EventInfo> &first_sync_events, int64_t current_logic_event_num,
                             LoweringGlobalData &global_data);
void LoweringLastSyncEvents(const std::vector<EventInfo> &last_sync_events, int64_t current_logic_event_num,
                            LoweringGlobalData &global_data);
void LoweringLastResourceCleanEvents(const std::vector<EventInfo> &last_sync_events, int64_t current_logic_event_num,
                                     LoweringGlobalData &global_data);

/*
 *
 *
 *     Const(logic_stream_id)           Const(event_ids)  CreateGertEvents   Data(rt_events)   SelectL2Allocator
 *                |        \                   |                 |                  |                 |
 *                |        GetStreamById       |                 |                  |                 |
 *                |            |               |                 |                  |                 |
 *                +-------->SendEvents<--------+-----------------+------------------+-----------------+
 *
 */
ValueHolderPtr SendEvents(int64_t logic_stream_id, const std::vector<int64_t> &logic_event_ids,
                          LoweringGlobalData &global_data);
/*
 *
 *
 *     Const(logic_stream_id)           Const(event_ids)  CreateGertEvents   Data(rt_events)   SelectL2Allocator
 *                |        \                   |                 |                  |                 |
 *                |        GetStreamById       |                 |                  |                 |
 *                |            |               |                 |                  |                 |
 *                +-------->WaitEvents<--------+-----------------+------------------+-----------------+
 *
 */
ValueHolderPtr WaitEvents(int64_t logic_stream_id, const std::vector<int64_t> &logic_event_ids,
                          LoweringGlobalData &global_data);

HyperStatus LoweringAccessMemCrossStream(const ge::NodePtr &curr_compute_node,
                                  std::vector<bg::DevMemValueHolderPtr> &input_addrs);
}  // namespace bg
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_RT_STREAM_H_
