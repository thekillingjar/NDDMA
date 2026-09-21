/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_SUBSCRIBER_HOST_EXECUTOR_DUMPER_H_
#define AIR_CXX_RUNTIME_SUBSCRIBER_HOST_EXECUTOR_DUMPER_H_

#include <vector>
#include <set>
#include <mutex>
#include "common/dump/dump_manager.h"
#include "common/dump/dump_op.h"
#include "core/builder/node_types.h"
#include "core/execution_data.h"
#include "exe_graph/runtime/context_extend.h"
#include "exe_graph/runtime/kernel_context.h"
#include "lowering/pass_changed_kernels_info.h"
#include "lowering/placement/placed_lowering_result.h"
#include "runtime/base.h"
#include "runtime/subscriber/executor_subscriber_c.h"
#include "runtime/subscriber/global_dumper.h"
#include "common/model/ge_root_model.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "subscriber/dumper/executor_dumper.h"

namespace gert {
class HostExecutorDumper : public ExecutorDumper {
 public:
  explicit HostExecutorDumper(const std::shared_ptr<const SubscriberExtendInfo> &extend_info);
  static void OnExecuteEvent(int32_t sub_exe_graph_type, HostExecutorDumper *dumper, ExecutorEvent event,
                             const void *node, KernelStatus result);

  ge::Status HostDataDump(const Node *node, ExecutorEvent event);
  ge::Status DoHostDataDump(NodeDumpUnit &dump_unit, const ge::DumpProperties &dump_properties);
  void ParseDumpStep();
  bool IsInDumpStep(const int64_t step_id, const std::string &dump_step);

private:
  ge::Status OnUpdateDumpUnitForHostDump(const Node &node);
  void SetOpDescInfo(NodeDumpUnit &dump_unit, ge::OpDescPtr &op_desc, ge::OpDescInfo &op_desc_info,
                     const std::vector<uintptr_t> &input_addrs, const std::vector<uintptr_t> &output_addrs) const;
  ge::Status GetDumpAddrFromChainAddrOnHost(const NodeDumpUnit &dump_unit, bool is_input,
                                            std::vector<uintptr_t> &dump_addrs) const;

private:
  std::shared_ptr<const SubscriberExtendInfo> extend_info_{nullptr};
  std::vector<std::pair<int64_t, int64_t>> step_range_;
  std::set<int64_t> step_set_;
  bool is_parsed_{false};
  static std::mutex mutex_;
};
}  // namespace gert
#endif
