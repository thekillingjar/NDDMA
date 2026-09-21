/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_SUBSCRIBER_EXECUTOR_DUMPER_H_
#define AIR_CXX_RUNTIME_SUBSCRIBER_EXECUTOR_DUMPER_H_
#include <unordered_map>
#include <utility>
#include <vector>
#include "common/dump/dump_manager.h"
#include "common/dump/dump_op.h"
#include "common/dump/exception_dumper.h"
#include "common/dump/data_dumper.h"
#include "common/dump/kernel_tracing_utils.h"
#include "common/dump/opdebug_register.h"
#include "core/builder/node_types.h"
#include "core/execution_data.h"
#include "exe_graph/runtime/context_extend.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/dfx_info_filler.h"
#include "register/kernel_registry.h"
#include "lowering/pass_changed_kernels_info.h"
#include "lowering/placement/placed_lowering_result.h"
#include "runtime/base.h"
#include "runtime/model_v2_executor.h"
#include "runtime/subscriber/built_in_subscriber_definitions.h"
#include "runtime/subscriber/executor_subscriber_c.h"
#include "runtime/subscriber/global_dumper.h"
#include "common/model/ge_root_model.h"
#include "exe_graph/runtime/extended_kernel_context.h"

namespace gert {
struct NodeDumpUnit {
  void UpdateInputShapes(ge::OpDescPtr &op_desc);
  void UpdateOutputShapes(ge::OpDescPtr &op_desc);
  void Clear() {
    cur_update_count = 0UL;
    need_overflow_dump = false;
    context_list.clear();
  }
  // compute node to dump
  ge::NodePtr node;
  size_t total_update_count{0UL};
  size_t cur_update_count{0UL};
  std::vector<Chain *> input_shapes{};
  std::vector<Chain *> output_shapes{};
  std::vector<Chain *> input_addrs{};
  std::vector<Chain *> output_addrs{};

  // overflow flag only for overflow dump
  bool need_overflow_dump = false;

  std::vector<ge::Context> context_list{};
  std::vector<std::pair<uintptr_t, int64_t>> workspace_info{};
};

class ExecutorDataDumpInfoWrapper : public DataDumpInfoWrapper {
 public:
  ExecutorDataDumpInfoWrapper() = default;
  explicit ExecutorDataDumpInfoWrapper(NodeDumpUnit *dump_unit): dump_unit_(dump_unit) {}

  ge::graphStatus CreateFftsCtxInfo(const uint32_t thread_id, const uint32_t context_id) override {
    for (const auto &ctx: dump_unit_->context_list) {
      GE_ASSERT_TRUE(ctx.thread_id != thread_id);
    }
    ge::Context new_ctx = {context_id, thread_id, {}, {}};
    dump_unit_->context_list.emplace_back(new_ctx);
    return ge::SUCCESS;
  }

  ge::graphStatus AddFftsCtxAddr(const uint32_t thread_id, const bool is_input,
                             const uint64_t address, const uint64_t size) override {
    for (auto &ctx: dump_unit_->context_list) {
      if (ctx.thread_id != thread_id) {
        continue;
      }
      ge::RealAddressAndSize addr_info = {address, size};
      if (is_input) {
        ctx.input.emplace_back(addr_info);
      } else {
        ctx.output.emplace_back(addr_info);
      }
      return ge::SUCCESS;
    }
    return ge::FAILED;
  }

  void AddWorkspace(uintptr_t addr, int64_t bytes) override {
    dump_unit_->workspace_info.emplace_back(std::make_pair(addr, bytes));
  }

  bool SetStrAttr(const std::string &name, const std::string &value) override {
    return ge::AttrUtils::SetStr(dump_unit_->node->GetOpDesc(), name, value);
  }

 private:
  NodeDumpUnit *dump_unit_{};
};

struct ExceptionDumpUint {
  std::string tiling_data;
  uint32_t tiling_key{0U};
  bool is_host_args{false};
  uintptr_t args{0U};
  size_t args_size{0UL};
  std::string args_before_execute;
  std::vector<std::pair<uintptr_t, int64_t>> workspace_info{};

  void Clear() {
    tiling_data.clear();
    tiling_key = 0U;
    is_host_args = false;
    args = 0U;
    args_size = 0UL;
    args_before_execute.clear();
    workspace_info.clear();
  }
};

class ExecutorExceptionDumpInfoWrapper : public ExceptionDumpInfoWrapper {
 public:
  ExecutorExceptionDumpInfoWrapper() = default;
  explicit ExecutorExceptionDumpInfoWrapper(ExceptionDumpUint *dump_unit) : dump_unit_(dump_unit) {}

  void SetTilingData(uintptr_t addr, size_t size) override {
    std::stringstream ss;
    PrintHex(reinterpret_cast<uint8_t *>(addr), size, ss);
    dump_unit_->tiling_data = ss.str();
  }

  void SetTilingKey(uint32_t key) override { dump_unit_->tiling_key = key; }

  void SetHostArgs(uintptr_t addr, size_t size) override {
    std::stringstream ss;
    ss << "args before execute: ";
    PrintHex(reinterpret_cast<void **>(addr), size / sizeof(void *), ss);
    dump_unit_->args_before_execute = ss.str();
    dump_unit_->args = addr;
    dump_unit_->args_size = size;
    dump_unit_->is_host_args = true;
  }

  void SetDeviceArgs(uintptr_t addr, size_t size) override {
    (void)addr;
    (void)size;
  }

  void AddWorkspace(uintptr_t addr, int64_t bytes) override {
    dump_unit_->workspace_info.emplace_back(addr, bytes);
  }

 private:
  ExceptionDumpUint *dump_unit_;
};

class ExecutorDumper {
 public:
  explicit ExecutorDumper(const std::shared_ptr<const SubscriberExtendInfo> &extend_info);
  ~ExecutorDumper();
  static void OnExecuteEvent(int32_t sub_exe_graph_type, ExecutorDumper *dumper, ExecutorEvent event,
                             const void *node, KernelStatus result);

  ge::Status Init();
  bool IsEnable(DumpType dump_type) const {
    return global_dumper_->IsEnable(dump_type);
  }

  void CountIterNum() {
    ++iteration_num_;
  }

  ge::Status SaveSessionId();

  ge::Status SaveRtStream();

  uint64_t GetSessionId() const {
    return session_id_;
  }

  uint64_t GetIterationNum() const {
    return iteration_num_;
  }

  ge::Status DataDump(const Node *node, ExecutorEvent event);
  ge::Status OverflowDump(const Node *node, ExecutorEvent event);
  ge::Status ExceptionDump(const Node *node, ExecutorEvent event);

  ge::Status DoDataDump(NodeDumpUnit &dump_unit, const ge::DumpProperties &dump_properties,
                        const Node *exe_node = nullptr);

  static ge::Status OnExecutorDumperSwitch(void *ins, uint64_t enable_flags);
protected:
  ge::Status SaveWorkSpaceAddrForAiCpuLaunchCCNode(const Node &node);

private:
  ge::Status UpdateFftsplusLaunchTask(const Node *node);
 KernelNameAndIdx GetKernelNameAndIdxAfterPass(const ge::OpDesc *op_desc, const KernelNameAndIdx &kernel_name_and_idx,
                                               const NodeDumpUnit *dump_unit) const;
 ge::Status InsertHcclDumpOp(const KernelRunContext &context, ExecutorEvent event);
  ge::Status DumpOpDebug();
  ge::Status ClearDumpOpDebug();
  void GetLastKernelDumpUnits(const Node &node, std::vector<NodeDumpUnit *> &dump_nodes);
  ge::Status SetOverflowDumpFlag(ExecutorEvent event, const Node &node);
  ge::Status OnUpdateDumpUnit(ExecutorEvent event, const Node &node, bool overflow_flag = false);
  ge::Status InitDumpUnits();
  ge::Status CollectLaunchKernelName();
  ge::Status InitOutputShapes(const PlacedLoweringResult &placed_lower_result, NodeDumpUnit *dump_unit);
  ge::Status InitOutputAddrs(const PlacedLoweringResult &placed_lower_result, NodeDumpUnit *dump_unit);

  ge::Status InitOutputChainFromEquivalentDataEdges(const KernelNameAndIdx &kernel_name_and_idx,
                                                    std::vector<Chain *> &output_chain);
  ge::Status InitOutputChainFromMainGraph(const KernelNameAndIdx &kernel_name_and_idx, NodeDumpUnit *dump_unit,
                                          vector<Chain *> &output_chain, bool is_input);
  ge::Status InitOutputChainFromInitGraph(const KernelNameAndIdx &kernel_name_and_idx, const NodeDumpUnit *dump_unit,
                                          std::vector<Chain *> &output_chain, bool is_input);
  ge::Status InitInputShapes(const LowerInputInfo &input_info, NodeDumpUnit *dump_unit);
  ge::Status InitInputAddrs(const LowerInputInfo &input_info, NodeDumpUnit *dump_unit);
  ge::Status InitOrderHoldersFromExeGraph(const std::string &name, NodeDumpUnit *dump_unit);
  ge::Status InitOrderHolders(const PlacedLoweringResult &placed_lower_result, NodeDumpUnit *dump_unit);
  ge::Status InitOrderHoldersFromControlNodes(const bg::ValueHolderPtr &order_holder, NodeDumpUnit *dump_unit);
  void ClearDumpUnits();
  ge::Status InitByGraphType(SubExeGraphType type);
  bool IsOpInDumpList(const ge::DumpProperties &dump_properties, const std::string &op_name) const;
  ge::Status SetDumpFlagForFfts(const std::string &kernel_type, const Node *node) const;
  ge::Status SetDumpFlagForMixl2(const Node *node) const;
  ge::Status FillDumpInfoByKernel(const Node &node);
  ge::Status FillExceptionDumpInfoByKernel(const Node &node);
  ge::Status FillExtraDumpInfo(const Node &node);
  ge::Status PrepareExceptionDump(const Node &node, const char *kernel_type, NodeDumpUnit &dump_unit);

  void LoadDumpTaskForDavinciModels(const bool dump_enable) const;
  bool IsDavinciModelExist() const;
  ge::Status UpdateStepNum();
  ge::Status DoStreamSyncAfterFftsTask(const Node *node);
  bool IsSingleOpScene() const;

  ge::Status ResetDumpFsmState();
  ge::Status SetDumpFsmState(const Node *node, const char *const node_type);
  bool IsInDumpOpRange(const std::string &op_name) const {
    return (dump_op_in_range_.count(op_name) == 1U);
  }

protected:
  std::unordered_map<std::string, ExceptionDumpUint> node_names_to_extra_units_{};  // key computer node name
  std::unordered_map<std::string, Node *> kernel_names_to_exe_nodes_{};
  std::unordered_map<std::string, Node *> init_kernel_names_to_exe_nodes_{};
  std::unordered_map<std::string, NodeDumpUnit> node_names_to_dump_units_{};
  std::vector<std::vector<NodeDumpUnit *>> kernel_idxes_to_dump_units_{};

private:
  std::shared_ptr<const SubscriberExtendInfo> extend_info_{nullptr};
  GlobalDumper *global_dumper_{nullptr};
  bool is_inited_{false};
  size_t iteration_num_{0UL};
  ge::DumpOp ffts_dump_op_;
  std::unordered_map<std::string, ExecutorDataDumpInfoWrapper> node_names_to_dump_unit_wrappers_{};
  std::vector<KernelRegistry::DataDumpInfoFiller> exe_node_id_to_data_dump_filler_{};
  std::vector<KernelRegistry::ExceptionDumpInfoFiller> exe_node_id_to_exception_dump_filler_{};
  uint64_t session_id_{0UL};
  ContinuousVector *streams_;
  std::vector<std::shared_ptr<ge::DataDumper>> data_dumpers_;
  std::vector<std::shared_ptr<ge::OpdebugRegister>> op_debug_registers_;
  bool is_op_debug_reg_;
  bool is_dumper_switch_reg_{false};
  uintptr_t global_step_addr_{0U};
  bool has_davinci_models_{false};
  std::vector<ge::DumpProcState> dump_fsm_state_;
  std::unordered_set<std::string> dump_op_in_range_;
  std::map<std::string, std::string> compute_node_name_to_launch_kernel_name_;
};
}  // namespace gert
#endif
