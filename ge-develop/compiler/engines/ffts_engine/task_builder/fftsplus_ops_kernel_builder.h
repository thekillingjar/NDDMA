/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_ENGIINE_TASK_BUILDER_FFTSPLUS_OPS_KERNEL_BUILDER_H_
#define FFTS_ENGIINE_TASK_BUILDER_FFTSPLUS_OPS_KERNEL_BUILDER_H_

#include "inc/ffts_error_codes.h"
#include "inc/memory_slice.h"
#include "common/opskernel/ops_kernel_builder.h"
#include "common/ffts_plugin_manager.h"
#include "task_builder/mode/data_task_builder.h"
#include "task_builder/data_ctx/cache_persistent_manual_task_builder.h"
#include "task_builder/mode/thread_task_builder.h"
#include "task_builder/mode/manual/manual_thread_task_builder.h"
#include "task_builder/mode/mixl2/mixl2_mode_task_builder.h"
#include "task_builder/mode/auto/auto_thread_task_builder.h"

namespace ffts {
using TheadTaskBuilderPtr = std::shared_ptr<TheadTaskBuilder>;
using ManualTheadTaskBuilderPtr = std::shared_ptr<ManualTheadTaskBuilder>;
using AutoTheadTaskBuilderPtr = std::shared_ptr<AutoTheadTaskBuilder>;
using Mixl2ModeTaskBuilderPtr = std::shared_ptr<Mixl2ModeTaskBuilder>;
using PluginManagerPtr = std::shared_ptr<PluginManager>;
using ScheculePolicyPassFunc = std::function<Status(domi::TaskDef &, std::vector<ffts::FftsPlusContextPath> &)>;

struct ContextParam {
  uint32_t ctx_id_;
  FftsPlusContextPath context_path_;
  std::vector<uint32_t> remove_ctx_list_;
  std::vector<uint32_t> reserve_ctx_list_;
};

struct TimeLineOptimizerContext {
  std::vector<FftsPlusContextPath> ctx_path_vector_;
  std::unordered_map<uint32_t, vector<uint32_t>> cmo_id_map_;
};
class FFTSPlusOpsKernelBuilder : public ge::OpsKernelBuilder {
 public:
  /**
   * Constructor for AICoreOpsKernelBuilder
   */
  FFTSPlusOpsKernelBuilder();

  /**
   * Deconstruction for AICoreOpsKernelBuilder
   */
  ~FFTSPlusOpsKernelBuilder() override;

  /**
   * Initialization
   * @param options
   * @return Status SUCCESS / FAILED or others
   */
  Status Initialize(const std::map<std::string, std::string> &options) override;

  /**
   * Finalization
   * @return Status SUCCESS / FAILED or others
   */
  Status Finalize() override;

  /**
   * Calculate the running parameters for node
   * @param node node object
   * @return Status SUCCESS / FAILED or others
   */
  Status CalcOpRunningParam(ge::Node &node) override;

  /**
   * Generate task for node
   * @param node node object
   * @param context context object
   * @param tasks Task list
   * @return Status SUCCESS / FAILED or others
   */
  using ge::OpsKernelBuilder::GenerateTask;
  Status GenerateTask(const ge::Node &node, ge::RunContext &context, std::vector<domi::TaskDef> &task_defs) override;

 private:
  TheadTaskBuilderPtr GetNormBuilder(const ge::Node &node, ge::ComputeGraphPtr &sgt_graph,
                                     domi::TaskDef &task_def, uint64_t &ready_num, uint64_t &total_num);
  Status GenPersistentContext(const ge::Node &node, uint64_t &ready_context_num, uint64_t &total_context_number,
                              domi::TaskDef &task_def) const;
  TheadTaskBuilderPtr GetFftsPlusMode(const ge::Node &part_node, const ge::ComputeGraph &sgt_graph);
  Status GenerateAutoThreadTask();
  Status GenerateManualThreadTask();
  Status WritePrefetchBitmapToFirst64Bytes(domi::FftsPlusTaskDef *ffts_plus_task_def);
  std::string ConvSqeTypeToStr(uint32_t context_type) const;
  Status GenSubGraphSqeDef(domi::TaskDef &task_def, const uint64_t &ready_context_num,
                           const ge::ComputeGraph &sgt_graph) const;
  Status ChooseGenFftsPlusContextId(TheadTaskBuilderPtr base_mode_ptr,
                                    ge::ComputeGraphPtr sgt_graph,
                                    std::vector<ge::NodePtr> &sub_graph_nodes,
                                    const ge::Node &node,
                                    std::pair<uint64_t, uint64_t> &context_num) const;
  Status InitLibPath();
  Status SetSingleCtxPolicyPri(uint32_t type, domi::FftsPlusCtxDef *ffts_plus_ctx_def,
                               FftsPlusContextPath &ctx_path) const;
  Status SetCtxsPolicyPri(uint64_t ready_context_num, domi::TaskDef &task_def,
                          TimeLineOptimizerContext &timeCtx) const;
  Status UpdateContextForRemoveDuplicate(domi::FftsPlusTaskDef *ffts_plus_task_def,
                                         const ContextParam &ctx_param,
                                         const std::unordered_map<uint32_t, size_t> &ctx_index_map,
                                         TimeLineOptimizerContext &timeCtx) const;
  Status RemoveDuplicateDependencies(domi::TaskDef &task_def, TimeLineOptimizerContext &timeCtx) const;
  bool IsNoCtx(const ge::NodePtr &node) const;
  Status GenSerialDependency(const ge::ComputeGraphPtr &sub_graph) const;
  Status TimelineLayoutOptimize(uint64_t ready_context_num, const ge::Node &node, domi::TaskDef &task_def) const;
  void PrintTaskDefContent(domi::FftsPlusTaskDef *ffts_plus_task_def) const;
  void SortContextPathByMaxPreIndex(const vector<FftsPlusContextPath> &context_paths,
                                    vector<FftsPlusContextPath> &sort_context_paths) const;
  void TransitiveReductionContextPath(const vector<FftsPlusContextPath> &sort_context_paths,
                                      map<uint32_t, vector<uint32_t>> &real_ctx_succ_list) const;
  Status InsertCmoCtxToSucclist(domi::FftsPlusTaskDef *ffts_plus_task_def,
                                uint32_t cmo_id, ContextParam &ctx_param,
                                std::unordered_map<uint32_t, size_t> &ctx_index_map) const;
  Status UpdateContexts(domi::FftsPlusTaskDef *ffts_plus_task_def,
                        map<uint32_t, vector<uint32_t>> &real_ctx_succ_list,
                        TimeLineOptimizerContext &timeCtx) const;
  Status InsertCxtListByPriority(const vector<FftsPlusContextPath> &context_paths, uint32_t insert_ctx_id,
                                 vector<uint32_t> &ctx_list,
                                 const std::unordered_map<uint32_t, size_t> &ctx_index_map) const;
  Status UpdateContextsPreList(domi::TaskDef &task_def, const vector<FftsPlusContextPath> &context_paths,
                               std::unordered_map<uint32_t, vector<uint32_t>> &cmo_id_map) const;
  Status GenLabelForPreList(domi::FftsPlusTaskDef *ffts_plus_task_def, const uint32_t ctx_id,
                            const vector<uint32_t> &pre_list) const;
  Status RemoveFromPreList(const uint32_t ctx_id, const uint32_t next_ctx_id, domi::FftsPlusTaskDef *ffts_plus_task_def,
                           TimeLineOptimizerContext &timeCtx,
                           const std::unordered_map<uint32_t, size_t> &ctx_index_map) const;
  bool GenerateCtxIdIdxMap(const vector<FftsPlusContextPath> &context_paths,
                           std::unordered_map<uint32_t, size_t> &ctx_index_map) const;
  Status GetIndexByCtxId(uint32_t ctx_id, size_t &path_index,
                         const std::unordered_map<uint32_t, size_t> &ctx_index_map) const;

  Status ReBuildCtxIdsRelation(domi::TaskDef &task_def,
                               std::unordered_map<uint32_t, uint32_t> &new_old_map,
                               std::unordered_map<uint32_t, uint32_t> &old_new_map) const;
  Status GenNewSubGraphTaskDef(const ge::Node &node,
                               domi::TaskDef &task_def,
                               domi::TaskDef &task_def_new,
                               std::unordered_map<uint32_t, uint32_t> &new_old_map,
                               std::unordered_map<uint32_t, uint32_t> &old_new_map) const;
  Status ConvertOldCtxInfoToNewCtx(domi::FftsPlusCtxDef *ctx_def_old,
                                   domi::FftsPlusCtxDef *ctx_def_new,
                                   std::unordered_map<uint32_t, uint32_t> &old_new_map) const;
  Status GenerateExtTask(const ge::Node &node, ge::RunContext &context, std::vector<domi::TaskDef> &task_defs);
  Status UpdateCtxSuccList(domi::FftsPlusCtxDef *ctx_def_old, domi::FftsPlusCtxDef *ctx_def_new,
                           std::unordered_map<uint32_t, uint32_t> &old_new_map) const;
  Status UpdateCtxSrcSlotId(domi::FftsPlusCtxDef *ctx_def_old, domi::FftsPlusCtxDef *ctx_def_new,
                            std::unordered_map<uint32_t, uint32_t> &old_new_map) const;
  template <typename T>
  static Status UpdateNewCtxSuccList(T *ctx_old, T *ctx_new, std::unordered_map<uint32_t, uint32_t> &old_new_map) {
    FFTS_CHECK_NOTNULL(ctx_old);
    FFTS_CHECK_NOTNULL(ctx_new);
    ctx_new->set_successor_num(ctx_old->successor_num());
    for (size_t i = 0; i < ctx_old->successor_num(); ++i) {
      ctx_new->set_successor_list(i, old_new_map[ctx_old->successor_list(i)]);
    }
    return SUCCESS;
  }
  template <typename T>
  static Status UpdateNewCtxSrcSlotId(T *ctx_old, T *ctx_new, std::unordered_map<uint32_t, uint32_t> &old_new_map) {
    FFTS_CHECK_NOTNULL(ctx_old);
    FFTS_CHECK_NOTNULL(ctx_new);
    for (size_t i = 0; i < static_cast<size_t>(ctx_old->src_slot_size()); ++i) {
      ctx_new->set_src_slot(i, old_new_map[ctx_old->src_slot(i)]);
    }
    return SUCCESS;
  }
  template <typename T>
  static Status UpdateNewCtxSuccListForCondSwitch(T *ctx_old, T *ctx_new,
                                                  std::unordered_map<uint32_t, uint32_t> &old_new_map) {
    FFTS_CHECK_NOTNULL(ctx_old);
    FFTS_CHECK_NOTNULL(ctx_new);
    ctx_new->set_true_successor_num(ctx_old->true_successor_num());
    ctx_new->set_false_successor_num(ctx_old->false_successor_num());
    for (uint32_t i = 0; i < ctx_old->true_successor_num(); ++i) {
      ctx_new->set_true_successor_list(i, old_new_map[ctx_old->true_successor_list(i)]);
    }
    for (uint32_t i = 0; i < ctx_old->false_successor_num(); ++i) {
      ctx_new->set_false_successor_list(i, old_new_map[ctx_old->false_successor_list(i)]);
    }
    return SUCCESS;
  }
 private:
  PluginManagerPtr sch_policy_pass_plugin_;
  ScheculePolicyPassFunc schecule_policy_pass_{nullptr};
  FFTSPlusTaskBuilderPtr ffts_plus_task_builder_ptr_{nullptr};
  std::string lib_path_;
  bool init_flag_{false};
  bool skip_schecule_policy_pass_{false};
};
}  // namespace ffts
#endif  // FFTS_ENGIINE_TASK_BUILDER_FFTSPLUS_OPS_KERNEL_BUILDER_H_
