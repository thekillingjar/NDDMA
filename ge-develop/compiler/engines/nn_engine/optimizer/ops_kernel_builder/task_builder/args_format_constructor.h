/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_ARGS_FROMAT_CONSTRUCTOR_H_
#define GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_ARGS_FROMAT_CONSTRUCTOR_H_
#include <vector>
#include "common/fe_log.h"
#include "proto/task.pb.h"
#include "inc/ffts_type.h"
#include "graph/utils/node_utils.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/args_format_desc.h"
#include "ops_store/ops_kernel_manager.h"

namespace fe {
class ArgsFormatConstructor {
 public:
  ArgsFormatConstructor(const ge::OpDescPtr &op_desc, bool is_ffts_plus) :
      op_desc_(op_desc), is_ffts_plus_(is_ffts_plus) {};
  ~ArgsFormatConstructor() {};
  Status ConstructNodeArgsDesc();
  Status GetArgsSize(size_t &args_size);
  std::string GetArgsFormatString();
 private:
  Status ConstructOutArgsDesc();
  Status ConstructInArgsDesc();
  void ConstructArgsDescByGraph();
  void AddDynamicDesc(const std::pair<size_t, size_t> &range, size_t ir_index, bool is_input);
  bool FindOptInsertPos(size_t ir_idx, const std::vector<InputOrOutputInfoPtr>& input_infos,
      std::vector<std::string> &input_name_list, size_t &insert_pos) const;
  bool InsertMissOptInput(std::vector<uint32_t> &input_type_list, std::vector<int32_t> &input_graph_idx,
                          std::vector<std::string> &input_name_list, size_t exp_num) const;
  bool GetOpInputInfo(std::vector<uint32_t> &input_type_list, std::vector<int32_t> &input_graph_idx,
      std::vector<std::string> &input_name_list, std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range);
  bool GetOpOutputInfo(std::vector<uint32_t> &output_type_list, std::vector<std::string> &output_name_list,
                       std::map<size_t, std::pair<size_t, size_t>> &ir_out_2_range);
  void ConstructOptOutputArgs(size_t ir_index);
  Status ConstructOutArgsDescByOps(const std::vector<std::pair<std::string, ge::IrOutputType>> &ir_outputs);
  Status ConstructInArgsDescByOps(const std::vector<std::pair<std::string, ge::IrInputType>> &ir_inputs);
  const ge::OpDescPtr op_desc_{nullptr};
  ge::ArgsFormatDesc format_desc_;
  std::vector<int64_t> dyn_io_v_;
  std::vector<std::vector<int64_t>> dyn_io_vv_;
  size_t dyn_add_num_{0};
  bool pre_is_dyn_{false};
  bool is_dy_folded_{false};
  bool is_input_gen_place_{false};
  bool is_output_gen_place_{false};
  bool is_ffts_plus_{false};
  bool by_ir_{false};
  bool need_sync_{false};
};
}  // namespace fe
#endif  // GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_ARGS_FROMAT_CONSTRUCTOR_H_
