/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DAVINCI_CPU_OPTIMIZER_H_
#define DAVINCI_CPU_OPTIMIZER_H_

#include <unordered_map>
#include <vector>
#include "common/aicpu_graph_optimizer/optimizer.h"
#include "graph/compute_graph.h"
#include "graph/detail/attributes_holder.h"
#include "graph/op_desc.h"
#include "proto/aicpu/cpu_attr.pb.h"
#include "proto/aicpu/cpu_node_def.pb.h"
#include "proto/aicpu/cpu_tensor.pb.h"
#include "proto/aicpu/cpu_tensor_shape.pb.h"
#include "util/constant.h"

namespace aicpu {

class CpuOptimizer : public Optimizer {
 public:
  /**
   * constructor
   * @param void
   */
  CpuOptimizer() = default;

  /**
   * Destructor
   */
  virtual ~CpuOptimizer() = default;

  /**
   * Optimize original graph, using in graph preparation stage
   * @param graph Computation graph
   * @param all_op_info, map stored all aicpu ops information
   * @return status whether this operation success
   */
  ge::Status OptimizeOriginalGraph(ge::ComputeGraph &graph,
                                   const std::map<std::string, OpFullInfo> &all_op_info) const override;

  /**
   * optimizer fused Graph, find aicpu ops which can be fused and fuse them
   * @param graph, Compute graph
   * @param all_op_info, map stored all aicpu ops information
   * @return result is success or not
   */
  ge::Status OptimizeFusedGraph(
      ge::ComputeGraph &graph,
      const std::map<std::string, OpFullInfo> &all_op_info) const override;

  /**
   * init optimizer
   * @return status whether this operation success
   */
  ge::Status Initialize() override;

  std::map<std::string, std::string> cust_user_infos_;

 private:
  /**
   * Read bin file
   * @param op_desc_ptr Op desc pointer
   * @param bin_folder_path bin file path
   * @param op_full_info op full info
   * @param graph_id graph id
   * @param exist_graph_id exist graph id
   * @return bool flag
   */
  bool PackageBinFile(ge::OpDescPtr op_desc_ptr,
                      const std::string &bin_folder_path,
                      const OpFullInfo &op_full_info, uint32_t graph_id,
                      bool exist_graph_id) const;

  /**
   * Get custom aicpu kernel so path
   * @param null
   * @return string path
   */
  const std::string GetCustKernelSoPath(const std::string op_type) const;

  /**
   * Get cpu kernel so path (libcpu_kernels.so)
   * @param null
   * @return string path
   */
  const std::string GetCpuKernelSoPath() const;

  /**
   * Init load cpu kernels type
   */
  void InitLoadCpuKernelsType();

  /**
   * Check so whether need load in model
   * @param op_full_info op full info
   * @param file_path file path
   * @return true or false
   */
  bool CheckSoNeedLoadInModel(const OpFullInfo &op_full_info,
                              std::string &file_path, const std::string op_type) const;

  /**
   * Get bin file name
   * @param op_full_info op full info
   * @param bin_folder_path bin folder path
   * @param bin_file_name bin file name
   * @param graph_id graph id
   * @return result is success or not
   */
  ge::Status GetBinFileName(const OpFullInfo &op_full_info,
                            const std::string &bin_folder_path,
                            std::string &bin_file_name) const;

  ge::Status SetCustKernelBinFile(
      ge::OpDescPtr op_desc_ptr,
      const std::map<std::string, OpFullInfo> &all_op_info, uint32_t graph_id,
      bool exist_graph_id) const;

  void CheckAndSetSocVersion(const std::string &soc_version_from_ge) const;
  void SetCustUserInfos(map<std::string, std::string> info) override;
  void GetCustUserInfos(map<std::string, std::string> &info) const;

 private:
  // load type for libcpu_kernels.so
  uint64_t load_type_for_cpu_kernels_ = kDefaultLoadTypeForCpuKernels;

  bool isVendorCustPathEmpty(const std::string &path) const;
};
}  // namespace aicpu

#endif  // DAVINCI_AICPU_OPTIMIZER_H
