/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_CPU_KERNEL_BUILDER_H
#define AICPU_CPU_KERNEL_BUILDER_H

#include "common/aicpu_ops_kernel_builder/kernel_builder.h"

namespace aicpu {
class CpuKernelBuilder : public KernelBuilder {
public:
  /**
   * constructor
   * @param void
   */
  CpuKernelBuilder() = default;
  /**
   * Destructor
   */
  virtual ~CpuKernelBuilder() = default;

  /**
   * Calc the running size of Operator,then GE will alloc the memsize from
   * runtime The size is consist of the part as follow: 1.kernel so
   * name; 2.kernel func name; 3.Kernel param.
   * @param node Node information, return task_memsize in node's attr
   * @return status whether this operation successful
   */
  ge::Status CalcOpRunningParam(const ge::Node &node) const override;

  /**
   * Copy the data from host to device, then address is alloced by GE, then
   * invoked the runtime's interface to generate the task
   * @param node Node information
   * @param run_context
   * @return status whether operation successful
   */
  ge::Status GenerateTask(const ge::Node &node,
                          const ge::RunContext &run_context,
                          std::vector<domi::TaskDef> &tasks) override;

  ge::Status UpdateTask(const ge::Node &node, std::vector<domi::TaskDef> &tasks) override;
private:
  /**
   * Build the struct AicpuKernelFixedParam, then launch the task
   * @param node Op description
   * @param kernel_def Kernel task info
   * @return status whether operation successful
   */
  ge::Status BuildAndLaunchKernel(const ge::Node &node,
                                  domi::KernelDef *&kernel_def) const;

  /**
   * Make aicpu kernel task extend info
   * @param op_desc_ptr Ge op description pointer
   * @param task_ext_info task extend info
   * @return whether handle success
   */
  ge::Status MakeAicpuKernelExtInfo(const ge::OpDescPtr &op_desc_ptr,
                                    std::vector<char> &task_ext_info,
                                    const FftsPlusInfo &ffts_info) const;

  /**
   * Generate ffts plus task node info
   * @param node Ge node description
   * @return status whether handle success
   */
  ge::Status GenerateFftsPlusTask(const ge::Node &node) const;

  /**
   * get ffts info from sgt and node info
   * @param op_desc_ptr Ge node description
   * @param ffts_info ffts info
   */
  ge::Status BuildFftsInfo(const ge::OpDescPtr &op_desc_ptr,
                           FftsPlusInfo &ffts_info) const;

  /**
   * build aicpu ctx for fftsdef
   * @param op_desc_ptr Ge node description
   * @param ffts_info ffts info
   * @param ctx aicpu ctx for fftsdef
   * @return status whether handle success
   */
  ge::Status BuildAiCpuCtx(const ge::OpDescPtr &op_desc_ptr,
                           const FftsPlusInfo &ffts_info,
                           domi::FftsPlusAicpuCtxDef *ctx) const;

  /**
   * build aicpu kerneldef for ffts
   * @param node Ge node description
   * @param ffts_info ffts info
   * @param aicpu_ctx aicpu ctx for fftsdef
   * @return status whether handle success
   */
  ge::Status BuildAiCpuFftsKernelDef(
      const ge::Node &node, FftsPlusInfo &ffts_info,
      domi::FftsPlusAicpuCtxDef *aicpu_ctx) const;
  uint32_t GetOpBlockDim(const ge::OpDescPtr &op_desc_ptr) const;
  int64_t CeilDivisor(const int64_t x, const int64_t base) const;
  uint32_t GetOpBlockDimForFftsPlus(const ge::OpDescPtr &op_desc_ptr,
                                    const FftsPlusInfo &ffts_info,
                                    const uint32_t thread_index) const;
  bool IsSupportBlockDim(const ge::OpDescPtr &op_desc_ptr) const;
  uint32_t CalcBlockDimByShapeSize(const int64_t total) const;
  ge::Status GenerateMemCopyTask(uint64_t data_info_size,
                                 const ge::RunContext &run_context,
                                 std::vector<domi::TaskDef> &tasks) const;
  ge::Status BuildMemCopyInfo(const ge::OpDescPtr &op_desc_ptr,
                              const ge::RunContext &run_context,
                              domi::KernelDef *&kernel_def) const;
};
}  // namespace aicpu
#endif  // AICPU_CPU_KERNEL_BUILDER_H
