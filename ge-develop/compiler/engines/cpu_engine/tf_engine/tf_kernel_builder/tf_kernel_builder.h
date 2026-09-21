/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TF_KERNEL_BUILDER_H_
#define AICPU_TF_KERNEL_BUILDER_H_

#include <mutex>
#include "common/aicpu_ops_kernel_builder/kernel_builder.h"
#include "proto/fwk_adapter.pb.h"

namespace aicpu {
using KernelBuilderPtr = std::shared_ptr<KernelBuilder>;
class TfKernelBuilder : public KernelBuilder {
 public:
  /**
   * Destructor
   */
  virtual ~TfKernelBuilder() = default;

  /**
   * @return kernel info object
   */
  static KernelBuilderPtr Instance();

  /**
   * Calc the running size of Operator,then GE will alloc the memsize from runtime
   * The size is consist of the part as follow:
   *   1.Input and output size;
   *   2.NodeDef in tf; 3.FuncDef in tf.
   * @param node Node information, return task_memsize in node's attr
   * @return status whether this operation successful
   */
  ge::Status CalcOpRunningParam(const ge::Node &node) const override;

  /**
   * Copy the data from host to device, then address is alloced by GE, then invoked
   * the runtime's interface to generate the task
   * @param node Node information
   * @param run_context
   * @return status whether operation successful
   */
  ge::Status GenerateTask(const ge::Node &node,
                          const ge::RunContext &run_context,
                          std::vector<domi::TaskDef> &tasks) override;

  /**
   * Generate the task
   * @param node Node information
   * @param task[out]
   * @param task_info[out]
   * @return status whether this operation success
   */
  ge::Status GenSingleOpRunTask(const ge::NodePtr &node, STR_FWK_OP_KERNEL &task, string &task_info) override;

  /**
   * Generate the task
   * @param count the memcopy times
   * @param task[out]
   * @param task_info[out]
   * @return status whether this operation success
   */
  ge::Status GenMemCopyTask(uint64_t count, STR_FWK_OP_KERNEL &task, string &task_info) override;

  /**
   * init optimizer
   * @return status whether this operation success
   */
  ge::Status Initialize() override;

  // Copy prohibited
  TfKernelBuilder(const TfKernelBuilder &tf_kernel_builder) = delete;

  // Move prohibited
  TfKernelBuilder(const TfKernelBuilder &&tf_kernel_builder) = delete;

  // Copy prohibited
  TfKernelBuilder &operator=(const TfKernelBuilder &tf_kernel_builder) = delete;

  // Move prohibited
  TfKernelBuilder &operator=(TfKernelBuilder &&tf_kernel_builder) = delete;
 protected:
  /**
   * get type of input and output
   * @param op_desc_ptr, ptr store op information used for ge
   * @param inputs_type, type of input
   * @param outputs_type, type of output
   */
  void GetInOutPutsDataType(const ge::OpDescPtr &op_desc_ptr,
                            std::vector<uint32_t> &inputs_type,
                            std::vector<uint32_t> &outputs_type) const override;
 private:
  /**
   * Constructor
   */
  TfKernelBuilder() = default;
  
  /*
   * Build the ffts plus task def
   * @param Node operator
   * @param run_context run context infomation
   * @return status whether operation successful
   */
  ge::Status GenerateFftsPlusTask(const ge::Node &node, const ge::RunContext &run_context);
  
  /*
   * Build the ffts plus task def
   * @param Node operator
   * @param run_context run context infomation
   * @param tasks vector of task_def
   * @return status whether operation successful
   */
  ge::Status GenerateTaskDef(const ge::Node &node, const ge::RunContext &run_context,
                             std::vector<domi::TaskDef> &tasks);

  /*
   * Build the ffts plus kernel
   * @param Node operator
   * @param str_fwkop_kernel struct of tf
   * @param run_context run context infomation
   * @param task_info task info args
   * @index index of thread dim
   * @param ffts_info struct which store ffts infomation
   * @return status whether operation successful
   */
  ge::Status BuildAicpuFftsPlusKernel(const ge::Node &node,
                                      STR_FWK_OP_KERNEL &str_fwkop_kernel,
                                      const ge::RunContext &run_context,
                                      std::string &task_info,
                                      const uint32_t &index,
                                      const FftsPlusInfo &ffts_info) const;

  /*
   * Build the ffts kernelRunParam
   * @param opDesc Op description
   * @param kernel_run_param fake kernel_run_param just the input and
   *  output data_addr is not real)
   * @param index index of thread dim
   * @param ffts_info struct which store ffts infomation
   * @return status whether operation successful
   */
  ge::Status BuildFFtsKernelRunParam(
      const ge::OpDesc &op_desc, FWKAdapter::KernelRunParam &kernel_run_param,
      const uint32_t &index, const FftsPlusInfo &ffts_info) const;

  /**
   * Set the aicpu::FWKAdapter::TensorDataInfo, the data is from Ge Tensor
   * @param ge_tensor_desc Original Ge Tensor
   * @param tensor_data_info The input or output data, defined by protobuf
   * @param tensor_shape tensor shape
   * @param is_ref if output is ref
   * @param skip_dim_check if skip dim
   * @return status whether operation successful
   */
  aicpu::State SetTensorDataInfo(
      const ge::GeTensorDesc &ge_tensor_desc,
      ::aicpu::FWKAdapter::TensorDataInfo *tensor_data_info,
      const std::vector<int64_t> &tensor_shape, bool is_ref = false,
      bool skip_dim_check = false) const;

  ge::Status ConstructTfKernel(const ge::Node &node, const ge::RunContext &run_context,
                               FWKAdapter::FWKOperateParam *str_tf_kernel) const;

  ge::Status ConstructExtendInfoAndTaskdef(const ge::Node &node,
                                           const ge::RunContext &run_context,
                                           FWKAdapter::FWKOperateParam *str_tf_kernel) const;
  ge::Status UpdateTask(const ge::Node &node, std::vector<domi::TaskDef> &tasks) override;

  /**
   * Build the struct StrFWKKernel and set the real value, then launch the task
   * @param node original GE node info
   * @param run_context GE's run_context, the sessionId is needed
   * @return status whether operation successful
   */
  ge::Status BuildAndLaunchKernel(const ge::Node &node, const ge::RunContext &run_context) const;

  /**
   * update op information in framework
   * @param op_desc_ptr Op description
   * @return status whether operation successful
   */
  ge::Status UpdateFmkOpInfo(std::shared_ptr<ge::OpDesc> &op_desc_ptr) const;

  /**
   * update op imply type information
   * @param op_desc_ptr Op description
   * @return status whether operation successful
   */
  ge::Status UpdateImplyTypeInfo(std::shared_ptr<ge::OpDesc> &op_desc_ptr) const;
  /**
   * Make task extend info for node
   * @param node original GE node info
   * @param task_ext_info runtime's task info name vector
   * @return status whether operation successful
   */
  ge::Status MakeTaskExtInfo(const ge::Node &node,
                             std::vector<char> &task_ext_info,
                             const FftsPlusInfo &ffts_info) const;

  /**
   * check if the address need update
   * @param node original GE node info
   * @param update_addr_flag, flag used to check if the address need update
   * @return whether the address need update
   */
  bool NeedUpdateAddr(const ge::Node &node, int32_t &update_addr_flag) const;

  /**
   * Check the known node whether in dynamic shape graph
   * @param node original GE node info
   * @return whether node is dynamic shape op
   */
  bool IsKnownNodeDynamic(const ge::Node &node) const;

  /**
    * Generate the task
    * @param node Node information
    * @param skip_dim_check skip dim check
    * @param str_tf_kernel tf kernel name
    * @param task_info task info
    * @return status whether this operation success
    */
  ge::Status GenTaskImply(const ge::NodePtr &node,
                          ::aicpu::FWKAdapter::FWKOperateParam *str_tf_kernel,
                          string &task_info,
                          bool skip_dim_check = false) const;

 private:
  // singleton instance
  static KernelBuilderPtr instance_;
  std::mutex mutex_;

  /**
   * get ffts info from sgt and node info
   * @param op_desc_ptr Ge node description
   * @param ffts_info ffts info
   */
  ge::Status BuildFftsInfo(const ge::OpDescPtr &op_desc_ptr,
                           FftsPlusInfo &ffts_info) const;
};
} // namespace aicpu
#endif // AICPU_TF_KERNEL_BUILDER_H_
