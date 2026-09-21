/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_FFTS_ADAPTER_TASK_BUILDER_ADAPTER_FACTORY_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_FFTS_ADAPTER_TASK_BUILDER_ADAPTER_FACTORY_H_

#include <string>
#include <mutex>
#include <map>
#include <string>
#include <vector>
#include "common/sgt_slice_type.h"
#include "adapter/tbe_adapter/tbe_task_builder_adapter.h"
#include "graph/op_kernel_bin.h"

namespace fe {
struct ThreadParamOffset {
  size_t thread_dim;
  vector<vector<void *>> thread_input_addrs;
  vector<vector<void *>> thread_output_addrs;
  vector<void *> first_thread_input_addrs;
  vector<void *> first_thread_output_addrs;
  vector<vector<void *>> thread_workspace_addrs;
  vector<vector<uint32_t>> thread_workspace_sizes;
  vector<int64_t> thread_addr_offset;
  vector<int64_t> input_tensor_sizes;
  vector<int64_t> output_tensor_sizes;
};

class FftsTaskBuilderAdapter : public TaskBuilderAdapter {
 public:
  FftsTaskBuilderAdapter(const ge::Node &node, TaskBuilderContext &context);
  ~FftsTaskBuilderAdapter() override;

  /*
   * @ingroup fe
   * @brief   Init TaskBuilderAdapter
   * @return  SUCCESS or FAILED
   */
  Status Init() override;
  void GetThreadParamOffset(ThreadParamOffset &param_offset) const;
  Status CheckInputAndOutputSize();

 protected:
  Status InitInput() override;
  Status InitOutput() override;
  Status InitWorkspace() override;
  Status ThreadInit();
  Status ThreadInitInput(vector<vector<vector<ffts::DimRange>>> &tensor_slice);
  Status ThreadInitOutput(vector<vector<vector<ffts::DimRange>>> &tensor_slice);
  Status RegTbeInfo();
  Status RegisterBinary(void * &bin_handle, std::string& kernel_name);
  void DebugThreadArgs() const;

 private:
  std::string stub_func_;
  ge::OpKernelBinPtr kernel_bin_;
  std::string kernel_name_;
  uint32_t block_dim_;
  ge::Buffer tbe_kernel_buffer_;
  uint32_t tbe_kernel_size_;

  size_t thread_dim_;
  vector<vector<void *>> thread_input_addrs_;
  vector<vector<void *>> thread_output_addrs_;
  vector<void *> first_thread_input_addrs_;
  vector<void *> first_thread_output_addrs_;
  // Workspace
  vector<vector<void *>> thread_workspace_addrs_;
  vector<vector<uint32_t>> thread_workspace_sizes_;

  vector<int64_t> thread_addr_offset_;
  vector<int64_t> input_tensor_sizes_;
  vector<int64_t> output_tensor_sizes_;

  Status HandleAnchorData(size_t &input_index, const size_t &anchor_index);
  void SetInputAddrFromDataBase(int64_t &input_offset);
  Status VerifyWeights() const;
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_FFTS_ADAPTER_TASK_BUILDER_ADAPTER_FACTORY_H_
