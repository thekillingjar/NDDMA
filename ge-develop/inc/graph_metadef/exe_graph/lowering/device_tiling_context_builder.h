/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMMON_RUNTIME_DEVICE_TILING_KERNEL_CONTEXT_BUILDER_H_
#define GE_COMMMON_RUNTIME_DEVICE_TILING_KERNEL_CONTEXT_BUILDER_H_

#include "graph/node.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/lowering/buffer_pool.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"

namespace gert {
struct AddrRefreshedTensor {
  gert::Tensor *host_addr;
  uint64_t device_addr;
};

struct TiledKernelContextHolder {
  uint64_t dev_op_type_addr_{0UL};
  uint64_t dev_op_name_addr_{0UL};
  KernelContext *host_context_{nullptr};
  uint64_t dev_context_addr_{0UL};
  std::vector<uint64_t> output_addrs_;
  uint8_t *host_compute_node_info_{nullptr};
  size_t compute_node_info_size_{0UL};
};

class DeviceTilingContextBuilder {
 public:
  static size_t CalcTotalTiledSize(const ge::OpDescPtr &op_desc);
  DeviceTilingContextBuilder &CompileInfo(void *compile_info);
  DeviceTilingContextBuilder &Deterministic(int32_t deterministic);
  DeviceTilingContextBuilder &DeterministicLevel(int32_t deterministic_level);
  DeviceTilingContextBuilder &PlatformInfo(void *platform_info);
  DeviceTilingContextBuilder &TilingData(void *tiling_data);
  DeviceTilingContextBuilder &AddrRefreshedInputTensor(const std::map<size_t, AddrRefreshedTensor> &index_to_tensor);
  DeviceTilingContextBuilder &TiledHolder(uint8_t *host_addr, uint64_t dev_addr, size_t max_mem_size);
  DeviceTilingContextBuilder &Workspace(void *workspace);
  ge::graphStatus Build(const ge::NodePtr &node, TiledKernelContextHolder &holder);

 private:
  ge::graphStatus BuildRtTensor(const ge::GeTensorDesc &tensor_desc, ConstTensorAddressPtr address);
  ge::graphStatus BuildPlacementRtTensor(const ge::GeTensorDesc &tensor_desc, Tensor *rt_tensor) const;
  ge::graphStatus BuildIOTensors(const ge::OpDesc *const op_desc);

  ge::graphStatus TiledBuild(const ge::NodePtr &node, TiledKernelContextHolder &holder);

  void *compile_info_{nullptr};
  void *platform_info_{nullptr};
  int32_t deterministic_{0};
  int32_t deterministic_level_{0};
  uint64_t dev_begin_{0UL};
  uint8_t *host_begin_{nullptr};
  size_t max_mem_size_{0UL};
  std::map<size_t, AddrRefreshedTensor> index_to_tensor_;
  std::vector<void *> inputs_;
  std::vector<void *> outputs_{TilingContext::kOutputNum};
};
}  // namespace gert
#endif  // GE_COMMMON_RUNTIME_DEVICE_TILING_KERNEL_CONTEXT_BUILDER_H_
