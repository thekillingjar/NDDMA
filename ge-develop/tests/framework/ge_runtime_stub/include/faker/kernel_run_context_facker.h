/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_UT_GE_RUNTIME_V2_FAKER_KERNEL_RUN_CONTEXT_FACKER_H_
#define AIR_CXX_TESTS_UT_GE_RUNTIME_V2_FAKER_KERNEL_RUN_CONTEXT_FACKER_H_
#include <memory>
#include <vector>
#include <cstring>
#include "exe_graph/runtime/kernel_run_context.h"
#include "exe_graph/runtime/context_extend.h"
#include "exe_graph/lowering/buffer_pool.h"
#include "exe_graph/runtime/storage_shape.h"
#include "graph/any_value.h"
#include "graph/node.h"
#include "exe_graph/runtime/tiling_context.h"
#include "kernel/common_kernel_impl/tiling.h"
#include "exe_graph/runtime/gert_mem_block.h"

namespace gert {
struct KernelRunContextHolder {
  ~KernelRunContextHolder() {
    FreeAll();
  }
  KernelRunContextHolder() = default;
  KernelRunContextHolder(KernelRunContextHolder &&) = default;
  KernelRunContextHolder &operator=(KernelRunContextHolder &&) = default;
  template<typename T>
  T *GetContext() {
    return reinterpret_cast<T*>(context);
  }
  ComputeNodeInfo *MutableComputeNodeInfo() {
    return reinterpret_cast<ComputeNodeInfo *>(compute_node_info_holder.get());
  }

  operator KernelContext*(){
    return GetContext<KernelContext>();
  }

  void FreeValue(size_t index){
    if(index < value_holder.size()){
      value_holder[index].Set(nullptr, nullptr);
    }
  }
  void FreeAll() {
    for (auto &chain : value_holder) {
      chain.Set(nullptr, nullptr);
    }
  }
  std::unique_ptr<uint8_t[]> context_holder;
  std::vector<Chain> value_holder;
  std::unique_ptr<uint8_t[]> compute_node_info_holder;
  bg::BufferPool buffer_pool;
  KernelRunContext *context;
  std::string kernel_name_holder;
  std::string kernel_type_holder;
  std::unique_ptr<uint8_t[]> kernel_extend_info_holder;
};
KernelRunContextHolder BuildKernelRunContext(size_t input_num, size_t output_num);

class KernelRunContextFaker {
 public:
  KernelRunContextFaker() = default;
  KernelRunContextFaker &KernelIONum(size_t input_num, size_t output_num);
  KernelRunContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  KernelRunContextFaker &IrInputNum(size_t input_num);
  KernelRunContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num);
  KernelRunContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                     ge::Format storage_format);
  KernelRunContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                      ge::Format storage_format);
  KernelRunContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value);
  KernelRunContextFaker &Inputs(std::vector<void *> inputs);
  KernelRunContextFaker &Outputs(std::vector<void *> outputs);

  KernelRunContextFaker &NodeName(std::string name);
  KernelRunContextFaker &NodeType(std::string node_type);
  KernelRunContextFaker &KernelName(std::string name);
  KernelRunContextFaker &KernelType(std::string kernel_type);

  KernelRunContextHolder Build() const;

 private:
  ge::NodePtr FakeNode(ge::ComputeGraphPtr &graph) const;

 private:
  size_t kernel_input_num_;
  size_t kernel_output_num_;
  size_t node_input_num_;
  size_t node_output_num_;
  std::vector<uint32_t> ir_instance_num_;
  std::vector<CompileTimeTensorDesc> node_input_tds_;
  std::vector<CompileTimeTensorDesc> node_output_tds_;
  std::vector<void *> inputs_;
  std::vector<void *> outputs_;
  std::vector<std::pair<std::string, ge::AnyValue>> attrs_;
  std::string kernel_name_;
  std::string kernel_type_;
  std::string node_name_{"node"};
  std::string node_type_{"node"};
};

class InferShapeContextFaker {
 public:
  InferShapeContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  InferShapeContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  InferShapeContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  InferShapeContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                      ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferShapeContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                       ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferShapeContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
    base_faker_.NodeAttrs(std::move(keys_to_value));
    return *this;
  }

  InferShapeContextFaker &InputShapes(std::vector<void *> input_shapes);
  InferShapeContextFaker &OutputShapes(std::vector<void *> output_shapes);

  KernelRunContextHolder Build() const;

 private:
  enum InputsAppend { kInputsInferShapeFunc, kInputsAppendEnd };

 private:
  KernelRunContextFaker base_faker_;
};

class TilingContextFaker {
 public:
  TilingContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  TilingContextFaker &FallibleContext();
  TilingContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  TilingContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  TilingContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format, ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  TilingContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                   ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  TilingContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
    base_faker_.NodeAttrs(std::move(keys_to_value));
    return *this;
  }
  TilingContextFaker &InputShapes(std::vector<gert::StorageShape *> input_shapes);
  TilingContextFaker &OutputShapes(std::vector<gert::StorageShape *> output_shapes);
  TilingContextFaker &CompileInfo(void *compile_info);
  TilingContextFaker &TilingData(void *tiling_data);
  TilingContextFaker &Workspace(ContinuousVector *workspace);
  TilingContextFaker &TilingFwkData(void* fwk_data);
  TilingContextFaker &Deterministic(void* deterministic);

  KernelRunContextHolder Build() const;

 private:
  void UpdateInputs();

 private:
  enum InputsAppend { kInputsCompileInfo, kInputsTilingFunc, kInputsAppendEnd };

  KernelRunContextFaker base_faker_;
  std::vector<gert::StorageShape *> input_shapes_;
  std::vector<gert::StorageShape *> output_shapes_;
  std::vector<void *> outputs_{static_cast<size_t>(kernel::TilingExOutputIndex::kNum)};
  void *fwk_data_ = nullptr;
  void *compile_info_ = nullptr;
  void *deterministic_ = nullptr;
  void *deterministic_level_ = nullptr;
};
}  // namespace gert
#endif  //AIR_CXX_TESTS_UT_GE_RUNTIME_V2_FAKER_KERNEL_RUN_CONTEXT_FACKER_H_
