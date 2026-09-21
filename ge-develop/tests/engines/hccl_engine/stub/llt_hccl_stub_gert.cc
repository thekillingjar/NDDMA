/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "llt_hccl_stub_gert.h"
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/lowering/frame_selector.h"
#include "runtime/allocator.h"
#include "ge/ge_error_codes.h"
#include "exe_graph/runtime/shape.h"
#include "llt_hccl_stub_kernel_run_ctx_faker.h"

namespace gert {
namespace bg {

 // namespace
std::atomic<int64_t> ValueHolder::id_generator_{0};
ValueHolder::~ValueHolder() = default;

ValueHolder::ValueHolder()
    : id_(id_generator_++), type_(ValueHolderType::kValueHolderTypeEnd), index_(0), placement_(0) {}

ValueHolderPtr ValueHolder::CreateConst(const void *data, size_t size, bool is_string)
{
    ValueHolderPtr holder = std::shared_ptr<ValueHolder>(new (std::nothrow) ValueHolder());
    return holder;
}

ValueHolderPtr ValueHolder::CreateSingleDataOutput(const char *node_type, const vector<ValueHolderPtr> &inputs)
{
    auto output_data = std::shared_ptr<ValueHolder>(new (std::nothrow) ValueHolder());
    return output_data;
}

std::vector<ValueHolderPtr> ValueHolder::CreateDataOutput(const char *node_type,
    const std::vector<ValueHolderPtr> &inputs, size_t out_count)
{
    std::vector<ValueHolderPtr> holders;
    return holders;
}


ValueHolderPtr ValueHolder::CreateVoidGuarder(const char *node_type, const ValueHolderPtr &resource,
    const vector<ValueHolderPtr> &args)
{
    auto guarder = std::shared_ptr<ValueHolder>(new (std::nothrow) ValueHolder());
    return guarder;
}

ValueHolderPtr ValueHolder::CreateMateFromNode(ge::FastNode *node, int32_t index, ValueHolderType type) {
    auto holder = std::shared_ptr<ValueHolder>(new (std::nothrow) ValueHolder());
    return holder;  
}

ValueHolderPtr DevMemValueHolder::CreateMateFromNode(ge::FastNode *node, int32_t index, ValueHolderType type) {
    auto holder = std::shared_ptr<ValueHolder>(
      new (std::nothrow) DevMemValueHolder(0)
    );
    return holder;
}

DevMemValueHolderPtr DevMemValueHolder::CreateSingleDataOutput(const char *node_type,
                                                               const std::vector<ValueHolderPtr> &inputs,
                                                               int64_t logic_stream_id) {
    auto holder = std::shared_ptr<DevMemValueHolder>(new (std::nothrow) DevMemValueHolder(0));
    return holder;
}

std::vector<DevMemValueHolderPtr> DevMemValueHolder::CreateDataOutput(const char *node_type,
                                                                      const std::vector<ValueHolderPtr> &inputs,
                                                                      size_t out_count, int64_t logic_stream_id) {
  std::vector<DevMemValueHolderPtr> holders;
  return holders;
}

GenerateExeGraph::ExeGraphGenerator GenerateExeGraph::generator_;
static std::vector<ValueHolderPtr> InferShape(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &shapes) {
    std::vector<ValueHolderPtr> holders;
    return holders;
}

static std::vector<ValueHolderPtr> AllocOutputMemory(TensorPlacement placement, const ge::NodePtr &node,
    const std::vector<ValueHolderPtr> &output_sizes, LoweringGlobalData &global_data) {
    std::vector<ValueHolderPtr> holders;
    return holders;
}

static std::vector<ValueHolderPtr> CalcTensorSize(const ge::NodePtr &node,
    const std::vector<ValueHolderPtr> &output_shapes) {
    std::vector<ValueHolderPtr> holders;
    size_t outputSize = output_shapes.size();
    ValueHolderPtr outputHolder;
    for (size_t i = 0; i < outputSize; i++) {
        holders.push_back(outputHolder);
    }
    return holders;
}

std::vector<ValueHolderPtr> FrameSelector::OnInitRoot(const std::function<std::vector<ValueHolderPtr>()> &builder) {
  std::vector<ValueHolderPtr> init_node_outputs;
  return init_node_outputs;
}

}  // namespace bg

bg::ValueHolderPtr LoweringGlobalData::GetStreamById(int64_t logic_stream_id) const {
	(void)logic_stream_id;
	return bg::ValueHolder::CreateSingleDataOutput("Stream", {});
}

bg::ValueHolderPtr LoweringGlobalData::GetOrCreateL2Allocator(int64_t logic_stream_id, const AllocatorDesc desc) {
	(void)logic_stream_id;
	bg::ValueHolderPtr init_selected_allocator = nullptr;
    auto placement_holder = bg::ValueHolder::CreateConst(&desc.placement, sizeof(desc.placement));
    auto memory_type_holder = bg::ValueHolder::CreateConst(&desc.usage, sizeof(desc.usage));
    auto created_allocator =
        bg::ValueHolder::CreateSingleDataOutput("CreateAllocator", {placement_holder, memory_type_holder});
    init_selected_allocator = created_allocator;
    return init_selected_allocator;
}

ge::graphStatus CalcAlignedSizeByShape(const Shape &shape, ge::DataType data_type, uint64_t &ret_tensor_size) {
  return ge::GRAPH_SUCCESS;
}

KernelRunContextFaker &KernelRunContextFaker::KernelIONum(size_t input_num, size_t output_num) {
  kernel_input_num_ = input_num;
  kernel_output_num_ = output_num;
  return *this;
}

KernelRunContextFaker &KernelRunContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  node_input_num_ = input_num;
  node_output_num_ = output_num;
  node_input_tds_.resize(input_num);
  node_output_tds_.resize(output_num);
  return *this;
}

GertMemBlock *AllocatorFaker::Malloc(size_t size) {
    GertMemBlock *block = reinterpret_cast<GertMemBlock *>(new GertMemBlockFaker(malloc(size)));
    return block;
}
void AllocatorFaker::Free(GertMemBlock *block) {
    free(block->GetAddr());
    delete block;
}

ge::OpDescPtr KernelRunContextFaker::FakeOp() const {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  size_t input_index = 0;
  for (size_t ir_index = 0; ir_index < ir_instance_num_.size(); ++ir_index) {
    auto ir_ins_num = ir_instance_num_[ir_index];
    auto prefix = "x_" + std::to_string(ir_index) + "_";
    for (size_t i = 0; i < ir_ins_num; ++i, ++input_index) {
      auto td = ge::GeTensorDesc();
      if (node_input_tds_.size() > input_index) {
        td.SetFormat(node_input_tds_[input_index].GetStorageFormat());
        td.SetDataType(node_input_tds_[input_index].GetDataType());
      }
      op_desc->AddInputDesc(prefix + std::to_string(i), td);
    }
  }
  // fill it when not set
  std::vector<uint32_t> ir_output_instance_num;
  if (ir_output_instance_num_.empty()) {
    for (size_t i = 0; i < node_output_num_; ++i) {
      ir_output_instance_num.emplace_back(1U);
    }
  } else {
    ir_output_instance_num = ir_output_instance_num_;
  }
  size_t output_index = 0;
  for (size_t ir_index = 0; ir_index < ir_output_instance_num.size(); ++ir_index) {
    auto ir_ins_num = ir_output_instance_num[ir_index];
    auto prefix = "y_" + std::to_string(ir_index) + "_";
    for (size_t i = 0; i < ir_ins_num; ++i, ++output_index) {
      auto td = ge::GeTensorDesc();
      if (node_output_tds_.size() > output_index) {
        td.SetFormat(node_output_tds_[output_index].GetStorageFormat());
        td.SetDataType(node_output_tds_[output_index].GetDataType());
      }
      op_desc->AddOutputDesc(prefix + std::to_string(i), td);
    }
  }

  for (const auto &attr : attrs_) {
  }
  return op_desc;
}

FakeKernelContextHolder KernelRunContextFaker::Build() const {
  FakeKernelContextHolder fake_holder;
  fake_holder.kernel_input_num = kernel_input_num_;
  fake_holder.kernel_output_num = kernel_output_num_;
  KernelRunContextBuilder kernel_context_builder;
  auto op_desc = FakeOp();
  if (inputs_.size() != kernel_input_num_ || outputs_.size() != kernel_output_num_) {
    std::vector<void *> inputs(kernel_input_num_, nullptr);
    std::vector<void *> outputs(kernel_output_num_, nullptr);
    fake_holder.holder = kernel_context_builder.Inputs(inputs).Outputs(outputs).Build(op_desc);
    return fake_holder;
  }
  fake_holder.holder = kernel_context_builder.Inputs(inputs_).Outputs(outputs_).Build(op_desc);
  return fake_holder;
}

}  // namespace gert
