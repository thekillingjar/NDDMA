/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "placed_lowering_result.h"

#include "common/checker.h"
#include "common/table_driven.h"
#include "core/builder/node_types.h"
#include "graph_builder/bg_memory.h"
#include "graph_builder/bg_tensor.h"
#include "exe_graph/lowering/value_holder_utils.h"

#include "lowering/placement/placed_lowering_register.h"
namespace gert {
namespace {
bg::ValueHolderPtr GetTensorSize(LoweringGlobalData &global_data, ge::DataType data_type, const OutputLowerResult &src,
                                 bg::ValueHolderPtr &dt_holder) {
  bg::ValueHolderPtr tensor_size;
  if (data_type != ge::DT_STRING) {
    tensor_size = bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromStorage", {dt_holder, src.shape});
  } else {
    std::vector<bg::ValueHolderPtr> calc_string_input{dt_holder};
    calc_string_input.emplace_back(src.shape);
    calc_string_input.emplace_back(src.address);
    calc_string_input.emplace_back(global_data.GetStream());

    tensor_size = bg::ValueHolder::CreateSingleDataOutput("CalcStringTensorSize", calc_string_input);
  }
  return tensor_size;
}

ge::graphStatus HostToDevice(bg::ValueHolderPtr &dst_allocator, LoweringGlobalData &global_data, ge::DataType data_type,
                             const OutputLowerResult &src, int64_t dst_logic_stream_id, OutputLowerResult &dst) {
  // 因为当前host计算是同步的、不是基于流的，因此H2D拷贝不需要流同步，调度到时直接拷贝就可以了
  auto dt_holder = bg::ValueHolder::CreateConst(&data_type, sizeof(data_type));
  bg::ValueHolderPtr tensor_size;
  if (data_type != ge::DT_STRING) {
    tensor_size = bg::ValueHolder::CreateSingleDataOutput("CalcUnalignedTensorSizeFromStorage", {dt_holder, src.shape});
  } else {
    std::vector<bg::ValueHolderPtr> calc_string_input{dt_holder};
    calc_string_input.emplace_back(src.shape);
    calc_string_input.emplace_back(src.address);
    calc_string_input.emplace_back(global_data.GetStream());

    tensor_size = bg::ValueHolder::CreateSingleDataOutput("CalcStringTensorSize", calc_string_input);
  }
  auto dst_stream = global_data.GetStreamById(dst_logic_stream_id);
  auto dst_address = bg::DevMemValueHolder::CreateSingleDataOutput(
      "CopyH2D", {dst_stream, dst_allocator, src.address, tensor_size, src.shape, dt_holder}, dst_logic_stream_id);
  GE_ASSERT_NOTNULL(bg::ValueHolder::CreateVoidGuarder("FreeMemory", dst_address, {}));

  for (const auto &src_ordered_holder : src.order_holders) {
    bg::ValueHolder::AddDependency(src_ordered_holder, dst_address);
  }
  dst.shape = src.shape;
  dst.address = dst_address;
  dst.address->SetPlacement(dst_allocator->GetPlacement());
  dst.has_init = true;
  dst.order_holders.emplace_back(dst_address);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus HostToDeviceHbm(const ge::Node *node, LoweringGlobalData &global_data, ge::DataType data_type,
                                const OutputLowerResult &src, int64_t dst_logic_stream_id, OutputLowerResult &dst) {
  (void)node;
  auto dst_allocator =
      global_data.GetOrCreateL2Allocator(dst_logic_stream_id, {kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  GE_ASSERT_NOTNULL(dst_allocator);
  return HostToDevice(dst_allocator, global_data, data_type, src, dst_logic_stream_id, dst);
}

ge::graphStatus HostToDeviceP2p(const ge::Node *node, LoweringGlobalData &global_data, ge::DataType data_type,
                                const OutputLowerResult &src, int64_t dst_logic_stream_id, OutputLowerResult &dst) {
  (void)node;
  auto dst_allocator =
      global_data.GetOrCreateL2Allocator(dst_logic_stream_id, {kOnDeviceP2p, AllocatorUsage::kAllocNodeOutput});
  GE_ASSERT_NOTNULL(dst_allocator);
  return HostToDevice(dst_allocator, global_data, data_type, src, dst_logic_stream_id, dst);
}

ge::graphStatus DeviceHbmToHost(const ge::Node *node, LoweringGlobalData &global_data, ge::DataType data_type,
                                const OutputLowerResult &src, int64_t dst_logic_stream_id, OutputLowerResult &dst) {
  (void)dst_logic_stream_id;
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const int64_t src_logic_stream_id = op_desc->GetStreamId();
  auto src_stream = global_data.GetStreamById(src_logic_stream_id);
  const auto sync_stream_key = "SyncStream_" + node->GetName();
  auto builder = [&src_stream]()->bg::ValueHolderPtr {
    return bg::ValueHolder::CreateVoid<bg::ValueHolder>("SyncStream", {src_stream});
  };
  auto sync_stream_holder = global_data.GetOrCreateUniqueValueHolder(sync_stream_key, builder);
  for (const auto &src_ordered_holder : src.order_holders) {
    bg::ValueHolder::AddDependency(src_ordered_holder, sync_stream_holder);
  }
  auto dt_holder = bg::ValueHolder::CreateConst(&data_type, sizeof(data_type));
  auto tensor_size = GetTensorSize(global_data, data_type, src, dt_holder);
  auto allocator_holder =
      global_data.GetOrCreateL2Allocator(src_logic_stream_id, {kOnHost, AllocatorUsage::kAllocNodeWorkspace});
  GE_ASSERT_NOTNULL(allocator_holder);
  auto dst_address = bg::DevMemValueHolder::CreateSingleDataOutput(
      "CopyD2H", {src.address, tensor_size, allocator_holder}, src_logic_stream_id);
  GE_ASSERT_NOTNULL(bg::ValueHolder::CreateVoidGuarder("FreeMemory", dst_address, {}));
  bg::ValueHolder::AddDependency(sync_stream_holder, dst_address);

  dst.order_holders.clear();
  dst.shape = src.shape;
  dst.address = dst_address;
  dst.address->SetPlacement(kOnHost);
  dst.has_init = true;
  dst.order_holders.emplace_back(dst_address);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MakeSureAtHostDDR(const ge::Node *node, LoweringGlobalData &global_data, ge::DataType data_type,
                                  const OutputLowerResult &src, int64_t dst_logic_stream_id, OutputLowerResult &dst) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const int64_t src_logic_stream_id = op_desc->GetStreamId();
  (void)dst_logic_stream_id;
  auto dt_holder = bg::ValueHolder::CreateConst(&data_type, sizeof(data_type));
  auto tensor_size = GetTensorSize(global_data, data_type, src, dt_holder);
  // host不需要傳入allocator
  std::vector<bg::ValueHolderPtr> copy_inputs{global_data.GetStream()};
  copy_inputs.emplace_back(
      global_data.GetOrCreateL2Allocator(src_logic_stream_id, {kOnHost, AllocatorUsage::kAllocNodeWorkspace}));

  copy_inputs.emplace_back(src.address);
  copy_inputs.emplace_back(tensor_size);

  auto dst_address = bg::DevMemValueHolder::CreateSingleDataOutput("MakeSureTensorAtHost",
                                                                   copy_inputs, src_logic_stream_id);

  for (const auto &src_ordered_holder : src.order_holders) {
    bg::ValueHolder::AddDependency(src_ordered_holder, dst_address);
  }

  dst.shape = src.shape;
  dst.address = dst_address;
  GE_ASSERT_NOTNULL(dst.address);
  dst.address->SetPlacement(kOnHost);
  dst.has_init = true;
  dst.order_holders.emplace_back(dst_address);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MakeSureAtDevice(const ge::Node *node, LoweringGlobalData &global_data, ge::DataType data_type,
    const OutputLowerResult &src, bg::ValueHolderPtr &allocator, OutputLowerResult &dst) {
  auto dt_holder = bg::ValueHolder::CreateConst(&data_type, sizeof(data_type));
  auto tensor_size = GetTensorSize(global_data, data_type, src, dt_holder);
  std::vector<bg::ValueHolderPtr> copy_inputs{global_data.GetStream(), allocator};

  copy_inputs.emplace_back(src.address);
  copy_inputs.emplace_back(tensor_size);
  copy_inputs.emplace_back(src.shape);
  copy_inputs.emplace_back(dt_holder);

  const auto &output_addr = bg::DevMemValueHolder::CreateSingleDataOutput("MakeSureTensorAtDevice", copy_inputs,
                                                                          node->GetOpDescBarePtr()->GetStreamId());
  GE_ASSERT_NOTNULL(output_addr);
  GE_ASSERT_NOTNULL(bg::ValueHolder::CreateVoidGuarder("FreeMemory", output_addr, {}));

  dst.shape = src.shape;
  dst.order_holders = src.order_holders;
  dst.address = output_addr;
  dst.address->SetPlacement(allocator->GetPlacement());
  dst.has_init = true;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MakeSureAtDeviceHbm(const ge::Node *node, LoweringGlobalData &global_data, ge::DataType data_type,
                                    const OutputLowerResult &src, int64_t dst_logic_stream_id, OutputLowerResult &dst) {
  (void)dst_logic_stream_id;
  auto current_allocator = global_data.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  GE_ASSERT_NOTNULL(current_allocator);
  GE_ASSERT_SUCCESS(MakeSureAtDevice(node, global_data, data_type, src, current_allocator, dst));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MakeSureAtDeviceP2p(const ge::Node *node, LoweringGlobalData &global_data, ge::DataType data_type,
                                    const OutputLowerResult &src, int64_t dst_logic_stream_id, OutputLowerResult &dst) {
  (void)dst_logic_stream_id;
  auto current_allocator = global_data.GetOrCreateAllocator({kOnDeviceP2p, AllocatorUsage::kAllocNodeOutput});
  GE_ASSERT_NOTNULL(current_allocator);
  GE_ASSERT_SUCCESS(MakeSureAtDevice(node, global_data, data_type, src, current_allocator, dst));
  return ge::GRAPH_SUCCESS;
}

using OutputGenerator = std::function<ge::graphStatus(
    const ge::Node *node, LoweringGlobalData &global_data, ge::DataType data_type,
    const OutputLowerResult &src, int64_t dst_logic_stream_id, OutputLowerResult &dst)>;
// 多预留一个kTensorPlacementEnd的位置，用于保存“不知道Tensor在哪”的场景
const auto kOutputDataCopyer = TableDriven2<static_cast<size_t>(kTensorPlacementEnd) + 1,
    static_cast<size_t>(kTensorPlacementEnd) + 1, OutputGenerator>(nullptr)
    .Add(kTensorPlacementEnd, kOnDeviceHbm, MakeSureAtDeviceHbm)
    .Add(kTensorPlacementEnd, kOnDeviceP2p, MakeSureAtDeviceP2p)
    .Add(kTensorPlacementEnd, kOnHost, MakeSureAtHostDDR)
    .Add(kOnHost, kOnDeviceHbm, HostToDeviceHbm)
    .Add(kOnHost, kOnDeviceP2p, HostToDeviceP2p)
    .Add(kOnDeviceHbm, kOnHost, DeviceHbmToHost)
    .Add(kOnDeviceHbm, kOnDeviceP2p, MakeSureAtDeviceP2p)
    .Add(kOnDeviceP2p, kOnHost, DeviceHbmToHost)
    .Add(kOnDeviceP2p, kOnDeviceHbm, MakeSureAtDeviceHbm);

bool ShouldGenOnInit(const bg::ValueHolderPtr &shape_on_init, const bg::ValueHolderPtr &addr_on_init,
                     const std::vector<bg::ValueHolderPtr> &ordered_holders,
                     std::vector<bg::ValueHolderPtr> &ordered_holders_on_init) {
  if ((shape_on_init == nullptr) || (addr_on_init == nullptr)) {
    return false;
  }
  for (const auto &order : ordered_holders) {
    if (!bg::ValueHolderUtils::IsNodeValid(order)) {
      GELOGW(
          "The ordered holder does not point to a node on execute graph, do not gen tensor copy nodes on init graph");
      return false;
    }
    // 如果ordered holders出现在了Init节点上，说明ordered_holders来自于Init图内部，如果都能找到对应init graph中的节点
    // 则order holder替换为init graph中的节点
    if (strcmp(GetExecuteGraphTypeStr(ExecuteGraphType::kInit), bg::ValueHolderUtils::GetNodeTypeBarePtr(order)) == 0) {
      GELOGD("There are ordered holders from Init node, node %s",
             bg::ValueHolderUtils::GetNodeNameBarePtr(addr_on_init));
      const auto order_on_init = HolderOnInit(order);
      if (order_on_init == nullptr) {
        return false;
      }
      ordered_holders_on_init.emplace_back(order_on_init);
    }
  }
  return true;
}
}  // namespace
PlacedLoweringResult::PlacedLoweringResult(const ge::NodePtr &node, LowerResult &&lower_result)
    : node_(node.get()), result_(std::move(lower_result)) {
  auto result_num = result_.out_addrs.size();
  if (result_num != result_.out_shapes.size()) {
    GELOGW("The num of shapes and address are different %zu, %zu", result_.out_addrs.size(), result_.out_shapes.size());
    return;
  }
  constexpr size_t dim1 = 1U;
  constexpr size_t dim2 = 3U;
  TableDriven2<static_cast<size_t>(kTensorPlacementEnd) + dim1, dim2, OutputLowerResult> un_init_value{
      OutputLowerResult()};
  outputs_results_.resize(result_num, un_init_value);
  for (size_t i = 0U; i < result_num; ++i) {
    auto &ordered_holders = result_.order_holders;
    auto &shape = result_.out_shapes[i];
    auto &address = result_.out_addrs[i];
    if (address == nullptr) {
      continue;
    }
    auto placement = address->GetPlacement();
    if (placement < 0) {
      // don't know what the placement is, e.g. the `Data` node lower result.
      placement = kTensorPlacementEnd;
    }
    outputs_results_[i].Add(placement, static_cast<int32_t>(false), OutputLowerResult(ordered_holders, shape, address));
  }
}
const LowerResult *PlacedLoweringResult::GetResult() const {
  return &result_;
}

const OutputLowerResult *PlacedLoweringResult::GetOutputTensorResult(LoweringGlobalData &global_data,
                                                                     int32_t output_index,
                                                                     TargetAddrDesc target_addr_desc) {
  bool is_data_dependency = TensorPlacementUtils::IsOnHostNotFollowing(
      static_cast<TensorPlacement>(target_addr_desc.address_placement));
  auto result = GetOutputResult(global_data, output_index, target_addr_desc, is_data_dependency);
  GE_ASSERT_NOTNULL(result);
  if (is_data_dependency) {
    return result;
  }
  return CreateRawTensorResult(*result, output_index, target_addr_desc.address_placement);
}

const OutputLowerResult *PlacedLoweringResult::GetOutputResult(LoweringGlobalData &global_data, int32_t output_index,
                                                               TargetAddrDesc target_addr_desc,
                                                               bool is_data_dependent) {
  if (output_index < 0 || static_cast<size_t>(output_index) >= outputs_results_.size()) {
    GELOGE(ge::PARAM_INVALID, "Invalid output index %d, max size %zu", output_index, outputs_results_.size());
    return nullptr;
  }

  if (target_addr_desc.address_placement < 0) {
    // don't care the placement of the address, so use the prev node's output placement
    target_addr_desc.address_placement = result_.out_addrs[output_index]->GetPlacement();
  }

  auto &results = outputs_results_[output_index];
  auto result = results.FindPointer(target_addr_desc.address_placement, static_cast<int32_t>(is_data_dependent));
  GE_ASSERT_NOTNULL(result);
  if (result->has_init) {
    return result;
  }
  auto no_dependent_result = GetOrCreateNoDataDependentResult(global_data, output_index, target_addr_desc);
  GE_ASSERT_NOTNULL(no_dependent_result);
  if (!is_data_dependent) {
    return no_dependent_result;
  }
  TargetAddrDesc host_addr_desc = {kOnHost, 0};
  auto host_result = GetOrCreateNoDataDependentResult(global_data, output_index, host_addr_desc);
  GE_ASSERT_NOTNULL(host_result);
  return CreateDataDependentResult(output_index, target_addr_desc, *no_dependent_result, *host_result);
}

const OutputLowerResult *PlacedLoweringResult::GetOrCreateNoDataDependentResult(LoweringGlobalData &global_data,
                                                                                int32_t output_index,
                                                                                TargetAddrDesc &target_addr_desc) {
  auto &results = outputs_results_[output_index];
  auto result = results.FindPointer(target_addr_desc.address_placement, static_cast<int32_t>(false));
  GE_ASSERT_NOTNULL(result);
  if (result->has_init) {
    return result;
  }
  return CreateNoDataDependentResult(global_data, output_index, target_addr_desc);
}
const OutputLowerResult *PlacedLoweringResult::CreateNoDataDependentResult(LoweringGlobalData &global_data,
                                                                           int32_t output_index,
                                                                           TargetAddrDesc &target_addr_desc) {
  auto &result = result_.out_addrs[output_index];
  auto &output_results = outputs_results_[output_index];
  auto output_result = output_results.FindPointer(result->GetPlacement(), static_cast<int32_t>(false));
  auto target_result = output_results.FindPointer(target_addr_desc.address_placement, static_cast<int32_t>(false));
  GE_ASSERT_NOTNULL(output_result);
  GE_ASSERT_NOTNULL(target_result);

  const auto op_desc = node_->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto &td = op_desc->GetOutputDescPtr(output_index);
  GE_ASSERT_NOTNULL(td);
  auto priority_generator = PlacedLoweringRegistry::GetInstance().FindPlacedLower(node_->GetType());
  OutputGenerator generator;
  if (priority_generator != nullptr) {
    generator = priority_generator;
  } else {
    generator = kOutputDataCopyer.Find(result->GetPlacement(), target_addr_desc.address_placement);
  }
  GE_ASSERT_NOTNULL(generator);
  GELOGI("Create copy nodes from placement %d to %d for the %d output of node %s",
         result->GetPlacement(), target_addr_desc.address_placement, output_index, node_->GetNamePtr());

  auto shape_holder_on_init = HolderOnInit(output_result->shape);
  auto address_holder_on_init = HolderOnInit(output_result->address);
  std::vector<bg::ValueHolderPtr> order_holders_on_init;
  if (ShouldGenOnInit(shape_holder_on_init, address_holder_on_init,
                      output_result->order_holders, order_holders_on_init)) {
    OutputLowerResult result_from_init = {order_holders_on_init, shape_holder_on_init,
                                          std::dynamic_pointer_cast<bg::DevMemValueHolder>(address_holder_on_init)};
    auto outputs = bg::FrameSelector::OnInitRoot(
        [&generator, &global_data, &td, &result_from_init, this]() -> std::vector<bg::ValueHolderPtr> {
          OutputLowerResult result_on_init;
          auto ret = generator(node_, global_data, td->GetDataType(), result_from_init, 0, result_on_init);
          if (ret != ge::GRAPH_SUCCESS) {
            return {};
          }
          return {result_on_init.shape, result_on_init.address};
        });
    GE_ASSERT_EQ(outputs.size(), 2U);

    target_result->shape = std::move(outputs[0U]);
    auto output_addr = std::dynamic_pointer_cast<bg::DevMemValueHolder>(outputs[1U]);
    target_result->address = std::move(output_addr);
    target_result->order_holders = output_result->order_holders;
  } else {
    GE_ASSERT_SUCCESS(generator(node_, global_data, td->GetDataType(), *output_result, target_addr_desc.logic_stream_id,
                                *target_result));
  }

  target_result->has_init = true;
  return target_result;
}

const OutputLowerResult *PlacedLoweringResult::CreateDataDependentResult(int32_t output_index,
                                                                         TargetAddrDesc &target_addr_desc,
                                                                         const OutputLowerResult &result,
                                                                         const OutputLowerResult &host_result) {
  auto &results = outputs_results_[output_index];
  auto tensor_result = results.FindPointer(target_addr_desc.address_placement, static_cast<int32_t>(true));
  GE_ASSERT_NOTNULL(tensor_result);

  tensor_result->address = result.address;
  auto guarder = host_result.address->GetGuarder();
  if (guarder != nullptr) {
    tensor_result->shape = bg::BuildTensor(node_->shared_from_this(), output_index, kOnHost,
                                           result_.out_shapes[output_index], host_result.address);
  } else {
    tensor_result->shape = bg::BuildTensorWithoutGuarder(node_->shared_from_this(), output_index, kOnHost,
                                                         result_.out_shapes[output_index], host_result.address);
  }

  GE_ASSERT_NOTNULL(tensor_result->shape);
  tensor_result->order_holders = host_result.order_holders;
  std::set<bg::ValueHolderPtr> holder_set{host_result.order_holders.begin(), host_result.order_holders.end()};
  for (const auto &order_holder : result.order_holders) {
    if (holder_set.count(order_holder) == 0) {
      tensor_result->order_holders.emplace_back(order_holder);
    }
  }

  tensor_result->has_init = true;
  return tensor_result;
}

const OutputLowerResult *PlacedLoweringResult::CreateRawTensorResult(const OutputLowerResult &result,
                                                                     int32_t output_index,
                                                                     int32_t address_placement) {
  auto &results = outputs_results_[output_index];
  constexpr int32_t raw_tensor_index = 2;
  auto tensor_result = results.FindPointer(address_placement, raw_tensor_index);
  GE_ASSERT_NOTNULL(tensor_result);
  if (tensor_result->has_init) {
    return tensor_result;
  }
  tensor_result->address = result.address;
  auto guarder = result.address->GetGuarder();
  if (guarder != nullptr) {
    tensor_result->shape = bg::BuildRefTensor(node_->shared_from_this(), output_index,
                                              static_cast<TensorPlacement>(address_placement),
                                              result_.out_shapes[output_index], result.address);
  } else {
    tensor_result->shape = bg::BuildTensorWithoutGuarder(node_->shared_from_this(), output_index,
                                                         static_cast<TensorPlacement>(address_placement),
                                                         result_.out_shapes[output_index], result.address);
  }
  GE_ASSERT_NOTNULL(tensor_result->shape);
  tensor_result->order_holders = result.order_holders;
  tensor_result->has_init = true;
  return tensor_result;
}

}  // namespace gert
