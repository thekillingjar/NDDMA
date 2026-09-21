/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_callback.h"
#include "graph_builder/bg_infer_shape.h"
#include "graph_builder/bg_memory.h"
#include "engine/aicpu/graph_builder/bg_aicpu_arg.h"
#include "engine/aicpu/graph_builder/bg_memcpy.h"
#include "engine/aicpu/graph_builder/bg_launch.h"
#include "engine/aicpu/graph_builder/bg_ext_info.h"
#include "engine/aicpu/kernel/aicpu_ext_info_handle.h"
#include "engine/aicpu/kernel/aicpu_resource_manager.h"
#include "framework/common/ge_types.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "lowering/lowering_utils.h"
#include "graph_builder/bg_rt_session.h"
#include "exe_graph/lowering/frame_selector.h"
#include "graph_metadef/common/ge_common/util.h"

namespace gert {
namespace {
void DependShapeRangeCallback(const bg::ValueHolderPtr &ext_info, const bg::ValueHolderPtr& launch_holder,
                              const bg::ValueHolderPtr &stream, std::vector<bg::ValueHolderPtr> &output_shapes,
                              const std::string &engine_name) {
  auto sync_stream = bg::ValueHolder::CreateVoid<bg::ValueHolder>("SyncStream", {stream});
  const size_t output_num = output_shapes.size();
  auto engine_name_holder = bg::ValueHolder::CreateConst(engine_name.c_str(), engine_name.size() + 1, true);
  auto ext_output_shapes = bg::ValueHolder::CreateDataOutput("GetExtOutputShapes", {ext_info, engine_name_holder}, output_num);

  bg::ValueHolder::AddDependency(launch_holder, sync_stream);
  for (auto shape : ext_output_shapes) {
    bg::ValueHolder::AddDependency(sync_stream, shape);
  }
  output_shapes = ext_output_shapes;
}

void DependShapeRangeHostCpuCallback(const bg::ValueHolderPtr &ext_info, const bg::ValueHolderPtr& launch_holder,
                                     std::vector<bg::ValueHolderPtr> &output_shapes) {
  const size_t output_num = output_shapes.size();
  const std::string host_engine_name = ge::kEngineNameHostCpu;
  auto engine_name_holder = bg::ValueHolder::CreateConst(host_engine_name.c_str(), host_engine_name.size() + 1, true);
  auto ext_output_shapes = bg::ValueHolder::CreateDataOutput("GetExtOutputShapes",
      {ext_info, engine_name_holder}, output_num);

  for (auto shape : ext_output_shapes) {
    bg::ValueHolder::AddDependency(launch_holder, shape);
  }
  output_shapes = ext_output_shapes;
}

std::vector<bg::DevMemValueHolderPtr> AllocMemcpyInput(const size_t output_num, LoweringGlobalData &global_data) {
  auto vec = bg::FrameSelector::OnInitRoot([&output_num, &global_data]()
      -> std::vector<bg::ValueHolderPtr> {
    std::vector<bg::ValueHolderPtr> res;
    // copy task need copy output_data and output_shape, max len is 2 * output_num
    const size_t copy_buf_len = output_num * 2U * sizeof(void *);
    auto copy_buf_len_holder = bg::ValueHolder::CreateConst(&copy_buf_len, sizeof(copy_buf_len));
    /**
     * Init graph must be in main stream
     * The four inputs represent the input parameters of the MemCopy operator,
     * which are release_flag, data_size, src_ptr, dst_ptr.
     * For details, see the enum class CopyInputs in memcpy_task.cc on the host
     * and file mem_copy_ops.cc on the device.
     */
    res.emplace_back(bg::AllocMem(kOnDeviceHbm, copy_buf_len_holder, global_data, bg::kMainStream));
    res.emplace_back(bg::AllocMem(kOnDeviceHbm, copy_buf_len_holder, global_data, bg::kMainStream));
    res.emplace_back(bg::AllocMem(kOnDeviceHbm, copy_buf_len_holder, global_data, bg::kMainStream));
    res.emplace_back(bg::AllocMem(kOnDeviceHbm, copy_buf_len_holder, global_data, bg::kMainStream));
    return res;
  });
  std::vector<bg::DevMemValueHolderPtr> memcpy_input;
  for (const auto &input : vec) {
    memcpy_input.emplace_back(std::dynamic_pointer_cast<bg::DevMemValueHolder>(input));
  }
  return memcpy_input;
}

bg::ValueHolderPtr PrepareCopyInputs(const ge::NodePtr &node,
                                     const bg::ValueHolderPtr &ordered_holder,
                                     const std::vector<bg::DevMemValueHolderPtr> &memcpy_input,
                                     LoweringGlobalData &global_data,
                                     NodeOutput &node_output) {
  auto output_num = node->GetAllOutDataAnchorsSize();
  auto host_summary = bg::GetHostSummary(node_output.addrs);
  auto data_sizes = bg::GetSummaryDataSizes(host_summary);
  auto shape_sizes = bg::GetSummaryShapeSizes(host_summary);
  auto output_memory = bg::AllocOutputMemory(kOnDeviceHbm, node, data_sizes, global_data);
  node_output.addrs.clear();
  node_output.addrs.insert(node_output.addrs.begin(), output_memory.begin(), output_memory.end());
  auto shape_buffer = bg::AllocOutputMemory(kOnDeviceHbm, node, shape_sizes, global_data);

  std::vector<bg::ValueHolderPtr> inputs;
  inputs.emplace_back(bg::ValueHolder::CreateConst(&output_num, sizeof(output_num)));
  inputs.insert(inputs.cend(), memcpy_input.cbegin(), memcpy_input.cend());
  inputs.insert(inputs.cend(), host_summary.cbegin(), host_summary.cend());
  inputs.insert(inputs.cend(), node_output.addrs.cbegin(), node_output.addrs.cend());
  inputs.insert(inputs.cend(), shape_buffer.cbegin(), shape_buffer.cend());
  inputs.emplace_back(global_data.GetStream());
  auto prepare_copy_inputs =  bg::ValueHolder::CreateVoid<bg::ValueHolder>("PrepareCopyInputs", inputs);

  for (auto &summary : host_summary) {
    bg::ValueHolder::AddDependency(ordered_holder, summary);
  }

  node_output.shapes = bg::GetOutputShapeFromHbmBuffer(host_summary, shape_buffer);
  return prepare_copy_inputs;
}

ge::Status AddDataNodeForMemCpy(ge::ComputeGraphPtr &graph, ge::NodePtr &memcpy_node, size_t input_size) {
  // add data node for output
  auto ret = ge::SUCCESS;
  for (size_t i = 0U; i < input_size; ++i) {
    auto data_op_desc =
        std::make_shared<ge::OpDesc>(memcpy_node->GetName() + "_Data_" + std::to_string(i + 1), "Data");
    GE_CHECK_NOTNULL(data_op_desc);
    if (data_op_desc->AddOutputDesc(ge::GeTensorDesc()) != ge::SUCCESS) {
      GELOGE(ge::FAILED, "data_op_desc add input desc failed, i = %zu", i);
      return ge::FAILED;
    }
    auto data_node = graph->AddNode(data_op_desc);
    GE_CHECK_NOTNULL(data_node);
    ret = ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), memcpy_node->GetInDataAnchor(i));
    if (ret != ge::SUCCESS) {
      GELOGE(ge::FAILED, "add edge between [%s] and [%s] failed", data_node->GetName().c_str(),
             memcpy_node->GetName().c_str());
      return ge::FAILED;
    }
  }
  return ret;
}

ge::NodePtr BuildMemCpyTaskNode(ge::ComputeGraphPtr &graph, const ge::NodePtr &origin_node, uint32_t input_size) {
  GELOGD("Generate cc/tf memcpy logic for node %s type %s.",
         origin_node->GetName().c_str(), origin_node->GetType().c_str());
  ge::OpDescPtr memcpy_op_desc =
      std::make_shared<ge::OpDesc>(origin_node->GetName() + "_FakeMemCpy", "MemCopy");
  if (memcpy_op_desc != nullptr) {
    for (uint32_t i = 0; i < input_size; ++i) {
      memcpy_op_desc->AppendIrInput("input" + std::to_string(i + 1), ge::kIrInputRequired);
      memcpy_op_desc->AddInputDesc("input" + std::to_string(i + 1), ge::GeTensorDesc());
    }
    GE_ASSERT_NOTNULL(origin_node->GetOpDescBarePtr());
    auto memcpy_node = LoweringUtils::CreateEngineTaskNode(graph, memcpy_op_desc, origin_node);
    if (memcpy_node != nullptr) {
      if (AddDataNodeForMemCpy(graph, memcpy_node, input_size) != ge::SUCCESS) {
        GELOGE(ge::FAILED, "add data node for memcpy node failed, input size = %u", input_size);
        return nullptr;
      }
      std::vector<ge::NodePtr> in_out_nodes;
      auto in_nodes = memcpy_node->GetInNodes();
      auto out_nodes = memcpy_node->GetOutNodes();
      in_out_nodes.insert(in_out_nodes.cend(), in_nodes.begin(), in_nodes.end());
      in_out_nodes.insert(in_out_nodes.cend(), out_nodes.begin(), out_nodes.end());
      memcpy_node->GetOpDescBarePtr()->SetExtAttr("_in_out_nodes_holder", in_out_nodes);
      return memcpy_node;
    }
  }
  return nullptr;
}

void LaunchTfMemcpyTask(const ge::NodePtr &node, bg::ValueHolderPtr& ordered_holder,
                        LoweringGlobalData &global_data, NodeOutput &node_output) {
  auto compile_result = global_data.FindCompiledResult(node);
  auto &task_def = compile_result->task_defs.back();
  auto &kernel_ex_def = task_def.kernel_ex();
  auto session_id = bg::GetSessionId(global_data);

  auto outputs_num = node->GetAllOutDataAnchorsSize();
  auto memcpy_input = AllocMemcpyInput(outputs_num, global_data);

  auto tmp_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  if (tmp_graph == nullptr) {
    return;
  }

  auto tf_memcpy_node = BuildMemCpyTaskNode(tmp_graph, node, outputs_num);
  if (tf_memcpy_node == nullptr) {
    GELOGE(ge::PARAM_INVALID, "build tf memcpy task node failed, tf 4th memcpy doest not finished yet.");
    return;
  }
  // Return value of this function will be implicitly used.
  // If the value is ignored, there will be functional problems.
  auto current_node_guarder = bg::ValueHolder::SetScopedCurrentComputeNode(tf_memcpy_node);

  // gen function handle
  auto rts_args = bg::BuildTfArgsBinHandle(node);

  // build tf memcpy task
  auto step_id = GetStepId(global_data);
  auto aicpu_tf_args = bg::BuildTfAicpuArg(node, {kernel_ex_def, memcpy_input.size(), session_id, step_id}, true);

  // update args(may move to init graph by CEM) & prepare inputs
  auto update_tf_holder = bg::UpdateAicpuIoAddr(aicpu_tf_args.args_handler, memcpy_input, {});
  auto prepare_copy_inputs = PrepareCopyInputs(node, ordered_holder, memcpy_input, global_data, node_output);

  // launch & sync
  auto launch_tf_holder = bg::AicpuTfLaunchKernel(aicpu_tf_args.args_handler, global_data.GetStream(), rts_args.bin_handle, node);
  auto sync_stream = bg::ValueHolder::CreateSingleDataOutput("SyncStream", {global_data.GetStream()});

  SetReleaseAfter(memcpy_input, launch_tf_holder);
  SetReleaseAfter(node_output.addrs, launch_tf_holder);

  bg::ValueHolder::AddDependency(update_tf_holder, launch_tf_holder);
  bg::ValueHolder::AddDependency(prepare_copy_inputs, launch_tf_holder);
  bg::ValueHolder::AddDependency(launch_tf_holder, sync_stream);
  for (auto &shape : node_output.shapes) {
    bg::ValueHolder::AddDependency(sync_stream, shape);
    ordered_holder = shape;
  }
}

void LaunchCCMemcpyTask(const ge::NodePtr &node, bg::ValueHolderPtr& ordered_holder,
                        LoweringGlobalData &global_data, NodeOutput &node_output) {
  auto compile_result = global_data.FindCompiledResult(node);
  if (compile_result == nullptr) {
    GELOGE(ge::PARAM_INVALID, "get compile result failed");
    return;
  }
  auto &task_def = compile_result->task_defs.back();
  auto &kernel_def = task_def.kernel();
  auto session_id = bg::GetSessionId(global_data);

  auto outputs_num = node->GetAllOutDataAnchorsSize();
  auto memcpy_input = AllocMemcpyInput(outputs_num, global_data);

  auto tmp_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  if (tmp_graph == nullptr) {
    return;
  }

  auto cc_memcpy_node = BuildMemCpyTaskNode(tmp_graph, node, outputs_num);
  if (cc_memcpy_node == nullptr) {
    GELOGE(ge::PARAM_INVALID, "build cc memcpy task node failed, cc 4th memcpy doest not finished yet.");
    return;
  }
  // Return value of this function will be implicitly used.
  // If the value is ignored, there will be functional problems.
  auto current_node_guarder = bg::ValueHolder::SetScopedCurrentComputeNode(cc_memcpy_node);
  // gen function handle
  auto rts_args = bg::BuildCCArgsBinHandle(node);

  // build cc memcpy task
  auto aicpu_cc_args = bg::BuildCCAicpuArg(node, kernel_def, memcpy_input.size(), session_id, true);

  // update args(may move to init graph by CEM) & prepare inputs
  auto update_cc_holder = bg::UpdateAicpuIoAddr(aicpu_cc_args.args_handler, memcpy_input, {});
  auto prepare_copy_inputs = PrepareCopyInputs(node, ordered_holder, memcpy_input, global_data, node_output);

  // launch & sync
  size_t block_dim = 1U;
  auto block_dim_holder = bg::ValueHolder::CreateConst(&block_dim, sizeof(block_dim));
  auto launch_cc_holder = bg::AicpuCCLaunchKernel(aicpu_cc_args.args_handler, global_data.GetStream(), block_dim_holder,
                                                  kernel_def, node->GetOpDesc(), aicpu_cc_args.ext_info_handler,
                                                  rts_args.bin_handle, node);
  auto sync_stream = bg::ValueHolder::CreateSingleDataOutput("SyncStream", {global_data.GetStream()});

  SetReleaseAfter(memcpy_input, launch_cc_holder);
  SetReleaseAfter(node_output.addrs, launch_cc_holder);

  bg::ValueHolder::AddDependency(prepare_copy_inputs, launch_cc_holder);
  bg::ValueHolder::AddDependency(update_cc_holder, launch_cc_holder);
  bg::ValueHolder::AddDependency(launch_cc_holder, sync_stream);
  for (auto &shape : node_output.shapes) {
    bg::ValueHolder::AddDependency(sync_stream, shape);
    ordered_holder = shape;
  }
}
} // namespace
bg::ValueHolderPtr GetStepId(LoweringGlobalData &global_data) {
  if (AicpuResourceManager::GetInstance().IsSingleOp()) {
    // must set node on init graph
    auto step_id = bg::FrameSelector::OnInitRoot([]() -> std::vector<bg::ValueHolderPtr> {
      int64_t step_id_local = 0;
      return std::vector<bg::ValueHolderPtr>({bg::ValueHolder::CreateConst(&step_id_local, sizeof(step_id_local))});
    });
    return step_id[0];
  }

  auto builder = [&global_data]() -> bg::ValueHolderPtr {
    auto step_id_holders = bg::FrameSelector::OnInitRoot([]() -> std::vector<bg::ValueHolderPtr> {
      return bg::ValueHolder::CreateDataOutput("CreateStepId", {}, 2U); // device & host step_id
    });

    bg::FrameSelector::OnMainRootLast([&step_id_holders, &global_data] () -> bg::ValueHolderPtr {
      // step id作为全局资源，需要保证读写step id时都在同一条流上，所以将其固定在stream 0上
      step_id_holders.emplace_back(global_data.GetStreamById(0));
      return bg::ValueHolder::CreateVoid<bg::ValueHolder>("IncreaseStepId", step_id_holders);
    });
    return step_id_holders[0];
  };
  return global_data.GetOrCreateUniqueValueHolder("step_id", builder);
}

void SetReleaseAfter(const std::vector<bg::ValueHolderPtr> &addrs, const bg::ValueHolderPtr &launch_holder) {
  for (const auto &addr : addrs) {
    addr->ReleaseAfter(launch_holder);
  }
}
void SetReleaseAfter(const std::vector<bg::DevMemValueHolderPtr> &addrs, const bg::ValueHolderPtr &launch_holder) {
  for (const auto &addr : addrs) {
    addr->ReleaseAfter(launch_holder);
  }
}

NodeOutput GetOutputShapeAndAddr(const ge::NodePtr &node, const std::vector<bg::ValueHolderPtr> &input_shapes,
                                 const std::vector<bg::DevMemValueHolderPtr> &input_addrs,
                                 LoweringGlobalData &global_data) {
  NodeOutput res;
  int32_t unknown_shape_type_val = 0;
  (void)ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  if ((bg::IsAicpuUnknownShape(node)) &&
      (unknown_shape_type_val == static_cast<int32_t>(ge::DEPEND_COMPUTE))) {
    auto output_num = node->GetAllOutDataAnchorsSize();
    auto summary_len = sizeof(aicpu::FWKAdapter::ResultSummary);
    auto len_holder = bg::ValueHolder::CreateConst(&summary_len, sizeof(summary_len));
    for (size_t i = 0U; i < output_num; ++i) {
      auto summary = bg::AllocMem(kOnDeviceHbm, len_holder, global_data, node->GetOpDescBarePtr()->GetStreamId());
      res.addrs.emplace_back(summary);
    }
  } else {
    res.shapes = bg::GetMemAllocShape(node, input_shapes, global_data);
    auto output_sizes = bg::CalcTensorSize(node, res.shapes);
    auto memory = bg::AllocOutputMemory(kOnDeviceHbm, node, output_sizes, input_addrs, global_data);
    res.addrs.insert(res.addrs.begin(), memory.begin(), memory.end());
 }
  return res;
}

void AicpuCallback(const ge::NodePtr &node, const bg::ValueHolderPtr &ext_info, bg::ValueHolderPtr &launch_holder,
                   LoweringGlobalData &global_data, NodeOutput &node_output) {
  int32_t unknown_shape_type_val = 0;
  (void) ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  if ((unknown_shape_type_val == static_cast<int32_t>(ge::DEPEND_SHAPE_RANGE)) &&
      bg::IsAicpuOutputUnknownShape(node)) {
    GELOGI("Op %s type %s is 3th op. engine:%s", node->GetName().c_str(),
        ge::NodeUtils::GetNodeType(node).c_str(), node->GetOpDescBarePtr()->GetOpKernelLibName().c_str());
    if (node->GetOpDescBarePtr()->GetOpKernelLibName() == ge::kEngineNameHostCpu) {
      DependShapeRangeHostCpuCallback(ext_info, launch_holder, node_output.shapes);
      return;
    }
    DependShapeRangeCallback(ext_info, launch_holder, global_data.GetStream(),
        node_output.shapes, node->GetOpDescBarePtr()->GetOpKernelLibName());
  } else if ((unknown_shape_type_val == static_cast<int32_t>(ge::DEPEND_COMPUTE)) &&
             bg::IsAicpuUnknownShape(node)) {
    auto sync_stream = bg::ValueHolder::CreateVoid<bg::ValueHolder>("SyncStream", {global_data.GetStream()});
    bg::ValueHolder::AddDependency(launch_holder, sync_stream);
    launch_holder = sync_stream;
    if (node->GetOpDescBarePtr()->GetOpKernelLibName() == ge::kEngineNameAiCpuTf) {
      GELOGI("Op %s type %s is 4th tf op.",
          node->GetName().c_str(), ge::NodeUtils::GetNodeType(node).c_str());
      return LaunchTfMemcpyTask(node, launch_holder, global_data, node_output);
    } else if (node->GetOpDescBarePtr()->GetOpKernelLibName() == ge::kEngineNameAiCpu) {
      GELOGI("Op %s type %s is 4th cc op.", node->GetName().c_str(), ge::NodeUtils::GetNodeType(node).c_str());
      return LaunchCCMemcpyTask(node, launch_holder, global_data, node_output);
    } else {
      GELOGI("Skip node kernel name: %s", node->GetOpDescBarePtr()->GetOpKernelLibName().c_str());
    }
  }
}
}  // namespace gert
