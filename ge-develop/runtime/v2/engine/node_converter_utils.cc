/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engine/node_converter_utils.h"
#include "common/checker.h"
#include "graph_metadef/common/ge_common/util.h"
#include "framework/common/taskdown_common.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_types.h"
#include "framework/runtime/subscriber/built_in_subscriber_definitions.h"
#include "framework/runtime/subscriber/global_dumper.h"
#include "graph_builder/value_holder_generator.h"
#include "graph_builder/bg_memory.h"
#include "graph_builder/bg_checker.h"
#include "graph/debug/ge_op_types.h"
#include "graph/utils/graph_utils_ex.h"
#include "ge/ge_error_codes.h"
#include "exe_graph/lowering/frame_selector.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "mmpa/mmpa_api.h" // 临时引用，待删除

namespace gert {
namespace {
constexpr int64_t kUnknownDimNum = -2;
const std::set<ge::ModelTaskType> TBE_TASK_SET = {
    ge::ModelTaskType::MODEL_TASK_KERNEL,
    ge::ModelTaskType::MODEL_TASK_VECTOR_KERNEL,
    ge::ModelTaskType::MODEL_TASK_ALL_KERNEL,
    ge::ModelTaskType::MODEL_TASK_VECTOR_ALL_KERNEL
};
}
std::vector<bg::ValueHolderPtr> NodeConverterUtils::CreateOutputShapes(const ge::OpDescPtr &op_desc) {
  std::vector<ge::ConstGeTensorDescPtr> output_tensor_descs;
  for (size_t i = 0U; i < op_desc->GetOutputsSize(); ++i) {
    const auto &output_desc = op_desc->GetOutputDescPtr(i);
    output_tensor_descs.emplace_back(output_desc);
  }
  return CreateOutputShapes(output_tensor_descs);
}

std::vector<bg::ValueHolderPtr> NodeConverterUtils::CreateOutputShapes(
    const std::vector<ge::ConstGeTensorDescPtr> &output_tensor_descs) {
  std::vector<bg::ValueHolderPtr> outputs;
  for (const auto &output_desc : output_tensor_descs) {
    Tensor shape_tensor;
    auto output_shape = NodeConverterUtils::CreateOutputShape(output_desc);;
    outputs.emplace_back(output_shape);
  }
  return outputs;
}

static bg::ValueHolderPtr CreateShapeTensor(StorageShape &shape, const ge::ConstGeTensorDescPtr &output_desc) {
  StorageFormat storage_format;
  storage_format.SetOriginFormat(output_desc->GetOriginFormat());
  storage_format.SetStorageFormat(output_desc->GetFormat());
  Tensor shape_tensor(shape, storage_format, output_desc->GetDataType());
  return bg::ValueHolder::CreateConst(&shape_tensor, sizeof(shape_tensor));
}

bg::ValueHolderPtr NodeConverterUtils::CreateOutputShape(const ge::ConstGeTensorDescPtr &tensor_desc) {
  if (tensor_desc == nullptr) {
    Tensor shape_tensor;
    return bg::ValueHolder::CreateConst(&shape_tensor, sizeof(shape_tensor));
  }
  StorageShape storage_shape;
  storage_shape.MutableStorageShape() = NodeConverterUtils::GetShapeFromGeShape(tensor_desc->GetShape());
  storage_shape.MutableOriginShape() = NodeConverterUtils::GetShapeFromGeShape(tensor_desc->GetOriginShape());
  return CreateShapeTensor(storage_shape, tensor_desc);
}

bg::ValueHolderPtr NodeConverterUtils::CreateOutputShape(const ge::ConstGeTensorDescPtr &tensor_desc,
                                                         StorageShape &shape) {
  if (tensor_desc == nullptr) {
    Tensor shape_tensor(shape, {}, {});
    return bg::ValueHolder::CreateConst(&shape_tensor, sizeof(shape_tensor));
  }
  return CreateShapeTensor(shape, tensor_desc);
}

std::vector<bg::ValueHolderPtr> GetOrCreateInputFeeds(
    LoweringGlobalData *global_data, const ge::ComputeGraphPtr &graph) {
  GE_ASSERT_NOTNULL(graph);
  int32_t data_num = static_cast<int32_t>(ge::GraphUtilsEx::GetUserInputDataNodes(graph).size());
  GE_ASSERT_NOTNULL(global_data);
  const std::string feed_tensors_key = graph->GetName() + "_input_feed_tensors";
  return global_data->GetOrCreateUniqueValueHolder(feed_tensors_key,
      [&data_num] ()->std::vector<bg::ValueHolderPtr> {
        std::vector<bg::ValueHolderPtr> feed_tensors;
        for (int32_t i = 0; i < data_num; i++) {
          feed_tensors.emplace_back(bg::ValueHolder::CreateFeed(i));
        }
        return feed_tensors;});
}

Shape NodeConverterUtils::GetShapeFromGeShape(const ge::GeShape &ge_shape) {
  Shape shape;
  if (ge_shape.IsUnknownDimNum()) {
    shape.AppendDim(kUnknownDimNum);
    return shape;
  }
  shape.SetDimNum(ge_shape.GetDimNum());
  for (size_t i = 0U; i < ge_shape.GetDimNum(); ++i) {
    shape.SetDim(i, ge_shape.GetDim(i));
  }
  return shape;
}

inline bool IsAICpuTaskdef(ge::ccKernelType kernel_type) {
  return (kernel_type == ge::ccKernelType::AI_CPU) || (kernel_type == ge::ccKernelType::CUST_AI_CPU) ||
      (kernel_type == ge::ccKernelType::HOST_CPU);
}

const domi::TaskDef *GetTaskdefByType(const ge::NodePtr &node, std::vector<const domi::TaskDef *> &aicore_task_defs,
                                      std::vector<const domi::TaskDef *> &aicpu_task_defs, TaskDefType type) {
  if (aicore_task_defs.size() > ge::kNumTaskWithAtomicAddrCleanTask) {
    GELOGE(ge::INTERNAL_ERROR, "[NodeConverter][GetTaskDef][%s(%s)] At most %zu task was supported, but got %zu",
           node->GetName().c_str(), node->GetType().c_str(), ge::kNumTaskWithAtomicAddrCleanTask,
           aicore_task_defs.size());
    return nullptr;
  }
  if (type == TaskDefType::kAICpu) {
    const domi::TaskDef *ret = aicpu_task_defs.empty() ? nullptr : aicpu_task_defs[0];
    return ret;
  } else if (type == TaskDefType::kAICore) {
    return aicore_task_defs.empty() ? nullptr : aicore_task_defs.back();
  }
  if (aicore_task_defs.size() != ge::kNumTaskWithAtomicAddrCleanTask) {
    GELOGD("Node[%s] Expect %zu tasks, but got %zu",
           node->GetName().c_str(), ge::kNumTaskWithAtomicAddrCleanTask, aicore_task_defs.size());
    return nullptr;
  }
  return aicore_task_defs.front();
}

const domi::TaskDef *GetTaskDef(const ge::NodePtr &node, const LoweringGlobalData::NodeCompileResult *compile_result,
                                TaskDefType type) {
  GE_ASSERT_NOTNULL(compile_result);
  std::vector<const domi::TaskDef *> aicore_task_defs;
  std::vector<const domi::TaskDef *> aicpu_task_defs;
  for (const auto &task_def : compile_result->GetTaskDefs()) {
    const auto task_type = static_cast<ge::ModelTaskType>(task_def.type());
    if (TBE_TASK_SET.count(task_type) == 1) {
      bool is_sta_kernel = (task_type == ge::ModelTaskType::MODEL_TASK_KERNEL) ||
          (task_type == ge::ModelTaskType::MODEL_TASK_VECTOR_KERNEL);
      const auto &context =
          is_sta_kernel ? task_def.kernel().context() : task_def.kernel_with_handle().context();
      const auto kernel_type = static_cast<ge::ccKernelType>(context.kernel_type());
      if (kernel_type == ge::ccKernelType::TE || kernel_type == ge::ccKernelType::MIX_AICORE ||
          kernel_type == ge::ccKernelType::MIX_VECTOR_CORE) {
        aicore_task_defs.emplace_back(&task_def);
      } else if (IsAICpuTaskdef(kernel_type)) {
        aicpu_task_defs.emplace_back(&task_def);
      } else {
        GELOGE(ACL_ERROR_GE_OP_KERNEL_TYPE_INVALID,
               "[NodeConverter][GetTaskDef]Only TBE, AI_CPU, CUST_AI_CPU kernel are supported, but got %d",
               static_cast<int32_t>(kernel_type));
        return nullptr;
      }
    } else if (task_type == ge::ModelTaskType::MODEL_TASK_KERNEL_EX) {
      aicpu_task_defs.emplace_back(&task_def);
    } else {
      GELOGE(ge::FAILED, "[NodeConverter][GetTaskDef]Only AI_CORE and AI_CPU support, but got task_type %u", task_type);
      return nullptr;
    }
  }
  return GetTaskdefByType(node, aicore_task_defs, aicpu_task_defs, type);
}

std::vector<bg::ValueHolderPtr> CreateOutputShapes(const ge::OpDescPtr &op_desc) {
  std::vector<bg::ValueHolderPtr> outputs;
  for (size_t i = 0U; i < op_desc->GetOutputsSize(); ++i) {
    const auto &output_desc = op_desc->GetOutputDescPtr(i);
    outputs.emplace_back(NodeConverterUtils::CreateOutputShape(output_desc));
  }
  return outputs;
}

std::vector<bg::ValueHolderPtr> CreateInputOutputShapes(bool is_input, bool is_orishape,
                                                        const ge::OpDescPtr &op_desc) {
  std::vector<bg::ValueHolderPtr> shapes;
  size_t num = is_input ? op_desc->GetInputsSize() : op_desc->GetOutputsSize();
  for (size_t i = 0U; i < num; ++i) {
    const auto &desc = is_input ? op_desc->MutableInputDesc(i) : op_desc->MutableOutputDesc(i);
    if (desc != nullptr) {
      StorageShape shape;
      if (is_orishape) {
        shape.MutableOriginShape() = NodeConverterUtils::GetShapeFromGeShape(desc->GetOriginShape());
      } else {
        shape.MutableStorageShape() = NodeConverterUtils::GetShapeFromGeShape(desc->GetShape());
      }
      shapes.emplace_back(CreateShapeTensor(shape, desc));
    } else {
      Tensor shape_tensor;
      shapes.emplace_back(bg::ValueHolder::CreateConst(&shape_tensor, sizeof(shape_tensor)));
    }
  }
  return shapes;
}

std::vector<bg::ValueHolderPtr> CalcInputOutputTensorSize(bool is_input, bool is_orishape,
                                                          const ge::NodePtr &node,
                                                          const std::vector<bg::ValueHolderPtr> &shapes) {
  std::vector<bg::ValueHolderPtr> shapes_sizes;
  for (size_t i = 0U; i < shapes.size(); ++i) {
    const auto &td = is_input ? node->GetOpDescBarePtr()->GetInputDescPtr(static_cast<int32_t>(i)) :
                     node->GetOpDescBarePtr()->GetOutputDescPtr(static_cast<int32_t>(i));
    if (td == nullptr) {
      return {};
    }
    auto shape_size = is_orishape ? bg::CalcTensorSizeFromShape(td->GetDataType(), shapes[i]) :
                                    bg::CalcTensorSizeFromStorage(td->GetDataType(), shapes[i]);
    if (shape_size == nullptr) {
      return {};
    }
    shapes_sizes.emplace_back(std::move(shape_size));
  }
  return shapes_sizes;
}

bg::ValueHolderPtr ConvertWorkspaceSize(const ge::NodePtr &node) {
  const std::vector<int64_t> &workspace_bytes = node->GetOpDescBarePtr()->GetWorkspaceBytes();
  if (workspace_bytes.empty()) {
    GELOGD("Node[%s] do not need workspace.", node->GetNamePtr());
  }
  return bg::CreateContVecHolder(workspace_bytes);
}

bool IsDataDumpOpen() {
  for (uint32_t i = 0; i < static_cast<uint32_t>(gert::DumpType::kNum); ++i) {
    auto i_type = static_cast<gert::DumpType>(i);
    if (gert::GlobalDumper::GetInstance()->IsEnable(i_type)) {
      return true;
    }
  }
  return false;
}

bool IsExceptionDumpOpen() {
  if (gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kExceptionDump)) {
    return true;
  }
  return false;
}

bool GetDfxOptFlagByType(const ge::NodePtr node, OpDfxOpt opt_type) {
  std::vector<std::string> dfx_opts;
  (void)ge::AttrUtils::GetListStr(node->GetOpDesc(), gert::kOpDfxOptions, dfx_opts);
  std::string opt_type_str = opt_type == OpDfxOpt::PRINT ? kOpDfxPrintf : kOpDfxAssert;
  if (std::find(dfx_opts.begin(), dfx_opts.end(), opt_type_str) != dfx_opts.end()) {
    GELOGD("Node[%s] need [%s] dfx.", node->GetNamePtr(), opt_type_str.c_str());
    return true;
  }
  return false;
}
} // namespace gert
