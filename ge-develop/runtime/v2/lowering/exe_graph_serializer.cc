/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph_serializer.h"

#include <unordered_map>

#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "core/builder/node_types.h"
#include "engine/aicore/fe_rt2_common.h"
#include "engine/node_converter_utils.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "exe_graph/runtime/context_extend.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/types.h"
#include "framework/runtime/model_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_type_utils.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "subscriber/profiler/dfx_extend_info.h"
#include "graph_metadef/common/ge_common/util.h"

namespace gert {
namespace {
const std::string kOpImplModeEnum = "_op_impl_mode_enum";

const std::unordered_map<std::string, MsprofGeTaskType> task_type_str_to_int = {
    {ge::kTaskTypeAiv, MSPROF_GE_TASK_TYPE_AIV},
    {ge::kTaskTypeMixAic, MSPROF_GE_TASK_TYPE_MIX_AIC},
    {ge::kTaskTypeMixAiv, MSPROF_GE_TASK_TYPE_MIX_AIV},
    {ge::kTaskTypeAicore, MSPROF_GE_TASK_TYPE_AI_CORE},
    {ge::kTaskTypeAicpu, MSPROF_GE_TASK_TYPE_AI_CPU}};
ShapeRange ConstructShapeRange(const std::vector<std::pair<int64_t, int64_t>> &range) {
  ShapeRange shape_range;

  shape_range.MutableMin().SetDimNum(range.size());
  shape_range.MutableMax().SetDimNum(range.size());
  for (size_t i = 0U; i < range.size(); ++i) {
    shape_range.MutableMin().SetDim(i, range[i].first);
    shape_range.MutableMax().SetDim(i, range[i].second);
  }
  return shape_range;
}
ge::graphStatus GetIoDescFromTensor(const std::vector<ge::GeTensorDesc> &ge_descs,
                                    const std::vector<std::string> &names, bg::BufferPool &buffer_pool,
                                    ModelIoDesc *io_descs) {
  for (size_t i = 0U; i < ge_descs.size(); ++i) {
    auto io_desc = io_descs + i;
    auto name_id = buffer_pool.AddStr(names[i].c_str());
    GE_ASSERT_NOTNULL(io_desc);
    io_desc->SetName(reinterpret_cast<const char *>(name_id));
    io_desc->SetDataType(ge_descs[i].GetDataType());

    io_desc->SetStorageFormat(ge_descs[i].GetFormat());
    io_desc->SetOriginFormat(ge_descs[i].GetOriginFormat());

    io_desc->MutableStorageShape() = NodeConverterUtils::GetShapeFromGeShape(ge_descs[i].GetShape());
    io_desc->MutableOriginShape() = NodeConverterUtils::GetShapeFromGeShape(ge_descs[i].GetOriginShape());

    std::vector<std::pair<int64_t, int64_t>> range;
    ge_descs[i].GetShapeRange(range);
    io_desc->MutableStorageShapeRange() = ConstructShapeRange(range);
    range.clear();
    ge_descs[i].GetOriginShapeRange(range);
    io_desc->MutableOriginShapeRange() = ConstructShapeRange(range);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetAippDims(ModelIoDesc *io_descs, const std::vector<std::vector<int64_t>> &aipp_all_input_dims) {
  GE_ASSERT_NOTNULL(io_descs);
  for (size_t i = 0U; i < aipp_all_input_dims.size(); ++i) {
    auto io_desc = io_descs + i;
    GE_ASSERT_NOTNULL(io_desc);
    io_desc->MutableAippShape().SetDimNum(aipp_all_input_dims[i].size());
    for (size_t j = 0U; j < aipp_all_input_dims[i].size(); ++j) {
      io_desc->MutableAippShape().SetDim(j, aipp_all_input_dims[i][j]);
    }
  }
  return ge::GRAPH_SUCCESS;
}

void GetCMOAttrs(const ge::OpDesc *const op_desc, std::vector<std::vector<uint32_t>> &cmo_context_ids,
                 std::vector<std::string> &cmo_context_names, std::vector<uint32_t> &cmo_context_types) {
  for (uint32_t i = 0U; i < CMO_IDX_KEY.size(); i++) {
    const std::string ctx_id_attr = kDataProfCtxIdV + CMO_IDX_KEY.at(i);
    const std::string name_attr = kDataProfName + CMO_IDX_KEY.at(i);
    const std::string type_attr = kDataProfType + CMO_IDX_KEY.at(i);
    std::vector<uint32_t> cmo_context_ids_tmp;
    std::string cmo_context_name_tmp;
    uint32_t cmo_context_type_tmp;
    (void)ge::AttrUtils::GetListInt(op_desc, ctx_id_attr, cmo_context_ids_tmp);
    const auto *cmo_name_ptr = ge::AttrUtils::GetStr(op_desc, name_attr);
    if (cmo_name_ptr != nullptr) {
      cmo_context_name_tmp = *cmo_name_ptr;
    }
    (void)ge::AttrUtils::GetInt(op_desc, type_attr, cmo_context_type_tmp);

    cmo_context_ids.push_back(cmo_context_ids_tmp);
    cmo_context_names.push_back(cmo_context_name_tmp);
    cmo_context_types.push_back(cmo_context_type_tmp);
  }
}
}  // namespace
ge::graphStatus ExeGraphSerializer::SaveSerialization(const std::vector<ge::FastNode *> &all_exe_nodes) const {
  auto &graph = execute_graph_;
  GE_ASSERT_NOTNULL(graph);
  size_t buffer_size;

  bg::BufferPool buffer_pool;
  bg::BufferPool compute_node_infos;
  bg::BufferPool kernel_extend_infos;
  bg::BufferPool model_desc;
  bg::BufferPool dfx_extend_infos;

  GE_ASSERT_SUCCESS(SerializeComputeNodeInfo(all_exe_nodes, compute_node_infos, buffer_pool));
  GE_ASSERT_SUCCESS(SerializeKernelExtendInfo(all_exe_nodes, kernel_extend_infos, buffer_pool));
  GE_ASSERT_SUCCESS(SerializeModelDesc(model_desc));
  GE_ASSERT_SUCCESS(SerializeDfxExtendInfo(dfx_extend_infos, buffer_pool));

  // buffer pool
  auto buffer = buffer_pool.Serialize(buffer_size);
  GE_ASSERT_NOTNULL(buffer);
  GE_ASSERT_TRUE(ge::AttrUtils::SetZeroCopyBytes(graph, kBuffer, ge::Buffer::CopyFrom(buffer.get(), buffer_size)));

  // compute node info
  buffer = compute_node_infos.Serialize(buffer_size);
  GE_ASSERT_NOTNULL(buffer);
  GE_ASSERT_TRUE(
      ge::AttrUtils::SetZeroCopyBytes(graph, kComputeNodeInfo, ge::Buffer::CopyFrom(buffer.get(), buffer_size)));

  // kernel extend info
  buffer = kernel_extend_infos.Serialize(buffer_size);
  GE_ASSERT_NOTNULL(buffer);
  GE_ASSERT_TRUE(
      ge::AttrUtils::SetZeroCopyBytes(graph, kKernelExtendInfo, ge::Buffer::CopyFrom(buffer.get(), buffer_size)));

  // dfx extend info
  buffer = dfx_extend_infos.Serialize(buffer_size);
  GE_ASSERT_NOTNULL(buffer);
  GE_ASSERT_TRUE(
      ge::AttrUtils::SetZeroCopyBytes(graph, "DfxExtendInfo", ge::Buffer::CopyFrom(buffer.get(), buffer_size)));

  // model_desc
  buffer = model_desc.Serialize(buffer_size);
  GE_ASSERT_NOTNULL(buffer);
  GE_ASSERT_TRUE(ge::AttrUtils::SetZeroCopyBytes(graph, kModelDesc, ge::Buffer::CopyFrom(buffer.get(), buffer_size)));
  // todo:先用扩展属性携带，后续改为真正的序列化
  GE_ASSERT_TRUE(graph->SetExtAttr(kSpaceRegistry, model_desc_holder_->GetSpaceRegistries()));
  GE_ASSERT_TRUE(graph->SetExtAttr(kExternalFileConstantDir, model_desc_holder_->GetFileConstantWeightDir()));
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus ExeGraphSerializer::SerializeDfxExtendInfo(bg::BufferPool &dfx_extend_infos,
                                                           bg::BufferPool &buffer_pool) const {
  GE_ASSERT_NOTNULL(compute_graph_);
  for (const auto compute_node : compute_graph_->GetAllNodesPtr()) {
    const auto node_name_index = buffer_pool.AddStr(compute_node->GetNamePtr());
    // 设置dfx属性，UINT32_MAX表示属性获取失败，设置非法值，用于反序列化时候指示
    uint32_t task_type = UINT32_MAX;
    const auto op_desc = compute_node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    auto task_type_str = ge::AttrUtils::GetStr(op_desc, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE);
    if (task_type_str != nullptr) {
      const auto iter = task_type_str_to_int.find(*task_type_str);
      if (iter != task_type_str_to_int.end()) {
        task_type = static_cast<uint32_t>(iter->second);
      } else {
        GELOGW("Unsupported task type %s. node name is %s", task_type_str->c_str(), compute_node->GetNamePtr());
      }
    }

    uint32_t ctx_type = 0U;
    if (!ge::AttrUtils::GetInt(op_desc, kFFTSProfCtxType, ctx_type)) {
      GELOGW("get attr _ffts_prof_ctx_type unsuccessful.");
      ctx_type = UINT32_MAX;
    }

    vector<uint32_t> ctx_id_vec;
    if (!ge::AttrUtils::GetListInt(op_desc, kContextIdList, ctx_id_vec)) {
      GELOGW("get attr _context_id_list unsuccessful.");
    }
    uint32_t op_impl_mode = 0U;
    if (!ge::AttrUtils::GetInt(op_desc, kOpImplModeEnum, op_impl_mode)) {
      GELOGW("get attr _op_impl_mode_enum unsuccessful.");
      op_impl_mode = UINT32_MAX;
    }

    // 用vector下表对应cmo属性
    std::vector<std::vector<uint32_t>> cmo_context_ids;
    std::vector<std::string> cmo_context_names;
    std::vector<uint32_t> cmo_context_types;
    GetCMOAttrs(op_desc, cmo_context_ids, cmo_context_names, cmo_context_types);

    // 初始化dfxExtendInfo POD
    size_t total_size = 0U;
    GE_ASSERT_SUCCESS(DfxExtendInfo::CalcSize(ctx_id_vec.size(), cmo_context_ids, total_size));
    std::unique_ptr<uint8_t[]> dfx_extend_info_holder = ge::MakeUnique<uint8_t[]>(total_size);
    GE_ASSERT_NOTNULL(dfx_extend_info_holder);
    auto dfx_extend_info = reinterpret_cast<DfxExtendInfo *>(dfx_extend_info_holder.get());
    GE_ASSERT_SUCCESS(dfx_extend_info->Init(ge::PtrToPtr<void, ge::char_t>(ge::ValueToPtr(node_name_index)),
                                            cmo_context_names, cmo_context_ids));
    dfx_extend_info->SetCtxInfo(task_type, ctx_type, op_impl_mode, ctx_id_vec);
    if (!cmo_context_ids.empty()) {
      dfx_extend_info->SetCmoInfo(cmo_context_types, cmo_context_ids);
    }

    dfx_extend_infos.AddBuf(dfx_extend_info_holder.get(), total_size);
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus ExeGraphSerializer::SerializeComputeNodeInfo(const std::vector<ge::FastNode *> &all_exe_nodes,
                                                             bg::BufferPool &compute_node_infos,
                                                             bg::BufferPool &buffer_pool) const {
  std::vector<size_t> index_2_unique_index;
  index_2_unique_index.reserve(frame_.GetIndexesToNode().size());

  for (const auto &compute_node : frame_.GetIndexesToNode()) {
    size_t total_size = 0U;
    OpImplKernelRegistry::PrivateAttrList private_attrs;
    GE_ASSERT_NOTNULL(compute_node);
    GE_ASSERT_NOTNULL(compute_node->GetOpDescBarePtr());
    auto space_registry = model_desc_holder_->GetSpaceRegistry(
        static_cast<gert::OppImplVersionTag>(compute_node->GetOpDescBarePtr()->GetOppImplVersion()));
    if (space_registry != nullptr && space_registry->GetOpImpl(compute_node->GetOpDescBarePtr()->GetType().c_str()) != nullptr) {
      private_attrs = space_registry->GetOpImpl(compute_node->GetOpDescBarePtr()->GetType().c_str())->private_attrs;
    } else {
      GELOGW("Space registry is null. Private attrs is set empty.");
    }
    // todo 临时方案：解决const weight在host侧内存占用太大的问题
    // 当前实现：const的weight value在序列化compute node info时会新申请内存，因为value属性是const的ir属性会做序列化，
    // 实际上该部分weight信息已经恢复到执行结点value属性上了,这里再重复申请内存进行序列化会有内存浪费,
    // 且序列化后的value属性在执行时也不会使用，因此考虑const计算结点的ir属性做特殊处理不做序列化
    // 正式方案：应该对const的weight(tensor)进行共享内存，全局使用同一份内存避免内存浪费
    std::unique_ptr<uint8_t[]> compute_node_info;
    if (IsConstType(compute_node->GetTypePtr())) {
      compute_node_info = CreateComputeNodeInfoWithoutIrAttr(compute_node, buffer_pool, private_attrs, total_size);
    } else {
      compute_node_info = CreateComputeNodeInfo(compute_node, buffer_pool, private_attrs, total_size);
    }
    auto buf_id = compute_node_infos.AddBuf(compute_node_info.get(), total_size);
    index_2_unique_index.emplace_back(buf_id);
  }
  for (const auto node : all_exe_nodes) {
    int64_t origin_index = 0;
    const auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    if (ge::AttrUtils::GetInt(op_desc, kComputeNodeIndex, origin_index)) {
      GE_ASSERT_TRUE(origin_index >= 0);
      GE_ASSERT_TRUE(static_cast<size_t>(origin_index) < index_2_unique_index.size());
      auto unique_index = static_cast<int64_t>(index_2_unique_index[origin_index]);
      if (unique_index != origin_index) {
        GE_ASSERT_TRUE(ge::AttrUtils::SetInt(op_desc, kComputeNodeIndex, unique_index));
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus ExeGraphSerializer::SerializeKernelExtendInfo(const std::vector<ge::FastNode *> &all_exe_nodes,
                                                              bg::BufferPool &kernel_extend_infos,
                                                              bg::BufferPool &buffer_pool) const {
  for (const auto exe_node : all_exe_nodes) {
    uint8_t holder[sizeof(KernelExtendInfo)];

    const auto name_id = buffer_pool.AddStr(exe_node->GetNamePtr());
    const auto type_id = buffer_pool.AddStr(exe_node->GetTypePtr());

    auto extend_info = reinterpret_cast<KernelExtendInfo *>(holder);
    extend_info->SetKernelName(reinterpret_cast<const char *>(name_id));
    extend_info->SetKernelType(reinterpret_cast<const char *>(type_id));

    auto buf_id = kernel_extend_infos.AddBuf(holder, sizeof(holder));
    if (!ge::AttrUtils::SetInt(exe_node->GetOpDescBarePtr(), kKernelExtendIndex, static_cast<int64_t>(buf_id))) {
      GELOGE(ge::FAILED, "Failed to add KernelExtendIndex for node %s.", exe_node->GetName().c_str());
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus ExeGraphSerializer::SerializeModelDesc(bg::BufferPool &model_desc_pool) const {
  GE_ASSERT_NOTNULL(compute_graph_);
  std::vector<ge::GeTensorDesc> input_descs;
  std::vector<std::string> input_names;
  std::vector<ge::GeTensorDesc> output_descs;
  std::vector<std::string> output_names;
  std::vector<std::vector<int64_t>> aipp_all_input_dims;
  std::map<uint32_t, ge::OpDesc *> input_map;
  for (const auto node : compute_graph_->GetDirectNodePtr()) {
    GE_ASSERT_NOTNULL(node);
    const auto op_desc = node->GetOpDescBarePtr();
    GE_CHECK_NOTNULL(op_desc);
    const auto &op_type = node->GetType();
    if (ge::OpTypeUtils::IsDataNode(op_type)) {
      uint32_t data_op_index = 0U;
      GE_ASSERT_TRUE(ge::AttrUtils::GetInt(op_desc, ge::ATTR_NAME_INDEX, data_op_index));
      GE_ASSERT_TRUE(input_map.find(data_op_index) == input_map.cend(), "node %s (index %u) already exists.",
                     op_desc->GetNamePtr(), data_op_index);
      input_map[data_op_index] = op_desc;
    } else if (op_type == ge::NETOUTPUT) {
      const std::vector<std::string> &src_name = op_desc->GetSrcName();
      const std::vector<int64_t> &src_index = op_desc->GetSrcIndex();
      const auto &out_size = op_desc->GetInputsSize();
      const auto &out_node_names = model_desc_holder_->GetOutputNodeName();
      GELOGD("Get output name success, size = %zu", out_node_names.size());
      for (size_t index = 0U; index < out_size; ++index) {
        std::string output_name;
        if (out_size == out_node_names.size()) {
          const bool contains_colon = out_node_names[static_cast<size_t>(index)].find(":") != std::string::npos;
          output_name = contains_colon ? out_node_names[static_cast<size_t>(index)]
                                       : out_node_names[static_cast<size_t>(index)] + ":" +
                                             std::to_string(src_index[static_cast<size_t>(index)]);
        } else {
          GE_ASSERT(index < src_name.size());
          GE_ASSERT(index < src_index.size());
          output_name = std::string("output_") + std::to_string(index) + "_" + src_name[static_cast<size_t>(index)] +
                        "_" + std::to_string(src_index[static_cast<size_t>(index)]);
        }
        GELOGD("Serialize model output, index = %d, output name = %s", index, output_name.c_str());
        output_names.push_back(output_name);
        output_descs.emplace_back(op_desc->GetInputDesc(index));
      }
    }
  }

  for (const auto &input_pair : input_map) {
    const auto op_desc = input_pair.second;
    GE_ASSERT(op_desc->GetOutputsSize() > 0U);
    // default value
    input_descs.emplace_back(op_desc->GetOutputDesc(0U));
    input_names.emplace_back(op_desc->GetName());
    std::vector<int64_t> input_dims = op_desc->GetOutputDescPtr(0U)->GetShape().GetDims();
    aipp_all_input_dims.emplace_back(std::move(input_dims));
    // for aipp
    if (op_desc->HasAttr(ge::ATTR_NAME_INPUT_DIMS)) {
      // When static aipp is set, need to get the model input dims which processed by aipp
      (void)ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_INPUT_DIMS, aipp_all_input_dims.back());
    }
    if (op_desc->HasAttr(ge::ATTR_DYNAMIC_AIPP_INPUT_DIMS)) {
      // judge if this data is linked dynamic aipp first, multiply batch has been considered
      (void)ge::AttrUtils::GetListInt(op_desc, ge::ATTR_DYNAMIC_AIPP_INPUT_DIMS, aipp_all_input_dims.back());
      input_descs.back().SetShape(ge::GeShape(aipp_all_input_dims.back()));
      input_descs.back().SetOriginShape(ge::GeShape(aipp_all_input_dims.back()));
    }
  }

  size_t input_num = input_descs.size();
  size_t output_num = output_descs.size();
  size_t total_size = ModelDesc::CalcSize(input_num, output_num);
  std::unique_ptr<uint8_t[]> model_desc_holder = ge::MakeUnique<uint8_t[]>(total_size);

  GE_CHECK_NOTNULL(model_desc_holder);
  auto model_desc = reinterpret_cast<ModelDesc *>(model_desc_holder.get());
  model_desc->SetInputNum(input_num);
  model_desc->SetOutputNum(output_num);
  model_desc->SetReusableStreamNum(model_desc_holder_->GetModelDesc().GetReusableStreamNum());
  model_desc->SetReusableEventNum(model_desc_holder_->GetModelDesc().GetReusableEventNum());
  model_desc->SetReusableNotifyNum(model_desc_holder_->GetModelDesc().GetReusableNotifyNum());
  model_desc->SetAttachedStreamNum(model_desc_holder_->GetModelDesc().GetAttachedStreamNum());
  auto model_io_descs = model_desc->AllMutableIoDesc(input_num, output_num);
  GE_ASSERT_SUCCESS(GetIoDescFromTensor(input_descs, input_names, model_desc_pool, model_io_descs));

  // 设置aipp shape v2
  GE_ASSERT_GRAPH_SUCCESS(SetAippDims(model_io_descs, aipp_all_input_dims));

  auto model_output_descs = model_io_descs + input_num;
  GE_ASSERT_SUCCESS(GetIoDescFromTensor(output_descs, output_names, model_desc_pool, model_output_descs));

  (void)model_desc_pool.AddBuf(model_desc_holder.get(), total_size);
  return ge::GRAPH_SUCCESS;
}
ExeGraphSerializer &ExeGraphSerializer::SetComputeGraph(ge::ComputeGraphPtr compute_graph) {
  compute_graph_ = std::move(compute_graph);
  return *this;
}
ExeGraphSerializer &ExeGraphSerializer::SetModelDescHolder(ModelDescHolder *model_desc_holder) {
  model_desc_holder_ = model_desc_holder;
  return *this;
}

ExeGraphSerializer &ExeGraphSerializer::SetExecuteGraph(ge::ExecuteGraph *const execute_graph) {
  execute_graph_ = execute_graph;
  return *this;
}
}  // namespace gert
