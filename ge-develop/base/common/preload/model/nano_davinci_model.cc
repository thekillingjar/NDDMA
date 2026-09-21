/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/preload/model/nano_davinci_model.h"
#include "common/preload/model/pre_model_partition_utils.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "mmpa/mmpa_api.h"
#include "framework/common/string_util.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {
namespace {
const std::string kFwkAcllibPath = "fwkacllib/lib64";

uint32_t GetOpIndexKernel(const domi::TaskDef &task_def) {
  return task_def.kernel().context().op_index();
}

uint32_t GetOpIndexSwitchByIndex(const domi::TaskDef &task_def) {
  return task_def.label_switch_by_index().op_index();
}

uint32_t GetOpIndexLabelGoto(const domi::TaskDef &task_def) {
  return task_def.label_goto_ex().op_index();
}

uint32_t GetOpIndexLabelSet(const domi::TaskDef &task_def) {
  return task_def.label_set().op_index();
}

Status GetBinRealPath(const std::string &switch_kernel_name, std::string &bin_real_path) {
  const char_t *ascend_home_path = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, ascend_home_path);
  GE_ASSERT_NOTNULL(ascend_home_path);
  std::string bin_path = std::string(ascend_home_path) + "/" + kFwkAcllibPath + "/" + switch_kernel_name;
  bin_real_path = RealPath(bin_path.c_str());
  GE_ASSERT_TRUE(!bin_real_path.empty(), "Get binary file path is invalid, path: %s", bin_path.c_str());
  return GRAPH_SUCCESS;
}

Status GetKernelBinByName(const std::string &bin_real_path, std::unique_ptr<char_t []> &buf, uint64_t &buf_len) {
  std::ifstream file(bin_real_path.c_str(), std::ios::binary | std::ios::in);
  GE_ASSERT_TRUE(file.is_open(), "file: %s does not exist or is unaccessible.", bin_real_path.c_str());
  GE_MAKE_GUARD(file_guard, [&file]() {
    (void)file.close();
  });
  const std::streampos begin = file.tellg();
  (void)file.seekg(0, std::ios::end);
  const std::streampos end = file.tellg();
  buf_len = static_cast<uint64_t>(end - begin);
  GE_ASSERT_TRUE(static_cast<int64_t>(buf_len) > 0, "file: %s is empty.", bin_real_path.c_str());
  buf = MakeUnique<char_t []>(buf_len);
  GE_ASSERT_NOTNULL(buf);
  (void)file.seekg(0, std::ios::beg);
  (void)file.read(buf.get(), static_cast<int64_t>(buf_len));
  return GRAPH_SUCCESS;
}

Status GetKernelBin(const std::string &switch_kernel_name, std::unique_ptr<char_t []> &buf, uint64_t &buf_len) {
  // nano rts replace
  std::string bin_real_path;
  GE_ASSERT_SUCCESS(GetBinRealPath(switch_kernel_name, bin_real_path));
  GELOGI("switch bin full path: %s", bin_real_path.c_str());
  return GetKernelBinByName(bin_real_path, buf, buf_len);
}

using GetOpIndexFunc = std::function<uint32_t(const domi::TaskDef &)>;
static const std::map<ModelTaskType, GetOpIndexFunc> task_map = {
    {ModelTaskType::MODEL_TASK_KERNEL, &GetOpIndexKernel},
    {ModelTaskType::MODEL_TASK_STREAM_LABEL_SWITCH_BY_INDEX, &GetOpIndexSwitchByIndex},
    {ModelTaskType::MODEL_TASK_STREAM_LABEL_GOTO, &GetOpIndexLabelGoto},
    {ModelTaskType::MODEL_TASK_LABEL_SET, &GetOpIndexLabelSet},
};

// COND_TASKID 是64Bits SPR 低16bit表示TaskID（task在模型中的位置编号）
uint64_t NanoSwitchCondTaskId(const uint32_t reg_taskid) {
  return static_cast<uint64_t>((reg_taskid) & 0xFFFFUL);
}

// COND_TASKID 是64Bits SPR 高48bit表示TaskOffset（动态描述符在内存中的相对位置z）
uint64_t NanoSwitchCondTaskOffset(const uint32_t reg_taskid) {
  return static_cast<uint64_t>((static_cast<uint64_t>(reg_taskid) << 16UL) & 0xFFFFFFFFFFFF0000UL);
}

constexpr int64_t kMemoryGlobalType = 2;
constexpr uint32_t kIoaOffsetSize = 8U;
}  // namespace

static std::atomic<std::uint32_t> g_task_id(0U);
static const std::set<ModelTaskType> kNanoModelTaskType{
    ModelTaskType::MODEL_TASK_KERNEL, ModelTaskType::MODEL_TASK_STREAM_LABEL_SWITCH_BY_INDEX, ModelTaskType::MODEL_TASK_STREAM_LABEL_GOTO};

Status NanoDavinciModel::Init() {
  GELOGI("begin init nano davinci model.");
  GE_ASSERT_NOTNULL(ge_model_, "[Check][Param] GeModel is null.");
  ComputeGraphPtr compute_graph = ge_model_->GetGraph();

  GE_ASSERT_NOTNULL(compute_graph, "[Get][ComputeGraph] failed, ret is nullptr.");
  DoReset();
  InitRuntimeParams();
  GE_ASSERT_SUCCESS(InitTaskId());
  GE_CHK_STATUS_RET(InitNodes(compute_graph), "[Init][Nodes] failed, graph:%s.", compute_graph->GetName().c_str());
  InitKernelOffset();
  GE_ASSERT_SUCCESS(InitSwitchWeightData(compute_graph));
  GE_TIMESTAMP_START(DoTaskSink);
  GE_CHK_STATUS_RET(DoTaskSink(EngineType::kNanoEngine), "[Call][DoTaskSink] failed, model_id:%u.", model_id_);
  GE_TIMESTAMP_END(DoTaskSink, "NanoDavinciModel::DoTaskSink");

  GE_TIMESTAMP_START(DoPartitionProcess);
  GE_CHK_STATUS_RET(DoPartitionProcess(), "[Call][DoPartitionProcess] failed, model_id:%u.", model_id_);
  GE_TIMESTAMP_END(DoPartitionProcess, "NanoDavinciModel::DoPartitionProcess");
  GELOGI("success init nano davinci model.");
  return SUCCESS;
}
Status NanoDavinciModel::DoPartitionProcess() {
  GE_CHK_STATUS_RET(PreModelPartitionUtils::GetInstance().InitTaskBuildMem(task_num_),
                    "[Call][PreModelPartitionUtils][InitTaskBuildMem] failed.");
  // refresh partition data
  GE_CHK_STATUS_RET(PreModelPartitionUtils::GetInstance().PreparePartitionData(EngineType::kNanoEngine),
                    "[Call][PreModelPartitionUtils][PreparePartitionData] failed.");
  return SUCCESS;
}

Status NanoDavinciModel::InitTaskId() {
  const auto &model_task_def = ge_model_->GetModelTaskDefPtr();
  GE_ASSERT_NOTNULL(model_task_def, "model_task_def is null");

  const int32_t task_num = static_cast<int32_t>(model_task_def->task_size());
  for (int32_t i = 0; i < task_num; ++i) {
    const auto &task_def = model_task_def->task(i);
    const auto task_type = static_cast<ModelTaskType>(task_def.type());
    GELOGI("get task type[%u]", static_cast<uint32_t>(task_type));
    const auto iter = task_map.find(task_type);
    if (iter == task_map.end()) {
      GELOGD("Skip task type:%d", task_def.type());
      continue;
    }
    // 映射task index和op index的表
    const uint32_t op_index = iter->second(task_def);
    task_list_[op_index] = i;
    GELOGI("op_index[%u] set task index[%d], task type[%u]", op_index, i, static_cast<uint32_t>(task_type));
    if (kNanoModelTaskType.count(task_type) > 0U) {
      GE_ASSERT_NOTNULL(model_task_def->mutable_task(i));
      model_task_def->mutable_task(i)->set_id(g_task_id++);
    }
  }
  return SUCCESS;
}

Status NanoDavinciModel::MatchIndexToTaskIndex(const uint32_t label_idx, uint32_t &task_index) const {
  const auto &model_task_def = ge_model_->GetModelTaskDefPtr();
  GE_ASSERT_NOTNULL(model_task_def, "model_task_def is null");

  bool match_flg = false;
  const int32_t task_num = static_cast<int32_t>(model_task_def->task_size());
  for (int32_t i = 0; i < task_num; ++i) {
    const auto &task_def = model_task_def->task(i);
    const auto task_type = static_cast<ModelTaskType>(task_def.type());
    const auto iter = task_map.find(task_type);
    if (iter == task_map.end()) {
      GELOGD("Match label[%u], skip task type:%d", label_idx, task_def.type());
      continue;
    }
    if (match_flg) {
      if (kNanoModelTaskType.count(task_type) > 0U) {
        task_index = static_cast<uint32_t>(i);
        GELOGD("Match label[%u], task index:%d task_id:%u task type: %u", label_idx, i, task_def.id(),
               static_cast<uint32_t>(task_type));
        return SUCCESS;
      }
      continue;
    }
    const auto match_op_desc = GetOpByIndex(iter->second(task_def));
    if (task_type == ModelTaskType::MODEL_TASK_LABEL_SET) {
      uint32_t match_label_idx = 0U;
      GE_ASSERT_TRUE(AttrUtils::GetInt(match_op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, match_label_idx));
      if (match_label_idx == label_idx) {
        match_flg = true;
        continue;
      }
    }
  }
  GELOGE(FAILED, "MatchIndexToTaskIndex fail label_index: %u.", label_idx);
  return FAILED;
}

Status NanoDavinciModel::NanoAddSwitchKernel(const OpDescPtr &op_desc) {
  (void)op_desc;
  std::unique_ptr<char_t []> buf = nullptr;
  uint64_t buf_len = 0U;
  const string switch_kernel_name = "switch_by_index.o";
  GE_ASSERT_SUCCESS(GetKernelBin(switch_kernel_name, buf, buf_len),
      "[Call][GetKernelBin]kernel[%s] get bin fail", switch_kernel_name.c_str());
  GE_ASSERT_NOTNULL(buf);
  std::vector<char_t> data(buf.get(), PtrToPtr<void, char_t>(ValueToPtr(PtrToValue(buf.get()) + buf_len)));
  const TBEKernelPtr tbe_kernel = MakeShared<OpKernelBin>(switch_kernel_name, std::move(data));
  GE_ASSERT_NOTNULL(tbe_kernel);
  GELOGD("Nano add switch kernel: %s", switch_kernel_name.c_str());
  auto &tbe_kernel_store = ge_model_->GetTBEKernelStore();
  tbe_kernel_store.AddTBEKernel(tbe_kernel);
  return SUCCESS;
}

Status NanoDavinciModel::GetTaskKernelOffset(const std::string &kernel_name, uint32_t &offset) const {
  const auto itr = names_to_bin_offset_.find(kernel_name);
  if (itr != names_to_bin_offset_.end()) {
    offset = itr->second;
    return SUCCESS;
  }
  GELOGW("there are unsupported kernel, kernel name:%s.", kernel_name.c_str());
  return SUCCESS;
}

Status NanoDavinciModel::NanoSetWeightData(OpDescPtr &op_desc) const {
  // Get const op weight pointer
  ge::GeTensorPtr weight = nullptr;
  (void)ge::AttrUtils::MutableTensor(op_desc, ATTR_NAME_WEIGHTS, weight);
  GE_ASSERT_NOTNULL(weight);
  // Get const op weight data
  auto weight_data = weight->MutableData();
  // copy const op weight data to buffer
  GELOGI("Move to buffer, name: %s size: %zu", op_desc->GetName().c_str(), weight_data.size());
  GE_ASSERT_SUCCESS(PreModelPartitionUtils::GetInstance().SaveNanoModelPartitionData(
      static_cast<uint8_t>(ModelPartitionType::WEIGHTS_DATA), PtrToPtr<const uint8_t, const void>(weight_data.data()),
      static_cast<uint32_t>(weight_data.size())));

  return SUCCESS;
}

Status NanoDavinciModel::NanoAddSwitchConstNode(const std::vector<uint64_t> &cond_task_id_list,
                                                const ge::NodePtr &sw_node, size_t &weight_offset,
                                                ComputeGraphPtr &graph) const {
  OpDescPtr const_op_desc = MakeShared<OpDesc>(sw_node->GetName() + "_switch_taskid", CONSTANT);
  GeTensorDesc data_desc(GeShape(), FORMAT_NCHW, DT_UINT64);
  const GeTensorPtr const_value =
      MakeShared<GeTensor>(data_desc, PtrToPtr<const uint64_t, const uint8_t>(cond_task_id_list.data()),
                           cond_task_id_list.size() * sizeof(uint64_t));
  GE_ASSERT_TRUE(AttrUtils::SetTensor(const_op_desc, ATTR_NAME_WEIGHTS, const_value));
  GE_ASSERT_SUCCESS(const_op_desc->AddOutputDesc(data_desc));

  GELOGI("Create Const op: %s.", const_op_desc->GetName().c_str());
  const NodePtr const_node = graph->AddNode(const_op_desc);
  GE_ASSERT_NOTNULL(const_node);
  auto sw_op_desc = sw_node->GetOpDesc();
  const uint32_t index = sw_node->GetAllInDataAnchorsSize();
  GE_ASSERT_SUCCESS(sw_node->AddLinkFrom(index, const_node));
  GE_ASSERT_SUCCESS(NanoSetWeightData(const_op_desc), "[Call][NanoSetWeightData]set weight data fail");
  // add weight offset & input offset
  const std::vector<GeTensorPtr> weights = ge::OpDescUtils::MutableWeights(const_node);
  GE_ASSERT_TRUE(!(weights.empty()), "Add const node, weights size of node %s is empty", const_node->GetName().c_str());

  const GeTensorPtr weight = weights[0];
  GE_ASSERT_NOTNULL(weight);
  GeTensorDesc &tensor_desc = weight->MutableTensorDesc();
  const size_t output_size = weight->GetData().size();
  const size_t current_offset = weight_offset;
  ge::TensorUtils::SetDataOffset(tensor_desc, static_cast<int64_t>(current_offset));
  GELOGI("sw_node: %s, input offset: %ld", sw_node->GetName().c_str(), current_offset);
  weight_offset += output_size;
  std::vector<int64_t> v_input_offset = sw_op_desc->GetInputOffset();
  std::vector<int64_t> v_output_offset;
  v_input_offset.push_back(static_cast<int64_t>(current_offset));
  v_output_offset.push_back(static_cast<int64_t>(current_offset));
  const_op_desc->SetOutputOffset(v_output_offset);
  sw_op_desc->SetInputOffset(v_input_offset);
  return SUCCESS;
}

Status NanoDavinciModel::NanoSwitchWeightDataInit(ComputeGraphPtr &compute_graph,
                                                  const ComputeGraph::Vistor<NodePtr> &all_nodes) {
  const auto &model_task_def = ge_model_->GetModelTaskDefPtr();
  size_t weight_offset = ge_model_->GetWeightSize();
  for (auto &node : all_nodes) {
    const auto &op_desc = node->GetOpDesc();
    vector<uint64_t> cond_task_id_list;
    if (node->GetType() == LABELSWITCHBYINDEX) {
      vector<uint32_t> task_index_list;
      GE_ASSERT_TRUE(AttrUtils::GetListInt(op_desc, ATTR_NAME_LABEL_SWITCH_LIST, task_index_list));
      for (size_t i = 0UL; i < task_index_list.size(); ++i) {
        const int32_t task_index = static_cast<int32_t>(task_index_list.at(i));
        const auto &task_def = model_task_def->task(task_index);
        uint64_t cond_task_id = 0UL;
        cond_task_id |= NanoSwitchCondTaskId(static_cast<uint32_t>(task_def.id()));
        // 获取offset
        const domi::KernelDef &kernel_def = task_def.kernel();
        uint32_t task_offset = 0U;
        GE_ASSERT_SUCCESS(GetTaskKernelOffset(kernel_def.kernel_name(), task_offset));
        cond_task_id |= NanoSwitchCondTaskOffset(task_offset);
        cond_task_id_list.push_back(cond_task_id);
      }
      GE_ASSERT_SUCCESS(NanoAddSwitchConstNode(cond_task_id_list, node, weight_offset, compute_graph));
    } else if (node->GetType() == LABELGOTOEX) {
      uint32_t task_index = 0U;
      uint64_t cond_task_id = 0UL;
      GE_ASSERT_TRUE(AttrUtils::GetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, task_index));
      const auto &task_def = model_task_def->task(static_cast<int32_t>(task_index));
      cond_task_id |= NanoSwitchCondTaskId(static_cast<uint32_t>(task_def.id()));
      // 获取offset
      const domi::KernelDef &kernel_def = task_def.kernel();
      uint32_t task_offset = 0U;
      GE_ASSERT_SUCCESS(GetTaskKernelOffset(kernel_def.kernel_name(), task_offset));
      cond_task_id |= NanoSwitchCondTaskOffset(task_offset);
      cond_task_id_list.push_back(cond_task_id);
      GE_ASSERT_SUCCESS(NanoAddSwitchConstNode(cond_task_id_list, node, weight_offset, compute_graph));
    } else {
      GELOGI("node type %s skip", node->GetType().c_str());
    }
  }
  return SUCCESS;
}

Status NanoDavinciModel::InitSwitchWeightData(ComputeGraphPtr &compute_graph) {
  // dirct node
  GE_ASSERT_SUCCESS(NanoSwitchWeightDataInit(compute_graph, compute_graph->GetDirectNode()));
  // sub node
  for (auto &subgraph : compute_graph->GetAllSubgraphs()) {
    GE_ASSERT_SUCCESS(NanoSwitchWeightDataInit(subgraph, subgraph->GetAllNodes()));
  }
  GE_DUMP(compute_graph, "AfterNanoInitSwitchWeightData");
  return SUCCESS;
}

Status NanoDavinciModel::InitSwitchNodes(const ComputeGraphPtr &compute_graph) {
  uint32_t nano_weight_data_size = 0U;
  const auto &nodes = compute_graph->GetAllNodes();
  for (size_t i = 0UL; i < nodes.size(); ++i) {
    const auto &node = nodes.at(i);
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (node->GetType() == LABELSWITCHBYINDEX) {
      vector<uint32_t> label_idx_list;
      GE_ASSERT_TRUE(AttrUtils::GetListInt(op_desc, ATTR_NAME_LABEL_SWITCH_LIST, label_idx_list));
      for (size_t j = 0UL; j < label_idx_list.size(); ++j) {
        uint32_t task_index = 0U;
        GE_ASSERT_SUCCESS(MatchIndexToTaskIndex(label_idx_list[j], task_index));
        label_idx_list[j] = task_index;
        nano_weight_data_size += static_cast<uint32_t>(sizeof(uint64_t));
      }
      GE_ASSERT_TRUE(AttrUtils::SetListInt(op_desc, ATTR_NAME_LABEL_SWITCH_LIST, label_idx_list));
      GE_ASSERT_SUCCESS(NanoAddSwitchKernel(op_desc));
    } else if (node->GetType() == LABELGOTOEX) {
      uint32_t index = 0U;
      GE_ASSERT_TRUE(AttrUtils::GetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, index));
      uint32_t task_index = 0U;
      GE_ASSERT_SUCCESS(MatchIndexToTaskIndex(index, task_index));
      nano_weight_data_size += static_cast<uint32_t>(sizeof(uint64_t));
      index = task_index;
      GE_ASSERT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, static_cast<int64_t>(index)));
      GE_ASSERT_SUCCESS(NanoAddSwitchKernel(op_desc));
    } else {
      GELOGI("node type %s skip", node->GetType().c_str());
    }
  }
  GE_ASSERT_SUCCESS(PreModelPartitionUtils::GetInstance().GenModelPartitionBuf(
                    static_cast<uint8_t>(ModelPartitionType::WEIGHTS_DATA), nano_weight_data_size),
                    "[Call][PreModelPartitionUtils][GenModelPartitionBuf] failed.");

  return SUCCESS;
}

Status NanoDavinciModel::SetAnchorsOffset(const ge::NodePtr &node, const uint32_t index, const bool is_input,
                                          const uint32_t offset) const {
  auto op_desc = node->GetOpDesc();
  auto logic_offset_list = is_input ? op_desc->GetInputOffset() : op_desc->GetOutputOffset();
  GE_ASSERT_TRUE(index < logic_offset_list.size(), "node[%s] %s index[%u] overflow", node->GetName().c_str(),
                 is_input ? "input" : "output", index);
  GELOGI("node[%s] %s no.%u replace offset [%u] to [%u]", node->GetName().c_str(), is_input ? "input" : "output", index,
         logic_offset_list[static_cast<size_t>(index)], offset);
  logic_offset_list[static_cast<size_t>(index)] = static_cast<int64_t>(offset);
  if (is_input) {
    op_desc->SetInputOffset(logic_offset_list);
  } else {
    op_desc->SetOutputOffset(logic_offset_list);
  }
  return SUCCESS;
}

Status NanoDavinciModel::SetPeerInDataOffset(const OutDataAnchorPtr out_anchor, const uint32_t offset) const {
  GE_CHECK_NOTNULL(out_anchor);
  for (const auto &in_anchor : out_anchor->GetPeerInDataAnchorsPtr()) {
    GE_IF_BOOL_EXEC(in_anchor == nullptr, continue);
    auto owner_node = in_anchor->GetOwnerNode();
    const auto in_index = in_anchor->GetIdx();
    GELOGI("node[%s] set input index[%u] offset [%u]", owner_node->GetName().c_str(), in_index, offset);
    auto in_op_desc = owner_node->GetOpDesc();
    const GeTensorDescPtr in_tensor_desc = in_op_desc->MutableInputDesc(static_cast<uint32_t>(in_index));
    GE_IF_BOOL_EXEC(in_tensor_desc == nullptr, continue);
    (void)ge::AttrUtils::SetInt(in_tensor_desc, ATTR_NAME_TENSOR_MEMORY_SCOPE, kMemoryGlobalType);
    GE_ASSERT_SUCCESS(SetAnchorsOffset(owner_node, static_cast<uint32_t>(in_index), true, offset));
  }
  return SUCCESS;
}

Status NanoDavinciModel::UpdateFifoWindowCacheRefOffset(const ge::NodePtr &node) const {
  uint32_t offset;
  auto op_desc = node->GetOpDesc();
  auto input_offset_list = op_desc->GetInputOffset();
  auto output_offset_list = op_desc->GetOutputOffset();
  bool no_need_update = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NOTASK, no_need_update);
  // no_task节点（split、concact）非FIFO相关的节点，无需更新
  if (!no_need_update) {
    for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
      int32_t reuse_in_index = -1;
      const bool reuse_input = GraphUtils::IsRefFromInput(out_anchor, reuse_in_index);
      // FIFO算子在fm内存申请阶段不划分内存，需要更新FIFO算子输入输出相关的ref节点
      if (reuse_input && reuse_in_index >= 0 && static_cast<size_t>(reuse_in_index) < input_offset_list.size()) {
        offset = static_cast<uint32_t>(input_offset_list[static_cast<size_t>(reuse_in_index)]);
        const auto out_index = out_anchor->GetIdx();
        GE_ASSERT_TRUE(static_cast<size_t>(out_index) < output_offset_list.size(),
                       "node[%s] output index[%u] overflow",
                       node->GetName().c_str(), out_index);
        // 非FIFO相关的ref节点，无需更新
        GE_IF_BOOL_EXEC(offset == static_cast<uint32_t>(output_offset_list[static_cast<size_t>(out_index)]),
                        GELOGD("ref node[%s] is not fifo node, no need to update offset", node->GetName().c_str());
                        continue);
        // FIFO算子为ref节点，获取输出对应ref的input index
        GELOGI("ref node[%s] set output index[%u] offset [%u]", node->GetName().c_str(), out_index, offset);
        GE_ASSERT_SUCCESS(SetAnchorsOffset(node, static_cast<uint32_t>(out_index), false, offset));
        // 同时设置该ref节点输出对端的input的offset
        GE_ASSERT_SUCCESS(SetPeerInDataOffset(out_anchor, offset));
      }
    }
  }
  return SUCCESS;
}

Status NanoDavinciModel::InitFifoWindowCacheOffset(const ge::NodePtr &node) {
  uint32_t offset;
  const auto op_desc = node->GetOpDesc();
  const size_t outputs_size = op_desc->GetOutputsSize();
  for (uint32_t i = 0U; i < static_cast<uint32_t>(outputs_size); i++) {
    const auto output_tensor_desc = op_desc->MutableOutputDesc(i);
    GE_IF_BOOL_EXEC(output_tensor_desc == nullptr,
                    GELOGW("op[%s] null output_tensor_desc, index[%u]", op_desc->GetName().c_str(), i);
                    continue);
    int32_t tensor_type = 0;
    const bool ret = ge::AttrUtils::GetInt(output_tensor_desc, ATTR_NAME_TENSOR_MEMORY_SCOPE, tensor_type);
    if (ret && tensor_type == kMemoryGlobalType) {
      const auto out_anchor = node->GetOutDataAnchor(static_cast<int32_t>(i));
      int32_t reuse_in_index = -1;
      const bool reuse_input = GraphUtils::IsRefFromInput(out_anchor, reuse_in_index);
      // 如果输出为复用节点，则不需要再另外分配内存划分offset
      // 复用场景节点的offset的设置在UpdateFifoWindowCacheRefOffset中统一设置
      GE_IF_BOOL_EXEC(reuse_input, GELOGI("node[%s] output index[%u] is reuse input[%d],continue",
                      node->GetName().c_str(), i, reuse_in_index); continue);
      int64_t tensor_size = 0;
      GE_ASSERT_SUCCESS(TensorUtils::GetSize(*output_tensor_desc, tensor_size));
      offset = (search_id_++) * kIoaOffsetSize;
      PreModelPartitionUtils::GetInstance().AddGlobalDataTensorSize(static_cast<uint64_t>(tensor_size));
      GELOGI("node[%s] set offset [%u] with tensor size [%ld]", node->GetName().c_str(), offset, tensor_size);
      // 继续在ioa offset后设置index对应输出节点的offset
      GE_ASSERT_SUCCESS(SetAnchorsOffset(node, static_cast<uint32_t>(i), false, offset));
      // 同时设置该index输出节点对端的input的offset
      GE_ASSERT_SUCCESS(SetPeerInDataOffset(out_anchor, offset));
    }
  }
  GE_ASSERT_SUCCESS(UpdateFifoWindowCacheRefOffset(node), "node[%s] update ref node error", node->GetName().c_str());

  return SUCCESS;
}

Status NanoDavinciModel::InitNodes(const ComputeGraphPtr &compute_graph) {
  const auto io_table = PreModelPartitionUtils::GetInstance().GetZeroCopyTable();
  search_id_ = static_cast<uint32_t>(io_table.size());
  GELOGI("ioa_table size: %u", search_id_);
  const auto &nodes = compute_graph->GetAllNodes();
  for (size_t i = 0UL; i < nodes.size(); ++i) {
    const auto &node = nodes.at(i);
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    GELOGI("add op[%d] name[%s] to list", op_desc->GetId(), op_desc->GetName().c_str());
    op_list_[op_desc->GetId()] = op_desc;
    GE_ASSERT_SUCCESS(InitFifoWindowCacheOffset(node));
  }

  GE_ASSERT_SUCCESS(InitSwitchNodes(compute_graph));
  auto &tbe_kernel_store = ge_model_->GetTBEKernelStore();
  GE_ASSERT_TRUE(tbe_kernel_store.PreBuild());

  return SUCCESS;
}
}  // namespace ge