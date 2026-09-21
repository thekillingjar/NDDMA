/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/ge_model_builder.h"
#include "faker/aicpu_taskdef_faker.h"
#include "faker/aicore_taskdef_faker.h"
#include "faker/dvpp_taskdef_faker.h"
#include "utils/tensor_utils.h"
#include "common/debug/log.h"
#include "graph/debug/ge_attr_define.h"
#include "common/host_resource_center/host_resource_center.h"
#include "common/env_path.h"
#include "common/ge_common/scope_guard.h"

namespace gert {
namespace {
std::vector<char> CreateStubBin() {
  auto ascend_install_path = ge::EnvPath().GetAscendInstallPath();
  std::string op_bin_path = ascend_install_path + "/fwkacllib/lib64/switch_by_index.o";
  std::vector<char> buf;
  std::ifstream file(op_bin_path.c_str(), std::ios::binary | std::ios::in);
  if (!file.is_open()) {
    std::cout << "file:" << op_bin_path << "does not exist or is unaccessible." << std::endl;
    return buf;
  }
  GE_MAKE_GUARD(file_guard, [&file]() {
    (void)file.close();
  });
  const std::streampos begin = file.tellg();
  (void)file.seekg(0, std::ios::end);
  const std::streampos end = file.tellg();
  auto buf_len = static_cast<uint32_t>(end - begin);
  buf.resize(buf_len + 1);
  (void)file.seekg(0, std::ios::beg);
  (void)file.read(const_cast<char *>(buf.data()), buf_len);
  std::cout << std::string(buf.data()) << std::endl;
  return buf;
}
}

GeModelBuilder::GeModelBuilder(ge::ComputeGraphPtr compute_graph) : compute_graph_(compute_graph),
                                                                    use_default_task_(true) {
  ge_model = std::make_shared<ge::GeModel>();
  ge_model->SetName(compute_graph->GetName());
  model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  weight_buffer_.reserve(1024);
}

GeModelBuilder &GeModelBuilder::ConstWeight(const std::string &node_name) {
  auto const_node = compute_graph_->FindNode(node_name);
  if (const_node->GetType() != ge::CONSTANT) {
    return *this;
  }
  ge::ConstGeTensorPtr weight;
  if (!ge::AttrUtils::GetTensor(const_node->GetOpDesc(), "value", weight)) {
    GELOGE(ge::INTERNAL_ERROR, "weight is empty. node  = %s", const_node->GetName().c_str());
    return *this;
  }
  ge::GeTensorDesc &tensor_desc = const_cast<ge::GeTensorDesc &>(weight->GetTensorDesc());
  ge::TensorUtils::SetSize(tensor_desc, weight->GetData().GetSize());
  ge::TensorUtils::SetDataOffset(tensor_desc, weight_buffer_.size());
  weight_buffer_.insert(weight_buffer_.end(), weight->GetData().GetData(),
                        weight->GetData().GetData() + weight->GetData().GetSize());
  return *this;
}

void GeModelBuilder::SetTaskDef(TaskDefFaker &task_def, int64_t node_id) {
  auto tasks = task_def.CreateTaskDef(node_id);
  for (auto &task : tasks) {
    task_indexes_to_node_id_[model_task_def->task_size()] = node_id;
    auto task_ptr = model_task_def->add_task();
    *task_ptr = task;
  }
}
void GeModelBuilder::SetTaskDefs() {
  for (const auto &faker : append_task_def_fakers_) {
    SetTaskDef(*faker, -1);
  }
  for (const auto &node : compute_graph_->GetDirectNode()) {
    auto iter = node_name_or_types_to_task_def_faker_.find(node->GetName());
    if (iter != node_name_or_types_to_task_def_faker_.end()) {
      SetTaskDef(*iter->second, node->GetOpDesc()->GetId());
      continue;
    }

    iter = node_name_or_types_to_task_def_faker_.find(node->GetType());
    if (iter != node_name_or_types_to_task_def_faker_.end()) {
      SetTaskDef(*iter->second, node->GetOpDesc()->GetId());
      continue;
    }

    if (all_task_def_faker_ != nullptr) {
      SetTaskDef(*all_task_def_faker_, node->GetOpDesc()->GetId());
      continue;
    }
  }
}
void GeModelBuilder::FakeTbeBinToNodes() {
  const char *compile_info_json =
      "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", \"ub_size\": 126464, "
      "\"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}";
  for (const auto &node : compute_graph_->GetDirectNode()) {
    auto iter = node_names_or_types_to_tbe_bin_fake_cfg_.find(node->GetName());
    if (iter == node_names_or_types_to_tbe_bin_fake_cfg_.end()) {
      iter = node_names_or_types_to_tbe_bin_fake_cfg_.find(node->GetType());
      if (iter == node_names_or_types_to_tbe_bin_fake_cfg_.end()) {
        std::cout << "Cannot find node: " << node->GetName() << ", type: " << node->GetType() << std::endl;
        continue;
      }
    }

    const auto &cfg = iter->second;
    ge::AttrUtils::SetStr(node->GetOpDesc(), "compile_info_json", compile_info_json);
    ge::AttrUtils::SetInt(node->GetOpDesc(), "op_para_size", 2048);
    auto name = node->GetName() + "_faked_kernel";
    auto bin = std::make_shared<ge::OpKernelBin>(name, CreateStubBin());
    node->GetOpDesc()->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, bin);
    ge::AttrUtils::SetStr(node->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
    ge::AttrUtils::SetStr(node->GetOpDesc(), ge::TVM_ATTR_NAME_METADATA, "FakeMeta");
    ge::AttrUtils::SetStr(node->GetOpDesc(), node->GetName() + "_kernelname", name);
    ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", name);

    if (cfg.need_atomic) {
      ge::AttrUtils::SetStr(node->GetOpDesc(), "_atomic_compile_info_json", "{}");
      ge::AttrUtils::SetInt(node->GetOpDesc(), "atomic_op_para_size", 2048);

      name = node->GetName() + "_faked_atomic_kernel";
      auto atomic_bin = std::make_shared<ge::OpKernelBin>(name, CreateStubBin());
      node->GetOpDesc()->SetExtAttr(ge::EXT_ATTR_ATOMIC_TBE_KERNEL, atomic_bin);
      ge::AttrUtils::SetStr(node->GetOpDesc(), ge::ATOMIC_ATTR_TVM_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
      ge::AttrUtils::SetStr(node->GetOpDesc(), ge::ATOMIC_ATTR_TVM_METADATA, "FakeAtomicMeta");
      ge::AttrUtils::SetStr(node->GetOpDesc(), node->GetName() + "_atomic_kernelname", name);
      ge::AttrUtils::SetStr(node->GetOpDesc(), ge::ATOMIC_ATTR_TBE_KERNEL_NAME, name);

    }
  }
}

GeModelBuilder &GeModelBuilder::AddTaskDef(const std::string &node_type_or_name, const TaskDefFaker &task_def) {
  use_default_task_ = false;
  node_name_or_types_to_task_def_faker_[node_type_or_name] = task_def.Clone();
  return *this;
}

GeModelBuilder &GeModelBuilder::AddTaskDefForAll(const TaskDefFaker &task_def) {
  use_default_task_ = false;
  all_task_def_faker_ = task_def.Clone();
  return *this;
}

GeModelBuilder &GeModelBuilder::AddWeight() {
  weight_buffer_ = std::vector<uint8_t>(1024, 100);
  return *this;
}

void GeModelBuilder::AddTbeKernelStore() {
  // todo 归一到 AddTbeKernelStoreFromNodes
  bool without_kernel_store = false;
  ge::AttrUtils::GetBool(compute_graph_, "without_kernel_store", without_kernel_store);
  ge::TBEKernelStore tbe_kernel_store;
  if (!without_kernel_store) {
    std::vector<char> kernel_bin(64, '\0');
    ge::TBEKernelPtr kernel_handle = ge::MakeShared<ge::OpKernelBin>("dumy_kernel_name", std::move(kernel_bin));
    tbe_kernel_store.AddTBEKernel(kernel_handle);
  }
  for (const auto &node : compute_graph_->GetDirectNode()) {
    auto bin = node->GetOpDesc()->TryGetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, ge::TBEKernelPtr());
    if (bin != nullptr) {
      tbe_kernel_store.AddTBEKernel(bin);
    }
    bin = node->GetOpDesc()->TryGetExtAttr(ge::EXT_ATTR_ATOMIC_TBE_KERNEL, ge::TBEKernelPtr());
    if (bin != nullptr) {
      tbe_kernel_store.AddTBEKernel(bin);
    }
  }
  tbe_kernel_store.Build();
  ge_model->SetTBEKernelStore(tbe_kernel_store);
}


ge::TBEKernelStore GeModelBuilder::BuildKernelStoreFromNodes() const {
  ge::TBEKernelStore tbe_kernel_store;
  for (const auto &node : compute_graph_->GetDirectNode()) {
    auto bin = node->GetOpDesc()->TryGetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, ge::TBEKernelPtr());
    if (bin != nullptr) {
      tbe_kernel_store.AddTBEKernel(bin);
    }

    bin = node->GetOpDesc()->TryGetExtAttr(ge::EXT_ATTR_ATOMIC_TBE_KERNEL, ge::TBEKernelPtr());
    if (bin != nullptr) {
      tbe_kernel_store.AddTBEKernel(bin);
    }
  }
  tbe_kernel_store.Build();
  return tbe_kernel_store;
}

ge::CustAICPUKernelStore GeModelBuilder::BuildCustAicpuKernelStoreFromNodes() const {
  ge::CustAICPUKernelStore cust_aicpu_kernel_store;
  for (const auto &node : compute_graph_->GetDirectNode()) {
    auto bin = node->GetOpDesc()->TryGetExtAttr(ge::OP_EXTATTR_CUSTAICPU_KERNEL, ge::OpKernelBinPtr());
    if (bin != nullptr) {
      cust_aicpu_kernel_store.AddCustAICPUKernel(bin);
    }
  }
  cust_aicpu_kernel_store.Build();
  return cust_aicpu_kernel_store;
}

void GeModelBuilder::AddTbeKernelStoreFromNodes() {
  FakeTbeBinToNodes();
  ge_model->SetTBEKernelStore(BuildKernelStoreFromNodes());
  ge_model->SetCustAICPUKernelStore(BuildCustAicpuKernelStoreFromNodes());
}

void GeModelBuilder::BuildCommon() {
  SetTaskDefs();
  ge_model->SetGraph(compute_graph_);
}

void GeModelBuilder::AddCustAicpuKernelStore(const std::string &name) {
  ge::CustAICPUKernelStore cust_aicpu_kernel_store;
  std::vector<char> kernel_bin(64, '\0');
  ge::OpKernelBinPtr cust_aicpu_kernel = std::make_shared<ge::OpKernelBin>(name, std::move(kernel_bin));
  cust_aicpu_kernel_store.AddCustAICPUKernel(cust_aicpu_kernel);
  ge_model->SetCustAICPUKernelStore(cust_aicpu_kernel_store);
}

void GeModelBuilder::SetAttrs() {
  ge::AttrUtils::SetInt(ge_model, ge::ATTR_MODEL_MEMORY_SIZE, 10240);
  ge::AttrUtils::SetInt(ge_model, ge::ATTR_MODEL_STREAM_NUM, 1);
  ge::AttrUtils::SetInt(ge_model, ge::ATTR_MODEL_EVENT_NUM, 1);
  ge::AttrUtils::SetInt(ge_model, ge::ATTR_MODEL_LABEL_NUM, 0);
  ge::AttrUtils::SetInt(ge_model, ge::MODEL_ATTR_TASK_GEN_BASE_ADDR, 0x100000000);
  ge::AttrUtils::SetInt(ge_model, ge::ATTR_MODEL_TASK_GEN_VAR_ADDR, 137438953472);
  ge::AttrUtils::SetInt(ge_model, ge::MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0x1000000000);
  ge::AttrUtils::SetInt(ge_model, ge::ATTR_MODEL_VAR_SIZE, 5120);

  ge::AttrUtils::SetInt(ge_model, ge::ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 5120);
  ge::AttrUtils::SetInt(ge_model, ge::ATTR_MODEL_SESSION_SCOPE_MEMORY_SIZE, 512);
  if (set_sub_mem_infos_) {
    std::vector<std::vector<int64_t>> sub_memory_infos;
    sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 2560});     // sub fm0
    sub_memory_infos.push_back({RT_MEMORY_HBM, 2560, 2560});  // sub fm1
    sub_memory_infos.push_back({RT_MEMORY_HBM, 5120, 5120});  // zero copy io
    (void)ge::AttrUtils::SetListListInt(ge_model, ge::ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);
  }
}

ge::GeModelPtr GeModelBuilder::Build() {
  AddTbeKernelStore();
  ge_model->SetCustAICPUKernelStore(BuildCustAicpuKernelStoreFromNodes());
  BuildCommon();
  SetAttrs();
  AddDefaultTasks();
  AddDefaultWeights();
  return ge_model;
}
GeModelBuilder::ModelResult GeModelBuilder::BuildDetail() {
  auto model = Build();
  return {model, task_indexes_to_node_id_};
}

ge::GeRootModelPtr GeModelBuilder::BuildGeRootModel() {
  compute_graph_->SetGraphUnknownFlag(true);
  AddTbeKernelStoreFromNodes();
  BuildCommon();
  AddDefaultTasks();
  AddDefaultWeights();
  GELOGD("Model has not been loaded!");
  std::shared_ptr<ge::GeRootModel> out_model = ge::MakeShared<ge::GeRootModel>();
  if (out_model == nullptr) {
    return nullptr;
  }

  ge::AttrUtils::SetStr(ge_model, ge::ATTR_MODEL_OPP_VERSION, "3.20.T100.0.B356");
  ge::AttrUtils::SetStr(ge_model, ge::ATTR_MODEL_HOST_ENV_OS, "linux");
  ge::AttrUtils::SetStr(ge_model, ge::ATTR_MODEL_HOST_ENV_CPU, "x86_64");
  ge::AttrUtils::SetInt(ge_model, ge::ATTR_MODEL_STREAM_NUM, root_model_stream_num_);
  ge::AttrUtils::SetInt(ge_model, ge::ATTR_MODEL_EVENT_NUM, root_model_event_num_);
  ge::AttrUtils::SetInt(ge_model, ge::ATTR_MODEL_VAR_SIZE, 512 *1024 * 1024);
  ge::AttrUtils::SetInt(ge_model, ge::ATTR_MODEL_TASK_GEN_VAR_ADDR, 137438953472);
  const auto root_graph = ge_model->GetGraph();
  if (root_graph != nullptr) {
    out_model->Initialize(root_graph);
    out_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);
  }

  for (auto subgraph : root_graph->GetAllSubgraphs()) {
    // exclude functional subgrapg in known graph
    const auto &parent_graph = subgraph->GetParentGraph();
    if ((parent_graph != nullptr) && (parent_graph != root_graph) && (!parent_graph->GetGraphUnknownFlag())) {
      continue;
    }
    // exclude ffts plus subgraph in known subgraph
    const auto &fuctional_node = subgraph->GetParentNode();
    if ((fuctional_node != nullptr) && (fuctional_node->GetOpDesc()->HasAttr(ge::ATTR_NAME_FFTS_PLUS_SUB_GRAPH))) {
      continue;
    }
    GeModelBuilder builder(subgraph);
    if (all_task_def_faker_ != nullptr) {
      builder.AddTaskDefForAll(*(all_task_def_faker_.get()));
    }
    out_model->SetSubgraphInstanceNameToModel(subgraph->GetName(), builder.Build());
  }
  return out_model;
}
GeModelBuilder &GeModelBuilder::FakeTbeBin(const std::vector<TbeConfig> &configs) {
  for (const auto &cfg : configs) {
    node_names_or_types_to_tbe_bin_fake_cfg_[cfg.node_name_or_type] = cfg;
  }
  return *this;
}

GeModelBuilder &GeModelBuilder::AddDefaultTasks() {
  if (!use_default_task_) {
    return *this;
  }
  for (auto node : compute_graph_->GetDirectNode()) {
    std::shared_ptr<TaskDefFaker> faker;
    std::string kernel_lib = node->GetOpDesc()->GetOpKernelLibName();
    if (kernel_lib == ge::kEngineNameDvpp) {
      faker = std::make_shared<DvppTaskDefFaker>();
    } else if (kernel_lib == ge::kEngineNameAiCore) {
      faker = std::make_shared<AiCoreTaskDefFaker>();
      FakeTbeBin({node->GetType().c_str()});
    } else if (kernel_lib == ge::kEngineNameAiCpu) {
      faker = std::make_shared<AiCpuCCTaskDefFaker>();
    } else if (kernel_lib == ge::kEngineNameAiCpuTf) {
      faker = std::make_shared<AiCpuTfTaskDefFaker>();
    } else {
      GELOGI("node[%s] appoint invalid engine type[%s], skip fake task.",
             node->GetName().c_str(), kernel_lib.c_str());
      continue;
    }
    SetTaskDef(*faker, node->GetOpDesc()->GetId());
  }
  return *this;
}

GeModelBuilder &GeModelBuilder::AddDefaultWeights() {
  for (auto const_node : compute_graph_->GetDirectNode()) {
    if (const_node->GetType() != "Const") {
      continue;
    }
    ge::GeTensorPtr weight;
    if (!ge::AttrUtils::MutableTensor(const_node->GetOpDesc(), "value", weight)) {
      GELOGE(ge::INTERNAL_ERROR, "weight is empty. node  = %s", const_node->GetName().c_str());
      continue;
    }
    auto op_desc = const_node->GetOpDesc();
    if (op_desc == nullptr) {
      GELOGE(ge::INTERNAL_ERROR, "op_desc is empty. node  = %s", const_node->GetName().c_str());
      continue;
    }
    ge::GeTensorDesc &tensor_desc = weight->MutableTensorDesc();
    auto tensor_data = weight->GetData();
    ge::TensorUtils::SetSize(tensor_desc, tensor_data.GetSize());
    ge::TensorUtils::SetWeightSize(tensor_desc, tensor_data.GetSize());
    ge::TensorUtils::SetDataOffset(tensor_desc, weight_buffer_.size());
    ge::TensorUtils::SetSize(*op_desc->MutableOutputDesc(0U), tensor_data.GetSize());
    weight_buffer_.insert(weight_buffer_.end(), tensor_data.GetData(),
                          tensor_data.GetData() + tensor_data.GetSize());
  }
  ge_model->SetWeight(ge::Buffer::CopyFrom((uint8_t *)weight_buffer_.data(), weight_buffer_.size()));
  GELOGI("model[%s], set weight size[%zu]", ge_model->GetName().c_str(), weight_buffer_.size());
  return *this;
}
GeModelBuilder &GeModelBuilder::AppendTaskDef(const TaskDefFaker &task_def) {
  append_task_def_fakers_.emplace_back(task_def.Clone());
  return *this;
}
}  // namespace gert
