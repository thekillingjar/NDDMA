/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/model/ge_root_model.h"
#include "common/op_so_store/op_so_store_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/op_tiling/op_tiling_rt2.h"
#include "common/checker.h"
#include "graph/ge_context.h"
#include "common/host_resource_center/host_resource_serializer.h"
#include "graph/manager/graph_var_manager.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {
namespace {
/*
 * P2P类型的feature map内存，目前只给hccl算子使用，而大部分hccl算子都是fixed地址的，所以这里简化处理，将全部p2p内存作为fixed内存，
 * 并将子图中最大的长度作为整图fix内存大小
 */
Status GetP2pFixedFeatureMemorySize(const GeRootModel *ge_root_model, size_t &p2p_mem_size) {
  size_t p2p_max_size = 0U;
  for (const auto &name_to_model : ge_root_model->GetSubgraphInstanceNameToModel()) {
    const auto &ge_model_temp = name_to_model.second;
    size_t p2p_size = 0U;
    (void)AttrUtils::GetInt(ge_model_temp, ATTR_MODEL_P2P_MEMORY_SIZE, p2p_size);
    if (p2p_size > p2p_max_size) {
      p2p_max_size = p2p_size;
    }
    GELOGI("[IMAS]model_name:%s p2p fixed_feature_memory size:%zu, p2p max size:%zu.",
           ge_model_temp->GetName().c_str(), p2p_size, p2p_max_size);
  }
  p2p_mem_size = p2p_max_size;
  return SUCCESS;
}
}
Status GeRootModel::Initialize(const ComputeGraphPtr &root_graph) {
  GE_ASSERT_NOTNULL(root_graph);
  SetRootGraph(root_graph);
  if (model_name_.empty()) {
    model_name_ = root_graph_->GetName();
  }
  GE_ASSERT_NOTNULL(host_resource_center_);
  GE_ASSERT_SUCCESS(host_resource_center_->TakeOverHostResources(root_graph));
  return SUCCESS;
}

Status GeRootModel::ModifyOwnerGraphForSubModels() {
  GE_ASSERT_NOTNULL(root_graph_);
  for (auto &iter : subgraph_instance_name_to_model_) {
    auto sub_graph = root_graph_->GetSubgraph(iter.first);
    if (sub_graph != nullptr) {
      iter.second->SetGraph(sub_graph);
    }
  }
  return SUCCESS;
}

void GeRootModel::SetSubgraphInstanceNameToModel(const std::string &instance_name, const GeModelPtr &ge_model) {
  (void)subgraph_instance_name_to_model_.insert(std::pair<std::string, GeModelPtr>(instance_name, ge_model));
}

void GeRootModel::RemoveInstanceSubgraphModel(const std::string &instance_name) {
  (void)subgraph_instance_name_to_model_.erase(instance_name);
}

Status GeRootModel::CheckIsUnknownShape(bool &is_dynamic_shape) const {
  if (root_graph_ == nullptr) {
    return FAILED;
  }
  is_dynamic_shape = false;
  (void)AttrUtils::GetBool(root_graph_, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape);
  is_dynamic_shape = (is_dynamic_shape || (root_graph_->GetGraphUnknownFlag()));
  return SUCCESS;
}

const uint8_t *GeRootModel::GetOpSoStoreData() const { return op_so_store_.Data(); }

size_t GeRootModel::GetOpStoreDataSize() const { return op_so_store_.DataSize(); }

uint16_t GeRootModel::GetSoInOmFlag() const { return so_in_om_; }

SoInOmInfo GeRootModel::GetSoInOmInfo() const { return so_info_; }

bool GeRootModel::LoadSoBinData(const uint8_t *const data, const size_t len) {
  return op_so_store_.Load(data, len);
}

std::vector<OpSoBinPtr> GeRootModel::GetAllSoBin() const {
  return op_so_store_.GetSoBin();
}

void GeRootModel::SetSoInOmInfo(const SoInOmInfo &so_info) {
  so_info_ = so_info;
  return;
}

Status GeRootModel::CheckAndSetNeedSoInOM() {
  GE_ASSERT_SUCCESS(CheckAndSetSpaceRegistry(), "Check space registry failed.");
  GE_ASSERT_SUCCESS(CheckAndSetOpMasterDevice(), "Check op master device failed.");
  GE_ASSERT_SUCCESS(CheckAndSetAutofuseSo(), "Check autofuse so failed.");
  GELOGI("so in om flag:0x%x", so_in_om_);
  return SUCCESS;
}

Status GeRootModel::CheckAndSetOpMasterDevice() {
  for (const auto &item : subgraph_instance_name_to_model_) {
    const auto &ge_model = item.second;
    GE_ASSERT_NOTNULL(ge_model);
    if (ge_model->GetModelTaskDefPtr() == nullptr) {
      GELOGI("Root model [%s][%d] has no task", ge_model->GetName().c_str(), ge_model->GetModelId());
      continue;
    }
    const auto &tasks = ge_model->GetModelTaskDefPtr()->task();
    for (int32_t i = 0; i < tasks.size(); ++i) {
      const domi::TaskDef &task_def = tasks[i];
      GELOGI("Task id = %d, task type = %u", i, task_def.type());
      if (static_cast<ModelTaskType>(task_def.type()) != ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL) {
        continue;
      }
      const auto &so_name = task_def.kernel().so_name();
      GE_ASSERT_TRUE(!so_name.empty(), "task [%u] has not set so_name", task_def.type());
      auto result = op_master_device_so_set_.insert(so_name);
      if (result.second) {
        GELOGI("[OpMasterDevice]Get so [%s] from model[%u] task[%u] kernel_type[%u].", so_name.c_str(),
               ge_model->GetModelId(), task_def.type(), task_def.kernel().context().kernel_type());
      }
    }
  }
  if (!op_master_device_so_set_.empty()) {
    OpSoStoreUtils::SetSoBinType(SoBinType::kOpMasterDevice, so_in_om_);
  }
  GELOGI("[OpMasterDevice]The num of so is %zu.", op_master_device_so_set_.size());
  return SUCCESS;
}

Status GeRootModel::CheckAndSetSpaceRegistry() {
  const ComputeGraphPtr &comp_graph = root_graph_;
  GE_ASSERT_NOTNULL(comp_graph);
  bool is_dynamic_shape = false;
  (void)AttrUtils::GetBool(comp_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape);
  is_dynamic_shape = (is_dynamic_shape || (comp_graph->GetGraphUnknownFlag()));
  if (is_dynamic_shape) {
    OpSoStoreUtils::SetSoBinType(SoBinType::kSpaceRegistry, so_in_om_);
    GELOGI("[SpaceRegistry]Has space registry as dynamic_shape.");
    return SUCCESS;
  }

  bool stc_to_dyn_soft_sync = false;
  for (const auto &node : comp_graph->GetAllNodes()) {
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), "_static_to_dynamic_softsync_op", stc_to_dyn_soft_sync);
    if (stc_to_dyn_soft_sync) {
      OpSoStoreUtils::SetSoBinType(SoBinType::kSpaceRegistry, so_in_om_);
      GELOGI("[SpaceRegistry]Has space registry as static_to_dynamic_softsync_op.");
      return SUCCESS;
    }
  }
  GELOGI("[SpaceRegistry]Has no space registry.");
  return SUCCESS;
}

Status GeRootModel::CheckAndSetAutofuseSo() {
  auto nodes = root_graph_->GetAllNodesPtr();
  for (auto node : nodes) {
    const std::string* so_path_ptr = ge::AttrUtils::GetStr(node->GetOpDesc(), "bin_file_path");
    if (so_path_ptr != nullptr) {
      auto result = autofuse_so_set_.insert(*so_path_ptr);
      if (result.second) {
          GELOGD("Added autofuse so %s.", (*so_path_ptr).c_str());
      }
    }
  }
  if (!autofuse_so_set_.empty()) {
    OpSoStoreUtils::SetSoBinType(SoBinType::kAutofuse, so_in_om_);
  }
  GELOGI("[AutofuseSo]The num of so is %zu.", autofuse_so_set_.size());
  return SUCCESS;
}

bool GeRootModel::IsNeedMallocFixedFeatureMem() const {
  bool is_unknown_shape = false;
  (void) CheckIsUnknownShape(is_unknown_shape);
  // 动态shape静态子图, 在init图中申请
  if (is_unknown_shape) {
    return false;
  }

  // 用户如果通过这个option设置了fixed内存，也不需要GE申请了
  std::string is_addr_fixed_opt;
  (void)ge::GetContext().GetOption("ge.exec.static_model_addr_fixed", is_addr_fixed_opt);
  if (!is_addr_fixed_opt.empty()) {
    GELOGI("user set ge.exec.static_model_addr_fixed option, return false");
    return false;
  }
  if (VarManager::IsGeUseExtendSizeMemory(false)) {
    GELOGI("enable extend memory, need to malloc fixed_feature_memory use session allocator, model_name: %s,"
        " model_id: %u", GetModelName().c_str(), GetCurModelId());
    return true;
  }
  std::string is_refreshable;
  (void)GetContext().GetOption(OPTION_FEATURE_BASE_REFRESHABLE, is_refreshable);
  static const std::string kEnabled = "1";
  // 如果用户没有设置feature base可刷新，ge也不需要申请fix内存
  if (is_refreshable != kEnabled) {
    GELOGI("feature base is not refreshable, return false, model_name: %s,"
           " model_id: %u", GetModelName().c_str(), GetCurModelId());
    return false;
  }
  return true;
}

/*
 * 如果用户设置了feature base可刷新，并且配置了正常的fix地址，Ge不再申请fix地址，need refresh为false。
 * 如果用户设置了feature base可刷新，没有配置fix地址，Ge需要兜底申请fix内存，need refresh为false。
 * 如果用户设置了feature base可刷新，并且配置了fix地址为nullptr，并且size也设置为0，Ge也不申请fix地址，并设置need refresh为true。
 * 如果用户没有设置feature base可刷新，配置了staticMemoryPolicy=4/2，图间共用fix优先内存, ge申请fix内存，need refresha为false.
 */
bool GeRootModel::IsNeedMallocFixedFeatureMemByType(const rtMemType_t rt_mem_type) const {
  const auto fixed_mem_iter = fixed_feature_mems_.find(rt_mem_type);
  // 没有配置fix地址, Ge需要兜底申请fix内存
  if (fixed_mem_iter == fixed_feature_mems_.end()) {
    GELOGI("fixed_feature_memory base is not set by user, return true, memory type: %s, model_name: %s, model_id: %u",
           MemTypeUtils::ToString(rt_mem_type).c_str(), GetModelName().c_str(), GetCurModelId());
    return true;
  }
  // 配置了正常的fix地址，Ge不再申请fix地址
  if (fixed_mem_iter->second.addr != nullptr) {
    GELOGI("fixed_feature_memory base[%p], return false, memory type: %s, model_name: %s, model_id: %u",
           fixed_mem_iter->second.addr, MemTypeUtils::ToString(rt_mem_type).c_str(), GetModelName().c_str(),
           GetCurModelId());
    return false;
  }
  // 配置了fix地址为nullptr，Ge也不申请fix地址
  if (fixed_mem_iter->second.user_alloc && (fixed_mem_iter->second.addr == nullptr)) {
    GELOGI("user set fixed_feature_memory base nullptr, return false, memory type: %s, model_name: %s, model_id: %u",
           MemTypeUtils::ToString(rt_mem_type).c_str(), GetModelName().c_str(), GetCurModelId());
    return false;
  }
  if (VarManager::IsGeUseExtendSizeMemory(false)) {
    GELOGI("enable extend memory, need to malloc fixed_feature_memory use session allocator, model_name: %s,"
           " model_id: %u", GetModelName().c_str(), GetCurModelId());
    return true;
  }
  GELOGI("fixed_feature_memory info: addr:%p, size:%zu, ge_alloc:%d, user_alloc:%d, memory type: %s, return false,"
      "model_name: %s, model_id: %u", fixed_mem_iter->second.addr, fixed_mem_iter->second.size,
      fixed_mem_iter->second.ge_alloc, fixed_mem_iter->second.user_alloc, MemTypeUtils::ToString(rt_mem_type).c_str(),
      GetModelName().c_str(), GetCurModelId());
  return false;
}

Status GeRootModel::GetSummaryFeatureMemory(std::vector<FeatureMemoryPtr> &all_feature_memory,
                                            size_t &hbm_fixed_feature_mem) {
  if (all_feature_memory_init_flag_) {
    all_feature_memory = all_feature_memory_;
    return SUCCESS;
  }
  hbm_fixed_feature_mem = 0U;
  for (const auto &name_to_model : GetSubgraphInstanceNameToModel()) {
    const auto &ge_model_temp = name_to_model.second;
    std::vector<std::vector<int64_t>> sub_mem_infos;
    (void)AttrUtils::GetListListInt(ge_model_temp, ATTR_MODEL_SUB_MEMORY_INFO, sub_mem_infos);
    size_t fixed_mem_size = 0UL;
    for (size_t index = 0UL; index < sub_mem_infos.size(); ++index) {
      const auto &sub_memory_info = sub_mem_infos[index];
      // 0U: memory_type, 1U:logic_memory_base, 2U:memory_size, 3U:is_fixed_addr_prior
      const bool is_fixed_addr_prior = (sub_memory_info.size() > 3U) && (sub_memory_info[3U] != 0L);
      // 2U:memory_size, is_fixed_addr_prior is true set memory size to fixed size
      fixed_mem_size += (is_fixed_addr_prior ? static_cast<size_t>(sub_memory_info[2U]) : 0UL);
    }
    if (fixed_mem_size > hbm_fixed_feature_mem) {
      hbm_fixed_feature_mem = fixed_mem_size;
    }
    GELOGI("[IMAS]model_name:%s hbm fixed_feature_memory size:%zu, hbm max size:%zu.",
           ge_model_temp->GetName().c_str(), fixed_mem_size, hbm_fixed_feature_mem);
    sub_mem_infos.clear();
  }
  if (hbm_fixed_feature_mem > 0U) {
    auto feature_memory = FeatureMemory::Builder::Build(MemoryType::MEMORY_TYPE_DEFAULT,
                                                        hbm_fixed_feature_mem, {true});
    GE_ASSERT_NOTNULL(feature_memory);
    (void)all_feature_memory_.emplace_back(std::move(feature_memory));
  }

  size_t p2p_fixed_size = 0U;
  GE_ASSERT_SUCCESS(GetP2pFixedFeatureMemorySize(this, p2p_fixed_size));
  if (p2p_fixed_size > 0U) {
    auto p2p_feature_memory = FeatureMemory::Builder::Build(MemoryType::MEMORY_TYPE_P2P,
                                                            p2p_fixed_size, {true});
    GE_ASSERT_NOTNULL(p2p_feature_memory);
    (void)all_feature_memory_.emplace_back(std::move(p2p_feature_memory));
  }
  all_feature_memory = all_feature_memory_;
  all_feature_memory_init_flag_ = true;
  return SUCCESS;
}

HostResourceCenterPtr GeRootModel::GetHostResourceCenterPtr() const {
  return host_resource_center_;
}
std::shared_ptr<GeRootModel> GeRootModel::Fork() {
  std::shared_ptr<GeRootModel> ge_root_model = MakeShared<ge::GeRootModel>();
  GE_ASSERT_NOTNULL(ge_root_model);
  ge_root_model->root_graph_ = root_graph_;
  ge_root_model->model_name_ = model_name_;
  ge_root_model->flatten_graph_ = this->flatten_graph_;
  ge_root_model->subgraph_instance_name_to_model_ = this->subgraph_instance_name_to_model_;
  ge_root_model->is_specific_stream_ = this->is_specific_stream_;
  ge_root_model->total_weight_size_ = this->total_weight_size_;
  ge_root_model->nodes_to_task_defs_ = this->nodes_to_task_defs_;
  ge_root_model->graph_to_static_models_ = this->graph_to_static_models_;
  ge_root_model->op_so_store_ = this->op_so_store_;
  ge_root_model->so_in_om_ = this->so_in_om_;
  ge_root_model->so_info_ = this->so_info_;
  ge_root_model->file_constant_weight_dir_ = this->file_constant_weight_dir_;
  ge_root_model->host_resource_center_ = this->host_resource_center_;
  ge_root_model->op_master_device_so_set_ = this->op_master_device_so_set_;
  ge_root_model->autofuse_so_set_ = this->autofuse_so_set_;
  return ge_root_model;
}
}  // namespace ge
