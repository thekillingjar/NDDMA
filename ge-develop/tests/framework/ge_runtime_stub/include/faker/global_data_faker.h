/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_UT_GE_RUNTIME_V2_FAKER_GLOBAL_DATA_FAKER_H_
#define AIR_CXX_TESTS_UT_GE_RUNTIME_V2_FAKER_GLOBAL_DATA_FAKER_H_
#include "aicore_taskdef_faker.h"
#include "common/ge_inner_attrs.h"
#include "exe_graph/lowering/frame_selector.h"
#include "exe_graph/lowering/lowering_global_data.h"
#include "faker/model_desc_holder_faker.h"
#include "faker/space_registry_faker.h"
#include "framework/common/taskdown_common.h"
#include "ge_model_builder.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "runtime/model_v2_executor.h"
#include "task_def_faker.h"
#include "graph/utils/execute_graph_adapter.h"
#include "common/host_resource_center/host_resource_center.h"

namespace gert {
class GlobalDataFaker {
 public:
  GlobalDataFaker() {};
  explicit GlobalDataFaker(const ge::GeRootModelPtr &root_model) : root_model_(root_model) , graph_(root_model->GetRootGraph()) {}

  GlobalDataFaker &FakeWithHandleAiCore(const std::string &node_type, bool need_atomic, bool need_aicpu = false) {
    node_types_to_faker_[node_type] = std::unique_ptr<AiCoreTaskDefFaker>(new AiCoreTaskDefFaker(true, need_atomic, "", need_aicpu));
    return *this;
  }
  GlobalDataFaker &FakeWithoutHandleAiCore(const std::string &node_type, bool need_atomic, bool need_aicpu = false) {
    node_types_to_faker_[node_type] =
        std::unique_ptr<AiCoreTaskDefFaker>(new AiCoreTaskDefFaker(false, need_atomic, node_type + "StubName", need_aicpu));
    return *this;
  }

  GlobalDataFaker &AddTaskDef(const std::string &node_type_or_name, const TaskDefFaker& task_def) {
    node_types_to_faker_[node_type_or_name] = task_def.Clone();
    return *this;
  }

  GlobalDataFaker &EnableSubMemInfo() {
    set_sub_mem_infos_ = true;
    return *this;
  }

  GlobalDataFaker &SetHostResourceCenter(void *host_resource_center) {
    host_resource_center_ = host_resource_center;
    return *this;
  }

  void BuildWithKnownSubgraph(LoweringGlobalData &global_data) {
    int64_t offset = 0;
    FakerFlattenOffset(graph_, offset);
    for (const auto &subgraph : graph_->GetAllSubgraphs()) {
      FakerFlattenOffset(subgraph, offset);
      if (subgraph == nullptr || subgraph->GetGraphUnknownFlag()) {
        continue;
      }
      if (subgraph->GetParentNode()->GetType() != ge::PARTITIONEDCALL && (!subgraph->HasAttr("_stub_force_known_shape"))) {
        continue;
      }
      bool without_kernel_flag = 0;
      if (ge::AttrUtils::GetBool(subgraph->GetParentNode()->GetOpDesc(),
                                 "without_kernel_store", without_kernel_flag)) {
        continue;
      }
      GeModelBuilder builder(subgraph);
      subgraph->SetGraphUnknownFlag(false);
      for (const auto &types2faker : node_types_to_faker_) {
        builder.AddTaskDef(types2faker.first, *types2faker.second);
      }
      auto model = builder.EnableSubMemInfo(set_sub_mem_infos_).Build();
      if (model == nullptr) {
        continue;
      }
      graph_to_ge_models_[subgraph->GetName()] = model;
      model->SetAttr(ge::ATTR_NAME_GRAPH_FLATTEN_OFFSET,
                    ge::GeAttrValue::CreateFrom<std::vector<int64_t>>({offset, static_cast<int64_t>(model->GetWeightSize())}));
      offset += model->GetWeightSize();
      global_data.AddStaticCompiledGraphModel(subgraph->GetName(), &(*model));
    }
    global_data.SetModelWeightSize(static_cast<size_t>(offset));
  }

  void FakerFlattenOffset(const ge::ComputeGraphPtr &graph, int64_t &offset) {
    int64_t graph_weight_size = 0;
    for (auto &node : graph->GetDirectNode()) {
      if (node->GetType() != "Const") {
        continue;
      }
      ge::GeTensorPtr weight;
      if (!ge::AttrUtils::MutableTensor(node->GetOpDesc(), "value", weight)) {
        GELOGE(ge::INTERNAL_ERROR, "weight is empty. node  = %s", node->GetName().c_str());
        continue;
      }
      graph_weight_size += weight->GetData().GetSize();
    }
    for (auto &node : graph->GetDirectNode()) {
      if (node->GetType() != "Const") {
        continue;
      }
      ge::GeTensorPtr weight;
      if (!ge::AttrUtils::MutableTensor(node->GetOpDesc(), "value", weight)) {
        GELOGE(ge::INTERNAL_ERROR, "weight is empty. node  = %s", node->GetName().c_str());
        continue;
      }
      ge::GeTensorDesc &tensor_desc = weight->MutableTensorDesc();
      tensor_desc.SetAttr(ge::ATTR_NAME_GRAPH_FLATTEN_OFFSET,
                          ge::GeAttrValue::CreateFrom<std::vector<int64_t>>({offset, graph_weight_size}));
    }
    offset += graph_weight_size;
  }

  LoweringGlobalData Build(size_t stream_num = 1) {
    LoweringGlobalData global_data;
    global_data.LoweringAndSplitRtStreams(stream_num);
    global_data.SetExternalAllocator(bg::ValueHolder::CreateFeed(
        static_cast<int64_t>(ExecuteArgIndex::kExternalAllocator)));
    for (const auto &node : graph_->GetAllNodes()) {
      auto iter = node_types_to_faker_.find(node->GetName());
      if (iter == node_types_to_faker_.end()) {
        auto node_type = ge::NodeUtils::GetNodeType(node);
        iter = node_types_to_faker_.find(node_type);
        if (iter == node_types_to_faker_.end()) {
          continue;
        }
      }
      global_data.AddCompiledResult(node, {iter->second->CreateTaskDef()});
    }
    BuildWithKnownSubgraph(global_data);
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    if (space_registry == nullptr) {
      space_registry = SpaceRegistryFaker().Build();
    }
    auto space_registry_array = OpImplSpaceRegistryV2Array();
    space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = space_registry;
    global_data.SetSpaceRegistriesV2(space_registry_array);

    // prepare stream num in init for l2 allocator
    const auto current_fame = bg::ValueHolder::GetCurrentFrame();
    if (current_fame != nullptr) {
      const auto is_parent_graph_exist = current_fame->GetExecuteGraph()->GetParentGraphBarePtr() != nullptr;
      if (is_parent_graph_exist) {
        auto init_out = bg::FrameSelector::OnInitRoot([&stream_num, &global_data]()-> std::vector<bg::ValueHolderPtr> {
          global_data.LoweringAndSplitRtStreams(1);
          auto stream_num_holder = bg::ValueHolder::CreateConst(&stream_num, sizeof(stream_num));
          global_data.SetUniqueValueHolder(kGlobalDataModelStreamNum, stream_num_holder);
          return {};
        });
      }
    }

    if (stream_num == 1 && graph_->GetDirectNodesSize() != 0) {
      // fake current compute node to ensure GetOrCreateAllocator can work in single stream UT
      // GetOrCreateAllocator depends on GetCurrentComputeNode to get current stream id
      bg::ValueHolder::SetCurrentComputeNode(*graph_->GetDirectNode().begin());
    }
    if (root_model_ == nullptr) {
      GELOGW("Global data faker get a null root model. May fail when lowering const");
      return {};
    }
    global_data.SetHostResourceCenter(root_model_->GetHostResourceCenterPtr().get());
    return global_data;
  }
  // If you only lowering something on ut, and your lowering logic need use stream, fake global data with this
  // This interface will help you easy to use stream
  LoweringGlobalData BuildWithCurrentStreamOnMainRoot(const ge::NodePtr &node) {
    auto global_data = Build();
    global_data.LoweringAndSplitRtStreams(1);
    return global_data;
  }

  LoweringGlobalData BuildEmptyGlobalData() const {
    LoweringGlobalData global_data;
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    if (space_registry == nullptr) {
      space_registry = SpaceRegistryFaker().Build();
    }
    auto space_registry_array = OpImplSpaceRegistryV2Array();
    space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = space_registry;
    global_data.SetSpaceRegistriesV2(space_registry_array);
    return global_data;
  }

 private:
  bool set_sub_mem_infos_{false};
  ge::GeRootModelPtr root_model_;
  ge::ComputeGraphPtr graph_;
  std::unordered_map<std::string, std::unique_ptr<TaskDefFaker>> node_types_to_faker_;
  std::unordered_map<std::string, ge::GeModelPtr> graph_to_ge_models_;
  void *host_resource_center_{nullptr};
};
}  // namespace gert
#endif  //AIR_CXX_TESTS_UT_GE_RUNTIME_V2_FAKER_GLOBAL_DATA_FAKER_H_
