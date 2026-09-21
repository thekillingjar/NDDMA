/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "analyzer/analyzer.h"
#include "graph/op_desc.h"
#include "graph/utils/graph_utils.h"
#include "common/host_resource_center/tiling_resource_manager.h"
#include "common/host_resource_center/host_resource_center.h"
#include "common/host_resource_center/host_resource_manager.h"
#include "common/host_resource_center/host_resource_serializer.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "register/op_tiling_info.h"
#include "graph/utils/node_utils.h"

namespace ge {
class TilingResourceManagerUT : public testing::Test {};

TEST_F(TilingResourceManagerUT, SetGetResource) {
  auto op_desc = std::make_shared<OpDesc>("fake", "fake");
  auto run_info = std::make_shared<optiling::utils::OpRunInfo>();
  auto host_resource = std::dynamic_pointer_cast<HostResource>(std::make_shared<TilingResource>(run_info));
  TilingResourceManager manager;
  ASSERT_EQ(manager.AddResource(op_desc, 0, host_resource), SUCCESS);
  const TilingResource *resource = dynamic_cast<const TilingResource *>(manager.GetResource(op_desc, 0));
  ASSERT_NE(resource, nullptr);
  EXPECT_EQ(resource->GetOpRunInfo().get(), run_info.get());
}

TEST_F(TilingResourceManagerUT, SerializeDeSerialize_Success) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", "Data")->EDGE(0, 0)->NODE("reduce_sum0", "ReduceSum"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("reduce_sum1", "ReduceSum"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 0)->NODE("reduce_sum2", "ReduceSum"));
    CHAIN(NODE("data3", "Data")->EDGE(0, 0)->NODE("reduce_sum3", "ReduceSum"));
    CHAIN(NODE("data4", "Data")->EDGE(0, 0)->NODE("reduce_sum4", "ReduceSum"));
    CHAIN(
        NODE("reduce_sum0", "ReduceSum")->EDGE(0, 0)->NODE("add0", "Add")->EDGE(0, 0)->NODE("net_output", "NetOutput"));
    CHAIN(NODE("reduce_sum0", "ReduceSum")->EDGE(0, 1)->NODE("add0", "Add"));
    CHAIN(
        NODE("reduce_sum0", "ReduceSum")->EDGE(0, 0)->NODE("add1", "Add")->EDGE(0, 1)->NODE("net_output", "NetOutput"));
    CHAIN(NODE("reduce_sum0", "ReduceSum")->EDGE(0, 1)->NODE("add1", "Add"));
  };
  auto graph = ToComputeGraph(g1);
  std::string tiling_data = "hahahaha";
  std::shared_ptr<optiling::utils::OpRunInfo> reduce_sum_run_info = std::make_shared<optiling::utils::OpRunInfo>();
  reduce_sum_run_info->AddTilingData(tiling_data.data(), tiling_data.size());
  std::shared_ptr<optiling::utils::OpRunInfo> reduce_sum_atomic_run_info =
      std::make_shared<optiling::utils::OpRunInfo>();
  reduce_sum_atomic_run_info->AddTilingData(tiling_data.data(), tiling_data.size());
  std::shared_ptr<optiling::utils::OpRunInfo> add_run_info = std::make_shared<optiling::utils::OpRunInfo>();
  add_run_info->AddTilingData(tiling_data.data(), tiling_data.size());

  for (auto node : graph->GetAllNodesPtr()) {
    if (node->GetType() == "ReduceSum") {
      node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_OP_RUN_INFO, reduce_sum_run_info);
      node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO, reduce_sum_atomic_run_info);
    }
    if (node->GetType() == "Add") {
      node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_OP_RUN_INFO, add_run_info);
    }
  }

  // mock add resource
  HostResourceCenter center = HostResourceCenter();
  TilingResourceManager *manager = const_cast<TilingResourceManager *>(
      dynamic_cast<const TilingResourceManager *>(center.GetHostResourceMgr(HostResourceType::kTilingData)));
  ASSERT_NE(manager, nullptr);

  for (auto node : graph->GetAllNodesPtr()) {
    auto op_desc = node->GetOpDesc();
    manager->TakeoverResources(op_desc);
  }

  HostResourceSerializer serializer;
  HostResourcePartition partition;
  partition.total_size_ = 0UL;
  ASSERT_EQ(serializer.SerializeTilingData(center, partition), SUCCESS);
  EXPECT_NE(partition.total_size_, 0UL);
  EXPECT_EQ(partition.total_size_, partition.buffer_.size());

  // unserialize
  for (auto node : graph->GetAllNodesPtr()) {
    auto op_desc = node->GetOpDescBarePtr();
    op_desc->DelExtAttr(ATTR_NAME_OP_RUN_INFO);
    op_desc->DelExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO);
  }
  //
  manager->resource_to_keys_.clear();
  manager->key_to_resources_.clear();
  //
  ASSERT_EQ(serializer.DeSerializeTilingData(center, partition.buffer_.data(), partition.total_size_), SUCCESS);

  EXPECT_EQ(manager->resource_to_keys_.size(), 3UL);
  EXPECT_EQ(manager->key_to_resources_.size(), 12UL);
  ASSERT_EQ(serializer.RecoverOpRunInfoToExtAttrs(center, graph), SUCCESS);
}

TEST_F(TilingResourceManagerUT, SerializeDeSerialize_Sk_Success) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", "Data")->EDGE(0, 0)->NODE("reduce_sum0", "ReduceSum"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("reduce_sum1", "ReduceSum"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 0)->NODE("reduce_sum2", "ReduceSum"));
    CHAIN(NODE("data3", "Data")->EDGE(0, 0)->NODE("reduce_sum3", "ReduceSum"));
    CHAIN(NODE("data4", "Data")->EDGE(0, 0)->NODE("reduce_sum4", "ReduceSum"));
    CHAIN(
        NODE("reduce_sum0", "ReduceSum")->EDGE(0, 0)->NODE("add0", "Add")->EDGE(0, 0)->NODE("net_output", "NetOutput"));
    CHAIN(NODE("reduce_sum0", "ReduceSum")->EDGE(0, 1)->NODE("add0", "Add"));
    CHAIN(
        NODE("reduce_sum0", "ReduceSum")->EDGE(0, 0)->NODE("add1", "Add")->EDGE(0, 1)->NODE("net_output", "NetOutput"));
    CHAIN(NODE("reduce_sum0", "ReduceSum")->EDGE(0, 1)->NODE("add1", "Add"));
  };
  auto graph = ToComputeGraph(g1);
  std::string tiling_data = "hahahaha";
  std::shared_ptr<optiling::utils::OpRunInfo> reduce_sum_run_info = std::make_shared<optiling::utils::OpRunInfo>();
  reduce_sum_run_info->AddTilingData(tiling_data.data(), tiling_data.size());
  std::shared_ptr<optiling::utils::OpRunInfo> reduce_sum_atomic_run_info =
      std::make_shared<optiling::utils::OpRunInfo>();
  reduce_sum_atomic_run_info->AddTilingData(tiling_data.data(), tiling_data.size());
  std::shared_ptr<optiling::utils::OpRunInfo> add_run_info = std::make_shared<optiling::utils::OpRunInfo>();
  add_run_info->AddTilingData(tiling_data.data(), tiling_data.size());

  for (auto node : graph->GetAllNodesPtr()) {
    if (node->GetType() == "ReduceSum") {
      node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_OP_RUN_INFO, reduce_sum_run_info);
      node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO, reduce_sum_atomic_run_info);
    }
    if (node->GetType() == "Add") {
      node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_OP_RUN_INFO, add_run_info);
    }
  }

  // mock add resource
  HostResourceCenter center = HostResourceCenter();
  TilingResourceManager *manager = const_cast<TilingResourceManager *>(
      dynamic_cast<const TilingResourceManager *>(center.GetHostResourceMgr(HostResourceType::kTilingData)));
  ASSERT_NE(manager, nullptr);

  DEF_GRAPH(sk_parent_graph) {
    CHAIN(NODE("data0", "Data")->EDGE(0, 0)->NODE("sk", "SuperKernel")->EDGE(0, 0)->NODE("net_output", "NetOutput"));
  };
  auto root_graph = ToComputeGraph(sk_parent_graph);
  for (auto node : root_graph->GetAllNodesPtr()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc->GetType() == "SuperKernel") {
      op_desc->SetExtAttr("_sk_sub_graph", graph);
    }
    manager->TakeoverResources(std::dynamic_pointer_cast<AttrHolder>(op_desc));
  }

  HostResourceSerializer serializer;
  HostResourcePartition partition;
  partition.total_size_ = 0UL;
  ASSERT_EQ(serializer.SerializeTilingData(center, partition), SUCCESS);
  EXPECT_NE(partition.total_size_, 0UL);
  EXPECT_EQ(partition.total_size_, partition.buffer_.size());

  manager->resource_to_keys_.clear();
  manager->key_to_resources_.clear();
  //
  ASSERT_EQ(serializer.DeSerializeTilingData(center, partition.buffer_.data(), partition.total_size_), SUCCESS);

  EXPECT_EQ(manager->resource_to_keys_.size(), 3UL);
  EXPECT_EQ(manager->key_to_resources_.size(), 12UL);
  ASSERT_EQ(serializer.RecoverOpRunInfoToExtAttrs(center, root_graph), SUCCESS);
}

TEST_F(TilingResourceManagerUT, SerializeDeSerialize_WithOldVersion_Success) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", "Data")->EDGE(0, 0)->NODE("reduce_sum0", "ReduceSum"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("reduce_sum1", "ReduceSum"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 0)->NODE("reduce_sum2", "ReduceSum"));
    CHAIN(NODE("data3", "Data")->EDGE(0, 0)->NODE("reduce_sum3", "ReduceSum"));
    CHAIN(NODE("data4", "Data")->EDGE(0, 0)->NODE("reduce_sum4", "ReduceSum"));
    CHAIN(
        NODE("reduce_sum0", "ReduceSum")->EDGE(0, 0)->NODE("add0", "Add")->EDGE(0, 0)->NODE("net_output", "NetOutput"));
    CHAIN(NODE("reduce_sum0", "ReduceSum")->EDGE(0, 1)->NODE("add0", "Add"));
    CHAIN(
        NODE("reduce_sum0", "ReduceSum")->EDGE(0, 0)->NODE("add1", "Add")->EDGE(0, 1)->NODE("net_output", "NetOutput"));
    CHAIN(NODE("reduce_sum0", "ReduceSum")->EDGE(0, 1)->NODE("add1", "Add"));
  };
  auto graph = ToComputeGraph(g1);
  std::string tiling_data = "hahahaha";
  std::shared_ptr<optiling::utils::OpRunInfo> reduce_sum_run_info = std::make_shared<optiling::utils::OpRunInfo>();
  reduce_sum_run_info->AddTilingData(tiling_data.data(), tiling_data.size());
  std::shared_ptr<optiling::utils::OpRunInfo> reduce_sum_atomic_run_info =
      std::make_shared<optiling::utils::OpRunInfo>();
  reduce_sum_atomic_run_info->AddTilingData(tiling_data.data(), tiling_data.size());
  std::shared_ptr<optiling::utils::OpRunInfo> add_run_info = std::make_shared<optiling::utils::OpRunInfo>();
  add_run_info->AddTilingData(tiling_data.data(), tiling_data.size());

  // mock add resource
  HostResourceCenter center = HostResourceCenter();
  TilingResourceManager *manager = const_cast<TilingResourceManager *>(
      dynamic_cast<const TilingResourceManager *>(center.GetHostResourceMgr(HostResourceType::kTilingData)));
  ASSERT_NE(manager, nullptr);

  for (auto node : graph->GetAllNodesPtr()) {
    if (node->GetType() == "ReduceSum") {
      auto aicore_id = TilingResourceManager::GenTilingId(*node->GetOpDesc(), TilingResourceType::kAiCore);
      std::shared_ptr<HostResource> host_resource = std::make_shared<TilingResource>(reduce_sum_run_info);
      manager->key_to_resources_[aicore_id] = host_resource;
      manager->resource_to_keys_[reduce_sum_run_info.get()].push_back(aicore_id);

      auto atomic_id = TilingResourceManager::GenTilingId(*node->GetOpDesc(), TilingResourceType::kAtomic);
      std::shared_ptr<HostResource> atomic_resource = std::make_shared<TilingResource>(reduce_sum_atomic_run_info);
      manager->key_to_resources_[atomic_id] = atomic_resource;
      manager->resource_to_keys_[reduce_sum_atomic_run_info.get()].push_back(atomic_id);

      node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_OP_RUN_INFO, reduce_sum_run_info);
      node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO, reduce_sum_atomic_run_info);
    }
    if (node->GetType() == "Add") {
      auto aicore_id = TilingResourceManager::GenTilingId(*node->GetOpDesc(), TilingResourceType::kAiCore);
      std::shared_ptr<HostResource> host_resource = std::make_shared<TilingResource>(add_run_info);
      manager->key_to_resources_[aicore_id] = host_resource;
      manager->resource_to_keys_[add_run_info.get()].push_back(aicore_id);

      node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_OP_RUN_INFO, add_run_info);
    }
  }

  HostResourceSerializer serializer;
  HostResourcePartition partition;
  partition.total_size_ = 0UL;
  ASSERT_EQ(serializer.SerializeTilingData(center, partition), SUCCESS);
  EXPECT_NE(partition.total_size_, 0UL);
  EXPECT_EQ(partition.total_size_, partition.buffer_.size());

  struct TilingHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t data_len;
  };

  auto tiling_header = PtrToPtr<uint8_t, TilingHeader>(partition.buffer_.data());
  tiling_header->version = 0x1;

  // unserialize
  for (auto node : graph->GetAllNodesPtr()) {
    auto op_desc = node->GetOpDescBarePtr();
    op_desc->DelExtAttr(ATTR_NAME_OP_RUN_INFO);
    op_desc->DelExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO);
  }
  //
  manager->resource_to_keys_.clear();
  manager->key_to_resources_.clear();
  //
  ASSERT_EQ(serializer.DeSerializeTilingData(center, partition.buffer_.data(), partition.total_size_), SUCCESS);

  EXPECT_EQ(manager->resource_to_keys_.size(), 3UL);
  EXPECT_EQ(manager->key_to_resources_.size(), 12UL);
  ASSERT_EQ(serializer.RecoverOpRunInfoToExtAttrs(center, graph), SUCCESS);

  std::shared_ptr<optiling::utils::OpRunInfo> default_empty;
  for (auto node : graph->GetAllNodesPtr()) {
    if (node->GetType() == "ReduceSum") {
      std::shared_ptr<optiling::utils::OpRunInfo> emp1 =
          node->GetOpDesc()->TryGetExtAttr(ATTR_NAME_OP_RUN_INFO, default_empty);
      std::shared_ptr<optiling::utils::OpRunInfo> emp2 =
          node->GetOpDesc()->TryGetExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO, default_empty);
      EXPECT_NE(emp1.get(), nullptr);
      EXPECT_NE(emp2.get(), nullptr);
    }
    if (node->GetType() == "Add") {
      std::shared_ptr<optiling::utils::OpRunInfo> emp1 =
          node->GetOpDesc()->TryGetExtAttr(ATTR_NAME_OP_RUN_INFO, default_empty);
      EXPECT_NE(emp1.get(), nullptr);
    }
  }
}

TEST_F(TilingResourceManagerUT, MultiNodesSerializeDeSerialize_Success) {
  ComputeGraphPtr root_graph = std::make_shared<ComputeGraph>("root_graph");

  std::string tiling_data = "hahahaha";
  std::shared_ptr<optiling::utils::OpRunInfo> reduce_sum_run_info = std::make_shared<optiling::utils::OpRunInfo>();
  reduce_sum_run_info->AddTilingData(tiling_data.data(), tiling_data.size());
  reduce_sum_run_info->SetTilingKey(10U);
  reduce_sum_run_info->SetBlockDim(10U);
  reduce_sum_run_info->SetWorkspaces({1000, 1000});

  std::shared_ptr<optiling::utils::OpRunInfo> reduce_sum_atomic_run_info =
      std::make_shared<optiling::utils::OpRunInfo>();
  reduce_sum_atomic_run_info->AddTilingData(tiling_data.data(), tiling_data.size());
  std::shared_ptr<optiling::utils::OpRunInfo> add_run_info = std::make_shared<optiling::utils::OpRunInfo>();
  add_run_info->AddTilingData(tiling_data.data(), tiling_data.size());
  reduce_sum_atomic_run_info->SetTilingKey(20U);
  reduce_sum_atomic_run_info->SetBlockDim(20U);
  reduce_sum_atomic_run_info->SetWorkspaces({2000, 2000});

  std::shared_ptr<optiling::utils::OpRunInfo> empty_data_run_info = std::make_shared<optiling::utils::OpRunInfo>();
  empty_data_run_info->SetTilingKey(30U);
  empty_data_run_info->SetBlockDim(30U);
  empty_data_run_info->SetWorkspaces({3000, 3000});

  vector<int64_t> empty_id;
  vector<int64_t> true_id;
  for (size_t i = 0UL; i < 1000UL; ++i) {
    OpDescPtr desc = std::make_shared<OpDesc>("reduce" + std::to_string(i), "ReduceSum");
    if (i % 4 == 0) {
      desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, empty_data_run_info);
      auto node = root_graph->AddNode(desc);
      empty_id.push_back(node->GetOpDesc()->GetId());
    } else {
      desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, reduce_sum_run_info);
      desc->SetExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO, reduce_sum_atomic_run_info);
      auto node = root_graph->AddNode(desc);
      true_id.push_back(node->GetOpDesc()->GetId());
    }
  }

  // mock add resource
  HostResourceCenter center = HostResourceCenter();
  TilingResourceManager *manager = const_cast<TilingResourceManager *>(
      dynamic_cast<const TilingResourceManager *>(center.GetHostResourceMgr(HostResourceType::kTilingData)));
  ASSERT_NE(manager, nullptr);

  for (auto node : root_graph->GetAllNodesPtr()) {
    auto op_desc = node->GetOpDesc();
    manager->TakeoverResources(op_desc);
  }

  HostResourceSerializer serializer;
  HostResourcePartition partition;
  partition.total_size_ = 0UL;
  ASSERT_EQ(serializer.SerializeTilingData(center, partition), SUCCESS);
  EXPECT_NE(partition.total_size_, 0UL);
  EXPECT_EQ(partition.total_size_, partition.buffer_.size());

  // unserialize
  for (auto node : root_graph->GetAllNodesPtr()) {
    auto op_desc = node->GetOpDescBarePtr();
    op_desc->DelExtAttr(ATTR_NAME_OP_RUN_INFO);
    op_desc->DelExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO);
  }
  //
  manager->resource_to_keys_.clear();
  manager->key_to_resources_.clear();
  //
  ASSERT_EQ(serializer.DeSerializeTilingData(center, partition.buffer_.data(), partition.total_size_), SUCCESS);

  EXPECT_EQ(manager->resource_to_keys_.size(), 3UL);
  EXPECT_EQ(manager->key_to_resources_.size(), 1750UL);
  ASSERT_EQ(serializer.RecoverOpRunInfoToExtAttrs(center, root_graph), SUCCESS);

  std::shared_ptr<optiling::utils::OpRunInfo> default_empty;
  {
    auto validate1 = root_graph->FindNode("reduce" + std::to_string(empty_id[0]));
    ASSERT_NE(validate1, nullptr);
    std::shared_ptr<optiling::utils::OpRunInfo> emp1 =
        validate1->GetOpDesc()->TryGetExtAttr(ATTR_NAME_OP_RUN_INFO, default_empty);
    ASSERT_NE(emp1, nullptr);
    EXPECT_EQ(emp1->GetAllTilingData().str(), "");
    EXPECT_EQ(emp1->GetTilingKey(), 30U);
    EXPECT_EQ(emp1->GetBlockDim(), 30U);
    std::vector<int64_t> golden_workspace{3000, 3000};
    EXPECT_EQ(emp1->GetAllWorkspaces(), golden_workspace);
  }

  {
    auto validate2 = root_graph->FindNode("reduce" + std::to_string(true_id[0]));
    ASSERT_NE(validate2, nullptr);

    std::shared_ptr<optiling::utils::OpRunInfo> res1 =
        validate2->GetOpDesc()->TryGetExtAttr(ATTR_NAME_OP_RUN_INFO, default_empty);
    ASSERT_NE(res1, nullptr);
    EXPECT_EQ(res1->GetAllTilingData().str(), "hahahaha");
    EXPECT_EQ(res1->GetTilingKey(), 10U);
    EXPECT_EQ(res1->GetBlockDim(), 10U);
    std::vector<int64_t> golden_workspace{1000, 1000};
    EXPECT_EQ(res1->GetAllWorkspaces(), golden_workspace);

    std::shared_ptr<optiling::utils::OpRunInfo> res2 =
        validate2->GetOpDesc()->TryGetExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO, default_empty);
    ASSERT_NE(res2, nullptr);
    EXPECT_EQ(res2->GetAllTilingData().str(), "hahahaha");
    EXPECT_EQ(res2->GetTilingKey(), 20U);
    EXPECT_EQ(res2->GetBlockDim(), 20U);
    std::vector<int64_t> golden_workspace2{2000, 2000};
    EXPECT_EQ(res2->GetAllWorkspaces(), golden_workspace2);
  }
}

TEST_F(TilingResourceManagerUT, MultiNodesSerializeDeSerialize_with_repeated_op_id_Success) {
  ComputeGraphPtr root_graph = std::make_shared<ComputeGraph>("root_graph");

  std::string tiling_data = "hahahaha";
  std::shared_ptr<optiling::utils::OpRunInfo> reduce_sum_run_info = std::make_shared<optiling::utils::OpRunInfo>();
  reduce_sum_run_info->AddTilingData(tiling_data.data(), tiling_data.size());
  reduce_sum_run_info->SetTilingKey(10U);
  reduce_sum_run_info->SetBlockDim(10U);
  reduce_sum_run_info->SetWorkspaces({1000, 1000});

  std::shared_ptr<optiling::utils::OpRunInfo> reduce_sum_atomic_run_info =
      std::make_shared<optiling::utils::OpRunInfo>();
  reduce_sum_atomic_run_info->AddTilingData(tiling_data.data(), tiling_data.size());
  std::shared_ptr<optiling::utils::OpRunInfo> add_run_info = std::make_shared<optiling::utils::OpRunInfo>();
  add_run_info->AddTilingData(tiling_data.data(), tiling_data.size());
  reduce_sum_atomic_run_info->SetTilingKey(20U);
  reduce_sum_atomic_run_info->SetBlockDim(20U);
  reduce_sum_atomic_run_info->SetWorkspaces({2000, 2000});

  std::shared_ptr<optiling::utils::OpRunInfo> empty_data_run_info = std::make_shared<optiling::utils::OpRunInfo>();
  empty_data_run_info->SetTilingKey(30U);
  empty_data_run_info->SetBlockDim(30U);
  empty_data_run_info->SetWorkspaces({3000, 3000});

  vector<OpDescPtr> empty_ops;
  vector<OpDescPtr> true_ops;
  for (size_t i = 0UL; i < 1000UL; ++i) {
    OpDescPtr desc = std::make_shared<OpDesc>("reduce" + std::to_string(i), "ReduceSum");
    if (i % 4 == 0) {
      desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, empty_data_run_info);
      auto node = root_graph->AddNode(desc);
      desc->SetId(0);
      empty_ops.push_back(node->GetOpDesc());
    } else {
      desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, reduce_sum_run_info);
      desc->SetExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO, reduce_sum_atomic_run_info);
      auto node = root_graph->AddNode(desc);
      desc->SetId(0);
      true_ops.push_back(node->GetOpDesc());
    }
  }

  // mock add resource
  HostResourceCenter center = HostResourceCenter();
  TilingResourceManager *manager = const_cast<TilingResourceManager *>(
      dynamic_cast<const TilingResourceManager *>(center.GetHostResourceMgr(HostResourceType::kTilingData)));
  ASSERT_NE(manager, nullptr);

  for (auto node : root_graph->GetAllNodesPtr()) {
    auto op_desc = node->GetOpDesc();
    manager->TakeoverResources(op_desc);
  }

  HostResourceSerializer serializer;
  HostResourcePartition partition;
  partition.total_size_ = 0UL;
  ASSERT_EQ(serializer.SerializeTilingData(center, partition), SUCCESS);
  EXPECT_NE(partition.total_size_, 0UL);
  EXPECT_EQ(partition.total_size_, partition.buffer_.size());

  // unserialize
  for (auto node : root_graph->GetAllNodesPtr()) {
    auto op_desc = node->GetOpDescBarePtr();
    op_desc->DelExtAttr(ATTR_NAME_OP_RUN_INFO);
    op_desc->DelExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO);
  }
  //
  manager->resource_to_keys_.clear();
  manager->key_to_resources_.clear();
  //
  ASSERT_EQ(serializer.DeSerializeTilingData(center, partition.buffer_.data(), partition.total_size_), SUCCESS);

  EXPECT_EQ(manager->resource_to_keys_.size(), 3UL);
  EXPECT_EQ(manager->key_to_resources_.size(), 1750UL);
  ASSERT_EQ(serializer.RecoverOpRunInfoToExtAttrs(center, root_graph), SUCCESS);

  std::shared_ptr<optiling::utils::OpRunInfo> default_empty;
  {
    ASSERT_TRUE(!empty_ops.empty());
    auto validate1 = empty_ops[0UL];
    ASSERT_NE(validate1, nullptr);
    EXPECT_EQ(validate1->GetId(), 0);
    std::shared_ptr<optiling::utils::OpRunInfo> emp1 = validate1->TryGetExtAttr(ATTR_NAME_OP_RUN_INFO, default_empty);
    ASSERT_NE(emp1, nullptr);
    EXPECT_EQ(emp1->GetAllTilingData().str(), "");
    EXPECT_EQ(emp1->GetTilingKey(), 30U);
    EXPECT_EQ(emp1->GetBlockDim(), 30U);
    std::vector<int64_t> golden_workspace{3000, 3000};
    EXPECT_EQ(emp1->GetAllWorkspaces(), golden_workspace);
  }

  {
    ASSERT_TRUE(!true_ops.empty());
    auto validate2 = true_ops[0UL];
    ASSERT_NE(validate2, nullptr);
    EXPECT_EQ(validate2->GetId(), 0);
    std::shared_ptr<optiling::utils::OpRunInfo> res1 = validate2->TryGetExtAttr(ATTR_NAME_OP_RUN_INFO, default_empty);
    ASSERT_NE(res1, nullptr);
    EXPECT_EQ(res1->GetAllTilingData().str(), "hahahaha");
    EXPECT_EQ(res1->GetTilingKey(), 10U);
    EXPECT_EQ(res1->GetBlockDim(), 10U);
    std::vector<int64_t> golden_workspace{1000, 1000};
    EXPECT_EQ(res1->GetAllWorkspaces(), golden_workspace);

    std::shared_ptr<optiling::utils::OpRunInfo> res2 =
        validate2->TryGetExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO, default_empty);
    ASSERT_NE(res2, nullptr);
    EXPECT_EQ(res2->GetAllTilingData().str(), "hahahaha");
    EXPECT_EQ(res2->GetTilingKey(), 20U);
    EXPECT_EQ(res2->GetBlockDim(), 20U);
    std::vector<int64_t> golden_workspace2{2000, 2000};
    EXPECT_EQ(res2->GetAllWorkspaces(), golden_workspace2);
  }
}
}  // namespace ge
