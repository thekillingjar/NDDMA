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
#include "common/ge_common/ge_types.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "graph/symbolizer/symbolic.h"
#include "common/util/mem_utils.h"
#include "can_fuse/fusion_strategy_solver.h"
#include "can_fuse/backend/fusion_decider_registry.h"
#include "can_fuse/backend/asc_backend_fusion_decider.h"
#include "utils/autofuse_attrs.h"
#include "utils/autofuse_utils.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"


using namespace std;
using namespace testing;

namespace ge {
class UtestFusionStrategySolverManual : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }

  static Status SetAttrsGroup(const NodePtr &node) {
    gert::SymbolShape ori_symbol_shape({Symbol(1), Symbol(2), Symbol(3), Symbol(4)});
    gert::SymbolShape symbol_shape({Symbol(1), Symbol(2), Symbol(3), Symbol(4)});
    const auto node_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(node_desc);
    for (const auto out_anchor : node->GetAllOutDataAnchorsPtr()) {
      GE_ASSERT_NOTNULL(out_anchor);
      auto output_tensor_desc = node_desc->MutableOutputDesc(out_anchor->GetIdx());
      if (output_tensor_desc != nullptr) {
        output_tensor_desc->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.MutableOriginSymbolShape() =
            symbol_shape;
      }
    }
    for (const auto in_anchor : node->GetAllInDataAnchorsPtr()) {
      GE_ASSERT_NOTNULL(in_anchor);
      auto input_tensor_desc = node_desc->MutableInputDesc(in_anchor->GetIdx());
      if (input_tensor_desc != nullptr) {
        input_tensor_desc->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.MutableOriginSymbolShape() =
            symbol_shape;
      }
    }
    return SUCCESS;
  }

  static bool MockCanFuse(const NodePtr &node, bool &can_fuse) {
    // 悬空输入，输出的会处理失败，这里返回不能融合
    for (const auto in_anchor : node->GetAllInDataAnchorsPtr()) {
      GE_ASSERT_NOTNULL(in_anchor);
      const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
      if (peer_out_anchor == nullptr) {
        can_fuse = false;
        return false;
      }
    }

    for (const auto out_anchor : node->GetAllOutDataAnchorsPtr()) {
      GE_ASSERT_NOTNULL(out_anchor);
      for(const auto &peer_in_anchor: out_anchor->GetPeerInDataAnchors()) {
        if (peer_in_anchor == nullptr) {
          can_fuse = false;
          return false;
        }
      }
    }

    if (!AutofuseUtils::IsAutoFuseNode(node->GetOpDesc())) {
      return false;
    }
    const std::string ATTR_NAME_OP_KERNEL_LIB_NAME = "_ge_attr_op_kernel_lib_name";
    std::string op_kernel_lib_name;
    (void)AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_OP_KERNEL_LIB_NAME, op_kernel_lib_name);
    if (op_kernel_lib_name == "DNN_VM_GE_LOCAL_OP_STORE") {
      return false;
    }
    return true;
  }
 public:
};

TEST_F(UtestFusionStrategySolverManual, Fuse_all) {
  class MockFusionDecider : public AscBackendFusionDecider {
    FusionPriority GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) {
      return FusionPriority::HIGH;
    }

    bool CanFuseVertical(const NodePtr &node1, const NodePtr &node2) {
      bool can_fuse = true;
      auto node1_can_fuse = MockCanFuse(node1, can_fuse);
      auto node2_can_fuse = MockCanFuse(node2, can_fuse);
      if (!can_fuse || (!node1_can_fuse && !node2_can_fuse)) {
        return false;
      }
      return true;
    }

    bool CanFuseHorizontal(const NodePtr &node1, const NodePtr &node2) {
      return CanFuseVertical(node1, node2);
    }

    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      auto op_desc = node1->GetOpDescBarePtr();
      GE_ASSERT_NOTNULL(op_desc);
      GetOrCreateAutoFuseAttrs(op_desc);
      op_desc = node2->GetOpDescBarePtr();
      GE_ASSERT_NOTNULL(op_desc);
      GetOrCreateAutoFuseAttrs(op_desc);
      NodeFuseInfo node_fuse_info;
      GE_ASSERT_SUCCESS(node_fuse_info.UpdateNodeFuseInfo(node1, node2));
      auto new_node = FuseNode(node1, node2, nullptr, node_fuse_info, counter);
      GE_ASSERT_NOTNULL(new_node);
      if (new_node != nullptr) {
        SetAttrsGroup(new_node);
      }
      return new_node;
    }
  };

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  std::string dump_file_name = "./ge_proto_AfterAssignedLogicStreams.txt";
  auto dump_file = getenv("GE_MT_DUMP_FILE");
  if (dump_file != nullptr) {
    dump_file_name = dump_file;
  }
  if (dump_file_name.find(".txt") == std::string::npos) {
    return;
  }
  bool load_success = GraphUtils::LoadGEGraph(dump_file_name.c_str(), graph);
  if(load_success) {
    for (const auto &node : graph->GetAllNodes()) {
      SetAttrsGroup(node);
    }
    FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new MockFusionDecider()));
    FusionStrategySolver  fusion_strategy_solver;
    GEEVENT("Before fuse all nodes size:%zu", graph->GetAllNodesSize());
    EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
    GEEVENT("After fuse all nodes size:%zu", graph->GetAllNodesSize());
  }
}

} // namespace ge
