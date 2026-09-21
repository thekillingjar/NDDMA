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
#include <iostream>
#include "test_structs.h"
#include "func_counter.h"
#include "graph/anchor.h"
#include "graph/node.h"
#include "graph_builder_utils.h"
#include "utils/graph_utils.h"

namespace ge {
namespace {
class SubInDataAnchor : public InDataAnchor
{
public:

  SubInDataAnchor(const NodePtr& owner_node, const int32_t idx);

  virtual ~SubInDataAnchor();

  bool EncaEq(const AnchorPtr anchor);
  bool EncaIsTypeOf(const TYPE type);
  void SetImpNull();
  template <class T>
  static Anchor::TYPE EncaTypeOf() {
    return Anchor::TypeOf<T>();
  }

};

SubInDataAnchor::SubInDataAnchor(const NodePtr &owner_node, const int32_t idx):InDataAnchor(owner_node,idx){}

SubInDataAnchor::~SubInDataAnchor() = default;

bool SubInDataAnchor::EncaEq(const AnchorPtr anchor){
  return Equal(anchor);
}

bool SubInDataAnchor::EncaIsTypeOf(const TYPE type){
  return IsTypeOf(type);
}

void SubInDataAnchor::SetImpNull(){
  impl_ = nullptr;
}

class SubOutDataAnchor : public OutDataAnchor
{
public:

  SubOutDataAnchor(const NodePtr& owner_node, const int32_t idx);

  virtual ~SubOutDataAnchor();

  bool EncaEq(const AnchorPtr anchor);
  bool EncaIsTypeOf(const TYPE type);
  void SetImpNull();

};

SubOutDataAnchor::SubOutDataAnchor(const NodePtr &owner_node, const int32_t idx):OutDataAnchor(owner_node,idx){}

SubOutDataAnchor::~SubOutDataAnchor() = default;

bool SubOutDataAnchor::EncaEq(const AnchorPtr anchor){
  return Equal(anchor);
}

bool SubOutDataAnchor::EncaIsTypeOf(const TYPE type){
  return IsTypeOf(type);
}

void SubOutDataAnchor::SetImpNull(){
  impl_ = nullptr;
}

class SubInControlAnchor : public InControlAnchor
{
public:

  SubInControlAnchor(const NodePtr &owner_node);
  SubInControlAnchor(const NodePtr& owner_node, const int32_t idx);

  virtual ~SubInControlAnchor();

  bool EncaEq(const AnchorPtr anchor);
  bool EncaIsTypeOf(const TYPE type);
  void SetImpNull();

};

SubInControlAnchor::SubInControlAnchor(const NodePtr &owner_node):InControlAnchor(owner_node){}
SubInControlAnchor::SubInControlAnchor(const NodePtr &owner_node, const int32_t idx):InControlAnchor(owner_node,idx){}

SubInControlAnchor::~SubInControlAnchor() = default;

bool SubInControlAnchor::EncaEq(const AnchorPtr anchor){
  return Equal(anchor);
}

bool SubInControlAnchor::EncaIsTypeOf(const TYPE type){
  return IsTypeOf(type);
}

void SubInControlAnchor::SetImpNull(){
  impl_ = nullptr;
}

class SubOutControlAnchor : public OutControlAnchor
{
public:

  SubOutControlAnchor(const NodePtr &owner_node);

  SubOutControlAnchor(const NodePtr& owner_node, const int32_t idx);

  virtual ~SubOutControlAnchor();

  bool EncaEq(const AnchorPtr anchor);
  bool EncaIsTypeOf(const TYPE type);
  void SetImpNull();

};

SubOutControlAnchor::SubOutControlAnchor(const NodePtr &owner_node):OutControlAnchor(owner_node){}
SubOutControlAnchor::SubOutControlAnchor(const NodePtr &owner_node, const int32_t idx):OutControlAnchor(owner_node,idx){}

SubOutControlAnchor::~SubOutControlAnchor() = default;

bool SubOutControlAnchor::EncaEq(const AnchorPtr anchor){
  return Equal(anchor);
}

bool SubOutControlAnchor::EncaIsTypeOf(const TYPE type){
  return IsTypeOf(type);
}

void SubOutControlAnchor::SetImpNull(){
  impl_ = nullptr;
}

}

using SubInDataAnchorPtr = std::shared_ptr<SubInDataAnchor>;
using SubOutDataAnchorPtr = std::shared_ptr<SubOutDataAnchor>;
using SubInControlAnchorPtr = std::shared_ptr<SubInControlAnchor>;
using SubOutControlAnchorPtr = std::shared_ptr<SubOutControlAnchor>;

class AnchorUt : public testing::Test {};

TEST_F(AnchorUt, SubInDataAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Data", "Data", 1, 1);
  SubInDataAnchorPtr in_anch = std::make_shared<SubInDataAnchor>(node, 111);
  in_anch->SetIdx(222);
  EXPECT_EQ(in_anch->GetIdx(), 222);
  EXPECT_EQ(in_anch->EncaEq(Anchor::DynamicAnchorCast<Anchor>(in_anch)), true);
  EXPECT_EQ(in_anch->GetPeerAnchorsSize(),0);
  EXPECT_EQ(in_anch->GetFirstPeerAnchor(),nullptr);
  EXPECT_EQ(in_anch->GetOwnerNode(), node);
  EXPECT_EQ(in_anch->GetOwnerNodeBarePtr(), node.get());
  EXPECT_EQ(in_anch->IsLinkedWith(nullptr), false);
  EXPECT_EQ(in_anch->GetPeerOutAnchor(), nullptr);
  EXPECT_EQ(in_anch->LinkFrom(nullptr), GRAPH_FAILED);
  auto node2 = builder.AddNode("Data", "Data", 2, 2);
  OutDataAnchorPtr peer = std::make_shared<OutDataAnchor>(node2, 22);
  EXPECT_EQ(in_anch->LinkFrom(peer), GRAPH_SUCCESS);
  EXPECT_EQ(in_anch->IsLinkedWith(peer), true);
  EXPECT_EQ(in_anch->GetPeerAnchorsSize(),1);
  EXPECT_EQ(in_anch->GetPeerAnchors().size(),1);
  EXPECT_NE(in_anch->GetFirstPeerAnchor(),nullptr);
  EXPECT_NE(in_anch->GetOwnerNode(),nullptr);
  auto node3 = builder.AddNode("Data", "Data", 3, 3);
  auto node4 = builder.AddNode("Data", "Data", 4, 4);
  OutDataAnchorPtr first = std::make_shared<OutDataAnchor>(node4, 44);
  SubInDataAnchorPtr second = std::make_shared<SubInDataAnchor>(node3, 33);
  EXPECT_EQ(in_anch->Insert(peer,first,second),GRAPH_SUCCESS);

  auto node5 = builder.AddNode("Data", "Data", 5, 5);
  OutDataAnchorPtr oa5 = std::make_shared<OutDataAnchor>(node5, 55);
  auto node6 = builder.AddNode("Data", "Data", 6, 6);
  SubInDataAnchorPtr ia6 = std::make_shared<SubInDataAnchor>(node, 66);
  EXPECT_EQ(ia6->LinkFrom(oa5), GRAPH_SUCCESS);
  EXPECT_EQ(ia6->Unlink(oa5), GRAPH_SUCCESS);

  EXPECT_EQ(in_anch->Unlink(nullptr),GRAPH_FAILED);
  EXPECT_EQ(in_anch->EncaEq(nullptr),false);
  EXPECT_EQ(in_anch->EncaIsTypeOf("nnn"),false);
  EXPECT_NE(in_anch->DynamicAnchorCast<InDataAnchor>(in_anch),nullptr);
  EXPECT_EQ(in_anch->DynamicAnchorCast<OutDataAnchor>(in_anch),nullptr);
}


TEST_F(AnchorUt, SubOutDataAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Data", "Data", 1, 1);
  auto attr = builder.AddNode("Attr", "Attr", 1, 1);
  SubOutDataAnchorPtr out_anch = std::make_shared<SubOutDataAnchor>(node, 111);
  out_anch->SetIdx(222);
  EXPECT_EQ(out_anch->EncaEq(Anchor::DynamicAnchorCast<Anchor>(out_anch)), true);
  EXPECT_EQ(out_anch->GetIdx(), 222);
  EXPECT_EQ(out_anch->GetPeerAnchorsSize(),0);
  EXPECT_EQ(out_anch->GetFirstPeerAnchor(),nullptr);
  EXPECT_EQ(out_anch->GetOwnerNode(),node);
  EXPECT_EQ(out_anch->IsLinkedWith(nullptr), false);
  auto node2 = builder.AddNode("Data", "Data", 2, 2);
  InDataAnchorPtr peer = std::make_shared<InDataAnchor>(node2, 22);
  EXPECT_EQ(out_anch->LinkTo(peer), GRAPH_SUCCESS);

  auto node3 = builder.AddNode("Data", "Data", 3, 3);
  InControlAnchorPtr peerctr = std::make_shared<InControlAnchor>(node3, 33);
  EXPECT_EQ(out_anch->LinkTo(peerctr), GRAPH_SUCCESS);
  EXPECT_EQ(peerctr->GetPeerOutDataAnchors().size(), 1);
  EXPECT_EQ(out_anch->GetPeerAnchorsSize(),2);
  EXPECT_EQ(out_anch->GetPeerAnchors().size(),2);
  EXPECT_NE(out_anch->GetFirstPeerAnchor(),nullptr);
  EXPECT_NE(out_anch->GetOwnerNode(),nullptr);
  auto node22 = builder.AddNode("Data", "Data", 22, 22);
  SubInDataAnchorPtr peerd2 = std::make_shared<SubInDataAnchor>(node2, 222);
  peerd2->SetImpNull();
  EXPECT_EQ(out_anch->LinkTo(peerd2), GRAPH_FAILED);
  auto node33 = builder.AddNode("Data", "Data", 33, 33);
  SubInControlAnchorPtr peerctr2 = std::make_shared<SubInControlAnchor>(node3, 333);
  peerctr2->SetImpNull();
  EXPECT_EQ(out_anch->LinkTo(peerctr2), GRAPH_FAILED);

  EXPECT_EQ(out_anch->Unlink(nullptr),GRAPH_FAILED);
  EXPECT_EQ(out_anch->EncaEq(nullptr),false);
  EXPECT_EQ(out_anch->EncaIsTypeOf("nnn"),false);
  out_anch->SetImpNull();
  auto nodelast = builder.AddNode("Data", "Data", 23, 23);
  SubInDataAnchorPtr peerd23 = std::make_shared<SubInDataAnchor>(nodelast, 223);
  EXPECT_EQ(out_anch->LinkTo(peerd23), GRAPH_FAILED);

  auto node4 = builder.AddNode("Data4", "Data", 1, 1);
  InDataAnchorPtr peerin2 = std::make_shared<InDataAnchor>(node4, 44);
  out_anch->impl_ = nullptr;
  EXPECT_EQ(out_anch->LinkTo(peerin2), GRAPH_FAILED);
}


TEST_F(AnchorUt, SubInControlAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node0 = builder.AddNode("Data", "Data", 111, 1);
  SubInControlAnchorPtr in_canch0 = std::make_shared<SubInControlAnchor>(node0);
  EXPECT_NE(in_canch0, nullptr);
  EXPECT_EQ(in_canch0->EncaEq(Anchor::DynamicAnchorCast<Anchor>(in_canch0)), true);
  auto node = builder.AddNode("Data", "Data", 1, 1);
  SubInControlAnchorPtr inc_anch = std::make_shared<SubInControlAnchor>(node, 111);
  inc_anch->SetIdx(222);
  EXPECT_EQ(inc_anch->GetIdx(), 222);
  EXPECT_EQ(inc_anch->GetPeerAnchorsSize(),0);
  EXPECT_EQ(inc_anch->GetFirstPeerAnchor(),nullptr);
  EXPECT_EQ(inc_anch->GetOwnerNode(),node);
  EXPECT_EQ(inc_anch->IsLinkedWith(nullptr), false);
  EXPECT_EQ(inc_anch->LinkFrom(nullptr), GRAPH_FAILED);
  auto node2 = builder.AddNode("Data", "Data", 2, 2);
  OutControlAnchorPtr peer = std::make_shared<OutControlAnchor>(node2, 22);
  EXPECT_EQ(inc_anch->LinkFrom(peer), GRAPH_SUCCESS);
  EXPECT_EQ(inc_anch->IsPeerOutAnchorsEmpty(),false);
  EXPECT_EQ(inc_anch->GetPeerAnchorsSize(),1);
  EXPECT_EQ(inc_anch->GetPeerAnchors().size(),1);
  EXPECT_EQ(inc_anch->GetPeerAnchorsPtr().size(),1);
  EXPECT_EQ(inc_anch->GetPeerOutDataAnchors().size(), 0);
  EXPECT_NE(inc_anch->GetFirstPeerAnchor(),nullptr);
  EXPECT_NE(inc_anch->GetOwnerNode(),nullptr);
  auto node3 = builder.AddNode("Data", "Data", 3, 3);
  SubInControlAnchorPtr second = std::make_shared<SubInControlAnchor>(node3, 33);
  auto node4 = builder.AddNode("Data", "Data", 4, 4);
  OutControlAnchorPtr first = std::make_shared<OutControlAnchor>(node4, 44);
  EXPECT_EQ(inc_anch->Insert(peer,first,second),GRAPH_SUCCESS);
  EXPECT_EQ(inc_anch->Unlink(nullptr),GRAPH_FAILED);
  EXPECT_EQ(inc_anch->EncaEq(nullptr),false);
  EXPECT_EQ(inc_anch->EncaIsTypeOf("nnn"),false);
  inc_anch->UnlinkAll();
  auto node24 = builder.AddNode("Data24", "Data", 2, 2);
  OutControlAnchorPtr peer24 = std::make_shared<OutControlAnchor>(node24, 24);
  inc_anch->impl_ = nullptr;
  EXPECT_EQ(inc_anch->LinkFrom(peer24), GRAPH_FAILED);
}


TEST_F(AnchorUt, SubOutControlAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node0 = builder.AddNode("Data", "Data", 111, 1);
  SubOutControlAnchorPtr out_canch0 = std::make_shared<SubOutControlAnchor>(node0);
  EXPECT_NE(out_canch0, nullptr);
  EXPECT_EQ(out_canch0->EncaEq(Anchor::DynamicAnchorCast<Anchor>(out_canch0)), true);
  auto node = builder.AddNode("Data", "Data", 1, 1);
  SubOutControlAnchorPtr outc_anch = std::make_shared<SubOutControlAnchor>(node, 111);
  outc_anch->SetIdx(222);
  EXPECT_EQ(outc_anch->GetIdx(), 222);
  EXPECT_EQ(outc_anch->GetPeerAnchorsSize(),0);
  EXPECT_EQ(outc_anch->GetFirstPeerAnchor(),nullptr);
  EXPECT_EQ(outc_anch->GetOwnerNode(),node);
  EXPECT_EQ(outc_anch->IsLinkedWith(nullptr), false);
  auto node2 = builder.AddNode("Data", "Data", 2, 2);
  InDataAnchorPtr peer = std::make_shared<InDataAnchor>(node2, 22);
  EXPECT_EQ(outc_anch->LinkTo(peer), GRAPH_SUCCESS);
  auto node3 = builder.AddNode("Data", "Data", 3, 3);
  InControlAnchorPtr peerctr = std::make_shared<InControlAnchor>(node3, 33);
  EXPECT_EQ(outc_anch->LinkTo(peerctr), GRAPH_SUCCESS);
  EXPECT_EQ(outc_anch->GetPeerAnchorsSize(),2);
  EXPECT_EQ(outc_anch->GetPeerAnchors().size(),2);
  EXPECT_NE(outc_anch->GetFirstPeerAnchor(),nullptr);
  EXPECT_NE(outc_anch->GetOwnerNode(),nullptr);
  auto node22 = builder.AddNode("Data", "Data", 22, 22);
  SubInDataAnchorPtr peerd2 = std::make_shared<SubInDataAnchor>(node2, 222);
  peerd2->SetImpNull();
  EXPECT_EQ(outc_anch->LinkTo(peerd2), GRAPH_FAILED);
  auto node33 = builder.AddNode("Data", "Data", 33, 33);
  SubInControlAnchorPtr peerctr2 = std::make_shared<SubInControlAnchor>(node3, 333);
  peerctr2->SetImpNull();
  EXPECT_EQ(outc_anch->LinkTo(peerctr2), GRAPH_FAILED);
  EXPECT_NE(outc_anch->GetPeerInControlAnchors().size(), 0);
  EXPECT_NE(outc_anch->GetPeerInControlAnchorsPtr().size(), 0);
  EXPECT_EQ(outc_anch->GetPeerInControlAnchorsPtr().size(), outc_anch->GetPeerInControlAnchors().size());
  EXPECT_NE(outc_anch->GetPeerInDataAnchors().size(), 0);

  EXPECT_EQ(outc_anch->Unlink(nullptr),GRAPH_FAILED);
  EXPECT_EQ(outc_anch->EncaEq(nullptr),false);
  EXPECT_EQ(outc_anch->EncaIsTypeOf("nnn"),false);
  outc_anch->SetImpNull();
  auto nodelast = builder.AddNode("Data", "Data", 23, 23);
  SubInDataAnchorPtr peerd23 = std::make_shared<SubInDataAnchor>(nodelast, 223);
  EXPECT_EQ(outc_anch->LinkTo(peerd23), GRAPH_FAILED);

  auto node4 = builder.AddNode("Data4", "Data", 1, 1);
  InControlAnchorPtr peerctr4 = std::make_shared<InControlAnchor>(node4, 44);
  outc_anch->impl_ = nullptr;
  EXPECT_EQ(outc_anch->LinkTo(peerctr4), GRAPH_FAILED);
}

//   node1 is replaced by node3
//    node0           node0
//    /   \    -->    /   |
// node1  node2    node3  node2
TEST_F(AnchorUt, CheckReplacePeerOrder) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node0 = builder.AddNode("Data", "Data", 1, 1);
  auto node1 = builder.AddNode("Data", "Data", 1, 1);
  auto node2 = builder.AddNode("Data", "Data", 1, 1);
  auto node3 = builder.AddNode("Data", "Data", 1, 1);

  graphStatus ret = ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0U), node1->GetInDataAnchor(0));
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0U), node2->GetInDataAnchor(0));
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  ret = node0->GetOutDataAnchor(0U)->ReplacePeer(node1->GetInDataAnchor(0),
                                                 node3->GetInDataAnchor(0));
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  bool check_same = node0->GetOutDataAnchor(0)->GetFirstPeerAnchor()->Equal(node3->GetInDataAnchor(0));
  EXPECT_TRUE(check_same);
  EXPECT_TRUE(node1->GetInDataAnchor(0U)->GetPeerAnchors().empty());
}

//   node1 is replaced by node4
//      node0                  node0
//    /   |   \       -->    /  |   |
// node1 node2 node3    node4 node2 node3
TEST_F(AnchorUt, ControlAnchorReplacePeer) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node0 = builder.AddNode("Data", "Data", 1, 1);
  auto node1 = builder.AddNode("Data", "Data", 1, 1);
  auto node2 = builder.AddNode("Data", "Data", 1, 1);
  auto node3 = builder.AddNode("Data", "Data", 1, 1);
  auto node4 = builder.AddNode("Data", "Data", 1, 1);
  (void)ge::GraphUtils::AddEdge(node0->GetOutControlAnchor(), node1->GetInControlAnchor());
  (void)ge::GraphUtils::AddEdge(node0->GetOutControlAnchor(), node2->GetInControlAnchor());
  (void) ge::GraphUtils::AddEdge(node0->GetOutControlAnchor(), node3->GetInControlAnchor());
  EXPECT_EQ(node1->GetInControlAnchor()->GetPeerOutControlAnchors().size(),
            node1->GetInControlAnchor()->GetPeerOutControlAnchorsPtr().size());
  graphStatus ret = node0->GetOutControlAnchor()->ReplacePeer(node1->GetInControlAnchor(), node4->GetInControlAnchor());
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  bool check_same = node0->GetOutControlAnchor()->GetFirstPeerAnchor()->Equal(node4->GetInControlAnchor());
  EXPECT_TRUE(check_same);
  EXPECT_TRUE(node1->GetInControlAnchor()->GetPeerAnchors().empty());
}

TEST_F(AnchorUt, ReplacePeerDifferentTypeFailed) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node0 = builder.AddNode("Data", "Data", 1, 1);
  auto node1 = builder.AddNode("Data", "Data", 1, 1);
  auto node2 = builder.AddNode("Data", "Data", 1, 1);

  graphStatus ret = ge::GraphUtils::AddEdge(node0->GetOutControlAnchor(), node1->GetInControlAnchor());
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = node0->GetOutControlAnchor()->ReplacePeer(node1->GetInControlAnchor(),
                                                  node2->GetInDataAnchor(0U));
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = node0->GetOutControlAnchor()->ReplacePeer(node1->GetInDataAnchor(0U),
                                                  node2->GetInDataAnchor(0U));
  EXPECT_EQ(ret, GRAPH_FAILED);
}

//   node0 is replaced by node2
//    node0           node2
//    /       -->    /
// node1         node1
TEST_F(AnchorUt, ReplacePeerOfOutDataAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node0 = builder.AddNode("Data", "Data", 1, 1);
  auto node1 = builder.AddNode("Data", "Data", 1, 1);
  auto node2 = builder.AddNode("Data", "Data", 1, 1);

  graphStatus ret = ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0U), node1->GetInDataAnchor(0));
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(node0->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(),
            node0->GetOutDataAnchor(0)->GetPeerInDataAnchorsPtr().size());
  ret = node1->GetInDataAnchor(0U)->ReplacePeer(node0->GetOutAnchor(0U), node2->GetOutAnchor(0U));
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  bool check_same = node1->GetInDataAnchor(0U)->GetFirstPeerAnchor()->Equal(node2->GetOutAnchor(0U));
  EXPECT_TRUE(check_same);
}

TEST_F(AnchorUt, CheckReplaceNewAnchorPeerNotEmpty) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node0 = builder.AddNode("Data", "Data", 1, 1);
  auto node1 = builder.AddNode("Data", "Data", 1, 1);
  auto node2 = builder.AddNode("Data", "Data", 1, 1);
  auto node3 = builder.AddNode("Data", "Data", 1, 1);

  graphStatus ret = ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0U), node1->GetInDataAnchor(0));
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = ge::GraphUtils::AddEdge(node2->GetOutDataAnchor(0U), node3->GetInDataAnchor(0));
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  ret = node0->GetOutDataAnchor(0U)->ReplacePeer(node1->GetInDataAnchor(0),
                                                 node3->GetInDataAnchor(0));
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(AnchorUt, InsertNotEmptyNodeFailed) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node0 = builder.AddNode("Data", "Data", 1, 1);
  auto node1 = builder.AddNode("Data", "Data", 1, 1);
  auto node2 = builder.AddNode("Data", "Data", 1, 1);
  auto node3 = builder.AddNode("Data", "Data", 1, 1);
  auto node4 = builder.AddNode("Data", "Data", 1, 1);

  graphStatus ret = ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0U), node2->GetInDataAnchor(0));
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = ge::GraphUtils::AddEdge(node2->GetOutDataAnchor(0U), node1->GetInDataAnchor(0));
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = ge::GraphUtils::AddEdge(node1->GetOutDataAnchor(0U), node4->GetInDataAnchor(0));
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  ret = node0->GetOutDataAnchor(0U)->Insert(node2->GetInDataAnchor(0),
                                            node1->GetInDataAnchor(0),
                                            node1->GetOutDataAnchor(0));
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(AnchorUt, InsertDifferentAnchorFailed) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node0 = builder.AddNode("Data", "Data", 1, 1);
  auto node1 = builder.AddNode("Data", "Data", 1, 1);
  auto node2 = builder.AddNode("Data", "Data", 1, 1);

  graphStatus ret = ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0U), node2->GetInDataAnchor(0));
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  ret = node0->GetOutDataAnchor(0U)->Insert(node2->GetInDataAnchor(0),
                                            node1->GetInControlAnchor(),
                                            node1->GetOutDataAnchor(0));
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = node0->GetOutDataAnchor(0U)->Insert(node2->GetInControlAnchor(),
                                            node1->GetInDataAnchor(0),
                                            node1->GetOutDataAnchor(0));
  EXPECT_EQ(ret, GRAPH_FAILED);
}

//    node0 -- node2
//       插入新节点
// node0--node1-- node2
TEST_F(AnchorUt, InsertSuccess) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node0 = builder.AddNode("Data", "Data", 1, 1);
  auto node1 = builder.AddNode("Data", "Data", 1, 1);
  auto node2 = builder.AddNode("Data", "Data", 1, 1);

  graphStatus ret = ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0U), node2->GetInDataAnchor(0U));
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  ret = node0->GetOutDataAnchor(0U)->Insert(node2->GetInDataAnchor(0U),
                                            node1->GetInDataAnchor(0U),
                                            node1->GetOutDataAnchor(0U));
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  bool the_same = node0->GetOutDataAnchor(0U)->GetFirstPeerAnchor()->Equal(node1->GetInDataAnchor(0U));
  EXPECT_TRUE(the_same);
  the_same = node2->GetInDataAnchor(0U)->GetFirstPeerAnchor()->Equal(node1->GetOutDataAnchor(0U));
  EXPECT_TRUE(the_same);

  auto node00 = builder.AddNode("Data", "Data", 1, 1);
  auto node11 = builder.AddNode("Data", "Data", 1, 1);
  auto node22 = builder.AddNode("Data", "Data", 1, 1);
  ret = ge::GraphUtils::AddEdge(node00->GetOutDataAnchor(0U), node22->GetInDataAnchor(0U));
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = node22->GetInDataAnchor(0U)->Insert(node00->GetOutDataAnchor(0U),
                                            node11->GetOutDataAnchor(0),
                                            node11->GetInDataAnchor(0));
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  the_same = node00->GetOutDataAnchor(0U)->GetFirstPeerAnchor()->Equal(node11->GetInDataAnchor(0U));
  EXPECT_TRUE(the_same);
  the_same = node22->GetInDataAnchor(0U)->GetFirstPeerAnchor()->Equal(node11->GetOutDataAnchor(0U));
  EXPECT_TRUE(the_same);
}
}  // namespace ge
