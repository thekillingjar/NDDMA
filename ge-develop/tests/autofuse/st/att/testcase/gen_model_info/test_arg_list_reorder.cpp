/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#define private public
#include "expr_gen/arg_list_reorder.h"


namespace att{
class TestArgListReorder : public ::testing::Test {
 public:
  static void TearDownTestCase()
  {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase()
  {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
     // Code here will be called immediately after the constructor (right
     // before each test).
  }

  void TearDown() override {
     // Code here will be called immediately after each test (right
     // before the destructor).
  }
};

TEST_F(TestArgListReorder, case0)
{
  //Define TuningSpace
  //Create node: MatMul
  //input : [m, k][k, n]
  //repeat : [M, K][K, N]
  //stride : [MM, KK][KK, NN]
  //output : [m, n, k]
  //repeat : [M, N, ONE]
  //stride : [MM, NN, ZERO]
  NodeInfo MatMul;
  Tensor Tensor00;
  std::shared_ptr<Tensor> tensor00 = std::make_shared<Tensor>(Tensor00);
  Tensor Tensor01;
  std::shared_ptr<Tensor> tensor01 = std::make_shared<Tensor>(Tensor01);
  Tensor Tensor02;
  std::shared_ptr<Tensor> tensor02 = std::make_shared<Tensor>(Tensor02);
  SubAxis subaxis_m;
  auto m = std::make_unique<SubAxis>(subaxis_m);
  SubAxis subaxis_k;
  auto k = std::make_unique<SubAxis>(subaxis_k);
  SubAxis subaxis_n;
  auto n = std::make_unique<SubAxis>(subaxis_n);
  auto M = ge::Symbol("M");
  auto K = ge::Symbol("K");
  auto N = ge::Symbol("N");
  auto MM = ge::Symbol("MM");
  auto KK = ge::Symbol("KK");
  auto NN = ge::Symbol("NN");
  auto ONE = ge::Symbol(1, "ONE");
  auto ZERO = ge::Symbol(0, "ZERO");
  m->name = "m";
  k->name = "k";
  n->name = "n";
  n->is_node_innerest_dim = true;
  tensor00->name = "MatMul_input_0";
  tensor00->dim_info = {m.get(), k.get()};
  tensor00->repeat = {M, K};
  tensor00->stride = {MM, KK};
  tensor01->name = "MatMul_input_1";
  tensor01->dim_info = {k.get(), n.get()};
  tensor01->repeat = {K, N};
  tensor01->stride = {KK, NN};
  tensor02->name = "MatMul_output_0";
  tensor02->dim_info = {m.get(), n.get(), k.get()};
  tensor02->repeat = {M, N, ONE};
  tensor02->stride = {MM, NN, ZERO};
  MatMul.name = "MatMul";
  MatMul.node_type = "MatMul";
  MatMul.inputs = {tensor00, tensor01};
  MatMul.outputs = {tensor02};

  //Create node: Load
  //input : [a, b]
  //repeat : [A, B]
  //stride : [AA, BB]
  //output : [a, b]
  //repeat : [A, B]
  //stride : [AA, BB]
  NodeInfo Load;
  Tensor Tensor10;
  std::shared_ptr<Tensor> tensor10 = std::make_shared<Tensor>(Tensor10);
  Tensor Tensor11;
  std::shared_ptr<Tensor> tensor11 = std::make_shared<Tensor>(Tensor11);
  SubAxis subaxis_a;
  auto a = std::make_unique<SubAxis>(subaxis_a);
  SubAxis subaxis_b;
  auto b = std::make_unique<SubAxis>(subaxis_b);
  auto A = ge::Symbol("A");
  auto B = ge::Symbol("B");
  auto AA = ge::Symbol("AA");
  auto BB = ge::Symbol("BB");
  a->name = "a";
  b->name = "b";
  tensor10->name = "Load_input";
  tensor10->dim_info = {a.get(), b.get()};
  tensor10->repeat = {A, B};
  tensor10->stride = {AA, BB};
  tensor11->name = "Load_output";
  tensor11->dim_info = {a.get(), b.get()};
  tensor11->repeat = {A, B};
  tensor11->stride = {AA, BB};
  Load.name = "Load";
  Load.node_type = "Load";
  Load.inputs = {tensor10};
  Load.outputs = {tensor11};

  auto tuning_space_ = std::make_shared<TuningSpace>();
  tuning_space_->node_infos = {MatMul, Load};
  tuning_space_->sub_axes.emplace_back(std::move(m));
  tuning_space_->sub_axes.emplace_back(std::move(k));
  tuning_space_->sub_axes.emplace_back(std::move(n));
  tuning_space_->sub_axes.emplace_back(std::move(a));
  tuning_space_->sub_axes.emplace_back(std::move(b));
  //End Define

  //Define Modelinfo
  ModelInfo model_info;
  AttAxis att_m;
  std::shared_ptr<AttAxis> attaxis_m = std::make_shared<AttAxis>(att_m);
  AttAxis att_k;
  std::shared_ptr<AttAxis> attaxis_k = std::make_shared<AttAxis>(att_k);
  AttAxis att_n;
  std::shared_ptr<AttAxis> attaxis_n = std::make_shared<AttAxis>(att_n);
  AttAxis att_a;
  std::shared_ptr<AttAxis> attaxis_a = std::make_shared<AttAxis>(att_a);
  AttAxis att_b;
  std::shared_ptr<AttAxis> attaxis_b = std::make_shared<AttAxis>(att_b);
  attaxis_m->name = "m";
  attaxis_k->name = "k";
  attaxis_n->name = "n";
  attaxis_a->name = "a";
  attaxis_b->name = "b";
  model_info.arg_list = {attaxis_m, attaxis_k, attaxis_n, attaxis_a, attaxis_b};
  //End Define

  ArgListReorder arg_list_reorder(tuning_space_);
  std::vector<AttAxisPtr> tiling_R_arg_list;
  EXPECT_EQ(arg_list_reorder.SortArgList(model_info.arg_list, tiling_R_arg_list), ge::SUCCESS);
  std::map<std::string, size_t> arg_id_map;
  for (size_t i=0; i < model_info.arg_list.size(); i++) {
    auto arg = model_info.arg_list[i];
    arg_id_map[arg->name] = i;
  }
  EXPECT_EQ(arg_id_map["k"], 0);
  EXPECT_EQ(arg_id_map["n"], 1);
}

TEST_F(TestArgListReorder, case1)
{
  NodeInfo node1;
  std::shared_ptr<Tensor> tensor0 = std::make_shared<Tensor>();
  std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>();

  auto z0 = std::make_unique<SubAxis>();
  auto z1 = std::make_unique<SubAxis>();
  auto z2 = std::make_unique<SubAxis>();
  auto z0z1 = std::make_unique<SubAxis>();
  auto z0z1t = std::make_unique<SubAxis>();
  auto z2t = std::make_unique<SubAxis>();
    
  z0->name = "z0";
  z1->name = "z1";
  z2->name = "z2";
  z0z1->name = "z0z1";
  z0z1t->name = "z0z1t";
  z2t->name = "z2t";

  z0z1t->is_node_innerest_dim = true;
  tensor0->name = "node1_input0";
  tensor0->dim_info = {z0z1t.get(), z2t.get()};
  tensor0->repeat = {ge::Symbol("z0z1t_size"), ge::Symbol("z2t_size")};
  tensor0->stride = {ge::Symbol("z2t_size"), ge::Symbol(1, "ONE")};
  tensor1->name = "node1_output0";
  tensor1->dim_info = {z0z1t.get(), z2t.get()};
  tensor1->repeat = {ge::Symbol(1, "ONE"), ge::Symbol("z2t_size")};
  tensor1->stride = {ge::Symbol(0, "ZERO"), ge::Symbol(1, "ONE")};
  node1.name = "Node1";
  node1.node_type = "LayerNorm";
  node1.inputs = {tensor0};
  node1.outputs = {tensor1};



  auto tuning_space_ = std::make_shared<TuningSpace>();
  tuning_space_->node_infos = {node1};
  tuning_space_->sub_axes.emplace_back(std::move(z0));
  tuning_space_->sub_axes.emplace_back(std::move(z1));
  tuning_space_->sub_axes.emplace_back(std::move(z2));
  tuning_space_->sub_axes.emplace_back(std::move(z0z1));
  tuning_space_->sub_axes.emplace_back(std::move(z0z1t));
  tuning_space_->sub_axes.emplace_back(std::move(z2t));
  //End Define

  //Define Modelinfo
  ModelInfo model_info;
  std::shared_ptr<AttAxis> att_z0 = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z1 = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z2 = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z0z1 = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z0z1t = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z2t = std::make_shared<AttAxis>();
  
  att_z0->name = "z0";
  att_z1->name = "z1";
  att_z2->name = "z2";
  att_z0z1->name = "z0z1";
  att_z0z1t->name = "z0z1t";
  att_z2t->name = "z2t";

  att_z0z1->from_axis.emplace_back(att_z0.get());
  att_z0z1->from_axis.emplace_back(att_z1.get());
  att_z0z1t->from_axis.emplace_back(att_z0z1.get());
  att_z2t->from_axis.emplace_back(att_z2.get());

  model_info.arg_list = {att_z0, att_z1, att_z2, att_z0z1, att_z0z1t, att_z2t};
  //End Define

  ArgListReorder arg_list_reorder(tuning_space_);
  std::vector<AttAxisPtr> tiling_R_arg_list;
  EXPECT_EQ(arg_list_reorder.SortArgList(model_info.arg_list, tiling_R_arg_list), ge::SUCCESS);
  std::map<std::string, size_t> arg_id_map;
  for (size_t i=0; i < model_info.arg_list.size(); i++) {
    auto arg = model_info.arg_list[i];
    arg_id_map[arg->name] = i;
  }
  EXPECT_EQ(arg_id_map["z0z1t"] < arg_id_map["z2t"], true);
}

TEST_F(TestArgListReorder, case2)
{
  NodeInfo node1;
  std::shared_ptr<Tensor> tensor0 = std::make_shared<Tensor>();
  std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>();

  auto z0 = std::make_unique<SubAxis>();
  auto z1 = std::make_unique<SubAxis>();
  auto z2 = std::make_unique<SubAxis>();
  auto z0z1 = std::make_unique<SubAxis>();
  auto z0z1t = std::make_unique<SubAxis>();
  auto z2t = std::make_unique<SubAxis>();
    
  z0->name = "z0";
  z1->name = "z1";
  z2->name = "z2";
  z0z1->name = "z0z1";
  z0z1t->name = "z0z1t";
  z2t->name = "z2t";

  z0z1t->is_node_innerest_dim = true;
  tensor0->name = "node1_output0";
  tensor0->dim_info = {z0z1t.get(), z2t.get()};
  tensor0->repeat = {ge::Symbol("z0z1t_size"), ge::Symbol("z2t_size")};
  tensor0->stride = {ge::Symbol("z2t_size"), ge::Symbol(1, "ONE")};
  tensor1->name = "node1_input0";
  tensor1->dim_info = {z0z1t.get(), z2t.get()};
  tensor1->repeat = {ge::Symbol(1, "ONE"), ge::Symbol("z2t_size")};
  tensor1->stride = {ge::Symbol(0, "ZERO"), ge::Symbol(1, "ONE")};
  node1.name = "Node1";
  node1.node_type = "LayerNorm";
  node1.inputs = {tensor1};
  node1.outputs = {tensor0};



  auto tuning_space_ = std::make_shared<TuningSpace>();
  tuning_space_->node_infos = {node1};
  tuning_space_->sub_axes.emplace_back(std::move(z0));
  tuning_space_->sub_axes.emplace_back(std::move(z1));
  tuning_space_->sub_axes.emplace_back(std::move(z2));
  tuning_space_->sub_axes.emplace_back(std::move(z0z1));
  tuning_space_->sub_axes.emplace_back(std::move(z0z1t));
  tuning_space_->sub_axes.emplace_back(std::move(z2t));
  //End Define

  //Define Modelinfo
  ModelInfo model_info;
  std::shared_ptr<AttAxis> att_z0 = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z1 = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z2 = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z0z1 = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z0z1t = std::make_shared<AttAxis>();
  std::shared_ptr<AttAxis> att_z2t = std::make_shared<AttAxis>();
  
  att_z0->name = "z0";
  att_z1->name = "z1";
  att_z2->name = "z2";
  att_z0z1->name = "z0z1";
  att_z0z1t->name = "z0z1t";
  att_z2t->name = "z2t";

  att_z0z1->from_axis.emplace_back(att_z0.get());
  att_z0z1->from_axis.emplace_back(att_z1.get());
  att_z0z1t->from_axis.emplace_back(att_z0z1.get());
  att_z2t->from_axis.emplace_back(att_z2.get());

  model_info.arg_list = {att_z0, att_z1, att_z2, att_z0z1, att_z0z1t, att_z2t};
  //End Define

  ArgListReorder arg_list_reorder(tuning_space_);
  std::vector<AttAxisPtr> tiling_R_arg_list;
  EXPECT_EQ(arg_list_reorder.SortArgList(model_info.arg_list, tiling_R_arg_list), ge::SUCCESS);
  std::map<std::string, size_t> arg_id_map;
  for (size_t i=0; i < model_info.arg_list.size(); i++) {
    auto arg = model_info.arg_list[i];
    arg_id_map[arg->name] = i;
  }
  EXPECT_EQ(arg_id_map["z0z1t"] < arg_id_map["z2t"], true);
}
} // namespace att


