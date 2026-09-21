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
#include "proto/ge_ir.pb.h"
#include "graph/attr_store.h"
#include "graph/symbolizer/guard_dfx_context.h"

#include "attribute_group/attr_group_shape_env.h"

#include <util/mem_utils.h>
#include "source_stub.h"
namespace ge {
namespace {
class AttributeGroupShapeEnvUt : public testing::Test {};

TEST_F(AttributeGroupShapeEnvUt, ShapeEnvAttrDeserialize_Succ) {
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(true, DynamicMode::kDynamic));
  Symbol s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(1, 2));
  Symbol s1 = shape_env.CreateSymbol(5, MakeShared<GraphInputShapeSourceStub>(2, 0));

  auto guard_1 = sym::Eq(s1 + Symbol(2), s0);
  auto guard_2 = sym::Gt(s0 + Symbol(1), s1);
  shape_env.AppendSymbolCheckInfo(guard_1);
  shape_env.AppendSymbolAssertInfo(guard_2);
  shape_env.AppendReplacement(s1 + Symbol(2), s0);
  proto::AttrGroupDef attr_group_def;
  auto ret = shape_env.Serialize(attr_group_def);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  auto shape_env1 = ShapeEnvAttr();
  ret = shape_env1.Deserialize(attr_group_def, nullptr);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(2, shape_env1.symbol_to_value_.size());
  EXPECT_EQ(2, shape_env1.value_to_symbol_.size());
  EXPECT_EQ(1, shape_env1.GetAllSymbolCheckInfos().size());
  EXPECT_EQ(1, shape_env1.GetAllSymbolAssertInfos().size());
  EXPECT_EQ(Symbol("s0"), shape_env1.value_to_symbol_[2][0]);
  EXPECT_EQ(Symbol("s1"), shape_env1.value_to_symbol_[5][0]);
  EXPECT_EQ(true, shape_env1.HasSymbolCheckInfo(sym::Eq(Symbol("s1") + Symbol(2), Symbol("s0"))));
  EXPECT_EQ(true, shape_env1.HasSymbolAssertInfo(sym::Gt(Symbol("s0") + Symbol(1), Symbol("s1"))));
}

TEST_F(AttributeGroupShapeEnvUt, CreateSymbolDuckMode_Succ) {
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDuck));
  Symbol sym = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 2));
  EXPECT_EQ(1, shape_env.symbol_to_value_.size());
  EXPECT_EQ(1, shape_env.value_to_symbol_.size());
  EXPECT_EQ(1, shape_env.symbol_to_source_.size());
  EXPECT_EQ(sym, shape_env.value_to_symbol_[2][0]);
  EXPECT_EQ(2, shape_env.symbol_to_value_[sym]);
  EXPECT_EQ(0, shape_env.symbol_to_source_[sym]->GetGlobalIndex());
  EXPECT_EQ(std::string(sym.Serialize().get()), "s0");
  Symbol sym1 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(1, 0));
  EXPECT_EQ(sym, sym1);
}

TEST_F(AttributeGroupShapeEnvUt, CreateSymbolSpecializeZeroOne_Succ) {
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(true, DynamicMode::kDynamic));
  Symbol sym = shape_env.CreateSymbol(1, MakeShared<GraphInputShapeSourceStub>(0, 2));
  EXPECT_EQ(0, shape_env.symbol_to_value_.size());
  EXPECT_EQ(0, shape_env.value_to_symbol_.size());
  EXPECT_EQ(0, shape_env.symbol_to_source_.size());
  EXPECT_EQ(std::string(sym.Serialize().get()), "1");
  Symbol sym1 = shape_env.CreateSymbol(0, MakeShared<GraphInputShapeSourceStub>(0, 3));
  EXPECT_EQ(0, shape_env.symbol_to_value_.size());
  EXPECT_EQ(0, shape_env.value_to_symbol_.size());
  EXPECT_EQ(0, shape_env.symbol_to_source_.size());
  EXPECT_EQ(std::string(sym1.Serialize().get()), "0");
}

TEST_F(AttributeGroupShapeEnvUt, CreateSymbolDynamicMode_Succ) {
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  Symbol sym = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto symbol_relation = shape_env.GetAllSym2Src();
  EXPECT_EQ(symbol_relation.size(), 1);
  EXPECT_EQ(symbol_relation[0].first, sym);
  EXPECT_EQ(symbol_relation[0].second->GetSourceStr(), MakeShared<GraphInputShapeSourceStub>(0, 2)->GetSourceStr());
  EXPECT_EQ(1, shape_env.symbol_to_value_.size());
  EXPECT_EQ(1, shape_env.value_to_symbol_.size());
  EXPECT_EQ(1, shape_env.symbol_to_source_.size());
  EXPECT_EQ(1, shape_env.value_to_symbol_[2].size());
  EXPECT_EQ(sym, shape_env.value_to_symbol_[2][0]);
  EXPECT_EQ(2, shape_env.symbol_to_value_[sym]);
  EXPECT_EQ(0, shape_env.symbol_to_source_[sym]->GetGlobalIndex());
  EXPECT_EQ(std::string(sym.Serialize().get()), "s0");
  Symbol sym1 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(1, 0));
  EXPECT_EQ(2, shape_env.symbol_to_value_.size());
  EXPECT_EQ(1, shape_env.value_to_symbol_.size());
  EXPECT_EQ(2, shape_env.symbol_to_source_.size());
  EXPECT_EQ(2, shape_env.value_to_symbol_[2].size());
  EXPECT_EQ(sym, shape_env.value_to_symbol_[2][0]);
  EXPECT_EQ(sym1, shape_env.value_to_symbol_[2][1]);
  EXPECT_EQ(2, shape_env.symbol_to_value_[sym1]);
  EXPECT_EQ(1, shape_env.symbol_to_source_[sym1]->GetGlobalIndex());
  EXPECT_EQ(std::string(sym1.Serialize().get()), "s1");
}

TEST_F(AttributeGroupShapeEnvUt, CreateSymbolStaticMode_Succ) {
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kStatic));
  SetCurShapeEnvContext(&shape_env);
  Symbol sym = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 2));
  EXPECT_EQ(1, shape_env.symbol_to_value_.size());
  EXPECT_EQ(1, shape_env.value_to_symbol_.size());
  EXPECT_EQ(1, shape_env.symbol_to_source_.size());
  EXPECT_EQ(1, shape_env.value_to_symbol_[2].size());
  EXPECT_EQ(sym, shape_env.value_to_symbol_[2][0]);
  EXPECT_EQ(2, shape_env.symbol_to_value_[sym]);
  EXPECT_EQ(0, shape_env.symbol_to_source_[sym]->GetGlobalIndex());
  EXPECT_EQ(std::string(sym.Serialize().get()), "s0");
  Symbol sym1 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 2));
  EXPECT_EQ(sym, sym1);
  EXPECT_EQ(true, shape_env.HasSymbolAssertInfo(sym::Eq(sym, Symbol(2))));
  SetCurShapeEnvContext(nullptr);
}

TEST_F(AttributeGroupShapeEnvUt, GetInputShapeSourceStr) {
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  Symbol sym_shape = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 2));
  Symbol sym_value = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(1, 2));

  auto shape_str = shape_env.GetAllSym2Src();
  EXPECT_EQ(shape_str.size(), 2);
  EXPECT_EQ(shape_str[1].second->GetSourceStr(), R"([&]() -> int64_t {
      const auto *tensor = context->GetGraphInputTensor(0);
      if (tensor == nullptr) {
        return -1;
      }
      return tensor->GetOriginShape().GetDim(2);
    }())");
  EXPECT_EQ(shape_str[0].second->GetSourceStr(), R"([&]() -> int64_t {
      const auto *tensor = context->GetGraphInputTensor(1);
      if (tensor == nullptr) {
        return -1;
      }
      return tensor->GetOriginShape().GetDim(2);
    }())");
  EXPECT_EQ(shape_str[0].second->GetGlobalIndex(), 1);
  EXPECT_EQ(shape_str[1].second->GetGlobalIndex(), 0);
  EXPECT_EQ(shape_str[0].second->GetGlobalIndexStr(), "context->GetInputPointer<int64_t>(1)");
  EXPECT_EQ(shape_str[1].second->GetGlobalIndexStr(), "context->GetInputPointer<int64_t>(0)");
}

TEST_F(AttributeGroupShapeEnvUt, GetSourceStrNew) {
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  auto source_shape = ge::MakeShared<InputValueSumSourceStub>(0, ge::DT_INT32);
  Symbol sym_shape = shape_env.CreateSymbol(2, source_shape);

  auto source_value = ge::MakeShared<InputValueSumSourceStub>(2, ge::DT_INT32);
  Symbol sym_value = shape_env.CreateSymbol(3, source_value);

  auto source_value2 = ge::MakeShared<InputValueSumSourceStub>(4, ge::DT_INT64);
  Symbol sym_value2 = shape_env.CreateSymbol(4, source_value2);

  auto source_value3 = ge::MakeShared<InputValueSumSourceStub>(6, ge::DT_UINT32);
  Symbol sym_value3 = shape_env.CreateSymbol(5, source_value3);

  auto source_value4 = ge::MakeShared<InputValueSumSourceStub>(8, ge::DT_UINT64);
  Symbol sym_value4 = shape_env.CreateSymbol(6, source_value4);

  auto source_value_unsupport = ge::MakeShared<InputValueSumSourceStub>(8, ge::DT_FLOAT);
  Symbol sym_unsupport = shape_env.CreateSymbol(7, source_value_unsupport);

  auto shape_str = shape_env.GetAllSym2Src();
  EXPECT_EQ(shape_str.size(), 6);
  EXPECT_EQ(shape_str[5].second->GetSourceStr(), R"([&]() -> int64_t {
              const auto* tensor = context->GetGraphInputTensor(0);
                if (tensor == nullptr) {
                  return -1;
                }
                const auto* data = tensor->GetData<int32_t>();
                int64_t sum = 0;
                for (size_t i = 0; i < tensor->GetSize() / sizeof(int32_t); ++i) {
                  sum += data[i];
                }
                return sum;
            }())");
  EXPECT_EQ(shape_str[4].second->GetSourceStr(), R"([&]() -> int64_t {
              const auto* tensor = context->GetGraphInputTensor(2);
                if (tensor == nullptr) {
                  return -1;
                }
                const auto* data = tensor->GetData<int32_t>();
                int64_t sum = 0;
                for (size_t i = 0; i < tensor->GetSize() / sizeof(int32_t); ++i) {
                  sum += data[i];
                }
                return sum;
            }())");
  EXPECT_EQ(shape_str[3].second->GetSourceStr(), R"([&]() -> int64_t {
              const auto* tensor = context->GetGraphInputTensor(4);
                if (tensor == nullptr) {
                  return -1;
                }
                const auto* data = tensor->GetData<int64_t>();
                int64_t sum = 0;
                for (size_t i = 0; i < tensor->GetSize() / sizeof(int64_t); ++i) {
                  sum += data[i];
                }
                return sum;
            }())");
  EXPECT_EQ(shape_str[2].second->GetSourceStr(), R"([&]() -> int64_t {
              const auto* tensor = context->GetGraphInputTensor(6);
                if (tensor == nullptr) {
                  return -1;
                }
                const auto* data = tensor->GetData<uint32_t>();
                int64_t sum = 0;
                for (size_t i = 0; i < tensor->GetSize() / sizeof(uint32_t); ++i) {
                  sum += data[i];
                }
                return sum;
            }())");
  EXPECT_EQ(shape_str[1].second->GetSourceStr(), R"([&]() -> int64_t {
              const auto* tensor = context->GetGraphInputTensor(8);
                if (tensor == nullptr) {
                  return -1;
                }
                const auto* data = tensor->GetData<uint64_t>();
                int64_t sum = 0;
                for (size_t i = 0; i < tensor->GetSize() / sizeof(uint64_t); ++i) {
                  sum += data[i];
                }
                return sum;
            }())");
  EXPECT_EQ(shape_str[0].second->GetSourceStr(), "");
}

TEST_F(AttributeGroupShapeEnvUt, ShapeEnvAttrClone_Succ) {
  auto s = AttrStore::Create(1);
  auto shape_env = s.GetOrCreateAttrsGroup<ShapeEnvAttr>();
  EXPECT_NE(shape_env, nullptr);
  SetCurShapeEnvContext(shape_env);
  shape_env->shape_env_setting_.specialize_zero_one = true;
  shape_env->shape_env_setting_.dynamic_mode = DynamicMode::kDynamic;
  auto s0 = shape_env->CreateSymbol(10, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto s1 = shape_env->CreateSymbol(10, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto s2 = shape_env->CreateSymbol(20, MakeShared<GraphInputShapeSourceStub>(0, 2));
  EXPECT_SYMBOL_EQ(s0 + s1, s2);

  // 测试Clone
  auto s_bak = s;
  auto shape_env_bak = s_bak.GetAttrsGroup<ShapeEnvAttr>();
  EXPECT_NE(shape_env_bak, nullptr);
  EXPECT_EQ(shape_env_bak->shape_env_setting_.specialize_zero_one, true);
  EXPECT_EQ(shape_env_bak->shape_env_setting_.dynamic_mode, DynamicMode::kDynamic);
  EXPECT_EQ(shape_env_bak->replacements_.size(), 2);
  EXPECT_EQ(shape_env_bak->symbol_to_value_.size(), 3);
  EXPECT_EQ(shape_env_bak->symbol_to_value_, shape_env->symbol_to_value_);
  EXPECT_EQ(shape_env_bak->symbol_to_source_.size(), 3);
  EXPECT_EQ(shape_env_bak->symbol_to_source_, shape_env->symbol_to_source_);
  EXPECT_EQ(shape_env_bak->value_to_symbol_.size(), 2);
  EXPECT_EQ(shape_env_bak->value_to_symbol_, shape_env->value_to_symbol_);
  EXPECT_EQ(shape_env_bak->symbol_check_infos_.size(), 1);
  EXPECT_EQ(shape_env_bak->symbol_check_infos_, shape_env->symbol_check_infos_);
  EXPECT_EQ(shape_env_bak->symbol_assert_infos_.size(),0);
  EXPECT_EQ(shape_env_bak->symbol_assert_infos_, shape_env->symbol_assert_infos_);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(AttributeGroupShapeEnvUt, ShapeEnvAttrCopy_Succ) {
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(true, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(10, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto s1 = shape_env.CreateSymbol(10, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto s2 = shape_env.CreateSymbol(20, MakeShared<GraphInputShapeSourceStub>(0, 2));
  EXPECT_SYMBOL_EQ(s0 + s1, s2);

  // 测试operator=
  auto shape_env_bak = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDuck));
  shape_env_bak = shape_env;
  EXPECT_EQ(shape_env_bak.shape_env_setting_.specialize_zero_one, true);
  EXPECT_EQ(shape_env_bak.shape_env_setting_.dynamic_mode, DynamicMode::kDynamic);
  EXPECT_EQ(shape_env_bak.replacements_.size(), 2);
  EXPECT_EQ(shape_env_bak.symbol_to_value_.size(), 3);
  EXPECT_EQ(shape_env_bak.symbol_to_value_, shape_env.symbol_to_value_);
  EXPECT_EQ(shape_env_bak.symbol_to_source_.size(), 3);
  EXPECT_EQ(shape_env_bak.symbol_to_source_, shape_env.symbol_to_source_);
  EXPECT_EQ(shape_env_bak.value_to_symbol_.size(), 2);
  EXPECT_EQ(shape_env_bak.value_to_symbol_, shape_env.value_to_symbol_);
  EXPECT_EQ(shape_env_bak.symbol_check_infos_.size(), 1);
  EXPECT_EQ(shape_env_bak.symbol_check_infos_, shape_env.symbol_check_infos_);
  EXPECT_EQ(shape_env_bak.symbol_assert_infos_.size(),0);
  EXPECT_EQ(shape_env_bak.symbol_assert_infos_, shape_env.symbol_assert_infos_);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(AttributeGroupShapeEnvUt, CreateShapeEnvFromGraph_Succ) {
  auto s = AttrStore::Create(1);
  auto shape_env = s.CreateAttrsGroup<ShapeEnvAttr>(ShapeEnvSetting(false, DynamicMode::kDuck));
  EXPECT_NE(shape_env, nullptr);
}

TEST_F(AttributeGroupShapeEnvUt, Get_Guard_Has_Dfx_Info_When_Set_Guard_Context) {
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(true, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  Symbol s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(1, 2));
  Symbol s1 = shape_env.CreateSymbol(5, MakeShared<GraphInputShapeSourceStub>(2, 0));

  auto guard_1 = sym::Eq(s1 + Symbol(2), s0);
  auto guard_2 = sym::Gt(s0 + Symbol(1), s1);

  GuardDfxContext dfx_context("node name:Add");

  shape_env.AppendSymbolCheckInfo(guard_1);
  shape_env.AppendSymbolAssertInfo(guard_2);

  auto check_infos = shape_env.GetAllSymbolCheckInfos();
  auto assert_infos = shape_env.GetAllSymbolAssertInfos();
  EXPECT_EQ(1, check_infos.size());
  EXPECT_EQ(1, assert_infos.size());

  EXPECT_EQ("node name:Add", check_infos[0].dfx_info);
  EXPECT_EQ("node name:Add", assert_infos[0].dfx_info);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(AttributeGroupShapeEnvUt, Get_Guard_Has_No_Dfx_Info_When_Clear_Guard_Context) {
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(true, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  Symbol s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(1, 2));
  Symbol s1 = shape_env.CreateSymbol(5, MakeShared<GraphInputShapeSourceStub>(2, 0));

  auto guard_1 = sym::Eq(s1 + Symbol(2), s0);
  auto guard_2 = sym::Gt(s0 + Symbol(1), s1);

  {
    GuardDfxContext dfx_context("node name:Add");
    shape_env.AppendSymbolCheckInfo(guard_1);
  }
  
  auto check_infos = shape_env.GetAllSymbolCheckInfos();
  EXPECT_EQ(1, check_infos.size());
  EXPECT_EQ("node name:Add", check_infos[0].dfx_info);

  shape_env.AppendSymbolAssertInfo(guard_2);
  auto assert_infos = shape_env.GetAllSymbolAssertInfos();
  EXPECT_EQ(1, assert_infos.size());
  EXPECT_EQ("", assert_infos[0].dfx_info);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(AttributeGroupShapeEnvUt, Get_Guard_Has_New_Dfx_Info_When_Set_Guard_Context_Twice) {
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(true, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  Symbol s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(1, 2));
  Symbol s1 = shape_env.CreateSymbol(5, MakeShared<GraphInputShapeSourceStub>(2, 0));

  auto guard_1 = sym::Eq(s1 + Symbol(2), s0);
  auto guard_2 = sym::Gt(s0 + Symbol(1), s1);

  GuardDfxContext dfx_context("node name:Add");
  shape_env.AppendSymbolCheckInfo(guard_1);

  GuardDfxContext dfx_context1("node name:Sub");
  shape_env.AppendSymbolAssertInfo(guard_2);

  auto check_infos = shape_env.GetAllSymbolCheckInfos();
  auto assert_infos = shape_env.GetAllSymbolAssertInfos();
  EXPECT_EQ(1, check_infos.size());
  EXPECT_EQ(1, assert_infos.size());

  EXPECT_EQ("node name:Add", check_infos[0].dfx_info);
  EXPECT_EQ("node name:Sub", assert_infos[0].dfx_info);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(AttributeGroupShapeEnvUt, Get_Deserialize_Guard_Has_New_Dfx_Info_When_Set_Guard_Context_Serialize) {
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(true, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  Symbol s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(1, 2));
  Symbol s1 = shape_env.CreateSymbol(5, MakeShared<GraphInputShapeSourceStub>(2, 0));

  auto guard_1 = sym::Eq(s1 + Symbol(2), s0);
  auto guard_2 = sym::Gt(s0 + Symbol(1), s1);
  GuardDfxContext dfx_context("node name:Add");
  shape_env.AppendSymbolCheckInfo(guard_1);
  shape_env.AppendSymbolAssertInfo(guard_2);

  proto::AttrGroupDef attr_group_def;
  auto ret = shape_env.Serialize(attr_group_def);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  auto shape_env1 = ShapeEnvAttr();
  ret = shape_env1.Deserialize(attr_group_def, nullptr);
  auto check_infos = shape_env1.GetAllSymbolCheckInfos();
  auto assert_infos = shape_env1.GetAllSymbolAssertInfos();
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(2, shape_env1.symbol_to_value_.size());
  EXPECT_EQ(2, shape_env1.value_to_symbol_.size());
  EXPECT_EQ(Symbol("s0"), shape_env1.value_to_symbol_[2][0]);
  EXPECT_EQ(Symbol("s1"), shape_env1.value_to_symbol_[5][0]);
  EXPECT_EQ(true, shape_env1.HasSymbolCheckInfo(sym::Eq(Symbol("s1") + Symbol(2), Symbol("s0"))));
  EXPECT_EQ(true, shape_env1.HasSymbolAssertInfo(sym::Gt(Symbol("s0") + Symbol(1), Symbol("s1"))));
  EXPECT_EQ("node name:Add", check_infos[0].dfx_info);
  EXPECT_EQ("node name:Add", assert_infos[0].dfx_info);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(AttributeGroupShapeEnvUt, CheckReplacementCycleTest) {
  ShapeEnvAttr shape_env;
  SetCurShapeEnvContext(&shape_env);
  Symbol s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  Symbol s1 = shape_env.CreateSymbol(0, MakeShared<GraphInputShapeSourceStub>(0, 1));
  Symbol s2 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 1));

  auto replacements = shape_env.replacements_;
  // 1、 repalcement直接包含key, s0 + s1表达式中包含s0, 不能加入replacement
  EXPECT_EQ(EXPECT_SYMBOL_EQ(s0 + s1, s0), true);
  if (shape_env.replacements_.find(s0) != shape_env.replacements_.end()) {
    EXPECT_NE(replacements[s0].replace_expr, (s0 + s1));
  }
  // 2、 化简后replace包含key, s0 + s1 化简后是s2 + s1包含s2, 不能加入replacement
  EXPECT_EQ(EXPECT_SYMBOL_EQ(s0, s2), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(s0 + s1, s2), true);
  if (shape_env.replacements_.find(s2) != shape_env.replacements_.end()) {
    EXPECT_NE(replacements[s2].replace_expr, (s0 + s1));
  }
  // 3、 root_expr包含不能替换, s0的root_expr是s2, s2 + s1包含s2, 不能加入replacement
  EXPECT_EQ(EXPECT_SYMBOL_EQ(s1 + s2, s0), true);
  if (shape_env.replacements_.find(s2) != shape_env.replacements_.end()) {
    EXPECT_NE(replacements[s2].replace_expr, (s1 + s2));
  }

  SetCurShapeEnvContext(nullptr);
}
}
}  // namespace ge
