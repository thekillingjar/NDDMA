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
#include <gmock/gmock.h>
#include <vector>

#include "register/scope/scope_pattern_impl.h"
#include "register/scope/scope_graph_impl.h"
#include "framework/common/debug/ge_log.h"
#include "graph/types.h"
#include "register/scope/scope_fusion_pass_register.h"

using namespace std;
using namespace testing;

namespace ge {

class ScopePatternUt : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(ScopePatternUt, ScopeAttrValue1) {
  ScopeAttrValue scope_attr_value;

  float32_t value = 0.2;
  scope_attr_value.SetFloatValue(value);
  EXPECT_EQ(scope_attr_value.impl_->GetFloatValue(), static_cast<float32_t>(0.2));

  int64_t value2 = 2;
  scope_attr_value.SetIntValue(value2);
  EXPECT_EQ(scope_attr_value.impl_->GetIntValue(), 2);

  scope_attr_value.SetStringValue("abc");
  EXPECT_EQ(scope_attr_value.impl_->GetStrValue(), string("abc"));

  scope_attr_value.SetStringValue(string("def"));
  EXPECT_EQ(scope_attr_value.impl_->GetStrValue(), string("def"));

  scope_attr_value.SetBoolValue(true);
  EXPECT_TRUE(scope_attr_value.impl_->GetBoolValue());

  ScopeAttrValue scope_attr_value2(scope_attr_value);
  EXPECT_EQ(scope_attr_value2.impl_->GetFloatValue(), static_cast<float32_t>(0.2));
  EXPECT_EQ(scope_attr_value2.impl_->GetIntValue(), 2);
  EXPECT_EQ(scope_attr_value2.impl_->GetStrValue(), string("def"));
  EXPECT_TRUE(scope_attr_value2.impl_->GetBoolValue());

  ScopeAttrValue scope_attr_value3;
  scope_attr_value3 = scope_attr_value;
  EXPECT_EQ(scope_attr_value3.impl_->GetFloatValue(), static_cast<float32_t>(0.2));
  EXPECT_EQ(scope_attr_value3.impl_->GetIntValue(), 2);
  EXPECT_EQ(scope_attr_value3.impl_->GetStrValue(), string("def"));
  EXPECT_TRUE(scope_attr_value3.impl_->GetBoolValue());
}

TEST_F(ScopePatternUt, ScopeAttrValue2) {
  ScopeAttrValue scope_attr_value;
  scope_attr_value.impl_ = nullptr;

  float32_t value = 0.2;
  scope_attr_value.SetFloatValue(value);

  int64_t value2 = 2;
  scope_attr_value.SetIntValue(value2);
  scope_attr_value.SetStringValue("abc");
  scope_attr_value.SetStringValue(string("def"));
  scope_attr_value.SetBoolValue(true);

  EXPECT_TRUE(scope_attr_value.impl_ == nullptr);
}

TEST_F(ScopePatternUt, NodeOpTypeFeature) {
  // construct
  string nodeType = string("add");
  int32_t num = 1;
  int32_t step = 100;
  NodeOpTypeFeature notf(nodeType, num, step);
  EXPECT_EQ(notf.impl_->step_, step);
  NodeOpTypeFeature notf2("edf", num, 0);
  EXPECT_EQ(notf2.impl_->node_type_, string("edf"));

  // match
  Scope scope;
  scope.Init("name", "sub_type", nullptr);
  EXPECT_FALSE(notf.Match(nullptr));
  EXPECT_FALSE(notf.Match(&scope));
  EXPECT_FALSE(notf2.Match(&scope));

  // copy
  NodeOpTypeFeature notf3(notf);
  EXPECT_EQ(notf3.impl_->node_type_, string("add"));
  notf3 = notf3;
  notf3 = notf2;
  EXPECT_EQ(notf3.impl_->node_type_, string("edf"));

  notf3.impl_.reset();
  EXPECT_FALSE(notf3.Match(nullptr));
  EXPECT_EQ(notf3.impl_, nullptr);
}

TEST_F(ScopePatternUt, NodeAttrFeature) {
  // construct
  ScopeAttrValue scope_attr_value;
  scope_attr_value.SetStringValue("abc");
  NodeAttrFeature naf("node_type", "attr_name", DT_INT8, scope_attr_value);
  NodeAttrFeature naf2(string("node_type_2"), string("attr_name_2"), DT_INT8, scope_attr_value);
  EXPECT_EQ(naf.impl_->attr_value_.impl_->GetStrValue(), string("abc"));

  // copy
  NodeAttrFeature naf3(naf2);
  EXPECT_EQ(naf3.impl_->node_type_, string("node_type_2"));
  naf3 = naf3;
  naf3 = naf;
  EXPECT_EQ(naf3.impl_->attr_name_, string("attr_name"));

  // match
  Scope scope;
  scope.Init("name", "sub_type", nullptr);
  EXPECT_FALSE(naf3.impl_->Match(nullptr));
  EXPECT_FALSE(naf3.impl_->Match(&scope));
}

TEST_F(ScopePatternUt, CheckNodeAttrFeatureData) {
  ScopeAttrValue scope_attr_value;
  scope_attr_value.SetStringValue("abc");
  NodeAttrFeature naf("node_type", "attr_name", DT_INT8, scope_attr_value);

  bool init_value = true;
  ge::OpDescPtr op_desc(new ge::OpDesc("add1", "Add"));
  Scope scope;
  scope.Init("name", "sub_type", nullptr);

  auto ret = naf.impl_->CheckNodeAttrFeatureData(init_value, op_desc, &scope);
  EXPECT_EQ(ret, PARAM_INVALID);

  string init_value2 = "init_value";
  ret = naf.impl_->CheckNodeAttrFeatureData(init_value2, op_desc, &scope);
  EXPECT_EQ(ret, PARAM_INVALID);

  int64_t init_value3 = 1;
  ret = naf.impl_->CheckNodeAttrFeatureData(init_value3, op_desc, &scope);
  EXPECT_EQ(ret, PARAM_INVALID);

  float32_t init_value4 = 0.2;
  ret = naf.impl_->CheckNodeAttrFeatureData(init_value4, op_desc, &scope);
  EXPECT_EQ(ret, PARAM_INVALID);

  // match
  EXPECT_FALSE(naf.Match(nullptr));
  EXPECT_FALSE(naf.Match(&scope));
}

TEST_F(ScopePatternUt, CheckNodeAttrFeatureDataSuccess) {
  {
    ScopeAttrValue scope_attr_value;
    bool init_value = true;
    scope_attr_value.SetBoolValue(init_value);
    string attr_name("attr_name");
    NodeAttrFeature naf("node_type", attr_name, DT_INT8, scope_attr_value);

    ge::OpDescPtr op_desc(new ge::OpDesc("add1", "Add"));
    ge::AttrUtils::SetBool(op_desc, attr_name, init_value);
    Scope scope;
    scope.Init("name", "sub_type", nullptr);

    auto ret = naf.impl_->CheckNodeAttrFeatureData(init_value, op_desc, &scope);
    EXPECT_EQ(ret, SUCCESS);
  }
  {
    ScopeAttrValue scope_attr_value;
    string init_value = "true";
    scope_attr_value.SetStringValue(init_value.c_str());
    string attr_name("attr_name");
    NodeAttrFeature naf("node_type", attr_name, DT_INT8, scope_attr_value);

    ge::OpDescPtr op_desc(new ge::OpDesc("add1", "Add"));
    ge::AttrUtils::SetStr(op_desc, attr_name, init_value);
    Scope scope;
    scope.Init("name", "sub_type", nullptr);

    auto ret = naf.impl_->CheckNodeAttrFeatureData(init_value, op_desc, &scope);
    EXPECT_EQ(ret, SUCCESS);
  }
  {
    ScopeAttrValue scope_attr_value;
    float32_t init_value = 0.0f;
    scope_attr_value.SetFloatValue(init_value);
    string attr_name("attr_name");
    NodeAttrFeature naf("node_type", attr_name, DT_INT8, scope_attr_value);

    ge::OpDescPtr op_desc(new ge::OpDesc("add1", "Add"));
    ge::AttrUtils::SetFloat(op_desc, attr_name, init_value);
    Scope scope;
    scope.Init("name", "sub_type", nullptr);

    auto ret = naf.impl_->CheckNodeAttrFeatureData(init_value, op_desc, &scope);
    EXPECT_EQ(ret, SUCCESS);
  }
  {
    ScopeAttrValue scope_attr_value;
    int64_t init_value = 0;
    scope_attr_value.SetIntValue(init_value);
    string attr_name("attr_name");
    NodeAttrFeature naf("node_type", attr_name, DT_INT8, scope_attr_value);

    ge::OpDescPtr op_desc(new ge::OpDesc("add1", "Add"));
    ge::AttrUtils::SetInt(op_desc, attr_name, init_value);
    Scope scope;
    scope.Init("name", "sub_type", nullptr);

    auto ret = naf.impl_->CheckNodeAttrFeatureData(init_value, op_desc, &scope);
    EXPECT_EQ(ret, SUCCESS);
  }
}

TEST_F(ScopePatternUt, ScopeFeature) {
  // construct
  string sub_type = "sub_type";
  int32_t num = 3;
  string suffix = "suffix";
  string sub_scope_mask = "sub_scope_mask";
  int32_t step = 0;

  ScopeFeature sf(sub_type, num, suffix, sub_scope_mask, step);
  EXPECT_EQ(sf.impl_->sub_type_, sub_type);

  ScopeFeature sf2("sub_type_2", num, "suffix_2", "sub_scope_mask_2", step);
  EXPECT_EQ(sf2.impl_->sub_type_, string("sub_type_2"));

  // copy
  ScopeFeature sf3(sf2);
  EXPECT_EQ(sf3.impl_->sub_type_, string("sub_type_2"));

  sf2 = sf2;
  sf2 = sf;
  EXPECT_EQ(sf2.impl_->sub_type_, sub_type);

  // match
  Scope scope;
  scope.Init("name", "sub_type", nullptr);
  EXPECT_FALSE(sf.Match(&scope));
}

TEST_F(ScopePatternUt, ScopeFeature_Match) {
  std::vector<Scope *> scopes;
  Scope scope;
  scope.Init("name", "sub_type", nullptr);
  scopes.emplace_back(&scope);
  Scope scope2;
  scope2.Init("name_2", "sub_type_2", nullptr);
  scopes.emplace_back(&scope2);

  ScopeFeature sf2("sub_type_2", 1, "suffix_2", "sub_scope_mask_2", 1);
  auto ret = sf2.impl_->SubScopesMatch(scopes);
  EXPECT_FALSE(ret);
}

TEST_F(ScopePatternUt, ScopePattern) {
  ScopePattern scope_pat;
  EXPECT_NE(scope_pat.impl_, nullptr);

  scope_pat.SetSubType("sub_type");
  scope_pat.SetSubType(string("sub_type_2"));
  EXPECT_EQ(scope_pat.impl_->sub_type_, string("sub_type_2"));

  scope_pat.impl_.reset();
  scope_pat.SetSubType("sub_type");
  scope_pat.SetSubType(string("sub_type_2"));
  EXPECT_EQ(scope_pat.impl_, nullptr);
}

TEST_F(ScopePatternUt, AddFeature) {
  ScopePattern scope_pat;

  NodeOpTypeFeature notf("abc", 1, 0);
  scope_pat.AddNodeOpTypeFeature(notf);
  EXPECT_TRUE(scope_pat.impl_->node_optype_features_.size() > 0);

  ScopeAttrValue scope_attr_value;
  scope_attr_value.SetStringValue("abc");
  NodeAttrFeature naf("node_type", "attr_name", DT_INT8, scope_attr_value);
  scope_pat.AddNodeAttrFeature(naf);
  EXPECT_TRUE(scope_pat.impl_->node_attr_features_.size() > 0);

  ScopeFeature sf("sub_type", 1, "suffix", "sub_scope_mask", 1);
  scope_pat.AddScopeFeature(sf);
  EXPECT_TRUE(scope_pat.impl_->scopes_features_.size() > 0);
}

TEST_F(ScopePatternUt, AddFeature_Null) {
  ScopePattern scope_pat;
  scope_pat.impl_.reset();

  NodeOpTypeFeature notf("abc", 1, 0);
  scope_pat.AddNodeOpTypeFeature(notf);

  ScopeAttrValue scope_attr_value;
  scope_attr_value.SetStringValue("abc");
  NodeAttrFeature naf("node_type", "attr_name", DT_INT8, scope_attr_value);
  scope_pat.AddNodeAttrFeature(naf);

  ScopeFeature sf("sub_type", 1, "suffix", "sub_scope_mask", 1);
  scope_pat.AddScopeFeature(sf);

  EXPECT_EQ(scope_pat.impl_, nullptr);
}
}  // namespace ge
