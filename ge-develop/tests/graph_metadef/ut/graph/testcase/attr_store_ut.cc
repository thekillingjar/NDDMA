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
#include "graph/attr_store.h"
#include "graph/op_desc.h"
#include "ge_common/debug/ge_log.h"
#include "graph/attribute_group/attr_group_serialize.h"
#include "graph/attribute_group/attr_group_base.h"
#include "graph/attribute_group/attr_group_serializer_registry.h"
#include "graph/compute_graph.h"
#include "graph/attribute_group/attr_group_shape_env.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "common/checker.h"
#include "test_structs.h"

namespace ge {
namespace {
struct TestStructB {
  int16_t a;
  int b;
  int64_t c;
  bool operator==(const TestStructB &other) const {
    return a == other.a && b == other.b && c == other.c;
  }
};

struct TestAscendCIROpAttrGroups : public AttrGroupsBase {
  std::string name;
  std::string type;

  graphStatus Serialize(proto::AttrGroupDef &attr_group_def) override;
  graphStatus Deserialize(const proto::AttrGroupDef &attr_group_def, AttrHolder *attr_holder) override;
  std::unique_ptr<AttrGroupsBase> Clone() override;

  bool operator==(const TestAscendCIROpAttrGroups &other) const {
    return name == other.name && type == other.type;
  }
};

std::unique_ptr<AttrGroupsBase> TestAscendCIROpAttrGroups::Clone() {
  return std::unique_ptr<AttrGroupsBase>(new (std::nothrow) TestAscendCIROpAttrGroups(*this));
}

graphStatus TestAscendCIROpAttrGroups::Serialize(proto::AttrGroupDef &attr_group_def)  {
  auto op_attr_groups = attr_group_def.mutable_op_attr_group();
  if (op_attr_groups == nullptr) {
    return GRAPH_FAILED;
  }

  op_attr_groups->set_name(name);
  op_attr_groups->set_type(type);
  return GRAPH_SUCCESS;
}

graphStatus TestAscendCIROpAttrGroups::Deserialize(const proto::AttrGroupDef &attr_group_def, AttrHolder *attr_holder) {
  (void) attr_holder;
  auto &op_attr_groups = attr_group_def.op_attr_group();

  name = op_attr_groups.name();
  type = op_attr_groups.type();
  return GRAPH_SUCCESS;
}

REG_ATTR_GROUP_SERIALIZER(TestAscendCIROpAttrGroups,
                          TestAscendCIROpAttrGroups,
                          GetTypeId<TestAscendCIROpAttrGroups>(),
                          proto::AttrGroupDef::kOpAttrGroup);

struct TestAscendCIROpAttrGroupsFailed : public AttrGroupsBase {
  std::string name;
  std::string type;
  std::unique_ptr<AttrGroupsBase> Clone() override {
    return std::unique_ptr<AttrGroupsBase>(new (std::nothrow) TestAscendCIROpAttrGroupsFailed(*this));
  }
};

}
class AttrStoreUt : public testing::Test {
};

TEST_F(AttrStoreUt, CreateAndGetOk1) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<bool>(0, true));
  EXPECT_TRUE(s.Set<bool>(1, false));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_NE(s.Get<bool>(0), nullptr);
  EXPECT_TRUE(*s.Get<bool>(0));
  EXPECT_NE(s.Get<bool>(1), nullptr);
  EXPECT_FALSE(*s.Get<bool>(1));

  EXPECT_NE(s.GetByName<bool>("transpose_x1"), nullptr);
  EXPECT_TRUE(*s.GetByName<bool>("transpose_x1"));
  EXPECT_NE(s.GetByName<bool>("transpose_x2"), nullptr);
  EXPECT_FALSE(*s.GetByName<bool>("transpose_x2"));
}

TEST_F(AttrStoreUt, CreateAndGetOk2) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set(0, true));
  EXPECT_TRUE(s.Set(1, false));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_NE(s.Get<bool>(0), nullptr);
  EXPECT_TRUE(*s.Get<bool>(0));
  EXPECT_NE(s.Get<bool>(1), nullptr);
  EXPECT_FALSE(*s.Get<bool>(1));
}

TEST_F(AttrStoreUt, CreateAndGetOk3) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set(0, true));
  EXPECT_TRUE(s.Set(1, TestStructB({1,2,3})));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_NE(s.Get<bool>(0), nullptr);
  EXPECT_TRUE(*s.Get<bool>(0));
  EXPECT_NE(s.Get<TestStructB>(1), nullptr);
  EXPECT_EQ(*s.Get<TestStructB>(1), TestStructB({1,2,3}));

  EXPECT_NE(s.GetByName<bool>("transpose_x1"), nullptr);
  EXPECT_TRUE(*s.GetByName<bool>("transpose_x1"));
  EXPECT_NE(s.GetByName<TestStructB>("transpose_x2"), nullptr);
  EXPECT_EQ(*s.GetByName<TestStructB>("transpose_x2"), TestStructB({1,2,3}));
}

TEST_F(AttrStoreUt, CreateAndGetOk_RLValue1) {
  int a = 10;
  int &b = a;
  const int c = 20;

  auto s = AttrStore::Create(4);
  EXPECT_TRUE(s.Set(0, a));
  EXPECT_TRUE(s.Set(1, b));
  EXPECT_TRUE(s.Set(2, c));
  EXPECT_TRUE(s.Set(3, 20));

  EXPECT_EQ(*s.Get<int>(0), 10);
  EXPECT_EQ(*s.Get<int>(1), 10);
  EXPECT_EQ(*s.Get<int>(2), 20);
  EXPECT_EQ(*s.Get<int>(3), 20);
}

TEST_F(AttrStoreUt, CreateAndGetOk_RLValue2) {
  TestStructB a = {10, 20, 30};
  TestStructB &b = a;
  const TestStructB c = {100, 200, 300};

  auto s = AttrStore::Create(4);
  EXPECT_TRUE(s.SetByName("attr_0", a));
  EXPECT_TRUE(s.SetByName("attr_1", b));
  EXPECT_TRUE(s.SetByName("attr_2", c));
  EXPECT_TRUE(s.SetByName("attr_3", TestStructB{100,200,300}));

  EXPECT_EQ(*s.GetByName<TestStructB>("attr_0"), a);
  EXPECT_EQ(*s.GetByName<TestStructB>("attr_1"), a);
  EXPECT_EQ(*s.GetByName<TestStructB>("attr_2"), c);
  EXPECT_EQ(*s.GetByName<TestStructB>("attr_3"), c);
}

TEST_F(AttrStoreUt, CreateAndGetByNameOk1) {
  auto s = AttrStore::Create(2);

  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_TRUE(s.SetByName("transpose_x1", true));
  EXPECT_TRUE(s.SetByName("transpose_x2", false));

  EXPECT_NE(s.Get<bool>(0), nullptr);
  EXPECT_TRUE(*s.Get<bool>(0));
  EXPECT_NE(s.Get<bool>(1), nullptr);
  EXPECT_FALSE(*s.Get<bool>(1));

  EXPECT_NE(s.GetByName<bool>("transpose_x1"), nullptr);
  EXPECT_TRUE(*s.GetByName<bool>("transpose_x1"));
  EXPECT_NE(s.GetByName<bool>("transpose_x2"), nullptr);
  EXPECT_FALSE(*s.GetByName<bool>("transpose_x2"));
}

TEST_F(AttrStoreUt, CreateAndGetByNameOk2) {
  auto s = AttrStore::Create(2);

  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_TRUE(s.SetByName("transpose_x3", true));
  EXPECT_TRUE(s.SetByName("transpose_x4", false));

  EXPECT_EQ(s.Get<bool>(0), nullptr);
  EXPECT_EQ(s.Get<bool>(1), nullptr);

  EXPECT_NE(s.GetByName<bool>("transpose_x3"), nullptr);
  EXPECT_NE(s.GetByName<bool>("transpose_x4"), nullptr);

  EXPECT_EQ(*s.GetByName<bool>("transpose_x3"), true);
  EXPECT_EQ(*s.GetByName<bool>("transpose_x4"), false);
}

TEST_F(AttrStoreUt, GetNotExists) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<bool>(0, true));
  EXPECT_TRUE(s.Set<bool>(1, false));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_EQ(s.Get<bool>(2), nullptr);
  EXPECT_EQ(s.MutableGet<bool>(2), nullptr);

  EXPECT_EQ(s.GetByName<bool>("transpose_x3"), nullptr);
  EXPECT_EQ(s.MutableGetByName<bool>("transpose_x3"), nullptr);
}

TEST_F(AttrStoreUt, DeleteOk) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<bool>(0, true));
  EXPECT_TRUE(s.Set<bool>(1, false));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_NE(s.Get<bool>(0), nullptr);
  EXPECT_NE(s.Get<bool>(1), nullptr);

  EXPECT_TRUE(s.Delete("transpose_x1"));
  EXPECT_EQ(s.Get<bool>(0), nullptr);
  EXPECT_FALSE(s.Delete("transpose_x1"));
}

TEST_F(AttrStoreUt, GetWithWrongType) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<bool>(0, true));
  EXPECT_TRUE(s.Set<TestStructB>(1, {1,2,10}));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_NE(s.Get<bool>(0), nullptr);
  EXPECT_NE(s.Get<TestStructB>(1), nullptr);
  EXPECT_EQ(s.Get<int>(0), nullptr);
  EXPECT_EQ(s.Get<int>(1), nullptr);
}

TEST_F(AttrStoreUt, ModifyOk) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<bool>(0, true));
  EXPECT_TRUE(s.Set<bool>(1, false));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_TRUE(*s.Get<bool>(0));
  EXPECT_FALSE(*s.Get<bool>(1));

  EXPECT_TRUE(s.Set<bool>(0, false));
  EXPECT_FALSE(*s.Get<bool>(0));
}

TEST_F(AttrStoreUt, ModifyByNameOk) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<int64_t>(0, 100));
  EXPECT_TRUE(s.Set<int64_t>(1, 200));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  auto p = s.MutableGetByName<int64_t>("transpose_x1");
  EXPECT_NE(p,  nullptr);
  *p = 101;
  EXPECT_EQ(*s.Get<int64_t>(0), 101);
  EXPECT_EQ(*s.GetByName<int64_t>("transpose_x1"), 101);


  p = s.MutableGetByName<int64_t>("transpose_x2");
  EXPECT_NE(p,  nullptr);
  *p = 201;
  EXPECT_EQ(*s.Get<int64_t>(1), 201);
  EXPECT_EQ(*s.GetByName<int64_t>("transpose_x2"), 201);
}

TEST_F(AttrStoreUt, ExistsOk) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<int64_t>(0, 100));
  EXPECT_TRUE(s.Set<int64_t>(1, 200));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);

  EXPECT_TRUE(s.Exists(0));
  EXPECT_TRUE(s.Exists(1));
  EXPECT_TRUE(s.Exists("transpose_x1"));
  EXPECT_TRUE(s.Exists("transpose_x2"));
  EXPECT_FALSE(s.Exists(2));
  EXPECT_FALSE(s.Exists("transpose_x3"));
}

TEST_F(AttrStoreUt, CopyOk) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<int64_t>(0, 100));
  EXPECT_TRUE(s.Set<bool>(1, true));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);
  s.SetByName("attr_3", TestStructB{10,200,3000});
  s.SetByName("attr_4", std::vector<int64_t>({1,2,3,4,5}));
  auto attr_group_ptr = s.GetOrCreateAttrsGroup<TestAscendCIROpAttrGroups>();

  auto s2(s);

  AttrStore s_copy(s);
  s_copy = AttrStore(s);
  EXPECT_EQ(*s_copy.Get<int64_t>(0), 100);

  EXPECT_NE(s2.Get<int64_t>(0), nullptr);
  EXPECT_NE(s2.Get<int64_t>(0), s.Get<int64_t>(0));
  EXPECT_EQ(*s2.Get<int64_t>(0), 100);

  EXPECT_NE(s2.Get<bool>(1), nullptr);
  EXPECT_NE(s2.Get<bool>(1), s.Get<bool>(1));
  EXPECT_EQ(*s2.Get<bool>(1), true);

  EXPECT_NE(s2.GetByName<int64_t>("transpose_x1"), nullptr);
  EXPECT_NE(s2.GetByName<int64_t>("transpose_x1"), s.GetByName<int64_t>("transpose_x1"));
  EXPECT_EQ(*s2.GetByName<int64_t>("transpose_x1"), 100);

  EXPECT_NE(s2.GetByName<bool>("transpose_x2"), nullptr);
  EXPECT_NE(s2.GetByName<bool>("transpose_x2"), s.GetByName<bool>("transpose_x2"));
  EXPECT_EQ(*s2.GetByName<bool>("transpose_x2"), true);

  EXPECT_NE(s2.GetByName<TestStructB>("attr_3"), nullptr);
  EXPECT_NE(s2.GetByName<TestStructB>("attr_3"), s.GetByName<TestStructB>("attr_3"));
  EXPECT_EQ(*s2.GetByName<TestStructB>("attr_3"), TestStructB({10,200,3000}));

  EXPECT_NE(s2.GetByName<std::vector<int64_t>>("attr_4"), nullptr);
  EXPECT_NE(s2.GetByName<std::vector<int64_t>>("attr_4"), s.GetByName<std::vector<int64_t>>("attr_4"));
  EXPECT_EQ(*s2.GetByName<std::vector<int64_t>>("attr_4"), std::vector<int64_t>({1,2,3,4,5}));

  EXPECT_NE(s2.GetOrCreateAttrsGroup<TestAscendCIROpAttrGroups>(), attr_group_ptr);
  EXPECT_EQ(*(s2.GetOrCreateAttrsGroup<TestAscendCIROpAttrGroups>()), *attr_group_ptr);

  auto s3 = AttrStore::Create(1);
  auto ptr = s3.GetOrCreateAttrsGroup<TestAscendCIROpAttrGroups>();
  EXPECT_NE(ptr, nullptr);
  s3 = s;
  EXPECT_NE(s3.GetOrCreateAttrsGroup<TestAscendCIROpAttrGroups>(), attr_group_ptr);
  EXPECT_EQ(*(s3.GetOrCreateAttrsGroup<TestAscendCIROpAttrGroups>()), *attr_group_ptr);
}

TEST_F(AttrStoreUt, MoveOk) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<int64_t>(0, 100));
  EXPECT_TRUE(s.Set<bool>(1, true));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);
  s.SetByName("attr_3", TestStructB{10,200,3000});
  s.SetByName("attr_4", std::vector<int64_t>({1,2,3,4,5}));

  auto attr_3 = s.GetByName<TestStructB>("attr_3");
  auto attr_4 = s.GetByName<std::vector<int64_t>>("attr_4");
  auto attr_group_ptr = s.GetOrCreateAttrsGroup<TestAscendCIROpAttrGroups>();

  auto s2(std::move(s));

  EXPECT_NE(s2.GetByName<TestStructB>("attr_3"), nullptr);
  EXPECT_EQ(s2.GetByName<TestStructB>("attr_3"), attr_3);
  EXPECT_EQ(*s2.GetByName<TestStructB>("attr_3"), TestStructB({10,200,3000}));

  EXPECT_NE(s2.GetByName<std::vector<int64_t>>("attr_4"), nullptr);
  EXPECT_EQ(s2.GetByName<std::vector<int64_t>>("attr_4"), attr_4);
  EXPECT_EQ(*s2.GetByName<std::vector<int64_t>>("attr_4"), std::vector<int64_t>({1,2,3,4,5}));

  EXPECT_EQ(s2.GetOrCreateAttrsGroup<TestAscendCIROpAttrGroups>(), attr_group_ptr);
}

TEST_F(AttrStoreUt, GetAllAttrNamesOk) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<int64_t>(0, 100));
  EXPECT_TRUE(s.Set<bool>(1, true));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);
  s.SetByName("attr_3", TestStructB{10,200,3000});
  s.SetByName("attr_4", std::vector<int64_t>({1,2,3,4,5}));

  EXPECT_EQ(s.GetAllAttrNames(), std::set<std::string>({"transpose_x1",
                                                        "transpose_x2",
                                                        "attr_3",
                                                        "attr_4"}));
}

TEST_F(AttrStoreUt, GetAllAttrsOk) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<int64_t>(0, 100));
  EXPECT_TRUE(s.Set<bool>(1, true));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);
  s.SetByName("attr_3", TestStructB{10,200,3000});
  s.SetByName("attr_4", std::vector<int64_t>({1,2,3,4,5}));

  auto attrs = s.GetAllAttrs();
  EXPECT_EQ(attrs.size(), 4);
  EXPECT_EQ(*attrs["transpose_x1"].Get<int64_t>(), 100);
  EXPECT_EQ(*attrs["transpose_x2"].Get<bool>(), true);
  EXPECT_EQ(*attrs["attr_3"].Get<TestStructB>(), TestStructB({10,200,3000}));
  EXPECT_EQ(*attrs["attr_4"].Get<std::vector<int64_t>>(), std::vector<int64_t>({1,2,3,4,5}));
}

TEST_F(AttrStoreUt, GetAllAttrs_EmptyPredefinedAttrsNotReturn) {
  auto s = AttrStore::Create(2);
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);
  EXPECT_TRUE(s.Set<int64_t>(0, 100));
  s.SetByName("attr_3", TestStructB{10,200,3000});
  s.SetByName("attr_4", std::vector<int64_t>({1,2,3,4,5}));

  auto attrs = s.GetAllAttrs();
  EXPECT_EQ(attrs.size(), 3);
  EXPECT_EQ(attrs.count("transpose_x2"), 0);
}

TEST_F(AttrStoreUt, GetAllAttrsWithFilterOk) {
  auto s = AttrStore::Create(2);
  EXPECT_TRUE(s.Set<int64_t>(0, 100));
  EXPECT_TRUE(s.Set<bool>(1, true));
  s.SetNameAndId("transpose_x1", 0);
  s.SetNameAndId("transpose_x2", 1);
  s.SetByName("attr_3", TestStructB{10,200,3000});
  s.SetByName("attr_4", std::vector<int64_t>({1,2,3,4,5}));

  const AttrNameFilter attr_filter = [](const std::string &attr_name) -> bool {
    return (attr_name != "transpose_x1") && (attr_name != "attr_4");
  };

  auto attrs = s.GetAllAttrsWithFilter(attr_filter);
  EXPECT_EQ(attrs.size(), 2);
  EXPECT_EQ(*attrs["transpose_x2"].Get<bool>(), true);
  EXPECT_EQ(*attrs["attr_3"].Get<TestStructB>(), TestStructB({10,200,3000}));
}

TEST_F(AttrStoreUt, TestGetAttrGroups) {
  auto s = AttrStore::Create(2);
  auto flag = s.CheckAttrGroupIsExist<TestAscendCIROpAttrGroups>();
  EXPECT_EQ(flag, false);

  auto ptr = s.GetOrCreateAttrsGroup<TestAscendCIROpAttrGroups>();
  EXPECT_NE(ptr, nullptr);

  flag = s.CheckAttrGroupIsExist<TestAscendCIROpAttrGroups>();
  EXPECT_EQ(flag, true);
}

TEST_F(AttrStoreUt, TestAttrGroup) {
  auto s = AttrStore::Create(1);
  auto flag = s.CheckAttrGroupIsExist<TestAscendCIROpAttrGroups>();
  ASSERT_EQ(flag, false);

  auto ptr = s.GetOrCreateAttrsGroup<TestAscendCIROpAttrGroups>();
  ASSERT_NE(ptr, nullptr);
  TestAscendCIROpAttrGroups &t = *ptr;
  t.name = "weqweqr";

  s.ClearAttrsGroups();

  flag = s.CheckAttrGroupIsExist<TestAscendCIROpAttrGroups>();
  ASSERT_EQ(flag, false);
}

TEST_F(AttrStoreUt, OtherAttrGroupTest) {
  auto s = AttrStore::Create(1);
  std::string attr_name = "Max memory";
  auto flag = s.CheckAttrIsExistInOtherGroup(attr_name);
  ASSERT_EQ(flag, false);

  AnyValue any_value;
  auto ret = s.GetAttrFromOtherGroup(attr_name, any_value);
  ASSERT_EQ(ret, GRAPH_FAILED);

  any_value.SetValue<int64_t>(1);

  ret = s.SetAttrToOtherGroup(attr_name, any_value);
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  AnyValue any_value1;
  ret = s.GetAttrFromOtherGroup(attr_name, any_value1);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  int64_t value = -1;
  int64_t value1 = -1;
  any_value.GetValue(value);
  any_value1.GetValue(value1);
  ASSERT_EQ(value, 1);
  ASSERT_EQ(value, value1);

  ret = s.GetAttrFromOtherGroup(attr_name, any_value1);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  value = -1;
  value1 = -1;
  any_value.GetValue(value);
  any_value1.GetValue(value1);
  ASSERT_EQ(value, 1);
  ASSERT_EQ(value, value1);

  bool delete_flag = s.DeleteSingleAttrsInOtherGroup(attr_name);
  ASSERT_EQ(delete_flag, true);

  ret = s.GetAttrFromOtherGroup(attr_name, any_value1);
  ASSERT_EQ(ret, GRAPH_FAILED);

  any_value.SetValue<int64_t>(10);
  ret = s.SetAttrToOtherGroup(attr_name, any_value);
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  ret = s.GetAttrFromOtherGroup(attr_name, any_value1);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  value = -1;
  value1 = -1;
  any_value.GetValue(value);
  any_value1.GetValue(value1);
  ASSERT_EQ(value, 10);
  ASSERT_EQ(value, value1);

  s.DeleteAllAttrsInOtherGroup();

  ret = s.GetAttrFromOtherGroup(attr_name, any_value1);
  ASSERT_EQ(ret, GRAPH_FAILED);
}

TEST_F(AttrStoreUt, ErrorTest) {
  auto s = AttrStore::Create(1);
  std::string no_exist_attr_name = "TEST2";

  AnyValue any_value;
  auto ret = s.SetAttrToOtherGroup(no_exist_attr_name, any_value);
  ASSERT_EQ(ret, GRAPH_FAILED);

  std::string attr_name = "Max memory";
  ret = s.SetAttrToOtherGroup(attr_name, any_value);
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  auto flag = s.CheckAttrIsExistInOtherGroup(attr_name);
  ASSERT_EQ(flag, true);
}

TEST_F(AttrStoreUt, ErrorTest2) {
  auto s = AttrStore::Create(1);
  auto ptr = s.GetOrCreateAttrsGroup<TestAscendCIROpAttrGroups>();
  ASSERT_NE(ptr, nullptr);
  ptr->name = "test attr group";
  ptr->type = "test type";

  AnyValue err_any_value;
  err_any_value.SetValue(1);
  std::string attr_name = "Max memory";
  auto ret = s.GetAttrFromOtherGroup(attr_name, err_any_value);
  ASSERT_EQ(ret, GRAPH_FAILED);

  std::string err_attr_name = "TEST2";
  auto flag = s.ClearAttrInOtherAttrs(err_attr_name);
  ASSERT_EQ(flag, false);

  AnyValue any_value;
  ret = s.GetAttrFromOtherGroup(attr_name, any_value);
  ASSERT_EQ(ret, GRAPH_FAILED);

  any_value.SetValue<int64_t>(1);

  ret = s.SetAttrToOtherGroup(attr_name, any_value);
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  s.ClearAllAttrsInOtherAttrs();
  flag = s.ClearAttrInOtherAttrs(attr_name);
  ASSERT_EQ(flag, false);

  ptr = s.GetOrCreateAttrsGroup<TestAscendCIROpAttrGroups>();
  ASSERT_NE(ptr, nullptr);
  ret = s.SetAttrToOtherGroup(attr_name, any_value);
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  s.ClearAllAttrs();

  ret = s.GetAttrFromOtherGroup(attr_name, any_value);
  ASSERT_EQ(ret, GRAPH_FAILED);
}

TEST_F(AttrStoreUt, AttrGroupSerializeAndDeSeralize) {
  auto s = AttrStore::Create(1);
  std::string attr_name = "Max memory";
  auto flag = s.CheckAttrIsExistInOtherGroup(attr_name);
  ASSERT_EQ(flag, false);
  AnyValue any_value;
  any_value.SetValue<int64_t>(1);

  auto ret = s.SetAttrToOtherGroup(attr_name, any_value);
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  auto m = s.GetAllAttrsFromOtherGroup();
  ASSERT_EQ(m.size(), 1);

  auto ptr = s.GetOrCreateAttrsGroup<TestAscendCIROpAttrGroups>();
  ASSERT_NE(ptr, nullptr);
  ptr->name = "test attr group";
  ptr->type = "test type";

  proto::AttrGroups attr_group;
  ret = AttrGroupSerialize::SerializeAllAttr(attr_group, s);
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  auto op_desc = std::make_shared<OpDesc>("Stub", "Stub");
  ASSERT_NE(op_desc, nullptr);
  ret = AttrGroupSerialize::DeserializeAllAttr(attr_group, op_desc.get());
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  auto ptr_new = op_desc->GetAttrsGroup<TestAscendCIROpAttrGroups>();
  ASSERT_NE(ptr_new, nullptr);
  ASSERT_EQ(*ptr_new, *ptr);
}

TEST_F(AttrStoreUt, AttrGroupSerializer_invalid) {
  EXPECT_NE(AttrGroupSerializerRegistry::GetInstance().GetSerializer(GetTypeId<TestAscendCIROpAttrGroups>()), nullptr);
  // invalid builder
  AttrGroupSerializerRegistry::GetInstance().RegisterAttrGroupSerialize([]() -> std::unique_ptr<AttrGroupsBase> { return nullptr; },
                                                                        GetTypeId<TestAscendCIROpAttrGroups>(),
                                                                        proto::AttrGroupDef::kOpAttrGroup);
  // repeat reg
  AttrGroupSerializerRegistry::GetInstance().RegisterAttrGroupSerialize([]() -> std::unique_ptr<ge::AttrGroupsBase> {
                                                                          return std::unique_ptr<ge::AttrGroupsBase>(new(std::nothrow)TestAscendCIROpAttrGroups());
                                                                        }, GetTypeId<TestAscendCIROpAttrGroups>(),
                                                                        proto::AttrGroupDef::kOpAttrGroup);
  // builder is null
  AttrGroupSerializerRegister attr_group_serializer_registrar
      (nullptr, GetTypeId<TestAscendCIROpAttrGroups>(), proto::AttrGroupDef::kOpAttrGroup);
}

TEST_F(AttrStoreUt, GetOrCreateAttrGroupWith0Args) {
  auto s = AttrStore::Create(1);
  auto ptr = s.CreateAttrsGroup<TestAttrGroup>();
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(ptr->a, 0);
  ASSERT_EQ(ptr->b, 0);

  auto ptr_1 = s.CreateAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ptr_1, nullptr);

  auto ptr_2 = s.CreateAttrsGroup<TestAttrGroup>(1);
  ASSERT_EQ(ptr_2, nullptr);

  auto ptr_3 = s.CreateAttrsGroup<TestAttrGroup>(1, 2);
  ASSERT_EQ(ptr_3, nullptr);

  auto ptr_4 = s.GetAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ptr_4, ptr);

  auto ptr_5 = s.GetOrCreateAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ptr_5, ptr);
}

TEST_F(AttrStoreUt, GetOrCreateAttrGroupWith1Args) {
  auto s = AttrStore::Create(1);
  auto ptr = s.CreateAttrsGroup<TestAttrGroup>(1);
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(ptr->a, 1);
  ASSERT_EQ(ptr->b, 0);

  auto ptr_1 = s.CreateAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ptr_1, nullptr);

  auto ptr_2 = s.CreateAttrsGroup<TestAttrGroup>(1);
  ASSERT_EQ(ptr_2, nullptr);

  auto ptr_3 = s.CreateAttrsGroup<TestAttrGroup>(1, 2);
  ASSERT_EQ(ptr_3, nullptr);

  auto ptr_4 = s.GetAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ptr_4, ptr);

  auto ptr_5 = s.GetOrCreateAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ptr_5, ptr);
}

TEST_F(AttrStoreUt, DeleteAttrsGroup) {
  auto s = AttrStore::Create(1);
  ASSERT_FALSE(s.DeleteAttrsGroup<TestAttrGroup>());
  s.CreateAttrsGroup<TestAttrGroup>();
  ASSERT_TRUE(s.DeleteAttrsGroup<TestAttrGroup>());
  ASSERT_FALSE(s.DeleteAttrsGroup<TestAttrGroup>());
  ASSERT_EQ(s.GetAttrsGroup<TestAttrGroup>(), nullptr);
  ASSERT_FALSE(s.CheckAttrGroupIsExist<TestAttrGroup>());
  s.CreateAttrsGroup<TestAttrGroup>();
  ASSERT_TRUE(s.DeleteAttrsGroup<TestAttrGroup>());

  ge::ComputeGraph cg("simple");
  auto graph_attr_group_ptr = cg.GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_TRUE(cg.DeleteAttrsGroup<ShapeEnvAttr>());
  ASSERT_EQ(cg.GetAttrsGroup<ShapeEnvAttr>(), nullptr);
  ASSERT_FALSE(cg.DeleteAttrsGroup<ShapeEnvAttr>());
  graph_attr_group_ptr = cg.CreateAttrsGroup<ShapeEnvAttr>();
  EXPECT_TRUE(graph_attr_group_ptr != nullptr);
}

TEST_F(AttrStoreUt, GetOrCreateAttrGroupWith2Args) {
  auto s = AttrStore::Create(1);
  auto ptr = s.CreateAttrsGroup<TestAttrGroup>(1, 2);
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(ptr->a, 1);
  ASSERT_EQ(ptr->b, 2);

  auto ptr_1 = s.CreateAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ptr_1, nullptr);

  auto ptr_2 = s.CreateAttrsGroup<TestAttrGroup>(1);
  ASSERT_EQ(ptr_2, nullptr);

  auto ptr_3 = s.CreateAttrsGroup<TestAttrGroup>(1, 2);
  ASSERT_EQ(ptr_3, nullptr);

  auto ptr_4 = s.GetAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ptr_4, ptr);

  auto ptr_5 = s.GetOrCreateAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ptr_5, ptr);
}
}
