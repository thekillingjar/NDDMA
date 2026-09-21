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
#include "graph/op_desc.h"
#include "graph/ge_attr_value.h"
#include "graph/utils/attr_utils.h"
#include "graph/attr_value.h"
#include "graph/tensor.h"
#include "graph/ascend_string.h"
#include "graph/types.h"
#include <string>
#include <limits>
#include <functional>

namespace ge {
class UtestGeAttrValue : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};


TEST_F(UtestGeAttrValue, GetAttrsStrAfterRid) {
  string name = "const";
  string type = "Constant";
  OpDescPtr op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(AttrUtils::GetAttrsStrAfterRid(op_desc, {}), "");

  std::set<std::string> names ={"qazwsx", "d"};
  op_desc->SetAttr("qazwsx", GeAttrValue::CreateFrom<int64_t>(132));
  op_desc->SetAttr("xswzaq", GeAttrValue::CreateFrom<int64_t>(123));
  auto tensor = GeTensor();
  op_desc->SetAttr("value", GeAttrValue::CreateFrom<GeTensor>(tensor));
  std::string res = AttrUtils::GetAttrsStrAfterRid(op_desc, names);
  EXPECT_TRUE(res.find("qazwsx") == string::npos);
  EXPECT_TRUE(res.find("xswzaq") != string::npos);
}

TEST_F(UtestGeAttrValue, GetAllAttrsStr) {
 // 属性序列化
  string name = "const";
  string type = "Constant";
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  EXPECT_TRUE(op_desc);
  EXPECT_EQ(AttrUtils::GetAllAttrsStr(op_desc), "");
  op_desc->SetAttr("seri_i", GeAttrValue::CreateFrom<int64_t>(1));
  auto tensor = GeTensor();
  op_desc->SetAttr("seri_value", GeAttrValue::CreateFrom<GeTensor>(tensor));
  op_desc->SetAttr("seri_input_desc", GeAttrValue::CreateFrom<GeTensorDesc>(GeTensorDesc()));
  string attr = AttrUtils::GetAllAttrsStr(op_desc);

  EXPECT_TRUE(attr.find("seri_i") != string::npos);
  EXPECT_TRUE(attr.find("seri_value") != string::npos);
  EXPECT_TRUE(attr.find("seri_input_desc") != string::npos);

}
TEST_F(UtestGeAttrValue, GetAllAttrs) {
  string name = "const";
  string type = "Constant";
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  EXPECT_TRUE(op_desc);
  op_desc->SetAttr("i", GeAttrValue::CreateFrom<int64_t>(100));
  op_desc->SetAttr("input_desc", GeAttrValue::CreateFrom<GeTensorDesc>(GeTensorDesc()));
  auto attrs = AttrUtils::GetAllAttrs(op_desc);
  EXPECT_EQ(attrs.size(), 2);
  int64_t attr_value = 0;
  EXPECT_EQ(attrs["i"].GetValue(attr_value), GRAPH_SUCCESS);
  EXPECT_EQ(attr_value, 100);

}

TEST_F(UtestGeAttrValue, TrySetExists) {
  string name = "const";
  string type = "Constant";
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  EXPECT_TRUE(op_desc);

  int64_t attr_value = 0;

  EXPECT_FALSE(AttrUtils::GetInt(op_desc, "i", attr_value));
  op_desc->TrySetAttr("i", GeAttrValue::CreateFrom<int64_t>(100));
  EXPECT_TRUE(AttrUtils::GetInt(op_desc, "i", attr_value));
  EXPECT_EQ(attr_value, 100);

  op_desc->TrySetAttr("i", GeAttrValue::CreateFrom<int64_t>(102));
  attr_value = 0;
  AttrUtils::GetInt(op_desc, "i", attr_value);
  EXPECT_EQ(attr_value, 100);

  uint64_t uint64_val = 0U;
  EXPECT_TRUE(AttrUtils::GetInt(op_desc, "i", uint64_val));
  EXPECT_EQ(uint64_val, 100U);
}

TEST_F(UtestGeAttrValue, CloneOpDesc_check_null) {
  OpDescPtr op_desc = nullptr;
  auto ret = AttrUtils::CloneOpDesc(op_desc);
  EXPECT_EQ(ret == nullptr, true);
}

TEST_F(UtestGeAttrValue, CopyOpDesc_check_null) {
  OpDescPtr op_desc = nullptr;
  auto ret = AttrUtils::CopyOpDesc(op_desc);
  EXPECT_EQ(ret == nullptr, true);
}

TEST_F(UtestGeAttrValue, SetGetListInt) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("const1", "Identity");
  EXPECT_TRUE(op_desc);

  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "li1", std::vector<int64_t>({1,2,3,4,5})));
  std::vector<int64_t> li1_out0;
  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "li1", li1_out0));
  EXPECT_EQ(li1_out0, std::vector<int64_t>({1,2,3,4,5}));
}

TEST_F(UtestGeAttrValue, SetListIntGetByGeAttrValue) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("const1", "Identity");
  EXPECT_TRUE(op_desc);

  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "li1", std::vector<int64_t>({1,2,3,4,5})));
  auto names_to_value = AttrUtils::GetAllAttrs(op_desc);
  auto iter = names_to_value.find("li1");
  EXPECT_NE(iter, names_to_value.end());

  std::vector<int64_t> li1_out;
  auto &ge_value = iter->second;
  EXPECT_EQ(ge_value.GetValue(li1_out), GRAPH_SUCCESS);
  EXPECT_EQ(li1_out, std::vector<int64_t>({1,2,3,4,5}));

  li1_out.clear();
  EXPECT_EQ(ge_value.GetValue<std::vector<int64_t>>(li1_out), GRAPH_SUCCESS);
  EXPECT_EQ(li1_out, std::vector<int64_t>({1,2,3,4,5}));
}

TEST_F(UtestGeAttrValue, SetGetAttr_GeTensor) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("const1", "Identity");
  GeTensorDesc td;
  td.SetShape(GeShape(std::vector<int64_t>({1,100})));
  td.SetOriginShape(GeShape(std::vector<int64_t>({1,100})));
  td.SetDataType(DT_FLOAT);
  td.SetFormat(FORMAT_ND);
  float data[100];
  for (size_t i = 0; i < 100; ++i) {
    data[i] = 1.0 * i;
  }
  auto tensor = std::make_shared<GeTensor>(td, reinterpret_cast<const uint8_t *>(data), sizeof(data));
  EXPECT_NE(tensor, nullptr);

  EXPECT_TRUE(AttrUtils::SetTensor(op_desc, "t", tensor));
  tensor = nullptr;

  EXPECT_TRUE(AttrUtils::MutableTensor(op_desc, "t", tensor));
  EXPECT_NE(tensor, nullptr);

  EXPECT_EQ(tensor->GetData().GetSize(), sizeof(data));
  auto attr_data = reinterpret_cast<const float *>(tensor->GetData().GetData());
  for (size_t i = 0; i < 100; ++i) {
    EXPECT_FLOAT_EQ(attr_data[i], data[i]);
  }
  tensor = nullptr;

  EXPECT_TRUE(AttrUtils::MutableTensor(op_desc, "t", tensor));
  EXPECT_NE(tensor, nullptr);

  EXPECT_EQ(tensor->GetData().GetSize(), sizeof(data));
  attr_data = reinterpret_cast<const float *>(tensor->GetData().GetData());
  for (size_t i = 0; i < 100; ++i) {
    EXPECT_FLOAT_EQ(attr_data[i], data[i]);
  }
  tensor = nullptr;
}

TEST_F(UtestGeAttrValue, GetStr) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  EXPECT_TRUE(op_desc);

  std::string add_info = "add_info";
  AttrUtils::SetStr(op_desc, "compile_info_key", add_info);
  const std::string *s2 = AttrUtils::GetStr(op_desc, "compile_info_key");
  EXPECT_NE(s2, nullptr);
  EXPECT_EQ(*s2, add_info);
}

TEST_F(UtestGeAttrValue, GetStr_for_2_name) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  EXPECT_TRUE(op_desc);

  std::string name1 = "compile_info_key1";
  std::string name2 = "compile_info_key2";
  std::string value1 = "add_info1";
  std::string value2 = "add_info2";
  std::string value;

  // name1未设置属性，name2设置属性，获取的值是name2的属性值
  AttrUtils::SetStr(op_desc, name2, value2);
  EXPECT_TRUE(AttrUtils::GetStr(op_desc, name1, name2, value));
  EXPECT_EQ(value, value2);

  // name1和name2均设置属性，获取的是name1的属性值
  AttrUtils::SetStr(op_desc, name1, value1);
  EXPECT_TRUE(AttrUtils::GetStr(op_desc, name1, name2, value));
  EXPECT_EQ(value, value1);

  // 异常场景
  EXPECT_FALSE(AttrUtils::GetStr(nullptr, name1, name2, value));
}

TEST_F(UtestGeAttrValue, SetNullObjectAttr) {
  OpDescPtr op_desc(nullptr);
  EXPECT_EQ(AttrUtils::SetStr(op_desc, "key", "value"), false);
  EXPECT_EQ(AttrUtils::SetInt(op_desc, "key", 0), false);
  EXPECT_EQ(AttrUtils::SetTensorDesc(op_desc, "key", GeTensorDesc()), false);
  GeTensorPtr ge_tensor;
  EXPECT_EQ(AttrUtils::SetTensor(op_desc, "key", ge_tensor), false);
  ConstGeTensorPtr const_ge_tensor;
  EXPECT_EQ(AttrUtils::SetTensor(op_desc, "key", const_ge_tensor), false);
  EXPECT_EQ(AttrUtils::SetBool(op_desc, "key", true), false);
  EXPECT_EQ(AttrUtils::SetBytes(op_desc, "key", Buffer()), false);
  EXPECT_EQ(AttrUtils::SetFloat(op_desc, "key", 1.0), false);
  EXPECT_EQ(AttrUtils::SetGraph(op_desc, "key", nullptr), false);
  EXPECT_EQ(AttrUtils::SetDataType(op_desc, "key", DT_UINT8), false);
  EXPECT_EQ(AttrUtils::SetListDataType(op_desc, "key", {DT_UINT8}), false);
  EXPECT_EQ(AttrUtils::SetListListInt(op_desc, "key", {}), false);
  EXPECT_EQ(AttrUtils::SetListInt(op_desc, "key", {}), false);
  EXPECT_EQ(AttrUtils::SetListTensor(op_desc, "key", {}), false);
  EXPECT_EQ(AttrUtils::SetListBool(op_desc, "key", {}), false);
  EXPECT_EQ(AttrUtils::SetListFloat(op_desc, "key", {}), false);
  EXPECT_EQ(AttrUtils::SetListBytes(op_desc, "key", {}), false);
  EXPECT_EQ(AttrUtils::SetListGraph(op_desc, "key", {}), false);
  EXPECT_EQ(AttrUtils::SetListListFloat(op_desc, "key", {}), false);
  EXPECT_EQ(AttrUtils::SetListTensorDesc(op_desc, "key", {}), false);
  EXPECT_EQ(AttrUtils::SetListStr(op_desc, "key", {}), false);
  std::vector<Buffer> buffer;
  EXPECT_EQ(AttrUtils::SetZeroCopyListBytes(op_desc, "key", buffer), false);
}
TEST_F(UtestGeAttrValue, GetNullObjectAttr) {
  OpDescPtr op_desc(nullptr);
  std::string value;
  EXPECT_EQ(AttrUtils::GetStr(op_desc, "key", value), false);
  int64_t i;
  EXPECT_EQ(AttrUtils::GetInt(op_desc, "key", i), false);
  GeTensorDesc ge_tensor_desc;
  EXPECT_EQ(AttrUtils::GetTensorDesc(op_desc, "key", ge_tensor_desc), false);
  ConstGeTensorPtr const_ge_tensor;
  EXPECT_EQ(AttrUtils::GetTensor(op_desc, "key", const_ge_tensor), false);
  bool flag;
  EXPECT_EQ(AttrUtils::GetBool(op_desc, "key", flag), false);
  Buffer buffer;
  EXPECT_EQ(AttrUtils::GetBytes(op_desc, "key", buffer), false);
  float j;
  EXPECT_EQ(AttrUtils::GetFloat(op_desc, "key", j), false);
  ComputeGraphPtr compute_graph;
  EXPECT_EQ(AttrUtils::GetGraph(op_desc, "key", compute_graph), false);
  DataType data_type;
  EXPECT_EQ(AttrUtils::GetDataType(op_desc, "key", data_type), false);
  std::vector<DataType> data_types;
  EXPECT_EQ(AttrUtils::GetListDataType(op_desc, "key", data_types), false);
  std::vector<std::vector<int64_t>> ints_list;
  EXPECT_EQ(AttrUtils::GetListListInt(op_desc, "key", ints_list), false);
  std::vector<int64_t> ints;
  EXPECT_EQ(AttrUtils::GetListInt(op_desc, "key", ints), false);
  std::vector<ConstGeTensorPtr> tensors;
  EXPECT_EQ(AttrUtils::GetListTensor(op_desc, "key", tensors), false);
  std::vector<bool> flags;
  EXPECT_EQ(AttrUtils::GetListBool(op_desc, "key", flags), false);
  std::vector<float> floats;
  EXPECT_EQ(AttrUtils::GetListFloat(op_desc, "key", floats), false);
  std::vector<Buffer> buffers;
  EXPECT_EQ(AttrUtils::GetListBytes(op_desc, "key", buffers), false);
  std::vector<ComputeGraphPtr> graphs;
  EXPECT_EQ(AttrUtils::GetListGraph(op_desc, "key", graphs), false);
  std::vector<std::vector<float>> floats_list;
  EXPECT_EQ(AttrUtils::GetListListFloat(op_desc, "key", floats_list), false);
  std::vector<GeTensorDesc> tensor_descs;
  EXPECT_EQ(AttrUtils::GetListTensorDesc(op_desc, "key", tensor_descs), false);
  std::vector<std::string> strings;
  EXPECT_EQ(AttrUtils::GetListStr(op_desc, "key", strings), false);
  EXPECT_EQ(AttrUtils::GetZeroCopyListBytes(op_desc, "key", buffers), false);
}

TEST_F(UtestGeAttrValue, SetGetAttrValue_Comprehensive) {
  // 综合测试所有功能
  AttrValue attr_value;

  // 测试所有支持的类型
  std::vector<std::pair<std::string, std::function<void()>>> test_cases = {
      {"int64_t", [&]() {
        int64_t val = 12345;
        EXPECT_EQ(attr_value.SetAttrValue(val), GRAPH_SUCCESS);
        int64_t get_val = 0;
        EXPECT_EQ(attr_value.GetAttrValue(get_val), GRAPH_SUCCESS);
        EXPECT_EQ(get_val, val);
      }},
      {"float32_t", [&]() {
        float32_t val = 3.14159f;
        EXPECT_EQ(attr_value.SetAttrValue(val), GRAPH_SUCCESS);
        float32_t get_val = 0.0f;
        EXPECT_EQ(attr_value.GetAttrValue(get_val), GRAPH_SUCCESS);
        EXPECT_FLOAT_EQ(get_val, val);
      }},
      {"bool", [&]() {
        bool val = true;
        EXPECT_EQ(attr_value.SetAttrValue(val), GRAPH_SUCCESS);
        bool get_val = false;
        EXPECT_EQ(attr_value.GetAttrValue(get_val), GRAPH_SUCCESS);
        EXPECT_EQ(get_val, val);
      }},
      {"DataType", [&]() {
        ge::DataType val = DT_FLOAT;
        EXPECT_EQ(attr_value.SetAttrValue(val), GRAPH_SUCCESS);
        ge::DataType get_val = DT_UNDEFINED;
        EXPECT_EQ(attr_value.GetAttrValue(get_val), GRAPH_SUCCESS);
        EXPECT_EQ(get_val, val);
      }},
      {"vector<int64_t>", [&]() {
        std::vector<int64_t> val = {1, 2, 3, 4, 5};
        EXPECT_EQ(attr_value.SetAttrValue(val), GRAPH_SUCCESS);
        std::vector<int64_t> get_val;
        EXPECT_EQ(attr_value.GetAttrValue(get_val), GRAPH_SUCCESS);
        EXPECT_EQ(get_val, val);
      }},
      {"vector<float32_t>", [&]() {
        std::vector<float32_t> val = {1.1f, 2.2f, 3.3f};
        EXPECT_EQ(attr_value.SetAttrValue(val), GRAPH_SUCCESS);
        std::vector<float32_t> get_val;
        EXPECT_EQ(attr_value.GetAttrValue(get_val), GRAPH_SUCCESS);
        EXPECT_EQ(get_val.size(), val.size());
        for (size_t i = 0; i < val.size(); ++i) {
          EXPECT_FLOAT_EQ(get_val[i], val[i]);
        }
      }},
      {"vector<bool>", [&]() {
        std::vector<bool> val = {true, false, true};
        EXPECT_EQ(attr_value.SetAttrValue(val), GRAPH_SUCCESS);
        std::vector<bool> get_val;
        EXPECT_EQ(attr_value.GetAttrValue(get_val), GRAPH_SUCCESS);
        EXPECT_EQ(get_val, val);
      }},
      {"vector<vector<int64_t>>", [&]() {
        std::vector<std::vector<int64_t>> val = {{1, 2}, {3, 4, 5}};
        EXPECT_EQ(attr_value.SetAttrValue(val), GRAPH_SUCCESS);
        std::vector<std::vector<int64_t>> get_val;
        EXPECT_EQ(attr_value.GetAttrValue(get_val), GRAPH_SUCCESS);
        EXPECT_EQ(get_val, val);
      }},
      {"vector<DataType>", [&]() {
        std::vector<ge::DataType> val = {DT_FLOAT, DT_INT32, DT_BOOL};
        EXPECT_EQ(attr_value.SetAttrValue(val), GRAPH_SUCCESS);
        std::vector<ge::DataType> get_val;
        EXPECT_EQ(attr_value.GetAttrValue(get_val), GRAPH_SUCCESS);
        EXPECT_EQ(get_val, val);
      }},
      {"AscendString", [&]() {
        AscendString val("test_ascend_string");
        EXPECT_EQ(attr_value.SetAttrValue(val), GRAPH_SUCCESS);
        AscendString get_val;
        EXPECT_EQ(attr_value.GetAttrValue(get_val), GRAPH_SUCCESS);
        EXPECT_STREQ(get_val.GetString(), val.GetString());
      }},
      {"vector<AscendString>", [&]() {
        std::vector<AscendString> val = {
            AscendString("str1"),
            AscendString("str2"),
            AscendString("str3")
        };
        EXPECT_EQ(attr_value.SetAttrValue(val), GRAPH_SUCCESS);
        std::vector<AscendString> get_val;
        EXPECT_EQ(attr_value.GetAttrValue(get_val), GRAPH_SUCCESS);
        EXPECT_EQ(get_val.size(), val.size());
        for (size_t i = 0; i < val.size(); ++i) {
          EXPECT_STREQ(get_val[i].GetString(), val[i].GetString());
        }
      }},
      {"Tensor", [&]() {
        TensorDesc tensor_desc(Shape({4, 4}), FORMAT_ND, DT_FLOAT);
        std::vector<uint8_t> tensor_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        Tensor val(tensor_desc, tensor_data);
        EXPECT_EQ(attr_value.SetAttrValue(val), GRAPH_SUCCESS);
        Tensor get_val;
        EXPECT_EQ(attr_value.GetAttrValue(get_val), GRAPH_SUCCESS);
        EXPECT_EQ(get_val.GetSize(), val.GetSize());
      }},
      {"vector<Tensor>", [&]() {
        TensorDesc tensor_desc(Shape({2, 2}), FORMAT_ND, DT_FLOAT);
        std::vector<uint8_t> tensor_data = {1, 2, 3, 4};
        Tensor tensor1(tensor_desc, tensor_data);
        Tensor tensor2(tensor_desc, tensor_data);
        std::vector<Tensor> val = {tensor1, tensor2};
        EXPECT_EQ(attr_value.SetAttrValue(val), GRAPH_SUCCESS);
        std::vector<Tensor> get_val;
        EXPECT_EQ(attr_value.GetAttrValue(get_val), GRAPH_SUCCESS);
        EXPECT_EQ(get_val.size(), val.size());
        for (size_t i = 0; i < val.size(); ++i) {
          EXPECT_EQ(get_val[i].GetSize(), val[i].GetSize());
        }
      }}
  };

  // 执行所有测试用例
  for (const auto &test_case : test_cases) {
    SCOPED_TRACE("Testing: " + test_case.first);
    test_case.second();
  }
}
// extern "C" wrapper for AttrValue SetAttrValue methods to avoid C++ name mangling
extern "C" {
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus aclCom_AttrValue_SetAttrValue_Tensor(void *attr_value_ptr,
                                                                                                const void *value) {
  if (attr_value_ptr == nullptr || value == nullptr) {
    return GRAPH_FAILED;
  }
  auto *attr_value = static_cast<ge::AttrValue *>(attr_value_ptr);
  auto *tensor = static_cast<const ge::Tensor *>(value);
  return attr_value->SetAttrValue(*tensor);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus aclCom_AttrValue_SetAttrValue_Int64(void *attr_value_ptr,
                                                                                               int64_t value) {
  if (attr_value_ptr == nullptr) {
    return GRAPH_FAILED;
  }
  auto *attr_value = static_cast<ge::AttrValue *>(attr_value_ptr);
  return attr_value->SetAttrValue(value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus aclCom_AttrValue_SetAttrValue_String(void *attr_value_ptr,
                                                                                                const char_t *value) {
  if (attr_value_ptr == nullptr || value == nullptr) {
    return GRAPH_FAILED;
  }
  auto *attr_value = static_cast<ge::AttrValue *>(attr_value_ptr);
  return attr_value->SetAttrValue(ge::AscendString(value));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus aclCom_AttrValue_SetAttrValue_Bool(void *attr_value_ptr,
                                                                                              bool value) {
  if (attr_value_ptr == nullptr) {
    return GRAPH_FAILED;
  }
  auto *attr_value = static_cast<ge::AttrValue *>(attr_value_ptr);
  return attr_value->SetAttrValue(value);
}
}
TEST_F(UtestGeAttrValue, ExternC_AttrValue_SetAttrValue_Tensor_Success) {
  AttrValue attr_value;
  Tensor tensor_value;

  // 测试成功情况
  EXPECT_EQ(aclCom_AttrValue_SetAttrValue_Tensor(&attr_value, &tensor_value), GRAPH_SUCCESS);

  // 测试nullptr参数
  EXPECT_EQ(aclCom_AttrValue_SetAttrValue_Tensor(nullptr, &tensor_value), GRAPH_FAILED);
}

TEST_F(UtestGeAttrValue, ExternC_AttrValue_SetAttrValue_Int64_Success) {
  AttrValue attr_value;
  int64_t int64_value = 12345;

  // 测试成功情况
  EXPECT_EQ(aclCom_AttrValue_SetAttrValue_Int64(&attr_value, int64_value), GRAPH_SUCCESS);

  // 验证设置的值
  int64_t get_value = 0;
  EXPECT_EQ(attr_value.GetAttrValue(get_value), GRAPH_SUCCESS);
  EXPECT_EQ(get_value, int64_value);

  // 测试nullptr参数
  EXPECT_EQ(aclCom_AttrValue_SetAttrValue_Int64(nullptr, int64_value), GRAPH_FAILED);
}

TEST_F(UtestGeAttrValue, ExternC_AttrValue_SetAttrValue_String_Success) {
  AttrValue attr_value;
  const char_t *char_value = "12345";

  // 测试成功情况
  EXPECT_EQ(aclCom_AttrValue_SetAttrValue_String(&attr_value, char_value), GRAPH_SUCCESS);

  // 验证设置的值
  ge::AscendString get_value = "0";
  EXPECT_EQ(attr_value.GetAttrValue(get_value), GRAPH_SUCCESS);
  EXPECT_EQ(get_value, char_value);

  // 测试nullptr参数
  EXPECT_EQ(aclCom_AttrValue_SetAttrValue_String(nullptr, char_value), GRAPH_FAILED);
}

TEST_F(UtestGeAttrValue, ExternC_AttrValue_SetAttrValue_Int64_EdgeCases) {
  AttrValue attr_value;

  // 测试边界值
  EXPECT_EQ(aclCom_AttrValue_SetAttrValue_Int64(&attr_value, std::numeric_limits<int64_t>::max()), GRAPH_SUCCESS);
  EXPECT_EQ(aclCom_AttrValue_SetAttrValue_Int64(&attr_value, std::numeric_limits<int64_t>::min()), GRAPH_SUCCESS);
  EXPECT_EQ(aclCom_AttrValue_SetAttrValue_Int64(&attr_value, 0), GRAPH_SUCCESS);
  EXPECT_EQ(aclCom_AttrValue_SetAttrValue_Int64(&attr_value, -1), GRAPH_SUCCESS);
}

TEST_F(UtestGeAttrValue, ExternC_AttrValue_SetAttrValue_Bool_Success) {
  AttrValue attr_value;
  bool bool_value = true;

  // 测试成功情况
  EXPECT_EQ(aclCom_AttrValue_SetAttrValue_Bool(&attr_value, bool_value), GRAPH_SUCCESS);

  // 验证设置的值
  bool get_value = false;
  EXPECT_EQ(attr_value.GetAttrValue(get_value), GRAPH_SUCCESS);
  EXPECT_EQ(get_value, bool_value);

  // 测试nullptr参数
  EXPECT_EQ(aclCom_AttrValue_SetAttrValue_Bool(nullptr, bool_value), GRAPH_FAILED);
}

TEST_F(UtestGeAttrValue, ExternC_AttrValue_SetAttrValue_Bool_EdgeCases) {
  AttrValue attr_value;

  // 测试true值
  EXPECT_EQ(aclCom_AttrValue_SetAttrValue_Bool(&attr_value, true), GRAPH_SUCCESS);
  bool get_value = false;
  EXPECT_EQ(attr_value.GetAttrValue(get_value), GRAPH_SUCCESS);
  EXPECT_EQ(get_value, true);

  // 测试false值
  EXPECT_EQ(aclCom_AttrValue_SetAttrValue_Bool(&attr_value, false), GRAPH_SUCCESS);
  EXPECT_EQ(attr_value.GetAttrValue(get_value), GRAPH_SUCCESS);
  EXPECT_EQ(get_value, false);
}
}
