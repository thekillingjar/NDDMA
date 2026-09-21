/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/bg_kernel_context_extend.h"
#include <gtest/gtest.h>
#include <memory>
#include <cmath>
#include "graph/compute_graph.h"
#include "graph/utils/node_utils.h"
#include "exe_graph/runtime/context_extend.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "runtime/runtime_attrs_def.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/lowering/bg_ir_attrs.h"
#include "graph/debug/ge_attr_define.h"
#include "expand_dimension.h"

namespace gert {
class BgIrAttrsUT : public testing::Test {};
// 构造tensorAttr，其shape小于size，测试AppendTensorAtrr函数能够正常内存拷贝
TEST_F(BgIrAttrsUT, ShapeSmallerThanSizeOfTensorAttr) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc ge_td;
  ge_td.SetOriginFormat(ge::FORMAT_NHWC);
  ge_td.SetFormat(ge::FORMAT_NHWC);
  ge_td.SetDataType(ge::DT_FLOAT16);
  ge_td.SetOriginShape(ge::GeShape({10, 10}));
  ge_td.SetShape(ge::GeShape({10, 10}));
  ge::GeTensor ge_tensor(ge_td);
  std::vector<uint16_t> fake_data(12 * 12);
  for (size_t i = 0; i < fake_data.size(); ++i) {
    fake_data[i] = static_cast<uint16_t>(i % std::numeric_limits<uint16_t>::max());
  }
  ge_tensor.SetData(reinterpret_cast<uint8_t *>(fake_data.data()), fake_data.size() * 2);
  ge::AttrUtils::SetTensor(op_desc, "a1", ge_tensor);
  op_desc->AppendIrAttrName("a1");

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  ASSERT_NE(ret, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  auto attrs = compute_node_info->GetAttrs();
  ASSERT_NE(attrs, nullptr);
  EXPECT_EQ(attrs->GetAttrNum(), 1);

  auto gert_tensor = attrs->GetAttrPointer<gert::Tensor>(0);
  EXPECT_EQ(attrs->GetTensor(0), gert_tensor);
  ASSERT_NE(gert_tensor, nullptr);
  EXPECT_EQ(gert_tensor->GetOriginShape(), gert::Shape({10, 10}));
  EXPECT_EQ(gert_tensor->GetStorageShape(), gert::Shape({10, 10}));
  EXPECT_EQ(gert_tensor->GetOriginFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(gert_tensor->GetStorageFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(gert_tensor->GetDataType(), ge::DT_FLOAT16);
  auto gert_tensor_ptr = gert_tensor->GetData<uint16_t>();
  EXPECT_NE(gert_tensor_ptr, nullptr);
  for (size_t i = 0; i < 10 * 10; ++i) {
    EXPECT_EQ(gert_tensor_ptr[i], static_cast<uint16_t>(i % std::numeric_limits<uint16_t>::max()));
  }
}
TEST_F(BgIrAttrsUT, CreateDataTypeAttrBuffer) {
  auto op_desc = std::make_shared<ge::OpDesc>("foo", "Foo");
  op_desc->AppendIrAttrName("dtype");
  EXPECT_EQ(op_desc->GetIrAttrNames().size(), 1U);
  EXPECT_EQ(op_desc->GetIrAttrNames().at(0), "dtype");
  ge::AttrUtils::SetDataType(op_desc, "dtype", ge::DT_INT32);
  auto node = ge::NodeUtils::CreatNodeWithoutGraph(op_desc);
  size_t attr_size;
  auto attr_buffer = bg::CreateAttrBuffer(node, attr_size);
  auto rt_attr_def = reinterpret_cast<RuntimeAttrsDef *>(attr_buffer.get());
  ASSERT_NE(rt_attr_def, nullptr);
  EXPECT_EQ(rt_attr_def->attr_num, 1U);
  for (size_t i = 0U; i < 40U; ++i) {
    EXPECT_EQ(rt_attr_def->reserved_[i], 0);
  }
  EXPECT_EQ(rt_attr_def->offset[0], 2 * sizeof(size_t) + sizeof(rt_attr_def->reserved_));
  auto base = reinterpret_cast<uint8_t *>(attr_buffer.get());
  EXPECT_EQ(*reinterpret_cast<ge::DataType *>(base + rt_attr_def->offset[0]), ge::DT_INT32);
}

TEST_F(BgIrAttrsUT, CreateAttrBufferSuccessOpLossAttr) {
  auto op_desc = std::make_shared<ge::OpDesc>("foo", "Foo");
  op_desc->AppendIrAttrName("dtype");
  EXPECT_EQ(op_desc->GetIrAttrNames().size(), 1U);
  EXPECT_EQ(op_desc->GetIrAttrNames().at(0), "dtype");
  auto node = ge::NodeUtils::CreatNodeWithoutGraph(op_desc);
  size_t attr_size;
  auto attr_buffer = bg::CreateAttrBuffer(node, attr_size);
  size_t gt_attr_size = sizeof(RuntimeAttrsDef);
  EXPECT_EQ(attr_size, gt_attr_size);
  auto base = reinterpret_cast<size_t*>(attr_buffer.get());
  EXPECT_EQ(base[0], 0U);
  EXPECT_EQ(base[1], 0U); // todo 原始用例，没加预留字段之前，base[1]为啥能取到值
}

TEST_F(BgIrAttrsUT, CreateListListIntAttrBuffer_Int64Ok) {
  auto op_desc = std::make_shared<ge::OpDesc>("foo", "Foo");
  op_desc->AppendIrAttrName("axes");
  EXPECT_EQ(op_desc->GetIrAttrNames().size(), 1U);
  EXPECT_EQ(op_desc->GetIrAttrNames().at(0), "axes");
  std::vector<std::vector<int64_t>> value{{}, {1}, {1, 2}, {1, 2, 3}};
  ge::AttrUtils::SetListListInt(op_desc, "axes", value);
  auto node = ge::NodeUtils::CreatNodeWithoutGraph(op_desc);
  size_t attr_size;
  auto attr_buffer = bg::CreateAttrBuffer(node, attr_size);
  auto rt_attr_def = reinterpret_cast<RuntimeAttrsDef *>(attr_buffer.get());
  ASSERT_NE(rt_attr_def, nullptr);
  EXPECT_EQ(rt_attr_def->attr_num, 1U);
  for (size_t i = 0U; i < 40U; ++i) {
    EXPECT_EQ(rt_attr_def->reserved_[i], 0);
  }
  EXPECT_EQ(rt_attr_def->offset[0], 2 * sizeof(size_t) + sizeof(rt_attr_def->reserved_));

  auto base = reinterpret_cast<uint8_t *>(attr_buffer.get());
  auto cvv = reinterpret_cast<ContinuousVectorVector *>(base + rt_attr_def->offset[0]);
  ASSERT_NE(cvv, nullptr);
  ASSERT_EQ(cvv->GetSize(), value.size());
  for (size_t i = 0U; i < value.size(); ++i) {
    auto cv = cvv->Get(i);
    ASSERT_NE(cv, nullptr);
    ASSERT_EQ(cv->GetSize(), value[i].size());
    ASSERT_EQ(cv->GetSize(), cv->GetCapacity());
    auto data = reinterpret_cast<const int64_t *>(cv->GetData());
    for (size_t j = 0U; j < value[i].size(); ++j) {
      EXPECT_EQ(data[j], value[i][j]);
    }
  }
}

TEST_F(BgIrAttrsUT, CreateListListIntAttrBuffer_Float64) {
  auto op_desc = std::make_shared<ge::OpDesc>("foo", "Foo");
  op_desc->AppendIrAttrName("axes");
  EXPECT_EQ(op_desc->GetIrAttrNames().size(), 1U);
  EXPECT_EQ(op_desc->GetIrAttrNames().at(0), "axes");
  std::vector<std::vector<float_t>> value{{}, {1}, {1, 2}, {1, 2, 3}};
  ge::AttrUtils::SetListListFloat(op_desc, "axes", value);
  auto node = ge::NodeUtils::CreatNodeWithoutGraph(op_desc);
  size_t attr_size;
  auto attr_buffer = bg::CreateAttrBuffer(node, attr_size);
  auto rt_attr_def = reinterpret_cast<RuntimeAttrsDef *>(attr_buffer.get());
  ASSERT_NE(rt_attr_def, nullptr);
  EXPECT_EQ(rt_attr_def->attr_num, 1U);
  for (size_t i = 0U; i < 40U; ++i) {
    EXPECT_EQ(rt_attr_def->reserved_[i], 0);
  }
  EXPECT_EQ(rt_attr_def->offset[0], 2 * sizeof(size_t) + sizeof(rt_attr_def->reserved_));

  auto base = reinterpret_cast<uint8_t *>(attr_buffer.get());
  auto cvv = reinterpret_cast<ContinuousVectorVector *>(base + rt_attr_def->offset[0]);
  ASSERT_NE(cvv, nullptr);
  ASSERT_EQ(cvv->GetSize(), value.size());
  for (size_t i = 0U; i < value.size(); ++i) {
    auto cv = cvv->Get(i);
    ASSERT_NE(cv, nullptr);
    ASSERT_EQ(cv->GetSize(), value[i].size());
    ASSERT_EQ(cv->GetSize(), cv->GetCapacity());
    auto data = reinterpret_cast<const float_t *>(cv->GetData());
    for (size_t j = 0U; j < value[i].size(); ++j) {
      EXPECT_EQ(data[j], value[i][j]);
    }
  }
}
TEST_F(BgIrAttrsUT, CreateStringAttrBuffer) {
  auto op_desc = std::make_shared<ge::OpDesc>("foo", "Foo");
  op_desc->AppendIrAttrName("demo_str");
  EXPECT_EQ(op_desc->GetIrAttrNames().size(), 1U);
  EXPECT_EQ(op_desc->GetIrAttrNames().at(0), "demo_str");
  std::string str_attr = "hello";
  ge::AttrUtils::SetStr(op_desc, "demo_str", str_attr);
  auto node = ge::NodeUtils::CreatNodeWithoutGraph(op_desc);
  size_t attr_size;
  auto attr_buffer = bg::CreateAttrBuffer(node, attr_size);

  auto base = reinterpret_cast<char *>(reinterpret_cast<RuntimeAttrsDef *>(attr_buffer.get())->offset + 1);
  EXPECT_STREQ(base, "hello");
}

TEST_F(BgIrAttrsUT, CreateListStringAttrBuffer) {
  auto op_desc = std::make_shared<ge::OpDesc>("foo", "Foo");
  op_desc->AppendIrAttrName("demo_str");
  EXPECT_EQ(op_desc->GetIrAttrNames().size(), 1U);
  EXPECT_EQ(op_desc->GetIrAttrNames().at(0), "demo_str");
  std::string str_attr1 = "hello";
  std::string str_attr2 = "world";
  std::string str_attr3 = "good";
  std::string str_attr4 = "job";
  std::vector<std::string> str_atts = {str_attr1, str_attr2, str_attr3, str_attr4};
  ge::AttrUtils::SetListStr(op_desc, "demo_str", str_atts);
  auto node = ge::NodeUtils::CreatNodeWithoutGraph(op_desc);
  size_t attr_size;
  auto attr_buffer = bg::CreateAttrBuffer(node, attr_size);
  auto attr_def = reinterpret_cast<RuntimeAttrsDef *>(attr_buffer.get());
  auto base =
    reinterpret_cast<const gert::ContinuousVector *>(ge::PtrToPtr<const RuntimeAttrsDef, const uint8_t>(attr_def)
        + attr_def->offset[0]);
  ASSERT_NE(base, nullptr);
  size_t str_attrs_len = 0U;
  for (const auto &str_attr : str_atts) {
    str_attrs_len += strlen(str_attr.c_str()) + 1U;
  }
  size_t gt_attr_size = sizeof(ContinuousVector) + sizeof(RuntimeAttrsDef) + 1 * sizeof(size_t) + +str_attrs_len;
  EXPECT_EQ(attr_size, gt_attr_size);
  ASSERT_EQ(base->GetSize(), 4);
  EXPECT_STREQ(reinterpret_cast<const char *>(base->GetData()), "hello");
  EXPECT_STREQ(reinterpret_cast<const char *>(base->GetData()) + 6, "world");
  EXPECT_STREQ(reinterpret_cast<const char *>(base->GetData()) + 12, "good");
  EXPECT_STREQ(reinterpret_cast<const char *>(base->GetData()) + 17, "job");
}

TEST_F(BgIrAttrsUT, CreateAttrBufferWithoutIrAttr) {
  auto op_desc = std::make_shared<ge::OpDesc>("foo", "Foo");
  op_desc->AppendIrAttrName("dtype");
  ge::AttrUtils::SetDataType(op_desc, "dtype", ge::DT_INT32);

  auto node = ge::NodeUtils::CreatNodeWithoutGraph(op_desc);
  size_t attr_size;
  ge::AnyValue value = ge::AnyValue::CreateFrom<int64_t>(2);
  auto attr_buffer = bg::CreateAttrBufferWithoutIr(node, {value}, attr_size);
  size_t gt_attr_size = sizeof(RuntimeAttrsDef) + sizeof(size_t) + sizeof(int64_t);
  EXPECT_EQ(attr_size, gt_attr_size);
  auto rt_attr_def = reinterpret_cast<RuntimeAttrsDef *>(attr_buffer.get());
  ASSERT_NE(rt_attr_def, nullptr);
  EXPECT_EQ(rt_attr_def->attr_num, 1U);
  EXPECT_EQ(rt_attr_def->offset[0], sizeof(size_t) + sizeof(RuntimeAttrsDef));
  auto base = reinterpret_cast<int8_t *>(rt_attr_def);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(base + rt_attr_def->offset[0]), 2);
}
}  // namespace gert
