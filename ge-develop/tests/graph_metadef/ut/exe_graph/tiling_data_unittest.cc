/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/tiling_data.h"
#include "common/util/tiling_utils.h"
#include "faker/kernel_run_context_faker.h"
#include "framework/common/debug/ge_log.h"
#include <gtest/gtest.h>
namespace gert {
class TilingDataUT : public testing::Test {};
namespace {
struct TestData {
  int64_t a;
  int32_t b;
  int16_t c;
  int16_t d;
};

FakeKernelContextHolder BuildTestContext() {
  auto holder = gert::KernelRunContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .NodeAttrs({{"str", ge::AnyValue::CreateFrom<std::string>("Hello!")},
                                {"float", ge::AnyValue::CreateFrom<float>(10.101)},
                                {"list_float", ge::AnyValue::CreateFrom<std::vector<float>>({1.2, 2.3, 3.4})},
                                {"list_list_float", ge::AnyValue::CreateFrom<std::vector<std::vector<float>>>(
                                    {{1.2, 2.3, 3.4}, {4.5, 5.6, 6.7}})},
                                {"int", ge::AnyValue::CreateFrom<int64_t>(0x7fUL)},
                                {"list_int", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 3})},
                                {"list_list_int", ge::AnyValue::CreateFrom<std::vector<std::vector<int64_t>>>(
                                    {{1, 2, 3}, {4, 5, 6}})},
                                {"bool", ge::AnyValue::CreateFrom<bool>(true)},
                                {"list_bool", ge::AnyValue::CreateFrom<std::vector<bool>>({true, false, true})}})
                    .Build();
  return holder;
}
}  // namespace

TEST_F(TilingDataUT, AppendSameTypesOk) {
  auto data = TilingData::CreateCap(2048);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  ASSERT_NE(tiling_data, nullptr);
  std::vector<int64_t> expect_vec;
  for (int64_t i = 0; i < 10; ++i) {
    tiling_data->Append(i);
    expect_vec.push_back(i);
  }
  ASSERT_EQ(tiling_data->GetDataSize(), 80);
  EXPECT_EQ(memcmp(tiling_data->GetData(), expect_vec.data(), tiling_data->GetDataSize()), 0);
}

TEST_F(TilingDataUT, ExpandTest) {
  auto data = TilingData::CreateCap(1024);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  ASSERT_NE(tiling_data, nullptr);
  std::vector<int64_t> gold;
  for (int64_t elem = 0; elem < 10; ++elem) {
    tiling_data->Append(elem);
    gold.push_back(elem);
  }
  int64_t test_data = 666;
  ASSERT_EQ(tiling_data->GetDataSize(), 80);
  EXPECT_EQ(memcmp(tiling_data->GetData(), gold.data(), tiling_data->GetDataSize()), 0);
  auto ptr = tiling_data->Expand(2048);
  EXPECT_EQ(ptr, nullptr);
  ptr = tiling_data->Expand(sizeof(int64_t));
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(tiling_data->GetDataSize(), 80 + sizeof(int64_t));
  *reinterpret_cast<int64_t *>(ptr) = test_data;
  gold.push_back(test_data);
  EXPECT_EQ(memcmp(tiling_data->GetData(), gold.data(), tiling_data->GetDataSize()), 0);
}

TEST_F(TilingDataUT, AppendStructOk) {
  auto data = TilingData::CreateCap(2048);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  ASSERT_NE(tiling_data, nullptr);
  TestData td{.a = 1024, .b = 512, .c = 256, .d = 128};
  tiling_data->Append(td);
  ASSERT_EQ(tiling_data->GetDataSize(), sizeof(td));
  EXPECT_EQ(memcmp(tiling_data->GetData(), &td, tiling_data->GetDataSize()), 0);
}

TEST_F(TilingDataUT, AppendDifferentTypes) {
  auto data = TilingData::CreateCap(2048);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  ASSERT_NE(tiling_data, nullptr);
  std::vector<int64_t> expect_vec1;
  for (int64_t i = 0; i < 10; ++i) {
    tiling_data->Append(i);
    expect_vec1.push_back(i);
  }

  std::vector<int32_t> expect_vec2;
  for (int32_t i = 0; i < 3; ++i) {
    tiling_data->Append(i);
    expect_vec2.push_back(i);
  }

  TestData td{.a = 1024, .b = 512, .c = 256, .d = 128};
  tiling_data->Append(td);

  ASSERT_EQ(tiling_data->GetDataSize(), 80 + 12 + sizeof(TestData));
  EXPECT_EQ(memcmp(tiling_data->GetData(), expect_vec1.data(), 80), 0);
  EXPECT_EQ(memcmp(reinterpret_cast<uint8_t *>(tiling_data->GetData()) + 80, expect_vec2.data(), 12), 0);
  EXPECT_EQ(memcmp(reinterpret_cast<uint8_t *>(tiling_data->GetData()) + 92, &td, sizeof(td)), 0);
}

TEST_F(TilingDataUT, AppendOutOfBounds) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  ASSERT_NE(tiling_data, nullptr);
  std::vector<int64_t> expect_vec1;
  for (int64_t i = 0; i < 10; ++i) {
    tiling_data->Append(i);
    expect_vec1.push_back(i);
  }

  std::vector<int64_t> expect_vec2;
  for (int64_t i = 0; i < 2; ++i) {
    tiling_data->Append(i);
    expect_vec2.push_back(i);
  }
  EXPECT_NE(tiling_data->Append(static_cast<int64_t>(3)), ge::GRAPH_SUCCESS);

  ASSERT_EQ(tiling_data->GetDataSize(), 16);
  EXPECT_EQ(memcmp(tiling_data->GetData(), expect_vec1.data(), 16), 0);
}

TEST_F(TilingDataUT, AppendAttrStrOk) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 0, AttrDataType::kString, AttrDataType::kString),
            ge::GRAPH_SUCCESS);
  std::string s1(reinterpret_cast<char *>(tiling_data->GetData()),
                 reinterpret_cast<char *>(tiling_data->GetData()) + 6);
  EXPECT_STREQ(s1.c_str(), "Hello!");
  EXPECT_EQ(tiling_data->GetDataSize(), 6);
  tiling_data->SetDataSize(0);
}

TEST_F(TilingDataUT, AppendAttrBoolOk) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 7, AttrDataType::kBool, AttrDataType::kBool),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<bool *>(tiling_data->GetData()), true);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(bool));
}

TEST_F(TilingDataUT, AppendAttrListBoolOk) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 8, AttrDataType::kListBool, AttrDataType::kListBool), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(bool) * 3);
  auto ele = reinterpret_cast<bool *>(tiling_data->GetData());
  std::vector<bool> expect_data{true, false, true};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrBoolToFloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 7, AttrDataType::kBool, AttrDataType::kFloat16),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint16_t *>(tiling_data->GetData()), optiling::OtherToFloat16<bool>(true));
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t));
}

TEST_F(TilingDataUT, AppendAttrListBoolToListFloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 8, AttrDataType::kListBool, AttrDataType::kListFloat16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 3);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<bool> expect_data{true, false, true};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], optiling::OtherToFloat16<bool>(expect_data[i]));
  }
}

TEST_F(TilingDataUT, AppendAttrBoolToBfloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 7, AttrDataType::kBool, AttrDataType::kBfloat16),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint16_t *>(tiling_data->GetData()), optiling::OtherToBfloat16<bool>(true));
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t));
}

TEST_F(TilingDataUT, AppendAttrListBoolToListBfloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 8, AttrDataType::kListBool, AttrDataType::kListBfloat16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 3);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<bool> expect_data{true, false, true};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], optiling::OtherToBfloat16<bool>(expect_data[i]));
  }
}

TEST_F(TilingDataUT, AppendAttrBoolToFloat32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 7, AttrDataType::kBool, AttrDataType::kFloat32),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<float *>(tiling_data->GetData()), 1.0f);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(float));
}

TEST_F(TilingDataUT, AppendAttrListBoolToListFloat32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 8, AttrDataType::kListBool, AttrDataType::kListFloat32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(float) * 3);
  auto ele = reinterpret_cast<float *>(tiling_data->GetData());
  std::vector<float> expect_data{1.0f, 0.0f, 1.0f};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrBoolToInt8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 7, AttrDataType::kBool, AttrDataType::kInt8),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int8_t *>(tiling_data->GetData()), 1);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int8_t));
}

TEST_F(TilingDataUT, AppendAttrListBoolToListInt8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 8, AttrDataType::kListBool, AttrDataType::kListInt8), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int8_t) * 3);
  auto ele = reinterpret_cast<int8_t *>(tiling_data->GetData());
  std::vector<int8_t> expect_data{1, 0, 1};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrBoolToInt16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 7, AttrDataType::kBool, AttrDataType::kInt16),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int16_t *>(tiling_data->GetData()), 1);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int16_t));
}

TEST_F(TilingDataUT, AppendAttrListBoolToListInt16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 8, AttrDataType::kListBool, AttrDataType::kListInt16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int16_t) * 3);
  auto ele = reinterpret_cast<int16_t *>(tiling_data->GetData());
  std::vector<int16_t> expect_data{1, 0, 1};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrBoolToInt32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 7, AttrDataType::kBool, AttrDataType::kInt32),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int32_t *>(tiling_data->GetData()), 1);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int32_t));
}

TEST_F(TilingDataUT, AppendAttrListBoolToListInt32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 8, AttrDataType::kListBool, AttrDataType::kListInt32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int32_t) * 3);
  auto ele = reinterpret_cast<int32_t *>(tiling_data->GetData());
  std::vector<int32_t> expect_data{1, 0, 1};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrBoolToInt64Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 7, AttrDataType::kBool, AttrDataType::kInt64),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(tiling_data->GetData()), 1);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int64_t));
}

TEST_F(TilingDataUT, AppendAttrListBoolToListInt64Ok) {
  auto data = TilingData::CreateCap(30);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 8, AttrDataType::kListBool, AttrDataType::kListInt64), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int64_t) * 3);
  auto ele = reinterpret_cast<int64_t *>(tiling_data->GetData());
  std::vector<int64_t> expect_data{1, 0, 1};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrBoolToUint8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 7, AttrDataType::kBool, AttrDataType::kUint8),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint8_t *>(tiling_data->GetData()), 1);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint8_t));
}

TEST_F(TilingDataUT, AppendAttrListBoolToListUint8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 8, AttrDataType::kListBool, AttrDataType::kListUint8), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint8_t) * 3);
  auto ele = reinterpret_cast<uint8_t *>(tiling_data->GetData());
  std::vector<uint8_t> expect_data{1, 0, 1};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrBoolToUint16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 7, AttrDataType::kBool, AttrDataType::kUint16),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint16_t *>(tiling_data->GetData()), 1);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t));
}

TEST_F(TilingDataUT, AppendAttrListBoolToListUint16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 8, AttrDataType::kListBool, AttrDataType::kListUint16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 3);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<uint16_t> expect_data{1, 0, 1};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrBoolToUint32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 7, AttrDataType::kBool, AttrDataType::kUint32),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(tiling_data->GetData()), 1);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint32_t));
}

TEST_F(TilingDataUT, AppendAttrListBoolToListUint32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 8, AttrDataType::kListBool, AttrDataType::kListUint32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint32_t) * 3);
  auto ele = reinterpret_cast<uint32_t *>(tiling_data->GetData());
  std::vector<uint32_t> expect_data{1, 0, 1};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrBoolToUint64Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 7, AttrDataType::kBool, AttrDataType::kUint64),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint64_t *>(tiling_data->GetData()), 1UL);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint64_t));
}

TEST_F(TilingDataUT, AppendAttrListBoolToListUint64Ok) {
  auto data = TilingData::CreateCap(30);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 8, AttrDataType::kListBool, AttrDataType::kListUint64), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int64_t) * 3);
  auto ele = reinterpret_cast<int64_t *>(tiling_data->GetData());
  std::vector<int64_t> expect_data{1UL, 0UL, 1UL};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrFloat32ToBoolOk) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 1, AttrDataType::kFloat32, AttrDataType::kBool), ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<bool *>(tiling_data->GetData()), true);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(bool));
}

TEST_F(TilingDataUT, AppendAttrListFloat32ToListBoolOk) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 2, AttrDataType::kListFloat32, AttrDataType::kListBool), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(bool) * 3);
  auto ele = reinterpret_cast<bool *>(tiling_data->GetData());
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], true);
  }
}

TEST_F(TilingDataUT, AppendAttrListListFloat32ToListListBoolOk) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 3, AttrDataType::kListListFloat32, AttrDataType::kListListBool), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(bool) * 6);
  auto ele = reinterpret_cast<bool *>(tiling_data->GetData());
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], true);
  }
}

TEST_F(TilingDataUT, AppendAttrFloat32ToFloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 1, AttrDataType::kFloat32, AttrDataType::kFloat16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint16_t *>(tiling_data->GetData()), optiling::Float32ToFloat16(10.101));
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t));
}

TEST_F(TilingDataUT, AppendAttrListFloat32ToListFloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 2, AttrDataType::kListFloat32, AttrDataType::kListFloat16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 3);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<float> expect_data = {1.2, 2.3, 3.4};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], optiling::Float32ToFloat16(expect_data[i]));
  }
}

TEST_F(TilingDataUT, AppendAttrListListFloat32ToListListFloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 3, AttrDataType::kListListFloat32, AttrDataType::kListListFloat16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 6);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<float> expect_data{1.2, 2.3, 3.4, 4.5, 5.6, 6.7};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], optiling::Float32ToFloat16(expect_data[i]));
  }
}

TEST_F(TilingDataUT, AppendAttrFloat32ToBfloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 1, AttrDataType::kFloat32, AttrDataType::kBfloat16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint16_t *>(tiling_data->GetData()), optiling::Float32ToBfloat16(10.101));
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t));
}

TEST_F(TilingDataUT, AppendAttrListFloat32ToListBfloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 2, AttrDataType::kListFloat32, AttrDataType::kListBfloat16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 3);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<float> expect_data = {1.2, 2.3, 3.4};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], optiling::Float32ToBfloat16(expect_data[i]));
  }
}

TEST_F(TilingDataUT, AppendAttrListListFloat32ToListListBfloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 3, AttrDataType::kListListFloat32, AttrDataType::kListListBfloat16),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 6);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<float> expect_data{1.2, 2.3, 3.4, 4.5, 5.6, 6.7};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], optiling::Float32ToBfloat16(expect_data[i]));
  }
}

TEST_F(TilingDataUT, AppendAttrFloat32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 1, AttrDataType::kFloat32, AttrDataType::kFloat32), ge::GRAPH_SUCCESS);
  EXPECT_FLOAT_EQ(*reinterpret_cast<float *>(tiling_data->GetData()), 10.101);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(float));
}

TEST_F(TilingDataUT, AppendAttrListFloat32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 2, AttrDataType::kListFloat32, AttrDataType::kListFloat32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(float) * 3);
  auto ele = reinterpret_cast<float *>(tiling_data->GetData());
  std::vector<float> expect_data{1.2, 2.3, 3.4};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListFloat32Ok) {
  auto data = TilingData::CreateCap(30);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 3, AttrDataType::kListListFloat32, AttrDataType::kListListFloat32),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(float) * 6);
  auto ele = reinterpret_cast<float *>(tiling_data->GetData());
  std::vector<float> expect_data{1.2, 2.3, 3.4, 4.5, 5.6, 6.7};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrFloat32ToInt8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 1, AttrDataType::kFloat32, AttrDataType::kInt8),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int8_t *>(tiling_data->GetData()), 10);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int8_t));
}

TEST_F(TilingDataUT, AppendAttrListFloat32ToListInt8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 2, AttrDataType::kListFloat32, AttrDataType::kListInt8), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int8_t) * 3);
  auto ele = reinterpret_cast<int8_t *>(tiling_data->GetData());
  std::vector<int8_t> expect_data{1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListFloat32ToListListInt8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 3, AttrDataType::kListListFloat32, AttrDataType::kListListInt8), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int8_t) * 6);
  auto ele = reinterpret_cast<int8_t *>(tiling_data->GetData());
  std::vector<int8_t> expect_data{1, 2, 3, 4, 5, 6};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrFloat32ToInt16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 1, AttrDataType::kFloat32, AttrDataType::kInt16),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int16_t *>(tiling_data->GetData()), 10);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int16_t));
}

TEST_F(TilingDataUT, AppendAttrListFloat32ToListInt16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 2, AttrDataType::kListFloat32, AttrDataType::kListInt16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int16_t) * 3);
  auto ele = reinterpret_cast<int16_t *>(tiling_data->GetData());
  std::vector<int16_t> expect_data{1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListFloat32ToListListInt16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 3, AttrDataType::kListListFloat32, AttrDataType::kListListInt16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int16_t) * 6);
  auto ele = reinterpret_cast<int16_t *>(tiling_data->GetData());
  std::vector<int16_t> expect_data{1, 2, 3, 4, 5, 6};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrFloat32ToInt32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 1, AttrDataType::kFloat32, AttrDataType::kInt32),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int32_t *>(tiling_data->GetData()), 10);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int32_t));
}

TEST_F(TilingDataUT, AppendAttrListFloat32ToListInt32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 2, AttrDataType::kListFloat32, AttrDataType::kListInt32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int32_t) * 3);
  auto ele = reinterpret_cast<int32_t *>(tiling_data->GetData());
  std::vector<int32_t> expect_data{1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListFloat32ToListListInt32Ok) {
  auto data = TilingData::CreateCap(30);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 3, AttrDataType::kListListFloat32, AttrDataType::kListListInt32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int32_t) * 6);
  auto ele = reinterpret_cast<int32_t *>(tiling_data->GetData());
  std::vector<int32_t> expect_data{1, 2, 3, 4, 5, 6};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrFloat32ToInt64Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 1, AttrDataType::kFloat32, AttrDataType::kInt64),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(tiling_data->GetData()), 10);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int64_t));
}

TEST_F(TilingDataUT, AppendAttrListFloat32ToListInt64Ok) {
  auto data = TilingData::CreateCap(30);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 2, AttrDataType::kListFloat32, AttrDataType::kListInt64), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int64_t) * 3);
  auto ele = reinterpret_cast<int64_t *>(tiling_data->GetData());
  std::vector<int64_t> expect_data{1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListFloat32ToListListInt64Ok) {
  auto data = TilingData::CreateCap(50);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 3, AttrDataType::kListListFloat32, AttrDataType::kListListInt64), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int64_t) * 6);
  auto ele = reinterpret_cast<int64_t *>(tiling_data->GetData());
  std::vector<int64_t> expect_data{1, 2, 3, 4, 5, 6};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrFloat32ToUint8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 1, AttrDataType::kFloat32, AttrDataType::kUint8),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint8_t *>(tiling_data->GetData()), 10);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint8_t));
}

TEST_F(TilingDataUT, AppendAttrListFloat32ToListUint8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 2, AttrDataType::kListFloat32, AttrDataType::kListUint8), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint8_t) * 3);
  auto ele = reinterpret_cast<uint8_t *>(tiling_data->GetData());
  std::vector<uint8_t> expect_data{1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListFloat32ToListListUint8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 3, AttrDataType::kListListFloat32, AttrDataType::kListListUint8), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint8_t) * 6);
  auto ele = reinterpret_cast<uint8_t *>(tiling_data->GetData());
  std::vector<uint8_t> expect_data{1, 2, 3, 4, 5, 6};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrFloat32ToUint16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 1, AttrDataType::kFloat32, AttrDataType::kUint16),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint16_t *>(tiling_data->GetData()), 10);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t));
}

TEST_F(TilingDataUT, AppendAttrListFloat32ToListUint16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 2, AttrDataType::kListFloat32, AttrDataType::kListUint16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 3);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<uint16_t> expect_data{1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListFloat32ToListListUint16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 3, AttrDataType::kListListFloat32, AttrDataType::kListListUint16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 6);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<uint16_t> expect_data{1, 2, 3, 4, 5, 6};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrFloat32ToUint32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 1, AttrDataType::kFloat32, AttrDataType::kUint32),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(tiling_data->GetData()), 10);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint32_t));
}

TEST_F(TilingDataUT, AppendAttrListFloat32ToListUint32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 2, AttrDataType::kListFloat32, AttrDataType::kListUint32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint32_t) * 3);
  auto ele = reinterpret_cast<uint32_t *>(tiling_data->GetData());
  std::vector<uint32_t> expect_data{1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListFloat32ToListListUint32Ok) {
  auto data = TilingData::CreateCap(30);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 3, AttrDataType::kListListFloat32, AttrDataType::kListListUint32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint32_t) * 6);
  auto ele = reinterpret_cast<uint32_t *>(tiling_data->GetData());
  std::vector<uint32_t> expect_data{1, 2, 3, 4, 5, 6};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrFloat32ToUint64Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 1, AttrDataType::kFloat32, AttrDataType::kUint64),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint64_t *>(tiling_data->GetData()), 10);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint64_t));
}

TEST_F(TilingDataUT, AppendAttrListFloat32ToListUint64Ok) {
  auto data = TilingData::CreateCap(30);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 2, AttrDataType::kListFloat32, AttrDataType::kListUint64), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint64_t) * 3);
  auto ele = reinterpret_cast<uint64_t *>(tiling_data->GetData());
  std::vector<uint64_t> expect_data{1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListFloat32ToListListUint64Ok) {
  auto data = TilingData::CreateCap(50);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 3, AttrDataType::kListListFloat32, AttrDataType::kListListUint64), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint64_t) * 6);
  auto ele = reinterpret_cast<uint64_t *>(tiling_data->GetData());
  std::vector<uint64_t> expect_data{1, 2, 3, 4, 5, 6};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrInt32ToBoolOk) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 4, AttrDataType::kInt32, AttrDataType::kBool), ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<bool *>(tiling_data->GetData()), true);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(bool));
}

TEST_F(TilingDataUT, AppendAttrListInt32ToListBoolOk) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt32, AttrDataType::kListBool), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(bool) * 3);
  std::vector<int32_t> expect_data{1, 0, 2};
  auto ele = reinterpret_cast<bool *>(tiling_data->GetData());
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], static_cast<bool>(expect_data[i]));
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt32ToListListBoolOk) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt32, AttrDataType::kListListBool), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(bool) * 6);
  std::vector<int32_t> expect_data{1, 0, 2, 4, 0, 5};
  auto ele = reinterpret_cast<bool *>(tiling_data->GetData());
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], static_cast<bool>(expect_data[i]));
  }
}

TEST_F(TilingDataUT, AppendAttrInt32ToFloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 4, AttrDataType::kInt32, AttrDataType::kFloat16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint16_t *>(tiling_data->GetData()), optiling::OtherToFloat16<int32_t>(0x7f));
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t));
}

TEST_F(TilingDataUT, AppendAttrListInt32ToListFloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt32, AttrDataType::kListFloat16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 3);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<int32_t> expect_data{1, 0, 2};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], optiling::OtherToFloat16<int32_t>(expect_data[i]));
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt32ToListListFloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt32, AttrDataType::kListListFloat16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 6);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<int32_t> expect_data{1, 0, 2, 4, 0, 5};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], optiling::OtherToFloat16<int32_t>(expect_data[i]));
  }
}

TEST_F(TilingDataUT, AppendAttrInt32ToBfloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 4, AttrDataType::kInt32, AttrDataType::kBfloat16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint16_t *>(tiling_data->GetData()), optiling::OtherToBfloat16<int32_t>(0x7f));
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t));
}

TEST_F(TilingDataUT, AppendAttrListInt32ToListBfloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt32, AttrDataType::kListBfloat16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 3);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<int32_t> expect_data{1, 0, 2};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], optiling::OtherToBfloat16<int32_t>(expect_data[i]));
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt32ToListListBfloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt32, AttrDataType::kListListBfloat16),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 6);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<int32_t> expect_data{1, 0, 2, 4, 0, 5};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], optiling::OtherToBfloat16<int32_t>(expect_data[i]));
  }
}

TEST_F(TilingDataUT, AppendAttrInt32ToFloat32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 4, AttrDataType::kInt32, AttrDataType::kFloat32), ge::GRAPH_SUCCESS);
  EXPECT_FLOAT_EQ(*reinterpret_cast<float *>(tiling_data->GetData()), static_cast<float>(0x7f));
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(float));
}

TEST_F(TilingDataUT, AppendAttrListInt32ToListFloat32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt32, AttrDataType::kListFloat32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(float) * 3);
  auto ele = reinterpret_cast<float *>(tiling_data->GetData());
  std::vector<float> expect_data{1.0, 0.0, 2.0};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt32ToListListFloat32Ok) {
  auto data = TilingData::CreateCap(30);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt32, AttrDataType::kListListFloat32),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(float) * 6);
  auto ele = reinterpret_cast<float *>(tiling_data->GetData());
  std::vector<float> expect_data{1.0, 0.0, 2.0, 4.0, 0.0, 5.0};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrInt32ToInt8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 4, AttrDataType::kInt32, AttrDataType::kInt8),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int8_t *>(tiling_data->GetData()), 0x7f);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int8_t));
}

TEST_F(TilingDataUT, AppendAttrListInt32ToListInt8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt32, AttrDataType::kListInt8), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int8_t) * 3);
  auto ele = reinterpret_cast<int8_t *>(tiling_data->GetData());
  std::vector<int8_t> expect_data{1, 0, 2};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt32ToListListInt8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt32, AttrDataType::kListListInt8), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int8_t) * 6);
  auto ele = reinterpret_cast<int8_t *>(tiling_data->GetData());
  std::vector<int8_t> expect_data{1, 0, 2, 4, 0, 5};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrInt32ToInt16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 4, AttrDataType::kInt32, AttrDataType::kInt16),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int16_t *>(tiling_data->GetData()), 0x7f);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int16_t));
}

TEST_F(TilingDataUT, AppendAttrListInt32ToListInt16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt32, AttrDataType::kListInt16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int16_t) * 3);
  auto ele = reinterpret_cast<int16_t *>(tiling_data->GetData());
  std::vector<int16_t> expect_data{1, 0, 2};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt32ToListListInt16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt32, AttrDataType::kListListInt16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int16_t) * 6);
  auto ele = reinterpret_cast<int16_t *>(tiling_data->GetData());
  std::vector<int16_t> expect_data{1, 0, 2, 4, 0, 5};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrInt32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 4, AttrDataType::kInt32, AttrDataType::kInt32),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int32_t *>(tiling_data->GetData()), 0x7f);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int32_t));
}

TEST_F(TilingDataUT, AppendAttrListInt32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt32, AttrDataType::kListInt32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int32_t) * 3);
  auto ele = reinterpret_cast<int32_t *>(tiling_data->GetData());
  std::vector<int32_t> expect_data{1, 0, 2};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt32Ok) {
  auto data = TilingData::CreateCap(30);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt32, AttrDataType::kListListInt32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int32_t) * 6);
  auto ele = reinterpret_cast<int32_t *>(tiling_data->GetData());
  std::vector<int32_t> expect_data{1, 0, 2, 4, 0, 5};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrInt32ToInt64Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 4, AttrDataType::kInt32, AttrDataType::kInt64),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(tiling_data->GetData()), 0x7f);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int64_t));
}

TEST_F(TilingDataUT, AppendAttrListInt32ToListInt64Ok) {
  auto data = TilingData::CreateCap(30);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt32, AttrDataType::kListInt64), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int64_t) * 3);
  auto ele = reinterpret_cast<int64_t *>(tiling_data->GetData());
  std::vector<int64_t> expect_data{1, 0, 2};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt32ToListListInt64Ok) {
  auto data = TilingData::CreateCap(50);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt32, AttrDataType::kListListInt64), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int64_t) * 6);
  auto ele = reinterpret_cast<int64_t *>(tiling_data->GetData());
  std::vector<int64_t> expect_data{1, 0, 2, 4, 0, 5};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrInt32ToUint8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 4, AttrDataType::kInt32, AttrDataType::kUint8),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint8_t *>(tiling_data->GetData()), 0x7f);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint8_t));
}

TEST_F(TilingDataUT, AppendAttrListInt32ToListUint8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt32, AttrDataType::kListUint8), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint8_t) * 3);
  auto ele = reinterpret_cast<uint8_t *>(tiling_data->GetData());
  std::vector<uint8_t> expect_data{1, 0, 2};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt32ToListListUint8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt32, AttrDataType::kListListUint8), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint8_t) * 6);
  auto ele = reinterpret_cast<uint8_t *>(tiling_data->GetData());
  std::vector<uint8_t> expect_data{1, 0, 2, 4, 0, 5};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrInt32ToUint16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 4, AttrDataType::kInt32, AttrDataType::kUint16),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint16_t *>(tiling_data->GetData()), 0x7f);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t));
}

TEST_F(TilingDataUT, AppendAttrListInt32ToListUint16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt32, AttrDataType::kListUint16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 3);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<uint16_t> expect_data{1, 0, 2};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt32ToListListUint16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt32, AttrDataType::kListListUint16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 6);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<uint16_t> expect_data{1, 0, 2, 4, 0, 5};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrInt32ToUint32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 4, AttrDataType::kInt32, AttrDataType::kUint32),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(tiling_data->GetData()), 0x7f);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint32_t));
}

TEST_F(TilingDataUT, AppendAttrListInt32ToListUint32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt32, AttrDataType::kListUint32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint32_t) * 3);
  auto ele = reinterpret_cast<uint32_t *>(tiling_data->GetData());
  std::vector<uint32_t> expect_data{1, 0, 2};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt32ToListListUint32Ok) {
  auto data = TilingData::CreateCap(30);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt32, AttrDataType::kListListUint32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint32_t) * 6);
  auto ele = reinterpret_cast<uint32_t *>(tiling_data->GetData());
  std::vector<uint32_t> expect_data{1, 0, 2, 4, 0, 5};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrInt32ToUint64Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 4, AttrDataType::kInt32, AttrDataType::kUint64),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint64_t *>(tiling_data->GetData()), 0x7f);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint64_t));
}

TEST_F(TilingDataUT, AppendAttrListInt32ToListUint64Ok) {
  auto data = TilingData::CreateCap(30);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt32, AttrDataType::kListUint64), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint64_t) * 3);
  auto ele = reinterpret_cast<uint64_t *>(tiling_data->GetData());
  std::vector<uint64_t> expect_data{1, 0, 2};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt32ToListListUint64Ok) {
  auto data = TilingData::CreateCap(50);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt32, AttrDataType::kListListUint64), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint64_t) * 6);
  auto ele = reinterpret_cast<uint64_t *>(tiling_data->GetData());
  std::vector<uint64_t> expect_data{1, 0, 2, 4, 0, 5};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrInt64ToBoolOk) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 4, AttrDataType::kInt64, AttrDataType::kBool), ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<bool *>(tiling_data->GetData()), true);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(bool));
}

TEST_F(TilingDataUT, AppendAttrListInt64ToListBoolOk) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt64, AttrDataType::kListBool), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(bool) * 3);
  auto ele = reinterpret_cast<bool *>(tiling_data->GetData());
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], true);
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt64ToListListBoolOk) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt64, AttrDataType::kListListBool), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(bool) * 6);
  auto ele = reinterpret_cast<bool *>(tiling_data->GetData());
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], true);
  }
}

TEST_F(TilingDataUT, AppendAttrInt64ToFloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 4, AttrDataType::kInt64, AttrDataType::kFloat16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint16_t *>(tiling_data->GetData()), optiling::OtherToFloat16<int64_t>(0x7f));
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t));
}

TEST_F(TilingDataUT, AppendAttrListInt64ToListFloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt64, AttrDataType::kListFloat16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 3);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<int64_t> expect_data = {1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], optiling::OtherToFloat16<int64_t>(expect_data[i]));
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt64ToListListFloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt64, AttrDataType::kListListFloat16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 6);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<int64_t> expect_data{1, 2, 3, 4, 5, 6};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], optiling::OtherToFloat16<int64_t>(expect_data[i]));
  }
}

TEST_F(TilingDataUT, AppendAttrInt64ToBfloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 4, AttrDataType::kInt64, AttrDataType::kBfloat16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint16_t *>(tiling_data->GetData()), optiling::OtherToBfloat16<int64_t>(0x7f));
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t));
}

TEST_F(TilingDataUT, AppendAttrListInt64ToListBfloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt64, AttrDataType::kListBfloat16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 3);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<int64_t> expect_data = {1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], optiling::OtherToBfloat16<int64_t>(expect_data[i]));
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt64ToListListBfloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt64, AttrDataType::kListListBfloat16),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 6);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<int64_t> expect_data{1, 2, 3, 4, 5, 6};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], optiling::OtherToBfloat16<int64_t>(expect_data[i]));
  }
}

TEST_F(TilingDataUT, AppendAttrInt64ToFloat32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 4, AttrDataType::kInt64, AttrDataType::kFloat32), ge::GRAPH_SUCCESS);
  EXPECT_FLOAT_EQ(*reinterpret_cast<float *>(tiling_data->GetData()), static_cast<float>(0x7f));
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(float));
}

TEST_F(TilingDataUT, AppendAttrListInt64ToListFloat32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt64, AttrDataType::kListFloat32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(float) * 3);
  auto ele = reinterpret_cast<float *>(tiling_data->GetData());
  std::vector<float> expect_data{1.0, 2.0, 3.0};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt64ToListListFloat32Ok) {
  auto data = TilingData::CreateCap(30);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt64, AttrDataType::kListListFloat32),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(float) * 6);
  auto ele = reinterpret_cast<float *>(tiling_data->GetData());
  std::vector<float> expect_data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrInt64ToInt8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 4, AttrDataType::kInt64, AttrDataType::kInt8),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int8_t *>(tiling_data->GetData()), 0x7f);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int8_t));
}

TEST_F(TilingDataUT, AppendAttrListInt64ToListInt8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt64, AttrDataType::kListInt8), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int8_t) * 3);
  auto ele = reinterpret_cast<int8_t *>(tiling_data->GetData());
  std::vector<int8_t> expect_data{1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt64ToListListInt8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt64, AttrDataType::kListListInt8), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int8_t) * 6);
  auto ele = reinterpret_cast<int8_t *>(tiling_data->GetData());
  std::vector<int8_t> expect_data{1, 2, 3, 4, 5, 6};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrInt64ToInt16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 4, AttrDataType::kInt64, AttrDataType::kInt16),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int16_t *>(tiling_data->GetData()), 0x7f);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int16_t));
}

TEST_F(TilingDataUT, AppendAttrListInt64ToListInt16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt64, AttrDataType::kListInt16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int16_t) * 3);
  auto ele = reinterpret_cast<int16_t *>(tiling_data->GetData());
  std::vector<int16_t> expect_data{1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt64ToListListInt16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt64, AttrDataType::kListListInt16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int16_t) * 6);
  auto ele = reinterpret_cast<int16_t *>(tiling_data->GetData());
  std::vector<int16_t> expect_data{1, 2, 3, 4, 5, 6};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrInt64ToInt32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 4, AttrDataType::kInt64, AttrDataType::kInt32),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int32_t *>(tiling_data->GetData()), 0x7f);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int32_t));
}

TEST_F(TilingDataUT, AppendAttrListInt64ToListInt32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt64, AttrDataType::kListInt32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int32_t) * 3);
  auto ele = reinterpret_cast<int32_t *>(tiling_data->GetData());
  std::vector<int64_t> expect_data{1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt64ListListInt32Ok) {
  auto data = TilingData::CreateCap(30);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt64, AttrDataType::kListListInt32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int32_t) * 6);
  auto ele = reinterpret_cast<int32_t *>(tiling_data->GetData());
  std::vector<int32_t> expect_data{1, 2, 3, 4, 5, 6};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrInt64Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 4, AttrDataType::kInt64, AttrDataType::kInt64),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int64_t *>(tiling_data->GetData()), 0x7f);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int64_t));
}

TEST_F(TilingDataUT, AppendAttrListInt64Ok) {
  auto data = TilingData::CreateCap(30);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt64, AttrDataType::kListInt64), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int64_t) * 3);
  auto ele = reinterpret_cast<int64_t *>(tiling_data->GetData());
  std::vector<int64_t> expect_data{1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt64Ok) {
  auto data = TilingData::CreateCap(50);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt64, AttrDataType::kListListInt64), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int64_t) * 6);
  auto ele = reinterpret_cast<int64_t *>(tiling_data->GetData());
  std::vector<int64_t> expect_data{1, 2, 3, 4, 5, 6};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrInt64ToUint8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 4, AttrDataType::kInt64, AttrDataType::kUint8),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint8_t *>(tiling_data->GetData()), 0x7f);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint8_t));
}

TEST_F(TilingDataUT, AppendAttrListInt64ToListUint8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt64, AttrDataType::kListUint8), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint8_t) * 3);
  auto ele = reinterpret_cast<uint8_t *>(tiling_data->GetData());
  std::vector<uint8_t> expect_data{1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt64ToListListUint8Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt64, AttrDataType::kListListUint8), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint8_t) * 6);
  auto ele = reinterpret_cast<uint8_t *>(tiling_data->GetData());
  std::vector<uint8_t> expect_data{1, 2, 3, 4, 5, 6};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrInt64ToUint16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 4, AttrDataType::kInt64, AttrDataType::kUint16),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint16_t *>(tiling_data->GetData()), 0x7f);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t));
}

TEST_F(TilingDataUT, AppendAttrListInt64ToListUint16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt64, AttrDataType::kListUint16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 3);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<uint16_t> expect_data{1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt64ToListListUint16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt64, AttrDataType::kListListUint16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 6);
  auto ele = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<uint16_t> expect_data{1, 2, 3, 4, 5, 6};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrInt64ToUint32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 4, AttrDataType::kInt64, AttrDataType::kUint32),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(tiling_data->GetData()), 0x7f);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint32_t));
}

TEST_F(TilingDataUT, AppendAttrListInt64ToListUint32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt64, AttrDataType::kListUint32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint32_t) * 3);
  auto ele = reinterpret_cast<uint32_t *>(tiling_data->GetData());
  std::vector<uint32_t> expect_data{1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt64ToListListUint32Ok) {
  auto data = TilingData::CreateCap(30);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt64, AttrDataType::kListListUint32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint32_t) * 6);
  auto ele = reinterpret_cast<uint32_t *>(tiling_data->GetData());
  std::vector<uint32_t> expect_data{1, 2, 3, 4, 5, 6};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrInt64ToUint64Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 4, AttrDataType::kInt64, AttrDataType::kUint64),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint64_t *>(tiling_data->GetData()), 0x7f);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint64_t));
}

TEST_F(TilingDataUT, AppendAttrListInt64ToListUint64Ok) {
  auto data = TilingData::CreateCap(30);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 5, AttrDataType::kListInt64, AttrDataType::kListUint64), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint64_t) * 3);
  auto ele = reinterpret_cast<uint64_t *>(tiling_data->GetData());
  std::vector<uint64_t> expect_data{1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListListInt64ToListListUint64Ok) {
  auto data = TilingData::CreateCap(50);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 6, AttrDataType::kListListInt64, AttrDataType::kListListUint64), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint64_t) * 6);
  auto ele = reinterpret_cast<uint64_t *>(tiling_data->GetData());
  std::vector<uint64_t> expect_data{1, 2, 3, 4, 5, 6};
  for (size_t i = 0UL; i < 6UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrIndexInvalid) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 13, AttrDataType::kInt64, AttrDataType::kInt32),
            ge::GRAPH_FAILED);
}

TEST_F(TilingDataUT, AppendAttrSrcTypeInvalid) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 7, AttrDataType::kString, AttrDataType::kInt32),
            ge::GRAPH_FAILED);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 7, AttrDataType::kListInt64, AttrDataType::kInt32),
            ge::GRAPH_FAILED);
}

TEST_F(TilingDataUT, AppendAttrDstTypeInvalid) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 7, AttrDataType::kBool, AttrDataType::kListListInt32), ge::GRAPH_FAILED);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(
            context->GetAttrs(), 7, AttrDataType::kBool, AttrDataType::kString), ge::GRAPH_FAILED);
}

TEST_F(TilingDataUT, AppendListOverCapacity) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  EXPECT_NE(tiling_data, nullptr);

  tiling_data->SetDataSize(5);
  std::vector<uint64_t> append_data = {1, 2};
  EXPECT_NE(tiling_data->Append<uint64_t>(append_data.data(), append_data.size()), ge::GRAPH_SUCCESS);
}

TEST_F(TilingDataUT, AppendListOk) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  EXPECT_NE(tiling_data, nullptr);

  tiling_data->SetDataSize(5);
  std::vector<uint8_t> append_data = {1, 2, 3, 4};
  EXPECT_EQ(tiling_data->Append<uint8_t>(append_data.data(), append_data.size()), ge::GRAPH_SUCCESS);

  EXPECT_EQ(tiling_data->GetDataSize(), 9);
  for (size_t i = 0UL; i < append_data.size(); ++i) {
    EXPECT_EQ(*(reinterpret_cast<uint8_t *>(tiling_data->GetData()) + 5 + i), append_data[i]);
  }
}
}  // namespace gert
