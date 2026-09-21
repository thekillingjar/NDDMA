/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensor_utils.h"

#include <gtest/gtest.h>
#include "common/memory/tensor_trans_utils.h"
#include "../../../graph_metadef/depends/checker/tensor_check_utils.h"
#include "ge/ge_api_error_codes.h"
#include "graph_metadef/graph/utils/tensor_adapter.h"
#include "runtime/gert_api.h"
namespace ge {
namespace {
bool IsEqualWith(const gert::Shape &rt_shape, const ge::Shape &ge_shape) {
  if (rt_shape.GetDimNum() != ge_shape.GetDimNum()) {
    return false;
  }

  for (size_t i = 0U; i < rt_shape.GetDimNum(); ++i) {
    if (rt_shape.GetDim(i) != ge_shape.GetDim(i)) {
      return false;
    }
  }
  return true;
}
}
class UtestTensorTransUtils : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestTensorTransUtils, TransDeviceRtTensorToHostTensor_without_value_Success) {
  //  Status TensorTransUtils::TransRtTensorToTensor(const std::vector<gert::Tensor> &srcs,
  //                                                           std::vector<Tensor> &dsts, bool with_value);
  gert::Tensor tensor0 = {{{1000, 1000, 1, 1000}, {3, 1000, 1, 1000}},  // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},   // format
                          gert::kOnDeviceHbm,                           // placement
                          ge::DT_FLOAT,                                 // data type
                          nullptr};
  gert::Tensor tensor1 = {{{100, 100, 1, 100}, {3, 100, 1, 100}},  // shape
                          {ge::FORMAT_ND, ge::FORMAT_NC1HWC0, {}},   // format
                          gert::kOnDeviceHbm,                           // placement
                          ge::DT_FLOAT,                                 // data type
                          nullptr};
  std::vector<gert::Tensor> srcs;
  srcs.emplace_back(std::move(tensor0));
  srcs.emplace_back(std::move(tensor1));

  std::vector<Tensor> dsts;
  EXPECT_EQ(TensorTransUtils::TransRtTensorToTensor(srcs, dsts, false), GRAPH_SUCCESS);

  EXPECT_EQ(dsts.size(), srcs.size());
  EXPECT_EQ(dsts[1].GetPlacement(), ge::kPlacementEnd);
  const auto dst_tensor_desc = dsts[1].GetTensorDesc();
  EXPECT_EQ(dst_tensor_desc.GetPlacement(), ge::kPlacementEnd);
  EXPECT_TRUE(IsEqualWith(srcs[1].GetOriginShape(), dst_tensor_desc.GetOriginShape()));
  EXPECT_TRUE(IsEqualWith(srcs[1].GetStorageShape(), dst_tensor_desc.GetShape()));
  EXPECT_EQ(srcs[1].GetOriginFormat(), dst_tensor_desc.GetOriginFormat());
  EXPECT_EQ(srcs[1].GetFormat().GetStorageFormat(), dst_tensor_desc.GetFormat());
  EXPECT_EQ(srcs[1].GetFormat().GetOriginFormat(), dst_tensor_desc.GetOriginFormat());
}

TEST_F(UtestTensorTransUtils, TransDeviceRtTensorToHostTensor_empty_rttensor_with_value_Success) {
  gert::Tensor tensor0 = {{{1000, 1000, 0, 1000}, {3, 1000, 0, 1000}},  // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},   // format
                          gert::kOnDeviceHbm,                           // placement
                          ge::DT_FLOAT,                                 // data type
                          nullptr};
  std::vector<gert::Tensor> srcs;
  srcs.emplace_back(std::move(tensor0));

  std::vector<Tensor> dsts;
  EXPECT_EQ(TensorTransUtils::TransRtTensorToTensor(srcs, dsts, true), GRAPH_SUCCESS);

  EXPECT_EQ(dsts.size(), srcs.size());
  EXPECT_EQ(dsts[0].GetPlacement(), ge::kPlacementHost);
  const auto dst_tensor_desc = dsts[0].GetTensorDesc();
  EXPECT_EQ(dst_tensor_desc.GetPlacement(), ge::kPlacementHost);
  EXPECT_TRUE(IsEqualWith(srcs[0].GetOriginShape(), dst_tensor_desc.GetOriginShape()));
  EXPECT_TRUE(IsEqualWith(srcs[0].GetStorageShape(), dst_tensor_desc.GetShape()));
  EXPECT_EQ(srcs[0].GetOriginFormat(), dst_tensor_desc.GetOriginFormat());
  EXPECT_EQ(srcs[0].GetFormat().GetStorageFormat(), dst_tensor_desc.GetFormat());
  EXPECT_EQ(srcs[0].GetFormat().GetOriginFormat(), dst_tensor_desc.GetOriginFormat());
  EXPECT_EQ(dsts[0].GetTensorDesc().GetSize(), 0);
}

TEST_F(UtestTensorTransUtils, GertTensors2GeTensors) {
  std::vector<gert::Tensor> gert_tensors;
  gert_tensors.resize(2);
  TensorCheckUtils::ConstructGertTensor(gert_tensors[0]);
  TensorCheckUtils::ConstructGertTensor(gert_tensors[1]);

  std::vector<GeTensor> ge_tensors;
  ASSERT_EQ(TensorTransUtils::GertTensors2GeTensors(gert_tensors, ge_tensors), SUCCESS);
  ASSERT_EQ(ge_tensors.size(), gert_tensors.size());
  for (size_t i = 0; i < gert_tensors.size(); i++) {
    EXPECT_EQ(TensorCheckUtils::CheckGeTensorEqGertTensor(ge_tensors[i], gert_tensors[i]), SUCCESS);
  }
  ge_tensors.clear();
  gert_tensors.clear();
}

TEST_F(UtestTensorTransUtils, GertTensors2Tensors) {
  std::vector<gert::Tensor> gert_tensors;
  gert_tensors.resize(2);
  TensorCheckUtils::ConstructGertTensor(gert_tensors[0]);
  TensorCheckUtils::ConstructGertTensor(gert_tensors[1]);

  std::vector<Tensor> tensors;
  ASSERT_EQ(TensorTransUtils::GertTensors2Tensors(gert_tensors, tensors), SUCCESS);
  ASSERT_EQ(tensors.size(), gert_tensors.size());
  for (size_t i = 0; i < gert_tensors.size(); i++) {
    EXPECT_EQ(TensorCheckUtils::CheckTensorEqGertTensor(tensors[i], gert_tensors[i]), SUCCESS);
  }
  tensors.clear();
  gert_tensors.clear();
}

/*
 * 构造ge_tensor, 转换成为gert_tensor, 释放ge_tensor, 校验gert_tensor的数据，表示ge_tensor释放后，gert_tensor还可以访问数据
 */
TEST_F(UtestTensorTransUtils, GeTensors2GertTensors_CheckDataAfterFreeGeTensors) {
  std::vector<GeTensor> ge_tensors;
  ge_tensors.resize(2);
  TensorCheckUtils::ConstructGeTensor(ge_tensors[0]);
  TensorCheckUtils::ConstructGeTensor(ge_tensors[1]);

  std::vector<gert::Tensor> gert_tensors;
  ASSERT_EQ(TensorTransUtils::GeTensors2GertTensors(ge_tensors, gert_tensors), SUCCESS);
  ASSERT_EQ(ge_tensors.size(), gert_tensors.size());
  for (size_t i = 0; i < gert_tensors.size(); i++) {
    EXPECT_EQ(TensorCheckUtils::CheckGeTensorEqGertTensor(ge_tensors[i], gert_tensors[i]), SUCCESS);
  }
  const auto buffer_size = ge_tensors[0].MutableData().GetSize();
  auto buffer = new (std::nothrow) uint32_t [buffer_size];
  ASSERT_NE(buffer, nullptr);
  memccpy(buffer, ge_tensors[0].MutableData().GetData(), buffer_size, buffer_size);
  ge_tensors.clear();

  // ge_tensor释放后，gert_tensor数据仍然可用，且数据正确
  auto dst_data = reinterpret_cast<uint32_t *>(gert_tensors[0].MutableTensorData().GetAddr());
  ASSERT_EQ(buffer_size, gert_tensors[0].GetSize());

  for (size_t i = 0; i < buffer_size / sizeof(uint32_t); i++) {
    EXPECT_EQ(buffer[i], dst_data[i]);
  }
  delete [] buffer;
  gert_tensors.clear();
}

/*
 * 构造ge_tensor, 转换成为gert_tensor, 释放ge_tensor
 * gert_tensor_share和gert_tensor共享数据，分别释放
 */
TEST_F(UtestTensorTransUtils, GeTensors2GertTensors_GertTensorShare) {
  std::vector<GeTensor> ge_tensors;
  ge_tensors.resize(2);
  TensorCheckUtils::ConstructGeTensor(ge_tensors[0]);
  TensorCheckUtils::ConstructGeTensor(ge_tensors[1]);

  std::vector<gert::Tensor> gert_tensors;
  ASSERT_EQ(TensorTransUtils::GeTensors2GertTensors(ge_tensors, gert_tensors), SUCCESS);
  ASSERT_EQ(ge_tensors.size(), gert_tensors.size());
  for (size_t i = 0; i < gert_tensors.size(); i++) {
    EXPECT_EQ(TensorCheckUtils::CheckGeTensorEqGertTensor(ge_tensors[i], gert_tensors[i]), SUCCESS);
  }
  const auto buffer_size = ge_tensors[0].MutableData().GetSize();
  auto buffer = new (std::nothrow) uint32_t [buffer_size];
  ASSERT_NE(buffer, nullptr);
  memccpy(buffer, ge_tensors[0].MutableData().GetData(), buffer_size, buffer_size);
  ge_tensors.clear();

  // share from
  gert::Tensor gert_tensor_share(gert_tensors[0].GetShape(), gert_tensors[0].GetFormat(), gert_tensors[0].GetDataType());
  gert_tensor_share.MutableTensorData().ShareFrom(gert_tensors[0].GetTensorData());
  gert_tensors.clear();

  // 原gert_tensor释放后，访问gert_tensor_share
  auto dst_data = reinterpret_cast<uint32_t *>(gert_tensor_share.MutableTensorData().GetAddr());
  ASSERT_EQ(buffer_size, gert_tensor_share.GetSize());

  for (size_t i = 0; i < buffer_size / sizeof(uint32_t); i++) {
    EXPECT_EQ(buffer[i], dst_data[i]);
  }
  delete [] buffer;
}

TEST_F(UtestTensorTransUtils, Tensors2GertTensors) {
  std::vector<GeTensor> ge_tensors;
  ge_tensors.resize(10);
  for (size_t i = 0U; i < ge_tensors.size(); i++) {
    TensorCheckUtils::ConstructGeTensor(ge_tensors[i]);
  }

  std::vector<Tensor> tensors;
  tensors.resize(ge_tensors.size());
  for (size_t i = 0; i < ge_tensors.size(); i++) {
    tensors[i] = TensorAdapter::AsTensor(ge_tensors[i]);
  }

  GELOGE(FAILED, "call 10 start");
  std::vector<gert::Tensor> gert_tensors;
  ASSERT_EQ(TensorTransUtils::Tensors2GertTensors(tensors, gert_tensors), SUCCESS);
  GELOGE(FAILED, "call 10 finish");

  ASSERT_EQ(tensors.size(), gert_tensors.size());
  for (size_t i = 0; i < gert_tensors.size(); i++) {
    EXPECT_EQ(TensorCheckUtils::CheckTensorEqGertTensor(tensors[i], gert_tensors[i]), SUCCESS);
  }
  ge_tensors.clear();
  tensors.clear();
  gert_tensors.clear();
}

TEST_F(UtestTensorTransUtils, AsTensorsView) {
  constexpr size_t tensor_num = 10;
  std::vector<GeTensor> ge_tensors;
  ge_tensors.resize(tensor_num);
  for (size_t i = 0U; i < ge_tensors.size(); i++) {
    TensorCheckUtils::ConstructGeTensor(ge_tensors[i]);
  }

  std::vector<Tensor> tensors;
  tensors.resize(ge_tensors.size());
  for (size_t i = 0; i < ge_tensors.size(); i++) {
    tensors[i] = TensorAdapter::AsTensor(ge_tensors[i]);
  }

  GELOGE(FAILED, "call 10 start");
  std::vector<gert::Tensor> gert_tensors;
  ASSERT_EQ(TensorTransUtils::AsTensorsView(tensors, gert_tensors), SUCCESS);
  GELOGE(FAILED, "call 10 finish");

  ASSERT_EQ(tensors.size(), gert_tensors.size());
  for (size_t i = 0; i < gert_tensors.size(); i++) {
    EXPECT_EQ(TensorCheckUtils::CheckTensorEqGertTensor(tensors[i], gert_tensors[i]), SUCCESS);
  }

  gert_tensors.clear();
  ge_tensors.clear();
  tensors.clear();
}

/*
 * device上的gert_tensor转换成host上的gert_tensor，释放device上gert_tensor，然后校验host gert_tensor数据
 * 测试host gert_tensor的share from功能
 */
TEST_F(UtestTensorTransUtils, TransGertTensorToHost_WithShareFrom) {
  gert::Tensor host_gert_tensor;
  GeTensor ge_tensor;
  gert::Tensor device_gert_tensor;
  TensorCheckUtils::ConstructGeTensor(ge_tensor);

  ASSERT_EQ(TensorTransUtils::GeTensor2GertTensor(ge_tensor, device_gert_tensor), SUCCESS);
  EXPECT_EQ(TensorCheckUtils::CheckGeTensorEqGertTensor(ge_tensor, device_gert_tensor), SUCCESS);

  ASSERT_EQ(TensorTransUtils::TransGertTensorToHost(device_gert_tensor,
    host_gert_tensor), SUCCESS);

  // address 64 aligned
  auto addr = host_gert_tensor.GetAddr();
  EXPECT_EQ(PtrToValue(addr) % 64, 0);

  // 释放device gert_tensor
  EXPECT_EQ(TensorCheckUtils::CheckGeTensorEqGertTensor(ge_tensor, device_gert_tensor), SUCCESS);
  device_gert_tensor.MutableTensorData().Free();

  EXPECT_EQ(TensorCheckUtils::CheckGeTensorEqGertTensorWithData(ge_tensor, host_gert_tensor), SUCCESS);

  // share from，测试两个gert_tensor可单独释放
  gert::Tensor host_gert_tensor_share(host_gert_tensor.GetShape(), host_gert_tensor.GetFormat(),
    host_gert_tensor.GetDataType());
  host_gert_tensor_share.MutableTensorData().ShareFrom(host_gert_tensor.GetTensorData());

  // 原始host gert_tensor释放
  host_gert_tensor.MutableTensorData().Free();

  // share出来的host_gert_tensor_share数据访问
  EXPECT_EQ(TensorCheckUtils::CheckGeTensorEqGertTensorWithData(ge_tensor, host_gert_tensor_share), SUCCESS);
}

TEST_F(UtestTensorTransUtils, GetDimsFromGertShape_Success) {
  gert::Tensor gert_tensor;
  std::initializer_list<int64_t> shape = {99, 2, 3};
  const auto exp_dims = std::vector<int64_t>(shape);
  TensorCheckUtils::ConstructGertTensor(gert_tensor, shape);

  const auto dims = TensorTransUtils::GetDimsFromGertShape(gert_tensor.GetStorageShape());
  ASSERT_EQ(exp_dims.size(), dims.size());
  for (size_t i = 0U; i < dims.size(); i++) {
    EXPECT_EQ(exp_dims.at(i), dims.at(i));
  }
}
TEST_F(UtestTensorTransUtils, ShareFromGertTensors) {
  std::initializer_list<int64_t> shape = {99, 2, 3};
  const auto exp_dims = std::vector<int64_t>(shape);
  std::vector<gert::Tensor> tensors;
  {
    gert::Tensor gert_tensor;
    TensorCheckUtils::ConstructGertTensor(gert_tensor, shape);
    tensors.emplace_back(std::move(gert_tensor));
  }
  {
    gert::Tensor gert_tensor;
    TensorCheckUtils::ConstructGertTensor(gert_tensor, shape);
    tensors.emplace_back(std::move(gert_tensor));
  }
  std::vector<gert::Tensor> share_tensors = TensorTransUtils::ShareFromGertTenosrs(tensors);
  tensors.clear();

  for (const auto &gert_tensor : tensors) {
    auto addr = PtrToPtr<void, uint8_t>(gert_tensor.GetAddr());
    auto data = reinterpret_cast<const uint32_t *>(addr);
    for (size_t i = 0; i < gert_tensor.GetSize() / sizeof(uint32_t); ++i) {
      EXPECT_EQ(data[i], i);
    }
  }
}

TEST_F(UtestTensorTransUtils, ContructRtShapeFromGeShape) {
  GeShape ge_shape;
  ge_shape.AppendDim(1);
  auto rt_shape =TensorTransUtils::ContructRtShapeFromGeShape(ge_shape);
  EXPECT_EQ(rt_shape.GetDimNum(), 1);
}

TEST_F(UtestTensorTransUtils, ContructRtShapeFromVector) {
  auto rt_shape =TensorTransUtils::ContructRtShapeFromVector({1});
  EXPECT_EQ(rt_shape.GetDimNum(), 1);
}

TEST_F(UtestTensorTransUtils, GertTensors2Tensors_Life) {
  std::vector<gert::Tensor> gert_tensors;
  gert_tensors.resize(1);
  TensorCheckUtils::ConstructGertTensor(gert_tensors[0]);

  std::vector<ge::Tensor> ge_tensors;
  Status status = TensorTransUtils::GertTensors2Tensors(gert_tensors, ge_tensors);

  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(ge_tensors.size(), 1);

  // 测试转换后的 tensor 数据有效性
  EXPECT_EQ(ge_tensors[0].GetSize(), gert_tensors[0].GetSize());
  EXPECT_EQ(ge_tensors[0].GetDataType(), gert_tensors[0].GetDataType());

  ge_tensors.clear();
}

/*
 * 模拟 coredump 场景：AsTensorView -> GertTensors2Tensors 转换链
 * 测试场景：
 * 1. 创建 ge::Tensor a（持有数据所有权）
 * 2. 使用 AsTensorView 转换为 gert::Tensor b（只引用，不持有所有权）
 * 3. 使用 GertTensors2Tensors 转换为 ge::Tensor c（创建 output_holder 和 deleter）
 * 4. 按顺序析构 c, b, a
 *
 * 这模拟了 Session::RunGraphAsync 中的调用链，可能会触发内存问题
 */
TEST_F(UtestTensorTransUtils, AsTensorView_To_GertTensors2Tensors_DestructorOrder) {
  // 1. 创建 ge::Tensor a（使用 TensorAdapter 转换 GeTensor）
  std::vector<Tensor> a_vec;
  {
    GeTensor ge_a;
    TensorCheckUtils::ConstructGeTensor(ge_a);
    a_vec.push_back(TensorAdapter::AsTensor(ge_a));
  }

  // 2. 使用 AsTensorView 转换为 gert::Tensor b（只引用，不持有所有权）
  std::vector<gert::Tensor> b_vec;
  Status status = TensorTransUtils::AsTensorsView(a_vec, b_vec);
  EXPECT_EQ(status, SUCCESS);

  // 3. 使用 GertTensors2Tensors 转换为 ge::Tensor c（创建 output_holder 和 deleter）
  std::vector<Tensor> c_vec;
  status = TensorTransUtils::GertTensors2Tensors(b_vec, c_vec);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(c_vec.size(), 1);

  // 4. 按顺序析构 c, b, a
  // 首先析构 c（这会触发 deleter，但应该不会释放 a 的内存）
  c_vec.clear();

  // 然后析构 b（AsTensorView 创建的 view，不持有所有权）
  b_vec.clear();

  // 最后析构 a_vec（此时 a 的内存才真正被释放）
  a_vec.clear();
}

/*
 * 模拟 coredump 场景：AsTensorView -> GertTensors2Tensors 转换链
 * 测试场景：
 * 1. 创建 ge::Tensor a（持有数据所有权）
 * 2. 使用 AsTensorView 转换为 gert::Tensor b（只引用，不持有所有权）
 * 3. 使用 GertTensors2Tensors 转换为 ge::Tensor c（创建 output_holder 和 deleter）
 * 4. 按顺序析构 c, b, a
 *
 * 这模拟了 Session::RunGraphAsync 中的调用链，可能会触发内存问题
 */
TEST_F(UtestTensorTransUtils, AsTensorView_To_GertTensors2Tensors_CopyOutTensor) {
  // 1. 创建 ge::Tensor a（使用 TensorAdapter 转换 GeTensor）
  std::vector<Tensor> a_vec;
  {
    GeTensor ge_a;
    TensorCheckUtils::ConstructGeTensor(ge_a);
    a_vec.push_back(TensorAdapter::AsTensor(ge_a));
  }

  // 2. 使用 AsTensorView 转换为 gert::Tensor b（只引用，不持有所有权）
  std::vector<gert::Tensor> b_vec;
  Status status = TensorTransUtils::AsTensorsView(a_vec, b_vec);
  EXPECT_EQ(status, SUCCESS);

  // 3. 使用 GertTensors2Tensors 转换为 ge::Tensor c（创建 output_holder 和 deleter）
  std::vector<Tensor> c_vec;
  status = TensorTransUtils::GertTensors2Tensors(b_vec, c_vec);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(c_vec.size(), 1);

  // 4. 复制给d_vec
  auto d_vec = c_vec;

  // 5. 按顺序析构 c, b, a, d
  // 首先析构 c（这会触发 deleter，但应该不会释放 a 的内存）
  c_vec.clear();

  // 然后析构 b（AsTensorView 创建的 view，不持有所有权）
  b_vec.clear();

  // 最后析构 a_vec（此时 a 的内存才真正被释放）
  a_vec.clear();

  d_vec.clear();
}

/*
 * 模拟真实 coredump 场景：AsTensorView -> GertTensors2Tensors -> ResetData -> 析构顺序
 *
 * 关键场景（模拟 TensorFlow BuildOutputTensorInfo 的行为）：
 * 1. 创建 ge::Tensor a（持有数据所有权）
 * 2. 使用 AsTensorView 转换为 gert::Tensor b（只引用，不持有所有权）
 * 3. 使用 GertTensors2Tensors 转换为 ge::Tensor c（创建 output_holder 和 deleter）
 * 4. 调用 c.ResetData() 转移所有权（模拟 TensorFlow 的行为）
 * 5. 析构 c_vec（模拟 RunGraphAsyncCallback 结束，ge_tensors 被析构）
 * 6. 销毁 ResetData() 返回的 unique_ptr（模拟 TensorFlow Tensor 析构）
 *
 * 这个场景下，ResetData() 返回的 unique_ptr 的 deleter 会调用 output_holder->Free()
 * 如果 output_holder 持有的内存来自 AsTensorsView 创建的视图，可能会触发 use-after-free
 */
TEST_F(UtestTensorTransUtils, AsTensorView_GertTensors2Tensors_ResetData_CrashScenario) {
  // 1. 创建 ge::Tensor a（持有数据所有权）
  std::vector<Tensor> a_vec;
  {
    GeTensor ge_a;
    TensorCheckUtils::ConstructGeTensor(ge_a);
    a_vec.push_back(TensorAdapter::AsTensor(ge_a));
  }

  // 保存原始数据指针用于后续验证
  const void* original_data_ptr = a_vec[0].GetData();

  // 2. 使用 AsTensorView 转换为 gert::Tensor b（只引用，不持有所有权）
  std::vector<gert::Tensor> b_vec;
  Status status = TensorTransUtils::AsTensorsView(a_vec, b_vec);
  EXPECT_EQ(status, SUCCESS);

  // 验证 b_vec 只是指向 a 的数据（通过地址比较）
  const void* b_data_ptr = b_vec[0].GetAddr();
  EXPECT_EQ(b_data_ptr, original_data_ptr);

  // 3. 使用 GertTensors2Tensors 转换为 ge::Tensor c（创建 output_holder 和 deleter）
  std::vector<Tensor> c_vec;
  status = TensorTransUtils::GertTensors2Tensors(b_vec, c_vec);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(c_vec.size(), 1);

  // 验证 c 的数据也指向相同地址（共享数据）
  const void* c_data_ptr_before_reset = c_vec[0].GetData();
  EXPECT_EQ(c_data_ptr_before_reset, original_data_ptr);

  // 4. 调用 ResetData() 转移所有权（模拟 TensorFlow BuildOutputTensorInfo 的行为）
  // 这会返回一个 unique_ptr，持有数据和 deleter
  auto unique_data_ptr = c_vec[0].ResetData();

  // 注意：ResetData() 后，c_vec[0].GetData() 仍然返回地址（aligned_addr_）
  // 但 unique_ptr 的所有权已经被转移，数据不会在 c_vec[0] 析构时被释放
  // 真正的所有权现在在 unique_data_ptr 中

  // 验证 unique_ptr 持有原始数据
  EXPECT_NE(unique_data_ptr.get(), nullptr);
  EXPECT_EQ(unique_data_ptr.get(), original_data_ptr);

  // 5. 析构 c_vec（模拟 RunGraphAsyncCallback 结束，ge_tensors 被析构）
  // 此时 c_vec 中的 Tensor 已经不再持有数据（ResetData 清空了）
  // 所以理论上不应该触发 deleter
  c_vec.clear();

  // 6. 销毁 b_vec（AsTensorView 创建的 view）
  b_vec.clear();

  // 注意：a_vec 仍然持有原始数据，所以不应该被析构
  // 但是，如果 unique_data_ptr 的 deleter 错误地释放了内存，
  // 后续访问 a_vec[0] 的数据可能会触发 coredump

  // 7. 销毁 unique_data_ptr（模拟 TensorFlow Tensor 析构）
  // 这会调用 deleter，deleter 会调用 output_holder->Free()
  // 如果 output_holder 持有的内存来自 AsTensorsView，可能会 use-after-free
  unique_data_ptr.reset();

  // 如果代码运行到这里没有 coredump，说明在这个简单场景下没有触发问题
  // 但真实场景中，内存所有权更复杂，可能触发问题

  // 最后清理 a_vec
  a_vec.clear();
}

/*
 * 更真实的场景：模拟 RunGraphAsyncCallback 和 callback 的完整流程
 *
 * 时间线：
 * 1. RunGraphAsyncCallback 创建局部变量 ge_tensors
 * 2. GertTensors2Tensors 转换 gert::Tensor → ge::Tensor，设置 deleter
 * 3. callback 被调用，接收 ge_tensors 的引用
 * 4. callback 中调用 ResetData() 转移所有权
 * 5. callback 结束，返回到 RunGraphAsyncCallback
 * 6. RunGraphAsyncCallback 结束，ge_tensors 被析构
 * 7. 之后，TensorFlow 的 unique_ptr 被析构，调用 deleter
 */
TEST_F(UtestTensorTransUtils, RunGraphAsyncCallback_FullSimulation) {
  // 模拟 RunGraphAsyncCallback 函数
  auto simulate_run_graph_async_callback = [](std::vector<ge::Tensor> &ge_tensors,
                                               std::unique_ptr<uint8_t[], Tensor::DeleteFunc> &out_data) {
    // 模拟 callback：调用 ResetData() 转移所有权
    out_data = ge_tensors[0].ResetData();
  };

  // 1. 创建输入 gert::Tensor（模拟 GE 运行时输出）
  std::vector<gert::Tensor> outputs(2);
  TensorCheckUtils::ConstructGertTensor(outputs[0]);
  TensorCheckUtils::ConstructGertTensor(outputs[1]);

  // 2. 调用模拟的 RunGraphAsyncCallback
  std::unique_ptr<uint8_t[], Tensor::DeleteFunc> tensorflow_data;
  {
    std::vector<gert::Tensor> host_outputs;
    EXPECT_EQ(TensorTransUtils::TransGertTensorsToHost(outputs, host_outputs), SUCCESS);
    std::vector<ge::Tensor> ge_tensors;
    Status status = TensorTransUtils::GertTensors2Tensors(host_outputs, ge_tensors);
    EXPECT_EQ(status, SUCCESS);
    simulate_run_graph_async_callback(ge_tensors, tensorflow_data);
  }
  // 3. 此时，ge_tensors host_outputs 已经被析构, 但 tensorflow_data 仍然持有数据和 deleter

  // 4. 销毁 outputs（AsTensorsView 创建的视图）
  outputs.clear();

  // 5. 销毁 tensorflow_data（模拟 TensorFlow Tensor 析构）
  //    这会调用 deleter，deleter 会调用 output_holder->Free()
  //    如果 output_holder 持有的内存所有权有问题，可能会触发 coredump
  tensorflow_data.reset();

  // 如果代码运行到这里没有 coredump，说明在这个场景下没有触发问题
}
}
