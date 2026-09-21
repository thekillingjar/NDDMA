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
#include "register/kernel_registry_impl.h"
#include "faker/kernel_run_context_facker.h"
#include <iostream>
#include "exe_graph/runtime/continuous_vector.h"
#include "framework/common/debug/ge_log.h"

namespace gert {
struct SplitToSequenceComputeTest : public testing::Test {
  SplitToSequenceComputeTest() {
    splitToSequenceCompute = KernelRegistry::GetInstance().FindKernelFuncs("StoreSplitTensorToSequence");
  }
  const KernelRegistry::KernelFuncs* splitToSequenceCompute;
};

TEST_F(SplitToSequenceComputeTest, success) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  int32_t axis = 1;
  int32_t keep_dims = 1;
  uint32_t input_num = 2;

  uint32_t shape_num = 2;
  auto shapes = gert::ContinuousVector::Create<gert::Shape>(shape_num);
  auto shapes_vec = reinterpret_cast<gert::ContinuousVector*>(shapes.get());
  shapes_vec->SetSize(shape_num);
  auto shapes_data = static_cast<gert::Shape*>(shapes_vec->MutableData());
  for (size_t i = 0; i < shape_num; i++) {
    shapes_data[i].SetDimNum(2);
    shapes_data[i].SetDim(0, 2);
    shapes_data[i].SetDim(1, 3);
  }

  vector<int64_t> x_vec {1,2,3,4,5,6,7,8,9,10,11,12};
  auto tensor_datas = gert::ContinuousVector::Create<gert::TensorData*>(shape_num);
  auto tensor_data_vec = reinterpret_cast<gert::ContinuousVector*>(tensor_datas.get());
  tensor_data_vec->SetSize(shape_num);
  auto tensor_data_addr = static_cast<gert::TensorData**>(tensor_data_vec->MutableData());
  for (size_t i = 0; i < shape_num; i++) {
    tensor_data_addr[i] = new (std::nothrow) gert::TensorData();
    tensor_data_addr[i]->SetAddr(x_vec.data(), nullptr);
  }

  gert::StorageShape x_shape = {{2,6}, {2,6}};

  // build input shape & input tensor
  size_t total_size = 0;
  auto output_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto output_tensor = reinterpret_cast<gert::Tensor*>(output_tensor_holder.get());
  output_tensor->SetOriginFormat(ge::FORMAT_ND);
  output_tensor->SetStorageFormat(ge::FORMAT_ND);
  output_tensor->SetPlacement(gert::kFollowing);
  output_tensor->MutableOriginShape() = {};
  output_tensor->MutableStorageShape() = {};

  auto output_tensor_data = reinterpret_cast<uint64_t*>(output_tensor->GetAddr());
  output_tensor_data[0] = 1;
  output_tensor->SetData(gert::TensorData{output_tensor_data, nullptr});
  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(9, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id),
                                   tensor_datas.get(), shapes.get(), reinterpret_cast<void*>(input_num),
                                   reinterpret_cast<void*>(keep_dims), reinterpret_cast<void*>(axis),
                                   reinterpret_cast<void*>(&x_shape), reinterpret_cast<void*>(output_tensor)})
                          .NodeIoNum(2, 1)
                          .IrInputNum({2})
                          .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  std::cout << "splitToSequenceCompute->run_func : " << (void*)(splitToSequenceCompute->run_func) << std::endl;
  ASSERT_EQ(splitToSequenceCompute->run_func(context), ge::GRAPH_SUCCESS);
  for (size_t i = 0; i < shape_num; i++) {
    delete tensor_data_addr[i];
    tensor_data_addr[i] = nullptr;
  }
}

TEST_F(SplitToSequenceComputeTest, success_1) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  int32_t axis = 1;
  int32_t keep_dims = 0;
  uint32_t input_num = 1;

  uint32_t shape_num = 2;
  auto shapes = gert::ContinuousVector::Create<gert::Shape>(shape_num);
  auto shapes_vec = reinterpret_cast<gert::ContinuousVector*>(shapes.get());
  shapes_vec->SetSize(shape_num);
  auto shapes_data = static_cast<gert::Shape*>(shapes_vec->MutableData());
  for (size_t i = 0; i < shape_num; i++) {
    shapes_data[i].SetDimNum(2);
    shapes_data[i].SetDim(0, 2);
    shapes_data[i].SetDim(1, 3);
  }

  vector<int64_t> x_vec {1,2,3,4,5,6,7,8,9,10,11,12};
  auto tensor_datas = gert::ContinuousVector::Create<gert::TensorData*>(shape_num);
  auto tensor_data_vec = reinterpret_cast<gert::ContinuousVector*>(tensor_datas.get());
  tensor_data_vec->SetSize(shape_num);
  auto tensor_data_addr = static_cast<gert::TensorData**>(tensor_data_vec->MutableData());
  for (size_t i = 0; i < shape_num; i++) {
    tensor_data_addr[i] = new (std::nothrow) gert::TensorData();
    tensor_data_addr[i]->SetAddr(x_vec.data(), nullptr);
  }

  gert::StorageShape x_shape = {{2,6}, {2,6}};

  // build input shape & input tensor
  size_t total_size = 0;
  auto output_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto output_tensor = reinterpret_cast<gert::Tensor*>(output_tensor_holder.get());
  output_tensor->SetOriginFormat(ge::FORMAT_ND);
  output_tensor->SetStorageFormat(ge::FORMAT_ND);
  output_tensor->SetPlacement(gert::kFollowing);
  output_tensor->MutableOriginShape() = {};
  output_tensor->MutableStorageShape() = {};

  auto output_tensor_data = reinterpret_cast<uint64_t*>(output_tensor->GetAddr());
  output_tensor_data[0] = 1;
  output_tensor->SetData(gert::TensorData{output_tensor_data, nullptr});
  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(9, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id),
                                   tensor_datas.get(), shapes.get(), reinterpret_cast<void*>(input_num),
                                   reinterpret_cast<void*>(keep_dims), reinterpret_cast<void*>(axis),
                                   reinterpret_cast<void*>(&x_shape), reinterpret_cast<void*>(output_tensor)})
                          .NodeIoNum(1, 1)
                          .IrInputNum({1})
                          .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  std::cout << "splitToSequenceCompute->run_func : " << (void*)(splitToSequenceCompute->run_func) << std::endl;
  ASSERT_EQ(splitToSequenceCompute->run_func(context), ge::GRAPH_SUCCESS);
  for (size_t i = 0; i < shape_num; i++) {
    delete tensor_data_addr[i];
    tensor_data_addr[i] = nullptr;
  }
}

TEST_F(SplitToSequenceComputeTest, inner_tensor_shape_null) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  int32_t axis = 1;
  int32_t keep_dims = 1;
  uint32_t input_num = 2;

  uint32_t shape_num = 2;
  auto shapes = gert::ContinuousVector::Create<gert::Shape>(shape_num);
  auto shapes_vec = reinterpret_cast<gert::ContinuousVector*>(shapes.get());
  shapes_vec->SetSize(shape_num);
  auto shapes_data = static_cast<gert::Shape*>(shapes_vec->MutableData());
  for (size_t i = 0; i < shape_num; i++) {
    shapes_data[i].SetDimNum(2);
    shapes_data[i].SetDim(0, 2);
    shapes_data[i].SetDim(1, 3);
  }

  vector<int64_t> x_vec {1,2,3,4,5,6,7,8,9,10,11,12};
  auto tensor_datas = gert::ContinuousVector::Create<gert::TensorData*>(shape_num);
  auto tensor_data_vec = reinterpret_cast<gert::ContinuousVector*>(tensor_datas.get());
  tensor_data_vec->SetSize(shape_num);
  auto tensor_data_addr = static_cast<gert::TensorData**>(tensor_data_vec->MutableData());
  for (size_t i = 0; i < shape_num; i++) {
    tensor_data_addr[i] = new (std::nothrow) gert::TensorData();
    tensor_data_addr[i]->SetAddr(x_vec.data(), nullptr);
  }

  gert::StorageShape x_shape = {{2,6}, {2,6}};

  // build input shape & input tensor
  size_t total_size = 0;
  auto output_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto output_tensor = reinterpret_cast<gert::Tensor*>(output_tensor_holder.get());
  output_tensor->SetOriginFormat(ge::FORMAT_ND);
  output_tensor->SetStorageFormat(ge::FORMAT_ND);
  output_tensor->SetPlacement(gert::kFollowing);
  output_tensor->MutableOriginShape() = {};
  output_tensor->MutableStorageShape() = {};

  auto output_tensor_data = reinterpret_cast<uint64_t*>(output_tensor->GetAddr());
  output_tensor_data[0] = 1;
  output_tensor->SetData(gert::TensorData{output_tensor_data, nullptr});
  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(9, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id),
                                   tensor_datas.get(), nullptr, reinterpret_cast<void*>(input_num),
                                   reinterpret_cast<void*>(keep_dims), reinterpret_cast<void*>(axis),
                                   reinterpret_cast<void*>(&x_shape), reinterpret_cast<void*>(output_tensor)})
                          .NodeIoNum(2, 1)
                          .IrInputNum({2})
                          .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  std::cout << "splitToSequenceCompute->run_func : " << (void*)(splitToSequenceCompute->run_func) << std::endl;
  ASSERT_EQ(splitToSequenceCompute->run_func(context), ge::PARAM_INVALID);
  for (size_t i = 0; i < shape_num; i++) {
    delete tensor_data_addr[i];
    tensor_data_addr[i] = nullptr;
  }
}

TEST_F(SplitToSequenceComputeTest, input_x_storage_shape_null) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  int32_t axis = 1;
  int32_t keep_dims = 1;
  uint32_t input_num = 2;

  uint32_t shape_num = 2;
  auto shapes = gert::ContinuousVector::Create<gert::Shape>(shape_num);
  auto shapes_vec = reinterpret_cast<gert::ContinuousVector*>(shapes.get());
  shapes_vec->SetSize(shape_num);
  auto shapes_data = static_cast<gert::Shape*>(shapes_vec->MutableData());
  for (size_t i = 0; i < shape_num; i++) {
    shapes_data[i].SetDimNum(2);
    shapes_data[i].SetDim(0, 2);
    shapes_data[i].SetDim(1, 3);
  }

  vector<int64_t> x_vec {1,2,3,4,5,6,7,8,9,10,11,12};
  auto tensor_datas = gert::ContinuousVector::Create<gert::TensorData*>(shape_num);
  auto tensor_data_vec = reinterpret_cast<gert::ContinuousVector*>(tensor_datas.get());
  tensor_data_vec->SetSize(shape_num);
  auto tensor_data_addr = static_cast<gert::TensorData**>(tensor_data_vec->MutableData());
  for (size_t i = 0; i < shape_num; i++) {
    tensor_data_addr[i] = new (std::nothrow) gert::TensorData();
    tensor_data_addr[i]->SetAddr(x_vec.data(), nullptr);
  }

  // build input shape & input tensor
  size_t total_size = 0;
  auto output_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto output_tensor = reinterpret_cast<gert::Tensor*>(output_tensor_holder.get());
  output_tensor->SetOriginFormat(ge::FORMAT_ND);
  output_tensor->SetStorageFormat(ge::FORMAT_ND);
  output_tensor->SetPlacement(gert::kFollowing);
  output_tensor->MutableOriginShape() = {};
  output_tensor->MutableStorageShape() = {};

  auto output_tensor_data = reinterpret_cast<uint64_t*>(output_tensor->GetAddr());
  output_tensor_data[0] = 1;
  output_tensor->SetData(gert::TensorData{output_tensor_data, nullptr});
  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(9, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id),
                                   tensor_datas.get(), shapes.get(), reinterpret_cast<void*>(input_num),
                                   reinterpret_cast<void*>(keep_dims), reinterpret_cast<void*>(axis),
                                   nullptr, reinterpret_cast<void*>(output_tensor)})
                          .NodeIoNum(2, 1)
                          .IrInputNum({2})
                          .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  std::cout << "splitToSequenceCompute->run_func : " << (void*)(splitToSequenceCompute->run_func) << std::endl;
  ASSERT_EQ(splitToSequenceCompute->run_func(context), ge::PARAM_INVALID);
  for (size_t i = 0; i < shape_num; i++) {
    delete tensor_data_addr[i];
    tensor_data_addr[i] = nullptr;
  }
}


TEST_F(SplitToSequenceComputeTest, axis) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  int32_t axis = -1;
  int32_t keep_dims = 1;
  uint32_t input_num = 2;

  uint32_t shape_num = 2;
  auto shapes = gert::ContinuousVector::Create<gert::Shape>(shape_num);
  auto shapes_vec = reinterpret_cast<gert::ContinuousVector*>(shapes.get());
  shapes_vec->SetSize(shape_num);
  auto shapes_data = static_cast<gert::Shape*>(shapes_vec->MutableData());
  for (size_t i = 0; i < shape_num; i++) {
    shapes_data[i].SetDimNum(2);
    shapes_data[i].SetDim(0, 2);
    shapes_data[i].SetDim(1, 3);
  }

  vector<int64_t> x_vec {1,2,3,4,5,6,7,8,9,10,11,12};
  auto tensor_datas = gert::ContinuousVector::Create<gert::TensorData*>(shape_num);
  auto tensor_data_vec = reinterpret_cast<gert::ContinuousVector*>(tensor_datas.get());
  tensor_data_vec->SetSize(shape_num);
  auto tensor_data_addr = static_cast<gert::TensorData**>(tensor_data_vec->MutableData());
  for (size_t i = 0; i < shape_num; i++) {
    tensor_data_addr[i] = new (std::nothrow) gert::TensorData();
    tensor_data_addr[i]->SetAddr(x_vec.data(), nullptr);
  }

  gert::StorageShape x_shape = {{2,6}, {2,6}};

  // build input shape & input tensor
  size_t total_size = 0;
  auto output_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto output_tensor = reinterpret_cast<gert::Tensor*>(output_tensor_holder.get());
  output_tensor->SetOriginFormat(ge::FORMAT_ND);
  output_tensor->SetStorageFormat(ge::FORMAT_ND);
  output_tensor->SetPlacement(gert::kFollowing);
  output_tensor->MutableOriginShape() = {};
  output_tensor->MutableStorageShape() = {};

  auto output_tensor_data = reinterpret_cast<uint64_t*>(output_tensor->GetAddr());
  output_tensor_data[0] = 1;
  output_tensor->SetData(gert::TensorData{output_tensor_data, nullptr});
  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(9, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id),
                                   tensor_datas.get(), shapes.get(), reinterpret_cast<void*>(input_num),
                                   reinterpret_cast<void*>(keep_dims), reinterpret_cast<void*>(axis),
                                   reinterpret_cast<void*>(&x_shape), reinterpret_cast<void*>(output_tensor)})
                          .NodeIoNum(2, 1)
                          .IrInputNum({2})
                          .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  std::cout << "splitToSequenceCompute->run_func : " << (void*)(splitToSequenceCompute->run_func) << std::endl;
  ASSERT_EQ(splitToSequenceCompute->run_func(context), ge::GRAPH_SUCCESS);
  for (size_t i = 0; i < shape_num; i++) {
    delete tensor_data_addr[i];
    tensor_data_addr[i] = nullptr;
  }
}


TEST_F(SplitToSequenceComputeTest, inner_tensor_addrs_null) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  int32_t axis = 1;
  int32_t keep_dims = 1;
  uint32_t input_num = 2;

  uint32_t shape_num = 2;
  auto shapes = gert::ContinuousVector::Create<gert::Shape>(shape_num);
  auto shapes_vec = reinterpret_cast<gert::ContinuousVector*>(shapes.get());
  shapes_vec->SetSize(shape_num);
  auto shapes_data = static_cast<gert::Shape*>(shapes_vec->MutableData());
  for (size_t i = 0; i < shape_num; i++) {
    shapes_data[i].SetDimNum(2);
    shapes_data[i].SetDim(0, 2);
    shapes_data[i].SetDim(1, 3);
  }

  vector<int64_t> x_vec {1,2,3,4,5,6,7,8,9,10,11,12};
  auto tensor_datas = gert::ContinuousVector::Create<gert::TensorData*>(shape_num);
  auto tensor_data_vec = reinterpret_cast<gert::ContinuousVector*>(tensor_datas.get());
  tensor_data_vec->SetSize(shape_num);
  auto tensor_data_addr = static_cast<gert::TensorData**>(tensor_data_vec->MutableData());
  for (size_t i = 0; i < shape_num; i++) {
    tensor_data_addr[i] = new (std::nothrow) gert::TensorData();
    tensor_data_addr[i]->SetAddr(x_vec.data(), nullptr);
  }

  gert::StorageShape x_shape = {{2,6}, {2,6}};

  // build input shape & input tensor
  size_t total_size = 0;
  auto output_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto output_tensor = reinterpret_cast<gert::Tensor*>(output_tensor_holder.get());
  output_tensor->SetOriginFormat(ge::FORMAT_ND);
  output_tensor->SetStorageFormat(ge::FORMAT_ND);
  output_tensor->SetPlacement(gert::kFollowing);
  output_tensor->MutableOriginShape() = {};
  output_tensor->MutableStorageShape() = {};

  auto output_tensor_data = reinterpret_cast<uint64_t*>(output_tensor->GetAddr());
  output_tensor_data[0] = 1;
  output_tensor->SetData(gert::TensorData{output_tensor_data, nullptr});
  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(9, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id),
                                   nullptr, shapes.get(), reinterpret_cast<void*>(input_num),
                                   reinterpret_cast<void*>(keep_dims), reinterpret_cast<void*>(axis),
                                   reinterpret_cast<void*>(&x_shape), reinterpret_cast<void*>(output_tensor)})
                          .NodeIoNum(2, 1)
                          .IrInputNum({2})
                          .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  std::cout << "splitToSequenceCompute->run_func : " << (void*)(splitToSequenceCompute->run_func) << std::endl;
  ASSERT_EQ(splitToSequenceCompute->run_func(context), ge::PARAM_INVALID);
  for (size_t i = 0; i < shape_num; i++) {
    delete tensor_data_addr[i];
    tensor_data_addr[i] = nullptr;
  }
}

TEST_F(SplitToSequenceComputeTest, output_null) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  int32_t axis = 1;
  int32_t keep_dims = 1;
  uint32_t input_num = 2;

  uint32_t shape_num = 2;
  auto shapes = gert::ContinuousVector::Create<gert::Shape>(shape_num);
  auto shapes_vec = reinterpret_cast<gert::ContinuousVector*>(shapes.get());
  shapes_vec->SetSize(shape_num);
  auto shapes_data = static_cast<gert::Shape*>(shapes_vec->MutableData());
  for (size_t i = 0; i < shape_num; i++) {
    shapes_data[i].SetDimNum(2);
    shapes_data[i].SetDim(0, 2);
    shapes_data[i].SetDim(1, 3);
  }

  vector<int64_t> x_vec {1,2,3,4,5,6,7,8,9,10,11,12};
  auto tensor_datas = gert::ContinuousVector::Create<gert::TensorData*>(shape_num);
  auto tensor_data_vec = reinterpret_cast<gert::ContinuousVector*>(tensor_datas.get());
  tensor_data_vec->SetSize(shape_num);
  auto tensor_data_addr = static_cast<gert::TensorData**>(tensor_data_vec->MutableData());
  for (size_t i = 0; i < shape_num; i++) {
    tensor_data_addr[i] = new (std::nothrow) gert::TensorData();
    tensor_data_addr[i]->SetAddr(x_vec.data(), nullptr);
  }

  gert::StorageShape x_shape = {{2,6}, {2,6}};

  // build input shape & input tensor
  size_t total_size = 0;
  auto output_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto output_tensor = reinterpret_cast<gert::Tensor*>(output_tensor_holder.get());
  output_tensor->SetOriginFormat(ge::FORMAT_ND);
  output_tensor->SetStorageFormat(ge::FORMAT_ND);
  output_tensor->SetPlacement(gert::kFollowing);
  output_tensor->MutableOriginShape() = {};
  output_tensor->MutableStorageShape() = {};

  auto output_tensor_data = reinterpret_cast<uint64_t*>(output_tensor->GetAddr());
  output_tensor_data[0] = 1;
  output_tensor->SetData(gert::TensorData{output_tensor_data, nullptr});
  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(9, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id),
                                   tensor_datas.get(), shapes.get(), reinterpret_cast<void*>(input_num),
                                   reinterpret_cast<void*>(keep_dims), reinterpret_cast<void*>(axis),
                                   reinterpret_cast<void*>(&x_shape), nullptr})
                          .NodeIoNum(2, 1)
                          .IrInputNum({2})
                          .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  std::cout << "splitToSequenceCompute->run_func : " << (void*)(splitToSequenceCompute->run_func) << std::endl;
  ASSERT_EQ(splitToSequenceCompute->run_func(context), ge::PARAM_INVALID);
  for (size_t i = 0; i < shape_num; i++) {
    delete tensor_data_addr[i];
    tensor_data_addr[i] = nullptr;
  }
}
}  // namespace gert