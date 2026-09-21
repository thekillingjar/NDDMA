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
#include "aicpu/converter/sequence/tensor_sequence.h"
#include "aicpu/converter/sequence/resource_mgr.h"

namespace gert {
class SequenceAtComputeTest : public testing::Test {
public:
  SequenceAtComputeTest() {
    SequenceAtCompute = KernelRegistry::GetInstance().FindKernelFuncs("SequenceAtCompute");
  }
  const KernelRegistry::KernelFuncs* SequenceAtCompute;
};


TEST_F(SequenceAtComputeTest, int32_success) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;

  // create handle and tensor sequence
  gert::ResourceMgrPtr out_rm;
  gert::SessionMgr::GetInstance()->GetRm(session_id, container_id, out_rm);
  uint64_t handle = out_rm->GetHandle();
  gert::TensorSeqPtr tensor_seq_ptr = std::make_shared<gert::TensorSeq>(ge::DT_INT32);
  size_t total_size = 0;
  auto ele_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_INT32, total_size);
  auto ele_tensor = reinterpret_cast<gert::Tensor*>(ele_tensor_holder.get());
  tensor_seq_ptr->Add(*ele_tensor);
  out_rm->Create(handle, tensor_seq_ptr);

  // build input handle tensor
  auto handle_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto handle_tensor = reinterpret_cast<gert::Tensor*>(handle_tensor_holder.get());
  handle_tensor->SetOriginFormat(ge::FORMAT_ND);
  handle_tensor->SetStorageFormat(ge::FORMAT_ND);
  handle_tensor->SetPlacement(gert::kFollowing);
  handle_tensor->MutableOriginShape() = {};
  handle_tensor->MutableStorageShape() = {};

  auto handle_tensor_data = reinterpret_cast<uint64_t*>(handle_tensor->GetAddr());
  handle_tensor_data[0] = handle;
  handle_tensor->SetData(gert::TensorData{handle_tensor_data, nullptr});

  // build input index tensor
  auto input_index_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_INT32, total_size);
  auto input_index_tensor = reinterpret_cast<gert::Tensor*>(input_index_tensor_holder.get());
  input_index_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_index_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_index_tensor->SetPlacement(gert::kFollowing);
  input_index_tensor->MutableOriginShape() = {};
  input_index_tensor->MutableStorageShape() = {2,3,4};
  input_index_tensor->SetDataType(ge::DT_INT32);

  auto input_index_tensor_data = reinterpret_cast<int32_t*>(input_index_tensor->GetAddr());
  input_index_tensor_data[0] = 0;
  input_index_tensor->SetData(gert::TensorData{input_index_tensor_data, nullptr});

  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(4, 1)
                          .Inputs({reinterpret_cast<void*>(session_id),
                                   reinterpret_cast<void*>(container_id),
                                   reinterpret_cast<void*>(&(handle_tensor->MutableTensorData())),
                                   reinterpret_cast<void*>(&(input_index_tensor->MutableTensorData()))})
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({2})
                          .NodeInputTd(0, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_EQ(SequenceAtCompute->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(SequenceAtCompute->run_func(context), ge::GRAPH_SUCCESS);
}

TEST_F(SequenceAtComputeTest, int64_success) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;

  // create handle and tensor sequence
  gert::ResourceMgrPtr out_rm;
  gert::SessionMgr::GetInstance()->GetRm(session_id, container_id, out_rm);
  uint64_t handle = out_rm->GetHandle();
  gert::TensorSeqPtr tensor_seq_ptr = std::make_shared<gert::TensorSeq>(ge::DT_INT64);
  size_t total_size = 0;
  auto ele_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_INT64, total_size);
  auto ele_tensor = reinterpret_cast<gert::Tensor*>(ele_tensor_holder.get());
  tensor_seq_ptr->Add(*ele_tensor);
  out_rm->Create(handle, tensor_seq_ptr);

  // build input handle tensor
  auto handle_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto handle_tensor = reinterpret_cast<gert::Tensor*>(handle_tensor_holder.get());
  handle_tensor->SetOriginFormat(ge::FORMAT_ND);
  handle_tensor->SetStorageFormat(ge::FORMAT_ND);
  handle_tensor->SetPlacement(gert::kFollowing);
  handle_tensor->MutableOriginShape() = {};
  handle_tensor->MutableStorageShape() = {};

  auto handle_tensor_data = reinterpret_cast<uint64_t*>(handle_tensor->GetAddr());
  handle_tensor_data[0] = handle;
  handle_tensor->SetData(gert::TensorData{handle_tensor_data, nullptr});

  // build input index tensor
  auto input_index_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_INT64, total_size);
  auto input_index_tensor = reinterpret_cast<gert::Tensor*>(input_index_tensor_holder.get());
  input_index_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_index_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_index_tensor->SetPlacement(gert::kFollowing);
  input_index_tensor->MutableOriginShape() = {};
  input_index_tensor->MutableStorageShape() = {};
  input_index_tensor->SetDataType(ge::DT_INT64);

  auto input_index_tensor_data = reinterpret_cast<int64_t*>(input_index_tensor->GetAddr());
  input_index_tensor_data[0] = 0;
  input_index_tensor->SetData(gert::TensorData{input_index_tensor_data, nullptr});

  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(4, 1)
                          .Inputs({reinterpret_cast<void*>(session_id),
                                   reinterpret_cast<void*>(container_id),
                                   reinterpret_cast<void*>(&(handle_tensor->MutableTensorData())),
                                   reinterpret_cast<void*>(&(input_index_tensor->MutableTensorData()))})
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({2})
                          .NodeInputTd(0, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_EQ(SequenceAtCompute->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(SequenceAtCompute->run_func(context), ge::GRAPH_SUCCESS);
}


TEST_F(SequenceAtComputeTest, uint64_fail) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;

  // create handle and tensor sequence
  gert::ResourceMgrPtr out_rm;
  gert::SessionMgr::GetInstance()->GetRm(session_id, container_id, out_rm);
  uint64_t handle = out_rm->GetHandle();
  gert::TensorSeqPtr tensor_seq_ptr = std::make_shared<gert::TensorSeq>(ge::DT_UINT64);
  size_t total_size = 0;
  auto ele_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto ele_tensor = reinterpret_cast<gert::Tensor*>(ele_tensor_holder.get());
  tensor_seq_ptr->Add(*ele_tensor);
  out_rm->Create(handle, tensor_seq_ptr);

  // build input handle tensor
  auto handle_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto handle_tensor = reinterpret_cast<gert::Tensor*>(handle_tensor_holder.get());
  handle_tensor->SetOriginFormat(ge::FORMAT_ND);
  handle_tensor->SetStorageFormat(ge::FORMAT_ND);
  handle_tensor->SetPlacement(gert::kFollowing);
  handle_tensor->MutableOriginShape() = {};
  handle_tensor->MutableStorageShape() = {};

  auto handle_tensor_data = reinterpret_cast<uint64_t*>(handle_tensor->GetAddr());
  handle_tensor_data[0] = handle;
  handle_tensor->SetData(gert::TensorData{handle_tensor_data, nullptr});

  // build input index tensor
  auto input_index_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto input_index_tensor = reinterpret_cast<gert::Tensor*>(input_index_tensor_holder.get());
  input_index_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_index_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_index_tensor->SetPlacement(gert::kFollowing);
  input_index_tensor->MutableOriginShape() = {};
  input_index_tensor->MutableStorageShape() = {};
  input_index_tensor->SetDataType(ge::DT_UINT64);

  auto input_index_tensor_data = reinterpret_cast<uint64_t*>(input_index_tensor->GetAddr());
  input_index_tensor_data[0] = 0;
  input_index_tensor->SetData(gert::TensorData{input_index_tensor_data, nullptr});

  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(4, 1)
                          .Inputs({reinterpret_cast<void*>(session_id),
                                   reinterpret_cast<void*>(container_id),
                                   reinterpret_cast<void*>(&(handle_tensor->MutableTensorData())),
                                   reinterpret_cast<void*>(&(input_index_tensor->MutableTensorData()))})
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({2})
                          .NodeInputTd(0, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_EQ(SequenceAtCompute->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(SequenceAtCompute->run_func(context), ge::PARAM_INVALID);
}

TEST_F(SequenceAtComputeTest, handle_fail) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;

  // create handle and tensor sequence
  gert::ResourceMgrPtr out_rm;
  gert::SessionMgr::GetInstance()->GetRm(session_id, container_id, out_rm);
  uint64_t handle = out_rm->GetHandle();
  gert::TensorSeqPtr tensor_seq_ptr = std::make_shared<gert::TensorSeq>(ge::DT_INT64);
  size_t total_size = 0;
  auto ele_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_INT64, total_size);
  auto ele_tensor = reinterpret_cast<gert::Tensor*>(ele_tensor_holder.get());
  tensor_seq_ptr->Add(*ele_tensor);
  out_rm->Create(handle, tensor_seq_ptr);

  // build input handle tensor
  auto handle_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto handle_tensor = reinterpret_cast<gert::Tensor*>(handle_tensor_holder.get());
  handle_tensor->SetOriginFormat(ge::FORMAT_ND);
  handle_tensor->SetStorageFormat(ge::FORMAT_ND);
  handle_tensor->SetPlacement(gert::kFollowing);
  handle_tensor->MutableOriginShape() = {};
  handle_tensor->MutableStorageShape() = {};

  auto handle_tensor_data = reinterpret_cast<uint64_t*>(handle_tensor->GetAddr());
  handle_tensor_data[0] = handle + 100;
  handle_tensor->SetData(gert::TensorData{handle_tensor_data, nullptr});

  // build input index tensor
  auto input_index_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_INT64, total_size);
  auto input_index_tensor = reinterpret_cast<gert::Tensor*>(input_index_tensor_holder.get());
  input_index_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_index_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_index_tensor->SetPlacement(gert::kFollowing);
  input_index_tensor->MutableOriginShape() = {};
  input_index_tensor->MutableStorageShape() = {};
  input_index_tensor->SetDataType(ge::DT_INT64);

  auto input_index_tensor_data = reinterpret_cast<int64_t*>(input_index_tensor->GetAddr());
  input_index_tensor_data[0] = 0;
  input_index_tensor->SetData(gert::TensorData{input_index_tensor_data, nullptr});

  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(4, 1)
                          .Inputs({reinterpret_cast<void*>(session_id),
                                   reinterpret_cast<void*>(container_id),
                                   reinterpret_cast<void*>(&(handle_tensor->MutableTensorData())),
                                   reinterpret_cast<void*>(&(input_index_tensor->MutableTensorData()))})
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({2})
                          .NodeInputTd(0, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_EQ(SequenceAtCompute->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(SequenceAtCompute->run_func(context), ge::PARAM_INVALID);
}

TEST_F(SequenceAtComputeTest, handle_null) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;

  // create handle and tensor sequence
  gert::ResourceMgrPtr out_rm;
  gert::SessionMgr::GetInstance()->GetRm(session_id, container_id, out_rm);
  uint64_t handle = out_rm->GetHandle();
  gert::TensorSeqPtr tensor_seq_ptr = std::make_shared<gert::TensorSeq>(ge::DT_INT64);
  size_t total_size = 0;
  auto ele_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_INT64, total_size);
  auto ele_tensor = reinterpret_cast<gert::Tensor*>(ele_tensor_holder.get());
  tensor_seq_ptr->Add(*ele_tensor);
  out_rm->Create(handle, tensor_seq_ptr);

  // build input handle tensor
  auto handle_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto handle_tensor = reinterpret_cast<gert::Tensor*>(handle_tensor_holder.get());
  handle_tensor->SetOriginFormat(ge::FORMAT_ND);
  handle_tensor->SetStorageFormat(ge::FORMAT_ND);
  handle_tensor->SetPlacement(gert::kFollowing);
  handle_tensor->MutableOriginShape() = {};
  handle_tensor->MutableStorageShape() = {};

  auto handle_tensor_data = reinterpret_cast<uint64_t*>(handle_tensor->GetAddr());
  handle_tensor_data[0] = handle;
  handle_tensor->SetData(gert::TensorData{handle_tensor_data, nullptr});

  // build input index tensor
  auto input_index_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_INT64, total_size);
  auto input_index_tensor = reinterpret_cast<gert::Tensor*>(input_index_tensor_holder.get());
  input_index_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_index_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_index_tensor->SetPlacement(gert::kFollowing);
  input_index_tensor->MutableOriginShape() = {};
  input_index_tensor->MutableStorageShape() = {};
  input_index_tensor->SetDataType(ge::DT_INT64);

  auto input_index_tensor_data = reinterpret_cast<int64_t*>(input_index_tensor->GetAddr());
  input_index_tensor_data[0] = 0;
  input_index_tensor->SetData(gert::TensorData{input_index_tensor_data, nullptr});

  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(4, 1)
                          .Inputs({reinterpret_cast<void*>(session_id),
                                   reinterpret_cast<void*>(container_id),
                                   nullptr,
                                   reinterpret_cast<void*>(&(input_index_tensor->MutableTensorData()))})
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({2})
                          .NodeInputTd(0, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_EQ(SequenceAtCompute->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(SequenceAtCompute->run_func(context), ge::PARAM_INVALID);
}

TEST_F(SequenceAtComputeTest, index_data_null) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;

  // create handle and tensor sequence
  gert::ResourceMgrPtr out_rm;
  gert::SessionMgr::GetInstance()->GetRm(session_id, container_id, out_rm);
  uint64_t handle = out_rm->GetHandle();
  gert::TensorSeqPtr tensor_seq_ptr = std::make_shared<gert::TensorSeq>(ge::DT_INT64);
  size_t total_size = 0;
  auto ele_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_INT64, total_size);
  auto ele_tensor = reinterpret_cast<gert::Tensor*>(ele_tensor_holder.get());
  tensor_seq_ptr->Add(*ele_tensor);
  out_rm->Create(handle, tensor_seq_ptr);

  // build input handle tensor
  auto handle_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto handle_tensor = reinterpret_cast<gert::Tensor*>(handle_tensor_holder.get());
  handle_tensor->SetOriginFormat(ge::FORMAT_ND);
  handle_tensor->SetStorageFormat(ge::FORMAT_ND);
  handle_tensor->SetPlacement(gert::kFollowing);
  handle_tensor->MutableOriginShape() = {};
  handle_tensor->MutableStorageShape() = {};

  auto handle_tensor_data = reinterpret_cast<uint64_t*>(handle_tensor->GetAddr());
  handle_tensor_data[0] = handle;
  handle_tensor->SetData(gert::TensorData{handle_tensor_data, nullptr});

  // build input index tensor
  auto input_index_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_INT64, total_size);
  auto input_index_tensor = reinterpret_cast<gert::Tensor*>(input_index_tensor_holder.get());
  input_index_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_index_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_index_tensor->SetPlacement(gert::kFollowing);
  input_index_tensor->MutableOriginShape() = {};
  input_index_tensor->MutableStorageShape() = {};
  input_index_tensor->SetDataType(ge::DT_INT64);

  auto input_index_tensor_data = reinterpret_cast<int64_t*>(input_index_tensor->GetAddr());
  input_index_tensor_data[0] = 0;
  input_index_tensor->SetData(gert::TensorData{input_index_tensor_data, nullptr});

  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(4, 1)
                          .Inputs({reinterpret_cast<void*>(session_id),
                                   reinterpret_cast<void*>(container_id),
                                   reinterpret_cast<void*>(&(handle_tensor->MutableTensorData())),
                                   nullptr})
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({2})
                          .NodeInputTd(0, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_EQ(SequenceAtCompute->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(SequenceAtCompute->run_func(context), ge::PARAM_INVALID);
}

TEST_F(SequenceAtComputeTest, tensor_ref_null) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;

  // create handle and tensor sequence
  gert::ResourceMgrPtr out_rm;
  gert::SessionMgr::GetInstance()->GetRm(session_id, container_id, out_rm);
  uint64_t handle = out_rm->GetHandle();
  gert::TensorSeqPtr tensor_seq_ptr = std::make_shared<gert::TensorSeq>(ge::DT_INT64);
  out_rm->Create(handle, tensor_seq_ptr);
  size_t total_size = 0;

  // build input handle tensor
  auto handle_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto handle_tensor = reinterpret_cast<gert::Tensor*>(handle_tensor_holder.get());
  handle_tensor->SetOriginFormat(ge::FORMAT_ND);
  handle_tensor->SetStorageFormat(ge::FORMAT_ND);
  handle_tensor->SetPlacement(gert::kFollowing);
  handle_tensor->MutableOriginShape() = {};
  handle_tensor->MutableStorageShape() = {};

  auto handle_tensor_data = reinterpret_cast<uint64_t*>(handle_tensor->GetAddr());
  handle_tensor_data[0] = handle;
  handle_tensor->SetData(gert::TensorData{handle_tensor_data, nullptr});

  // build input index tensor
  auto input_index_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_INT64, total_size);
  auto input_index_tensor = reinterpret_cast<gert::Tensor*>(input_index_tensor_holder.get());
  input_index_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_index_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_index_tensor->SetPlacement(gert::kFollowing);
  input_index_tensor->MutableOriginShape() = {};
  input_index_tensor->MutableStorageShape() = {};
  input_index_tensor->SetDataType(ge::DT_INT64);

  auto input_index_tensor_data = reinterpret_cast<int64_t*>(input_index_tensor->GetAddr());
  input_index_tensor_data[0] = 0;
  input_index_tensor->SetData(gert::TensorData{input_index_tensor_data, nullptr});

  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(4, 1)
                          .Inputs({reinterpret_cast<void*>(session_id),
                                   reinterpret_cast<void*>(container_id),
                                   reinterpret_cast<void*>(&(handle_tensor->MutableTensorData())),
                                   reinterpret_cast<void*>(&(input_index_tensor->MutableTensorData()))})
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({2})
                          .NodeInputTd(0, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_EQ(SequenceAtCompute->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(SequenceAtCompute->run_func(context), ge::PARAM_INVALID);
}
}  // namespace gert
