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
struct SequenceEraseComputeTest : public testing::Test {
  SequenceEraseComputeTest() {
    sequenceEraseCompute = KernelRegistry::GetInstance().FindKernelFuncs("SequenceEraseCompute");
  }
  const KernelRegistry::KernelFuncs* sequenceEraseCompute;
  KernelRegistry& registry = KernelRegistry::GetInstance();
};

TEST_F(SequenceEraseComputeTest, failed) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  uint32_t input_num = 0;
  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(3, 0)
                          .Inputs({
                              reinterpret_cast<void*>(session_id),
                              reinterpret_cast<void*>(container_id),
                              reinterpret_cast<void*>(input_num),
                          })
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_EQ(sequenceEraseCompute->run_func(context), ge::PARAM_INVALID);
}

TEST_F(SequenceEraseComputeTest, failed_index) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  uint32_t input_num = 1;

  gert::ResourceMgrPtr out_rm;
  gert::SessionMgr::GetInstance()->GetRm(session_id, container_id, out_rm);
  uint64_t handle = out_rm->GetHandle();
  gert::TensorSeqPtr tensor_seq_ptr = std::make_shared<gert::TensorSeq>(ge::DT_INT32);
  out_rm->Create(handle, tensor_seq_ptr);

  // build input handle tensor

  size_t total_size = 0;
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

  // build input shape & input tensor
  auto input_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_INT32, total_size);
  auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
  input_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_tensor->SetPlacement(gert::kFollowing);
  input_tensor->MutableOriginShape() = {};
  input_tensor->MutableStorageShape() = {};

  auto input_tensor_data = reinterpret_cast<int32_t*>(input_tensor->GetAddr());
  input_tensor_data[0] = 0;
  input_tensor->SetData(gert::TensorData{input_tensor_data, nullptr});

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
                          .KernelIONum(6, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id),
                                   reinterpret_cast<void*>(input_num), 
                                   reinterpret_cast<void*>(&(handle_tensor->MutableTensorData())),
                                   reinterpret_cast<void*>(&(input_tensor->MutableTensorData())),
                                   reinterpret_cast<void*>(output_tensor)})
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({2})
                          .NodeInputTd(0, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_EQ(sequenceEraseCompute->run_func(context), ge::PARAM_INVALID);
}

TEST_F(SequenceEraseComputeTest, index_type_invalid) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  uint32_t input_num = 2;

  gert::ResourceMgrPtr out_rm;
  gert::SessionMgr::GetInstance()->GetRm(session_id, container_id, out_rm);
  uint64_t handle = out_rm->GetHandle();
  gert::TensorSeqPtr tensor_seq_ptr = std::make_shared<gert::TensorSeq>(ge::DT_INT32);
  out_rm->Create(handle, tensor_seq_ptr);

  // build input handle tensor
  size_t total_size = 0;
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
  auto input_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_FLOAT, total_size);
  auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
  input_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_tensor->SetPlacement(gert::kFollowing);
  input_tensor->MutableOriginShape() = {};
  input_tensor->MutableStorageShape() = {};

  auto input_tensor_data = reinterpret_cast<float*>(input_tensor->GetAddr());
  input_tensor_data[0] = 0.0F;
  input_tensor->SetData(gert::TensorData{input_tensor_data, nullptr});

  //build output tensor
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
                          .KernelIONum(6, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id),
                                   reinterpret_cast<void*>(input_num), 
                                   reinterpret_cast<void*>(&(handle_tensor->MutableTensorData())),
                                   reinterpret_cast<void*>(&(input_tensor->MutableTensorData())),
                                   reinterpret_cast<void*>(output_tensor)})
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({2})
                          .NodeInputTd(0, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_EQ(sequenceEraseCompute->run_func(context), ge::PARAM_INVALID);
}

TEST_F(SequenceEraseComputeTest, success) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  uint32_t input_num = 2;

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
  auto input_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_INT32, total_size);
  auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
  input_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_tensor->SetPlacement(gert::kFollowing);
  input_tensor->MutableOriginShape() = {};
  input_tensor->MutableStorageShape() = {};

  auto input_tensor_data = reinterpret_cast<float*>(input_tensor->GetAddr());
  input_tensor_data[0] = 0;
  input_tensor->SetData(gert::TensorData{input_tensor_data, nullptr});

  //build output tensor
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
                          .KernelIONum(6, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id),
                                   reinterpret_cast<void*>(input_num), 
                                   reinterpret_cast<void*>(&(handle_tensor->MutableTensorData())),
                                   reinterpret_cast<void*>(&(input_tensor->MutableTensorData())),
                                   reinterpret_cast<void*>(output_tensor)})
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({2})
                          .NodeInputTd(0, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_EQ(sequenceEraseCompute->run_func(context), ge::GRAPH_SUCCESS);
}

TEST_F(SequenceEraseComputeTest, input_handle_null) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  uint32_t input_num = 2;

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
  auto input_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_INT32, total_size);
  auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
  input_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_tensor->SetPlacement(gert::kFollowing);
  input_tensor->MutableOriginShape() = {};
  input_tensor->MutableStorageShape() = {};

  auto input_tensor_data = reinterpret_cast<float*>(input_tensor->GetAddr());
  input_tensor_data[0] = 0;
  input_tensor->SetData(gert::TensorData{input_tensor_data, nullptr});

  //build output tensor
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
                          .KernelIONum(6, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id),
                                   reinterpret_cast<void*>(input_num), 
                                   nullptr,
                                   reinterpret_cast<void*>(&(input_tensor->MutableTensorData())),
                                   reinterpret_cast<void*>(output_tensor)})
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({2})
                          .NodeInputTd(0, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_EQ(sequenceEraseCompute->run_func(context), ge::PARAM_INVALID);
}

TEST_F(SequenceEraseComputeTest, input_handle_addr_null) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  uint32_t input_num = 2;

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
  handle_tensor->SetData(gert::TensorData{nullptr, nullptr});

  // build input index tensor
  auto input_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_INT32, total_size);
  auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
  input_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_tensor->SetPlacement(gert::kFollowing);
  input_tensor->MutableOriginShape() = {};
  input_tensor->MutableStorageShape() = {};

  auto input_tensor_data = reinterpret_cast<float*>(input_tensor->GetAddr());
  input_tensor_data[0] = 0;
  input_tensor->SetData(gert::TensorData{input_tensor_data, nullptr});

  //build output tensor
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
                          .KernelIONum(6, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id),
                                   reinterpret_cast<void*>(input_num), 
                                   reinterpret_cast<void*>(&(handle_tensor->MutableTensorData())),
                                   reinterpret_cast<void*>(&(input_tensor->MutableTensorData())),
                                   reinterpret_cast<void*>(output_tensor)})
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({2})
                          .NodeInputTd(0, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_EQ(sequenceEraseCompute->run_func(context), ge::PARAM_INVALID);
}

TEST_F(SequenceEraseComputeTest, int64_fail) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  uint32_t input_num = 2;

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
  auto input_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_INT64, total_size);
  auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
  input_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_tensor->SetPlacement(gert::kFollowing);
  input_tensor->MutableOriginShape() = {};
  input_tensor->MutableStorageShape() = {};

  auto input_tensor_data = reinterpret_cast<float*>(input_tensor->GetAddr());
  input_tensor_data[0] = 0;
  input_tensor->SetData(gert::TensorData{input_tensor_data, nullptr});

  //build output tensor
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
                          .KernelIONum(6, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id),
                                   reinterpret_cast<void*>(input_num), 
                                   reinterpret_cast<void*>(&(handle_tensor->MutableTensorData())),
                                   reinterpret_cast<void*>(&(input_tensor->MutableTensorData())),
                                   reinterpret_cast<void*>(output_tensor)})
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({2})
                          .NodeInputTd(0, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_EQ(sequenceEraseCompute->run_func(context), ge::PARAM_INVALID);
}

TEST_F(SequenceEraseComputeTest, output_null) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  uint32_t input_num = 2;

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
  auto input_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_INT32, total_size);
  auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
  input_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_tensor->SetPlacement(gert::kFollowing);
  input_tensor->MutableOriginShape() = {};
  input_tensor->MutableStorageShape() = {};

  auto input_tensor_data = reinterpret_cast<float*>(input_tensor->GetAddr());
  input_tensor_data[0] = 0;
  input_tensor->SetData(gert::TensorData{input_tensor_data, nullptr});

  //build output tensor
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
                          .KernelIONum(6, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id),
                                   reinterpret_cast<void*>(input_num), 
                                   reinterpret_cast<void*>(&(handle_tensor->MutableTensorData())),
                                   reinterpret_cast<void*>(&(input_tensor->MutableTensorData())),
                                   nullptr})
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({2})
                          .NodeInputTd(0, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_EQ(sequenceEraseCompute->run_func(context), ge::PARAM_INVALID);
}
}  // namespace gert