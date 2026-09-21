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
#include<fstream>
#include "graph_metadef/graph/utils/file_utils.h"
#include "mmpa/src/mmpa_stub.h"
#include "faker/kernel_run_context_facker.h"
#include "faker/kernel_outputs_faker.h"
#include "faker/fake_value.h"
#include "register/kernel_registry.h"
#include "kernel/tensor_attr.h"
#include "faker/rt2_var_manager_faker.h"
#include "exe_graph/runtime/tensor_data.h"
#include "framework/runtime/rt_session.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "common/checker.h"
#include "graph/ge_tensor.h"
#include "stub/gert_runtime_stub.h"
#include "checker/memory_profiling_log_matcher.h"
#include "kernel/memory/host_mem_allocator.h"
#include "kernel/ge_local_kernel/file_constant_kernel.h"

namespace gert {
namespace {
KernelRegistry &registry = KernelRegistry::GetInstance();
const std::string kKernelName = "FileConstantKernel";
const size_t kOutputRankSize = 1U;

ge::Status CreateFileConstantFile(const std::string &file_dir, const std::string &file_name, const size_t output_size) {
  if (!file_dir.empty()) {
    GE_ASSERT_TRUE(ge::CreateDirectory(file_dir) == 0);
  }
  std::unique_ptr<int32_t[]> int32_t_buf(new int32_t[output_size / sizeof(int32_t)]);
  for (size_t i = 0U; i < output_size / sizeof(int32_t); ++i) {
    int32_t_buf[i] = i;
  }
  const auto file_constant_file = file_dir + file_name;
  std::ofstream out1(file_constant_file, std::ios::binary);
  GE_ASSERT_TRUE(out1.is_open());
  out1.write((char *)int32_t_buf.get(), output_size);
  out1.close();
  return ge::SUCCESS;
}
}  // namespace

struct FileConstantKernelUt : public testing::Test {
 public:
  struct KernelFakeInputs {
    KernelFakeInputs(RtSession *rt_session, std::string &var_id, std::string &file_constant_weight_dir,
                     std::string &file_name)
        : rt_session_(rt_session),
          var_id_(var_id),
          allocator_(0, RT_MEMORY_TYPE_HOST),
          stream_(reinterpret_cast<void *>(0x11)),
          output_size_(100U),
          file_constant_weight_dir_(file_constant_weight_dir),
          file_name_(file_name) {}
    RtSession* rt_session_;
    std::string var_id_;
    memory::CachingMemAllocator allocator_;
    rtStream_t stream_;
    size_t output_size_;
    std::string file_constant_weight_dir_;
    std::string file_name_;
  };

  static void TestFileConstantKernel(const std::string &attr_file_path, const std::string &location,
                                     const size_t total_execute_cnt, const ge::graphStatus expected_exc_status,
                                     KernelFakeInputs &kernel_fake_inputs) {
    EXPECT_EQ(CreateFileConstantFile(kernel_fake_inputs.file_constant_weight_dir_, kernel_fake_inputs.file_name_,
                                     kernel_fake_inputs.output_size_),
              ge::SUCCESS);

    // fake kernel inputs
    auto var_id_data = static_cast<void *>(const_cast<char *>(kernel_fake_inputs.var_id_.c_str()));
    memory::SingleStreamL2Allocator single_stream_l2_allocator(&(kernel_fake_inputs.allocator_));

    auto file_constant_weight_dir_data =
        static_cast<void *>(const_cast<char *>(kernel_fake_inputs.file_constant_weight_dir_.c_str()));
    auto file_name_data = static_cast<void *>(const_cast<char *>(kernel_fake_inputs.file_name_.c_str()));
    // fake attr param
    int64_t offset = 0;
    int64_t length = 0;
    auto context_holder =
        KernelRunContextFaker()
            .KernelIONum(static_cast<size_t>(kernel::FileConstantKernelInputIdx::kEnd),
                         static_cast<size_t>(kernel::FileConstantKernelOutputIdx::kEnd))
            .Inputs({kernel_fake_inputs.rt_session_, var_id_data, &single_stream_l2_allocator,
                     kernel_fake_inputs.stream_, reinterpret_cast<void *>(kernel_fake_inputs.output_size_),
                     file_constant_weight_dir_data, file_name_data})
            .NodeAttrs({{"file_path", ge::AnyValue::CreateFrom<std::string>(attr_file_path)},
                        {"file_id", ge::AnyValue::CreateFrom<std::string>("")},
                        {"shape", ge::AnyValue::CreateFrom<std::vector<int64_t>>({5, 5})},
                        {"dtype", ge::AnyValue::CreateFrom<int64_t>(ge::DataType::DT_INT32)},
                        {"offset", ge::AnyValue::CreateFrom<int64_t>(offset)},
                        {"length", ge::AnyValue::CreateFrom<int64_t>(length)},
                        {"location", ge::AnyValue::CreateFrom<std::string>(location)}})
            .Build();
    auto context = context_holder.GetContext<KernelContext>();
    auto funcs = registry.FindKernelFuncs(kKernelName);
    ASSERT_NE(funcs, nullptr);
    ASSERT_EQ(funcs->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
    for (size_t execute_cnt = 0U; execute_cnt < total_execute_cnt; ++execute_cnt) {
      auto ret = funcs->run_func(context);
      EXPECT_EQ(ret, expected_exc_status);
      if (ret == ge::GRAPH_SUCCESS) {
        auto td = context->GetOutputPointer<GertTensorData>(static_cast<size_t>(0));
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->GetPlacement(), kOnDeviceHbm);
        ASSERT_TRUE(td->GetSize() >= kernel_fake_inputs.output_size_);
        for (size_t i = 0U; i < kernel_fake_inputs.output_size_ / sizeof(int32_t); ++i) {
           EXPECT_EQ(static_cast<int32_t *>(td->GetAddr())[i], i);
        }
      }
    }
    context_holder.FreeAll();
    if (!kernel_fake_inputs.file_constant_weight_dir_.empty()) {
      ASSERT_TRUE(mmRmdir(kernel_fake_inputs.file_constant_weight_dir_.c_str()) == 0);
    } else {
      system(std::string("rm -rf ").append(kernel_fake_inputs.file_name_).c_str());
    }
  }

  static void TestFileConstantUserMemKernel(const std::string &attr_file_path, const std::string &location,
                                            const size_t total_execute_cnt, const ge::graphStatus expected_exc_status,
                                            KernelFakeInputs &kernel_fake_inputs, void *mem_addr, size_t mem_size) {
    EXPECT_EQ(CreateFileConstantFile(kernel_fake_inputs.file_constant_weight_dir_, kernel_fake_inputs.file_name_,
                                     kernel_fake_inputs.output_size_),
              ge::SUCCESS);

    // fake attr param
    int64_t offset = 0;
    int64_t length = 0;
    auto context_holder =
        KernelRunContextFaker()
            .KernelIONum(static_cast<size_t>(kernel::FileConstantKernelInputIdx::kEnd),
                         static_cast<size_t>(kernel::FileConstantKernelOutputIdx::kEnd))
            .Inputs({mem_addr, static_cast<void *>(&mem_size)})
            .NodeAttrs({{"file_path", ge::AnyValue::CreateFrom<std::string>(attr_file_path)},
                        {"file_id", ge::AnyValue::CreateFrom<std::string>("")},
                        {"shape", ge::AnyValue::CreateFrom<std::vector<int64_t>>({5, 5})},
                        {"dtype", ge::AnyValue::CreateFrom<int64_t>(ge::DataType::DT_INT32)},
                        {"offset", ge::AnyValue::CreateFrom<int64_t>(offset)},
                        {"length", ge::AnyValue::CreateFrom<int64_t>(length)},
                        {"location", ge::AnyValue::CreateFrom<std::string>(location)}})
            .Build();
    auto context = context_holder.GetContext<KernelContext>();
    auto funcs = registry.FindKernelFuncs("FileConstantUserMemKernel");
    ASSERT_NE(funcs, nullptr);
    ASSERT_EQ(funcs->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
    for (size_t execute_cnt = 0U; execute_cnt < total_execute_cnt; ++execute_cnt) {
      auto ret = funcs->run_func(context);
      EXPECT_EQ(ret, expected_exc_status);
      if (ret == ge::GRAPH_SUCCESS) {
        auto td = context->GetOutputPointer<GertTensorData>(static_cast<size_t>(0));
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->GetPlacement(), kOnDeviceHbm);
        ASSERT_TRUE(td->GetSize() >= kernel_fake_inputs.output_size_);
        for (size_t i = 0U; i < kernel_fake_inputs.output_size_ / sizeof(int32_t); ++i) {
           EXPECT_EQ(static_cast<int32_t *>(td->GetAddr())[i], i);
        }
      }
    }
    context_holder.FreeAll();
    if (!kernel_fake_inputs.file_constant_weight_dir_.empty()) {
      ASSERT_TRUE(mmRmdir(kernel_fake_inputs.file_constant_weight_dir_.c_str()) == 0);
    } else {
      system(std::string("rm -rf ").append(kernel_fake_inputs.file_name_).c_str());
    }
  }
};

// 用例描述: 在线场景下, 不走var_mgr, FileConstantKernel功能正常:权重加载成功 + 权重数据正确.
TEST_F(FileConstantKernelUt, FileConstantKernel_WeightLoadSuccessWithoutVarManager) {
  RtSession rt_session;
  string var_id("file_constant_node_0");
  std::string file_constant_weight_dir;
  std::string file_name = "test_copy_one_weight.bin";
  KernelFakeInputs kernel_fake_inputs(&rt_session, var_id, file_constant_weight_dir, file_name);
  // fake attr param
  std::string location;
  std::string attr_file_path = file_name;
  const ge::graphStatus expected_exc_status = ge::GRAPH_SUCCESS;
  TestFileConstantKernel(attr_file_path, location, 1, expected_exc_status, kernel_fake_inputs);
}

// 用例描述: 在线场景下, 走var_mgr, FileConstantKernel功能正常: 变量管理器中获取到输出地址 + 权重数据正确.
TEST_F(FileConstantKernelUt, FileConstantKernel_WeightLoadSuccessWithVarManager) {
  // fake var mgr
  UtestRt2VarManager var_mgr;
  string var_id("file_constant_node_0");
  StorageShape ss{{5,5}, {5,5}};
  std::unique_ptr<int32_t[]> magic_addr(new int32_t[25U]);
  for (size_t i = 0U; i < 25U; ++i) {
    magic_addr[i] = i;
  }
  TensorData tensor_data(magic_addr.get());
  tensor_data.SetPlacement(TensorPlacement::kOnDeviceHbm);
  tensor_data.SetSize(25U * sizeof(int32_t));
  var_mgr.SetVarShapeAndMemory(var_id, ss, tensor_data);
  // fake kernel inputs
  RtSession rt_session;
  rt_session.SetVarManager(&var_mgr);
  std::string file_constant_weight_dir;
  std::string file_name = "test_copy_one_weight.bin";
  KernelFakeInputs kernel_fake_inputs(&rt_session, var_id, file_constant_weight_dir, file_name);
  // fake attr param
  std::string location;
  std::string attr_file_path = file_name;
  const ge::graphStatus expected_exc_status = ge::GRAPH_SUCCESS;
  TestFileConstantKernel(attr_file_path, location, 1, expected_exc_status, kernel_fake_inputs);
}

// 用例描述: 离线场景下, 权重目录和权重文件名(location)拼接成功，权重加载成功
TEST_F(FileConstantKernelUt, FileConstantKernel_TestOfflineLocationOk) {
  RtSession rt_session;
  string var_id("file_constant_node_0");
  std::string file_constant_weight_dir = "./weight/";
  std::string file_name = "test_copy_one_weight.bin";
  KernelFakeInputs kernel_fake_inputs(&rt_session, var_id, file_constant_weight_dir, file_name);

  std::string location = "test_copy_one_weight.bin";
  std::string attr_file_path;
  const ge::graphStatus expected_exc_status = ge::GRAPH_SUCCESS;
  GertRuntimeStub runtime_stub;
  runtime_stub.Clear();
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevel(DLOG_DEBUG);
  TestFileConstantKernel(attr_file_path, location, 1, expected_exc_status, kernel_fake_inputs);
  const std::string expected_log = "Get file constant info from private attr.";
  EXPECT_NE(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, expected_log.c_str()), 0);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kAllocRe, {{2, "0"}}) >= 0);
  runtime_stub.GetSlogStub().SetLevel(DLOG_ERROR);
}

// 用例描述: 离线场景下, 权重目录和权重文件名(location)拼接失败
TEST_F(FileConstantKernelUt, FileConstantKernel_TestOfflineLocationFailed) {
  RtSession rt_session;
  string var_id("file_constant_node_0");
  std::string file_constant_weight_dir = "./weight/";
  std::string file_name = "test_copy_one_weight.bin";
  KernelFakeInputs kernel_fake_inputs(&rt_session, var_id, file_constant_weight_dir, file_name);

  std::string location;
  std::string attr_file_path;
  const ge::graphStatus expected_exc_status = ge::PARAM_INVALID;
  TestFileConstantKernel(attr_file_path, location, 1, expected_exc_status, kernel_fake_inputs);
}

// 用例描述: 离线场景下, 权重目录和权重文件名(location)拼接成功，权重加载执行2次成功
TEST_F(FileConstantKernelUt, FileConstantKernel_ExeFileconstantKernel2Times) {
  RtSession rt_session;
  string var_id("file_constant_node_0");
  std::string file_constant_weight_dir = "./weight/";
  std::string file_name = "test_copy_one_weight.bin";
  KernelFakeInputs kernel_fake_inputs(&rt_session, var_id, file_constant_weight_dir, file_name);

  std::string location = "test_copy_one_weight.bin";
  std::string attr_file_path;
  const ge::graphStatus expected_exc_status = ge::GRAPH_SUCCESS;
  TestFileConstantKernel(attr_file_path, location, 2, expected_exc_status, kernel_fake_inputs);
}

// 用例描述: 用户设置了FileConstant的device地址，直接使用用户设置的device地址
TEST_F(FileConstantKernelUt, FileConstantKernel_UserMem) {
  RtSession rt_session;
  string var_id("file_constant_node_0");
  std::string file_constant_weight_dir = "./weight/";
  std::string file_name = "test_copy_one_weight.bin";
  KernelFakeInputs kernel_fake_inputs(&rt_session, var_id, file_constant_weight_dir, file_name);

  std::string attr_file_path;
  const size_t mem_size = 100U;
  int32_t user_mem[mem_size];
  for (size_t i = 0U; i < mem_size; ++i) {
    user_mem[i] = i;
  }
  const ge::graphStatus expected_exc_status = ge::GRAPH_SUCCESS;
  TestFileConstantUserMemKernel(attr_file_path, file_name, 2, expected_exc_status, kernel_fake_inputs,
                                &user_mem, mem_size * sizeof(int32_t));
}
}  // namespace gert