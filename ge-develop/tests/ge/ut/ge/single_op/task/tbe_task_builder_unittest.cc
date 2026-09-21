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
#include <vector>

#include "runtime/rt.h"

#include "macro_utils/dt_public_scope.h"
#include "hybrid/model/graph_item.h"
#include "hybrid/model/node_item.h"
#include "single_op/single_op.h"
#include "single_op/single_op_manager.h"
#include "single_op/task/build_task_utils.h"
#include "single_op/task/build_task_utils.h"
#include "single_op/task/tbe_task_builder.h"
#include "common/dump/dump_manager.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "register/op_impl_registry.h"
#include "faker/space_registry_faker.h"
#include "macro_utils/dt_public_unscope.h"
#include "common/global_variables/diagnose_switch.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "common/opskernel/ops_kernel_info_types.h"

using namespace std;
using namespace ge;
using namespace hybrid;

class UtestTbeTaskBuilder : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
    TBEHandleStore::GetInstance().bin_key_to_handle_.clear();
  }
  ObjectPool<GeTensor> tensor_pool_;
};

TEST_F(UtestTbeTaskBuilder, test_KernelHolder_construct) {
  std::shared_ptr<ge::OpKernelBin> kernel_bin = std::make_shared<ge::OpKernelBin>("bin_name", std::vector<char>());
  void *stub_func = ValueToPtr(1234U);
  KernelHolder holder((const char_t*)stub_func, kernel_bin);
  holder.bin_handle_ = malloc(1);
  EXPECT_EQ(holder.stub_func_, stub_func);
  free((void*)holder.bin_handle_);
}

TEST_F(UtestTbeTaskBuilder, test_KernelBinRegistry_GetUnique) {
  std::string test = "test";
  const char *out1 = KernelBinRegistry::GetInstance().GetUnique(test.c_str());
  const char *out2 = KernelBinRegistry::GetInstance().GetUnique(test.c_str());
  EXPECT_TRUE(out1 == out2);
}

TEST_F(UtestTbeTaskBuilder, test_KernelBinRegistry_GetStubFunc) {
  EXPECT_EQ(KernelBinRegistry::GetInstance().GetStubFunc("test"), nullptr);
  std::shared_ptr<ge::OpKernelBin> kernel_bin = std::make_shared<ge::OpKernelBin>("bin_name", std::vector<char>());
  void *stub_func = ValueToPtr(1234U);
  KernelBinRegistry::GetInstance().AddKernel("test",
      std::unique_ptr<KernelHolder>(new KernelHolder((const char_t*)stub_func, kernel_bin)));
  EXPECT_EQ(KernelBinRegistry::GetInstance().GetStubFunc("test"), stub_func);
}

TEST_F(UtestTbeTaskBuilder, test_TbeTaskBuilder) {
  std::string model_name = "test_model";

  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);

  domi::TaskDef task_def;
  task_def.mutable_kernel()->set_args_size(sizeof(void *));
  TbeTaskBuilder builder(model_name, node, task_def);

  (void)AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", "test_kernel_name");
  (void)AttrUtils::SetStr(op_desc, op_desc->GetName() + "_atomic_kernelname", "test_kernel_name");
  std::string out;
  builder.GetKernelName(op_desc, out);
  EXPECT_EQ(out, "test_kernel_name");

  AttrUtils::SetStr(op_desc, builder.GetKeyForTvmMetaData(), "meta_data_test");

  EXPECT_EQ(builder.DoRegisterMeta(reinterpret_cast<void *>(98)), SUCCESS);
  EXPECT_EQ(builder.DoRegisterMeta(reinterpret_cast<void *>(99)), FAILED);

  std::string test = "test";
  EXPECT_EQ(builder.DoRegisterFunction(reinterpret_cast<void *>(98), test.c_str(), test.c_str()), SUCCESS);
  EXPECT_NE(builder.DoRegisterFunction(reinterpret_cast<void *>(99), test.c_str(), test.c_str()), SUCCESS);

  std::shared_ptr<ge::OpKernelBin> kernel_bin = std::make_shared<ge::OpKernelBin>("bin_name", std::vector<char>());
  void *bin_handle = nullptr;
  SingleOpModelParam param;
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  EXPECT_EQ(builder.DoRegisterKernel(*kernel_bin, test.c_str(), &bin_handle, param), SUCCESS);

  TbeOpTask task;
  builder.stub_name_ = "test1";
  EXPECT_EQ(builder.DoRegisterFunction(reinterpret_cast<void *>(98), test.c_str(), test.c_str()), SUCCESS);
  op_desc->SetExtAttr(EXT_ATTR_ATOMIC_TBE_KERNEL, kernel_bin);
  op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_bin);
  EXPECT_EQ(builder.RegisterKernel(task, param), SUCCESS);

  EXPECT_EQ(builder.RegisterKernelWithHandle(param), SUCCESS);

  EXPECT_EQ(builder.GetKeyForOpParamSize(), "op_para_size");
  EXPECT_EQ(builder.GetKeyForTvmMetaData(), "tvm_metadata");
  EXPECT_EQ(builder.InitKernelArgs(reinterpret_cast<void *>(98), 10, param), SUCCESS);

  uint32_t magic;
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  EXPECT_EQ(builder.GetMagic(magic), SUCCESS);
  EXPECT_EQ(magic, RT_DEV_BINARY_MAGIC_ELF_AIVEC);

  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AICUBE");
  EXPECT_EQ(builder.GetMagic(magic), SUCCESS);
  EXPECT_EQ(magic, RT_DEV_BINARY_MAGIC_ELF_AICUBE);

  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "invalid");
  EXPECT_EQ(builder.GetMagic(magic), PARAM_INVALID);

  (void)AttrUtils::SetInt(op_desc, builder.GetKeyForOpParamSize(), -1);
  EXPECT_EQ(builder.InitTilingInfo(task), ACL_ERROR_GE_PARAM_INVALID);

  EXPECT_NE(builder.BuildTask(task, param), SUCCESS);  // ??

  task.need_tiling_ = true;
  task.args_ = std::unique_ptr<uint8_t[]>(new uint8_t[24]());
  task.arg_size_ = 24;
  task.max_tiling_size_ = 0;
  param.graph_is_dynamic = true;
  int64_t buffer = 4;
  task.overflow_addr_ =  &buffer;
  task.has_overflow_attr_ = true;

  EXPECT_NE(builder.SetKernelArgs(task, param, op_desc), SUCCESS);  // ??

  (void)AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", "ZZZZ");     // ??
  EXPECT_EQ(builder.RegisterKernelWithHandle(param), ACL_ERROR_GE_INTERNAL_ERROR);
}

UINT32 StubTilingSingleopMix(gert::TilingContext *context) {
  context->SetNeedAtomic(false);
  context->SetTilingKey(666U);
  context->SetBlockDim(666U);
  auto tiling_data = context->GetTilingData<uint64_t>();
  *tiling_data = 100;
  return ge::GRAPH_SUCCESS;
}

UINT32 StubTilingParseSingleopMix(gert::KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

void* CompileInfoCreatorSingleopMix() {
  auto tmp =  ge::MakeUnique<char>();
  return tmp.get();
}

void TestMixL2TaskBuilder(std::function<void(MixL2OpTask &)> extend_check_func = nullptr, bool dynamic_kernel = false) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto funcs = space_registry->CreateOrGetOpImpl(MATMUL);
  ASSERT_NE(funcs, nullptr);
  funcs->tiling = StubTilingSingleopMix;
  funcs->tiling_parse = StubTilingParseSingleopMix;
  funcs->compile_info_creator = CompileInfoCreatorSingleopMix;
  funcs->compile_info_deleter = nullptr;

  std::string model_name = "test_model";

  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("MATMUL", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  AttrUtils::SetBool(op_desc, ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);

  ge::NodePtr node = graph->AddNode(op_desc);

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(op_desc->GetId());
  ffts_plus_task_def->set_addr_size(20);
  domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_MIX_AIC);
  std::vector<std::string> name_prefix = {"_mix_aic"};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefix);

  (void)AttrUtils::SetBool(op_desc, "support_dynamicshape", false);
  (void)AttrUtils::SetInt(op_desc, "op_para_size", 18 * sizeof(uintptr_t));

  // set tbe kernel
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  std::vector<char> conv_bin(64, '\0');
  TBEKernelPtr conv_kernel = MakeShared<ge::OpKernelBin>("MATMUL", std::move(conv_bin));
  op_desc->SetExtAttr(std::string("_mix_aic") + OP_EXTATTR_NAME_TBE_KERNEL, conv_kernel);
  AttrUtils::SetStr(op_desc, "_mix_aic" + op_desc->GetName() + "_kernelname", "MATMUL");

  (void) AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, std::string("MIX_AIC"));
  AttrUtils::SetStr(op_desc, std::string("_mix_aic") + ATTR_NAME_TBE_KERNEL_NAME, "MATMUL");

  MixL2TaskBuilder builder(model_name, node, task_def);
  SingleOpModelParam param;
  param.space_registries_ = ge::MakeShared<gert::OpImplSpaceRegistryV2Array>();
  param.space_registries_->at(static_cast<size_t>(OppImplVersion::kOpp)) = space_registry;
  MixL2OpTask op_task(node);
  op_task.space_registries_ = ge::MakeShared<gert::OpImplSpaceRegistryV2Array>();
  op_task.space_registries_->at(static_cast<size_t>(OppImplVersion::kOpp)) = space_registry;
  auto stream_resource = MakeUnique<StreamResource>(0);
  stream_resource->allocator_ = &stream_resource->internal_allocator_;
  op_task.stream_resource_ = stream_resource.get();
  if (dynamic_kernel) {
    AttrUtils::SetBool(op_desc, "_mix_enhanced_kernel_list_first_name", true);
  }

  EXPECT_EQ(builder.BuildMixL2Task(op_task, param), SUCCESS);

  // build dynamic op
  MixL2OpTask op_task2(node);
  op_task2.space_registries_ = ge::MakeShared<gert::OpImplSpaceRegistryV2Array>();
  op_task2.space_registries_->at(static_cast<size_t>(OppImplVersion::kOpp)) = space_registry;
  op_task2.stream_resource_ = stream_resource.get();
  (void)AttrUtils::SetBool(op_desc, "support_dynamicshape", true);
  EXPECT_EQ(builder.BuildMixL2Task(op_task2, param), SUCCESS);
  if (extend_check_func != nullptr) {
    extend_check_func(op_task2);
    ge::diagnoseSwitch::EnableProfiling({gert::ProfilingType::kDevice});
    op_task2.ReportProfilingData(0);
    ge::diagnoseSwitch::DisableProfiling();
  }
}

TEST_F(UtestTbeTaskBuilder, test_Mixl2TaskBuilder) {
  TestMixL2TaskBuilder();
}

TEST_F(UtestTbeTaskBuilder, test_Mixl2TaskBuilder_dynamic) {
  TestMixL2TaskBuilder(nullptr, true);
}

TEST_F(UtestTbeTaskBuilder, Mixl2Task_GetValidTilingKeyAndTilingData_WithExceptionDumpOn) {
  auto check_func = [](MixL2OpTask &op_task) -> void {
    uint32_t tiling_key;
    std::string tiling_data;
    op_task.GetTilingKeyAndData(tiling_key, tiling_data);
    EXPECT_EQ(tiling_key, 666);
    std::string tiling_data_head = tiling_data.substr(0, 40);
    EXPECT_EQ(tiling_data_head, "0x64 0x00 0x00 0x00 0x00 0x00 0x00 0x00 ");
    uintptr_t args = 0;
    size_t args_size = 0;
    op_task.GetHostArgsAndSize(args, args_size);
    EXPECT_NE(args, 0);
    EXPECT_NE(args_size, 0);
    EXPECT_EQ(op_task.GetTaskType(), kTaskTypeMixAic);
  };
  TestMixL2TaskBuilder(check_func);
}

TEST_F(UtestTbeTaskBuilder, MemcpyAsyncTask_GetHostArgsAndSize) {
  MemcpyAsyncTask task{};
  uintptr_t args = 0;
  size_t args_size = 0;
  EXPECT_NO_THROW(task.GetHostArgsAndSize(args, args_size));
}

TEST_F(UtestTbeTaskBuilder, test_updatetilingargs) {
  std::string model_name = "test_model";

  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);

  TbeOpTask task;
  domi::TaskDef task_def;
  task_def.mutable_kernel()->set_args_size(sizeof(void *));
  TbeTaskBuilder builder(model_name, node, task_def);
  task.need_tiling_ = true;
  size_t kernel_def_size = 256U;
  size_t index1 = 1;
  size_t index2 = 2;
  EXPECT_EQ(builder.UpdateTilingArgs(task, index1, kernel_def_size), SUCCESS);
  EXPECT_EQ(builder.UpdateTilingArgs(task, index2, kernel_def_size), SUCCESS);
}

TEST_F(UtestTbeTaskBuilder, test_AtomicAddrCleanTaskBuilder) {
  std::string model_name = "test_model";

  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);

  domi::TaskDef task_def;
  task_def.mutable_kernel()->set_args_size(sizeof(void *));
  AtomicAddrCleanTaskBuilder builder(model_name, node, task_def);

  SingleOpModelParam param;
  EXPECT_EQ(builder.InitKernelArgs(nullptr, 10, param), SUCCESS);
  EXPECT_EQ(builder.GetKeyForOpParamSize(), "atomic_op_para_size");
  EXPECT_EQ(builder.GetKeyForTvmMetaData(), "_atomic_tvm_metadata");

  EXPECT_EQ(builder.GetKeyForTvmMetaData(), "_atomic_tvm_metadata");

  (void)AttrUtils::SetStr(op_desc, op_desc->GetName() + "_atomic_kernelname", "test_kernel_name");
  std::string out;
  builder.GetKernelName(op_desc, out);
  EXPECT_EQ(out, "test_kernel_name");

  std::shared_ptr<ge::OpKernelBin> kernel_bin = std::make_shared<ge::OpKernelBin>("bin_name", std::vector<char>());
  op_desc->SetExtAttr(EXT_ATTR_ATOMIC_TBE_KERNEL, kernel_bin);
  EXPECT_EQ(builder.GetTbeKernel(op_desc)->GetName(), kernel_bin->GetName());
}

TEST_F(UtestTbeTaskBuilder, test_TbeTaskBuilderWithInputAddress) {
  std::string model_name = "test_model";

  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  vector<int64_t> input_offset = {1024};
  op_desc->SetInputOffset(input_offset);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);

  vector<bool> is_input_const = {true};
  op_desc->SetIsInputConst(is_input_const);
  auto tensor_desc = op_desc->MutableInputDesc(0);
  EXPECT_NE(tensor_desc, nullptr);
  TensorUtils::SetSize(*tensor_desc, 64);
  tensor_desc->SetShape(GeShape({4}));
  tensor_desc->SetOriginShape(GeShape({4}));
  TensorUtils::SetDataOffset(*tensor_desc, 0);

  domi::TaskDef task_def;
  task_def.mutable_kernel()->set_args_size(sizeof(void *));
  TbeTaskBuilder builder(model_name, node, task_def);

  SingleOpModelParam param;
  uint8_t weight_base_addr = 0;
  param.runtime_param.session_id = 0;
  param.runtime_param.weight_base = reinterpret_cast<uintptr_t>(&weight_base_addr);
  param.runtime_param.weight_size = 64;
  void *dst = (void *)malloc(10);
  EXPECT_EQ(builder.InitKernelArgs(dst, sizeof(void *) * 10, param), SUCCESS);
  free(dst);
}
