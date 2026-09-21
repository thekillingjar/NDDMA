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
#include <nlohmann/json.hpp>
#include <iostream>
#include <list>
#define private public
#define protected public
#include "adapter/ffts_adapter/ffts_task_builder_adapter.h"
#include "adapter/adapter_itf/task_builder_adapter.h"
#include "ops_kernel_builder/task_builder/ffts_task_builder.h"
#include "graph/debug/ge_attr_define.h"
#include "common/util/op_info_util.h"
#include "common/fe_inner_error_codes.h"
#include "../fe_test_utils.h"
#include "rt_error_codes.h"
#include "../../../graph_constructor/graph_constructor.h"


using namespace std;
using namespace fe;
using namespace ge;
using FftsTaskBuilderAdapterPtr = shared_ptr<FftsTaskBuilderAdapter>;
using FftsTaskBuilderPtr = shared_ptr<FftsTaskBuilder>;

static void DestroyContext(TaskBuilderContext &context) {
}

class FFTSTaskBuilderAdapterSTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "FFTSTaskBuilderSTest SetUpTestCase" << endl;
  }
  static void TearDownTestCase() {
    cout << "FFTSTaskBuilderSTest TearDownTestCase" << endl;
  }

  virtual void SetUp() {
    context_ = CreateContext();
    NodePtr node_ptr = CreateNode();
    ffts_task_builder_adapter_ptr = std::make_shared<FftsTaskBuilderAdapter>(*node_ptr, context_);
  }

  virtual void TearDown() {
    cout << "a test Tear Down" << endl;
    DestroyContext(context_);
  }

  void GenerateTensorSlice(vector<vector<vector<ffts::DimRange>>> &tensor_slice, const vector<vector<int64_t>> &shape) {
    for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < shape.size(); j++) {
        vector<int64_t> temp = shape[j];
        vector<ffts::DimRange> vdr;
        for (size_t z = 0; z < temp.size() - 1;) {
            ffts::DimRange dr;
            dr.lower = temp[z];
            dr.higher = temp[z + 1];
            vdr.push_back(dr);
            z = z + 2;
        }
        vector<vector<ffts::DimRange>> threadSlice;
        threadSlice.push_back(vdr);
        tensor_slice.push_back(threadSlice);
      }
    }
    return ;
  }

  NodePtr CreateNode()
  {
    FeTestOpDescBuilder builder;
    builder.SetName("test_tvm");
    builder.SetType("test_tvm");

    builder.AddInputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT);
    builder.AddOutputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT);
    NodePtr node =  builder.Finish();
    for (auto anchor : node->GetAllInDataAnchors()) {
      ge::AnchorUtils::SetStatus(anchor, ge::ANCHOR_DATA);
    }
    for (auto anchor : node->GetAllOutDataAnchors()) {
      ge::AnchorUtils::SetStatus(anchor, ge::ANCHOR_DATA);
    }
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
    OpKernelBinPtr tbe_kernel_ptr = std::make_shared<OpKernelBin>(node->GetName(), std::move(buffer));
    node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
    ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", "my_kernel");
    ge::AttrUtils::SetInt(node->GetOpDesc(), "tvm_blockdim", 10);
    ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
    ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_metadata", "my_metadata");

    vector<vector<vector<ffts::DimRange>>> tensor_slice_;
    vector<int64_t> shape_input = {0, 288, 0, 32, 0, 16, 0, 16};
    vector<vector<int64_t>> shape_;
    shape_.push_back(shape_input);
    GenerateTensorSlice(tensor_slice_, shape_);
    ffts::ThreadSliceMap thread_slice_map;
    thread_slice_map.thread_mode = 0;
    thread_slice_map.input_tensor_slice = tensor_slice_;
    thread_slice_map.output_tensor_slice = tensor_slice_;
    thread_slice_map.slice_instance_num = 2;
    ffts::ThreadSliceMapPtr thread_slice_map_ptr = std::make_shared<ffts::ThreadSliceMap>(thread_slice_map);
    node->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, thread_slice_map_ptr);
    return node;
  }
  NodePtr CreateDynAndOptNode()
  {
    FeTestOpDescBuilder builder;
    builder.SetName("test_tvm");
    builder.SetType("test_tvm");
    builder.AddInputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT, 10, "__input0");
    builder.AddInputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT, 10, "__input1");
    builder.AddInputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT, 10, "__input2");
    builder.AddInputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT, 10, "__input30");
    builder.AddInputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT, 10, "__input31");
    builder.AddInputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT, 10, "__input4");
    builder.AddOutputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT, 10, "__output0");
    builder.AddOutputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT, 10, "__output10");
    builder.AddOutputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT, 10, "__output20");
    NodePtr node =  builder.Finish();
    for (auto anchor : node->GetAllInDataAnchors()) {
      ge::AnchorUtils::SetStatus(anchor, ge::ANCHOR_DATA);
    }
    for (auto anchor : node->GetAllOutDataAnchors()) {
      ge::AnchorUtils::SetStatus(anchor, ge::ANCHOR_DATA);
    }
    auto op_desc = node->GetOpDesc();
    op_desc->AppendIrInput("__input0", ge::kIrInputRequired);
    op_desc->AppendIrInput("__input1", ge::kIrInputOptional);
    op_desc->AppendIrInput("__input2", ge::kIrInputRequired);
    op_desc->AppendIrInput("__input3", ge::kIrInputDynamic);
    op_desc->AppendIrInput("__input4", ge::kIrInputRequired);

    op_desc->AppendIrOutput("__output0", ge::kIrOutputRequired);
    op_desc->AppendIrOutput("__output1", ge::kIrOutputDynamic);
    op_desc->AppendIrOutput("__output2", ge::kIrOutputDynamic);

    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
    OpKernelBinPtr tbe_kernel_ptr = std::make_shared<OpKernelBin>(node->GetName(), std::move(buffer));
    node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
    ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", "my_kernel");
    ge::AttrUtils::SetInt(node->GetOpDesc(), "tvm_blockdim", 10);
    ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
    ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_metadata", "my_metadata");

    vector<vector<vector<ffts::DimRange>>> tensor_slice_;
    vector<vector<vector<ffts::DimRange>>> out_tensor_slice;
    vector<int64_t> shape_input = {0, 288, 0, 32, 0, 16, 0, 16};
    vector<vector<int64_t>> shape_;
    shape_.push_back(shape_input);
    shape_.push_back(shape_input);

    GenerateTensorSlice(out_tensor_slice, shape_);

    shape_.push_back(shape_input);
    shape_.push_back(shape_input);
    shape_.push_back(shape_input);
    shape_.push_back(shape_input);
    GenerateTensorSlice(tensor_slice_, shape_);
    ffts::ThreadSliceMap thread_slice_map;
    thread_slice_map.thread_mode = 0;
    thread_slice_map.input_tensor_slice = tensor_slice_;
    thread_slice_map.output_tensor_slice = out_tensor_slice;
    thread_slice_map.slice_instance_num = 2;
    ffts::ThreadSliceMapPtr thread_slice_map_ptr = std::make_shared<ffts::ThreadSliceMap>(thread_slice_map);
    node->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, thread_slice_map_ptr);
    return node;
  }

  NodePtr CreateDynAndOptNode1()
  {
    FeTestOpDescBuilder builder;
    builder.SetName("test_tvm");
    builder.SetType("test_tvm");
    builder.AddInputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT, 10, "__input00");
    builder.AddInputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT, 10, "__input01");
    builder.AddInputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT, 10, "__input1");
    builder.AddInputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT, 10, "__input2");
    builder.AddInputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT, 10, "__input30");
    builder.AddInputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT, 10, "__input4");

    builder.AddOutputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT, 10, "__output00");
    builder.AddOutputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT, 10, "__output1");
    builder.AddOutputDesc({288,32,16,16}, ge::FORMAT_NCHW, ge::DT_FLOAT, 10, "__output20");
    NodePtr node =  builder.Finish();
    for (auto anchor : node->GetAllInDataAnchors()) {
      ge::AnchorUtils::SetStatus(anchor, ge::ANCHOR_DATA);
    }
    for (auto anchor : node->GetAllOutDataAnchors()) {
      ge::AnchorUtils::SetStatus(anchor, ge::ANCHOR_DATA);
    }
    auto op_desc = node->GetOpDesc();
    op_desc->AppendIrInput("__input0", ge::kIrInputDynamic);
    op_desc->AppendIrInput("__input1", ge::kIrInputOptional);
    op_desc->AppendIrInput("__input2", ge::kIrInputRequired);
    op_desc->AppendIrInput("__input3", ge::kIrInputDynamic);
    op_desc->AppendIrInput("__input4", ge::kIrInputRequired);
    op_desc->AppendIrInput("__input5", ge::kIrInputOptional);
    op_desc->AppendIrInput("__input6", ge::kIrInputOptional);

    op_desc->AppendIrOutput("__output0", ge::kIrOutputDynamic);
    op_desc->AppendIrOutput("__output1", ge::kIrOutputRequired);
    op_desc->AppendIrOutput("__output2", ge::kIrOutputDynamic);

    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
    OpKernelBinPtr tbe_kernel_ptr = std::make_shared<OpKernelBin>(node->GetName(), std::move(buffer));
    node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
    ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", "my_kernel");
    ge::AttrUtils::SetInt(node->GetOpDesc(), "tvm_blockdim", 10);
    ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
    ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_metadata", "my_metadata");

    vector<vector<vector<ffts::DimRange>>> tensor_slice_;
    vector<vector<vector<ffts::DimRange>>> out_tensor_slice;
    vector<int64_t> shape_input = {0, 288, 0, 32, 0, 16, 0, 16};
    vector<vector<int64_t>> shape_;
    shape_.push_back(shape_input);
    shape_.push_back(shape_input);

    GenerateTensorSlice(out_tensor_slice, shape_);

    shape_.push_back(shape_input);
    shape_.push_back(shape_input);
    shape_.push_back(shape_input);
    shape_.push_back(shape_input);
    GenerateTensorSlice(tensor_slice_, shape_);
    ffts::ThreadSliceMap thread_slice_map;
    thread_slice_map.thread_mode = 0;
    thread_slice_map.input_tensor_slice = tensor_slice_;
    thread_slice_map.output_tensor_slice = out_tensor_slice;
    thread_slice_map.slice_instance_num = 2;
    ffts::ThreadSliceMapPtr thread_slice_map_ptr = std::make_shared<ffts::ThreadSliceMap>(thread_slice_map);
    node->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, thread_slice_map_ptr);
    return node;
  }
  static TaskBuilderContext CreateContext()
  {
    TaskBuilderContext context;
    context.dataMemSize = 100;
    context.dataMemBase = (uint8_t *) (intptr_t) 1000;
    context.weightMemSize = 200;
    context.weightMemBase = (uint8_t *) (intptr_t) 1100;
    context.weightBufferHost = Buffer(20);

    return context;
  }

 public:
  FftsTaskBuilderAdapterPtr ffts_task_builder_adapter_ptr;
  TaskBuilderContext context_;
};

TEST_F(FFTSTaskBuilderAdapterSTest, Init_Workspace_SUCCESS) {
  vector<int64_t> work_space_bytes = {1, 2};
  vector<int64_t> work_space = {1, 2};
  int64_t non_tail_workspace_size = 1;
  (void)ge::AttrUtils::SetInt(ffts_task_builder_adapter_ptr->op_desc_, fe::NON_TAIL_WORKSPACE_SIZE,
  non_tail_workspace_size);
  ffts_task_builder_adapter_ptr->op_desc_->SetWorkspaceBytes(work_space_bytes);
  ffts_task_builder_adapter_ptr->op_desc_->SetWorkspace(work_space);
  Status ret = ffts_task_builder_adapter_ptr->InitWorkspace();
  EXPECT_EQ(fe::SUCCESS, ret);
  domi::TaskDef task_def = {};
  ret = ffts_task_builder_adapter_ptr->Run(task_def);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(FFTSTaskBuilderAdapterSTest, Get_handle_anchor_data_SUCCESS) {
  Status ret = ffts_task_builder_adapter_ptr->Init();
  ge::InDataAnchorPtr anchor;
  size_t input_index = 0;
  size_t anchor_index = 0;
  ret = ffts_task_builder_adapter_ptr->HandleAnchorData(input_index, anchor_index);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(FFTSTaskBuilderAdapterSTest, Init_workspace_failed) {
  ffts_task_builder_adapter_ptr->workspace_addrs_.clear();
  ffts_task_builder_adapter_ptr->workspace_sizes_.clear();
  int64_t invalid_workspace_size = 10;
  (void)ge::AttrUtils::SetInt(ffts_task_builder_adapter_ptr->op_desc_, fe::NON_TAIL_WORKSPACE_SIZE,
      invalid_workspace_size);
  auto ret = ffts_task_builder_adapter_ptr->Init();
  EXPECT_NE(fe::SUCCESS, ret);
}

TEST_F(FFTSTaskBuilderAdapterSTest, gen_mix_l2_taskdef) {
  auto node = CreateNode();
  vector<void *> input_addrs(2, nullptr);
  vector<void *> output_addrs(1, nullptr);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR,
                                              0xFFFFFFFFU);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 1);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_OUTPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0);
  FftsTaskBuilderPtr ffts_task_builder = std::make_shared<FftsTaskBuilder>();
  ffts_task_builder->manual_thread_param_.input_addrs = input_addrs;
  ffts_task_builder->manual_thread_param_.output_addrs = output_addrs;
  ffts_task_builder->manual_thread_param_.kernel_args_info = ffts_task_builder_adapter_ptr->kernel_args_info_;

  FftsPlusCtxDefPtr ctx = std::make_shared<domi::FftsPlusCtxDef>();
  node->GetOpDesc()->AppendIrInput("__input0", ge::kIrInputRequired);
  std::vector<uint32_t> input_type_list = {0};
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kInputParaTypeList, input_type_list);
  node->GetOpDesc()->AppendIrOutput("__output0", ge::kIrOutputRequired);
  std::vector<uint32_t> output_type_list = {0};
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kOutputParaTypeList, output_type_list);
  EXPECT_EQ(ffts_task_builder->GenMixL2CtxDef(node->GetOpDesc(), ctx), fe::SUCCESS);
}

TEST_F(FFTSTaskBuilderAdapterSTest, gen_dyn_and_opt_mix_l2_taskdef) {
  auto node = CreateDynAndOptNode();
  vector<void *> input_addrs(6, nullptr);
  vector<void *> output_addrs(3, nullptr);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0xFFFFFFFFU);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 1);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 2);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 3);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 4);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 5);

  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_OUTPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_OUTPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 1);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_OUTPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 2);
  FftsTaskBuilderPtr ffts_task_builder = std::make_shared<FftsTaskBuilder>();
  ffts_task_builder->manual_thread_param_.input_addrs = input_addrs;
  ffts_task_builder->manual_thread_param_.output_addrs = output_addrs;
  ffts_task_builder->manual_thread_param_.kernel_args_info = ffts_task_builder_adapter_ptr->kernel_args_info_;

  (void) ge::AttrUtils::SetInt(node->GetOpDesc(), kModeInArgsFirstField, 1);
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), fe::kAttrDynamicParamMode, fe::kFoldedWithDesc);
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), fe::kAttrOptionalInputMode, fe::kGenPlaceholder);
  auto op_desc = node->GetOpDesc();

  std::vector<uint32_t> input_type_list = {0,1,0,2,2,0};
  std::vector<uint32_t> output_type_list = {0,2,2};
  std::vector<std::string> input_name_list = {"__input0", "__input1", "__input2", "__input3", "__input3", "__input4"};
  (void)ge::AttrUtils::SetListStr(op_desc, kInputNameList, input_name_list);
  (void)ge::AttrUtils::SetListInt(op_desc, kInputParaTypeList, input_type_list);
  std::vector<std::string> output_name_list = {"__output0", "__output1", "__output2"};
  (void)ge::AttrUtils::SetListStr(op_desc, kOutputNameList, output_name_list);
  (void)ge::AttrUtils::SetListInt(op_desc, kOutputParaTypeList, output_type_list);
  op_desc->SetWorkspaceBytes(vector<int64_t>{32});
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, "test");
  FftsPlusCtxDefPtr ctx = std::make_shared<domi::FftsPlusCtxDef>();
  auto ret = ffts_task_builder->GenMixL2CtxDef(op_desc, ctx);
  EXPECT_EQ(ret, fe::SUCCESS);
  std::vector<std::vector<int64_t>> dyn_io_vv;
  std::vector<std::vector<int64_t>> ep_dyn_io_vv = {{3,4}};
  (void)ge::AttrUtils::GetListListInt(node->GetOpDesc(), kDyInputsIndexes, dyn_io_vv);
  EXPECT_NE(dyn_io_vv, ep_dyn_io_vv);
  ep_dyn_io_vv = {{1},{2}};
  (void)ge::AttrUtils::GetListListInt(node->GetOpDesc(), kDyOutputsIndexes, dyn_io_vv);
  EXPECT_NE(dyn_io_vv, ep_dyn_io_vv);
}


TEST_F(FFTSTaskBuilderAdapterSTest, gen_dyn_and_opt_mix_l2_taskdef1) {
  auto node = CreateDynAndOptNode1();
  vector<void *> input_addrs(6, nullptr);
  vector<void *> output_addrs(3, nullptr);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 1);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 2);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 3);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 4);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 5);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_OUTPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_OUTPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 1);
  ffts_task_builder_adapter_ptr->FeedArgsInfo(domi::ArgsInfo_ArgsType_OUTPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 2);
  FftsTaskBuilderPtr ffts_task_builder = std::make_shared<FftsTaskBuilder>();
  ffts_task_builder->manual_thread_param_.input_addrs = input_addrs;
  ffts_task_builder->manual_thread_param_.output_addrs = output_addrs;
  ffts_task_builder->manual_thread_param_.kernel_args_info = ffts_task_builder_adapter_ptr->kernel_args_info_;

  auto op_desc = node->GetOpDesc();
  (void) ge::AttrUtils::SetInt(op_desc, kModeInArgsFirstField, 1);
  (void)ge::AttrUtils::SetStr(op_desc, fe::kAttrDynamicParamMode, fe::kFoldedWithDesc);
  (void)ge::AttrUtils::SetStr(op_desc, fe::kAttrOptionalInputMode, fe::kGenPlaceholder);


  (void)ge::AttrUtils::SetInt(op_desc, kOpKernelAllInputSize, 7);

  std::vector<std::string> input_name_list = {"__input0", "__input0", "__input1", "__input2", "__input3", "__input4"};
  (void)ge::AttrUtils::SetListStr(op_desc, kInputNameList, input_name_list);
  std::vector<std::string> output_name_list = {"__output0", "__output1", "__output2"};
  (void)ge::AttrUtils::SetListStr(op_desc, kOutputNameList, output_name_list);

  std::vector<uint32_t> input_type_list = {2,2,0,1,2,1};
  std::vector<uint32_t> output_type_list = {2,0,2};
  (void)ge::AttrUtils::SetListInt(op_desc, kInputParaTypeList, input_type_list);
  (void)ge::AttrUtils::SetListInt(op_desc, kOutputParaTypeList, output_type_list);
  (void)ge::AttrUtils::SetBool(op_desc, "_unknown_shape", true);
  FftsPlusCtxDefPtr ctx = std::make_shared<domi::FftsPlusCtxDef>();
  auto ret = ffts_task_builder->GenMixL2CtxDef(node->GetOpDesc(), ctx);
  EXPECT_EQ(ret, fe::SUCCESS);
  std::vector<std::vector<int64_t>> dyn_io_vv;
  std::vector<std::vector<int64_t>> ep_dyn_io_vv = {{0,1},{4}};
  (void)ge::AttrUtils::GetListListInt(op_desc, kDyInputsIndexes, dyn_io_vv);
  EXPECT_NE(dyn_io_vv, ep_dyn_io_vv);
  ep_dyn_io_vv = {{0},{2}};
  (void)ge::AttrUtils::GetListListInt(op_desc, kDyOutputsIndexes, dyn_io_vv);
  EXPECT_NE(dyn_io_vv, ep_dyn_io_vv);
  domi::FftsPlusMixAicAivCtxDef *mix_aic_aiv_ctx_def = ctx->mutable_mix_aic_aiv_ctx();
  auto args_str = mix_aic_aiv_ctx_def->args_format();
  EXPECT_STRNE(args_str.c_str(), "{i_desc0}{i1*}{i2*}{i_desc3}{i4*}{i5*}{i6*}{o_desc0}{o1*}{o_desc2}{ws*}{t_ffts.tail}");
}
