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

#include <vector>
#include "runtime/rt_model.h"
#include "rt_error_codes.h"
#include "graph/ge_tensor.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "../../fe_test_utils.h"
#include "securec.h"
#include "graph/debug/ge_attr_define.h"
#define protected public
#define private public
#include "common/fe_log.h"
#include "common/resource_def.h"
#include "common/l2_stream_info.h"
#include "common/constants_define.h"
#include "common/fe_inner_attr_define.h"
#include "common/comm_error_codes.h"
#include "common/fe_inner_error_codes.h"
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "common/graph_comm.h"
#include "common/large_bitmap.h"
#include "common/util/op_info_util.h"
#include "adapter/tbe_adapter/tbe_task_builder_adapter.h"
#include "adapter/adapter_itf/task_builder_adapter.h"
#include "adapter/tbe_adapter/kernel_launch/tbe_kernel_launch.h"
#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_optimizer.h"
#include "common/util/constants.h"
#undef private
#undef protected

using namespace std;
using namespace testing;
using namespace ge;
using namespace fe;

using NodePtr = std::shared_ptr<Node>;
#define SET_SIZE 128
#define SET_SIZE_2 100352

class UTEST_TbeTaskBuilderAdapter : public testing::Test
{
    friend class TbeTaskBuilderAdapter;

protected:

    static void CheckSizeFuction(ge::GeTensorDesc& tensor)
    {
      vector<int64_t> dims = {1,4,5,2,7};
      ge::GeShape input1_shape(dims);
      tensor.SetShape(input1_shape);
      tensor.SetDataType(ge::DT_DOUBLE);
      tensor.SetFormat(ge::FORMAT_NC1HWC0_C04);
    }

    static void SetOpDecSize(NodePtr& node, const int set_size){
        OpDesc::Vistor<GeTensorDesc> tensors = node->GetOpDesc()->GetAllInputsDesc();
        for (int i = 0; i < tensors.size(); i++){
            ge::GeTensorDesc tensor = tensors.at(i);
            ge::TensorUtils::SetSize(tensor, set_size);
            node->GetOpDesc()->UpdateInputDesc(i, tensor);
        }
        OpDesc::Vistor<GeTensorDesc> tensors_output = node->GetOpDesc()->GetAllOutputsDesc();
        for (int i = 0; i < tensors_output.size(); i++){
            ge::GeTensorDesc tensor_output = tensors_output.at(i);
            ge::TensorUtils::SetSize(tensor_output, set_size);
            node->GetOpDesc()->UpdateOutputDesc(i, tensor_output);
        }
    }
    void SetUp()
    {
        rtContext_t rtContext;
        assert(rtCtxCreate(&rtContext, RT_CTX_GEN_MODE, 0) == ACL_RT_SUCCESS);
        assert(rtCtxSetCurrent(rtContext) == ACL_RT_SUCCESS);

        node_ = CreateNode();
        context_ = CreateContext();
        SetOpDecSize(node_, SET_SIZE);
        adapter_ = shared_ptr<TbeTaskBuilderAdapter> (new (nothrow) TbeTaskBuilderAdapter(*node_, context_));
    }
    void TearDown()
    {
        adapter_.reset();
        DestroyContext(context_);
        node_.reset();

        rtContext_t rtContext;
        assert(rtCtxGetCurrent(&rtContext) == ACL_RT_SUCCESS);
        assert(rtCtxDestroy(rtContext) == ACL_RT_SUCCESS);
    }
  static void CheckGraphReadMode(GeTensorDescPtr tensor_desc, L2CacheReadMode expect) {
    int32_t read_mode;
    bool get_status = ge::AttrUtils::GetInt(tensor_desc, "_fe_l2cache_graph_read_mode", read_mode);
    EXPECT_EQ(get_status, true);
    EXPECT_EQ(read_mode, static_cast<int32_t>(expect));
  }

    void InitGraph1(ComputeGraphPtr& graph) {
      const std::string CONV2D = "Conv2D";
      const std::string CONCATD = "ConcatD";
      const std::string CONCAT_DIM = "concat_dim";
      OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", CONV2D);
      OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", CONV2D);
      OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
      (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 1);

      // add descriptor
      ge::GeShape shape1({1, -1, 14, 14});
      ge::GeShape shape2({1, -1, 14, 14});
      GeTensorDesc out_desc1(shape1, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
      out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
      out_desc1.SetOriginDataType(ge::DT_FLOAT);
      out_desc1.SetOriginShape(shape1);
      GeTensorDesc out_desc2(shape2, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
      out_desc2.SetOriginFormat(ge::FORMAT_NCHW);
      out_desc2.SetOriginDataType(ge::DT_FLOAT);
      out_desc2.SetOriginShape(shape2);
      conv1->AddOutputDesc(out_desc1);
      conv2->AddOutputDesc(out_desc2);
      concat->AddInputDesc("__input00", out_desc1);
      concat->AddInputDesc("__input10", out_desc2);
      std::vector<int64_t> shape_vec;
      ge::GeShape null_shape(shape_vec);
      GeTensorDesc out_desc_null(null_shape, ge::FORMAT_RESERVED, ge::DT_UNDEFINED);
      concat->AddInputDesc("__input2", out_desc_null);
      // create nodes
      NodePtr conv1_node = graph->AddNode(conv1);
      NodePtr conv2_node = graph->AddNode(conv2);
      NodePtr concat_node = graph->AddNode(concat);
      ge::NodeUtils::AppendInputAnchor(concat_node, 4);
      concat->AddInputDesc("__input3", out_desc2);
      /*
      *  Conv2d     Conv2d
      *      \       /
      *        Concat(concat_dim=0)
      *          |
      */
      ge::GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                              concat_node->GetInDataAnchor(0));
      ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                              concat_node->GetInDataAnchor(1));
      ge::GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0),
                              concat_node->GetInDataAnchor(3));
    }

    static NodePtr CreateNodeWithoutAttrs(bool has_weight = false)
    {
        FeTestOpDescBuilder builder;
        builder.SetName("test_tvm");
        builder.SetType("test_tvm");
        builder.SetInputs({1});
        builder.SetOutputs({1});
        builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
        builder.AddOutputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
        if (has_weight) {
            size_t len = 10;
            unique_ptr<float[]> buf(new float[len]);
            auto weight = builder.AddWeight((uint8_t*) buf.get(), len * sizeof(float), { 1, 1, 2, 5 },
                    ge::FORMAT_NCHW, ge::DT_FLOAT);
            ge::TensorUtils::SetWeightSize(weight->MutableTensorDesc(), len * sizeof(float));
        }

        return builder.Finish();
    }

    static NodePtr CreateNode(bool has_weight = false)
    {
        NodePtr node = CreateNodeWithoutAttrs(has_weight);

        const char tbe_bin[] = "tbe_bin";
        vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
        OpKernelBinPtr tbe_kernel_ptr = std::make_shared<OpKernelBin>(node->GetName(), std::move(buffer));
        node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

        ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", "my_kernel");
        ge::AttrUtils::SetInt(node->GetOpDesc(), "tvm_blockdim", 10);
        ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
        ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_metadata", "my_metadata");

        SetOpDecSize(node, SET_SIZE);
        return node;
    }

    static NodePtr CreateNodeWithoutAttrs3(bool has_weight = false)
    {
      FeTestOpDescBuilder builder;
      builder.SetName("test_tvm");
      builder.SetType("test_tvm");
      builder.SetInputs({1});
      builder.SetOutputs({1});
      builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
      builder.AddOutputDesc({1, 8, 7, 56, 32}, ge::FORMAT_NC1HWC0, ge::DT_INT8);
      if (has_weight) {
        size_t len = 10;
        unique_ptr<float[]> buf(new float[len]);
        auto weight = builder.AddWeight((uint8_t*) buf.get(), len * sizeof(float), { 1, 1, 2, 5 },
                                        ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorUtils::SetWeightSize(weight->MutableTensorDesc(), len * sizeof(float));
      }

      return builder.Finish();
    }

    static NodePtr CreateNode3(bool has_weight = false)
    {
      NodePtr node = CreateNodeWithoutAttrs3(has_weight);

      const char tbe_bin[] = "tbe_bin";
      vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
      OpKernelBinPtr tbe_kernel_ptr = std::make_shared<OpKernelBin>(node->GetName(), std::move(buffer));
      node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

      ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", "my_kernel");
      ge::AttrUtils::SetInt(node->GetOpDesc(), "tvm_blockdim", 10);
      ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
      ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_metadata", "my_metadata");

      SetOpDecSize(node, SET_SIZE_2);
      return node;
    }

      static NodePtr CreateUnknownShapeNodeWithoutAttrs(bool has_weight = false)
      {
          FeTestOpDescBuilder builder;
          builder.SetName("test_tvm");
          builder.SetType("test_tvm");
          builder.SetInputs({1});
          builder.SetOutputs({1});
          builder.AddInputDesc({1,-1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
          builder.AddOutputDesc({1,-1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
          if (has_weight) {
              size_t len = 10;
              unique_ptr<float[]> buf(new float[len]);
              auto weight = builder.AddWeight((uint8_t*) buf.get(), len * sizeof(float), { 1, 1, 2, 5 },
                                              ge::FORMAT_NCHW, ge::DT_FLOAT);
              ge::TensorUtils::SetWeightSize(weight->MutableTensorDesc(), len * sizeof(float));
          }

          return builder.Finish();
      }

      static NodePtr CreateUnknownShapeNode(bool has_weight = false)
      {
          NodePtr node = CreateUnknownShapeNodeWithoutAttrs(has_weight);

          const char tbe_bin[] = "tbe_bin";
          vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
          OpKernelBinPtr tbe_kernel_ptr = std::make_shared<OpKernelBin>(node->GetName(), std::move(buffer));
          node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

          ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", "my_kernel");
          ge::AttrUtils::SetInt(node->GetOpDesc(), "tvm_blockdim", 10);
          ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
          ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_metadata", "my_metadata");

          return node;
      }

    static NodePtr CreateNodeWithInputNode(string input_op)
    {
        FeTestOpDescBuilder builder;
        builder.SetName("test_tvm");
        builder.SetType("test_tvm");
        builder.SetInputs({1});
        builder.SetOutputs({1});
        builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
        builder.AddOutputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
        NodePtr node =  builder.Finish();
        const char tbe_bin[] = "tbe_bin";
        vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
        OpKernelBinPtr tbe_kernel_ptr = std::make_shared<OpKernelBin>(node->GetName(), std::move(buffer));
        node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
        ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", "my_kernel");
        ge::AttrUtils::SetInt(node->GetOpDesc(), "tvm_blockdim", 10);
        ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
        ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_metadata", "my_metadata");

        FeTestOpDescBuilder builder1;
        builder1.SetName(input_op);
        builder1.SetType(input_op);
        builder1.AddOutputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
        NodePtr input_node =  builder1.Finish();
        node->AddLinkFrom(input_node);
        SetOpDecSize(node, SET_SIZE);
        return node;
   }

  static NodePtr CreateNodeWithoutAttrsWithLargeWeightOffset(bool has_weight = false)
  {
      FeTestOpDescBuilder builder;
      builder.SetName("test_tvm");
      builder.SetInputs({1});
      builder.SetOutputs({1});
      builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
      builder.AddOutputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
      if (has_weight) {
          size_t len = 10;
          unique_ptr<float[]> buf(new float[len]);
          auto weight = builder.AddWeight((uint8_t*) buf.get(), len * sizeof(float), { 1, 1, 2, 5 },
                                          ge::FORMAT_NCHW, ge::DT_FLOAT);
          ge::TensorUtils::SetWeightSize(weight->MutableTensorDesc(), len * sizeof(float));
          ge::TensorUtils::SetDataOffset(weight->MutableTensorDesc(), 30*1024*1024*1024L + 1L);
      }

      return builder.Finish();
  }

  static NodePtr CreateNodeWithVarInputAttr(bool has_weight = false)
  {
      NodePtr node = CreateNodeWithoutAttrsWithLargeWeightOffset(has_weight);

      const char tbe_bin[] = "tbe_bin";
      vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
      OpKernelBinPtr tbe_kernel_ptr = std::make_shared<OpKernelBin>(node->GetName(), std::move(buffer));
      node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

      ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", "my_kernel");
      ge::AttrUtils::SetInt(node->GetOpDesc(), "tvm_blockdim", 10);
      ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
      ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_metadata", "my_metadata");

      vector<bool> input_is_addr_var = {true, true, true};

      ge::AttrUtils::SetListBool(node->GetOpDesc(), "INPUT_IS_VAR", input_is_addr_var);
      SetOpDecSize(node, SET_SIZE);
      return node;
  }

    static NodePtr CreateNodeWithoutAttrs2(bool has_weight = false)
    {
        FeTestOpDescBuilder builder;
        builder.SetName("test_tvm");
        builder.SetInputs({0,1});
        builder.SetOutputs({1});
        builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
        builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
        builder.AddOutputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
        if (has_weight) {
            size_t len = 10;
            unique_ptr<float[]> buf(new float[len]);
            auto weight = builder.AddWeight((uint8_t*) buf.get(), len * sizeof(float), { 1, 1, 2, 5 },
                    ge::FORMAT_NCHW, ge::DT_FLOAT);
            ge::TensorUtils::SetWeightSize(weight->MutableTensorDesc(), len * sizeof(float));
        }

        return builder.Finish2();
    }

    static NodePtr CreateNode2(bool has_weight = false)
    {
        NodePtr node = CreateNodeWithoutAttrs2(has_weight);

        const char tbe_bin[] = "tbe_bin";
        vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
        OpKernelBinPtr tbe_kernel_ptr = std::make_shared<OpKernelBin>(node->GetName(), std::move(buffer));
        node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

        ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", "my_kernel");
        ge::AttrUtils::SetInt(node->GetOpDesc(), "tvm_blockdim", 10);
        ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
        ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_metadata", "my_metadata");

        SetOpDecSize(node, SET_SIZE);
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

    static void DestroyContext(TaskBuilderContext &context) {
    }

protected:
    NodePtr node_ { nullptr };
    TaskBuilderContext context_;
    std::shared_ptr<TbeTaskBuilderAdapter> adapter_;
};

TEST_F(UTEST_TbeTaskBuilderAdapter, case_no_bin_attr_error)
{
    NodePtr node = CreateNodeWithoutAttrs();
    ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", "my_kernel");
    ge::AttrUtils::SetInt(node->GetOpDesc(), "tvm_blockdim", 10);
    ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
    ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_metadata", "my_metadata");

    TbeTaskBuilderAdapter adapter(*node, context_);
    EXPECT_NE(adapter.Init(), fe::SUCCESS);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, case_no_magic_attr_error)
{
    NodePtr node = CreateNodeWithoutAttrs();

    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
    OpKernelBinPtr tbe_kernel_ptr = std::make_shared<OpKernelBin>(node->GetName(), std::move(buffer));
    node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

    ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", "my_kernel");
    ge::AttrUtils::SetInt(node->GetOpDesc(), "tvm_blockdim", 10);
    ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_metadata", "my_metadata");

    TbeTaskBuilderAdapter adapter(*node, context_);
    EXPECT_NE(adapter.Init(), fe::SUCCESS);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, check_size_failed_change_input)
{
  NodePtr node = CreateNode(true);
  ge::OpDesc::Vistor<ge::GeTensorDesc> tensors_input = node->GetOpDesc()->GetAllInputsDesc();
  for (size_t i = 0; i < tensors_input.size(); i++) {
    ge::GeTensorDesc tensor_input = tensors_input.at(i);
    CheckSizeFuction(tensor_input);
    node->GetOpDesc()->UpdateInputDesc(i, tensor_input);
  }
  TbeTaskBuilderAdapter adapter(*node, context_);
  EXPECT_EQ(adapter.Init(), fe::FAILED);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, check_size_failed_change_output)
{
  NodePtr node = CreateNode(true);
  ge::OpDesc::Vistor<ge::GeTensorDesc> tensors_output = node->GetOpDesc()->GetAllOutputsDesc();
  for (size_t i = 0; i < tensors_output.size(); i++) {
    ge::GeTensorDesc tensor_output = tensors_output.at(i);
    CheckSizeFuction(tensor_output);
    node->GetOpDesc()->UpdateOutputDesc(i, tensor_output);
  }
  TbeTaskBuilderAdapter adapter(*node, context_);
  EXPECT_EQ(adapter.Init(), fe::FAILED);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, check_size_success_change_output_real_calc_flag)
{
  NodePtr node = CreateNode3(true);
  ge::OpDesc op_desc = *(node->GetOpDesc().get());
  ge::AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_GET_TENSOR_ACTUAL_SIZE, 1);
  for (size_t i = 0; i < op_desc.GetOutputsSize(); i++) {
    ge::GeTensorDesc tensor_output = op_desc.GetOutputDesc(i);
    node->GetOpDesc()->UpdateOutputDesc(i, tensor_output);
  }
  TbeTaskBuilderAdapter adapter(*node, context_);
  EXPECT_EQ(adapter.Init(), fe::SUCCESS);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, check_size_failed_change_output_real_calc_flag)
{
  NodePtr node = CreateNode3(true);
  ge::OpDesc op_desc = *(node->GetOpDesc().get());
  ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_GET_TENSOR_ACTUAL_SIZE, 0);
  for (size_t i = 0; i < op_desc.GetOutputsSize(); i++) {
    ge::GeTensorDesc tensor_output = op_desc.GetOutputDesc(i);
    node->GetOpDesc()->UpdateOutputDesc(i, tensor_output);
  }
  TbeTaskBuilderAdapter adapter(*node, context_);
  EXPECT_EQ(adapter.Init(), fe::FAILED);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, l2cache_rc_cache_01_no_args) {
  fe::SetFunctionState(FuncParamType::FUSION_L2, false);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2CacheMode)] = static_cast<int64_t>(L2CacheMode::DEFAULT);
  NodePtr node_ptr = CreateNode();

  OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  auto input_desc = op_desc_ptr->MutableInputDesc(0);
  ge::AttrUtils::SetBool(input_desc, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, true);
  vector<int32_t> read_dist_vec = {0, 0};
  ge::AttrUtils::SetListInt(input_desc, ge::ATTR_NAME_DATA_VISIT_DISTANCE, read_dist_vec);
  ge::AttrUtils::SetInt(op_desc_ptr, "globalworkspace_spec_workspace_bytes", 32);
  ge::AttrUtils::SetBool(op_desc_ptr, kStaticToDynamicSoftSyncOp, true);
  ge::AttrUtils::SetInt(op_desc_ptr, "op_para_size", 1);
  std::shared_ptr<TbeTaskBuilderAdapter> tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node_ptr, context_));
  tbe_task_builder_adapter->Init();
  domi::TaskDef task_def = {};
  Status ret = tbe_task_builder_adapter->Run(task_def);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(ge::AttrUtils::HasAttr(input_desc, "_fe_l2cache_graph_read_mode"), false);
  auto kernel_task_def = task_def.mutable_kernel();
  EXPECT_EQ(kernel_task_def->args_size(), 24);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, l2cache_rc_cache_01_no_args_02) {
  fe::SetFunctionState(FuncParamType::FUSION_L2, false);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2CacheMode)] = static_cast<int64_t>(L2CacheMode::DEFAULT);
  NodePtr node_ptr = CreateNode();

  OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  ge::AttrUtils::SetStr(op_desc_ptr, ATTR_NAME_KERNEL_LIST_FIRST_NAME, op_desc_ptr->GetName());
  auto input_desc = op_desc_ptr->MutableInputDesc(0);
  ge::AttrUtils::SetBool(input_desc, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, true);
  vector<int32_t> read_dist_vec = {0, 0};
  ge::AttrUtils::SetListInt(input_desc, ge::ATTR_NAME_DATA_VISIT_DISTANCE, read_dist_vec);
  std::shared_ptr<TbeTaskBuilderAdapter> tbe_task_builder_adapter =
          shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node_ptr, context_));
  tbe_task_builder_adapter->Init();
  domi::TaskDef task_def = {};
  Status ret = tbe_task_builder_adapter->Run(task_def);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(ge::AttrUtils::HasAttr(input_desc, "_fe_l2cache_graph_read_mode"), false);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, l2cache_rc_cache_02_withlifecycle_nnw) {
  fe::SetFunctionState(FuncParamType::FUSION_L2, false);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2CacheMode)] = static_cast<int64_t>(L2CacheMode::RC);

  NodePtr node = CreateNode();
  OpDescPtr op = node->GetOpDesc();
  auto op_input_desc = op->MutableInputDesc(0);
  ge::AttrUtils::SetBool(op_input_desc, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, true);
  vector<int32_t> read_dist_vec = {1, 2};
  ge::AttrUtils::SetListInt(op_input_desc, ge::ATTR_NAME_DATA_VISIT_DISTANCE, read_dist_vec);
  auto tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node, context_));
  tbe_task_builder_adapter->Init();
  domi::TaskDef task_def = {};
  Status ret = tbe_task_builder_adapter->Run(task_def);
  EXPECT_EQ(fe::SUCCESS, ret);
  CheckGraphReadMode(op_input_desc, L2CacheReadMode::READ_INVALID);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, l2cache_rc_cache_03_withlifecycle_ri) {
  fe::SetFunctionState(FuncParamType::FUSION_L2, false);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2CacheMode)] = static_cast<int64_t>(L2CacheMode::RC);

  NodePtr node = CreateNode();
  OpDescPtr op = node->GetOpDesc();
  auto op_input_desc = op->MutableInputDesc(0);
  ge::AttrUtils::SetBool(op_input_desc, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, true);
  vector<int32_t> read_dist_vec = {6};
  ge::AttrUtils::SetListInt(op_input_desc, ge::ATTR_NAME_DATA_VISIT_DISTANCE, read_dist_vec);

  auto tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node, context_));
  tbe_task_builder_adapter->Init();
  domi::TaskDef task_def = {};
  Status ret = tbe_task_builder_adapter->Run(task_def);
  EXPECT_EQ(fe::SUCCESS, ret);
  CheckGraphReadMode(op_input_desc, L2CacheReadMode::READ_INVALID);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, l2cache_rc_cache_04_withlifecycle_distance_notexist) {
  fe::SetFunctionState(FuncParamType::FUSION_L2, false);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2CacheMode)] = static_cast<int64_t>(L2CacheMode::RC);

  NodePtr node = CreateNode();
  OpDescPtr op = node->GetOpDesc();
  auto op_input_desc = op->MutableInputDesc(0);
  ge::AttrUtils::SetBool(op_input_desc, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, true);
  auto tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node, context_));
  tbe_task_builder_adapter->Init();
  domi::TaskDef task_def = {};
  Status ret = tbe_task_builder_adapter->Run(task_def);
  EXPECT_EQ(fe::SUCCESS, ret);
  CheckGraphReadMode(op_input_desc, L2CacheReadMode::NOT_NEED_WRITEBACK);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, l2cache_rc_cache_05_withoutlifecycle_rl) {
  fe::SetFunctionState(FuncParamType::FUSION_L2, false);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2CacheMode)] = static_cast<int64_t>(L2CacheMode::RC);

  NodePtr node = CreateNode();
  OpDescPtr op = node->GetOpDesc();
  auto op_input_desc = op->MutableInputDesc(0);
  vector<int32_t> read_dist_vec = {0, 2};
  ge::AttrUtils::SetListInt(op_input_desc, ge::ATTR_NAME_DATA_VISIT_DISTANCE, read_dist_vec);
  auto tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node, context_));
  tbe_task_builder_adapter->Init();
  domi::TaskDef task_def = {};
  Status ret = tbe_task_builder_adapter->Run(task_def);
  EXPECT_EQ(fe::SUCCESS, ret);
  CheckGraphReadMode(op_input_desc, L2CacheReadMode::READ_LAST);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, l2cache_rc_cache_06_withoutlifecycle_ri) {
  fe::SetFunctionState(FuncParamType::FUSION_L2, false);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2CacheMode)] = static_cast<int64_t>(L2CacheMode::RC);

  NodePtr node = CreateNode();
  OpDescPtr op = node->GetOpDesc();
  auto op_input_desc = op->MutableInputDesc(0);
  vector<int32_t> read_dist_vec = {0, 6};
  ge::AttrUtils::SetListInt(op_input_desc, ge::ATTR_NAME_DATA_VISIT_DISTANCE, read_dist_vec);
  auto tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node, context_));
  tbe_task_builder_adapter->Init();
  domi::TaskDef task_def = {};
  Status ret = tbe_task_builder_adapter->Run(task_def);
  EXPECT_EQ(fe::SUCCESS, ret);
  CheckGraphReadMode(op_input_desc, L2CacheReadMode::READ_INVALID);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, l2cache_rc_cache_07_withoutlifecycle_distance_notexist) {
  fe::SetFunctionState(FuncParamType::FUSION_L2, false);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2CacheMode)] = static_cast<int64_t>(L2CacheMode::RC);

  NodePtr node = CreateNode();
  OpDescPtr op = node->GetOpDesc();
  auto op_input_desc = op->MutableInputDesc(0);
  auto tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node, context_));
  tbe_task_builder_adapter->Init();
  domi::TaskDef task_def = {};
  Status ret = tbe_task_builder_adapter->Run(task_def);
  EXPECT_EQ(fe::SUCCESS, ret);
  CheckGraphReadMode(op_input_desc, L2CacheReadMode::READ_LAST);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, l2cache_rc_cache_08_datainput) {
  fe::SetFunctionState(FuncParamType::FUSION_L2, false);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2CacheMode)] = static_cast<int64_t>(L2CacheMode::RC);

  NodePtr node_ptr = CreateNodeWithInputNode(DATA);
  OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  auto input_desc = op_desc_ptr->MutableInputDesc(1);
  vector<int32_t> read_dist_vec = {-1, 2};
  ge::AttrUtils::SetListInt(input_desc, ge::ATTR_NAME_DATA_VISIT_DISTANCE, read_dist_vec);

  auto tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node_ptr, context_));
  tbe_task_builder_adapter->Init();
  domi::TaskDef task_def = {};
  string session_graph_id = "1";
  ge::AttrUtils::SetStr(op_desc_ptr, ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
  Status ret = tbe_task_builder_adapter->Run(task_def);
  EXPECT_EQ(fe::SUCCESS, ret);
  CheckGraphReadMode(input_desc, L2CacheReadMode::NOT_NEED_WRITEBACK);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, l2cache_rc_cache_09_constinput) {
  fe::SetFunctionState(FuncParamType::FUSION_L2, false);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2CacheMode)] = static_cast<int64_t>(L2CacheMode::RC);

  NodePtr node_ptr = CreateNodeWithInputNode(CONSTANTOP);
  OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  auto input_desc = op_desc_ptr->MutableInputDesc(1);
  vector<int32_t> read_dist_vec = {-1, 2};
  ge::AttrUtils::SetListInt(input_desc, ge::ATTR_NAME_DATA_VISIT_DISTANCE, read_dist_vec);

  auto tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node_ptr, context_));
  tbe_task_builder_adapter->Init();
  domi::TaskDef task_def = {};
  Status ret = tbe_task_builder_adapter->Run(task_def);
  EXPECT_EQ(fe::SUCCESS, ret);
  CheckGraphReadMode(input_desc, L2CacheReadMode::NOT_NEED_WRITEBACK);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, l2cache_rc_cache_10_variableinput) {
  fe::SetFunctionState(FuncParamType::FUSION_L2, false);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2CacheMode)] = static_cast<int64_t>(L2CacheMode::RC);

  NodePtr node_ptr = CreateNodeWithInputNode(VARIABLE);
  OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  auto input_desc = op_desc_ptr->MutableInputDesc(1);
  vector<int32_t> read_dist_vec = {-1, 2};
  ge::AttrUtils::SetListInt(input_desc, ge::ATTR_NAME_DATA_VISIT_DISTANCE, read_dist_vec);

  auto tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node_ptr, context_));
  tbe_task_builder_adapter->Init();
  domi::TaskDef task_def = {};
  Status ret = tbe_task_builder_adapter->Run(task_def);
  EXPECT_EQ(fe::SUCCESS, ret);
  CheckGraphReadMode(input_desc, L2CacheReadMode::READ_LAST);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, MemCpyForL2IdAndL2Addr_1) {
  NodePtr node_ptr = CreateNodeWithInputNode(VARIABLE);
  auto tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node_ptr, context_));

  uint64_t cur_ptr = 0;
  uint32_t l2_args_size = 1;
  int64_t data_in_l2_id;
  uint64_t data_in_l2_addr;
  tbe_task_builder_adapter->MemCpyForL2IdAndL2Addr(cur_ptr, l2_args_size, data_in_l2_id, data_in_l2_addr);
  EXPECT_EQ(cur_ptr, 0);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, MemCpyForL2IdAndL2Addr_2) {
  NodePtr node_ptr = CreateNodeWithInputNode(VARIABLE);
  auto tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node_ptr, context_));

  uint64_t cur_ptr = 0;
  uint32_t l2_args_size = 10;
  int64_t data_in_l2_id;
  uint64_t data_in_l2_addr;
  tbe_task_builder_adapter->MemCpyForL2IdAndL2Addr(cur_ptr, l2_args_size, data_in_l2_id, data_in_l2_addr);
  EXPECT_EQ(cur_ptr, 0);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, DealInputOutputWithDdr) {
  NodePtr node_ptr = CreateNodeWithInputNode(VARIABLE);
  auto tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node_ptr, context_));

  int32_t data_num = 1;
  uint64_t cur_ptr = 0;
  uint32_t l2_args_size;
  tbe_task_builder_adapter->DealInputOutputWithDdr(data_num, cur_ptr, l2_args_size);
  EXPECT_EQ(cur_ptr, 0);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, DealInputOutputL2_DataMap_1) {
  NodePtr node_ptr = CreateNodeWithInputNode(VARIABLE);
  auto tbe_task_builder_adapter = shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node_ptr, context_));

  L2DataMap l2datamap;
  int32_t data_num = 1;
  vector<void *> x_addrs;
  vector<void *> y_addrs;
  uint64_t cur_ptr = 0;
  uint32_t l2_args_size = 0;
  bool is_input;

  tbe_task_builder_adapter->DealInputOutputL2DataMap(l2datamap, data_num, (const void **)x_addrs.data(),
                                                     (const void **)y_addrs.data(),
                                                     cur_ptr, l2_args_size, is_input);
  EXPECT_EQ(l2_args_size, 0);
}

  // L2Data l2_data_ = {1, 2, 3};
  // l2datamap.emplace(1, l2_data_);

TEST_F(UTEST_TbeTaskBuilderAdapter, DealInputOutputL2_DataMap_2) {
  NodePtr node_ptr = CreateNodeWithInputNode(VARIABLE);
  auto tbe_task_builder_adapter = shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node_ptr, context_));

  L2DataMap l2datamap;
  L2Data l2_data_;
  l2_data_.l2Addr = 1;
  l2_data_.l2Index = 1;
  l2_data_.l2PageNum = 1;
  l2datamap.emplace(0, l2_data_);
  int32_t data_num = 1;
  vector<void *> x_addrs = {nullptr};
  vector<void *> y_addrs = {nullptr};
  uint64_t cur_ptr = 0;
  uint32_t l2_args_size = 0;
  bool is_input = false;
  tbe_task_builder_adapter->DealInputOutputL2DataMap(l2datamap, data_num, (const void **)x_addrs.data(), (const void **)y_addrs.data(),
                                                     cur_ptr, l2_args_size, is_input);
  EXPECT_EQ(l2_args_size, 0);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, DealInputOutputL2_FusionDataMap_1) {
  NodePtr node_ptr = CreateNodeWithInputNode(VARIABLE);
  auto tbe_task_builder_adapter = shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node_ptr, context_));

  L2FusionDataMap_t l2datamap;
  int32_t data_num = 1;
  vector<void *> x_addrs;
  vector<void *> y_addrs;
  uint64_t cur_ptr = 0;
  uint32_t l2_args_size = 0;
  bool is_input;

  tbe_task_builder_adapter->DealInputOutputL2DataMap(l2datamap, data_num, (const void **)x_addrs.data(), (const void **)y_addrs.data(),
                                                     cur_ptr, l2_args_size, is_input);
  EXPECT_EQ(l2_args_size, 0);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, DealInputOutputL2_FusionDataMap_2) {
  NodePtr node_ptr = CreateNodeWithInputNode(VARIABLE);
  auto tbe_task_builder_adapter = shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node_ptr, context_));

  L2FusionDataMap_t l2datamap;
  L2FusionData_t l2_fusion_data_;
  l2_fusion_data_.l2Addr = 1;
  l2_fusion_data_.l2Index = 1;
  l2_fusion_data_.l2PageNum = 1;
  l2datamap.emplace(0, l2_fusion_data_);
  int32_t data_num = 1;
  vector<void *> x_addrs = {nullptr};
  vector<void *> y_addrs = {nullptr};
  uint64_t cur_ptr = 0;
  uint32_t l2_args_size = 0;
  bool is_input;

  tbe_task_builder_adapter->DealInputOutputL2DataMap(l2datamap, data_num, (const void **)x_addrs.data(), (const void **)y_addrs.data(),
                                                     cur_ptr, l2_args_size, is_input);
  EXPECT_EQ(l2_args_size, 0);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, SaveTeCoreL2FlowDataForL2Fusion_failed1) {
  NodePtr node_ptr = CreateNodeWithInputNode(VARIABLE);
  auto tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node_ptr, context_));

  rtSmDesc_t rt_sm_desct;
  rt_sm_desct.data->L2_mirror_addr = 2;
  
  rtStream_t stream_id;
  int32_t input_num;
  int32_t output_num;
  uint64_t cur_ptr;
  vector<void *> x_addrs;
  vector<void *> y_addrs;
  rtL2Ctrl_t tel2ctrl = rt_sm_desct;
  uint32_t l2_args_size;
  uint32_t workspace_num;

  Status ret = tbe_task_builder_adapter->SaveTeCoreL2FlowDataForL2Fusion(input_num, output_num,
                                                                         cur_ptr,(const void **)x_addrs.data(),
                                                                         (const void **)y_addrs.data(), tel2ctrl, l2_args_size,
                                                                         workspace_num);
  EXPECT_EQ(ret, fe::PARAM_INVALID);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, SaveTeCoreL2FlowDataForL2Fusion_failed2) {
  NodePtr node_ptr = CreateNodeWithInputNode(VARIABLE);
  auto tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node_ptr, context_));

  rtSmDesc_t rt_sm_desct;
  rt_sm_desct.data->L2_mirror_addr = 2;

  rtStream_t stream_id;
  int32_t input_num;
  int32_t output_num;
  uint64_t cur_ptr;
  vector<void *> x_addrs;
  vector<void *> y_addrs;
  rtL2Ctrl_t tel2ctrl = rt_sm_desct;
  uint32_t l2_args_size;
  uint32_t workspace_num;

  Status ret = tbe_task_builder_adapter->SaveTeCoreL2FlowDataForL2Fusion(input_num, output_num,
                                                                         cur_ptr,(const void **)x_addrs.data(),
                                                                         (const void **)y_addrs.data(), tel2ctrl, l2_args_size,
                                                                         workspace_num);
  EXPECT_EQ(ret, fe::PARAM_INVALID);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, display_rt_l2_ctrl_info_succ) {
  NodePtr node_ptr = CreateNodeWithInputNode(VARIABLE);
  auto tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node_ptr, context_));

  rtSmDesc_t rt_sm_desct;
  rt_sm_desct.data->L2_mirror_addr = 2;
  rtL2Ctrl_t l2ctrl = rt_sm_desct;
  bool enable_l2 = false;
  tbe_task_builder_adapter->DisplayRtL2CtrlInfo(l2ctrl, enable_l2);
  bool ret = false;
  if (rt_sm_desct.data->L2_mirror_addr != 0) {
    ret = true;
  }
  EXPECT_EQ(ret, true);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, kernel_lanuch_1) {
  std::string str = "hello...";
  char* str_p = "hello...";
  domi::TaskDef task_def = {};
  rtSmDesc_t sm_desc;
  bool ret = TbeKernelLaunch::KernelLaunch(str, 4, str_p, 8, &sm_desc, task_def);
  EXPECT_EQ(ret, true);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, kernel_launch_with_handle_1) {
  std::string str = "hello...";
  const char* const_str_p = str.c_str();
  char* str_p = "hello...";
  domi::TaskDef task_def = {};
  rtSmDesc_t sm_desc;
  bool ret = TbeKernelLaunch::KernelLaunchWithHandle(4, str_p, 8, &sm_desc, task_def);
  EXPECT_EQ(ret, true);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, coverage_01) {
  int32_t input_num = 0;
  int32_t output_num;
  uint64_t cur_ptr;
  void *x = nullptr;
  const void **xx = (const void **)(&x);
  void *y = nullptr;
  const void **yy = (const void **)(&y);
  rtL2Ctrl_t tel2ctrl;
  uint32_t l2_args_size;
  uint32_t workspace_num;

  adapter_->SaveTeCoreL2FlowDataForL2Fusion(input_num, output_num, cur_ptr, xx, yy, tel2ctrl, l2_args_size,
      workspace_num);
  EXPECT_EQ(input_num, 0);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, coverage_02) {
  int32_t input_num = 1;
  int32_t output_num = 1;
  uint64_t *ptr = new uint64_t[100];
  uint64_t cur_ptr = (uint64_t)ptr;
  int a[5] = {0};
  void *x = (void *)a;
  const void **xx = (const void **)(&x);
  int b[5] = {0};
  void *y = (void *)b;
  const void **yy = (const void **)(&y);
  rtL2Ctrl_t tel2ctrl;
  uint32_t args_size = 0;
  uint32_t l2_args_size = 0;

  std::string stub_dev_func = "hello...";

  uint32_t core_dim;
  void *tmp_buf;
  int32_t workspace_num = 1;
  domi::TaskDef task_def;

  adapter_->DealKernelLaunchForL2Fusion(input_num, output_num, cur_ptr, xx, yy, tel2ctrl, args_size, l2_args_size,
      stub_dev_func, core_dim, tmp_buf, workspace_num, task_def);
  EXPECT_EQ(l2_args_size, 0);
  delete []ptr;
}

TEST_F(UTEST_TbeTaskBuilderAdapter, coverage_02_1) {
  int32_t input_num = 1;
  int32_t output_num = 1;
  uint64_t *ptr = new uint64_t[100];
  uint64_t cur_ptr = (uint64_t)ptr;
  int a[5] = {0};
  void *x = (void *)a;
  const void **xx = (const void **)(&x);
  int b[5] = {0};
  void *y = (void *)b;
  const void **yy = (const void **)(&y);
  rtL2Ctrl_t tel2ctrl;
  uint32_t args_size = 0;
  uint32_t l2_args_size = 0;

  std::string stub_dev_func = "hello...";

  uint32_t core_dim;
  void *tmp_buf;
  int32_t workspace_num = 1;
  domi::TaskDef task_def;

  ge::AttrUtils::SetStr(adapter_->node_.GetOpDesc(), ATTR_NAME_KERNEL_LIST_FIRST_NAME, "test");
  adapter_->DealKernelLaunchForL2Fusion(input_num, output_num, cur_ptr, xx, yy, tel2ctrl, args_size, l2_args_size,
      stub_dev_func, core_dim, tmp_buf, workspace_num, task_def);
  EXPECT_EQ(l2_args_size, 0);
  delete []ptr;
}

TEST_F(UTEST_TbeTaskBuilderAdapter, set_func_state_test) {
  FuncParamType type = FuncParamType::MAX_NUM;
  bool isopen = false;
  bool ret = fe::SetFunctionState(type, false);
  EXPECT_EQ(ret, false);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, set_bit_test) {
  size_t index = 0;
  auto largebitmap = fe::LargeBitMap(index);
  largebitmap.SetBit(index);
  EXPECT_EQ(largebitmap.GetBit(index), false);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, get_bit_test) {
  size_t index = 0;
  auto largebitmap = fe::LargeBitMap(index);
  bool ret = largebitmap.GetBit(index);
  EXPECT_EQ(ret, false);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, set_input_addr_from_database) {
  int64_t input_offset = 1;
  size_t input_index = 1;
  std::map<int64_t, uint8_t *> mem_type_to_data_mem_base1;
  uint8_t zero = 0;
  uint8_t one = 1;
  mem_type_to_data_mem_base1[0] = &zero;
  mem_type_to_data_mem_base1[1] = &one;
  adapter_->context_.mem_type_to_data_mem_base = mem_type_to_data_mem_base1;
  ge::OpDescPtr op_desc_ = std::make_shared<OpDesc>("test", "test");
  (void)ge::AttrUtils::SetInt(op_desc_, "_tensor_memory_type", -1);
  adapter_->SetInputAddrFromDataBase(input_index, input_offset);
  EXPECT_EQ(adapter_->input_addrs_.empty(), false);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, init_input) {
  auto node = CreateUnknownShapeNode("Conv_test");
  auto op_desc = node->GetOpDesc();
  adapter_->op_desc_ = op_desc;
  adapter_->InitInput();
  auto node_static = CreateNodeWithInputNode("Conv_test_1");
  auto op_desc_static = node_static->GetOpDesc();
  adapter_->op_desc_ = op_desc_static;
  fe::Status res = adapter_->InitInput();
  EXPECT_EQ(res, fe::SUCCESS);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, InitInputOnEachMode) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph");
  InitGraph1(graph);
  ge::OpDescPtr op_desc = nullptr;
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "ConcatD") {
      op_desc = node->GetOpDesc();
      adapter_->op_desc_ = op_desc;
      adapter_ = shared_ptr<TbeTaskBuilderAdapter> (new (nothrow) TbeTaskBuilderAdapter(*node, context_));
      for (auto anchor : node->GetAllInDataAnchors()) {
        ge::AnchorUtils::SetStatus(anchor, ge::ANCHOR_DATA);
      }
    }
  }
  vector<bool> input_is_addr_var = {true, true, true, true};
  (void)ge::AttrUtils::SetListBool(op_desc, ATTR_NAME_INPUT_IS_VAR, input_is_addr_var);
  op_desc->SetInputOffset({10,20,30,40});

  // i0 i1 i2 null i3
  std::vector<uint32_t> input_type_list;
  input_type_list.emplace_back(2);
  input_type_list.emplace_back(2);
  input_type_list.emplace_back(0);
  input_type_list.emplace_back(1);

  op_desc->AppendIrInput("__input0", ge::kIrInputDynamic);
  op_desc->AppendIrInput("__input1", ge::kIrInputDynamic);
  op_desc->AppendIrInput("__input2", ge::kIrInputRequired);
  op_desc->AppendIrInput("__input3", ge::kIrInputOptional);

  (void)ge::AttrUtils::SetInt(adapter_->op_desc_, kOpKernelAllInputSize, 4);
  (void)ge::AttrUtils::SetListInt(adapter_->op_desc_, kInputParaTypeList, input_type_list);
  auto res = adapter_->InitInputNoPlaceholder();
  EXPECT_EQ(res, fe::SUCCESS);
  adapter_->input_addrs_.clear();
  res = adapter_->InitInputGenPlaceholder();
  EXPECT_EQ(res, fe::SUCCESS);
  vector<void *> input_addrs_;
  input_addrs_.emplace_back(reinterpret_cast<void *>(10));
  input_addrs_.emplace_back(reinterpret_cast<void *>(20));
  input_addrs_.emplace_back(reinterpret_cast<void *>(kTaskPlaceHolderAddr));
  input_addrs_.emplace_back(reinterpret_cast<void *>(30));
  EXPECT_EQ(adapter_->input_addrs_, input_addrs_);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, CheckArrayValue) {
  NodePtr node_ptr = CreateNodeWithInputNode(VARIABLE);
  auto tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node_ptr, context_));

  const std::string name = "input1";
  int32_t input_num = 2;
  vector<void *> array(2);
  int32_t array_size = 2;
  Status ret = tbe_task_builder_adapter->CheckArrayValue(const_cast<const void **>(array.data()),
                                                         array_size, input_num, name);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, CheckArrayValue_02) {
  NodePtr node_ptr = CreateNodeWithInputNode(VARIABLE);
  auto tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node_ptr, context_));

  const std::string name = "input1";
  int32_t input_num = 3;
  vector<void *> array(2);
  int32_t array_size = 2;
  Status ret = tbe_task_builder_adapter->CheckArrayValue(const_cast<const void **>(array.data()),
                                                         array_size, input_num, name);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, AppendGlobalData) {
  NodePtr node_ptr = CreateNodeWithInputNode(VARIABLE);
  auto tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node_ptr, context_));

  OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  ge::AttrUtils::SetInt(op_desc_ptr, "globalworkspace_size", 32);
  ge::AttrUtils::SetInt(op_desc_ptr, "globalworkspace_type", 32);
  vector<void *> device_addrs;

  tbe_task_builder_adapter->AppendGlobalData(device_addrs);
}

TEST_F(UTEST_TbeTaskBuilderAdapter, MemCpyForL2IdAndL2Addr_0) {
  NodePtr node_ptr = CreateNodeWithInputNode(VARIABLE);
  auto tbe_task_builder_adapter =
      shared_ptr<TbeTaskBuilderAdapter>(new (nothrow) TbeTaskBuilderAdapter(*node_ptr, context_));

  uint64_t cur_ptr = 0;
  uint32_t l2_args_size = 9;
  int64_t data_in_l2_id = 0;
  uint64_t data_in_l2_addr = 0;
  tbe_task_builder_adapter->MemCpyForL2IdAndL2Addr(cur_ptr, l2_args_size, data_in_l2_id, data_in_l2_addr);
}
