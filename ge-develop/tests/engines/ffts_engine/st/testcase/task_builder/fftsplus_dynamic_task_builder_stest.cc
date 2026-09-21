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
#include "common/sgt_slice_type.h"
#include "inc/ffts_log.h"
#include "inc/ffts_error_codes.h"
#include "inc/ffts_type.h"
#include "task_builder/fftsplus_ops_kernel_builder.h"
#include "task_builder/mode/auto/auto_thread_task_builder.h"
#include "task_builder/mode/manual/manual_thread_task_builder.h"
#include "graph/node.h"
#include "graph_builder_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/compute_graph.h"
#include "graph/op_kernel_bin.h"
#include "runtime/context.h"
#include "runtime/stream.h"
#include "runtime/rt_model.h"
#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "../test_utils.h"

using namespace std;
using namespace ffts;
using namespace ge;

#define SET_SIZE 1000

using AICAIVDynamicTaskBuilderPtr = shared_ptr<AICAIVDynamicTaskBuilder>;

class FFTSPlusDynamicTaskBuilderSTest : public testing::Test
{
protected:
  void SetUp()
  {
    aic_aiv_dy_task_builder_ptr_ = make_shared<AICAIVDynamicTaskBuilder>();
    ffts_plus_def_ptr_ = new domi::FftsPlusTaskDef;
    node_ = CreateNode();
  }
  void TearDown() {
    delete ffts_plus_def_ptr_;
  }
  static void SetOpDecSize(NodePtr& node) {
    OpDesc::Vistor<GeTensorDesc> tensors = node->GetOpDesc()->GetAllInputsDesc();
    for (int i = 0; i < node->GetOpDesc()->GetAllInputsDesc().size(); i++) {
      ge::GeTensorDesc tensor = node->GetOpDesc()->GetAllInputsDesc().at(i);
      ge::TensorUtils::SetSize(tensor, SET_SIZE);
      node->GetOpDesc()->UpdateInputDesc(i, tensor);
    }
    OpDesc::Vistor<GeTensorDesc> tensorsOutput = node->GetOpDesc()->GetAllOutputsDesc();
    for (int i = 0; i < tensorsOutput.size(); i++) {
      ge::GeTensorDesc tensorOutput = tensorsOutput.at(i);
      ge::TensorUtils::SetSize(tensorOutput, SET_SIZE);
      node->GetOpDesc()->UpdateOutputDesc(i, tensorOutput);
    }
  }
  static NodePtr CreateNode()
  {
    FeTestOpDescBuilder builder;
    builder.SetName("test_tvm");
    builder.SetType("conv");
    builder.SetInputs({ 1 });
    builder.SetOutputs({ 1 });
    builder.AddInputDesc({ -1, -1, 4, 4 }, ge::FORMAT_NCHW, ge::DT_FLOAT);
    builder.AddOutputDesc({ -1, -1, 4, 4 }, ge::FORMAT_NCHW, ge::DT_FLOAT);
    auto node = builder.Finish();

    const char tbeBin[] = "tbe_bin";
    vector<char> buffer(tbeBin, tbeBin + strlen(tbeBin));
    OpKernelBinPtr tbeKernelPtr = std::make_shared<OpKernelBin>("test_tvm", std::move(buffer));
    node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbeKernelPtr);
    ge::AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    ge::AttrUtils::SetInt(node->GetOpDesc(), "_fe_imply_type", (int64_t)ffts::OpImplType::EN_IMPL_CUSTOM_TBE);
    ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
    ge::AttrUtils::SetBool(node->GetOpDesc(), "is_first_node", true);
    ge::AttrUtils::SetBool(node->GetOpDesc(), "is_last_node", true);
    ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", "kernelname");

    vector<uint32_t> context_id_list = {3, 4};
    ge::AttrUtils::SetListInt(node->GetOpDesc(), kAutoCtxIdList, context_id_list);

    SetOpDecSize(node);
    ffts::ThreadSliceMapPtr tsmp_ptr = make_shared<ffts::ThreadSliceMap>();
    tsmp_ptr->slice_instance_num = 1;
    tsmp_ptr->parallel_window_size = 1;
    tsmp_ptr->thread_mode = 1;

    ffts::DimRange dim_rang;
    dim_rang.higher = 3;
    dim_rang.lower = 0;
    vector<ffts::DimRange> input_tensor_slice_v;
    input_tensor_slice_v.push_back(dim_rang);
    dim_rang.higher = 3;
    dim_rang.lower = 0;
    input_tensor_slice_v.push_back(dim_rang);
    input_tensor_slice_v.push_back(dim_rang);
    input_tensor_slice_v.push_back(dim_rang);
    vector<vector<ffts::DimRange>> input_tensor_slice_vv;
    input_tensor_slice_vv.push_back(input_tensor_slice_v);
    vector<vector<vector<ffts::DimRange>>> input_tensor_slice_vvv = { input_tensor_slice_vv };
    vector<vector<vector<ffts::DimRange>>> output_tensor_slice = { input_tensor_slice_vv };
    tsmp_ptr->input_tensor_slice = input_tensor_slice_vvv;
    tsmp_ptr->output_tensor_slice = output_tensor_slice;
    node->GetOpDesc()->SetExtAttr("_sgt_struct_info", tsmp_ptr);
    std::string core_type = "AIC";
    (void)ge::AttrUtils::SetStr(node->GetOpDesc(), ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
    FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();
    node->GetOpDesc()->SetExtAttr("_aicore_ctx_def", ctxDefPtr);
    return node;
	}

public:
  AICAIVDynamicTaskBuilderPtr aic_aiv_dy_task_builder_ptr_;
	domi::FftsPlusTaskDef *ffts_plus_def_ptr_;
	NodePtr node_{ nullptr };
};

TEST_F(FFTSPlusDynamicTaskBuilderSTest, Gen_AICAIV_DYContextDef_SUCCESS)
{
	Status ret = aic_aiv_dy_task_builder_ptr_->GenContextDef(node_, ffts_plus_def_ptr_);
	EXPECT_EQ(fe::SUCCESS, ret);
}
