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
#include <sstream>

#define private public
#define protected public
#include "task_builder/fftsplus_ops_kernel_builder.h"
#include "testcase/test_utils.h"

using namespace std;
using namespace fe;
using namespace ge;
using namespace ffts;

using AicpuAutoTaskBuilderPtr = shared_ptr<AicpuAutoTaskBuilder>;
class FFTSPlusAicpuAutoTaskBuilderSTest : public testing::Test
{
protected:
	void SetUp()
	{
		aicpu_auto_task_builder_ptr_ = make_shared<AicpuAutoTaskBuilder>();
		ffts_plus_def_ptr_ = new domi::FftsPlusTaskDef;
		node_ = CreateNode();
	}
	void TearDown() {
		delete ffts_plus_def_ptr_;
	}
	static NodePtr CreateNode()
	{
		FeTestOpDescBuilder builder;
		builder.SetName("aicpu_node_name");
		builder.SetType("aicpu_node_type");
		auto node = builder.Finish();

    FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();
    node->GetOpDesc()->SetExtAttr("_ffts_plus_aicpu_ctx_def", ctxDefPtr);
    vector<uint32_t> context_id_list = {3, 4};
		ge::AttrUtils::SetListInt(node->GetOpDesc(), kAutoCtxIdList, context_id_list);

		ThreadSliceMapPtr tsmp_ptr = make_shared<ThreadSliceMap>();
    tsmp_ptr->thread_mode = 1;
    tsmp_ptr->parallel_window_size = 2;
		node->GetOpDesc()->SetExtAttr("_sgt_struct_info", tsmp_ptr);
		return node;
	}

public:
	AicpuAutoTaskBuilderPtr aicpu_auto_task_builder_ptr_;
	domi::FftsPlusTaskDef *ffts_plus_def_ptr_;
	NodePtr node_{ nullptr };
};

TEST_F(FFTSPlusAicpuAutoTaskBuilderSTest, Gen_Aicpu_AUTO_ContextDef_SUCCESS)
{
	Status ret = aicpu_auto_task_builder_ptr_->GenContextDef(node_, ffts_plus_def_ptr_);
	EXPECT_EQ(fe::SUCCESS, ret);
}
