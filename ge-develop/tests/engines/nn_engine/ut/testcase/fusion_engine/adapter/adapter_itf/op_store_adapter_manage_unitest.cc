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

#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/buffer.h"
#include "../../fe_test_utils.h"
#include "fe_llt_utils.h"
#include "securec.h"
#define protected public
#define private public
#include "common/fe_log.h"
#include "adapter/adapter_itf/task_builder_adapter.h"
#include "adapter/adapter_itf/op_store_adapter.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "runtime/kernel.h"
#include "common/configuration.h"
using namespace std;
using namespace testing;
using namespace ge;
using namespace fe;

using OpStoreAdapterManagerPtr = std::shared_ptr<fe::OpStoreAdapterManager>;

FEOpsStoreInfo cce_custom_opinfo_adapter {
      0,
      "cce-custom",
      EN_IMPL_CUSTOM_CONSTANT_CCE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/cce_custom_opinfo",
      ""
};
FEOpsStoreInfo tik_custom_opinfo_adapter  {
      1,
      "tik-custom",
      EN_IMPL_CUSTOM_TIK,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tik_custom_opinfo",
      ""
};
FEOpsStoreInfo tbe_custom_opinfo_adapter  {
      2,
      "tbe-custom",
      EN_IMPL_CUSTOM_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
      ""
};
FEOpsStoreInfo cce_constant_opinfo_adapter  {
      3,
      "cce-constant",
      EN_IMPL_HW_CONSTANT_CCE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/cce_constant_opinfo",
      ""
};
FEOpsStoreInfo cce_general_opinfo_adapter  {
      4,
      "cce-general",
      EN_IMPL_HW_GENERAL_CCE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/cce_general_opinfo",
      ""
};
FEOpsStoreInfo tik_opinfo_adapter  {
      5,
      "tik-builtin",
      EN_IMPL_HW_TIK,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tik_opinfo",
      ""
};
FEOpsStoreInfo tbe_opinfo_adapter  {
      6,
      "tbe-builtin",
      EN_IMPL_HW_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_opinfo",
      ""
};
FEOpsStoreInfo rl_opinfo_adapter  {
      7,
      "rl-builtin",
      EN_IMPL_RL,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/rl_opinfo",
      ""
};


std::vector<FEOpsStoreInfo> all_fe_ops_store_info_adapter{
      cce_custom_opinfo_adapter ,
      tik_custom_opinfo_adapter ,
      tbe_custom_opinfo_adapter ,
      cce_constant_opinfo_adapter ,
      cce_general_opinfo_adapter ,
      tik_opinfo_adapter ,
      tbe_opinfo_adapter ,
      rl_opinfo_adapter
};

class UTEST_OpStoreAdapterManage : public testing::Test
{
protected:
    void SetUp()
    {
        OpsAdapterManagePtr = make_shared<OpStoreAdapterManager>(fe::AI_CORE_NAME);
        Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (all_fe_ops_store_info_adapter);
        std::map<string, string> options;
        OpsAdapterManagePtr->Initialize(options);
        cout << OpsAdapterManagePtr->map_all_op_store_adapter_.size() << endl;
        std::cout << "One Test SetUP" << std::endl;
    }

    void TearDown()
    {
        std::cout << "a test SetUP" << std::endl;
        Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
        config.is_init_ = false;
        config.content_map_.clear();

        std::map<string, string> options;
        config.Initialize(options);
        OpsAdapterManagePtr->Finalize();

    }

protected:
    OpStoreAdapterManagerPtr OpsAdapterManagePtr;
};

TEST_F(UTEST_OpStoreAdapterManage, get_opstore_adapter_failed)
{
    OpStoreAdapterPtr adapter_ptr = nullptr;
    EXPECT_NE(OpsAdapterManagePtr->GetOpStoreAdapter(EN_RESERVED, adapter_ptr), fe::SUCCESS);
    EXPECT_EQ(adapter_ptr, nullptr);
    EXPECT_NE(OpsAdapterManagePtr->GetOpStoreAdapter(EN_IMPL_RL, adapter_ptr), fe::SUCCESS);
    EXPECT_EQ(adapter_ptr, nullptr);
}
