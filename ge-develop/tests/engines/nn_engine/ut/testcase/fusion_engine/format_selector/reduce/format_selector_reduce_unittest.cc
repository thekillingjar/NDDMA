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
#include <string>
#include <memory>
#include <map>
#include <utility>

#include "common/util/op_info_util.h"
#include "common/fe_type_utils.h"
#include "fe_llt_utils.h"
#define private public
#define protected public
#include "graph/ge_tensor.h"
#include "common/platform_utils.h"
#include "fusion_manager/fusion_manager.h"
#include "ops_store/ops_kernel_manager.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "format_selector/manager/format_dtype_querier.h"
#include "format_selector/builtin/reduce/reduce_format_selector/reduce_process_nz.h"
using namespace std;
using namespace testing;
using namespace fe;

using fe::FEOpsKernelInfoStore;
using ge::GeTensorDesc;
using ge::GeShape;
using ge::AttrUtils;
using ge::Format;
using ge::DataType;
using ge::ConstGeTensorDescPtr;
using ge::GeTensorDescPtr;
using ge::OpDescPtr;
using ge::OpDesc;
using fe::InputOrOutputInfoPtr;
using ge::GeAttrValue;
using std::vector;
using std::map;
using namespace ge;

using OpImplTypeJudgePtr = std::shared_ptr<OpImplTypeJudge>;

class FormatDtypeSelectorManagerReduceUTest : public testing::Test{
protected:
    static void SetUpTestCase() {
        cout << "FEOpsKernelInfoStoreTest SetUP" << endl;
    }
    static void TearDownTestCase() {
        cout << "FEOpsKernelInfoStoreTest SetUP" << endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp(){
      FEOpsStoreInfo tbe_builtin {
              6,
              "tbe-builtin",
              EN_IMPL_HW_TBE,
              GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/format_selector/fe_config/tbe_dynamic_opinfo",
              ""
      };
      vector<FEOpsStoreInfo> store_info;
      store_info.emplace_back(tbe_builtin);
      Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
      std::map<std::string, std::string> options;
      options.emplace(ge::SOC_VERSION, "Ascend910B");
      PlatformUtils::Instance().is_init_ = false;
      PlatformUtils::Instance().Initialize(options);
      OpsKernelManager::Instance(AI_CORE_NAME).Finalize();
      OpsKernelManager::Instance(AI_CORE_NAME).Initialize();

      cout << "a test Set Up" << endl;
    }
    virtual void TearDown(){
        cout << "a test Tear Down" << endl;
    }

protected:
    static void CreateOneOpGraph(ComputeGraphPtr graph,  OpDescPtr op_desc_ptr) {
        graph->AddNode(op_desc_ptr);
    }
    static GeTensorDesc CreateTensorDesc(vector<int64_t> dim_data,  ge::Format format, ge::DataType data_type) {
        GeShape shape_data(dim_data);
        GeTensorDesc data_desc(shape_data, format, data_type);
        data_desc.SetOriginFormat(format);
        data_desc.SetOriginShape(shape_data);
        data_desc.SetOriginDataType(data_type);
        return data_desc;
    }
public:
  const string EQUAL_ORIGIN_FORMAT="NCHW,NCHW,NHWC,NHWC,HWCN,HWCN,CHWN,CHWN,NDHWC,NDHWC,NCDHW,NCDHW,DHWCN,DHWCN,DHWNC,DHWNC,ND,ND";
};

TEST_F(FormatDtypeSelectorManagerReduceUTest, op_reduce_selector_5hd_6hd_success) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {16, 32, 16, 16};
  vector<int64_t> dim_data2 = {16, 8, 1, 1};
  op_desc_ptr->AddInputDesc("x",
                          CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT16));
  op_desc_ptr->AddOutputDesc("y",
                           CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT16));

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      "tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

  ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, true);
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {2, 3});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
      op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  bool support_flag = false;
  map<string, vector<ge::Format>>::reverse_iterator iter;
  for (iter = format_dtype_info.format_map.rbegin(); iter != format_dtype_info.format_map.rend(); iter++) {
    vector<ge::Format>::iterator ret;
    ret =
        std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_NC1HWC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, true);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_C1HWNCoC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, true);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_FRACTAL_Z);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, true);
  }
}

TEST_F(FormatDtypeSelectorManagerReduceUTest, op_reduce_selector_5hd_6hd_ubsize_greater) {
  PlatformUtils::Instance().soc_version_ = "Ascend910B";
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {16, 25600, 160, 32};
  vector<int64_t> dim_data2 = {16, 8, 1, 1};
  op_desc_ptr->AddInputDesc("x",
                          CreateTensorDesc(dim_data1, FORMAT_NHWC, DT_FLOAT16));
  op_desc_ptr->AddOutputDesc("y",
                           CreateTensorDesc(dim_data1, FORMAT_NHWC, DT_FLOAT16));
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
    "tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

  ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, false);
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {1, 2});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);
  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
    op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  bool support_flag = false;
  map<string, vector<ge::Format>>::reverse_iterator iter;
  for (iter = format_dtype_info.format_map.rbegin(); iter != format_dtype_info.format_map.rend(); iter++) {
    vector<ge::Format>::iterator ret;
    ret =
      std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_NC1HWC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_C1HWNCoC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_FRACTAL_Z);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);
  }
}

TEST_F(FormatDtypeSelectorManagerReduceUTest, op_reduce_selector_5hd_6hd_ubsize_success) {
  PlatformUtils::Instance().soc_version_ = "Ascend910B";
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {16, 32, 256, 16};
  vector<int64_t> dim_data2 = {16, 8, 1, 1};
  op_desc_ptr->AddInputDesc("x",
                          CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT16));
  op_desc_ptr->AddOutputDesc("y",
                           CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT16));

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
    "tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

  ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, false);
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {2, 3});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
    op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  bool support_flag = false;
  map<string, vector<ge::Format>>::reverse_iterator iter;
  for (iter = format_dtype_info.format_map.rbegin(); iter != format_dtype_info.format_map.rend(); iter++) {
    vector<ge::Format>::iterator ret;
    ret =
      std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_NC1HWC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    int pos = iter->first.find("input");
    if (pos != -1) {
      EXPECT_EQ(support_flag, true);
    } else {
      EXPECT_EQ(support_flag, true);
    }

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_C1HWNCoC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_FRACTAL_Z);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);
  }
}

TEST_F(FormatDtypeSelectorManagerReduceUTest, op_reduce_selector_5hd_6hd_failed1) {
  PlatformUtils::Instance().soc_version_ = "Ascend910B";
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {16, 32, 16, 15};
  op_desc_ptr->AddInputDesc("x",
                          CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("y",
                           CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      "tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

  ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, false);
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {2, 3});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
      op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  map<string, vector<ge::Format>>::reverse_iterator iter;
  for (iter = format_dtype_info.format_map.rbegin(); iter != format_dtype_info.format_map.rend(); iter++) {
    bool support_flag = false;
    vector<ge::Format>::iterator ret;
    ret =
        std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_NC1HWC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    int pos = iter->first.find("input");
    if (pos != -1) {
      EXPECT_EQ(support_flag, true);
    } else {
      EXPECT_EQ(support_flag, true);
    }

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_C1HWNCoC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_FRACTAL_Z);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_FRACTAL_NZ);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);
  }
}

TEST_F(FormatDtypeSelectorManagerReduceUTest, op_reduce_selector_5hd_6hd_failed2) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {16, 32, 16, 16};
  GeShape shape_data1(dim_data1);
  op_desc_ptr->AddInputDesc("x",
                          CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("y",
                           CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      "tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

  ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, true);
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {0, 1});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
      op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  map<string, vector<ge::Format>>::iterator iter;
  for (iter = format_dtype_info.format_map.begin(); iter != format_dtype_info.format_map.end(); iter++) {
    bool support_flag = false;
    vector<ge::Format>::iterator ret;
    ret =
        std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_NC1HWC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    // EXPECT_EQ(support_flag, true);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_C1HWNCoC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_FRACTAL_Z);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_FRACTAL_NZ);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, true);
  }
}

TEST_F(FormatDtypeSelectorManagerReduceUTest, op_reduce_selector_5hd_6hd_failed3) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {16, 32};
  GeShape shape_data1(dim_data1);
  op_desc_ptr->AddInputDesc("x",
                          CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));
  op_desc_ptr->AddOutputDesc("y",
                           CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT));

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      "tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

  ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, true);
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {2, 3});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
      op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);

  map<string, vector<ge::Format>>::reverse_iterator iter;
  for (iter = format_dtype_info.format_map.rbegin(); iter != format_dtype_info.format_map.rend(); iter++) {
    bool support_flag = false;
    vector<ge::Format>::iterator ret;
    ret =
        std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_NC1HWC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_C1HWNCoC0);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_FRACTAL_Z);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, false);

    support_flag = false;
    ret = std::find(iter->second.begin(), iter->second.end(),
                    ge::FORMAT_FRACTAL_NZ);
    if (ret != iter->second.end()) {
      support_flag = true;
    }
    EXPECT_EQ(support_flag, true);
  }
}


TEST_F(FormatDtypeSelectorManagerReduceUTest, op_reduce_selector_6hd_success_NDHWC) {
    const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
    vector<int64_t> dim_data1 = {16, 32, 16, 16, 16};
    op_desc_ptr->AddInputDesc("x",
                            CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT16));
    op_desc_ptr->AddOutputDesc(
            "y", CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT16));

    OpKernelInfoPtr op_kernel_info_ptr;
   op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
            "tbe-builtin", op_desc_ptr->GetType());
    EXPECT_NE(op_kernel_info_ptr, nullptr);

    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

    ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, true);
    ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {2, 3});
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr node = graph->AddNode(op_desc_ptr);

    // check inputs
    FormatDtypeInfo format_dtype_info;
    Status result = format_dtype_querier.GetSupportFormatAndDtype(
            op_kernel_info_ptr, node, false, format_dtype_info);
    EXPECT_EQ(fe::SUCCESS, result);

    bool support_flag = false;
    map<string, vector<ge::Format>>::reverse_iterator iter;
    for (iter = format_dtype_info.format_map.rbegin(); iter != format_dtype_info.format_map.rend(); iter++) {
        vector<ge::Format>::iterator ret;
        ret = std::find(iter->second.begin(), iter->second.end(),
                        ge::FORMAT_NDC1HWC0);
        if (ret != iter->second.end()) {
            support_flag = true;
        }
        EXPECT_EQ(support_flag, true);

        support_flag = false;
        ret =
                std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_NC1HWC0);
        if (ret != iter->second.end()) {
            support_flag = true;
        }
        EXPECT_EQ(support_flag, false);

        support_flag = false;
        ret = std::find(iter->second.begin(), iter->second.end(),
                        ge::FORMAT_C1HWNCoC0);
        if (ret != iter->second.end()) {
            support_flag = true;
        }
        EXPECT_EQ(support_flag, true);

        support_flag = false;
        ret = std::find(iter->second.begin(), iter->second.end(),
                        ge::FORMAT_FRACTAL_Z);
        if (ret != iter->second.end()) {
            support_flag = true;
        }
        EXPECT_EQ(support_flag, true);
    }
}

// NCDHW
TEST_F(FormatDtypeSelectorManagerReduceUTest, op_reduce_selector_6hd_success_NCDHW) {
    const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
    vector<int64_t> dim_data1 = {16, 16, 32, 16, 16};
    op_desc_ptr->AddInputDesc("x",
                            CreateTensorDesc(dim_data1, FORMAT_NCDHW, DT_FLOAT16));
    op_desc_ptr->AddOutputDesc(
            "y", CreateTensorDesc(dim_data1, FORMAT_NCDHW, DT_FLOAT16));

    OpKernelInfoPtr op_kernel_info_ptr;
   op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
            "tbe-builtin", op_desc_ptr->GetType());
    EXPECT_NE(op_kernel_info_ptr, nullptr);

    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

    ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, true);
    ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {2, 3});
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr node = graph->AddNode(op_desc_ptr);

    // check inputs
    FormatDtypeInfo format_dtype_info;
    Status result = format_dtype_querier.GetSupportFormatAndDtype(
            op_kernel_info_ptr, node, false, format_dtype_info);
    EXPECT_EQ(fe::SUCCESS, result);

    bool support_flag = false;
    map<string, vector<ge::Format>>::reverse_iterator iter;
    for (iter = format_dtype_info.format_map.rbegin(); iter != format_dtype_info.format_map.rend(); iter++) {
        vector<ge::Format>::iterator ret;
        ret = std::find(iter->second.begin(), iter->second.end(),
                        ge::FORMAT_NDC1HWC0);
        if (ret != iter->second.end()) {
            support_flag = true;
        }
        EXPECT_EQ(support_flag, true);

        support_flag = false;
        ret =
                std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_NC1HWC0);
        if (ret != iter->second.end()) {
            support_flag = true;
        }
        EXPECT_EQ(support_flag, false);

        support_flag = false;
        ret = std::find(iter->second.begin(), iter->second.end(),
                        ge::FORMAT_C1HWNCoC0);
        if (ret != iter->second.end()) {
            support_flag = true;
        }
        EXPECT_EQ(support_flag, true);

        support_flag = false;
        ret = std::find(iter->second.begin(), iter->second.end(),
                        ge::FORMAT_FRACTAL_Z);
        if (ret != iter->second.end()) {
            support_flag = true;
        }
        EXPECT_EQ(support_flag, true);
    }
}

/* shape of two tensor is not 5d */
TEST_F(FormatDtypeSelectorManagerReduceUTest, op_reduce_selector_6hd_failed) {
    const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
    vector<int64_t> dim_data1 = {16, 32, 16, 16};
    vector<int64_t> dim_data2 = {16, 8, 1, 1};
    op_desc_ptr->AddInputDesc("x",
                            CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT16));
    op_desc_ptr->AddOutputDesc(
            "y", CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT16));

    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
            "tbe-builtin", op_desc_ptr->GetType());
    EXPECT_NE(op_kernel_info_ptr, nullptr);

    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

    ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, true);
    ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {2, 3});
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr node = graph->AddNode(op_desc_ptr);

    // check inputs
    FormatDtypeInfo format_dtype_info;
    Status result = format_dtype_querier.GetSupportFormatAndDtype(
            op_kernel_info_ptr, node, false, format_dtype_info);
    EXPECT_EQ(fe::SUCCESS, result);

    bool support_flag;
    map<string, vector<ge::Format>>::reverse_iterator iter;
    for (iter = format_dtype_info.format_map.rbegin(); iter != format_dtype_info.format_map.rend(); iter++) {
        vector<ge::Format>::iterator ret;
        support_flag = false;
        ret = std::find(iter->second.begin(), iter->second.end(),
                        ge::FORMAT_NDC1HWC0);
        if (ret != iter->second.end()) {
            support_flag = true;
        }
        EXPECT_EQ(support_flag, false);

        support_flag = false;
        ret =
                std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_NC1HWC0);
        if (ret != iter->second.end()) {
            support_flag = true;
        }
        EXPECT_EQ(support_flag, true);

        support_flag = false;
        ret = std::find(iter->second.begin(), iter->second.end(),
                        ge::FORMAT_C1HWNCoC0);
        if (ret != iter->second.end()) {
            support_flag = true;
        }
        EXPECT_EQ(support_flag, true);

        support_flag = false;
        ret = std::find(iter->second.begin(), iter->second.end(),
                        ge::FORMAT_FRACTAL_Z);
        if (ret != iter->second.end()) {
            support_flag = true;
        }
        EXPECT_EQ(support_flag, true);
    }
}

TEST_F(FormatDtypeSelectorManagerReduceUTest, op_reduce_selector_6hd_failed2) {
    const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
    vector<int64_t> dim_data1 = {16, 32, 16, 16, 16};
    vector<int64_t> dim_data2 = {16, 8, 1, 1, 19};
    op_desc_ptr->AddInputDesc("x",
                            CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT16));
    op_desc_ptr->AddOutputDesc(
            "y", CreateTensorDesc(dim_data1, FORMAT_NDHWC, DT_FLOAT16));

    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin",
            op_desc_ptr->GetType());
    EXPECT_NE(op_kernel_info_ptr, nullptr);

    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

    ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, true);
    ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {2, 4});
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr node = graph->AddNode(op_desc_ptr);

    // check inputs
    FormatDtypeInfo format_dtype_info;
    Status result = format_dtype_querier.GetSupportFormatAndDtype(
            op_kernel_info_ptr, node, false, format_dtype_info);
    EXPECT_EQ(fe::SUCCESS, result);

    bool support_flag = false;
    map<string, vector<ge::Format>>::reverse_iterator iter;
    uint32_t count = 0;
    for (iter = format_dtype_info.format_map.rbegin(); iter != format_dtype_info.format_map.rend(); iter++) {
        vector<ge::Format>::iterator ret;
        support_flag = false;
        ret = std::find(iter->second.begin(), iter->second.end(),
                        ge::FORMAT_NDC1HWC0);
        if (ret != iter->second.end()) {
            support_flag = true;
        }
        EXPECT_EQ(support_flag, false);

        support_flag = false;
        ret =
                std::find(iter->second.begin(), iter->second.end(), ge::FORMAT_NC1HWC0);
        if (ret != iter->second.end()) {
            support_flag = true;
        }
        if (count == 0) {
            EXPECT_EQ(support_flag, false);
        } else {
            EXPECT_EQ(support_flag, false);
        }

        support_flag = false;
        ret = std::find(iter->second.begin(), iter->second.end(),
                        ge::FORMAT_C1HWNCoC0);
        if (ret != iter->second.end()) {
            support_flag = true;
        }
        EXPECT_EQ(support_flag, false);

        support_flag = false;
        ret = std::find(iter->second.begin(), iter->second.end(),
                        ge::FORMAT_FRACTAL_Z);
        if (ret != iter->second.end()) {
            support_flag = true;
        }
        EXPECT_EQ(support_flag, false);
        count++;
    }
}

TEST_F(FormatDtypeSelectorManagerReduceUTest, check_origin_format_and_shape_test) {
  vector<int64_t> dim = {1};
  ge::GeShape shape(dim);
  size_t dim_min = 2;
  vector<ge::Format> input_formats;
  vector<ge::Format> output_formats;
  vector<ge::GeShape> shapes;
  shapes.emplace_back(shape);
  ReduceProcessNz reduceprocessnz;
  bool ret = reduceprocessnz.CheckOriginFormatAndShape(input_formats, output_formats, shapes, dim_min);
  EXPECT_EQ(ret, false);
}

TEST_F(FormatDtypeSelectorManagerReduceUTest, op_reduce_selector_5hd_6hd) {
  const OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce1", "ReduceOp");
  vector<int64_t> dim_data1 = {16, 32, 16, 16};
  vector<int64_t> dim_data2 = {16, 8, 1, 1};
  op_desc_ptr->AddInputDesc("x",
                          CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT16));
  op_desc_ptr->AddOutputDesc("y",
                           CreateTensorDesc(dim_data1, FORMAT_ND, DT_FLOAT16));

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      "tbe-builtin", op_desc_ptr->GetType());
  EXPECT_NE(op_kernel_info_ptr, nullptr);

  FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
  vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

  ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, true);
  ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {2, 3});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  // check inputs
  FormatDtypeInfo format_dtype_info;
  Status result = format_dtype_querier.GetSupportFormatAndDtype(
      op_kernel_info_ptr, node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, result);
}

TEST_F(FormatDtypeSelectorManagerReduceUTest, op_reduce_selector_5hd_6hd_1)
{
    const OpDescPtr op_desc_ptr= std::make_shared<OpDesc>("reduce1", "ReduceOp");
    vector<int64_t> dim_data1 = {16, 32, 16, 16, 1};
    vector<int64_t> dim_data2 = {16, 8, 1 ,1, 1};
    op_desc_ptr->AddInputDesc("x", CreateTensorDesc(dim_data1, FORMAT_NCHW, DT_FLOAT16));
    op_desc_ptr->AddOutputDesc("y", CreateTensorDesc(dim_data1, FORMAT_ND, DT_FLOAT16));
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    auto node = graph->AddNode(op_desc_ptr);

    FormatDtypeQuerier format_dtype_querier(AI_CORE_NAME);
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-builtin", op_desc_ptr->GetType());
    vector<InputOrOutputInfoPtr> inputs = op_kernel_info_ptr->GetAllInputInfo();

    ge::AttrUtils::SetBool(*(op_desc_ptr.get()), KEEP_DIMS, true);
    ge::AttrUtils::SetListInt(*(op_desc_ptr.get()), AXES_ATTR_NAME, {2, 3});

    // check inputs
    FormatDtypeInfo format_dtype_info;
    Status result = format_dtype_querier.GetSupportFormatAndDtype(op_kernel_info_ptr, node,
                                                               false, format_dtype_info);
    EXPECT_EQ(fe::SUCCESS, result);
}
