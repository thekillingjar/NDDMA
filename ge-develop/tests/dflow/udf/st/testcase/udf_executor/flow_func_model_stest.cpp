/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#define private public
#include "model/flow_func_model.h"
#undef private
#include "ff_udf_def.pb.h"
#include "ff_deployer.pb.h"
#include "utils/udf_utils.h"
#include "mmpa/mmpa_api.h"
#include "config/global_config.h"

namespace FlowFunc {
class FlowFuncModelSTest : public testing::Test {
 protected:
  virtual void SetUp() {
    CreateModelDir();
    std::string cmd =
        "cd " + GetUdfModelPath() + "; mkdir release; cd release; touch libudf.so; tar -zcf release.tar.gz *";
    (void)system(cmd.c_str());
  }

  virtual void TearDown() {
    DeleteModelDir();
    GlobalMockObject::verify();
  }
};

TEST_F(FlowFuncModelSTest, normal) {
  std::map<AscendString, ff::udf::AttrValue> attrs;
  ff::udf::AttrValue attr_str;
  AscendString attr_val("attrValue");
  attr_str.set_s(attr_val.GetString());
  attrs["attrName"] = attr_str;
  std::string batch_model_path = CreateBatchModels(2, "flow_func", attrs);
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 2);
  for (auto &model : models) {
    EXPECT_FALSE(model->GetFlowFuncName().empty());
    EXPECT_FALSE(model->GetLibPath().empty());
    EXPECT_FALSE(model->GetNodeAttrMap().empty());
    EXPECT_FALSE(model->GetInputQueues().empty());
    EXPECT_FALSE(model->GetOutputQueues().empty());
    EXPECT_EQ(model->model_esched_process_priority_, kUserUnsetESchedPriority);
    EXPECT_EQ(model->model_esched_event_priority_, kUserUnsetESchedPriority);
    const auto &attr_map = model->GetNodeAttrMap();
    EXPECT_FALSE(attr_map.empty());
    auto attr_iter = attr_map.find("attrName");
    EXPECT_TRUE(attr_iter != attr_map.end());
    AscendString s_val;
    auto ret = attr_iter->second->GetVal(s_val);
    EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
    EXPECT_EQ(s_val, attr_val);
    int64_t int_val = 0;
    ret = attr_iter->second->GetVal(int_val);
    EXPECT_EQ(ret, FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH);
  }
}

TEST_F(FlowFuncModelSTest, invalid_buf_cfg) {
  std::map<AscendString, ff::udf::AttrValue> attrs;
  ff::udf::AttrValue attr_str;
  AscendString attr_val("attrValue");
  attr_str.set_s(attr_val.GetString());
  attrs["attrName"] = attr_str;
  // 配置中size有0值
  BufferConfigItem cfg0 = {2 * 1024 * 1024, 0, 8 * 1024, "normal"};
  std::string batch_model_path = CreateBatchModels(2, "flow_func", attrs, false, {}, {}, {cfg0});
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);

  // totalSize小于maxBufferSize
  cfg0 = {2 * 1024 * 1024, 0, 8 * 1024, "normal"};
  batch_model_path = CreateBatchModels(2, "flow_func", attrs, false, {}, {}, {cfg0});
  models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);

  // normal类型totalSize不是4K的倍数
  cfg0 = {2 * 1024, 256, 256, "normal"};
  batch_model_path = CreateBatchModels(2, "flow_func", attrs, false, {}, {}, {cfg0});
  models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);

  // huge类型totalsize不是2M的倍数
  cfg0 = {1 * 1024 * 1024, 256, 8 * 1024, "huge"};
  batch_model_path = CreateBatchModels(2, "flow_func", attrs, false, {}, {}, {cfg0});
  models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);

  // pageType不在有效范围内
  cfg0 = {64 * 1024 * 1024, 256, 8 * 1024, "dvpp"};
  batch_model_path = CreateBatchModels(2, "flow_func", attrs, false, {}, {}, {cfg0});
  models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);

  // blksize不是2^n
  cfg0 = {64 * 1024 * 1024, 258, 8 * 1024, "normal"};
  batch_model_path = CreateBatchModels(2, "flow_func", attrs, false, {}, {}, {cfg0});
  models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);

  // blksize超过2M
  cfg0 = {64 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024, "normal"};
  batch_model_path = CreateBatchModels(2, "flow_func", attrs, false, {}, {}, {cfg0});
  models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);

  // blksize不能被total size 整除
  cfg0 = {64 * 1024 * 1024 + 4 * 1024, 8 * 1024, 8 * 1024 * 1024, "normal"};
  batch_model_path = CreateBatchModels(2, "flow_func", attrs, false, {}, {}, {cfg0});
  models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
}

TEST_F(FlowFuncModelSTest, normal_built_in) {
  std::map<AscendString, ff::udf::AttrValue> model_attrs;
  ff::udf::AttrValue attr_str;
  AscendString attr_name("attrName");
  AscendString attr_val("attrValue");
  attr_str.set_s(attr_val.GetString());
  model_attrs[attr_name] = attr_str;
  ff::udf::AttrValue visible_attr;
  visible_attr.set_b(false);
  model_attrs["_dflow_visible_device_enable"] = visible_attr;
  ff::udf::AttrValue visible_attr1;
  visible_attr1.set_s("test_df_scope");
  model_attrs["_dflow_data_flow_scope"] = visible_attr1;
  std::map<std::string, std::string> attrs;
  attrs["_eschedProcessPriority"] = "0";
  attrs["_eschedEventPriority"] = "2";
  std::string batch_model_path = CreateBatchModels(2, "flow_func", model_attrs, true, attrs);
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 2);
  for (auto &model : models) {
    EXPECT_FALSE(model->GetFlowFuncName().empty());
    EXPECT_FALSE(model->GetLibPath().empty());
    EXPECT_FALSE(model->GetNodeAttrMap().empty());
    EXPECT_FALSE(model->GetInputQueues().empty());
    EXPECT_FALSE(model->GetOutputQueues().empty());
    EXPECT_EQ(model->model_esched_process_priority_, 0);
    EXPECT_EQ(model->model_esched_event_priority_, 2);
    const auto &attr_map = model->GetNodeAttrMap();
    EXPECT_FALSE(attr_map.empty());
    auto attr_iter = attr_map.find("attrName");
    EXPECT_TRUE(attr_iter != attr_map.end());
    AscendString s_val;
    auto ret = attr_iter->second->GetVal(s_val);
    EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
    EXPECT_EQ(s_val, attr_val);
    int64_t int_val = 0;
    ret = attr_iter->second->GetVal(int_val);
    EXPECT_EQ(ret, FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH);
  }
}

TEST_F(FlowFuncModelSTest, file_not_exit) {
  auto models = FlowFuncModel::ParseModels("./not_exit_path");
  EXPECT_TRUE(models.empty());
}

TEST_F(FlowFuncModelSTest, parse_process_priority_failed) {
  std::map<AscendString, ff::udf::AttrValue> model_attrs;
  ff::udf::AttrValue attr_str;
  AscendString attr_name("attrName");
  AscendString attr_val("attrValue");
  attr_str.set_s(attr_val.GetString());
  model_attrs[attr_name] = attr_str;
  std::map<std::string, std::string> attrs;
  attrs["_eschedProcessPriority"] = std::to_string(INT64_MAX);
  attrs["_eschedEventPriority"] = "2";
  std::string batch_model_path = CreateBatchModels(2, "flow_func", model_attrs, true, attrs);
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
}

TEST_F(FlowFuncModelSTest, parse_event_priority_failed) {
  std::map<AscendString, ff::udf::AttrValue> model_attrs;
  ff::udf::AttrValue attr_str;
  AscendString attr_name("attrName");
  AscendString attr_val("attrValue");
  attr_str.set_s(attr_val.GetString());
  model_attrs[attr_name] = attr_str;
  std::map<std::string, std::string> attrs;
  attrs["_eschedProcessPriority"] = "0";
  attrs["_eschedEventPriority"] = "afdasdfa;lk";
  std::string batch_model_path = CreateBatchModels(2, "flow_func", model_attrs, true, attrs);
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
}

TEST_F(FlowFuncModelSTest, normal_function_compile) {
  std::string with_release_pkg_model_path = CreateModelsWithReleasePkg(1, "flow_func", GetUdfModelPath());
  auto models = FlowFuncModel::ParseModels(with_release_pkg_model_path);
  EXPECT_EQ(models.size(), 1);
  EXPECT_EQ(models[0]->GetWorkPath(), GetUdfModelPath() + "flow_func_0.proto_dir");
}

TEST_F(FlowFuncModelSTest, visible_device_enable_success) {
  std::map<AscendString, ff::udf::AttrValue> model_attrs;
  ff::udf::AttrValue visible_attr;
  visible_attr.set_b(true);
  model_attrs["_dflow_visible_device_enable"] = visible_attr;
  std::string batch_model_path = CreateBatchModels(2, "flow_func", model_attrs, true, {});
  auto phy_device_id_bk = GlobalConfig::Instance().GetPhyDeviceId();
  GlobalConfig::Instance().SetPhyDeviceId(0);
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 2);
  auto visible_devices = getenv("ASCEND_RT_VISIBLE_DEVICES");
  EXPECT_NE(visible_devices, nullptr);
  unsetenv("ASCEND_RT_VISIBLE_DEVICES");
  GlobalConfig::Instance().SetPhyDeviceId(phy_device_id_bk);
}

TEST_F(FlowFuncModelSTest, visible_device_enable_failed) {
  std::map<AscendString, ff::udf::AttrValue> model_attrs;
  ff::udf::AttrValue visible_attr;
  visible_attr.set_b(true);
  model_attrs["_dflow_visible_device_enable"] = visible_attr;
  std::string batch_model_path = CreateBatchModels(2, "flow_func", model_attrs, true, {});
  // not support -1
  auto phy_device_id_bk = GlobalConfig::Instance().GetPhyDeviceId();
  GlobalConfig::Instance().SetPhyDeviceId(-1);
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);

  // invalid type
  visible_attr.set_s("true");
  model_attrs["_dflow_visible_device_enable"] = visible_attr;
  batch_model_path = CreateBatchModels(2, "flow_func", model_attrs, true, {});
  models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
  GlobalConfig::Instance().SetPhyDeviceId(phy_device_id_bk);
}

TEST_F(FlowFuncModelSTest, set_data_flow_scope_failed) {
  std::map<AscendString, ff::udf::AttrValue> attr_map;
  ff::udf::AttrValue visible_attr;

  // invalid type
  visible_attr.set_b(true);
  attr_map["_dflow_data_flow_scope"] = visible_attr;
  std::string batch_model_path = CreateBatchModels(2, "flow_func", attr_map, true, {});
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
}
}  // namespace FlowFunc