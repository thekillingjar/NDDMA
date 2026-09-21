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
#include "ascend_hal.h"
#include "fstream"
#include "mmpa/mmpa_api.h"
#define private public
#include "model/flow_func_model.h"
#include "config/global_config.h"
#undef private
#include "ff_udf_def.pb.h"
#include "ff_deployer.pb.h"
#include "flow_func/mbuf_flow_msg.h"

extern int32_t FlowFuncTestMain(int32_t argc, char *argv[]);
namespace FlowFunc {
namespace {
const std::string g_udfModelPath = "./udf_models/";
struct MbufImpl {
  uint32_t mbuf_size;
  uint8_t reserve_head[256];
  RuntimeTensorDesc tensor_desc;
  uint8_t data[1024];
};
int halMbufAllocStub(uint64_t size, Mbuf **mbuf) {
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  memset_s(mbuf_impl, sizeof(MbufImpl), 0, sizeof(MbufImpl));
  mbuf_impl->mbuf_size = size;
  *mbuf = reinterpret_cast<Mbuf *>(mbuf_impl);
  return DRV_ERROR_NONE;
}
int halMbufFreeStub(Mbuf *mbuf) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  delete mbuf_impl;
  return DRV_ERROR_NONE;
}
int halMbufGetBuffAddrStub(Mbuf *mbuf, void **buf) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  *buf = &(mbuf_impl->tensor_desc);
  return DRV_ERROR_NONE;
}
int halMbufGetBuffSizeStub(Mbuf *mbuf, uint64_t *total_size) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  *total_size = mbuf_impl->mbuf_size;
  return DRV_ERROR_NONE;
}
int halMbufSetDataLenStub(Mbuf *mbuf, uint64_t len) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  mbuf_impl->mbuf_size = len;
  return DRV_ERROR_NONE;
}
int halMbufGetPrivInfoStub(Mbuf *mbuf, void **priv, unsigned int *size) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  // start from first byte.
  *priv = &mbuf_impl->reserve_head;
  *size = 256;
  return DRV_ERROR_NONE;
}

int halMbufGetDataLenStub(Mbuf *mbuf, uint64_t *len) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  *len = mbuf_impl->mbuf_size;
  return DRV_ERROR_NONE;
}

int halMbufAllocExStub(uint64_t size, unsigned int align, unsigned long flag, int grp_id, Mbuf **mbuf) {
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  memset_s(mbuf_impl, sizeof(MbufImpl), 0, sizeof(MbufImpl));
  mbuf_impl->mbuf_size = size;
  *mbuf = reinterpret_cast<Mbuf *>(mbuf_impl);
  return DRV_ERROR_NONE;
}

drvError_t halQueueDeQueueStub(unsigned int dev_id, unsigned int qid, void **mbuf) {
  Mbuf *buf = nullptr;
  (void)halMbufAllocExStub(sizeof(RuntimeTensorDesc) + 1024, 0, 0, 0, &buf);
  *mbuf = buf;
  void *data_buf = nullptr;
  halMbufGetBuffAddrStub(buf, &data_buf);
  auto tensor_desc = static_cast<RuntimeTensorDesc *>(data_buf);
  tensor_desc->dtype = static_cast<int64_t>(TensorDataType::DT_INT8);
  tensor_desc->shape[0] = 1;
  tensor_desc->shape[1] = 1024;
  return DRV_ERROR_NONE;
}

drvError_t halQueueEnQueueStub(unsigned int dev_id, unsigned int qid, void *mbuf) {
  halMbufFreeStub(static_cast<Mbuf *>(mbuf));
  return DRV_ERROR_NONE;
}
}  // namespace
class FlowFuncModelUTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    mmMkdir(g_udfModelPath.c_str(), 0700);
    std::string cmd =
        "cd " + g_udfModelPath + "; mkdir release; cd release; touch libudf.so; tar -zcf release.tar.gz *";
    (void)system(cmd.c_str());
  }

  static void TearDownTestCase() {
    mmRmdir(g_udfModelPath.c_str());
  }

  virtual void SetUp() {
    back_is_on_device_ = GlobalConfig::Instance().IsOnDevice();
    MOCKER(halMbufFree).defaults().will(invoke(halMbufFreeStub));
    MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStub));
    MOCKER(halMbufGetBuffSize).defaults().will(invoke(halMbufGetBuffSizeStub));
    MOCKER(halMbufSetDataLen).defaults().will(invoke(halMbufSetDataLenStub));
    MOCKER(halMbufGetPrivInfo).defaults().will(invoke(halMbufGetPrivInfoStub));
    MOCKER(halMbufGetDataLen).defaults().will(invoke(halMbufGetDataLenStub));
    MOCKER(halMbufAlloc).defaults().will(invoke(halMbufAllocStub));
    MOCKER(halQueueEnQueue).defaults().will(invoke(halQueueEnQueueStub));
  }

  virtual void TearDown() {
    for (const auto &del_file : del_file_list) {
      unlink(del_file.c_str());
    }
    del_file_list.clear();
    GlobalMockObject::verify();
    GlobalConfig::on_device_ = back_is_on_device_;
  }

  std::string CreateBatchModelsWithAttr(int32_t model_num, std::map<std::string, ff::udf::AttrValue> attr_map,
                                        const bool is_built_in = false,
                                        const std::map<std::string, std::string> attrs = {},
                                        const std::string scope = "", const bool enable_exp = false,
                                        const std::vector<BufferConfigItem> &buf_cfg = {}, bool stream_input = false) {
    uint32_t group_idx = 0;
    ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;
    std::vector<std::string> model_list;
    model_list.reserve(model_num);
    for (int32_t i = 0; i < model_num; ++i) {
      ff::udf::UdfModelDef model_def;
      std::string flow_func_name("flow_func_");
      flow_func_name.append(std::to_string(i));
      CreateUdfModel(model_def, flow_func_name, attr_map, is_built_in, buf_cfg, stream_input);
      std::string path(g_udfModelPath + flow_func_name + ".proto");
      WriteProto(model_def, path);
      model_list.emplace_back(path);

      auto model = batch_load_model_req.add_models();
      model->set_model_path(path);
      model->set_scope(scope);
      model->set_enable_exception_catch(enable_exp);
      auto model_queues_attr = model->mutable_model_queues_attrs();
      auto input_queue_attr = model_queues_attr->add_input_queues_attrs();
      input_queue_attr->set_device_id(0);
      input_queue_attr->set_device_type(0);
      input_queue_attr->set_queue_id(group_idx++);

      auto output_queue_attr = model_queues_attr->add_output_queues_attrs();
      output_queue_attr->set_device_id(0);
      output_queue_attr->set_device_type(0);
      output_queue_attr->set_queue_id(group_idx++);
      auto model_attrs = model->mutable_attrs();
      for (const auto &attr : attrs) {
        (*model_attrs)[attr.first] = attr.second;
      }
      auto input_align_attr = model->mutable_input_align_attrs();
      input_align_attr->set_align_max_cache_num(100);
      input_align_attr->set_align_timeout(3000);
      input_align_attr->set_drop_when_not_align(false);
      del_file_list.emplace_back(path);
      if (enable_exp) {
        auto status_queues = model->mutable_status_queues();
        auto status_output_queue_attr = status_queues->add_output_queues_attrs();
        status_output_queue_attr->set_device_id(0);
        status_output_queue_attr->set_device_type(0);
        status_output_queue_attr->set_queue_id(group_idx++);
      }
    }

    std::string batch_model_path(g_udfModelPath + "batch_model.proto");
    WriteProto(batch_load_model_req, batch_model_path);
    del_file_list.emplace_back(batch_model_path);
    return batch_model_path;
  }

  std::string CreateBatchModels(int32_t model_num, const bool is_built_in = false,
                                const std::map<std::string, std::string> attrs = {}, const std::string scope = "",
                                const bool enable_exp = false, const std::vector<BufferConfigItem> &buf_cfg = {},
                                bool stream_input = false) {
    std::map<std::string, ff::udf::AttrValue> attr_map;
    ff::udf::AttrValue attr_str;
    attr_str.set_s("attrValue");
    attr_map["attrName"] = attr_str;
    ff::udf::AttrValue visible_attr;
    visible_attr.set_b(false);
    attr_map["_dflow_visible_device_enable"] = visible_attr;
    ff::udf::AttrValue visible_attr1;
    visible_attr1.set_s("test_scope");
    attr_map["_dflow_data_flow_scope"] = visible_attr1;
    ff::udf::AttrValue visible_attr2;
    visible_attr2.mutable_array()->add_s("test_invoked_socpe");
    attr_map["_dflow_data_flow_invoked_scopes"] = visible_attr2;
    return CreateBatchModelsWithAttr(model_num, attr_map, is_built_in, attrs, scope, enable_exp, buf_cfg, stream_input);
  }

  std::string CreateModelWithInvokedModel(int32_t model_num, int32_t queue_device_id = 0, int32_t queue_device_type = 0,
                                          bool duplicate_invoke_queue = false) {
    uint32_t queue_id = 0;
    ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;
    std::vector<std::string> model_list;
    model_list.reserve(model_num);
    for (int32_t i = 0; i < model_num; ++i) {
      ff::udf::UdfModelDef model_def;
      std::string flow_func_name("flow_func_");
      flow_func_name.append(std::to_string(i));
      std::map<std::string, ff::udf::AttrValue> attr_map;
      ff::udf::AttrValue attr_str;
      attr_str.set_s("attrValue");
      attr_map["attrName"] = attr_str;
      ff::udf::AttrValue visible_attr;
      visible_attr.set_b(false);
      attr_map["_dflow_visible_device_enable"] = visible_attr;
      CreateUdfModel(model_def, flow_func_name, attr_map, false);
      std::string path(g_udfModelPath + flow_func_name + ".proto");
      WriteProto(model_def, path);
      model_list.emplace_back(path);

      auto model = batch_load_model_req.add_models();
      model->set_model_path(path);
      auto model_queues_attr = model->mutable_model_queues_attrs();
      auto input_queue_attr = model_queues_attr->add_input_queues_attrs();
      input_queue_attr->set_device_id(queue_device_id);
      input_queue_attr->set_device_type(queue_device_type);
      input_queue_attr->set_queue_id(queue_id++);

      auto output_queue_attr = model_queues_attr->add_output_queues_attrs();
      output_queue_attr->set_device_id(queue_device_id);
      output_queue_attr->set_device_type(queue_device_type);
      output_queue_attr->set_queue_id(queue_id++);

      model->set_is_dynamic_sched(true);
      model->set_need_report_status(true);
      auto status_queues = model->mutable_status_queues();
      auto status_output_queue_attr = status_queues->add_output_queues_attrs();
      status_output_queue_attr->set_device_id(queue_device_id);
      status_output_queue_attr->set_device_type(queue_device_type);
      status_output_queue_attr->set_queue_id(queue_id++);

      auto invoked_model_map = model->mutable_invoked_model_queues_attrs();
      ff::deployer::ExecutorRequest_ModelQueuesAttrs invoke_model_queue_attrs;
      int32_t invoke_queue_id = 1001;
      if (!duplicate_invoke_queue) {
        for (int32_t j = 1; j <= 2; ++j) {
          auto invoke_input_attr = invoke_model_queue_attrs.add_input_queues_attrs();
          invoke_input_attr->set_device_id(queue_device_id);
          invoke_input_attr->set_device_type(queue_device_type);
          invoke_input_attr->set_queue_id(invoke_queue_id++);
        }
      } else {
        auto invoke_input_attr = invoke_model_queue_attrs.add_input_queues_attrs();
        invoke_input_attr->set_device_id(queue_device_id);
        invoke_input_attr->set_device_type(queue_device_type);
        invoke_input_attr->set_queue_id(1001);
        invoke_input_attr = invoke_model_queue_attrs.add_input_queues_attrs();
        invoke_input_attr->set_device_id(queue_device_id);
        invoke_input_attr->set_device_type(queue_device_type);
        invoke_input_attr->set_queue_id(1002);
        invoke_input_attr = invoke_model_queue_attrs.add_input_queues_attrs();
        invoke_input_attr->set_device_id(queue_device_id);
        invoke_input_attr->set_device_type(queue_device_type);
        invoke_input_attr->set_queue_id(1001);
      }
      auto invoke_output_attr = invoke_model_queue_attrs.add_output_queues_attrs();
      invoke_output_attr->set_device_id(queue_device_id);
      invoke_output_attr->set_device_type(queue_device_type);
      invoke_output_attr->set_queue_id(invoke_queue_id++);

      (*invoked_model_map)["modek_key"] = invoke_model_queue_attrs;
      del_file_list.emplace_back(path);
    }
    std::string batch_model_path(g_udfModelPath + "with_invoked_model.proto");
    WriteProto(batch_load_model_req, batch_model_path);
    del_file_list.emplace_back(batch_model_path);
    return batch_model_path;
  }

  void CreateUdfModelWithReleasePkg(ff::udf::UdfModelDef &model_def, const std::string &flow_func_name) {
    auto udf_def = model_def.add_udf_def();
    udf_def->set_name(flow_func_name);
    udf_def->set_func_name(flow_func_name);
    udf_def->set_bin_name("libudf.so");
  }

  std::string CreateReleasePkgModel(int32_t model_num, const std::string &udf_model_path) {
    uint32_t group_idx = 0;
    ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;
    std::vector<std::string> model_list;
    model_list.reserve(model_num);
    for (int32_t i = 0; i < model_num; ++i) {
      ff::udf::UdfModelDef model_def;
      std::string flow_func_name("flow_func_");
      flow_func_name.append(std::to_string(i));
      CreateUdfModelWithReleasePkg(model_def, flow_func_name);
      std::string path(udf_model_path + flow_func_name + ".proto");
      WriteProto(model_def, path);
      model_list.emplace_back(path);

      auto model = batch_load_model_req.add_models();
      model->set_model_path(path);
      auto model_queues_attr = model->mutable_model_queues_attrs();
      auto input_queue_attr = model_queues_attr->add_input_queues_attrs();
      input_queue_attr->set_device_id(0);
      input_queue_attr->set_device_type(0);
      input_queue_attr->set_queue_id(group_idx++);

      auto output_queue_attr = model_queues_attr->add_output_queues_attrs();
      output_queue_attr->set_device_id(0);
      output_queue_attr->set_device_type(0);
      output_queue_attr->set_queue_id(group_idx++);
      del_file_list.emplace_back(path);
      unlink((path + ".tar.gz").c_str());
      del_file_list.emplace_back(path + ".tar.gz");
    }
    std::string batch_model_path(udf_model_path + "with_release_pkg.proto");
    WriteProto(batch_load_model_req, batch_model_path);
    del_file_list.emplace_back(batch_model_path);
    return batch_model_path;
  }

  std::string CreateModelWithReleasePkg(int32_t model_num) {
    return CreateReleasePkgModel(model_num, g_udfModelPath);
  }

  void WriteProto(google::protobuf::MessageLite &proto_msg, const std::string &path) {
    std::ofstream out_stream;
    out_stream.open(path.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
    proto_msg.SerializeToOstream(&out_stream);
    out_stream.close();
  }

  void CreateUdfModel(ff::udf::UdfModelDef &model_def, const std::string &flow_func_name,
                      const std::map<std::string, ff::udf::AttrValue> &attr_map, bool is_built_in = false,
                      const std::vector<BufferConfigItem> &buf_cfg = {}, bool stream_input = false) {
    auto udf_def = model_def.add_udf_def();
    udf_def->set_name(flow_func_name);
    if (is_built_in) {
      udf_def->set_func_name("_BuiltIn_TimeBatch");
      udf_def->set_bin_name("libbuilt_in_flowfunc.so");
    } else {
      udf_def->set_func_name(flow_func_name);
      udf_def->set_bin_name(flow_func_name + ".so");
    }
    if (stream_input) {
      udf_def->add_stream_input_func_name(flow_func_name);
    }
    auto proto_attrs = udf_def->mutable_attrs();
    for (auto &attr : attr_map) {
      proto_attrs->insert({attr.first, attr.second});
    }
    ff::udf::AttrValue cpu_numvalue;
    cpu_numvalue.set_i(2);
    proto_attrs->insert({std::string("__cpu_num"), cpu_numvalue});
    for (auto &cfg : buf_cfg) {
      auto cfg_proto = udf_def->add_user_buf_cfg();
      cfg_proto->set_total_size(cfg.total_size);
      cfg_proto->set_blk_size(cfg.blk_size);
      cfg_proto->set_max_buf_size(cfg.max_buf_size);
      cfg_proto->set_page_type(cfg.page_type);
    }
  }

 protected:
  std::vector<std::string> del_file_list;
  bool back_is_on_device_ = false;
};

TEST_F(FlowFuncModelUTest, normal) {
  std::string batch_model_path = CreateBatchModels(2, false, {}, "world", true);
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
    uint32_t cpu_num = 0U;
    bool is_attr_get = false;
    EXPECT_EQ(model->GetCpuNumFromAttr(cpu_num, is_attr_get), FLOW_FUNC_SUCCESS);
    EXPECT_EQ(is_attr_get, true);
    EXPECT_EQ(cpu_num, 2U);
    EXPECT_EQ(model->GetScope(), "world");
    EXPECT_EQ(model->GetEnableRaiseException(), true);
  }
}

TEST_F(FlowFuncModelUTest, normal_with_stream_input) {
  std::string batch_model_path = CreateBatchModels(1, false, {}, "world", false, {}, true);
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 1);
  EXPECT_FALSE(models[0]->GetFlowFuncName().empty());
  EXPECT_FALSE(models[0]->GetLibPath().empty());
  EXPECT_FALSE(models[0]->GetNodeAttrMap().empty());
  EXPECT_FALSE(models[0]->GetInputQueues().empty());
  EXPECT_FALSE(models[0]->GetOutputQueues().empty());
  EXPECT_FALSE(models[0]->GetStreamInputFuncNames().empty());
  EXPECT_EQ(models[0]->model_esched_process_priority_, kUserUnsetESchedPriority);
  EXPECT_EQ(models[0]->model_esched_event_priority_, kUserUnsetESchedPriority);
  uint32_t cpu_num = 0U;
  bool is_attr_get = false;
  EXPECT_EQ(models[0]->GetCpuNumFromAttr(cpu_num, is_attr_get), FLOW_FUNC_SUCCESS);
  EXPECT_EQ(is_attr_get, true);
  EXPECT_EQ(cpu_num, 2U);
  EXPECT_EQ(models[0]->GetScope(), "world");
  EXPECT_EQ(models[0]->GetEnableRaiseException(), false);
}

TEST_F(FlowFuncModelUTest, normal_with_correct_buf_cfg) {
  BufferConfigItem cfg0 = {2 * 1024 * 1024, 256, 8 * 1024, "normal"};
  BufferConfigItem cfg1 = {32 * 1024 * 1024, 8 * 1024, 8 * 1024 * 1024, "normal"};
  BufferConfigItem cfg2 = {2 * 1024 * 1024, 256, 8 * 1024, "huge"};
  BufferConfigItem cfg3 = {52 * 1024 * 1024, 8 * 1024, 50 * 1024 * 1024, "huge"};
  BufferConfigItem cfg4 = {66 * 1024 * 1024, 8 * 1024, 64 * 1024 * 1024, "huge"};
  std::string batch_model_path = CreateBatchModels(2, false, {}, "world", true, {cfg0, cfg1, cfg2, cfg3, cfg4});
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
    uint32_t cpu_num = 0U;
    bool is_attr_get = false;
    EXPECT_EQ(model->GetCpuNumFromAttr(cpu_num, is_attr_get), FLOW_FUNC_SUCCESS);
    EXPECT_EQ(is_attr_get, true);
    EXPECT_EQ(cpu_num, 2U);
    EXPECT_EQ(model->GetScope(), "world");
    EXPECT_EQ(model->GetEnableRaiseException(), true);
  }
  std::vector<BufferConfigItem> buf_cfg = GlobalConfig::Instance().GetBufCfg();
  EXPECT_EQ(buf_cfg.size(), 5);
  EXPECT_EQ(buf_cfg[0].total_size, cfg0.total_size);
  EXPECT_EQ(buf_cfg[1].blk_size, cfg1.blk_size);
  EXPECT_EQ(buf_cfg[2].max_buf_size, cfg2.max_buf_size);
  EXPECT_EQ(buf_cfg[3].page_type, cfg3.page_type);
  EXPECT_EQ(buf_cfg[4].total_size, cfg4.total_size);
  GlobalConfig::Instance().SetBufCfg({});
}

TEST_F(FlowFuncModelUTest, normal_with_incorrect_buf_cfg_size_zero) {
  // 配置中size有0值
  BufferConfigItem cfg0 = {2 * 1024 * 1024, 0, 8 * 1024, "normal"};
  std::string batch_model_path = CreateBatchModels(2, false, {}, "world", true, {cfg0});
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
}

TEST_F(FlowFuncModelUTest, normal_with_incorrect_buf_cfg_total_blk_err) {
  // 配置中size有0值
  BufferConfigItem cfg0 = {2 * 1024 * 1024 + 4 * 1024, 8 * 1024, 16 * 1024, "normal"};
  std::string batch_model_path = CreateBatchModels(2, false, {}, "world", true, {cfg0});
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
}

TEST_F(FlowFuncModelUTest, normal_with_incorrect_buf_cfg_total_error) {
  // totalSize小于maxBufferSize
  BufferConfigItem cfg0 = {2 * 1024 * 1024, 256, 8 * 1024 * 1024, "normal"};
  std::string batch_model_path = CreateBatchModels(2, false, {}, "world", true, {cfg0});
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
}

TEST_F(FlowFuncModelUTest, normal_with_incorrect_buf_cfg_normal_not_sorted) {
  // 配置项没有按照totalSize排序
  BufferConfigItem cfg0 = {64 * 1024 * 1024, 256, 8 * 1024 * 1024, "normal"};
  BufferConfigItem cfg1 = {32 * 1024 * 1024, 8 * 1024, 4 * 1024 * 1024, "normal"};
  std::string batch_model_path = CreateBatchModels(2, false, {}, "world", true, {cfg0, cfg1});
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
}

TEST_F(FlowFuncModelUTest, normal_with_incorrect_buf_cfg_normal_total_error) {
  // normal类型totalSize不是4K的倍数
  BufferConfigItem cfg0 = {2 * 1024, 256, 256, "normal"};
  std::string batch_model_path = CreateBatchModels(2, false, {}, "world", true, {cfg0});
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
}

TEST_F(FlowFuncModelUTest, normal_with_incorrect_buf_cfg_huge_not_sorted) {
  // huge类型没有按照totalSize排序
  BufferConfigItem cfg0 = {64 * 1024 * 1024, 256, 8 * 1024, "normal"};
  BufferConfigItem cfg1 = {64 * 1024 * 1024, 256, 8 * 1024 * 1024, "huge"};
  BufferConfigItem cfg2 = {32 * 1024 * 1024, 8 * 1024, 4 * 1024 * 1024, "huge"};
  std::string batch_model_path = CreateBatchModels(2, false, {}, "world", true, {cfg0, cfg1, cfg2});
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
}

TEST_F(FlowFuncModelUTest, normal_with_incorrect_buf_cfg_huge_total_error) {
  // huge类型totalsize不是2M的倍数
  BufferConfigItem cfg0 = {64 * 1024 * 1024, 256, 8 * 1024, "normal"};
  BufferConfigItem cfg1 = {1 * 1024 * 1024, 256, 8 * 1024, "huge"};
  BufferConfigItem cfg2 = {32 * 1024 * 1024, 8 * 1024, 8 * 1024 * 1024, "huge"};
  std::string batch_model_path = CreateBatchModels(2, false, {}, "world", true, {cfg0, cfg1, cfg2});
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
}

TEST_F(FlowFuncModelUTest, normal_with_incorrect_buf_cfg_type_invalid) {
  // pageType不在有效范围内
  BufferConfigItem cfg0 = {64 * 1024 * 1024, 256, 8 * 1024, "dvpp"};
  std::string batch_model_path = CreateBatchModels(2, false, {}, "world", true, {cfg0});
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
}

TEST_F(FlowFuncModelUTest, normal_with_incorrect_buf_cfg_blk_error) {
  // blksize不是2^n
  BufferConfigItem cfg0 = {64 * 1024 * 1024, 258, 8 * 1024, "normal"};
  std::string batch_model_path = CreateBatchModels(2, false, {}, "world", true, {cfg0});
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
}

TEST_F(FlowFuncModelUTest, normal_with_incorrect_buf_cfg_blk_over_limit) {
  // blksize超过2M
  BufferConfigItem cfg0 = {64 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024, "normal"};
  std::string batch_model_path = CreateBatchModels(2, false, {}, "world", true, {cfg0});
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
}

TEST_F(FlowFuncModelUTest, with_invoked_models) {
  int32_t queue_device_id = GlobalConfig::Instance().GetDeviceId();
  int32_t queue_device_type = Common::kDeviceTypeCpu;
  std::string with_invoked_model_path = CreateModelWithInvokedModel(2, queue_device_id, queue_device_type);
  auto models = FlowFuncModel::ParseModels(with_invoked_model_path);
  EXPECT_EQ(models.size(), 2);
  for (auto &model : models) {
    EXPECT_FALSE(model->GetFlowFuncName().empty());
    EXPECT_FALSE(model->GetLibPath().empty());
    EXPECT_FALSE(model->GetNodeAttrMap().empty());
    EXPECT_FALSE(model->GetInputQueues().empty());
    for (const auto &input : model->GetInputQueues()) {
      EXPECT_EQ(input.device_id, queue_device_id);
      EXPECT_EQ(input.device_type, queue_device_type);
      EXPECT_FALSE(input.is_proxy_queue);
    }
    EXPECT_FALSE(model->GetOutputQueues().empty());
    for (const auto &output : model->GetOutputQueues()) {
      EXPECT_EQ(output.device_id, queue_device_id);
      EXPECT_EQ(output.device_type, queue_device_type);
      EXPECT_FALSE(output.is_proxy_queue);
    }
    EXPECT_EQ(model->model_esched_process_priority_, kUserUnsetESchedPriority);
    EXPECT_EQ(model->model_esched_event_priority_, kUserUnsetESchedPriority);
    EXPECT_FALSE(model->GetInvokedModelQueueInfos().empty());
    for (const auto &invoke_model_queue_info : model->GetInvokedModelQueueInfos()) {
      for (const auto &feed_queue : invoke_model_queue_info.second.feed_queue_infos) {
        EXPECT_EQ(feed_queue.device_id, queue_device_id);
        EXPECT_EQ(feed_queue.device_type, queue_device_type);
        EXPECT_FALSE(feed_queue.is_proxy_queue);
      }
      for (const auto &fetch_queue : invoke_model_queue_info.second.fetch_queue_infos) {
        EXPECT_EQ(fetch_queue.device_id, queue_device_id);
        EXPECT_EQ(fetch_queue.device_type, queue_device_type);
        EXPECT_FALSE(fetch_queue.is_proxy_queue);
      }
    }
    EXPECT_EQ(model->GetScope(), "");
    EXPECT_EQ(model->GetEnableRaiseException(), false);
  }
}

TEST_F(FlowFuncModelUTest, with_invoked_models_duplicate_feed_queues) {
  int32_t queue_device_id = GlobalConfig::Instance().GetDeviceId();
  int32_t queue_device_type = Common::kDeviceTypeCpu;
  std::string with_invoked_model_path = CreateModelWithInvokedModel(1, queue_device_id, queue_device_type, true);
  auto models = FlowFuncModel::ParseModels(with_invoked_model_path);
  EXPECT_EQ(models.size(), 1);
  for (auto &model : models) {
    EXPECT_FALSE(model->GetFlowFuncName().empty());
    EXPECT_FALSE(model->GetLibPath().empty());
    EXPECT_FALSE(model->GetNodeAttrMap().empty());
    EXPECT_FALSE(model->GetInputQueues().empty());
    for (const auto &input : model->GetInputQueues()) {
      EXPECT_EQ(input.device_id, queue_device_id);
      EXPECT_EQ(input.device_type, queue_device_type);
      EXPECT_FALSE(input.is_proxy_queue);
    }
    EXPECT_FALSE(model->GetOutputQueues().empty());
    for (const auto &output : model->GetOutputQueues()) {
      EXPECT_EQ(output.device_id, queue_device_id);
      EXPECT_EQ(output.device_type, queue_device_type);
      EXPECT_FALSE(output.is_proxy_queue);
    }
    EXPECT_EQ(model->model_esched_process_priority_, kUserUnsetESchedPriority);
    EXPECT_EQ(model->model_esched_event_priority_, kUserUnsetESchedPriority);
    EXPECT_FALSE(model->GetInvokedModelQueueInfos().empty());
    for (const auto &invoke_model_queue_info : model->GetInvokedModelQueueInfos()) {
      EXPECT_EQ(invoke_model_queue_info.second.feed_queue_infos.size(), 2);
    }
  }
}

TEST_F(FlowFuncModelUTest, invalid_as_device_use_cpu_queue) {
  GlobalConfig::Instance().on_device_ = true;
  int32_t current_device_id = GlobalConfig::Instance().GetDeviceId();
  int32_t queue_device_id = 4;
  int32_t queue_device_type = Common::kDeviceTypeCpu;
  std::string with_invoked_model_path = CreateModelWithInvokedModel(2, queue_device_id, queue_device_type);
  auto models = FlowFuncModel::ParseModels(with_invoked_model_path);
  EXPECT_EQ(models.size(), 0);
}

TEST_F(FlowFuncModelUTest, with_invoked_models_on_device) {
  GlobalConfig::Instance().on_device_ = true;
  int32_t current_device_id = GlobalConfig::Instance().GetDeviceId();
  int32_t queue_device_id = 4;
  int32_t queue_device_type = Common::kDeviceTypeNpu;
  std::string with_invoked_model_path = CreateModelWithInvokedModel(2, queue_device_id, queue_device_type);
  auto models = FlowFuncModel::ParseModels(with_invoked_model_path);
  EXPECT_EQ(models.size(), 2);
  for (auto &model : models) {
    EXPECT_FALSE(model->GetFlowFuncName().empty());
    EXPECT_FALSE(model->GetLibPath().empty());
    EXPECT_FALSE(model->GetNodeAttrMap().empty());
    EXPECT_FALSE(model->GetInputQueues().empty());
    for (const auto &input : model->GetInputQueues()) {
      EXPECT_EQ(input.device_id, current_device_id);
      EXPECT_EQ(input.device_type, queue_device_type);
      EXPECT_FALSE(input.is_proxy_queue);
    }
    EXPECT_FALSE(model->GetOutputQueues().empty());
    for (const auto &output : model->GetOutputQueues()) {
      EXPECT_EQ(output.device_id, current_device_id);
      EXPECT_EQ(output.device_type, queue_device_type);
      EXPECT_FALSE(output.is_proxy_queue);
    }
    EXPECT_EQ(model->model_esched_process_priority_, kUserUnsetESchedPriority);
    EXPECT_EQ(model->model_esched_event_priority_, kUserUnsetESchedPriority);
    EXPECT_FALSE(model->GetInvokedModelQueueInfos().empty());
    for (const auto &invoke_model_queue_info : model->GetInvokedModelQueueInfos()) {
      for (const auto &feed_queue : invoke_model_queue_info.second.feed_queue_infos) {
        EXPECT_EQ(feed_queue.device_id, current_device_id);
        EXPECT_EQ(feed_queue.device_type, queue_device_type);
        EXPECT_FALSE(feed_queue.is_proxy_queue);
      }
      for (const auto &fetch_queue : invoke_model_queue_info.second.fetch_queue_infos) {
        EXPECT_EQ(fetch_queue.device_id, current_device_id);
        EXPECT_EQ(fetch_queue.device_type, queue_device_type);
        EXPECT_FALSE(fetch_queue.is_proxy_queue);
      }
    }
  }
}

TEST_F(FlowFuncModelUTest, with_invoked_models_with_proxy) {
  int32_t queue_device_id = 2;
  int32_t queue_device_type = Common::kDeviceTypeNpu;
  std::string with_invoked_model_path = CreateModelWithInvokedModel(2, queue_device_id, queue_device_type);
  auto models = FlowFuncModel::ParseModels(with_invoked_model_path);
  EXPECT_EQ(models.size(), 2);
  for (auto &model : models) {
    EXPECT_FALSE(model->GetFlowFuncName().empty());
    EXPECT_FALSE(model->GetLibPath().empty());
    EXPECT_FALSE(model->GetNodeAttrMap().empty());
    EXPECT_FALSE(model->GetInputQueues().empty());
    for (const auto &input : model->GetInputQueues()) {
      EXPECT_EQ(input.device_id, queue_device_id);
      EXPECT_EQ(input.device_type, queue_device_type);
      EXPECT_TRUE(input.is_proxy_queue);
    }
    EXPECT_FALSE(model->GetOutputQueues().empty());
    for (const auto &output : model->GetOutputQueues()) {
      EXPECT_EQ(output.device_id, queue_device_id);
      EXPECT_EQ(output.device_type, queue_device_type);
      EXPECT_TRUE(output.is_proxy_queue);
    }
    EXPECT_EQ(model->model_esched_process_priority_, kUserUnsetESchedPriority);
    EXPECT_EQ(model->model_esched_event_priority_, kUserUnsetESchedPriority);
    EXPECT_FALSE(model->GetInvokedModelQueueInfos().empty());
    for (const auto &invoke_model_queue_info : model->GetInvokedModelQueueInfos()) {
      for (const auto &feed_queue : invoke_model_queue_info.second.feed_queue_infos) {
        EXPECT_EQ(feed_queue.device_id, queue_device_id);
        EXPECT_EQ(feed_queue.device_type, queue_device_type);
        EXPECT_TRUE(feed_queue.is_proxy_queue);
      }
      for (const auto &fetch_queue : invoke_model_queue_info.second.fetch_queue_infos) {
        EXPECT_EQ(fetch_queue.device_id, queue_device_id);
        EXPECT_EQ(fetch_queue.device_type, queue_device_type);
        EXPECT_TRUE(fetch_queue.is_proxy_queue);
      }
    }
  }
}

TEST_F(FlowFuncModelUTest, with_release_pkg) {
  std::string with_release_pkg_model_path = CreateModelWithReleasePkg(1);
  auto models = FlowFuncModel::ParseModels(with_release_pkg_model_path);
  EXPECT_EQ(models.size(), 1);
  EXPECT_EQ(models[0]->GetWorkPath(), "./udf_models/flow_func_0.proto_dir");
  EXPECT_FALSE(models[0]->GetFlowFuncName().empty());
  EXPECT_EQ(models[0]->GetLibPath(), "./udf_models/flow_func_0.proto_dir/libudf.so");
  EXPECT_FALSE(models[0]->GetInputQueues().empty());
  EXPECT_FALSE(models[0]->GetOutputQueues().empty());
}

TEST_F(FlowFuncModelUTest, normal_built_in) {
  std::map<std::string, std::string> attrs;
  attrs["_eschedProcessPriority"] = "0";
  attrs["_eschedEventPriority"] = "2";
  std::string batch_model_path = CreateBatchModels(2, true, attrs);
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
  }
}

TEST_F(FlowFuncModelUTest, test_from_main) {
  std::string batch_model_path = CreateBatchModels(1);

  char process_name[] = "flow_func_executor";
  char param_device_id_ok[] = "--device_id=1";
  char param_req_queue_id[] = "--req_queue_id=100";
  char param_rsp_queue_id[] = "--rsp_queue_id=101";
  char param_grp_name_ok[] = "--group_name=Grp1";
  char param_load_path_ok[1024] = "--load_path=";
  strcat(param_load_path_ok, batch_model_path.c_str());

  char *argv[] = {process_name,      param_device_id_ok, param_load_path_ok,
                  param_grp_name_ok, param_req_queue_id, param_rsp_queue_id};
  int32_t argc = 6;
  MOCKER(halQueueDeQueue).defaults().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_EQ(ret, 0);
}

TEST_F(FlowFuncModelUTest, file_not_exit) {
  auto models = FlowFuncModel::ParseModels("./not_exit_path");
  EXPECT_TRUE(models.empty());
}

TEST_F(FlowFuncModelUTest, parse_process_priority_failed) {
  std::map<std::string, std::string> attrs;
  attrs["_eschedProcessPriority"] = std::to_string(INT64_MAX);
  attrs["_eschedEventPriority"] = "2";
  std::string batch_model_path = CreateBatchModels(2, false, attrs);
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
}

TEST_F(FlowFuncModelUTest, parse_event_priority_failed) {
  std::map<std::string, std::string> attrs;
  attrs["_eschedProcessPriority"] = "0";
  attrs["_eschedEventPriority"] = "afdasdfa;lk";
  std::string batch_model_path = CreateBatchModels(2, false, attrs);
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
}

TEST_F(FlowFuncModelUTest, visible_device_enable_success) {
  std::map<std::string, ff::udf::AttrValue> attr_map;
  ff::udf::AttrValue visible_attr;
  visible_attr.set_b(true);
  attr_map["_dflow_visible_device_enable"] = visible_attr;
  std::string batch_model_path = CreateBatchModelsWithAttr(2, attr_map, true);
  auto phy_device_id_bk = GlobalConfig::Instance().GetPhyDeviceId();
  GlobalConfig::Instance().SetPhyDeviceId(0);
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 2);
  auto visible_devices = getenv("ASCEND_RT_VISIBLE_DEVICES");
  EXPECT_NE(visible_devices, nullptr);
  unsetenv("ASCEND_RT_VISIBLE_DEVICES");
  GlobalConfig::Instance().SetPhyDeviceId(phy_device_id_bk);
}

TEST_F(FlowFuncModelUTest, visible_device_enable_failed) {
  std::map<std::string, ff::udf::AttrValue> attr_map;
  ff::udf::AttrValue visible_attr;
  visible_attr.set_b(true);
  attr_map["_dflow_visible_device_enable"] = visible_attr;
  std::string batch_model_path = CreateBatchModelsWithAttr(2, attr_map, true);
  // not support -1
  auto phy_device_id_bk = GlobalConfig::Instance().GetPhyDeviceId();
  GlobalConfig::Instance().SetPhyDeviceId(-1);
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);

  // invalid type
  visible_attr.set_s("true");
  attr_map["_dflow_visible_device_enable"] = visible_attr;
  batch_model_path = CreateBatchModelsWithAttr(2, attr_map, true);
  models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
  GlobalConfig::Instance().SetPhyDeviceId(phy_device_id_bk);
}

TEST_F(FlowFuncModelUTest, set_data_flow_scope_failed) {
  std::map<std::string, ff::udf::AttrValue> attr_map;
  ff::udf::AttrValue visible_attr;

  // invalid type
  visible_attr.set_b(true);
  attr_map["_dflow_data_flow_scope"] = visible_attr;
  std::string batch_model_path = CreateBatchModelsWithAttr(2, attr_map, true);
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
}

TEST_F(FlowFuncModelUTest, set_data_flow_invoked_scopes_failed) {
  std::map<std::string, ff::udf::AttrValue> attr_map;
  ff::udf::AttrValue visible_attr;

  // invalid type
  visible_attr.set_s("scope");
  attr_map["_dflow_data_flow_invoked_scopes"] = visible_attr;
  std::string batch_model_path = CreateBatchModelsWithAttr(2, attr_map, true);
  auto models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(models.size(), 0);
}
}  // namespace FlowFunc