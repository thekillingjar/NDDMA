/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "udf_utils.h"
#include <fstream>
#include "mmpa/mmpa_api.h"
#include "stub/udf_stub.h"

namespace FlowFunc {
namespace {
const std::string g_udfModelPath = "./udf_models" + std::to_string(getpid()) + "/";
}

const std::string &GetUdfModelPath() {
  return g_udfModelPath;
}

void CreateModelDir() {
  mmMkdir(g_udfModelPath.c_str(), 0700);
}

void DeleteModelDir() {
  mmRmdir(g_udfModelPath.c_str());
}

std::string CreateBatchModels(int32_t model_num, const std::string &flow_func_name,
                              const std::map<AscendString, ff::udf::AttrValue> &attr_map, bool is_built_in,
                              const std::map<std::string, std::string> &model_attrs, int32_t device_type,
                              const std::vector<BufferConfigItem> &buf_cfg) {
  uint32_t queue_idx = 0;
  ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;
  for (int32_t i = 0; i < model_num; ++i) {
    ff::udf::UdfModelDef model_def;
    CreateUdfModel(model_def, flow_func_name, flow_func_name, attr_map, is_built_in, {}, {}, buf_cfg);
    std::string model_file_name(flow_func_name + "_" + std::to_string(i) + ".proto");
    auto proto_path = WriteProto(model_def, model_file_name);
    uint32_t input_qid = queue_idx++;
    uint32_t output_qid = queue_idx++;
    AddModelToBatchModel(batch_load_model_req, proto_path, {input_qid}, {output_qid}, model_attrs, device_type);
  }
  return WriteProto(batch_load_model_req, "batch_model.proto");
}

void CreateUdfModelWithReleasePkg(ff::udf::UdfModelDef &model_def, const std::string &flow_func_name) {
  auto udf_def = model_def.add_udf_def();
  udf_def->set_name(flow_func_name);
  udf_def->set_func_name(flow_func_name);
  udf_def->set_bin_name("libudf.so");
}

std::string CreateModelsWithReleasePkg(int32_t model_num, const std::string &flow_func_name,
                                       const std::string &model_path) {
  uint32_t queue_idx = 0;
  ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;
  for (int32_t i = 0; i < model_num; ++i) {
    ff::udf::UdfModelDef model_def;
    CreateUdfModelWithReleasePkg(model_def, flow_func_name);
    std::string model_file_name(flow_func_name + "_" + std::to_string(i) + ".proto");
    auto proto_path = WriteProto(model_def, model_file_name, model_path);
    uint32_t input_qid = queue_idx++;
    uint32_t output_qid = queue_idx++;
    // std::map<std::string, std::string> attrs;
    AddModelToBatchModel(batch_load_model_req, proto_path, {input_qid}, {output_qid}, {});
  }
  return WriteProto(batch_load_model_req, "batch_model.proto", model_path);
}

void AddModelToBatchModel(ff::deployer::ExecutorRequest_BatchLoadModelMessage &batch_load_model_req,
                          const std::string &model_path, const std::vector<uint32_t> &input_queues,
                          const std::vector<uint32_t> &output_queues,
                          const std::map<std::string, std::string> &model_attrs, int32_t device_type,
                          const bool report_status, const bool &catch_exceptions,
                          const std::vector<uint32_t> status_output_queues,
                          const std::vector<uint32_t> logic_input_queues) {
  auto model = batch_load_model_req.add_models();
  model->set_model_path(model_path);
  auto queue_attrs = model->mutable_model_queues_attrs();
  int32_t index = 0;
  for (auto qid : input_queues) {
    auto input_queue = queue_attrs->add_input_queues_attrs();
    input_queue->set_queue_id(qid);
    input_queue->set_device_type(device_type);
    input_queue->set_device_id(0);
    if (report_status) {
      input_queue->set_global_logic_id(logic_input_queues[index]);
      index++;
    }
  }
  for (auto qid : output_queues) {
    auto output_queue = queue_attrs->add_output_queues_attrs();
    output_queue->set_queue_id(qid);
    output_queue->set_device_type(device_type);
    output_queue->set_device_id(0);
  }
  if (report_status || catch_exceptions) {
    model->set_is_dynamic_sched(true);
    model->set_need_report_status(report_status);
    model->set_enable_exception_catch(catch_exceptions);
    auto status_queues = model->mutable_status_queues();
    for (const auto &status_output_queue : status_output_queues) {
      auto status_output_queue_attrs = status_queues->add_output_queues_attrs();
      status_output_queue_attrs->set_queue_id(status_output_queue);
      status_output_queue_attrs->set_device_type(device_type);
      status_output_queue_attrs->set_device_id(0);
    }
  }
  auto attrs = model->mutable_attrs();
  for (const auto &attr : model_attrs) {
    (*attrs)[attr.first] = attr.second;
  }
}
std::string WriteProto(google::protobuf::MessageLite &proto_msg, const std::string &proto_name,
                       const std::string &proto_path) {
  std::ofstream out_stream;
  std::string path(proto_path);
  if (!proto_path.empty() && proto_path.at(proto_path.length() - 1) != '/') {
    path.append("/");
  }
  path.append(proto_name);
  out_stream.open(path.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
  proto_msg.SerializeToOstream(&out_stream);
  out_stream.close();
  return path;
}
std::string WriteProto(google::protobuf::MessageLite &proto_msg, const std::string &proto_name) {
  return WriteProto(proto_msg, proto_name, g_udfModelPath);
}

void CreateUdfModel(ff::udf::UdfModelDef &model_def, const std::string &flow_func_name, const std::string &pp_name,
                    const std::map<AscendString, ff::udf::AttrValue> &attr_map, bool is_built_in,
                    const std::map<std::string, std::vector<uint32_t>> &func_input_maps,
                    const std::map<std::string, std::vector<uint32_t>> &func_output_maps,
                    const std::vector<BufferConfigItem> &buf_cfg, bool stream_input) {
  auto udf_def = model_def.add_udf_def();
  udf_def->set_name(pp_name);
  if (is_built_in) {
    udf_def->set_bin_name("libbuilt_in_flowfunc.so");
  } else {
    udf_def->set_bin_name(flow_func_name + ".so");
  }
  udf_def->set_func_name(flow_func_name);
  if (stream_input) {
    udf_def->add_stream_input_func_name(flow_func_name);
  }
  auto proto_attrs = udf_def->mutable_attrs();
  for (auto &attr : attr_map) {
    proto_attrs->insert({attr.first.GetString(), attr.second});
  }

  if (!func_input_maps.empty()) {
    for (auto &map : func_input_maps) {
      auto input_map = udf_def->mutable_func_inputs_map();
      auto &inputs = (*input_map)[map.first];
      for (auto index : map.second) {
        inputs.add_num(index);
      }
    }
  }
  if (!func_output_maps.empty()) {
    for (auto &map : func_output_maps) {
      auto output_map = udf_def->mutable_func_outputs_map();
      for (auto index : map.second) {
        (*output_map)[map.first].add_num(index);
      }
    }
  }
  for (auto &cfg : buf_cfg) {
    auto cfg_proto = udf_def->add_user_buf_cfg();
    cfg_proto->set_total_size(cfg.total_size);
    cfg_proto->set_blk_size(cfg.blk_size);
    cfg_proto->set_max_buf_size(cfg.max_buf_size);
    cfg_proto->set_page_type(cfg.page_type);
  }
}

void SetMbufRetCode(Mbuf *mbuf, int32_t ret_code) {
  MbufStImpl *impl = (MbufStImpl *)mbuf;
  impl->head_msg.ret_code = ret_code;
}

void SetMbufTransId(Mbuf *mbuf, uint64_t trans_id) {
  MbufStImpl *impl = (MbufStImpl *)mbuf;
  impl->head_msg.trans_id = trans_id;
}

void SetMbufDataLabel(Mbuf *mbuf, uint32_t data_label) {
  MbufStImpl *impl = (MbufStImpl *)mbuf;
  impl->head_msg.data_label = data_label;
}

uint64_t GetMbufTransId(Mbuf *mbuf) {
  MbufStImpl *impl = (MbufStImpl *)mbuf;
  return impl->head_msg.trans_id;
}

void SetMbufTensorDesc(Mbuf *mbuf, const std::vector<int64_t> &shape, TensorDataType data_type) {
  MbufStImpl *impl = (MbufStImpl *)mbuf;
  impl->tensor_desc.dtype = (int64_t)data_type;
  impl->tensor_desc.shape[0] = shape.size();
  for (size_t i = 0; i < shape.size(); ++i) {
    impl->tensor_desc.shape[i + 1] = shape[i];
  }
}

void SetMbufHeadMsg(Mbuf *mbuf, const MbufHeadMsg &head_msg) {
  MbufStImpl *impl = (MbufStImpl *)mbuf;
  impl->head_msg = head_msg;
}

const MbufHeadMsg *GetMbufHeadMsg(Mbuf *mbuf) {
  MbufStImpl *impl = (MbufStImpl *)mbuf;
  return &impl->head_msg;
}
}  // namespace FlowFunc
