/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_ST_UDF_UTILS_H
#define UDF_ST_UDF_UTILS_H

#include <string>
#include <map>
#include "gtest/gtest.h"
#include "ff_udf_def.pb.h"
#include "ff_deployer.pb.h"
#include "common/data_utils.h"
#include "common/common_define.h"
#include "stub/udf_stub.h"
#include "ascend_hal.h"
#include "flow_func/tensor_data_type.h"
#include "flow_func/ascend_string.h"

namespace FlowFunc {
const std::string &GetUdfModelPath();

void CreateModelDir();

void DeleteModelDir();

std::string CreateBatchModels(int32_t model_num, const std::string &flow_func_name,
                              const std::map<AscendString, ff::udf::AttrValue> &attr_map, bool is_built_in = false,
                              const std::map<std::string, std::string> &model_attrs = {},
                              int32_t device_type = Common::kDeviceTypeCpu,
                              const std::vector<BufferConfigItem> &buf_cfg = {});

std::string CreateModelsWithReleasePkg(int32_t model_num, const std::string &flow_func_name,
                                       const std::string &model_path);

std::string WriteProto(google::protobuf::MessageLite &proto_msg, const std::string &proto_name);

std::string WriteProto(google::protobuf::MessageLite &proto_msg, const std::string &proto_name,
                       const std::string &proto_path);

void AddModelToBatchModel(ff::deployer::ExecutorRequest_BatchLoadModelMessage &batch_load_model_req,
                          const std::string &model_path, const std::vector<uint32_t> &input_queues,
                          const std::vector<uint32_t> &output_queues,
                          const std::map<std::string, std::string> &model_attrs = {},
                          int32_t device_type = Common::kDeviceTypeCpu, const bool report_status = false,
                          const bool &catch_exceptions = false, const std::vector<uint32_t> status_output_queues = {},
                          const std::vector<uint32_t> logic_input_queues = {});

void CreateUdfModel(ff::udf::UdfModelDef &model_def, const std::string &flow_func_name, const std::string &pp_name,
                    const std::map<AscendString, ff::udf::AttrValue> &attr_map, bool is_built_in = false,
                    const std::map<std::string, std::vector<uint32_t>> &func_input_maps = {},
                    const std::map<std::string, std::vector<uint32_t>> &func_output_maps = {},
                    const std::vector<BufferConfigItem> &buf_cfg = {}, bool stream_input = false);

void SetMbufRetCode(Mbuf *mbuf, int32_t ret_code);
void SetMbufTransId(Mbuf *mbuf, uint64_t trans_id);
void SetMbufDataLabel(Mbuf *mbuf, uint32_t data_label);

uint64_t GetMbufTransId(Mbuf *mbuf);

void SetMbufTensorDesc(Mbuf *mbuf, const std::vector<int64_t> &shape, TensorDataType data_type);

void SetMbufHeadMsg(Mbuf *mbuf, const MbufHeadMsg &head_msg);

const MbufHeadMsg *GetMbufHeadMsg(Mbuf *mbuf);

template <typename T>
void DataEnqueue(uint32_t queue_id, const std::vector<int64_t> &shape, TensorDataType data_type, T data,
                 uint64_t trans_id = 0, uint32_t data_label = 0) {
  uint32_t data_count = CalcElementCnt(shape);
  int64_t data_size = CalcDataSize(shape, data_type);
  Mbuf *mbuf = nullptr;
  auto drv_ret = halMbufAllocEx(sizeof(RuntimeTensorDesc) + data_size, 0, 0, 0, &mbuf);
  EXPECT_EQ(drv_ret, DRV_ERROR_NONE);
  SetMbufTensorDesc(mbuf, shape, data_type);
  SetMbufTransId(mbuf, trans_id);
  SetMbufDataLabel(mbuf, data_label);
  void *buf_data = nullptr;
  drv_ret = halMbufGetBuffAddr(mbuf, &buf_data);
  EXPECT_EQ(drv_ret, DRV_ERROR_NONE);
  T *raw_data = (T *)((uint8_t *)buf_data + sizeof(RuntimeTensorDesc));
  for (uint32_t i = 0; i < data_count; ++i) {
    raw_data[i] = data;
  }
  constexpr uint32_t max_wait_second = 30;
  uint32_t wait_second = 0;
  bool need_release = true;
  while (wait_second < max_wait_second) {
    drv_ret = halQueueEnQueue(0, queue_id, mbuf);
    if (drv_ret == DRV_ERROR_NONE) {
      need_release = false;
      break;
    } else if (drv_ret == DRV_ERROR_QUEUE_FULL) {
      sleep(1);
      wait_second++;
      continue;
    } else {
      break;
    }
  }
  if (need_release) {
    halMbufFree(mbuf);
  }
}

template <typename T>
void DataEnqueue(uint32_t queue_id, const std::vector<int64_t> &shape, TensorDataType data_type,
                 const MbufHeadMsg &head_msg, const T *data) {
  uint32_t data_count = CalcElementCnt(shape);
  int64_t data_size = CalcDataSize(shape, data_type);
  Mbuf *mbuf = nullptr;
  auto drv_ret = halMbufAllocEx(sizeof(RuntimeTensorDesc) + data_size, 0, 0, 0, &mbuf);
  EXPECT_EQ(drv_ret, DRV_ERROR_NONE);
  SetMbufHeadMsg(mbuf, head_msg);
  SetMbufTensorDesc(mbuf, shape, data_type);
  void *buf_data = nullptr;
  drv_ret = halMbufGetBuffAddr(mbuf, &buf_data);
  EXPECT_EQ(drv_ret, DRV_ERROR_NONE);
  T *raw_data = (T *)((uint8_t *)buf_data + sizeof(RuntimeTensorDesc));
  for (uint32_t i = 0; i < data_count; ++i) {
    raw_data[i] = data[i];
  }
  constexpr uint32_t max_wait_second = 30;
  uint32_t wait_second = 0;
  bool need_release = true;
  while (wait_second < max_wait_second) {
    drv_ret = halQueueEnQueue(0, queue_id, mbuf);
    if (drv_ret == DRV_ERROR_NONE) {
      need_release = false;
      break;
    } else if (drv_ret == DRV_ERROR_QUEUE_FULL) {
      sleep(1);
      wait_second++;
      continue;
    } else {
      break;
    }
  }
  if (need_release) {
    halMbufFree(mbuf);
  }
}

template <typename T>
void EnqueueControlMsg(uint32_t queue_id, const T &control_msg) {
  Mbuf *mbuf = nullptr;
  auto drv_ret = halMbufAllocEx(control_msg.ByteSizeLong(), 0, 0, 0, &mbuf);
  EXPECT_EQ(drv_ret, DRV_ERROR_NONE);
  void *buf_data = nullptr;
  drv_ret = halMbufGetBuffAddr(mbuf, &buf_data);
  EXPECT_EQ(drv_ret, DRV_ERROR_NONE);
  drv_ret = halMbufSetDataLen(mbuf, control_msg.ByteSizeLong());
  EXPECT_EQ(drv_ret, DRV_ERROR_NONE);
  EXPECT_EQ(control_msg.SerializeToArray(buf_data, control_msg.ByteSizeLong()), true);
  constexpr uint32_t max_wait_second = 30;
  uint32_t wait_second = 0;
  bool need_release = true;
  drv_ret = halQueueEnQueue(0, queue_id, mbuf);
  if (drv_ret != DRV_ERROR_NONE) {
    halMbufFree(mbuf);
  }
  EXPECT_EQ(drv_ret, DRV_ERROR_NONE);
}

template <typename T>
void CheckMbufData(Mbuf *mbuf, const std::vector<int64_t> &shape, TensorDataType data_type, T *data,
                   uint32_t data_num) {
  uint32_t data_count = CalcElementCnt(shape);
  void *out_buf_data = nullptr;
  auto drv_ret = halMbufGetBuffAddr(mbuf, &out_buf_data);
  EXPECT_EQ(drv_ret, DRV_ERROR_NONE);
  uint64_t out_buff_size = 0;
  drv_ret = halMbufGetBuffSize(mbuf, &out_buff_size);
  EXPECT_EQ(drv_ret, DRV_ERROR_NONE);
  int64_t expect_out_data_size = CalcDataSize(shape, data_type) + sizeof(RuntimeTensorDesc);
  EXPECT_EQ(out_buff_size, expect_out_data_size);
  RuntimeTensorDesc *tensor_desc = (RuntimeTensorDesc *)out_buf_data;
  EXPECT_EQ(tensor_desc->shape[0], shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    EXPECT_EQ(tensor_desc->shape[i + 1], shape[i]);
  }
  T *out_buf_raw_data = reinterpret_cast<T *>(static_cast<uint8_t *>(out_buf_data) + sizeof(RuntimeTensorDesc));
  for (uint32_t i = 0; i < data_count; ++i) {
    EXPECT_EQ(out_buf_raw_data[i], data[i]);
  }
}

template <typename T>
void CheckMbufData(Mbuf *mbuf, T *data, uint32_t data_num) {
  void *out_buf_data = nullptr;
  auto drv_ret = halMbufGetBuffAddr(mbuf, &out_buf_data);
  EXPECT_EQ(drv_ret, DRV_ERROR_NONE);
  uint64_t out_buff_size = 0;
  drv_ret = halMbufGetBuffSize(mbuf, &out_buff_size);
  EXPECT_EQ(drv_ret, DRV_ERROR_NONE);
  const auto head_msg = GetMbufHeadMsg(mbuf);
  int64_t data_offset =
      (head_msg->msg_type == static_cast<uint16_t>(MsgType::MSG_TYPE_TENSOR_DATA)) ? sizeof(RuntimeTensorDesc) : 0;
  int64_t expect_out_data_size = sizeof(T) * data_num + data_offset;
  EXPECT_EQ(out_buff_size, expect_out_data_size);
  RuntimeTensorDesc *tensor_desc = (RuntimeTensorDesc *)out_buf_data;
  T *out_buf_raw_data = reinterpret_cast<T *>(static_cast<uint8_t *>(out_buf_data) + data_offset);
  for (uint32_t i = 0; i < data_num; ++i) {
    EXPECT_EQ(out_buf_raw_data[i], data[i]);
  }
}

template <typename T>
void GetMbufDataAndLen(Mbuf *&mbuf, const T *&data, uint32_t &data_num) {
  MbufStImpl *impl = (MbufStImpl *)mbuf;
  data = reinterpret_cast<const T *>(impl->data);
  data_num = (impl->mbuf_size - sizeof(RuntimeTensorDesc)) / sizeof(T);
}
}  // namespace FlowFunc
#endif  // UDF_ST_UDF_UTILS_H
