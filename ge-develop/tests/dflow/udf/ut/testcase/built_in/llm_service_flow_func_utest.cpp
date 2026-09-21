/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascend_hal.h"
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "securec.h"

#define private public

#include <dlfcn.h>
#include "entity/llm_comm_entity.h"
#include "entity/llm_comm_entity_mgr.h"
#include "flow_func/flow_func_params.h"
#include "flow_func/mbuf_flow_msg.h"
#include "llm_common/hccl_so_manager.h"
#include "llm_func/llm_service_flow_func.h"
#include "model/attr_value_impl.h"
#include "llm_common/cache_manager.h"

#undef private
#include "common/data_utils.h"
#include "entity/link_req_handler.h"

#include "flow_func/flow_func_run_context.h"
#include "hccl_stub.h"
#include "llm_common/kv_cache_manager.h"
#include "llm_common/hccl_proxy.h"

namespace FlowFunc {
namespace {
constexpr uint64_t kBatchSize = 2UL;
constexpr uint64_t kFirstDimForKvShape = 1UL;
constexpr uint64_t kSecondDimForKvShape = 256UL;
constexpr uint64_t kBlocksNum = 4UL;
constexpr uint64_t kBlockSize = 16UL;
constexpr uint64_t kHiddenSize = 16UL;
constexpr uint64_t kKvSize = 2UL;
inline uint64_t PtrToValue(const void *const ptr) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
}
inline void *ValueToPtr(const uint64_t value) {
  return reinterpret_cast<void *>(static_cast<uintptr_t>(value));
}

struct MbufImpl {
  uint32_t mbuf_size;
  uint8_t reserve_head[256 - sizeof(MbufHeadMsg)];
  MbufHeadMsg head_msg;
  RuntimeTensorDesc tensor_desc;
  uint8_t data[4096];
};

int halGrpQuerySuc(GroupQueryCmdType cmd, void *in_buff, unsigned int in_len, void *out_buff, unsigned int *out_len) {
  auto ptr = static_cast<GroupQueryOutput *>(out_buff);
  if (cmd == GRP_QUERY_GROUPS_OF_PROCESS) {
    *out_len = sizeof(ptr->grpQueryGroupsOfProcInfo[0]);
    std::string grp = "group";
    errno_t ret = strcpy_s(ptr->grpQueryGroupsOfProcInfo[0].groupName, grp.size() + 1, grp.data());
    if (ret != EOK) {
      return DRV_ERROR_INNER_ERR;
    }
    return DRV_ERROR_NONE;
  } else if (cmd == GRP_QUERY_GROUP_ADDR_INFO) {
    *out_len = sizeof(GrpQueryGroupAddrInfo);
    ptr->grpQueryGroupAddrInfo[0].addr = 0;
    ptr->grpQueryGroupAddrInfo[0].size = 1;
  }
  return DRV_ERROR_NONE;
}

int halGrpQueryFail(GroupQueryCmdType cmd, void *in_buff, unsigned int in_len, void *out_buff, unsigned int *out_len) {
  return DRV_ERROR_INNER_ERR;
}

int halGrpQueryFail2(GroupQueryCmdType cmd, void *in_buff, unsigned int in_len, void *out_buff, unsigned int *out_len) {
  if (cmd == GRP_QUERY_GROUPS_OF_PROCESS) {
    *out_len = 0;
  }
  return DRV_ERROR_NONE;
}

int halGrpQueryFail3(GroupQueryCmdType cmd, void *in_buff, unsigned int in_len, void *out_buff, unsigned int *out_len) {
  auto ptr = static_cast<GroupQueryOutput *>(out_buff);
  if (cmd == GRP_QUERY_GROUPS_OF_PROCESS) {
    *out_len = 2 * sizeof(ptr->grpQueryGroupsOfProcInfo[0]);
  }
  return DRV_ERROR_NONE;
}

int halMbufAllocExStub(uint64_t size, unsigned int align, unsigned long flag, int grp_id, Mbuf **mbuf) {
  if (size >= 32000000000) {
    return 1;
  }
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  memset_s(mbuf_impl, sizeof(MbufImpl), 0, sizeof(MbufImpl));
  mbuf_impl->mbuf_size = size;
  *mbuf = reinterpret_cast<Mbuf *>(mbuf_impl);
  return DRV_ERROR_NONE;
}

int halMbufAllocStub(uint64_t count, Mbuf **mbuf) {
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  mbuf_impl->mbuf_size = count;
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

HcclResult HcclRawForceCloseStub(HcclConn conn) {
  return HCCL_SUCCESS;
}
HcclResult HcclRawForceCloseFail(HcclConn conn) {
  return HCCL_E_INTERNAL;
}

HcclResult HcclRawImprobeStub(HcclConn conn, int32_t *flag, HcclMessage *msg, HcclStatus *status) {
  *flag = 1;
  return HCCL_SUCCESS;
}

HcclResult HcclRawImprobeFail(HcclConn conn, int32_t *flag, HcclMessage *msg, HcclStatus *status) {
  return HCCL_E_INTERNAL;
}

HcclResult HcclRawImrecvFail(void *buf, int32_t count, HcclDataType data_type, HcclMessage *msg, HcclRequest *request) {
  return HCCL_E_INTERNAL;
}

HcclResult HcclRawTestSomeSuccForUnlink(int32_t count, HcclRequest request_array[], int32_t *comp_count,
                                        int32_t comp_indices[], HcclStatus comp_status[]) {
  *comp_count = 1;
  HcclStatus status;
  status.error = 0;
  comp_status[0] = status;
  return HCCL_SUCCESS;
}

bool get_count_first_flag = true;
HcclResult HcclRawGetCountStubForSyncKv(const HcclStatus *status, HcclDataType data_type, int *count) {
  if (get_count_first_flag) {
    *count = static_cast<int32_t>(sizeof(SyncKvMetaInfo));
    get_count_first_flag = false;
  } else {
    *count = kFirstDimForKvShape * kSecondDimForKvShape * sizeof(float);
  }
  return HCCL_SUCCESS;
}

bool recv_first_flag = true;
SyncKvReqInfo *sync_req_info;
HcclResult HcclRawImRecvStubForSyncKv(void *buf, int32_t count, HcclDataType data_type, HcclMessage *msg,
                                      HcclRequest *request) {
  if (recv_first_flag) {
    SyncKvMetaInfo *resp_info = static_cast<SyncKvMetaInfo *>(buf);
    resp_info->req_id = 0;
    resp_info->transfer_count = sync_req_info->buffer_count_per_layer * kKvSize;
  }
  return HCCL_SUCCESS;
}

bool recv_kv_fist_flag = true;
HcclResult HcclRawImRecvStubForSyncKvNeedTryAgain(void *buf, int32_t count, HcclDataType data_type, HcclMessage *msg,
                                                  HcclRequest *request) {
  if (recv_first_flag) {
    SyncKvMetaInfo *resp_info = static_cast<SyncKvMetaInfo *>(buf);
    resp_info->req_id = 0;
    resp_info->transfer_count = sync_req_info->buffer_count_per_layer * kKvSize;
    recv_first_flag = false;
    return HCCL_SUCCESS;
  }
  if (recv_kv_fist_flag) {
    recv_kv_fist_flag = false;
    return HCCL_E_AGAIN;
  }
  return HCCL_SUCCESS;
}

int fd = 1;

bool accept_first_time = true;
HcclResult HcclRawAcceptSuc(HcclConn listen_conn, HcclAddr *accept_addr, HcclConn *accept_conn) {
  if (accept_first_time) {
    *accept_conn = &fd;
    accept_first_time = false;
  } else {
    return HCCL_E_AGAIN;
  }
  return HCCL_SUCCESS;
}

int probe_first_time = 0;
HcclResult HcclRawImprobeSuc(HcclConn conn, int32_t *flag, HcclMessage *msg, HcclStatus *status) {
  if (probe_first_time < 2) {
    *flag = 1;
  }
  probe_first_time++;
  return HCCL_SUCCESS;
}

HcclResult HcclRawOpenSuc(HcclConn *conn) {
  *conn = &fd;
  return HCCL_SUCCESS;
}

HcclResult HcclRawOpenFail(HcclConn *conn) {
  return HCCL_E_INTERNAL;
}

HcclResult HcclRawBindFail(HcclConn conn, HcclAddr *bind_addr) {
  return HCCL_E_INTERNAL;
}

HcclResult HcclRawConnectFail(HcclConn conn, HcclAddr *connect_addr) {
  return HCCL_E_INTERNAL;
}

HcclResult HcclRawIsendFail(const void *buf, int32_t count, HcclDataType data_type, HcclConn conn,
                            HcclRequest *request) {
  return HCCL_E_INTERNAL;
}

bool raw_send_eagain_and_fail = true;
HcclResult HcclRawIsendEagainAndFail(const void *buf, int32_t count, HcclDataType data_type, HcclConn conn,
                                     HcclRequest *request) {
  if (raw_send_eagain_and_fail) {
    raw_send_eagain_and_fail = false;
    return HCCL_E_AGAIN;
  }
  return HCCL_E_INTERNAL;
}

bool send_first_time = false;
HcclResult HcclRawIsendFailForTransferKv(const void *buf, int32_t count, HcclDataType data_type, HcclConn conn,
                                         HcclRequest *request) {
  if (send_first_time) {
    send_first_time = false;
    return HCCL_SUCCESS;
  }
  return HCCL_E_INTERNAL;
}

HcclResult HcclRawTestSomeFail(int32_t count, HcclRequest request_array[], int32_t *comp_count, int32_t comp_indices[],
                               HcclStatus comp_status[]) {
  return HCCL_E_INTERNAL;
}

HcclResult HcclRawTestSomeFail2(int32_t count, HcclRequest request_array[], int32_t *comp_count, int32_t comp_indices[],
                                HcclStatus comp_status[]) {
  *comp_count = 1;
  comp_status[0].error = 1;
  return HCCL_SUCCESS;
}

HcclResult HcclRawTestSomeFail3(int32_t count, HcclRequest request_array[], int32_t *comp_count, int32_t comp_indices[],
                                HcclStatus comp_status[]) {
  *comp_count = 2;
  return HCCL_SUCCESS;
}

HcclResult HcclRawImprobeSucOfLink(HcclConn conn, int32_t *flag, HcclMessage *msg, HcclStatus *status) {
  *flag = 1;
  return HCCL_SUCCESS;
}

HcclResult HcclRawGetCountSucOfLink(const HcclStatus *status, HcclDataType data_type, int *count) {
  *count = 0;
  return HCCL_SUCCESS;
}

HcclResult HcclRawGetCountFail(const HcclStatus *status, HcclDataType data_type, int *count) {
  return HCCL_E_INTERNAL;
}

HcclResult HcclRawTestSomeSucOfLink(int32_t count, HcclRequest request_array[], int32_t *comp_count,
                                    int32_t comp_indices[], HcclStatus comp_status[]) {
  *comp_count = 1;
  comp_status[0].error = 0;
  return HCCL_SUCCESS;
}

HcclResult HcclRawTestSomeTimeout(int32_t count, HcclRequest request_array[], int32_t *comp_count,
                                  int32_t comp_indices[], HcclStatus comp_status[]) {
  *comp_count = 0;
  comp_status[0].error = 0;
  return HCCL_SUCCESS;
}

bool get_count_first_time = false;
HcclResult HcclRawGetCountSuc(const HcclStatus *status, HcclDataType data_type, int *count) {
  if (get_count_first_time) {
    get_count_first_time = false;
    *count = sizeof(ClientClusterInfo);
  } else {
    *count = sizeof(SyncKvReqInfo) + 3 * sizeof(SyncBufferInfo);
  }
  return HCCL_SUCCESS;
}

HcclResult HcclRawGetCountSuc2(const HcclStatus *status, HcclDataType data_type, int *count) {
  if (get_count_first_time) {
    *count = sizeof(ClientClusterInfo);
    get_count_first_time = false;
  } else {
    *count = sizeof(SyncKvReqInfo) + 2 * sizeof(SyncBufferInfo);
  }
  return HCCL_SUCCESS;
}

HcclResult HcclRawTestSomeSuc(int32_t count, HcclRequest request_array[], int32_t *comp_count, int32_t comp_indices[],
                              HcclStatus comp_status[]) {
  *comp_count = 1;
  comp_status[0].error = 0;
  return HCCL_SUCCESS;
}

SyncKvReqInfo *sync_kv_req_info;

HcclResult HcclRawImrecvSuc(void *buf, int32_t count, HcclDataType data_type, HcclMessage *msg, HcclRequest *request) {
  memcpy_s(buf, count, sync_kv_req_info, count);
  return HCCL_SUCCESS;
}

HcclResult HcclRawGetCountOfUnlink(const HcclStatus *status, HcclDataType data_type, int *count) {
  *count = 0;
  return HCCL_SUCCESS;
}

HcclResult HcclRawImprobeSucOfUnLink(HcclConn conn, int32_t *flag, HcclMessage *msg, HcclStatus *status) {
  *flag = 1;
  return HCCL_SUCCESS;
}

bool test_recv_first_flag = true;
bool test_recv_second_flag = true;
HcclResult HcclRawTestSomeFailAfterTryAgain(int32_t count, HcclRequest request_array[], int32_t *comp_count,
                                            int32_t comp_indices[], HcclStatus comp_status[]) {
  if (test_recv_first_flag) {
    *comp_count = 1;
    comp_status[0].error = 0;
    test_recv_first_flag = false;
    return HCCL_SUCCESS;
  }
  if (test_recv_second_flag) {
    *comp_count = 1;
    comp_status[0].error = 0;
    test_recv_second_flag = false;
    return HCCL_SUCCESS;
  }
  return HCCL_E_INTERNAL;
}

HcclResult HcclRawTestSomeFail2AfterTryAgain(int32_t count, HcclRequest request_array[], int32_t *comp_count,
                                             int32_t comp_indices[], HcclStatus comp_status[]) {
  if (test_recv_first_flag) {
    *comp_count = 1;
    comp_status[0].error = 0;
    test_recv_first_flag = false;
    return HCCL_SUCCESS;
  }
  if (test_recv_second_flag) {
    *comp_count = 1;
    comp_status[0].error = 0;
    test_recv_second_flag = false;
    return HCCL_SUCCESS;
  }
  *comp_count = 0;
  comp_status[0].error = 1;
  return HCCL_E_INTERNAL;
}
TransferToRemoteReq *push_kv_req_info;
std::vector<uint8_t> push_kv_req_info_data;
uint32_t recv_push_kv_probe_first_time = 0U;
HcclResult HcclRawImprobeSucForRecvTransferKv(HcclConn conn, int32_t *flag, HcclMessage *msg, HcclStatus *status) {
  if (recv_push_kv_probe_first_time <= 2U) {
    *flag = 1;
  }
  recv_push_kv_probe_first_time++;
  return HCCL_SUCCESS;
}

uint32_t recv_push_get_count_first_time = 0U;
HcclResult HcclRawGetCountStubForRecvTransferKv(const HcclStatus *status, HcclDataType data_type, int *count) {
  if (recv_push_get_count_first_time == 0U) {
    *count = sizeof(TransferToRemoteReq) + 1 * sizeof(TransferSlotInfo);
    recv_push_get_count_first_time++;
  } else if (recv_push_get_count_first_time <= 2U) {
    *count = kFirstDimForKvShape * kSecondDimForKvShape * 4;
    recv_push_get_count_first_time++;
  }
  return HCCL_SUCCESS;
}

uint32_t recv_push_imrecv_first_time = 0U;
HcclResult HcclRawImrecvSucForRecvTransferKv(void *buf, int32_t count, HcclDataType data_type, HcclMessage *msg,
                                             HcclRequest *request) {
  size_t mem_size = 0U;
  if (recv_push_imrecv_first_time == 0U) {
    mem_size = sizeof(TransferToRemoteReq) + 1 * sizeof(TransferSlotInfo);
    memcpy_s(buf, mem_size, push_kv_req_info, mem_size);
    recv_push_imrecv_first_time++;
  } else if (recv_push_imrecv_first_time <= 2U) {
    UDF_LOG_INFO("Memset buf:%p", buf);
    uint32_t val = 2;
    memcpy_s(buf, 4, &val, 4);
    recv_push_imrecv_first_time++;
  }
  return HCCL_SUCCESS;
}

HcclResult HcclRawImrecvSucForRecvTransferKvEgain(void *buf, int32_t count, HcclDataType data_type, HcclMessage *msg,
                                                  HcclRequest *request) {
  size_t mem_size = 0U;
  if (recv_push_imrecv_first_time == 0U) {
    mem_size = sizeof(TransferToRemoteReq) + 1 * sizeof(TransferSlotInfo);
    memcpy_s(buf, mem_size, push_kv_req_info, mem_size);
    recv_push_imrecv_first_time++;
  } else if (recv_push_imrecv_first_time == 1U) {
    recv_push_imrecv_first_time++;
    return HCCL_E_AGAIN;
  } else {
    return HCCL_E_INTERNAL;
  }
  return HCCL_SUCCESS;
}

HcclResult HcclRawGetCountStubForTransferKv(const HcclStatus *status, HcclDataType data_type, int *count) {
  *count = sizeof(TransferKvMetaInfo);
  return HCCL_SUCCESS;
}

HcclResult HcclRawImRecvStubForTransferKv(void *buf, int32_t count, HcclDataType data_type, HcclMessage *msg,
                                          HcclRequest *request) {
  auto *resp_info = static_cast<TransferKvMetaInfo *>(buf);
  resp_info->err_code = static_cast<int32_t>(FsmStatus::kFsmSuccess);
  return HCCL_SUCCESS;
}

HcclResult HcclRawImRecvStubForTransferKvValidateMem(void *buf, int32_t count, HcclDataType data_type, HcclMessage *msg,
                                                     HcclRequest *request) {
  auto *resp_info = static_cast<TransferKvMetaInfo *>(buf);
  resp_info->err_code = static_cast<int32_t>(FsmStatus::kFsmParamInvalid);
  return HCCL_SUCCESS;
}

uint32_t recv_push_kv_probe_blocks_first_time = 0U;
HcclResult HcclRawImprobeSucForRecvBlocksTransferKv(HcclConn conn, int32_t *flag, HcclMessage *msg,
                                                    HcclStatus *status) {
  if (recv_push_kv_probe_blocks_first_time <= 4U) {
    *flag = 1;
  }
  recv_push_kv_probe_blocks_first_time++;
  return HCCL_SUCCESS;
}

uint32_t recv_push_blocks_get_count_first_time = 0U;
HcclResult HcclRawGetCountStubForRecvBlocksTransferKv(const HcclStatus *status, HcclDataType data_type, int *count) {
  if (recv_push_blocks_get_count_first_time == 0U) {
    *count = sizeof(TransferToRemoteReq) + 3 * sizeof(TransferSlotInfo);
    recv_push_blocks_get_count_first_time++;
  } else if (recv_push_blocks_get_count_first_time == 1U || recv_push_blocks_get_count_first_time == 3U) {
    *count = 20;
    recv_push_blocks_get_count_first_time++;
  } else if (recv_push_blocks_get_count_first_time == 2U || recv_push_blocks_get_count_first_time == 4U) {
    *count = 10;
    recv_push_blocks_get_count_first_time++;
  }
  return HCCL_SUCCESS;
}

uint32_t recv_push_imrecv_blocks_first_time = 0U;
HcclResult HcclRawImrecvSucForRecvBlocksTransferKv(void *buf, int32_t count, HcclDataType data_type, HcclMessage *msg,
                                                   HcclRequest *request) {
  size_t mem_size = 0U;
  if (recv_push_imrecv_blocks_first_time == 0U) {
    mem_size = sizeof(TransferToRemoteReq) + 3 * sizeof(TransferSlotInfo);
    memcpy_s(buf, mem_size, push_kv_req_info, mem_size);
    recv_push_imrecv_blocks_first_time++;
  } else if (recv_push_imrecv_blocks_first_time <= 2U) {
    UDF_LOG_INFO("Memset buf:%p", buf);
    uint32_t val = 2;
    memcpy_s(buf, 4, &val, 4);
    recv_push_imrecv_blocks_first_time++;
  }
  return HCCL_SUCCESS;
}

HcclResult HcclRawImrecvScatterStub(void *buf[], int counts[], int buf_count, HcclDataType data_type, HcclMessage *msg,
                                    HcclRequest *request) {
  UDF_LOG_INFO("Memset buf:%p", buf[0]);
  uint32_t val = 2;
  memcpy_s(buf[0], 4, &val, 4);
  return HCCL_SUCCESS;
}

std::shared_ptr<FlowFuncParams> CreateAttrsByRole(const std::string &role, bool set_env = false) {
  auto func_params = std::make_shared<FlowFuncParams>("params", 4, 4, 0, 0);
  ff::udf::AttrValue device_index;
  device_index.set_i(0);
  auto device_index_attr = std::make_shared<AttrValueImpl>(device_index);
  func_params->attr_map_["device_index"] = device_index_attr;
  if (role == "Prompt") {
    ff::udf::AttrValue is_prompt;
    is_prompt.set_i(1);
    auto is_prompt_attr = std::make_shared<AttrValueImpl>(is_prompt);
    func_params->attr_map_["is_prompt"] = is_prompt_attr;

    ff::udf::AttrValue listen_ip;
    listen_ip.set_i(1);
    auto listen_ip_attr = std::make_shared<AttrValueImpl>(listen_ip);
    func_params->attr_map_["ip"] = listen_ip_attr;

    ff::udf::AttrValue port;
    port.set_i(1);
    auto port_attr = std::make_shared<AttrValueImpl>(port);
    func_params->attr_map_["port"] = port_attr;
  }
  if (set_env) {
    ff::udf::AttrValue env_names;
    env_names.mutable_array()->add_s("HCCL_RDMA_TC");
    env_names.mutable_array()->add_s("HCCL_RDMA_SL");
    env_names.mutable_array()->add_s("HCCL_RDMA_TIMEOUT");
    env_names.mutable_array()->add_s("HCCL_RDMA_RETRY_CNT");
    auto env_names_attr = std::make_shared<AttrValueImpl>(env_names);
    func_params->attr_map_["ENV_NAMES"] = env_names_attr;

    ff::udf::AttrValue env_values;
    env_values.mutable_array()->add_s("1");
    env_values.mutable_array()->add_s("2");
    env_values.mutable_array()->add_s("3");
    env_values.mutable_array()->add_s("4");
    auto env_values_attr = std::make_shared<AttrValueImpl>(env_values);
    func_params->attr_map_["ENV_VALUES"] = env_values_attr;
  }
  ff::udf::AttrValue output_indices;
  auto array = output_indices.mutable_array();
  for (int32_t i = 0; i < 12; ++i) {
    array->add_i(0);
  }
  auto output_indices_attr = std::make_shared<AttrValueImpl>(output_indices);
  func_params->attr_map_["output_indices"] = output_indices_attr;
  ff::udf::AttrValue role_attr_value;
  role_attr_value.set_s(role);
  auto role_attr = std::make_shared<AttrValueImpl>(role_attr_value);
  func_params->attr_map_["role"] = role_attr;
  return func_params;
}

std::shared_ptr<FlowFuncParams> CreateAttrs(bool is_prompt = false, bool set_env = false) {
  return CreateAttrsByRole(is_prompt ? "Prompt" : "Decoder", set_env);
}

std::vector<uint8_t> sync_req_info_data;
std::vector<uint8_t> sync_kv_req_info_data;
}  // namespace

class LlmServiceFlowFuncUTest : public testing::Test {
 public:
  static void SetUpTestCase() {
    sync_req_info_data.resize(sizeof(SyncKvReqInfo) + 3 * sizeof(SyncBufferInfo));
    sync_kv_req_info_data.resize(sizeof(SyncKvReqInfo) + 4 * sizeof(SyncBufferInfo));
    push_kv_req_info_data.resize(sizeof(TransferToRemoteReq) + 3 * sizeof(TransferSlotInfo));
    sync_req_info = (SyncKvReqInfo *)sync_req_info_data.data();
    sync_kv_req_info = (SyncKvReqInfo *)sync_kv_req_info_data.data();
    push_kv_req_info = (TransferToRemoteReq *)push_kv_req_info_data.data();
    push_kv_req_info->dst_cache_id = -1;
    push_kv_req_info->tensor_num_per_layer = 2;
    sync_req_info->cache_id = -1;
    sync_req_info->prefix_id = UINT64_MAX;
    sync_kv_req_info->cache_id = -1;
    sync_kv_req_info->prefix_id = UINT64_MAX;
    for (int i = 0; i < 2; ++i) {
      sync_req_info->transfer_infos[i].buffer_info.block_index = UINT64_MAX;
    }
    for (int i = 0; i < 2; ++i) {
      sync_kv_req_info->transfer_infos[i].buffer_info.block_index = UINT64_MAX;
    }
  }

 protected:
  virtual void SetUp() {
    dlog_setlevel(UDF, DLOG_INFO, 1);
    HcclSoManager::GetInstance()->func_map_.clear();
    void *func = reinterpret_cast<void *>(&HcclRawAcceptStub);
    HcclSoManager::GetInstance()->func_map_.emplace(kHcclRawAcceptFuncName, func);
    void *func_raw_connect_func = reinterpret_cast<void *>(&HcclRawConnectStub);
    HcclSoManager::GetInstance()->func_map_.emplace(kHcclRawConnectFuncName, func_raw_connect_func);
    void *func_raw_listen_func = reinterpret_cast<void *>(&HcclRawListenStub);
    HcclSoManager::GetInstance()->func_map_.emplace(kHcclRawListenFuncName, func_raw_listen_func);
    void *func_raw_isend_func = reinterpret_cast<void *>(&HcclRawIsendStub);
    HcclSoManager::GetInstance()->func_map_.emplace(kHcclRawIsendFuncName, func_raw_isend_func);
    void *func_raw_improbe_func = reinterpret_cast<void *>(&HcclRawImprobeStub);
    HcclSoManager::GetInstance()->func_map_.emplace(kHcclRawImprobeFuncName, func_raw_improbe_func);
    void *func_raw_im_recv_func = reinterpret_cast<void *>(&HcclRawImrecvStub);
    HcclSoManager::GetInstance()->func_map_.emplace(kHcclRawImRecvFuncName, func_raw_im_recv_func);
    void *func_raw_test_some_func = reinterpret_cast<void *>(&HcclRawTestSomeStub);
    HcclSoManager::GetInstance()->func_map_.emplace(kHcclRawTestSomeFuncName, func_raw_test_some_func);
    void *func_raw_get_count_func = reinterpret_cast<void *>(&HcclRawGetCountStub);
    HcclSoManager::GetInstance()->func_map_.emplace(kHcclRawGetCountFuncName, func_raw_get_count_func);
    void *func_raw_open_func = reinterpret_cast<void *>(&HcclRawOpenStub);
    HcclSoManager::GetInstance()->func_map_.emplace(kHcclRawOpenFuncName, func_raw_open_func);
    void *func_raw_close_func = reinterpret_cast<void *>(&HcclRawCloseStub);
    HcclSoManager::GetInstance()->func_map_.emplace(kHcclRawCloseFuncName, func_raw_close_func);
    void *func_raw_force_close_func = reinterpret_cast<void *>(&HcclRawForceCloseStub);
    HcclSoManager::GetInstance()->func_map_.emplace(kHcclRawForceCloseFuncName, func_raw_force_close_func);
    void *func_raw_bind_func = reinterpret_cast<void *>(&HcclRawBindStub);
    HcclSoManager::GetInstance()->func_map_.emplace(kHcclRawBindFuncName, func_raw_bind_func);
    void *func_register_global_memory_func = reinterpret_cast<void *>(&HcclRegisterGlobalMemoryStub);
    HcclSoManager::GetInstance()->func_map_.emplace(kHcclRegisterGlobalMemoryFuncName,
                                                    func_register_global_memory_func);
    void *func_unregister_global_memory_func = reinterpret_cast<void *>(&HcclUnregisterGlobalMemoryStub);
    HcclSoManager::GetInstance()->func_map_.emplace(kHcclUnregisterGlobalMemoryFuncName,
                                                    func_unregister_global_memory_func);
    void *func_raw_im_recv_scatter_func = reinterpret_cast<void *>(&HcclRawImrecvScatterStub);
    HcclSoManager::GetInstance()->func_map_.emplace(kHcclRawImRecvScatterFuncName, func_raw_im_recv_scatter_func);

    MOCKER_CPP(&HcclSoManager::LoadSo).defaults().will(returnValue(FLOW_FUNC_SUCCESS));
    MOCKER(HcclRawOpen).defaults().will(invoke(HcclRawOpenSuc));
    MOCKER(halGrpQuery).defaults().will(invoke(halGrpQuerySuc));
    MOCKER(halMbufAllocEx).defaults().will(invoke(halMbufAllocExStub));
    MOCKER(halMbufAlloc).defaults().will(invoke(halMbufAllocStub));
    MOCKER(halMbufFree).defaults().will(invoke(halMbufFreeStub));
    MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStub));
    MOCKER(halMbufGetBuffSize).defaults().will(invoke(halMbufGetBuffSizeStub));
    MOCKER(halMbufSetDataLen).defaults().will(invoke(halMbufSetDataLenStub));
    MOCKER(halMbufGetPrivInfo).defaults().will(invoke(halMbufGetPrivInfoStub));
    MOCKER(halMbufGetDataLen).defaults().will(invoke(halMbufGetDataLenStub));
    MOCKER(HcclRawImprobe).defaults().will(invoke(HcclRawImprobeSucOfLink));
    MOCKER(HcclRawGetCount).defaults().will(invoke(HcclRawGetCountSucOfLink));
  }

  virtual void TearDown() {
    LlmCommEntityMgr::GetInstance().ClearEntities();
    GlobalMockObject::verify();
  }

  FsmStatus AllocateCache(const std::shared_ptr<FlowFuncRunContext> &run_context, std::shared_ptr<FlowMsg> &out_msg,
                          LlmServiceFlowFunc &llm_service_flow_func, int64_t cache_id,
                          const std::vector<uint64_t> &req_ids) {
    size_t req_size = sizeof(AllocateCacheReqInfo) + sizeof(uint64_t) * req_ids.size();
    std::shared_ptr<FlowMsg> flow_msg =
        run_context->AllocTensorMsg({static_cast<int64_t>(req_size)}, TensorDataType::DT_UINT8);
    std::vector<std::shared_ptr<FlowMsg>> msgs;
    msgs.emplace_back(flow_msg);

    auto &req_info = *static_cast<AllocateCacheReqInfo *>(flow_msg->GetTensor()->GetData());
    req_info.cache_id = cache_id;
    req_info.num_tensors = kKvSize;
    req_info.num_dims = 2;
    req_info.dims[0] = kFirstDimForKvShape;
    req_info.dims[1] = kSecondDimForKvShape;
    req_info.num_requests = req_ids.size();
    for (size_t i = 0U; i < req_ids.size(); ++i) {
      req_info.req_ids[i] = req_ids[i];
    }
    if (llm_service_flow_func.AllocateCache(run_context, msgs) != 0) {
      return FsmStatus::kFsmFailed;
    }
    auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
    return static_cast<FsmStatus>(*output_data);
  }

  FsmStatus AllocateBlockCaches(const std::shared_ptr<FlowFuncRunContext> &run_context,
                                std::shared_ptr<FlowMsg> &out_msg, LlmServiceFlowFunc &llm_service_flow_func,
                                int64_t cache_id, const std::vector<uint64_t> &req_ids) {
    size_t req_size = sizeof(AllocateCacheReqInfo) + sizeof(uint64_t) * req_ids.size();
    std::shared_ptr<FlowMsg> flow_msg =
        run_context->AllocTensorMsg({static_cast<int64_t>(req_size)}, TensorDataType::DT_UINT8);
    std::vector<std::shared_ptr<FlowMsg>> msgs;
    msgs.emplace_back(flow_msg);

    auto &req_info = *static_cast<AllocateCacheReqInfo *>(flow_msg->GetTensor()->GetData());
    req_info.cache_id = cache_id;
    req_info.num_tensors = kKvSize;
    req_info.num_dims = 2;
    req_info.dims[0] = 4;
    req_info.dims[1] = 10;
    req_info.dtype = 2;  // UINT8
    req_info.is_allocate_blocks = true;
    req_info.num_requests = req_ids.size();
    for (size_t i = 0U; i < req_ids.size(); ++i) {
      req_info.req_ids[i] = req_ids[i];
    }
    if (llm_service_flow_func.AllocateCache(run_context, msgs) != 0) {
      return FsmStatus::kFsmFailed;
    }
    auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
    return static_cast<FsmStatus>(*output_data);
  }

  FsmStatus DeallocateCache(const std::shared_ptr<FlowFuncRunContext> &run_context, std::shared_ptr<FlowMsg> &out_msg,
                            LlmServiceFlowFunc &llm_service_flow_func, int64_t cache_id) {
    std::shared_ptr<FlowMsg> flow_msg = run_context->AllocTensorMsg({1}, TensorDataType::DT_INT64);
    std::vector<std::shared_ptr<FlowMsg>> msgs;
    msgs.emplace_back(flow_msg);
    auto req_info = static_cast<int64_t *>(flow_msg->GetTensor()->GetData());
    *req_info = cache_id;
    if (llm_service_flow_func.DeallocateCache(run_context, msgs) != 0) {
      return FsmStatus::kFsmFailed;
    }
    auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
    return static_cast<FsmStatus>(*output_data);
  }

  FsmStatus RemoveCacheIndex(const std::shared_ptr<FlowFuncRunContext> &run_context, std::shared_ptr<FlowMsg> &out_msg,
                             LlmServiceFlowFunc &llm_service_flow_func, const RemoveCacheIndexReqInfo &req_info) {
    std::shared_ptr<FlowMsg> flow_msg =
        run_context->AllocTensorMsg({static_cast<int64_t>(sizeof(req_info))}, TensorDataType::DT_UINT8);
    std::vector<std::shared_ptr<FlowMsg>> msgs;
    msgs.emplace_back(flow_msg);
    *static_cast<RemoveCacheIndexReqInfo *>(flow_msg->GetTensor()->GetData()) = req_info;
    if (llm_service_flow_func.RemoveCacheIndex(run_context, msgs) != 0) {
      return FsmStatus::kFsmFailed;
    }
    auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
    return static_cast<FsmStatus>(*output_data);
  }
};

TEST_F(LlmServiceFlowFuncUTest, init_success) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(false, true);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;
  ret = llm_service_flow_func.Initialize(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(std::string(std::getenv("HCCL_RDMA_TC")), "1");
  EXPECT_EQ(std::string(std::getenv("HCCL_RDMA_SL")), "2");
  EXPECT_EQ(std::string(std::getenv("HCCL_RDMA_TIMEOUT")), "3");
  EXPECT_EQ(std::string(std::getenv("HCCL_RDMA_RETRY_CNT")), "4");
  unsetenv("HCCL_RDMA_TC");
  unsetenv("HCCL_RDMA_SL");
  unsetenv("HCCL_RDMA_TIMEOUT");
  unsetenv("HCCL_RDMA_RETRY_CNT");
}

TEST_F(LlmServiceFlowFuncUTest, init_failed) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  params->attr_map_.erase("device_index");
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_ERR_ATTR_NOT_EXITS);
}

TEST_F(LlmServiceFlowFuncUTest, init_group_query_failed) {
  MOCKER(halGrpQuery).stubs().will(invoke(halGrpQueryFail2));
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_FAILED);
}

TEST_F(LlmServiceFlowFuncUTest, init_group_size_over_one) {
  MOCKER(halGrpQuery).stubs().will(invoke(halGrpQueryFail3));
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(LlmServiceFlowFuncUTest, init_load_so_failed) {
  MOCKER_CPP(&HcclSoManager::LoadSo).stubs().will(returnValue(FLOW_FUNC_FAILED));
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_FAILED);
}

TEST_F(LlmServiceFlowFuncUTest, update_link_with_one_cluster) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  uint32_t cluster_num = 1U;
  uint32_t ip_num = 2U;
  uint64_t cluster_id = 1U;
  uint64_t size = sizeof(UpdateLinkReqInfo) + cluster_num * sizeof(ClusterInfo) + ip_num * sizeof(IpInfo);
  std::shared_ptr<FlowMsg> flow_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(size)}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> msgs;
  msgs.emplace_back(flow_msg);

  auto link_req_ptr = static_cast<UpdateLinkReqInfo *>(flow_msg->GetTensor()->GetData());
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kLink);
  link_req_ptr->cluster_num = cluster_num;
  link_req_ptr->timeout = 10000;
  link_req_ptr->force_flag = 0U;
  auto cluster_addr = PtrToValue(flow_msg->GetTensor()->GetData()) + sizeof(UpdateLinkReqInfo);
  auto cluster_info_ptr = static_cast<ClusterInfo *>(ValueToPtr(cluster_addr));
  cluster_info_ptr->ip_num = ip_num;
  cluster_info_ptr->remote_cluster_id = cluster_id;
  auto ip1_addr = cluster_addr + sizeof(ClusterInfo);
  auto ip1_ptr = static_cast<IpInfo *>(ValueToPtr(ip1_addr));
  ip1_ptr->local_ip = 1;
  ip1_ptr->remote_ip = 2;
  auto ip2_addr = cluster_addr + sizeof(ClusterInfo) + sizeof(IpInfo);
  auto ip2_ptr = static_cast<IpInfo *>(ValueToPtr(ip2_addr));
  ip2_ptr->local_ip = 3;
  ip2_ptr->remote_ip = 4;

  // Link failed
  MOCKER(HcclRawOpen).stubs().will(invoke(HcclRawOpenFail));
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmLinkFailed));
  MOCKER(HcclRawOpen).reset();

  MOCKER(HcclRawBind).stubs().will(invoke(HcclRawBindFail));
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmLinkFailed));
  MOCKER(HcclRawBind).reset();

  MOCKER(HcclRawConnect).stubs().will(invoke(HcclRawConnectFail));
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmLinkFailed));
  MOCKER(HcclRawConnect).reset();

  MOCKER(HcclRawIsend).stubs().will(invoke(HcclRawIsendFail));
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmLinkFailed));
  MOCKER(HcclRawIsend).reset();
  MOCKER(HcclRawIsend).stubs().will(invoke(HcclRawIsendStub));

  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeFail));
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmLinkFailed));
  MOCKER(HcclRawTestSome).reset();

  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeFail2));
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmLinkFailed));
  MOCKER(HcclRawTestSome).reset();

  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeFail3));
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmLinkFailed));
  MOCKER(HcclRawTestSome).reset();

  // link timeout
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeTimeout));
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmTimeout));
  MOCKER(HcclRawTestSome).reset();

  // link suc
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSucOfLink));
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmSuccess));
  MOCKER(HcclRawTestSome).reset();

  // already linked
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmAlreadyLinked));

  // unlink failed
  MOCKER(HcclRawIsend).reset();
  MOCKER(HcclRawIsend).stubs().will(invoke(HcclRawIsendEagainAndFail));
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kUnlink);
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmNotifyPromptUnlinkFailed));
  MOCKER(HcclRawIsend).reset();

  // link suc
  MOCKER(HcclRawIsend).stubs().will(invoke(HcclRawIsendStub));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSucOfLink));
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kLink);
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmSuccess));
  MOCKER(HcclRawTestSome).reset();

  // unlink failed
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeFail));
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kUnlink);
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmNotifyPromptUnlinkFailed));
  MOCKER(HcclRawTestSome).reset();

  // link suc
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSucOfLink));
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kLink);
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmSuccess));
  MOCKER(HcclRawTestSome).reset();

  // unlink failed
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuccForUnlink));
  MOCKER(HcclRawForceClose).stubs().will(invoke(HcclRawForceCloseFail));
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kUnlink);
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmUnlinkFailed));
  MOCKER(HcclRawTestSome).reset();
  MOCKER(HcclRawForceClose).reset();

  // link suc
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSucOfLink));
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kLink);
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmSuccess));
  MOCKER(HcclRawTestSome).reset();

  // unlink failed
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeFail2));
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kUnlink);
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmNotifyPromptUnlinkFailed));
  MOCKER(HcclRawTestSome).reset();

  // link suc
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSucOfLink));
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kLink);
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmSuccess));
  MOCKER(HcclRawTestSome).reset();

  // unlink suc
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuccForUnlink));
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kUnlink);
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmSuccess));
  MOCKER(HcclRawTestSome).reset();

  // unlink fail not_link
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kUnlink);
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmNotLink));
}

TEST_F(LlmServiceFlowFuncUTest, update_link_with_multi_cluster) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  uint32_t cluster_num = 2U;
  uint32_t ip_num = 2U;
  uint64_t first_remote_cluster_id = 0;
  uint64_t second_remote_cluster_id = 1;
  uint64_t cluster_size = sizeof(ClusterInfo) + ip_num * sizeof(IpInfo);
  uint64_t total_size = sizeof(UpdateLinkReqInfo) + cluster_num * (cluster_size);
  std::shared_ptr<FlowMsg> flow_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(total_size)}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> msgs;
  msgs.emplace_back(flow_msg);

  auto link_req_ptr = static_cast<UpdateLinkReqInfo *>(flow_msg->GetTensor()->GetData());
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kLink);
  link_req_ptr->cluster_num = cluster_num;
  link_req_ptr->force_flag = 0U;
  auto cluster_addr = reinterpret_cast<uintptr_t>(link_req_ptr->cluster_infos);
  auto cluster_info_ptr = link_req_ptr->cluster_infos;
  cluster_info_ptr->ip_num = ip_num;
  cluster_info_ptr->remote_cluster_id = first_remote_cluster_id;
  cluster_info_ptr->ip_infos[0].local_ip = 1;
  cluster_info_ptr->ip_infos[0].remote_ip = 2;
  cluster_info_ptr->ip_infos[1].local_ip = 3;
  cluster_info_ptr->ip_infos[1].remote_ip = 4;
  cluster_addr += cluster_size;
  cluster_info_ptr = reinterpret_cast<ClusterInfo *>(cluster_addr);
  cluster_info_ptr->ip_num = ip_num;
  cluster_info_ptr->remote_cluster_id = second_remote_cluster_id;
  cluster_info_ptr->ip_infos[0].local_ip = 5;
  cluster_info_ptr->ip_infos[0].remote_ip = 6;
  cluster_info_ptr->ip_infos[1].local_ip = 7;
  cluster_info_ptr->ip_infos[1].remote_ip = 8;

  // link suc
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSucOfLink));
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(output_data[0], static_cast<int32_t>(FsmStatus::kFsmSuccess));
  EXPECT_EQ(output_data[1], static_cast<int32_t>(FsmStatus::kFsmSuccess));
  MOCKER(HcclRawTestSome).reset();

  // unlink suc
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuccForUnlink));
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kUnlink);
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(output_data[0], static_cast<int32_t>(FsmStatus::kFsmSuccess));
  EXPECT_EQ(output_data[1], static_cast<int32_t>(FsmStatus::kFsmSuccess));
  MOCKER(HcclRawTestSome).reset();

  // link suc
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSucOfLink));
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kLink);
  // repeat cluster id
  cluster_info_ptr->remote_cluster_id = first_remote_cluster_id;
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(output_data[0], static_cast<int32_t>(FsmStatus::kFsmSuccess));
  EXPECT_EQ(output_data[1], static_cast<int32_t>(FsmStatus::kFsmRepeatRequest));
  MOCKER(HcclRawTestSome).reset();

  // unlink suc
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuccForUnlink));
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kUnlink);
  cluster_info_ptr->remote_cluster_id = first_remote_cluster_id;
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(output_data[0], static_cast<int32_t>(FsmStatus::kFsmSuccess));
  EXPECT_EQ(output_data[1], static_cast<int32_t>(FsmStatus::kFsmParamInvalid));
  MOCKER(HcclRawTestSome).reset();
}

TEST_F(LlmServiceFlowFuncUTest, copy_cache_not_exist) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  std::shared_ptr<FlowMsg> input_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(sizeof(CopyCacheReqInfo))}, TensorDataType::DT_INT8);
  input_msgs.emplace_back(input_msg);
  const Tensor *req_info_tensor = input_msg->GetTensor();
  CopyCacheReqInfo *req_info = static_cast<CopyCacheReqInfo *>(req_info_tensor->GetData());
  *req_info = CopyCacheReqInfo{};
  // call merge kv cache proc: kv not exist
  int32_t merge_ret = llm_service_flow_func.CopyCache(run_context, input_msgs);
  EXPECT_EQ(merge_ret, FLOW_FUNC_SUCCESS);
  // test output: kv not exist
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmKvNotExist));
}

TEST_F(LlmServiceFlowFuncUTest, copy_cache_param_invalid) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);

  // [1, 256], float
  int64_t src_cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, src_cache_id, {}), FsmStatus::kFsmSuccess);
  int64_t dst_cache_id = 2;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, dst_cache_id, {}), FsmStatus::kFsmSuccess);
  std::shared_ptr<FlowMsg> flow_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(sizeof(CopyCacheReqInfo))}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> msgs;
  msgs.emplace_back(flow_msg);
  auto &req_info = *static_cast<CopyCacheReqInfo *>(flow_msg->GetTensor()->GetData());
  req_info = CopyCacheReqInfo{};
  req_info.src_cache_id = src_cache_id;
  req_info.dst_cache_id = dst_cache_id;
  // size out of range
  req_info.copy_size = 1025;
  EXPECT_EQ(llm_service_flow_func.CopyCache(run_context, msgs), FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(static_cast<FsmStatus>(*output_data), FsmStatus::kFsmParamInvalid);

  // offset + size out of range
  req_info.copy_size = 512;
  req_info.offset = 513;
  EXPECT_EQ(llm_service_flow_func.CopyCache(run_context, msgs), FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(static_cast<FsmStatus>(*output_data), FsmStatus::kFsmParamInvalid);

  // offset out of range
  req_info.copy_size = -1;
  req_info.offset = 1024;
  EXPECT_EQ(llm_service_flow_func.CopyCache(run_context, msgs), FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(static_cast<FsmStatus>(*output_data), FsmStatus::kFsmParamInvalid);

  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, src_cache_id), FsmStatus::kFsmSuccess);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, dst_cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, sync_kv_cache_link_not_exist) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  std::shared_ptr<FlowMsg> input_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(sizeof(PullKvReqInfo))}, TensorDataType::DT_INT8);
  input_msgs.emplace_back(input_msg);
  const Tensor *req_info_tensor = input_msg->GetTensor();
  PullKvReqInfo *req_info = static_cast<PullKvReqInfo *>(req_info_tensor->GetData());
  req_info->prompt_cache_id = -1L;
  req_info->req_id = 0UL;
  req_info->model_id = 0UL;
  req_info->prompt_cluster_id = 1001UL;
  req_info->timeout = 3000000UL;
  req_info->cache_id = cache_id;
  // call sync kv cache proc: not link
  int32_t sync_ret = llm_service_flow_func.PullCache(run_context, input_msgs);
  EXPECT_EQ(sync_ret, FLOW_FUNC_SUCCESS);
  int32_t *output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmNotLink));
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, sync_kv_cache_success) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  size_t total_len = sizeof(PullKvReqInfo);
  std::shared_ptr<FlowMsg> input_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(total_len)}, TensorDataType::DT_INT8);
  input_msgs.emplace_back(input_msg);
  const Tensor *req_info_tensor = input_msg->GetTensor();
  PullKvReqInfo *req_info = static_cast<PullKvReqInfo *>(req_info_tensor->GetData());
  req_info->req_id = 0UL;
  req_info->model_id = 0UL;
  req_info->prompt_cluster_id = 1001UL;
  req_info->timeout = 3000000UL;
  req_info->cache_id = cache_id;
  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  HcclConn conn = reinterpret_cast<HcclConn>(1);
  LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityClient, conn, local_hccl_addr, remote_hccl_addr,
                                               req_info->prompt_cluster_id);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(req_info->prompt_cluster_id);
  EXPECT_NE(entity, nullptr);
  // mock for hccl api
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSucOfLink));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeStub));
  get_count_first_flag = true;
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountStubForSyncKv));
  recv_first_flag = true;
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImRecvStubForSyncKv));
  sync_req_info->req_id = req_info->req_id;
  sync_req_info->model_id = req_info->model_id;
  sync_req_info->buffer_count_per_layer = 1;
  sync_req_info->tensor_index_count = 0;
  sync_req_info->transfer_infos[0].buffer_info.block_index = 0;
  sync_req_info->transfer_infos[0].buffer_info.buffer_len = kFirstDimForKvShape * kSecondDimForKvShape * sizeof(float);
  auto sync_ret = llm_service_flow_func.PullCache(run_context, input_msgs);
  EXPECT_EQ(sync_ret, FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmSuccess));
  LlmCommEntityMgr::GetInstance().ClearEntities();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).reset();
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, remove_cache_index_success) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);

  std::vector<uint64_t> req_ids{1};
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, req_ids), FsmStatus::kFsmSuccess);
  RemoveCacheIndexReqInfo req_info{};
  req_info.model_id = 0;
  req_info.req_id = 2;
  EXPECT_EQ(RemoveCacheIndex(run_context, out_msg, llm_service_flow_func, req_info), FsmStatus::kFsmSuccess);
  req_info.req_id = 1;
  EXPECT_EQ(RemoveCacheIndex(run_context, out_msg, llm_service_flow_func, req_info), FsmStatus::kFsmSuccess);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, test_server_sync_kv_success) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(true);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  std::vector<uint64_t> req_ids{1};
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, req_ids), FsmStatus::kFsmSuccess);

  MOCKER(HcclRawAccept).stubs().will(invoke(HcclRawAcceptSuc));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSuc));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountSuc));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSuc));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  probe_first_time = 0;
  get_count_first_time = true;
  accept_first_time = true;
  sync_kv_req_info->cache_id = -1;
  sync_kv_req_info->req_id = req_ids[0];
  sync_kv_req_info->buffer_count_per_layer = 1;
  sync_kv_req_info->tensor_index_count = 0;
  sync_kv_req_info->transfer_infos[0].buffer_info.block_index = UINT64_MAX;
  sync_kv_req_info->transfer_infos[0].buffer_info.buffer_len =
      kFirstDimForKvShape * kSecondDimForKvShape * sizeof(float);
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 1);
  auto it = LlmCommEntityMgr::GetInstance().server_entity_map_.begin();
  EXPECT_EQ(it->second->GetCurState(), FsmState::kFsmProbeState);
  // unlink.
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSucOfUnLink));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountOfUnlink));
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 0);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, test_server_sync_kv_by_cache_id_success) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(true);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  std::vector<uint64_t> req_ids{1};
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, req_ids), FsmStatus::kFsmSuccess);

  MOCKER(HcclRawAccept).stubs().will(invoke(HcclRawAcceptSuc));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSuc));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountSuc));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSuc));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  probe_first_time = 0;
  get_count_first_time = true;
  accept_first_time = true;
  sync_kv_req_info->cache_id = cache_id;
  sync_kv_req_info->req_id = UINT64_MAX;
  sync_kv_req_info->buffer_count_per_layer = 1;
  sync_kv_req_info->tensor_index_count = 0;
  sync_kv_req_info->transfer_infos[0].buffer_info.block_index = UINT64_MAX;
  sync_kv_req_info->transfer_infos[0].buffer_info.buffer_len =
      kFirstDimForKvShape * kSecondDimForKvShape * sizeof(float);
  // sync by cache_id, maps to CacheKey
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 1);
  auto it = LlmCommEntityMgr::GetInstance().server_entity_map_.begin();
  EXPECT_EQ(it->second->GetCurState(), FsmState::kFsmProbeState);
  // sync by cache_id, maps to NO CacheKey
  probe_first_time = 0;
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  // unlink.
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSucOfUnLink));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountOfUnlink));
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 0);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, test_copy_cache_success) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);

  int64_t src_cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, src_cache_id, {}), FsmStatus::kFsmSuccess);
  int64_t dst_cache_id = 2;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, dst_cache_id, {}), FsmStatus::kFsmSuccess);
  std::shared_ptr<FlowMsg> flow_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(sizeof(CopyCacheReqInfo))}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> msgs;
  msgs.emplace_back(flow_msg);
  auto &req_info = *static_cast<CopyCacheReqInfo *>(flow_msg->GetTensor()->GetData());
  req_info = CopyCacheReqInfo{};
  req_info.src_cache_id = src_cache_id;
  req_info.dst_cache_id = dst_cache_id;
  EXPECT_EQ(llm_service_flow_func.CopyCache(run_context, msgs), FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(static_cast<FsmStatus>(*output_data), FsmStatus::kFsmSuccess);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, src_cache_id), FsmStatus::kFsmSuccess);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, dst_cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, test_prompt_send_blocks) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(true);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  std::vector<uint64_t> req_ids{UINT64_MAX};
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateBlockCaches(run_context, out_msg, llm_service_flow_func, cache_id, req_ids),
            FsmStatus::kFsmSuccess);

  MOCKER(HcclRawAccept).stubs().will(invoke(HcclRawAcceptSuc));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSuc));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountSuc));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSuc));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  probe_first_time = 0;
  get_count_first_time = true;
  accept_first_time = true;
  sync_kv_req_info->is_pull_block = 1;
  sync_kv_req_info->cache_id = 0;
  sync_kv_req_info->req_id = req_ids[0];
  sync_kv_req_info->buffer_count_per_layer = 1;
  sync_kv_req_info->tensor_index_count = 0;
  sync_kv_req_info->transfer_infos[0].buffer_info.block_index = 4;
  sync_kv_req_info->transfer_infos[0].buffer_info.buffer_len = 10;
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 1);
  sync_kv_req_info->cache_id = -1;
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  // unlink.
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSucOfUnLink));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountOfUnlink));
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 0);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
  RemoveCacheIndexReqInfo req_info1{};
  req_info1.model_id = 0;
  req_info1.req_id = UINT64_MAX;
  EXPECT_EQ(RemoveCacheIndex(run_context, out_msg, llm_service_flow_func, req_info1), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, test_prompt_send_blocks1) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(true);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  std::vector<uint64_t> req_ids{UINT64_MAX};
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateBlockCaches(run_context, out_msg, llm_service_flow_func, cache_id, req_ids),
            FsmStatus::kFsmSuccess);

  MOCKER(HcclRawAccept).stubs().will(invoke(HcclRawAcceptSuc));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSuc));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountSuc));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSuc));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  probe_first_time = 0;
  get_count_first_time = true;
  accept_first_time = true;
  sync_kv_req_info->is_pull_block = 1;
  sync_kv_req_info->cache_id = -1;
  sync_kv_req_info->req_id = req_ids[0];
  sync_kv_req_info->buffer_count_per_layer = 1;
  sync_kv_req_info->tensor_index_count = 0;
  sync_kv_req_info->transfer_infos[0].buffer_info.block_index = 3;
  sync_kv_req_info->transfer_infos[0].buffer_info.buffer_len = 11;
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 1);
  // unlink.
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSucOfUnLink));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountOfUnlink));
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 0);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
  RemoveCacheIndexReqInfo req_info1{};
  req_info1.model_id = 0;
  req_info1.req_id = UINT64_MAX;
  EXPECT_EQ(RemoveCacheIndex(run_context, out_msg, llm_service_flow_func, req_info1), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, test_pull_copy_blocks) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);

  int64_t src_cache_id = 1;
  EXPECT_EQ(AllocateBlockCaches(run_context, out_msg, llm_service_flow_func, src_cache_id, {UINT64_MAX}),
            FsmStatus::kFsmSuccess);

  std::shared_ptr<FlowMsg> pull_flow_msg = run_context->AllocTensorMsg(
      {static_cast<int64_t>(sizeof(PullKvReqInfo) + 1 * sizeof(uint64_t))}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> pull_msgs;
  pull_msgs.emplace_back(pull_flow_msg);
  auto &req_info = *static_cast<PullKvReqInfo *>(pull_flow_msg->GetTensor()->GetData());
  req_info = PullKvReqInfo{};
  req_info.prompt_cache_id = src_cache_id;
  req_info.req_id = UINT64_MAX;
  req_info.model_id = 0UL;
  req_info.prompt_cluster_id = 1001UL;
  req_info.timeout = 3000000UL;
  req_info.cache_id = src_cache_id;
  req_info.block_count = 1;
  req_info.block_indices[0] = 4;
  EXPECT_EQ(llm_service_flow_func.PullCache(run_context, pull_msgs), FLOW_FUNC_SUCCESS);
  auto output_data1 = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(static_cast<FsmStatus>(*output_data1), FsmStatus::kFsmParamInvalid);

  std::shared_ptr<FlowMsg> flow_msg = run_context->AllocTensorMsg(
      {static_cast<int64_t>(sizeof(CopyCacheReqInfo) + 2 * sizeof(BlockCopyInfo))}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> msgs;
  msgs.emplace_back(flow_msg);
  auto &copy_cache_req_info = *static_cast<CopyCacheReqInfo *>(flow_msg->GetTensor()->GetData());
  copy_cache_req_info = CopyCacheReqInfo{};
  copy_cache_req_info.src_cache_id = src_cache_id;
  copy_cache_req_info.dst_cache_id = src_cache_id;
  copy_cache_req_info.copy_block_count = 2;
  copy_cache_req_info.block_copy_infos[0].src_block_index = 0;
  copy_cache_req_info.block_copy_infos[0].dst_block_index = 2;
  copy_cache_req_info.block_copy_infos[1].src_block_index = 1;
  copy_cache_req_info.block_copy_infos[1].dst_block_index = 3;
  CacheEntry cache_entry;
  CacheManager::GetInstance().GetCacheEntry(src_cache_id, cache_entry);
  int8_t *ptr = reinterpret_cast<int8_t *>(cache_entry.tensors[0]->GetTensor()->GetData());
  for (int i = 0; i < 10; ++i) {
    *(ptr + i) = 1;
  }
  for (int i = 10; i < 20; ++i) {
    *(ptr + i) = 2;
  }
  EXPECT_EQ(llm_service_flow_func.CopyCache(run_context, msgs), FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(static_cast<FsmStatus>(*output_data), FsmStatus::kFsmSuccess);
  EXPECT_EQ(*(ptr + 20), 1);
  EXPECT_EQ(*(ptr + 30), 2);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, src_cache_id), FsmStatus::kFsmSuccess);
  RemoveCacheIndexReqInfo req_info1{};
  req_info1.model_id = 0;
  req_info1.req_id = UINT64_MAX;
  EXPECT_EQ(RemoveCacheIndex(run_context, out_msg, llm_service_flow_func, req_info1), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, test_get_cache_success) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  {
    std::shared_ptr<FlowMsg> invalid_flow_msg = run_context->AllocTensorMsg({}, TensorDataType::DT_UINT8);
    EXPECT_EQ(llm_service_flow_func.GetCache(run_context, {invalid_flow_msg}), FLOW_FUNC_SUCCESS);
    auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
    EXPECT_EQ(static_cast<FsmStatus>(*output_data), FsmStatus::kFsmParamInvalid);
  }

  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);
  std::shared_ptr<FlowMsg> flow_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(sizeof(GetCacheReqInfo))}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> msgs;
  msgs.emplace_back(flow_msg);

  auto &get_cache_req = *static_cast<GetCacheReqInfo *>(flow_msg->GetTensor()->GetData());
  get_cache_req.cache_id = 2;
  get_cache_req.tensor_index = 0;
  EXPECT_EQ(llm_service_flow_func.GetCache(run_context, msgs), FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(static_cast<FsmStatus>(*output_data), FsmStatus::kFsmKvNotExist);

  get_cache_req.cache_id = cache_id;
  EXPECT_EQ(llm_service_flow_func.GetCache(run_context, msgs), FLOW_FUNC_SUCCESS);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, allocate_cache_failed) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);

  size_t req_size = sizeof(AllocateCacheReqInfo);
  std::shared_ptr<FlowMsg> flow_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(req_size)}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> msgs;
  msgs.emplace_back(flow_msg);

  auto &req_info = *static_cast<AllocateCacheReqInfo *>(flow_msg->GetTensor()->GetData());
  req_info.cache_id = 666;
  req_info.num_tensors = 1;
  req_info.num_dims = 1;
  req_info.dims[0] = 128000000000;
  EXPECT_EQ(llm_service_flow_func.AllocateCache(run_context, msgs), FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(static_cast<FsmStatus>(*output_data), FsmStatus::kFsmOutOfMemory);
}

TEST_F(LlmServiceFlowFuncUTest, switch_role_success) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);

  size_t req_size = sizeof(SwitchRoleReqInfo);
  std::shared_ptr<FlowMsg> flow_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(req_size)}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> msgs;
  msgs.emplace_back(flow_msg);

  auto &req_info = *static_cast<SwitchRoleReqInfo *>(flow_msg->GetTensor()->GetData());
  req_info.role_id = 0;
  req_info.prompt_listen_ip = 1;
  req_info.prompt_listen_port = 1;
  EXPECT_EQ(llm_service_flow_func.SwitchRole(run_context, msgs), FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(static_cast<FsmStatus>(*output_data), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, test_pull_blocks_validate) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);

  std::shared_ptr<FlowMsg> pull_flow_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(sizeof(PullKvReqInfo))}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> pull_msgs;
  pull_msgs.emplace_back(pull_flow_msg);
  auto &req_info = *static_cast<PullKvReqInfo *>(pull_flow_msg->GetTensor()->GetData());
  req_info = PullKvReqInfo{};
  req_info.prompt_cache_id = 1;
  req_info.req_id = UINT64_MAX;
  req_info.model_id = 0UL;
  req_info.prompt_cluster_id = 1001UL;
  req_info.timeout = 3000000UL;
  req_info.cache_id = 1;
  req_info.block_count = 1;
  EXPECT_EQ(llm_service_flow_func.PullCache(run_context, pull_msgs), FLOW_FUNC_FAILED);
}

//  validate req struct dynamic array size
TEST_F(LlmServiceFlowFuncUTest, test_prompt_send_blocks_validate_req_size) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(true);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  std::vector<uint64_t> req_ids{UINT64_MAX};
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateBlockCaches(run_context, out_msg, llm_service_flow_func, cache_id, req_ids),
            FsmStatus::kFsmSuccess);

  MOCKER(HcclRawAccept).stubs().will(invoke(HcclRawAcceptSuc));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSuc));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountSuc));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSuc));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  probe_first_time = 0;
  get_count_first_time = true;
  accept_first_time = true;
  sync_kv_req_info->is_pull_block = 1;
  sync_kv_req_info->cache_id = -1;
  sync_kv_req_info->req_id = req_ids[0];
  // HcclRawGetCount size < req struct size
  sync_kv_req_info->buffer_count_per_layer = 2;
  sync_kv_req_info->tensor_index_count = 0;
  sync_kv_req_info->transfer_infos[0].buffer_info.block_index = 3;
  sync_kv_req_info->transfer_infos[0].buffer_info.buffer_len = 11;
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 1);

  LlmCommEntityMgr::GetInstance().ClearEntities();
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
  RemoveCacheIndexReqInfo req_info1{};
  req_info1.model_id = 0;
  req_info1.req_id = UINT64_MAX;
  EXPECT_EQ(RemoveCacheIndex(run_context, out_msg, llm_service_flow_func, req_info1), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, test_prompt_validate_not_blocks_req_param) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(true);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  std::vector<uint64_t> req_ids{UINT64_MAX};
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateBlockCaches(run_context, out_msg, llm_service_flow_func, cache_id, req_ids),
            FsmStatus::kFsmSuccess);

  MOCKER(HcclRawAccept).stubs().will(invoke(HcclRawAcceptSuc));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSuc));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountSuc2));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSuc));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  probe_first_time = 0;
  get_count_first_time = true;
  accept_first_time = true;
  sync_kv_req_info->is_pull_block = 0;
  sync_kv_req_info->cache_id = -1;
  sync_kv_req_info->req_id = req_ids[0];
  sync_kv_req_info->buffer_count_per_layer = 2;
  sync_kv_req_info->tensor_index_count = 0;
  sync_kv_req_info->transfer_infos[0].buffer_info.block_index = 1;  // invalid
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 1);
  // unlink.
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSucOfUnLink));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountOfUnlink));
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 0);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
  RemoveCacheIndexReqInfo req_info1{};
  req_info1.model_id = 0;
  req_info1.req_id = UINT64_MAX;
  EXPECT_EQ(RemoveCacheIndex(run_context, out_msg, llm_service_flow_func, req_info1), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, test_copy_blocks_failed) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);

  int64_t src_cache_id = 1;
  EXPECT_EQ(AllocateBlockCaches(run_context, out_msg, llm_service_flow_func, src_cache_id, {UINT64_MAX}),
            FsmStatus::kFsmSuccess);

  std::shared_ptr<FlowMsg> flow_msg = run_context->AllocTensorMsg(
      {static_cast<int64_t>(sizeof(CopyCacheReqInfo) + 1 * sizeof(BlockCopyInfo))}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> msgs;
  msgs.emplace_back(flow_msg);
  auto &req_info = *static_cast<CopyCacheReqInfo *>(flow_msg->GetTensor()->GetData());
  req_info = CopyCacheReqInfo{};
  req_info.src_cache_id = src_cache_id;
  req_info.dst_cache_id = src_cache_id;
  req_info.copy_block_count = 1;
  req_info.block_copy_infos[0].src_block_index = 4;
  req_info.block_copy_infos[0].dst_block_index = 4;
  EXPECT_EQ(llm_service_flow_func.CopyCache(run_context, msgs), FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(static_cast<FsmStatus>(*output_data), FsmStatus::kFsmParamInvalid);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, src_cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, test_pull_blocks_scatter) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);

  int64_t src_cache_id = 1;
  EXPECT_EQ(AllocateBlockCaches(run_context, out_msg, llm_service_flow_func, src_cache_id, {UINT64_MAX}),
            FsmStatus::kFsmSuccess);

  std::shared_ptr<FlowMsg> pull_flow_msg = run_context->AllocTensorMsg(
      {static_cast<int64_t>(sizeof(PullKvReqInfo) + 2 * 65 * sizeof(uint64_t))}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> pull_msgs;
  pull_msgs.emplace_back(pull_flow_msg);
  auto &req_info = *static_cast<PullKvReqInfo *>(pull_flow_msg->GetTensor()->GetData());
  req_info = PullKvReqInfo{};
  req_info.prompt_cache_id = src_cache_id;
  req_info.req_id = UINT64_MAX;
  req_info.model_id = 0UL;
  req_info.prompt_cluster_id = 1001UL;
  req_info.timeout = 3000000UL;
  req_info.cache_id = src_cache_id;
  req_info.block_count = 65;
  req_info.prompt_block_count = 65;
  std::vector<int> blocks = {};
  for (int i = 0; i < 65; ++i) {
    req_info.block_indices[i] = 1;
  }
  for (int i = 0; i < 65; ++i) {
    req_info.block_indices[65 + i] = i;
  }
  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  HcclConn conn = reinterpret_cast<HcclConn>(1);
  LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityClient, conn, local_hccl_addr, remote_hccl_addr,
                                               req_info.prompt_cluster_id);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(req_info.prompt_cluster_id);
  EXPECT_NE(entity, nullptr);
  EXPECT_EQ(llm_service_flow_func.PullCache(run_context, pull_msgs), FLOW_FUNC_SUCCESS);

  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, src_cache_id), FsmStatus::kFsmSuccess);
  RemoveCacheIndexReqInfo req_info1{};
  req_info1.model_id = 0;
  req_info1.req_id = UINT64_MAX;
  EXPECT_EQ(RemoveCacheIndex(run_context, out_msg, llm_service_flow_func, req_info1), FsmStatus::kFsmSuccess);
  LlmCommEntityMgr::GetInstance().ClearEntities();
}

TEST_F(LlmServiceFlowFuncUTest, test_pull_blocks_scatter1) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);

  int64_t src_cache_id = 1;
  EXPECT_EQ(AllocateBlockCaches(run_context, out_msg, llm_service_flow_func, src_cache_id, {UINT64_MAX}),
            FsmStatus::kFsmSuccess);

  std::shared_ptr<FlowMsg> pull_flow_msg = run_context->AllocTensorMsg(
      {static_cast<int64_t>(sizeof(PullKvReqInfo) + 2 * 66 * sizeof(uint64_t))}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> pull_msgs;
  pull_msgs.emplace_back(pull_flow_msg);
  auto &req_info = *static_cast<PullKvReqInfo *>(pull_flow_msg->GetTensor()->GetData());
  req_info = PullKvReqInfo{};
  req_info.prompt_cache_id = src_cache_id;
  req_info.req_id = UINT64_MAX;
  req_info.model_id = 0UL;
  req_info.prompt_cluster_id = 1001UL;
  req_info.timeout = 3000000UL;
  req_info.cache_id = src_cache_id;
  req_info.block_count = 66;
  req_info.prompt_block_count = 66;
  std::vector<int> blocks = {};
  // 63 not continuous
  for (int i = 0; i < 63; ++i) {
    req_info.block_indices[i] = 0;
  }
  // just need 64 slots, one scatter
  req_info.block_indices[63] = 2;
  req_info.block_indices[64] = 3;
  // new scatter
  req_info.block_indices[65] = 0;
  // prompt is continuous
  for (int i = 0; i < 66; ++i) {
    req_info.block_indices[66 + i] = i;
  }
  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  HcclConn conn = reinterpret_cast<HcclConn>(1);
  LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityClient, conn, local_hccl_addr, remote_hccl_addr,
                                               req_info.prompt_cluster_id);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(req_info.prompt_cluster_id);
  EXPECT_NE(entity, nullptr);
  EXPECT_EQ(llm_service_flow_func.PullCache(run_context, pull_msgs), FLOW_FUNC_SUCCESS);

  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, src_cache_id), FsmStatus::kFsmSuccess);
  RemoveCacheIndexReqInfo req_info1{};
  req_info1.model_id = 0;
  req_info1.req_id = UINT64_MAX;
  EXPECT_EQ(RemoveCacheIndex(run_context, out_msg, llm_service_flow_func, req_info1), FsmStatus::kFsmSuccess);
  LlmCommEntityMgr::GetInstance().ClearEntities();
}

TEST_F(LlmServiceFlowFuncUTest, test_copy_large_data) {
  std::vector<int32_t> dst_buffer(1024 * 1024 * 1024 + 1);
  std::vector<int32_t> src_buffer(1024 * 1024 * 1024 + 1);
  src_buffer.front() = 1;
  src_buffer.back() = 127;
  const auto size = sizeof(int32_t) * dst_buffer.size();
  EXPECT_EQ(D2DCopy(dst_buffer.data(), size, src_buffer.data(), size), DRV_ERROR_NONE);
  EXPECT_EQ(dst_buffer.front(), 1);
  EXPECT_EQ(dst_buffer.back(), 127);
}

TEST_F(LlmServiceFlowFuncUTest, test_force_unlink) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  uint32_t cluster_num = 1U;
  uint32_t ip_num = 2U;
  uint64_t cluster_id = 1U;
  uint64_t size = sizeof(UpdateLinkReqInfo) + cluster_num * sizeof(ClusterInfo) + ip_num * sizeof(IpInfo);
  std::shared_ptr<FlowMsg> flow_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(size)}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> msgs;
  msgs.emplace_back(flow_msg);

  auto link_req_ptr = static_cast<UpdateLinkReqInfo *>(flow_msg->GetTensor()->GetData());
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kLink);
  link_req_ptr->cluster_num = cluster_num;
  link_req_ptr->force_flag = 1U;
  link_req_ptr->timeout = 10000;
  auto cluster_addr = PtrToValue(flow_msg->GetTensor()->GetData()) + sizeof(UpdateLinkReqInfo);
  auto cluster_info_ptr = static_cast<ClusterInfo *>(ValueToPtr(cluster_addr));
  cluster_info_ptr->ip_num = ip_num;
  cluster_info_ptr->remote_cluster_id = cluster_id;
  auto ip1_addr = cluster_addr + sizeof(ClusterInfo);
  auto ip1_ptr = static_cast<IpInfo *>(ValueToPtr(ip1_addr));
  ip1_ptr->local_ip = 1;
  ip1_ptr->remote_ip = 2;
  auto ip2_addr = cluster_addr + sizeof(ClusterInfo) + sizeof(IpInfo);
  auto ip2_ptr = static_cast<IpInfo *>(ValueToPtr(ip2_addr));
  ip2_ptr->local_ip = 3;
  ip2_ptr->remote_ip = 4;

  // link suc
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSucOfLink));
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kLink);
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmSuccess));
  MOCKER(HcclRawTestSome).reset();

  // unlink suc
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuccForUnlink));
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kUnlink);

  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(LlmServiceFlowFuncUTest, test_force_unlink_fail) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  uint32_t cluster_num = 1U;
  uint32_t ip_num = 2U;
  uint64_t cluster_id = 1U;
  uint64_t size = sizeof(UpdateLinkReqInfo) + cluster_num * sizeof(ClusterInfo) + ip_num * sizeof(IpInfo);
  std::shared_ptr<FlowMsg> flow_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(size)}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> msgs;
  msgs.emplace_back(flow_msg);

  auto link_req_ptr = static_cast<UpdateLinkReqInfo *>(flow_msg->GetTensor()->GetData());
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kLink);
  link_req_ptr->cluster_num = cluster_num;
  link_req_ptr->force_flag = 1U;
  link_req_ptr->timeout = 10000;
  auto cluster_addr = PtrToValue(flow_msg->GetTensor()->GetData()) + sizeof(UpdateLinkReqInfo);
  auto cluster_info_ptr = static_cast<ClusterInfo *>(ValueToPtr(cluster_addr));
  cluster_info_ptr->ip_num = ip_num;
  cluster_info_ptr->remote_cluster_id = cluster_id;
  auto ip1_addr = cluster_addr + sizeof(ClusterInfo);
  auto ip1_ptr = static_cast<IpInfo *>(ValueToPtr(ip1_addr));
  ip1_ptr->local_ip = 1;
  ip1_ptr->remote_ip = 2;
  auto ip2_addr = cluster_addr + sizeof(ClusterInfo) + sizeof(IpInfo);
  auto ip2_ptr = static_cast<IpInfo *>(ValueToPtr(ip2_addr));
  ip2_ptr->local_ip = 3;
  ip2_ptr->remote_ip = 4;

  // link suc
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSucOfLink));
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kLink);
  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmSuccess));
  MOCKER(HcclRawTestSome).reset();

  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuccForUnlink));
  MOCKER(HcclRawForceClose).stubs().will(invoke(HcclRawForceCloseFail));
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kUnlink);
  ret = llm_service_flow_func.UnlinkData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmUnlinkFailed));
  MOCKER(HcclRawForceClose).reset();
}

TEST_F(LlmServiceFlowFuncUTest, push_kv_cache_success) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  size_t total_len = sizeof(TransferKvReqInfo);
  std::shared_ptr<FlowMsg> input_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(total_len)}, TensorDataType::DT_INT8);
  input_msgs.emplace_back(input_msg);
  const Tensor *req_info_tensor = input_msg->GetTensor();
  auto *req_info = static_cast<TransferKvReqInfo *>(req_info_tensor->GetData());
  req_info->cache_id = cache_id;
  req_info->src_batch_index = 0;
  req_info->src_layer_index = 0;
  req_info->dst_cluster_id = 1;
  req_info->dst_key_addr = 8000UL;
  req_info->dst_value_addr = 8001UL;
  req_info->block_len = 0;
  req_info->block_count = 0;
  req_info->prompt_block_count = 0;
  req_info->timeout = 3000000UL;
  req_info->dst_cache_id = -1;
  req_info->tensor_num_per_layer = 2;
  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  auto conn = reinterpret_cast<HcclConn>(1);
  auto new_entity = LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityServer, conn, local_hccl_addr,
                                                                 remote_hccl_addr, req_info->dst_cluster_id);
  LlmCommEntityMgr::GetInstance().AddClientEntityMap(req_info->dst_cluster_id, new_entity);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(req_info->dst_cluster_id);
  EXPECT_NE(entity, nullptr);
  // mock for hccl api
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSucOfLink));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeStub));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountStubForTransferKv));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImRecvStubForTransferKv));

  auto sync_ret = llm_service_flow_func.TransferCache(run_context, input_msgs);
  EXPECT_EQ(sync_ret, FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmSuccess));
  LlmCommEntityMgr::GetInstance().ClearEntities();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).reset();
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, test_receive_push_kv_success) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(false);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  auto conn = reinterpret_cast<HcclConn>(1);
  LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityClient, conn, local_hccl_addr, remote_hccl_addr, 0);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(0);
  EXPECT_NE(entity, nullptr);

  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSucForRecvTransferKv));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountStubForRecvTransferKv));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSucForRecvTransferKv));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  recv_push_kv_probe_first_time = 0U;
  recv_push_get_count_first_time = 0U;
  recv_push_imrecv_first_time = 0U;
  CacheEntry cache_entry;
  const auto cache_entry_exist = CacheManager::GetInstance().GetCacheEntry(cache_id, cache_entry);
  EXPECT_EQ(cache_entry_exist, true);
  push_kv_req_info->key_addr = reinterpret_cast<uintptr_t>(cache_entry.tensors[0]->GetTensor()->GetData());
  push_kv_req_info->value_addr = reinterpret_cast<uintptr_t>(cache_entry.tensors[1]->GetTensor()->GetData());
  push_kv_req_info->max_size = 1024;
  push_kv_req_info->send_nums_per_tensor = 1;
  push_kv_req_info->total_slot_nums = 1;
  push_kv_req_info->slot_infos[0].slot_size = 1024;
  push_kv_req_info->slot_infos[0].slot_nums_per_transfer = 1;
  push_kv_req_info->slot_infos[0].slot_offset = 0;

  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(entity->GetCurState(), FsmState::kFsmProbeState);
  uint32_t *ptr = reinterpret_cast<uint32_t *>(cache_entry.tensors[0]->GetTensor()->GetData());
  UDF_LOG_INFO("get %p value", ptr);
  EXPECT_EQ(*ptr, 2);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, test_receive_blocks) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(false);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  int64_t cache_id = 1;
  EXPECT_EQ(AllocateBlockCaches(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  auto conn = reinterpret_cast<HcclConn>(1);
  LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityClient, conn, local_hccl_addr, remote_hccl_addr, 0);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(0);
  EXPECT_NE(entity, nullptr);

  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSucForRecvBlocksTransferKv));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountStubForRecvBlocksTransferKv));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSucForRecvBlocksTransferKv));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  recv_push_kv_probe_blocks_first_time = 0U;
  recv_push_blocks_get_count_first_time = 0U;
  recv_push_imrecv_blocks_first_time = 0U;
  CacheEntry cache_entry;
  const auto cache_entry_exist = CacheManager::GetInstance().GetCacheEntry(cache_id, cache_entry);
  EXPECT_EQ(cache_entry_exist, true);
  push_kv_req_info->key_addr = reinterpret_cast<uintptr_t>(cache_entry.tensors[0]->GetTensor()->GetData());
  push_kv_req_info->value_addr = reinterpret_cast<uintptr_t>(cache_entry.tensors[0]->GetTensor()->GetData());
  push_kv_req_info->max_size = 40;
  push_kv_req_info->send_nums_per_tensor = 2;
  push_kv_req_info->total_slot_nums = 3;
  push_kv_req_info->slot_infos[0].slot_offset = 0;
  push_kv_req_info->slot_infos[0].slot_size = 10;
  push_kv_req_info->slot_infos[0].slot_nums_per_transfer = 2;
  push_kv_req_info->slot_infos[1].slot_offset = 10;
  push_kv_req_info->slot_infos[1].slot_size = 10;
  push_kv_req_info->slot_infos[1].slot_nums_per_transfer = 2;
  push_kv_req_info->slot_infos[2].slot_offset = 30;
  push_kv_req_info->slot_infos[2].slot_size = 10;
  push_kv_req_info->slot_infos[2].slot_nums_per_transfer = 1;

  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(entity->GetCurState(), FsmState::kFsmProbeState);
  uint32_t *ptr = reinterpret_cast<uint32_t *>(cache_entry.tensors[0]->GetTensor()->GetData());
  UDF_LOG_INFO("get %p value", ptr);
  EXPECT_EQ(*ptr, 2);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, transfer_blocks_cache_success) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateBlockCaches(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  size_t total_len = sizeof(TransferKvReqInfo) + 4 * sizeof(uint64_t);
  std::shared_ptr<FlowMsg> input_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(total_len)}, TensorDataType::DT_INT8);
  input_msgs.emplace_back(input_msg);
  const Tensor *req_info_tensor = input_msg->GetTensor();
  auto *req_info = static_cast<TransferKvReqInfo *>(req_info_tensor->GetData());
  req_info->cache_id = cache_id;
  req_info->src_batch_index = 0;
  req_info->src_layer_index = 0;
  req_info->dst_cluster_id = 1;
  req_info->dst_key_addr = 8000UL;
  req_info->dst_value_addr = 8001UL;
  req_info->block_len = 10;
  req_info->block_count = 2;
  req_info->prompt_block_count = 2;
  req_info->timeout = 3000000UL;
  req_info->block_indices[0] = 0;
  req_info->block_indices[1] = 1;
  req_info->block_indices[2] = 2;
  req_info->block_indices[3] = 3;
  req_info->dst_cache_id = -1;
  req_info->tensor_num_per_layer = 2;
  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  auto conn = reinterpret_cast<HcclConn>(1);
  auto new_entity = LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityServer, conn, local_hccl_addr,
                                                                 remote_hccl_addr, req_info->dst_cluster_id);
  LlmCommEntityMgr::GetInstance().AddClientEntityMap(req_info->dst_cluster_id, new_entity);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(req_info->dst_cluster_id);
  EXPECT_NE(entity, nullptr);
  // mock for hccl api
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSucOfLink));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeStub));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountStubForTransferKv));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImRecvStubForTransferKv));

  auto sync_ret = llm_service_flow_func.TransferCache(run_context, input_msgs);
  EXPECT_EQ(sync_ret, FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmSuccess));
  LlmCommEntityMgr::GetInstance().ClearEntities();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).reset();
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, transfer_cache_validate_mem_not_valid) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  size_t total_len = sizeof(TransferKvReqInfo);
  std::shared_ptr<FlowMsg> input_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(total_len)}, TensorDataType::DT_INT8);
  input_msgs.emplace_back(input_msg);
  const Tensor *req_info_tensor = input_msg->GetTensor();
  auto *req_info = static_cast<TransferKvReqInfo *>(req_info_tensor->GetData());
  req_info->cache_id = cache_id;
  req_info->src_batch_index = 0;
  req_info->src_layer_index = 0;
  req_info->dst_cluster_id = 1;
  req_info->dst_key_addr = 8000UL;
  req_info->dst_value_addr = 8001UL;
  req_info->block_len = 0;
  req_info->block_count = 0;
  req_info->prompt_block_count = 0;
  req_info->timeout = 3000000UL;
  req_info->dst_cache_id = -1;
  req_info->tensor_num_per_layer = 2;
  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  auto conn = reinterpret_cast<HcclConn>(1);
  auto new_entity = LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityServer, conn, local_hccl_addr,
                                                                 remote_hccl_addr, req_info->dst_cluster_id);
  LlmCommEntityMgr::GetInstance().AddClientEntityMap(req_info->dst_cluster_id, new_entity);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(req_info->dst_cluster_id);
  EXPECT_NE(entity, nullptr);
  // mock for hccl api
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSucOfLink));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeStub));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountStubForTransferKv));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImRecvStubForTransferKvValidateMem));

  auto sync_ret = llm_service_flow_func.TransferCache(run_context, input_msgs);
  EXPECT_EQ(sync_ret, FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmParamInvalid));
  LlmCommEntityMgr::GetInstance().ClearEntities();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).reset();
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, transfer_cache_send_req_fail) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  size_t total_len = sizeof(TransferKvReqInfo);
  std::shared_ptr<FlowMsg> input_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(total_len)}, TensorDataType::DT_INT8);
  input_msgs.emplace_back(input_msg);
  const Tensor *req_info_tensor = input_msg->GetTensor();
  auto *req_info = static_cast<TransferKvReqInfo *>(req_info_tensor->GetData());
  req_info->cache_id = cache_id;
  req_info->src_batch_index = 0;
  req_info->src_layer_index = 0;
  req_info->dst_cluster_id = 1;
  req_info->dst_key_addr = 8000UL;
  req_info->dst_value_addr = 8001UL;
  req_info->block_len = 0;
  req_info->block_count = 0;
  req_info->prompt_block_count = 0;
  req_info->timeout = 3000000UL;
  req_info->dst_cache_id = -1;
  req_info->tensor_num_per_layer = 2;
  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  auto conn = reinterpret_cast<HcclConn>(1);
  auto new_entity = LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityServer, conn, local_hccl_addr,
                                                                 remote_hccl_addr, req_info->dst_cluster_id);
  LlmCommEntityMgr::GetInstance().AddClientEntityMap(req_info->dst_cluster_id, new_entity);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(req_info->dst_cluster_id);
  EXPECT_NE(entity, nullptr);
  // mock for hccl api
  MOCKER(HcclRawIsend).reset();
  MOCKER(HcclRawIsend).stubs().will(invoke(HcclRawIsendFail));

  auto sync_ret = llm_service_flow_func.TransferCache(run_context, input_msgs);
  EXPECT_EQ(sync_ret, FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmFailed));
  LlmCommEntityMgr::GetInstance().ClearEntities();
  MOCKER(HcclRawIsend).reset();
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, transfer_cache_send_cache_fail) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  size_t total_len = sizeof(TransferKvReqInfo);
  std::shared_ptr<FlowMsg> input_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(total_len)}, TensorDataType::DT_INT8);
  input_msgs.emplace_back(input_msg);
  const Tensor *req_info_tensor = input_msg->GetTensor();
  auto *req_info = static_cast<TransferKvReqInfo *>(req_info_tensor->GetData());
  req_info->cache_id = cache_id;
  req_info->src_batch_index = 0;
  req_info->src_layer_index = 0;
  req_info->dst_cluster_id = 1;
  req_info->dst_key_addr = 8000UL;
  req_info->dst_value_addr = 8001UL;
  req_info->block_len = 0;
  req_info->block_count = 0;
  req_info->prompt_block_count = 0;
  req_info->timeout = 3000000UL;
  req_info->dst_cache_id = -1;
  req_info->tensor_num_per_layer = 2;
  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  auto conn = reinterpret_cast<HcclConn>(1);
  auto new_entity = LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityServer, conn, local_hccl_addr,
                                                                 remote_hccl_addr, req_info->dst_cluster_id);
  LlmCommEntityMgr::GetInstance().AddClientEntityMap(req_info->dst_cluster_id, new_entity);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(req_info->dst_cluster_id);
  EXPECT_NE(entity, nullptr);
  // mock for hccl api
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawIsend).reset();
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSucOfLink));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeStub));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountStubForTransferKv));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImRecvStubForTransferKv));
  MOCKER(HcclRawIsend).stubs().will(invoke(HcclRawIsendFailForTransferKv));
  send_first_time = true;

  auto sync_ret = llm_service_flow_func.TransferCache(run_context, input_msgs);
  EXPECT_EQ(sync_ret, FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmHcclFailed));
  LlmCommEntityMgr::GetInstance().ClearEntities();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).reset();
  MOCKER(HcclRawIsend).reset();
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, test_receive_push_kv_validate_key_mem_fail) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(false);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  auto conn = reinterpret_cast<HcclConn>(1);
  LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityClient, conn, local_hccl_addr, remote_hccl_addr, 0);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(0);
  EXPECT_NE(entity, nullptr);

  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSucForRecvTransferKv));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountStubForRecvTransferKv));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSucForRecvTransferKv));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  recv_push_kv_probe_first_time = 0U;
  recv_push_get_count_first_time = 0U;
  recv_push_imrecv_first_time = 0U;
  CacheEntry cache_entry;
  const auto cache_entry_exist = CacheManager::GetInstance().GetCacheEntry(cache_id, cache_entry);
  EXPECT_EQ(cache_entry_exist, true);
  push_kv_req_info->key_addr = 1001;
  push_kv_req_info->value_addr = reinterpret_cast<uintptr_t>(cache_entry.tensors[1]->GetTensor()->GetData());
  push_kv_req_info->max_size = 1024;
  push_kv_req_info->send_nums_per_tensor = 1;
  push_kv_req_info->total_slot_nums = 1;
  push_kv_req_info->slot_infos[0].slot_size = 1024;
  push_kv_req_info->slot_infos[0].slot_nums_per_transfer = 1;
  push_kv_req_info->slot_infos[0].slot_offset = 0;

  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(entity->GetCurState(), FsmState::kFsmProbeState);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, test_receive_push_kv_validate_value_mem_fail) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(false);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  auto conn = reinterpret_cast<HcclConn>(1);
  LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityClient, conn, local_hccl_addr, remote_hccl_addr, 0);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(0);
  EXPECT_NE(entity, nullptr);

  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSucForRecvTransferKv));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountStubForRecvTransferKv));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSucForRecvTransferKv));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  recv_push_kv_probe_first_time = 0U;
  recv_push_get_count_first_time = 0U;
  recv_push_imrecv_first_time = 0U;
  CacheEntry cache_entry;
  const auto cache_entry_exist = CacheManager::GetInstance().GetCacheEntry(cache_id, cache_entry);
  EXPECT_EQ(cache_entry_exist, true);
  push_kv_req_info->key_addr = reinterpret_cast<uintptr_t>(cache_entry.tensors[0]->GetTensor()->GetData());
  push_kv_req_info->value_addr = 1001;
  push_kv_req_info->max_size = 1024;
  push_kv_req_info->send_nums_per_tensor = 1;
  push_kv_req_info->total_slot_nums = 1;
  push_kv_req_info->slot_infos[0].slot_size = 1024;
  push_kv_req_info->slot_infos[0].slot_nums_per_transfer = 1;
  push_kv_req_info->slot_infos[0].slot_offset = 0;

  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(entity->GetCurState(), FsmState::kFsmProbeState);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, transfer_cache_validate_param_cache_not_exist) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  size_t total_len = sizeof(TransferKvReqInfo);
  std::shared_ptr<FlowMsg> input_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(total_len)}, TensorDataType::DT_INT8);
  input_msgs.emplace_back(input_msg);
  const Tensor *req_info_tensor = input_msg->GetTensor();
  auto *req_info = static_cast<TransferKvReqInfo *>(req_info_tensor->GetData());
  req_info->cache_id = cache_id + 1;
  req_info->src_batch_index = 0;
  req_info->src_layer_index = 0;
  req_info->dst_cluster_id = 1;
  req_info->dst_key_addr = 8000UL;
  req_info->dst_value_addr = 8001UL;
  req_info->block_len = 0;
  req_info->block_count = 0;
  req_info->prompt_block_count = 0;
  req_info->timeout = 3000000UL;
  req_info->dst_cache_id = -1;
  req_info->tensor_num_per_layer = 2;
  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  auto conn = reinterpret_cast<HcclConn>(1);
  auto new_entity = LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityServer, conn, local_hccl_addr,
                                                                 remote_hccl_addr, req_info->dst_cluster_id);
  LlmCommEntityMgr::GetInstance().AddClientEntityMap(req_info->dst_cluster_id, new_entity);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(req_info->dst_cluster_id);
  EXPECT_NE(entity, nullptr);

  auto sync_ret = llm_service_flow_func.TransferCache(run_context, input_msgs);
  EXPECT_EQ(sync_ret, FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmKvNotExist));

  req_info->cache_id = cache_id;
  req_info->block_len = kSecondDimForKvShape * 4 + 1;
  sync_ret = llm_service_flow_func.TransferCache(run_context, input_msgs);
  EXPECT_EQ(sync_ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmParamInvalid));

  req_info->src_batch_index = kFirstDimForKvShape + 1;
  sync_ret = llm_service_flow_func.TransferCache(run_context, input_msgs);
  EXPECT_EQ(sync_ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmParamInvalid));

  req_info->src_layer_index = 1;
  sync_ret = llm_service_flow_func.TransferCache(run_context, input_msgs);
  EXPECT_EQ(sync_ret, FLOW_FUNC_SUCCESS);
  output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmParamInvalid));

  LlmCommEntityMgr::GetInstance().ClearEntities();
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, transfer_blocks_validate_block_len) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateBlockCaches(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  size_t total_len = sizeof(TransferKvReqInfo) + 4 * sizeof(uint64_t);
  std::shared_ptr<FlowMsg> input_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(total_len)}, TensorDataType::DT_INT8);
  input_msgs.emplace_back(input_msg);
  const Tensor *req_info_tensor = input_msg->GetTensor();
  auto *req_info = static_cast<TransferKvReqInfo *>(req_info_tensor->GetData());
  req_info->cache_id = cache_id;
  req_info->src_batch_index = 0;
  req_info->src_layer_index = 0;
  req_info->dst_cluster_id = 1;
  req_info->dst_key_addr = 8000UL;
  req_info->dst_value_addr = 8001UL;
  req_info->block_len = 9;
  req_info->block_count = 2;
  req_info->prompt_block_count = 2;
  req_info->timeout = 3000000UL;
  req_info->block_indices[0] = 0;
  req_info->block_indices[1] = 1;
  req_info->block_indices[2] = 2;
  req_info->block_indices[3] = 3;
  req_info->dst_cache_id = -1;
  req_info->tensor_num_per_layer = 2;
  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  auto conn = reinterpret_cast<HcclConn>(1);
  auto new_entity = LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityServer, conn, local_hccl_addr,
                                                                 remote_hccl_addr, req_info->dst_cluster_id);
  LlmCommEntityMgr::GetInstance().AddClientEntityMap(req_info->dst_cluster_id, new_entity);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(req_info->dst_cluster_id);
  EXPECT_NE(entity, nullptr);
  // mock for hccl api
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSucOfLink));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeStub));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountStubForTransferKv));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImRecvStubForTransferKv));

  auto sync_ret = llm_service_flow_func.TransferCache(run_context, input_msgs);
  EXPECT_EQ(sync_ret, FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmParamInvalid));
  LlmCommEntityMgr::GetInstance().ClearEntities();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).reset();
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, test_receive_push_kv_egain) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(false);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  auto conn = reinterpret_cast<HcclConn>(1);
  LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityClient, conn, local_hccl_addr, remote_hccl_addr, 0);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(0);
  EXPECT_NE(entity, nullptr);

  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSucForRecvTransferKv));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountStubForRecvTransferKv));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSucForRecvTransferKvEgain));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  recv_push_kv_probe_first_time = 0U;
  recv_push_get_count_first_time = 0U;
  recv_push_imrecv_first_time = 0U;
  CacheEntry cache_entry;
  const auto cache_entry_exist = CacheManager::GetInstance().GetCacheEntry(cache_id, cache_entry);
  EXPECT_EQ(cache_entry_exist, true);
  push_kv_req_info->key_addr = reinterpret_cast<uintptr_t>(cache_entry.tensors[0]->GetTensor()->GetData());
  push_kv_req_info->value_addr = reinterpret_cast<uintptr_t>(cache_entry.tensors[1]->GetTensor()->GetData());
  push_kv_req_info->max_size = 1024;
  push_kv_req_info->send_nums_per_tensor = 1;
  push_kv_req_info->total_slot_nums = 1;
  push_kv_req_info->slot_infos[0].slot_size = 1024;
  push_kv_req_info->slot_infos[0].slot_nums_per_transfer = 1;
  push_kv_req_info->slot_infos[0].slot_offset = 0;

  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(entity->GetCurState(), FsmState::kFsmErrorState);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, transfer_blocks_scatter) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateBlockCaches(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  size_t total_len = sizeof(TransferKvReqInfo) + 4 * sizeof(uint64_t);
  std::shared_ptr<FlowMsg> input_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(total_len)}, TensorDataType::DT_INT8);
  input_msgs.emplace_back(input_msg);
  const Tensor *req_info_tensor = input_msg->GetTensor();
  auto *req_info = static_cast<TransferKvReqInfo *>(req_info_tensor->GetData());
  req_info->cache_id = cache_id;
  req_info->src_batch_index = 0;
  req_info->src_layer_index = 0;
  req_info->dst_cluster_id = 1;
  req_info->dst_key_addr = 8000UL;
  req_info->dst_value_addr = 8001UL;
  req_info->block_len = 10;
  req_info->block_count = 2;
  req_info->prompt_block_count = 2;
  req_info->timeout = 3000000UL;
  req_info->block_indices[0] = 1;
  req_info->block_indices[1] = 3;
  req_info->block_indices[2] = 0;
  req_info->block_indices[3] = 2;
  req_info->dst_cache_id = -1;
  req_info->tensor_num_per_layer = 2;
  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  auto conn = reinterpret_cast<HcclConn>(1);
  auto new_entity = LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityServer, conn, local_hccl_addr,
                                                                 remote_hccl_addr, req_info->dst_cluster_id);
  LlmCommEntityMgr::GetInstance().AddClientEntityMap(req_info->dst_cluster_id, new_entity);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(req_info->dst_cluster_id);
  EXPECT_NE(entity, nullptr);
  // mock for hccl api
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSucOfLink));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeStub));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountStubForTransferKv));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImRecvStubForTransferKv));

  auto sync_ret = llm_service_flow_func.TransferCache(run_context, input_msgs);
  EXPECT_EQ(sync_ret, FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmSuccess));
  LlmCommEntityMgr::GetInstance().ClearEntities();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).reset();
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, transfer_blocks_no_prompt_block_len) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateBlockCaches(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  size_t total_len = sizeof(TransferKvReqInfo) + 2 * sizeof(uint64_t);
  std::shared_ptr<FlowMsg> input_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(total_len)}, TensorDataType::DT_INT8);
  input_msgs.emplace_back(input_msg);
  const Tensor *req_info_tensor = input_msg->GetTensor();
  auto *req_info = static_cast<TransferKvReqInfo *>(req_info_tensor->GetData());
  req_info->cache_id = cache_id;
  req_info->src_batch_index = 0;
  req_info->src_layer_index = 0;
  req_info->dst_cluster_id = 1;
  req_info->dst_key_addr = 8000UL;
  req_info->dst_value_addr = 8001UL;
  req_info->block_len = 10;
  req_info->block_count = 2;
  req_info->prompt_block_count = 0;
  req_info->timeout = 3000000UL;
  req_info->block_indices[0] = 0;
  req_info->block_indices[1] = 1;
  req_info->dst_cache_id = -1;
  req_info->tensor_num_per_layer = 2;
  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  auto conn = reinterpret_cast<HcclConn>(1);
  auto new_entity = LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityServer, conn, local_hccl_addr,
                                                                 remote_hccl_addr, req_info->dst_cluster_id);
  LlmCommEntityMgr::GetInstance().AddClientEntityMap(req_info->dst_cluster_id, new_entity);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(req_info->dst_cluster_id);
  EXPECT_NE(entity, nullptr);
  // mock for hccl api
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSucOfLink));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeStub));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountStubForTransferKv));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImRecvStubForTransferKv));

  auto sync_ret = llm_service_flow_func.TransferCache(run_context, input_msgs);
  EXPECT_EQ(sync_ret, FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmParamInvalid));
  LlmCommEntityMgr::GetInstance().ClearEntities();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).reset();
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
  LlmCommEntityMgr::GetInstance().ClearEntities();
}

TEST_F(LlmServiceFlowFuncUTest, test_pull_blocks_no_prompt_block_len) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);

  int64_t src_cache_id = 1;
  EXPECT_EQ(AllocateBlockCaches(run_context, out_msg, llm_service_flow_func, src_cache_id, {}), FsmStatus::kFsmSuccess);

  std::shared_ptr<FlowMsg> pull_flow_msg = run_context->AllocTensorMsg(
      {static_cast<int64_t>(sizeof(PullKvReqInfo) + 65 * sizeof(uint64_t))}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> pull_msgs;
  pull_msgs.emplace_back(pull_flow_msg);
  auto &req_info = *static_cast<PullKvReqInfo *>(pull_flow_msg->GetTensor()->GetData());
  req_info = PullKvReqInfo{};
  req_info.prompt_cache_id = src_cache_id;
  req_info.req_id = UINT64_MAX;
  req_info.model_id = 0UL;
  req_info.prompt_cluster_id = 1001UL;
  req_info.timeout = 3000000UL;
  req_info.cache_id = src_cache_id;
  req_info.block_count = 65;
  req_info.prompt_block_count = 0;
  std::vector<int> blocks = {};
  for (int i = 0; i < 65; ++i) {
    req_info.block_indices[i] = 1;
  }
  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  HcclConn conn = reinterpret_cast<HcclConn>(1);
  LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityClient, conn, local_hccl_addr, remote_hccl_addr,
                                               req_info.prompt_cluster_id);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(req_info.prompt_cluster_id);
  EXPECT_NE(entity, nullptr);
  EXPECT_EQ(llm_service_flow_func.PullCache(run_context, pull_msgs), FLOW_FUNC_SUCCESS);

  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmParamInvalid));

  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, src_cache_id), FsmStatus::kFsmSuccess);
  LlmCommEntityMgr::GetInstance().ClearEntities();
}

HcclResult HcclRawImprobeTwoTimes(HcclConn conn, int32_t *flag, HcclMessage *msg, HcclStatus *status) {
  if (probe_first_time == 0) {
    *flag = 0;
  } else if (probe_first_time == 1) {
    *flag = 1;
  } else {
    *flag = 0;
  }
  probe_first_time++;
  return HCCL_SUCCESS;
}
TEST_F(LlmServiceFlowFuncUTest, test_server_probe_2_times_link_suc) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(true);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  MOCKER(HcclRawAccept).stubs().will(invoke(HcclRawAcceptSuc));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeTwoTimes));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountSuc));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSuc));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  probe_first_time = 0;
  get_count_first_time = true;
  accept_first_time = true;
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 1);
  auto it = LlmCommEntityMgr::GetInstance().server_entity_map_.begin();
  EXPECT_EQ(it->second->GetCurState(), FsmState::kFsmProbeState);
  // unlink.
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSucOfUnLink));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountOfUnlink));
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 0);
}

HcclResult HcclRawAcceptFailed(HcclConn listen_conn, HcclAddr *accept_addr, HcclConn *accept_conn) {
  if (accept_first_time) {
    accept_first_time = false;
    return HCCL_E_INTERNAL;
  }
  return HCCL_SUCCESS;
}

bool open_first_time = true;

HcclResult HcclRawOpenFail1(HcclConn *conn) {
  if (open_first_time) {
    open_first_time = false;
    return HCCL_E_INTERNAL;
  }
  return HCCL_SUCCESS;
}
TEST_F(LlmServiceFlowFuncUTest, test_server_accept_failed) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(true);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  MOCKER(HcclRawAccept).stubs().will(invoke(HcclRawAcceptFailed));
  MOCKER(HcclRawOpen).stubs().will(invoke(HcclRawOpenFail1));
  accept_first_time = true;
  open_first_time = true;
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 0);
}

HcclResult HcclRawGetCountStubForRecvTransferKvFail1(const HcclStatus *status, HcclDataType data_type, int *count) {
  if (recv_push_get_count_first_time == 0U) {
    *count = sizeof(SyncKvMetaInfo);
    recv_push_get_count_first_time++;
  } else if (recv_push_get_count_first_time <= 2U) {
    *count = kFirstDimForKvShape * kSecondDimForKvShape * 4;
    recv_push_get_count_first_time++;
  }
  return HCCL_SUCCESS;
}
TEST_F(LlmServiceFlowFuncUTest, test_receive_push_kv_probe_msg_invalid) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(false);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  auto conn = reinterpret_cast<HcclConn>(1);
  LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityClient, conn, local_hccl_addr, remote_hccl_addr, 0);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(0);
  EXPECT_NE(entity, nullptr);

  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSucForRecvTransferKv));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountStubForRecvTransferKvFail1));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSucForRecvTransferKv));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  recv_push_kv_probe_first_time = 0U;
  recv_push_get_count_first_time = 0U;
  recv_push_imrecv_first_time = 0U;
  CacheEntry cache_entry;
  const auto cache_entry_exist = CacheManager::GetInstance().GetCacheEntry(cache_id, cache_entry);
  EXPECT_EQ(cache_entry_exist, true);
  push_kv_req_info->key_addr = reinterpret_cast<uintptr_t>(cache_entry.tensors[0]->GetTensor()->GetData());
  push_kv_req_info->value_addr = reinterpret_cast<uintptr_t>(cache_entry.tensors[1]->GetTensor()->GetData());
  push_kv_req_info->max_size = 1024;
  push_kv_req_info->send_nums_per_tensor = 1;
  push_kv_req_info->total_slot_nums = 1;
  push_kv_req_info->slot_infos[0].slot_size = 1024;
  push_kv_req_info->slot_infos[0].slot_nums_per_transfer = 1;
  push_kv_req_info->slot_infos[0].slot_offset = 0;

  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(entity->GetCurState(), FsmState::kFsmProbeState);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

HcclResult HcclRawGetCountFail1(const HcclStatus *status, HcclDataType data_type, int *count) {
  if (get_count_first_time) {
    get_count_first_time = false;
    *count = sizeof(ClientClusterInfo);
  } else {
    *count = sizeof(TransferKvMetaInfo);
  }
  return HCCL_SUCCESS;
}
TEST_F(LlmServiceFlowFuncUTest, test_server_sync_kv_probe_msg_invalid) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(true);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  std::vector<uint64_t> req_ids{};
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, req_ids), FsmStatus::kFsmSuccess);

  MOCKER(HcclRawAccept).stubs().will(invoke(HcclRawAcceptSuc));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSuc));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountFail1));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSuc));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  probe_first_time = 0;
  get_count_first_time = true;
  accept_first_time = true;
  sync_kv_req_info->cache_id = -1;
  sync_kv_req_info->req_id = UINT64_MAX;
  sync_kv_req_info->buffer_count_per_layer = 1;
  sync_kv_req_info->tensor_index_count = 0;
  sync_kv_req_info->transfer_infos[0].buffer_info.block_index = 0;
  sync_kv_req_info->transfer_infos[0].buffer_info.buffer_len =
      kFirstDimForKvShape * kSecondDimForKvShape * sizeof(float);
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 1);
  auto it = LlmCommEntityMgr::GetInstance().server_entity_map_.begin();
  EXPECT_EQ(it->second->GetCurState(), FsmState::kFsmProbeState);
  // unlink.
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSucOfUnLink));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountOfUnlink));
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 0);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, update_link_over_limit) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  uint32_t cluster_num = 1U;
  uint32_t ip_num = 1U;
  uint64_t size = sizeof(UpdateLinkReqInfo) + cluster_num * sizeof(ClusterInfo) + ip_num * sizeof(IpInfo);
  std::shared_ptr<FlowMsg> flow_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(size)}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> msgs;
  msgs.emplace_back(flow_msg);

  auto link_req_ptr = static_cast<UpdateLinkReqInfo *>(flow_msg->GetTensor()->GetData());
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kLink);
  link_req_ptr->cluster_num = cluster_num;
  link_req_ptr->timeout = 10000;
  link_req_ptr->force_flag = 0U;
  auto cluster_addr = PtrToValue(flow_msg->GetTensor()->GetData()) + sizeof(UpdateLinkReqInfo);
  auto cluster_info_ptr = static_cast<ClusterInfo *>(ValueToPtr(cluster_addr));
  cluster_info_ptr->ip_num = ip_num;
  cluster_info_ptr->remote_cluster_id = 1;
  auto ip1_addr = cluster_addr + sizeof(ClusterInfo);
  auto ip1_ptr = static_cast<IpInfo *>(ValueToPtr(ip1_addr));
  ip1_ptr->local_ip = 1;
  ip1_ptr->remote_ip = 2;

  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSucOfLink));
  for (int i = 0; i <= 512; ++i) {
    cluster_info_ptr->remote_cluster_id = i + 1;
    ret = llm_service_flow_func.UpdateLink(run_context, msgs);
    EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  }
  MOCKER(HcclRawTestSome).reset();
}

TEST_F(LlmServiceFlowFuncUTest, update_link_req_invalid) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  uint32_t cluster_num = 17U;
  uint64_t size = sizeof(UpdateLinkReqInfo) + cluster_num * sizeof(ClusterInfo);
  std::shared_ptr<FlowMsg> flow_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(size)}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> msgs;
  msgs.emplace_back(flow_msg);

  auto link_req_ptr = static_cast<UpdateLinkReqInfo *>(flow_msg->GetTensor()->GetData());
  link_req_ptr->operator_type = static_cast<uint32_t>(LinkReqType::kLink);
  link_req_ptr->cluster_num = cluster_num;
  link_req_ptr->timeout = 10000;
  link_req_ptr->force_flag = 0U;

  ret = llm_service_flow_func.UpdateLink(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_FAILED);
  MOCKER(HcclRawTestSome).reset();
}

TEST_F(LlmServiceFlowFuncUTest, test_pull_with_tensor_indices) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);

  int64_t src_cache_id = 1;
  EXPECT_EQ(AllocateBlockCaches(run_context, out_msg, llm_service_flow_func, src_cache_id, {UINT64_MAX}),
            FsmStatus::kFsmSuccess);

  std::shared_ptr<FlowMsg> pull_flow_msg = run_context->AllocTensorMsg(
      {static_cast<int64_t>(sizeof(PullKvReqInfo) + 3 * sizeof(uint64_t))}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> pull_msgs;
  pull_msgs.emplace_back(pull_flow_msg);
  auto &req_info = *static_cast<PullKvReqInfo *>(pull_flow_msg->GetTensor()->GetData());
  req_info = PullKvReqInfo{};
  req_info.prompt_cache_id = src_cache_id;
  req_info.req_id = UINT64_MAX;
  req_info.model_id = 0UL;
  req_info.prompt_cluster_id = 1001UL;
  req_info.timeout = 3000000UL;
  req_info.cache_id = src_cache_id;
  req_info.block_count = 1;
  req_info.dst_tensor_index_count = 1;
  req_info.src_tensor_index_count = 1;
  req_info.block_indices[0] = 0;
  req_info.block_indices[1] = 666;
  req_info.block_indices[2] = 0;
  EXPECT_EQ(llm_service_flow_func.PullCache(run_context, pull_msgs), FLOW_FUNC_SUCCESS);
  auto output_data1 = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(static_cast<FsmStatus>(*output_data1), FsmStatus::kFsmParamInvalid);

  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, src_cache_id), FsmStatus::kFsmSuccess);
  RemoveCacheIndexReqInfo req_info1{};
  req_info1.model_id = 0;
  req_info1.req_id = UINT64_MAX;
  EXPECT_EQ(RemoveCacheIndex(run_context, out_msg, llm_service_flow_func, req_info1), FsmStatus::kFsmSuccess);
}

HcclResult HcclRawGetCountSucWithTensorIndices(const HcclStatus *status, HcclDataType data_type, int *count) {
  if (get_count_first_time) {
    get_count_first_time = false;
    *count = sizeof(ClientClusterInfo);
  } else {
    *count = sizeof(SyncKvReqInfo) + 3 * sizeof(SyncBufferInfo);
  }
  return HCCL_SUCCESS;
}
TEST_F(LlmServiceFlowFuncUTest, test_server_sync_kv_with_tensor_indices_success) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(true);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  std::vector<uint64_t> req_ids{1};
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, req_ids), FsmStatus::kFsmSuccess);

  MOCKER(HcclRawAccept).stubs().will(invoke(HcclRawAcceptSuc));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSuc));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountSucWithTensorIndices));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSuc));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  probe_first_time = 0;
  get_count_first_time = true;
  accept_first_time = true;
  sync_kv_req_info->cache_id = -1;
  sync_kv_req_info->req_id = req_ids[0];
  sync_kv_req_info->buffer_count_per_layer = 1;
  sync_kv_req_info->tensor_index_count = 2;
  sync_kv_req_info->transfer_infos[0].buffer_info.block_index = UINT64_MAX;
  sync_kv_req_info->transfer_infos[0].buffer_info.buffer_len =
      kFirstDimForKvShape * kSecondDimForKvShape * sizeof(float);
  sync_kv_req_info->transfer_infos[1].tensor_index = 0;
  sync_kv_req_info->transfer_infos[2].tensor_index = 666;
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 1);
  auto it = LlmCommEntityMgr::GetInstance().server_entity_map_.begin();
  EXPECT_EQ(it->second->GetCurState(), FsmState::kFsmProbeState);
  // unlink.
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSucOfUnLink));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountOfUnlink));
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 0);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);

  sync_kv_req_info->tensor_index_count = 0;
}

TEST_F(LlmServiceFlowFuncUTest, test_pull_with_offset) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);

  int64_t src_cache_id = 1;
  EXPECT_EQ(AllocateBlockCaches(run_context, out_msg, llm_service_flow_func, src_cache_id, {UINT64_MAX}),
            FsmStatus::kFsmSuccess);

  std::shared_ptr<FlowMsg> pull_flow_msg = run_context->AllocTensorMsg(
      {static_cast<int64_t>(sizeof(PullKvReqInfo) + 1 * sizeof(uint64_t))}, TensorDataType::DT_UINT8);
  std::vector<std::shared_ptr<FlowMsg>> pull_msgs;
  pull_msgs.emplace_back(pull_flow_msg);
  auto &req_info = *static_cast<PullKvReqInfo *>(pull_flow_msg->GetTensor()->GetData());
  req_info = PullKvReqInfo{};
  req_info.prompt_cache_id = src_cache_id;
  req_info.req_id = UINT64_MAX;
  req_info.model_id = 0UL;
  req_info.prompt_cluster_id = 1001UL;
  req_info.timeout = 3000000UL;
  req_info.cache_id = src_cache_id;
  req_info.block_count = 1;
  req_info.block_indices[0] = 0;
  req_info.block_len = 10;
  req_info.is_pull_with_offset = 1;
  req_info.src_cache_offset = 10;
  req_info.dst_cache_offset = 10;
  EXPECT_EQ(llm_service_flow_func.PullCache(run_context, pull_msgs), FLOW_FUNC_SUCCESS);
  auto output_data1 = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(static_cast<FsmStatus>(*output_data1), FsmStatus::kFsmParamInvalid);

  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, src_cache_id), FsmStatus::kFsmSuccess);
  RemoveCacheIndexReqInfo req_info1{};
  req_info1.model_id = 0;
  req_info1.req_id = UINT64_MAX;
  EXPECT_EQ(RemoveCacheIndex(run_context, out_msg, llm_service_flow_func, req_info1), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, test_server_sync_kv_with_offset_success) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(true);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  std::vector<uint64_t> req_ids{1};
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, req_ids), FsmStatus::kFsmSuccess);

  MOCKER(HcclRawAccept).stubs().will(invoke(HcclRawAcceptSuc));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSuc));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountSuc));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSuc));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  probe_first_time = 0;
  get_count_first_time = true;
  accept_first_time = true;
  sync_kv_req_info->cache_id = -1;
  sync_kv_req_info->req_id = req_ids[0];
  sync_kv_req_info->buffer_count_per_layer = 1;
  sync_kv_req_info->tensor_index_count = 0;
  sync_kv_req_info->is_pull_with_offset = 1;
  sync_kv_req_info->offset = 256;
  sync_kv_req_info->transfer_infos[0].buffer_info.block_index = UINT64_MAX;
  sync_kv_req_info->transfer_infos[0].buffer_info.buffer_len =
      kFirstDimForKvShape * kSecondDimForKvShape * sizeof(float);
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 1);
  auto it = LlmCommEntityMgr::GetInstance().server_entity_map_.begin();
  EXPECT_EQ(it->second->GetCurState(), FsmState::kFsmProbeState);
  // unlink.
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSucOfUnLink));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountOfUnlink));
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 0);
  RemoveCacheIndexReqInfo req_info1{};
  req_info1.model_id = 0;
  req_info1.req_id = 1;
  EXPECT_EQ(RemoveCacheIndex(run_context, out_msg, llm_service_flow_func, req_info1), FsmStatus::kFsmSuccess);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

HcclResult HcclRawImprobeFail1(HcclConn conn, int32_t *flag, HcclMessage *msg, HcclStatus *status) {
  return HCCL_E_INTERNAL;
}
HcclResult HcclRawGetCountFailCall(const HcclStatus *status, HcclDataType data_type, int *count) {
  return HCCL_E_INTERNAL;
}
HcclResult HcclRawImrecvFailCall1(void *buf, int32_t count, HcclDataType data_type, HcclMessage *msg,
                                  HcclRequest *request) {
  return HCCL_E_INTERNAL;
}
HcclResult HcclRawTestSomeFailCall1(int32_t count, HcclRequest request_array[], int32_t *comp_count,
                                    int32_t comp_indices[], HcclStatus comp_status[]) {
  return HCCL_E_INTERNAL;
}
bool HcclRawTestSomeFail2_first = true;
HcclResult HcclRawTestSomeFailCall2(int32_t count, HcclRequest request_array[], int32_t *comp_count,
                                    int32_t comp_indices[], HcclStatus comp_status[]) {
  if (HcclRawTestSomeFail2_first) {
    *comp_count = 0;
    HcclRawTestSomeFail2_first = false;
  } else {
    *comp_count = 1;
    HcclStatus status;
    status.error = 1;
    comp_status[0] = status;
  }
  return HCCL_SUCCESS;
}
HcclResult HcclRawGetCountSucCall(const HcclStatus *status, HcclDataType data_type, int *count) {
  *count = sizeof(SyncKvReqInfo) + 1 * sizeof(SyncBufferInfo);
  return HCCL_SUCCESS;
}
TEST_F(LlmServiceFlowFuncUTest, test_server_sync_kv_failed) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(true);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;
  std::vector<uint64_t> req_ids{1};
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, req_ids), FsmStatus::kFsmSuccess);
  sync_kv_req_info->cache_id = -1;
  sync_kv_req_info->req_id = req_ids[0];
  sync_kv_req_info->buffer_count_per_layer = 1;
  sync_kv_req_info->tensor_index_count = 0;
  sync_kv_req_info->transfer_infos[0].buffer_info.block_index = 0;
  sync_kv_req_info->transfer_infos[0].buffer_info.buffer_len =
      kFirstDimForKvShape * kSecondDimForKvShape * sizeof(float);

  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  HcclConn conn = reinterpret_cast<HcclConn>(1);

  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeFail1));
  auto entity = LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityServer, conn, local_hccl_addr,
                                                             remote_hccl_addr, 1);
  entity->cur_state_ = FsmState::kFsmProbeState;
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 1);
  auto it = LlmCommEntityMgr::GetInstance().server_entity_map_.begin();
  EXPECT_EQ(it->second->GetCurState(), FsmState::kFsmErrorState);
  LlmCommEntityMgr::GetInstance().ClearEntities();

  entity = LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityServer, conn, local_hccl_addr,
                                                        remote_hccl_addr, 1);
  entity->cur_state_ = FsmState::kFsmProbeState;
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).reset();
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSuc));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountFailCall));
  probe_first_time = 0;
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 1);
  it = LlmCommEntityMgr::GetInstance().server_entity_map_.begin();
  EXPECT_EQ(it->second->GetCurState(), FsmState::kFsmErrorState);
  LlmCommEntityMgr::GetInstance().ClearEntities();

  entity = LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityServer, conn, local_hccl_addr,
                                                        remote_hccl_addr, 1);
  entity->cur_state_ = FsmState::kFsmProbeState;
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).reset();
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSuc));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountSucCall));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvFailCall1));
  probe_first_time = 0;
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 1);
  it = LlmCommEntityMgr::GetInstance().server_entity_map_.begin();
  EXPECT_EQ(it->second->GetCurState(), FsmState::kFsmErrorState);
  LlmCommEntityMgr::GetInstance().ClearEntities();

  entity = LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityServer, conn, local_hccl_addr,
                                                        remote_hccl_addr, 1);
  entity->cur_state_ = FsmState::kFsmProbeState;
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).reset();
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSuc));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountSucCall));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSuc));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeFailCall1));
  probe_first_time = 0;
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 1);
  it = LlmCommEntityMgr::GetInstance().server_entity_map_.begin();
  EXPECT_EQ(it->second->GetCurState(), FsmState::kFsmErrorState);
  LlmCommEntityMgr::GetInstance().ClearEntities();

  entity = LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityServer, conn, local_hccl_addr,
                                                        remote_hccl_addr, 1);
  entity->cur_state_ = FsmState::kFsmProbeState;
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).reset();
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSuc));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountSucCall));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSuc));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeFailCall2));
  HcclRawTestSomeFail2_first = true;
  probe_first_time = 0;
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 1);
  it = LlmCommEntityMgr::GetInstance().server_entity_map_.begin();
  EXPECT_EQ(it->second->GetCurState(), FsmState::kFsmErrorState);
  LlmCommEntityMgr::GetInstance().ClearEntities();

  // unlink.
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImprobe).reset();
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 0);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
  RemoveCacheIndexReqInfo req_info{};
  req_info.model_id = 0;
  req_info.req_id = 1;
  EXPECT_EQ(RemoveCacheIndex(run_context, out_msg, llm_service_flow_func, req_info), FsmStatus::kFsmSuccess);
}
TEST_F(LlmServiceFlowFuncUTest, test_server_hanlde_push_kv_failed) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(true);
  auto ret = llm_service_flow_func.Init(params);
  llm_service_flow_func.role_ = "Decoder";
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;
  std::vector<uint64_t> req_ids{};
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, req_ids), FsmStatus::kFsmSuccess);
  CacheEntry cache_entry;
  const auto cache_entry_exist = CacheManager::GetInstance().GetCacheEntry(cache_id, cache_entry);
  EXPECT_EQ(cache_entry_exist, true);
  push_kv_req_info->key_addr = reinterpret_cast<uintptr_t>(cache_entry.tensors[0]->GetTensor()->GetData());
  push_kv_req_info->value_addr = reinterpret_cast<uintptr_t>(cache_entry.tensors[1]->GetTensor()->GetData());
  push_kv_req_info->max_size = 1024;
  push_kv_req_info->send_nums_per_tensor = 1;
  push_kv_req_info->total_slot_nums = 1;
  push_kv_req_info->slot_infos[0].slot_size = 1024;
  push_kv_req_info->slot_infos[0].slot_nums_per_transfer = 1;
  push_kv_req_info->slot_infos[0].slot_offset = 0;

  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  HcclConn conn = reinterpret_cast<HcclConn>(1);

  auto entity = LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityClient, conn, local_hccl_addr,
                                                             remote_hccl_addr, 1);
  entity->cur_state_ = FsmState::kFsmProbeState;
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).reset();
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSuc));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountStubForRecvTransferKv));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvFailCall1));
  probe_first_time = 0;
  recv_push_get_count_first_time = 0;
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(entity->GetCurState(), FsmState::kFsmErrorState);
  LlmCommEntityMgr::GetInstance().ClearEntities();

  entity = LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityClient, conn, local_hccl_addr,
                                                        remote_hccl_addr, 1);
  entity->cur_state_ = FsmState::kFsmProbeState;
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).reset();
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSuc));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountStubForRecvTransferKv));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSuc));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeFailCall1));
  probe_first_time = 0;
  recv_push_get_count_first_time = 0;
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(entity->GetCurState(), FsmState::kFsmErrorState);
  LlmCommEntityMgr::GetInstance().ClearEntities();

  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImprobe).reset();
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 0);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, test_server_sync_kv_not_exist) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(true);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  std::vector<uint64_t> req_ids{1};
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, req_ids), FsmStatus::kFsmSuccess);

  MOCKER(HcclRawAccept).stubs().will(invoke(HcclRawAcceptSuc));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSuc));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountSuc));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSuc));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  probe_first_time = 0;
  get_count_first_time = true;
  accept_first_time = true;
  sync_kv_req_info->cache_id = -1;
  sync_kv_req_info->req_id = 100;
  sync_kv_req_info->buffer_count_per_layer = 1;
  sync_kv_req_info->tensor_index_count = 0;
  sync_kv_req_info->transfer_infos[0].buffer_info.block_index = 0;
  sync_kv_req_info->transfer_infos[0].buffer_info.buffer_len =
      kFirstDimForKvShape * kSecondDimForKvShape * sizeof(float);
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 1);
  auto it = LlmCommEntityMgr::GetInstance().server_entity_map_.begin();
  // unlink.
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSucOfUnLink));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountOfUnlink));
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 0);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
  RemoveCacheIndexReqInfo req_info{};
  req_info.model_id = 0;
  req_info.req_id = 1;
  EXPECT_EQ(RemoveCacheIndex(run_context, out_msg, llm_service_flow_func, req_info), FsmStatus::kFsmSuccess);
}

HcclResult HcclRawGetCountStubForSyncKvFail(const HcclStatus *status, HcclDataType data_type, int *count) {
  if (get_count_first_flag) {
    *count = static_cast<int32_t>(sizeof(SyncKvMetaInfo)) + 1;
    get_count_first_flag = false;
  } else {
    *count = kFirstDimForKvShape * kSecondDimForKvShape * sizeof(float);
  }
  return HCCL_SUCCESS;
}
TEST_F(LlmServiceFlowFuncUTest, sync_kv_probe_meta_size_invalid) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  size_t total_len = sizeof(PullKvReqInfo);
  std::shared_ptr<FlowMsg> input_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(total_len)}, TensorDataType::DT_INT8);
  input_msgs.emplace_back(input_msg);
  const Tensor *req_info_tensor = input_msg->GetTensor();
  PullKvReqInfo *req_info = static_cast<PullKvReqInfo *>(req_info_tensor->GetData());
  req_info->req_id = 0UL;
  req_info->model_id = 0UL;
  req_info->prompt_cluster_id = 1001UL;
  req_info->timeout = 3000000UL;
  req_info->cache_id = cache_id;
  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  HcclConn conn = reinterpret_cast<HcclConn>(1);
  LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityClient, conn, local_hccl_addr, remote_hccl_addr,
                                               req_info->prompt_cluster_id);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(req_info->prompt_cluster_id);
  EXPECT_NE(entity, nullptr);
  // mock for hccl api
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSucOfLink));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeStub));
  get_count_first_flag = true;
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountStubForSyncKvFail));
  recv_first_flag = true;
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImRecvStubForSyncKv));
  sync_req_info->req_id = req_info->req_id;
  sync_req_info->model_id = req_info->model_id;
  sync_req_info->buffer_count_per_layer = 1;
  sync_req_info->tensor_index_count = 0;
  sync_req_info->transfer_infos[0].buffer_info.block_index = 0;
  sync_req_info->transfer_infos[0].buffer_info.buffer_len = kFirstDimForKvShape * kSecondDimForKvShape * sizeof(float);
  auto sync_ret = llm_service_flow_func.PullCache(run_context, input_msgs);
  EXPECT_EQ(sync_ret, FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmFailed));
  LlmCommEntityMgr::GetInstance().ClearEntities();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImrecv).reset();
  MOCKER(HcclRawTestSome).reset();
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

int32_t accept_times = 0U;
HcclResult HcclRawAcceptSucTwoTimes(HcclConn listen_conn, HcclAddr *accept_addr, HcclConn *accept_conn) {
  if (accept_times < 2) {
    *accept_conn = &fd;
    accept_first_time = false;
  } else {
    return HCCL_E_AGAIN;
  }
  accept_times++;
  return HCCL_SUCCESS;
}
TEST_F(LlmServiceFlowFuncUTest, test_server_accept_force_link) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(true);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  MOCKER(HcclRawAccept).stubs().will(invoke(HcclRawAcceptSucTwoTimes));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSuc));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountSuc));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSuc));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  accept_first_time = true;
  probe_first_time = 0;
  get_count_first_time = true;
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  // unlink.
  LlmCommEntityMgr::GetInstance().ClearEntities();
}

HcclResult HcclRawGetCountSucOfCheckLink(const HcclStatus *status, HcclDataType data_type, int *count) {
  if (get_count_first_time) {
    get_count_first_time = false;
    *count = sizeof(ClientClusterInfo);
  } else {
    *count = sizeof(uint8_t);
  }
  return HCCL_SUCCESS;
}
TEST_F(LlmServiceFlowFuncUTest, test_server_handle_check_link_suc) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(true);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  MOCKER(HcclRawAccept).stubs().will(invoke(HcclRawAcceptSuc));
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSuc));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountSucOfCheckLink));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSuc));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  accept_first_time = true;
  probe_first_time = 0;
  get_count_first_time = true;
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 1);
  auto it = LlmCommEntityMgr::GetInstance().server_entity_map_.begin();
  EXPECT_EQ(it->second->GetCurState(), FsmState::kFsmProbeState);
  // unlink.
  MOCKER(HcclRawGetCount).reset();
  MOCKER(HcclRawImprobe).reset();
  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSucOfUnLink));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountOfUnlink));
  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(LlmCommEntityMgr::GetInstance().server_entity_map_.size(), 0);
}

TEST_F(LlmServiceFlowFuncUTest, check_link_suc) {
  LlmServiceFlowFunc llm_service_flow_func;
  // init
  std::shared_ptr<FlowFuncParams> params = CreateAttrs();
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  // create input
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);

  std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  size_t total_len = sizeof(CheckLinkReqInfo);
  std::shared_ptr<FlowMsg> input_msg =
      run_context->AllocTensorMsg({static_cast<int64_t>(total_len)}, TensorDataType::DT_INT8);
  input_msgs.emplace_back(input_msg);
  const Tensor *req_info_tensor = input_msg->GetTensor();
  CheckLinkReqInfo *req_info = static_cast<CheckLinkReqInfo *>(req_info_tensor->GetData());
  req_info->dst_cluster_id = 1001U;
  req_info->timeout = 3000000UL;
  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  HcclConn conn = reinterpret_cast<HcclConn>(1);
  LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityClient, conn, local_hccl_addr, remote_hccl_addr,
                                               req_info->dst_cluster_id);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(req_info->dst_cluster_id);
  EXPECT_NE(entity, nullptr);

  MOCKER(HcclRawIsend).reset();
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  auto sync_ret = llm_service_flow_func.CheckLink(run_context, input_msgs);
  EXPECT_EQ(sync_ret, FLOW_FUNC_SUCCESS);
  auto output_data = static_cast<int32_t *>(out_msg->GetTensor()->GetData());
  EXPECT_EQ(*output_data, static_cast<int32_t>(FsmStatus::kFsmSuccess));
  LlmCommEntityMgr::GetInstance().ClearEntities();
}

TEST_F(LlmServiceFlowFuncUTest, test_receive_with_cacheid_push_kv_success) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(false);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  auto conn = reinterpret_cast<HcclConn>(1);
  LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityClient, conn, local_hccl_addr, remote_hccl_addr, 0);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(0);
  EXPECT_NE(entity, nullptr);

  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSucForRecvTransferKv));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountStubForRecvTransferKv));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSucForRecvTransferKv));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  recv_push_kv_probe_first_time = 0U;
  recv_push_get_count_first_time = 0U;
  recv_push_imrecv_first_time = 0U;
  CacheEntry cache_entry;
  const auto cache_entry_exist = CacheManager::GetInstance().GetCacheEntry(cache_id, cache_entry);
  EXPECT_EQ(cache_entry_exist, true);
  push_kv_req_info->key_addr = reinterpret_cast<uintptr_t>(cache_entry.tensors[0]->GetTensor()->GetData());
  push_kv_req_info->value_addr = reinterpret_cast<uintptr_t>(cache_entry.tensors[1]->GetTensor()->GetData());
  push_kv_req_info->max_size = 1024;
  push_kv_req_info->send_nums_per_tensor = 1;
  push_kv_req_info->total_slot_nums = 1;
  push_kv_req_info->dst_cache_id = cache_id;
  push_kv_req_info->dst_batch_index = 0;
  push_kv_req_info->dst_layer_index = 0;
  push_kv_req_info->tensor_num_per_layer = 2;
  push_kv_req_info->slot_infos[0].slot_size = 1024;
  push_kv_req_info->slot_infos[0].slot_nums_per_transfer = 1;
  push_kv_req_info->slot_infos[0].slot_offset = 0;

  ret = llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(entity->GetCurState(), FsmState::kFsmProbeState);
  uint32_t *ptr = reinterpret_cast<uint32_t *>(cache_entry.tensors[0]->GetTensor()->GetData());
  UDF_LOG_INFO("get %p value", ptr);
  EXPECT_EQ(*ptr, 2);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

TEST_F(LlmServiceFlowFuncUTest, test_receive_with_cacheid_push_not_exist_kv) {
  LlmServiceFlowFunc llm_service_flow_func;
  std::shared_ptr<FlowFuncParams> params = CreateAttrs(false);
  auto ret = llm_service_flow_func.Init(params);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowMsg> out_msg;
  WriterCallback lambda = [&out_msg](uint32_t, const std::shared_ptr<FlowMsg> &msg) -> int32_t {
    out_msg = msg;
    return 0;
  };
  std::shared_ptr<FlowFuncRunContext> run_context = std::make_shared<FlowFuncRunContext>(0, params, lambda);
  std::vector<std::shared_ptr<FlowMsg>> msgs;

  int64_t cache_id = 1;
  EXPECT_EQ(AllocateCache(run_context, out_msg, llm_service_flow_func, cache_id, {}), FsmStatus::kFsmSuccess);

  // create link
  HcclAddr local_hccl_addr;
  local_hccl_addr.info.tcp.ipv4Addr = 3232235777;  // 192.168.1.1
  local_hccl_addr.info.tcp.port = 8001;
  HcclAddr remote_hccl_addr;
  remote_hccl_addr.info.tcp.ipv4Addr = 3232235778;  // 192.168.1.2
  remote_hccl_addr.info.tcp.port = 8002;
  auto conn = reinterpret_cast<HcclConn>(1);
  LlmCommEntityMgr::GetInstance().CreateEntity(EntityType::kEntityClient, conn, local_hccl_addr, remote_hccl_addr, 0);
  auto entity = LlmCommEntityMgr::GetInstance().GetEntityByRemoteClusterId(0);
  EXPECT_NE(entity, nullptr);

  MOCKER(HcclRawImprobe).stubs().will(invoke(HcclRawImprobeSucForRecvTransferKv));
  MOCKER(HcclRawGetCount).stubs().will(invoke(HcclRawGetCountStubForRecvTransferKv));
  MOCKER(HcclRawImrecv).stubs().will(invoke(HcclRawImrecvSucForRecvTransferKv));
  MOCKER(HcclRawTestSome).stubs().will(invoke(HcclRawTestSomeSuc));
  recv_push_kv_probe_first_time = 0U;
  recv_push_get_count_first_time = 0U;
  recv_push_imrecv_first_time = 0U;
  CacheEntry cache_entry;
  const auto cache_entry_exist = CacheManager::GetInstance().GetCacheEntry(cache_id, cache_entry);
  EXPECT_EQ(cache_entry_exist, true);
  push_kv_req_info->key_addr = reinterpret_cast<uintptr_t>(cache_entry.tensors[0]->GetTensor()->GetData());
  push_kv_req_info->value_addr = reinterpret_cast<uintptr_t>(cache_entry.tensors[1]->GetTensor()->GetData());
  push_kv_req_info->max_size = 1024;
  push_kv_req_info->send_nums_per_tensor = 1;
  push_kv_req_info->total_slot_nums = 1;
  push_kv_req_info->dst_cache_id = cache_id + 1;
  push_kv_req_info->dst_batch_index = 0;
  push_kv_req_info->dst_layer_index = 0;
  push_kv_req_info->tensor_num_per_layer = 2;
  push_kv_req_info->slot_infos[0].slot_size = 1024;
  push_kv_req_info->slot_infos[0].slot_nums_per_transfer = 1;
  push_kv_req_info->slot_infos[0].slot_offset = 0;

  llm_service_flow_func.SyncKvData(run_context, msgs);
  EXPECT_EQ(entity->GetCurState(), FsmState::kFsmErrorState);
  EXPECT_EQ(DeallocateCache(run_context, out_msg, llm_service_flow_func, cache_id), FsmStatus::kFsmSuccess);
}

}  // namespace FlowFunc
