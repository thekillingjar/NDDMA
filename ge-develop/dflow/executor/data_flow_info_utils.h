/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_MODEL_MANAGER_DATA_FLOW_INFO_UTILS_H_
#define GE_GRAPH_LOAD_MODEL_MANAGER_DATA_FLOW_INFO_UTILS_H_
#include "ge/ge_data_flow_api.h"
#include "dflow/base/deploy/exchange_service.h"

namespace ge {
class DataFlowInfoUtils {
 public:
  static bool HasCustomTransactionId(const DataFlowInfo &info);
  static void InitDataFlowInfoByMsgInfo(DataFlowInfo &info, const ExchangeService::MsgInfo &msg_info);
  static void InitMsgInfoByDataFlowInfo(ExchangeService::MsgInfo &msg_info, const DataFlowInfo &info,
                                        bool contains_n_mapping_node);
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_MODEL_MANAGER_DATA_FLOW_INFO_UTILS_H_
