/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func_helper.h"
#include "common/udf_log.h"

namespace FlowFunc {
namespace {
constexpr uint32_t kBalanceMatrixMaxSize = 65536U;
}

bool FlowFuncHelper::CheckMatrixValid(int32_t row_num, int32_t col_num) {
    if ((row_num <= 0) || (col_num <= 0)) {
        UDF_LOG_ERROR("row_num=%d and col_num=%d can't be <=0.", row_num, col_num);
        return false;
    }
    uint64_t matrix_size = static_cast<uint64_t>(row_num) * static_cast<uint64_t>(col_num);
    if (matrix_size > kBalanceMatrixMaxSize) {
        UDF_LOG_ERROR("weight matrix size(row_num[%d] * col_num[%d]) can't be > %u.",
            row_num, col_num, kBalanceMatrixMaxSize);
        return false;
    }
    return true;
}

bool FlowFuncHelper::IsBalanceConfigValid(const BalanceConfig &config) {
    const auto &balance_weight = config.GetBalanceWeight();
    const auto &data_pos = config.GetDataPos();
    if ((config.GetAffinityPolicy() < AffinityPolicy::NO_AFFINITY) ||
        (config.GetAffinityPolicy() > AffinityPolicy::COL_AFFINITY)) {
        UDF_LOG_ERROR("BalanceConfig policy=%d must be in range [%d, %d].",
            static_cast<int32_t>(config.GetAffinityPolicy()),
            static_cast<int32_t>(AffinityPolicy::NO_AFFINITY),
            static_cast<int32_t>(AffinityPolicy::COL_AFFINITY));
        return false;
    }
    if (!CheckMatrixValid(balance_weight.rowNum, balance_weight.colNum)) {
        UDF_LOG_ERROR("BalanceConfig row_num=%d or col_num=%d check failed.", balance_weight.rowNum,
            balance_weight.colNum);
        return false;
    }

    for (size_t i = 0; i < data_pos.size(); ++i) {
        if ((data_pos[i].first < 0) || (data_pos[i].first >= balance_weight.rowNum)) {
            UDF_LOG_ERROR("BalanceConfig data_pos[%zu] row index=%d must > 0 and < balance_weight.rowNum=%d.",
                i, data_pos[i].first, balance_weight.rowNum);
            return false;
        }
        if ((data_pos[i].second < 0) || (data_pos[i].second >= balance_weight.colNum)) {
            UDF_LOG_ERROR("BalanceConfig data_pos[%zu] col index=%d must > 0 and < balance_weight.colNum=%d.",
                i, data_pos[i].second, balance_weight.colNum);
            return false;
        }
    }
    return true;
}

void FlowFuncHelper::CalcRouteLabelAndDataLabel(const BalanceConfig &config, const std::pair<int32_t, int32_t> &pos,
    uint32_t &data_label, uint32_t &route_label) {
    const auto &balance_weight = config.GetBalanceWeight();
    uint32_t pos_idx = static_cast<uint32_t>(pos.first * balance_weight.colNum + pos.second);
    // data label = 0 means no need align, so need add 1.
    data_label = pos_idx + 1;
    route_label = 0;
    switch (config.GetAffinityPolicy()) {
        case AffinityPolicy::ROW_AFFINITY:
            route_label = pos.first;
            break;
        case AffinityPolicy::COL_AFFINITY:
            route_label = pos.second;
            break;
        case AffinityPolicy::NO_AFFINITY:
        default:
            route_label = pos_idx;
            break;
    }
    // as route_label 0 may ignore by flow gw, so add 1
    ++route_label;
}
}