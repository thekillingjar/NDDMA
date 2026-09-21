/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_flow_attr_define.h"

namespace ge {
namespace dflow {
// Public attribute
const char *const ATTR_NAME_DATA_FLOW_PROCESS_POINTS = "_dflow_process_points";
const char *const ATTR_NAME_IS_DATA_FLOW_GRAPH = "_dflow_is_data_flow_graph";
const char *const ATTR_NAME_DATA_FLOW_INPUT = "input";
const char *const ATTR_NAME_DATA_FLOW_OUTPUT = "output";

const char *const ATTR_NAME_DATA_FLOW_CONTAINS_N_MAPPING_NODE = "_contains_n-mapping_node";
const char *const ATTR_NAME_DATA_FLOW_ENABLE_EXCEPTION_CATCH = "_enable_exception_catch";
// For data align
const char *const ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_TIMEOUT = "_inputs_align_timeout";
const char *const ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_MAX_CACHE_NUM = "_inputs_align_max_cache_num";
// whether dropout data when no align
const char *const ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_DROPOUT = "_inputs_align_dropout";

// For count batch
const char *const ATTR_NAME_COUNT_BATCH_BATCH_SIZE = "_count_batch_batch_size";
const char *const ATTR_NAME_COUNT_BATCH_SLIDE_STRIDE = "_count_batch_slide_stride";
const char *const ATTR_NAME_COUNT_BATCH_TIMEOUT = "_count_batch_timeout";
const char *const ATTR_NAME_COUNT_BATCH_BATCH_DIM = "_count_batch_dim";
const char *const ATTR_NAME_COUNT_BATCH_FLAG = "_count_batch_flag";
const char *const ATTR_NAME_COUNT_BATCH_PADDING = "_count_batch_padding";
const char *const ATTR_NAME_COUNT_BATCH_DROP_REMAINDER = "_count_batch_drop_remainder";

// For time batch
const char *const ATTR_NAME_TIME_BATCH_TIME_WINDOW = "_time_batch_window";
const char *const ATTR_NAME_TIME_BATCH_TIME_INTERVAL = "_time_batch_time_interval";
const char *const ATTR_NAME_TIME_BATCH_TIMEOUT = "_time_batch_timeout";
const char *const ATTR_NAME_TIME_BATCH_BATCH_DIM = "_time_batch_dim";
const char *const ATTR_NAME_TIME_BATCH_FLAG = "_time_batch_flag";
const char *const ATTR_NAME_TIME_BATCH_PADDING = "_time_batch_padding";
const char *const ATTR_NAME_TIME_BATCH_DROP_REMAINDER = "_time_batch_drop_remainder";

// FlowFunc
const char *const ATTR_NAME_FLOW_FUNC_BIN_PATH = "_flow_func_bin_path";
const char *const ATTR_NAME_FLOW_FUNC_FUNC_LIST = "_flow_func_func_list";
const char *const ATTR_NAME_FLOW_FUNC_FUNC_INPUTS_INDEX = "_flow_func_func_inputs_index";
const char *const ATTR_NAME_FLOW_FUNC_FUNC_OUTPUTS_INDEX = "_flow_func_func_outputs_index";
const char *const ATTR_NAME_FLOW_FUNC_FUNC_STREAM_INPUT = "_flow_func_func_stream_input";
const char *const ATTR_NAME_FLOW_FUNC_INVOKE_KEYS = "_flow_func_invoke_keys";

// for balance
const char *const ATTR_NAME_BALANCE_SCATTER = "_balance_scatter";
const char *const ATTR_NAME_BALANCE_GATHER = "_balance_gather";
const char *const ATTR_NAME_DYNAMIC_BALANCED_DISTRIBUTION_HCOM = "_dynamic_balanced_distribution_hcom";
const char *const ATTR_NAME_DYNAMIC_BALANCED_DISTRIBUTION_HCOM_GROUP = "_dynamic_balanced_distribution_hcom_group";
const char *const ATTR_NAME_DYNAMIC_BALANCED_DISTRIBUTION_HCOM_TAG = "_dynamic_balanced_distribution_hcom_tag";
const char *const ATTR_NAME_DYNAMIC_BALANCED_DISTRIBUTION_HCOM_IS_RECV = "_dynamic_balanced_distribution_hcom_is_recv";
}  // namespace dflow
}  // namespace ge
