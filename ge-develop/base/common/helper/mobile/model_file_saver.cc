/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_file_saver.h"
#include "common/checker.h"
#include "common/helper/file_saver.h"

namespace ge {

Status ModelFileSaver::SaveWeightBufferToFile(
    const char* output_weight_dir,
    const std::map<std::string, ge::BaseBuffer>& weights_list_external) const
{
    GE_ASSERT_NOTNULL(output_weight_dir, "[Mobile] output_weight_dir is nullptr");

    GELOGI("[Mobile] save weights list as external data.");
    for (auto it = weights_list_external.cbegin(); it != weights_list_external.cend(); it++) {
        std::string weight_path = std::string(output_weight_dir) + std::string("/") + it->first;
        const void* weight_addr = it->second.GetData();
        GE_ASSERT_NOTNULL(weight_addr, "[Mobile] weight_addr is nullptr");
        const size_t weight_size = it->second.GetSize();
        const auto ret = FileSaver::SaveToFile(weight_path, weight_addr, weight_size, false);
        GE_ASSERT_TRUE(ret == SUCCESS, "[Mobile] save subgraph weight buffer to file failed.");
    }
    return SUCCESS;
}

Status ModelFileSaver::SaveCompiledModelToFile(
    std::shared_ptr<CompiledModel> model,
    const char* output_model_file,
    const char* output_weight_dir,
    bool save_weights_as_external_data) const
{
    GELOGI("[Mobile] SaveCompiledModelToFile, output_model_file: %s output_weight_dir: %s",
        std::string(output_model_file).c_str(), std::string(output_weight_dir).c_str());
    GE_ASSERT_NOTNULL(output_model_file, "[Mobile] output_model_file is nullptr");

    ge::BaseBuffer buffer;
    // xxx.weight, weight buffer
    std::map<std::string, ge::BaseBuffer> weights_list_external;
    Status ret = model->SaveToBuffer(buffer, save_weights_as_external_data, &weights_list_external);
    GE_ASSERT_TRUE(ret == SUCCESS, "[Mobile] save to buffer failed.");

    if (save_weights_as_external_data && !(weights_list_external.empty())) {
        ret = SaveWeightBufferToFile(output_weight_dir, weights_list_external);
        GE_ASSERT_TRUE(ret == SUCCESS, "[Mobile] save weight buffer to file failed.");
    }

    uint8_t* data = buffer.GetData();
    GE_ASSERT_NOTNULL(data, "[Mobile] buffer data is nullptr");
    ret = FileSaver::SaveToFile(
        output_model_file,
        reinterpret_cast<void*>(data),
        buffer.GetSize(), false);
    GE_ASSERT_TRUE(ret == SUCCESS, "[Mobile] save model buffer to file failed.");
    GELOGI("[Mobile] save compiled model to file success.");
    return SUCCESS;
}

} // namespace ge
