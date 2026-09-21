/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MBUF_READER_H
#define MBUF_READER_H

#include <vector>
#include <memory>
#include <functional>
#include "ascend_hal.h"
#include "base_reader.h"
#include "queue_wrapper.h"
#include "data_aligner.h"

namespace FlowFunc {

// if callback need own mbuf, need move or set it null in vector.
using DATA_CALLBACK_FUNC = std::function<void(std::vector<Mbuf *> &)>;

class FLOW_FUNC_VISIBILITY MbufReader : public BaseReader {
public:
    MbufReader(const std::string &name, std::vector<std::shared_ptr<QueueWrapper>> &queue_wrappers,
        DATA_CALLBACK_FUNC data_callback, std::unique_ptr<DataAligner> data_aligner);

    ~MbufReader() override;

    void ReadMessage() override;

    void DiscardAllInputData();

    void Reset();

    bool StatusOk() const;

    bool NeedRetry() const {
        return need_retry_;
    }

    void DumpReaderStatus() const;

    void AddExceptionTransId(uint64_t trans_id);

    void DeleteExceptionTransId(uint64_t trans_id);
private:
    void ReportData(std::vector<Mbuf *> &data) const;
    void ReadAllData(bool &read_finish);

    void QueryQueueSize(std::vector<int32_t> &queue_size_list, uint32_t &not_empty_queue_num) const;

    void ReadWithDataAlign();

    static void ClearData(std::vector<Mbuf *> &mbuf_vec) noexcept;

    std::string name_;
    bool queue_failed_ = false;
    bool queue_empty_ = false;
    bool need_retry_ = false;
    std::vector<std::shared_ptr<QueueWrapper>> queue_wrappers_;
    std::vector<Mbuf *> read_data_;
    DATA_CALLBACK_FUNC data_callback_;
    std::unique_ptr<DataAligner> data_aligner_;
};
}

#endif // MBUF_READER_H
