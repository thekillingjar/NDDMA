/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MBUF_WRITER_H
#define MBUF_WRITER_H

#include <memory>
#include "ascend_hal.h"
#include "base_writer.h"
#include "queue_wrapper.h"

namespace FlowFunc {

class FLOW_FUNC_VISIBILITY MbufWriter : public BaseWriter {
public:
    explicit MbufWriter(const std::shared_ptr<QueueWrapper> &queue_wrapper);

    ~MbufWriter() override = default;

    int32_t WriteData(Mbuf *buf) override;

    bool NeedRetry() const {
        return queue_wrapper_->NeedRetry();
    }
private:
    std::shared_ptr<QueueWrapper> queue_wrapper_;
};

}

#endif // MBUF_WRITER_H
