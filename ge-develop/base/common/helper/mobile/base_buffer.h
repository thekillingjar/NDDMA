/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef BASE_COMMON_HELPER_MOBILE_BASE_BUFFER_H
#define BASE_COMMON_HELPER_MOBILE_BASE_BUFFER_H

#include <cstdlib>
#include <cstdint>

namespace ge {

class BaseBuffer {
public:
    BaseBuffer() : data_(nullptr), size_(0)
    {
    }

    BaseBuffer(void* data, std::size_t size)
        : data_(data), size_(size)
    {
    }

    BaseBuffer(std::uint8_t* data, std::size_t size)
        : data_(data), size_(size)
    {
    }

    ~BaseBuffer() = default;

    BaseBuffer(const BaseBuffer& other)
    {
        if (&other != this) {
            data_ = other.data_;
            size_ = other.size_;
        }
    }

    BaseBuffer& operator=(const BaseBuffer& other)
    {
        if (&other != this) {
            data_ = other.data_;
            size_ = other.size_;
        }
        return *this;
    }

    inline const std::uint8_t* GetData() const
    {
        return static_cast<const uint8_t*>(data_);
    }

    inline std::uint8_t* GetData()
    {
        return static_cast<uint8_t*>(data_);
    }

    inline std::size_t GetSize() const
    {
        return size_;
    }

    inline void SetData(std::uint8_t* data)
    {
        data_ = data;
    }

    inline void SetData(void* data)
    {
        data_ = data;
    }

    inline void SetSize(std::size_t size)
    {
        size_ = size;
    }

private:
    void* data_;
    std::size_t size_;
};

} // namespace ge

#endif // BASE_COMMON_HELPER_MOBILE_BASE_BUFFER_H
