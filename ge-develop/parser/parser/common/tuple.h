/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_TUPLE_H_
#define GE_COMMON_TUPLE_H_

#include <array>
#include <stdexcept>
#include <initializer_list>
#include <memory>
#include <variant>

namespace ge {
template <typename T, std::size_t InlineN = 4>
class Tuple {
public:
    explicit Tuple(const std::initializer_list<T>& init) {
        InitFromRange(init.begin(), init.size());
    }

    explicit Tuple(const std::vector<T>& init) {
        InitFromRange(init.begin(), init.size());
    }

    T& operator[](std::size_t i) {
        if (i >= ndim_) {
            throw std::out_of_range("Tuple index out of range");
        }
        return data()[i];
    }

    const T& operator[](std::size_t i) const {
        if (i >= ndim_) {
            throw std::out_of_range("Tuple index out of range");
        }
        return data()[i];
    }

    [[nodiscard]] std::size_t ndim() const {
        return ndim_;
    }

private:
    template <typename It>
    void InitFromRange(It first, std::size_t n) {
        ndim_ = n;
        if (n <= InlineN) {
            std::array<T, InlineN> buf{};
            for (std::size_t i = 0; i < n; ++i) {
                buf[i] = first[i];
            }
            storage_ = std::move(buf);
        } else {
            std::unique_ptr<T[]> buf = std::make_unique<T[]>(n);
            for (std::size_t i = 0; i < n; ++i) {
                buf[i] = first[i];
            }
            storage_ = std::move(buf);
        }
    }

    T* data() {
        return std::visit([](auto& s) -> T* { return PointerOf(s); }, storage_);
    }

    const T* data() const {
        return std::visit([](const auto& s) -> const T* { return PointerOf(s); }, storage_);
    }

    static T* PointerOf(std::array<T, InlineN>& arr) {
        return arr.data();
    }

    static const T* PointerOf(const std::array<T, InlineN>& arr) {
        return arr.data();
    }

    static T* PointerOf(std::unique_ptr<T[]>& ptr) {
        return ptr.get();
    }

    static const T* PointerOf(const std::unique_ptr<T[]>& ptr) {
        return ptr.get();
    }

    std::size_t ndim_{0};
    std::variant<std::array<T, InlineN>, std::unique_ptr<T[]>> storage_;
};

using UintTuple = Tuple<uint32_t>;
using IntTuple = Tuple<int64_t>;
using FloatTuple = Tuple<float>;
using BoolTuple = Tuple<bool>;
using StringTuple = Tuple<std::string>;
} // namespace ge

#endif // GE_COMMON_TUPLE_H_
