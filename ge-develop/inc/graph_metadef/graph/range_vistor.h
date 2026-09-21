/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_RANGE_VISTOR_H_
#define INC_GRAPH_RANGE_VISTOR_H_

#include <vector>
#include <list>
#include <memory>

template <class E, class O>
class RangeVistor {
 public:
  using Iterator = typename std::vector<E>::iterator;
  using ConstIterator = typename std::vector<E>::const_iterator;
  using ReverseIterator = typename std::vector<E>::reverse_iterator;
  using ConstReverseIterator = typename std::vector<E>::const_reverse_iterator;

  RangeVistor(const O owner, const std::vector<E> &vs) : owner_(owner), elements_(vs) {}
  RangeVistor(const O owner, const std::list<E> &vs) : owner_(owner), elements_(vs.begin(), vs.end()) {}

  ~RangeVistor() {}

  Iterator begin() { return elements_.begin(); }

  Iterator end() { return elements_.end(); }

  ConstIterator begin() const { return elements_.begin(); }

  ConstIterator end() const { return elements_.end(); }

  ReverseIterator rbegin() { return elements_.rbegin(); }

  ReverseIterator rend() { return elements_.rend(); }

  ConstReverseIterator rbegin() const { return elements_.rbegin(); }

  ConstReverseIterator rend() const { return elements_.rend(); }

  std::size_t size() const { return elements_.size(); }

  bool empty() const { return elements_.empty(); }

  E &at(const std::size_t index) { return elements_.at(index); }

  const E &at(const std::size_t index) const { return elements_.at(index); }

 private:
  O owner_;
  std::vector<E> elements_;
};

#endif  // INC_GRAPH_RANGE_VISTOR_H_
