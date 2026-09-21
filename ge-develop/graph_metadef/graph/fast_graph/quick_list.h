/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// This header file provides a C-style list, which is a data structure that manages ListElement object pointers.
// Note that it does not manage object ownership. when the list is released, the memory occupied by elements
// is not released. To ensure the normal running of the program, use the following method when destructing the list:
// 1. Erase each element form the list where the upper-layer application uses the list.
// 2. After the erase command is executed, the upper-layer application releases the memory occupied by the
//    corresponding element.

#ifndef D_INC_GRAPH_QUICK_LIST_H
#define D_INC_GRAPH_QUICK_LIST_H
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <list>
#include <unordered_map>
#include <vector>
#include "graph/fast_graph/list_element.h"

namespace ge {
template <typename T>
struct QuickIterator {
  using Self = QuickIterator<T>;
  using Element = ListElement<T>;
  using difference_type = ptrdiff_t;
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = T;
  using pointer = T *;
  using reference = T &;

  QuickIterator() noexcept : element_() {}

  explicit QuickIterator(Element *const x) noexcept : element_(x) {}

  pointer operator->() const noexcept {
    return &(operator*());
  }

  // Must downcast from _List_element_base to _List_element to get to value.
  Element *operator*() const noexcept {
    return element_;
  }

  Self &operator++() noexcept {
    element_ = element_->next;
    return *this;
  }

  Self &operator--() noexcept {
    element_ = element_->prev;
    return *this;
  }

  Self operator++(int) noexcept {
    Self tmp = *this;
    element_ = element_->next;
    return tmp;
  }

  Self operator--(int) noexcept {
    Self tmp = *this;
    element_ = element_->prev;
    return tmp;
  }

  friend bool operator!=(const Self &x, const Self &y) noexcept {
    return x.element_ != y.element_;
  }

  friend bool operator==(const Self &x, const Self &y) noexcept {
    return x.element_ == y.element_;
  }

  Element *element_;
};

template <typename T>
struct ConstQuickIterator {
  using Self = ConstQuickIterator<T>;
  using Element = const ListElement<T>;
  using iterator_category = std::bidirectional_iterator_tag;
  using iterator = QuickIterator<T>;
  using difference_type = ptrdiff_t;
  using value_type = T;
  using pointer = const T *;
  using reference = const T &;

  ConstQuickIterator() noexcept : element_() {}

  explicit ConstQuickIterator(Element *const x) noexcept : element_(x) {}

  explicit ConstQuickIterator(const iterator &x) noexcept : element_(x.element_) {}

  Element *operator*() const noexcept {
    return element_;
  }

  pointer operator->() const noexcept {
    return &(operator*());
  }

  Self &operator++() noexcept {
    element_ = element_->next;
    return *this;
  }

  Self operator++(int) noexcept {
    Self tmp = *this;
    element_ = element_->next;
    return tmp;
  }

  Self &operator--() noexcept {
    element_ = element_->prev;
    return *this;
  }

  Self operator--(int) noexcept {
    Self tmp = *this;
    element_ = element_->prev;
    return tmp;
  }

  friend bool operator==(const Self &x, const Self &y) noexcept {
    return x.element_ == y.element_;
  }

  friend bool operator!=(const Self &x, const Self &y) noexcept {
    return x.element_ != y.element_;
  }

  Element *element_;
};

// in traversal, it is quickly to list
template <class T>
class QuickList {
  using Element = ListElement<T>;

 public:
  using iterator = QuickIterator<T>;
  using const_iterator = ConstQuickIterator<T>;

  /**
   * Please note that this function does not release memory except head.
   * it follow the principle: who applies for who release.
   */
  ~QuickList() {
    if (head_ != nullptr) {
      clear();
      delete head_;
      head_ = nullptr;
      tail_ = nullptr;
    }
  }

  QuickList() {
    init();
  }

  QuickList(const QuickList &list) = delete;
  QuickList &operator=(const QuickList &list) = delete;

  QuickList(QuickList &&list) {
    if (this == &list) {
      return;
    }
    clear();
    iterator iter = list.begin();
    while (iter != list.end()) {
      auto element = *iter;
      iter = erase(iter);
      push_back(element);
    }
  }

  QuickList &operator=(QuickList &&list) {
    if (this == &list) {
      return *this;
    }

    clear();
    iterator iter = list.begin();
    while (iter != list.end()) {
      auto element = *iter;
      iter = erase(iter);
      push_back(element);
    }
    return *this;
  }

  void push_back(Element *const element, ListMode mode) {
    tail_->next = element;

    element->prev = tail_;
    element->next = head_;
    element->owner = this;
    element->mode = mode;
    tail_ = element;
    head_->prev = element;

    total_size_++;
  }

  void insert(iterator pos, Element *element, ListMode mode) {
    if (pos == begin()) {
      Element *src_node = head_->next;
      head_->next = element;
      element->next = src_node;
      element->prev = head_;
      src_node->prev = element;

      if (tail_ == head_) {
        tail_ = element;
      }
    } else if (pos == end()) {
      element->next = tail_->next;
      element->prev = tail_;
      tail_->next = element;
      tail_ = element;
      head_->prev = tail_;
    } else {
      Element *cur = pos.element_;
      Element *prev = cur->prev;
      element->prev = prev;
      element->next = cur;

      prev->next = element;
      cur->prev = element;
    }

    element->owner = this;
    element->mode = mode;
    total_size_++;
  }

  int move(Element *const src_pos_value, Element *const dst_pos_value, bool before_flag = true) {
    // action for src
    Element *cur = src_pos_value;
    Element *next = cur->next;
    Element *prev = cur->prev;
    prev->next = next;
    next->prev = prev;
    if (tail_ == cur) {
      tail_ = prev;
    }

    // action for dst
    Element *dst = dst_pos_value;
    if (before_flag) {
      Element *dst_prev = dst->prev;
      dst->prev = src_pos_value;
      dst_prev->next = src_pos_value;
      src_pos_value->prev = dst_prev;
      src_pos_value->next = dst;
    } else {
      Element *dst_next = dst->next;
      dst_next->prev = src_pos_value;
      dst->next = src_pos_value;
      src_pos_value->prev = dst;
      src_pos_value->next = dst_next;
      if (tail_ == dst) {
        tail_ = src_pos_value;
      }
    }

    return 0;
  }

  void push_front(Element *const x, ListMode mode) {
    insert(begin(), x, mode);
  }

  iterator erase(iterator &pos) {
    auto element = pos.element_;
    Element *next = element->next;
    (void)erase(element);
    return iterator(next);
  }

  int erase(Element *const x) {
    if (x->owner != this) {
      return -1;
    }
    if (x->prev == nullptr) {
      return 0;
    }
    if (tail_ == x) {
      tail_ = x->prev;
    }
    Element *prev = x->prev;
    Element *next = x->next;

    x->prev = nullptr;
    x->next = nullptr;
    x->owner = nullptr;
    x->mode = ListMode::kFreeMode;

    prev->next = next;
    next->prev = prev;

    total_size_--;
    return 0;
  }

  size_t size() const {
    return total_size_;
  }

  bool empty() const {
    return head_->next == head_;
  }

  void swap(QuickList<T> &list) {
    Element *tmp_head = this->head_;
    Element *tmp_tail = this->tail_;
    size_t tmp_total_size = this->total_size_;

    for (auto iter = list.begin(); iter != list.end(); ++iter) {
      auto item = *iter;
      item->owner = this;
    }

    for (auto iter = begin(); iter != end(); ++iter) {
      auto item = *iter;
      item->owner = &list;
    }

    this->head_ = list.head_;
    this->tail_ = list.tail_;
    this->head_->owner = this;
    this->total_size_ = list.total_size_;

    list.head_ = tmp_head;
    list.tail_ = tmp_tail;
    list.total_size_ = tmp_total_size;
    list.head_->owner = &list;
  }

  void clear() {
    iterator iter = begin();
    while (iter != end()) {
      iter = erase(iter);
    }
  }

  iterator begin() {
    return iterator(head_->next);
  }

  iterator end() {
    return iterator(head_);
  }

  const_iterator begin() const {
    return const_iterator(head_->next);
  }

  const_iterator end() const {
    return const_iterator(head_);
  }

  void sort(const std::function<bool(ListElement<T> *, ListElement<T> *b)> &comp) {
    if (total_size_ <= 1) {
      return;
    }

    std::list<Element *> carry;
    std::list<Element *> tmp[64];
    std::list<Element *> *fill = tmp;
    std::list<Element *> *counter = nullptr;

    do {
      auto iter = begin();
      auto element = iter.element_;
      erase(iter);
      carry.insert(carry.begin(), element);

      for (counter = tmp; counter != fill && !counter->empty(); ++counter) {
        counter->merge(carry, comp);
        carry.swap(*counter);
      }
      carry.swap(*counter);
      if (counter == fill) {
        ++fill;
      }
    } while (!empty());

    for (counter = tmp + 1; counter != fill; ++counter) {
      counter->merge(*(counter - 1), comp);
    }
    clear();
    std::for_each((*(fill - 1)).begin(), (*(fill - 1)).end(),
                  [this](Element *element) { push_back(element, ListMode::kWorkMode); });
  }

  // for T is pointer
  std::vector<T> CollectAllItemToVector() const {
    std::vector<T> tmp;
    tmp.reserve(size());
    for (auto iter = begin(); iter != end(); ++iter) {
      auto item = *iter;
      tmp.push_back(item->data);
    }

    return tmp;
  }

  // for T is obj
  std::vector<T *> CollectAllPtrItemToVector() {
    std::vector<T *> tmp;
    tmp.reserve(size());
    for (auto iter = begin(); iter != end(); ++iter) {
      auto item = *iter;
      tmp.push_back(&(item->data));
    }

    return tmp;
  }

 private:
  void init() {
    Element *mem = new Element;
    head_ = mem;
    tail_ = mem;
    head_->next = head_;
    head_->prev = head_;
    head_->owner = this;
  }

 private:
  Element *head_ = nullptr;
  Element *tail_ = nullptr;
  size_t total_size_ = 0U;
};
}  // namespace ge

#endif  // D_INC_GRAPH_QUICK_LIST_H
