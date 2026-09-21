/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "kernel/memory/util/link.h"

using namespace gert;

namespace {
struct Foo : LinkNode<Foo> {
  Foo(int a) : x(a) {}

  int value() const {
    return x;
  }

  void value(int v) {
    this->x = v;
  }

 private:
  int x;
};

Link<Foo> CreateFooLink(size_t num) {
  Link<Foo> foos;

  for (size_t i = 0U; i < num; i++) {
    auto f = new Foo(i);
    foos.push_front(*f);
  }
  return foos;
}

void DestroyFooLink(Link<Foo> &&foos) {
  while (auto f = foos.pop_front()) {
    delete f;
  }
}
}  // namespace

struct LinkTest : public testing::Test {
 protected:
  Link<Foo> link;
};

TEST_F(LinkTest, should_be_empty_when_init) {
  ASSERT_TRUE(link.front() == nullptr);
  ASSERT_TRUE(link.back() == nullptr);
  ASSERT_TRUE(link.empty());
  ASSERT_EQ(0, link.size());
  ASSERT_EQ(link.begin(), link.end());
}

TEST_F(LinkTest, should_push_back_and_front) {
  Foo elem1(1);
  Foo elem2(2);
  Foo elem3(3);

  link.push_back(elem1);
  link.push_back(elem2);
  link.push_front(elem3);

  ASSERT_EQ(3, link.size());

  Foo *first = link.pop_front();
  ASSERT_EQ(3, first->value());

  ASSERT_EQ(1, link.front()->value());
  ASSERT_EQ(2, link.back()->value());
}

TEST_F(LinkTest, get_elem_when_link_is_not_empty) {
  Foo elem(1);

  link.push_back(elem);

  ASSERT_FALSE(link.empty());
  ASSERT_EQ(1, link.size());
  ASSERT_EQ(&elem, link.front());
  ASSERT_EQ(&elem, link.back());

  Foo *first = link.pop_front();
  ASSERT_EQ(1, first->value());
  ASSERT_TRUE(link.empty());
}

TEST_F(LinkTest, travel_link_by_order) {
  Foo elem1(1), elem2(2), elem3(3);

  link.push_back(elem1);
  link.push_back(elem2);
  link.push_back(elem3);

  int i = 1;
  for (const auto &node : link) {
    ASSERT_EQ(i++, node.value());
  }
}

TEST_F(LinkTest, travel_link_by_reverse_order) {
  Foo elem1(1), elem2(2), elem3(3);

  link.push_back(elem1);
  link.push_back(elem2);
  link.push_back(elem3);

  int i = link.size();

  for (auto it = link.rbegin(); it != link.rend(); ++it) {
    ASSERT_EQ(i--, it->value());
  }
}

TEST_F(LinkTest, travel_link_by_modifing) {
  Foo elem1(1), elem2(2), elem3(3);

  link.push_back(elem1);
  link.push_back(elem2);
  link.push_back(elem3);

  for (auto &node : link) {
    node.value(node.value() * 2);
  }

  int i = link.size() * 2;
  int sum = 0;

  for (auto it = link.rbegin(); it != link.rend(); ++it) {
    ASSERT_EQ(i, it->value());
    sum += it->value();
    i -= 2;
  }

  ASSERT_EQ(12, sum);
}

TEST_F(LinkTest, should_point_to_the_correct_addr_when_get_next) {
  Foo elem(1);
  link.push_back(elem);

  ASSERT_EQ(&elem, link.begin().value());
  ASSERT_NE(&elem, link.end().value());
  gert::Link<Foo>::Iterator p = link.begin();
  ASSERT_EQ(link.end(), link.next_of(p));
}

TEST_F(LinkTest, should_move_link) {
  Link<Foo> link = CreateFooLink(5);
  ASSERT_EQ(5, link.size());

  auto f = link.pop_front();
  ASSERT_EQ(4, f->value());
  ASSERT_EQ(4, link.size());
  delete f;

  Link<Foo> link2 = std::move(link);
  ASSERT_EQ(0, link.size());
  ASSERT_EQ(4, link2.size());

  f = link.pop_front();
  ASSERT_EQ(nullptr, f);
  ASSERT_EQ(0, link.size());

  f = link2.pop_front();
  ASSERT_EQ(3, f->value());
  ASSERT_EQ(3, link2.size());
  delete f;

  link = std::move(link2);
  ASSERT_EQ(3, link.size());
  ASSERT_EQ(0, link2.size());
  f = link2.pop_front();
  ASSERT_EQ(nullptr, f);
  ASSERT_EQ(0, link2.size());

  DestroyFooLink(std::move(link));
  ASSERT_EQ(0, link.size());

  f = link.pop_front();
  ASSERT_EQ(nullptr, f);
  ASSERT_EQ(0, link.size());
}
