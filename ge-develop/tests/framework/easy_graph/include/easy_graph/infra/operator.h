/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H05B2224D_B927_4FC0_A936_97B52B8A99DB
#define H05B2224D_B927_4FC0_A936_97B52B8A99DB

//////////////////////////////////////////////////////////////
#define __DECL_EQUALS(cls)                                                                                             \
  bool operator!=(const cls &rhs) const;                                                                               \
  bool operator==(const cls &rhs) const

//////////////////////////////////////////////////////////////
#define __FIELD_EQ(name) this->name == rhs.name
#define __FIELD_LT(name) this->name < rhs.name

//////////////////////////////////////////////////////////////
#define __SUPER_EQ(super) static_cast<const super &>(*this) == rhs
#define __SUPER_LT(super) static_cast<const super &>(*this) < rhs

//////////////////////////////////////////////////////////////
#define __DEF_EQUALS(cls)                                                                                              \
  bool cls::operator!=(const cls &rhs) const {                                                                         \
    return !(*this == rhs);                                                                                            \
  }                                                                                                                    \
  bool cls::operator==(const cls &rhs) const

/////////////////////////////////////////////////////////////
#define __INLINE_EQUALS(cls)                                                                                           \
  bool operator!=(const cls &rhs) const {                                                                              \
    return !(*this == rhs);                                                                                            \
  }                                                                                                                    \
  bool operator==(const cls &rhs) const

/////////////////////////////////////////////////////////////
#define __DECL_COMP(cls)                                                                                               \
  __DECL_EQUALS(cls);                                                                                                  \
  bool operator<(const cls &) const;                                                                                   \
  bool operator>(const cls &) const;                                                                                   \
  bool operator<=(const cls &) const;                                                                                  \
  bool operator>=(const cls &) const

/////////////////////////////////////////////////////////////
#define __DEF_COMP(cls)                                                                                                \
  bool cls::operator>(const cls &rhs) const {                                                                          \
    return !(*this <= rhs);                                                                                            \
  }                                                                                                                    \
  bool cls::operator>=(const cls &rhs) const {                                                                         \
    return !(*this < rhs);                                                                                             \
  }                                                                                                                    \
  bool cls::operator<=(const cls &rhs) const {                                                                         \
    return (*this < rhs) || (*this == rhs);                                                                            \
  }                                                                                                                    \
  bool cls::operator<(const cls &rhs) const

#endif
