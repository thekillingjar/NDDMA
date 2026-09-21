/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_TILING_DATA_STUB_H_
#define ATT_TILING_DATA_STUB_H_
#include <memory>
#include <map>
#include <cstring>

#define BEGIN_TILING_DATA_DEF(struct_name)         \
class struct_name : public TilingDef {

#define TILING_DATA_FIELD_DEF(type, field_name)    \
 public:                                           \
  void set_##field_name(type field_name) {         \
    this->field_name##_ = field_name;              \
  }                                                \
  type get_##field_name() const {                  \
    return this->field_name##_;                    \
  }                                                \
  type field_name##_ = 0;

#define TILING_DATA_FIELD_DEF_STRUCT(struct_type, field_name)  \
  struct_type field_name;

#define END_TILING_DATA_DEF };

struct CharPtrCmp {
  bool operator()(const char *strLeft, const char *strRight) const
  {
    return strcmp(strLeft, strRight) < 0;
  }
};

class TilingDef {};

using TilingDataConstructor = std::shared_ptr<TilingDef> (*)();

class CTilingDataClassFactory {
public:
  static CTilingDataClassFactory &GetInstance();
  void RegisterTilingData(const char *op_type, const TilingDataConstructor constructor);
  std::shared_ptr<TilingDef> CreateTilingDataInstance(const char *op_type);

private:
  CTilingDataClassFactory() { };
  ~CTilingDataClassFactory() { };
  CTilingDataClassFactory(const CTilingDataClassFactory &);
  CTilingDataClassFactory &operator=(const CTilingDataClassFactory &);
  std::map<const char *, TilingDataConstructor, CharPtrCmp> instance_;
};

#define REGISTER_TILING_DATA_CLASS(op_type, class_name)                                                                \
  class op_type##class_name##Helper {                                                                                  \
   public:                                                                                                             \
    op_type##class_name##Helper() {}                                                                                                                  \
    static std::shared_ptr<TilingDef> CreateTilingDataInstance() { return std::make_shared<class_name>(); }            \
  };                                                                                                                   \
  static class_name g_##op_type##class_name##init;                                                                     \
  static op_type##class_name##Helper g_tilingdata_##op_type##class_name##helper;

#endif             