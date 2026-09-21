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
#include <vector>
#include <string.h>

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
  type* get_addr_##field_name()  {                 \
    return &this->field_name##_;                   \
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
class FieldInfo {
public:
  FieldInfo(const char *dtype, const char *name)
      : dtype_(dtype), name_(name), classType_("0") {}
  FieldInfo(const char *dtype, const char *name, size_t arrSize)
      : dtype_(dtype), name_(name), arrSize_(arrSize), classType_("1") {}
  FieldInfo(const char *dtype, const char *name, const char *structType,
            size_t structSize)
      : dtype_(dtype), name_(name), structType_(structType), structSize_(structSize), classType_("2") {}

public:
  const char *dtype_;
  const char *name_;
  size_t arrSize_;
  const char *structType_;
  size_t structSize_;
  const char *classType_;
};

class TilingDef {
  public:
  ~TilingDef()
  {
    if (!inited_data_ptr && data_ptr_ != nullptr) {
      delete[] data_ptr_;
    }
    data_ptr_ = nullptr;
    class_name_ = nullptr;
  }
  void SaveToBuffer(void *pdata, size_t capacity) const {
    if (inited_data_ptr) {
      return;
    }
    // copy tilingdata to buffer without struct tiling data.
    memcpy(pdata, data_ptr_, data_size_);
  }
  std::vector<FieldInfo> GetFieldInfo() const;
  const char *GetTilingClassName() const;
  size_t GetDataSize() const;
  void SetDataPtr(void *dataPtr);
  void CheckAlignAndGenPlaceHolder(const char *name, size_t typeSize);
protected:
  void InitData();
  void GeLogError(const std::string& str) const;
  // dtype, name
  std::vector<FieldInfo> field_info_;
  uint8_t *data_ptr_ = nullptr;
  size_t data_size_ = 0;
  const char *class_name_;
  std::vector<std::pair<void *, size_t>> saveBufferPtr;
  size_t struct_size_ = 0;
  bool inited_data_ptr = false;
  uint32_t feature_bit_flag = 0;
  uint8_t reserved_buf[128] = {0};
};

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