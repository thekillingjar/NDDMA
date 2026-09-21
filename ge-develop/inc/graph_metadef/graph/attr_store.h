/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTE_GRAPH_ATTR_STORE_H
#define EXECUTE_GRAPH_ATTR_STORE_H
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <memory>
#include <set>
#include "graph/any_value.h"
#include "attribute_group/attr_group_base.h"

namespace ge {
using AttrId = uint64_t;
using AttrSubId = uint32_t;
using AttrNameFilter = std::function<bool(const std::string &attr_name)>;
using AttrGroupsMap = std::unordered_map<TypeId, std::unique_ptr<AttrGroupsBase>>;

enum class AttrType : uint32_t {
  kAttrPredefinedInIr = 0U,  // IR预定义的属性
  kAttrGeneral = 1U,         // 通用属性
  kAttrTypeEnd = 2U
};
constexpr inline uint32_t GetAttrType(const AttrId id) {
  return static_cast<uint32_t>(id >> 32U);
}
constexpr inline uint32_t GetSubAttrId(const AttrId id) {
  return static_cast<uint32_t>(id & 0xffffffffU);
}
constexpr inline AttrId GetAttrId(const uint32_t type, const uint32_t sub_id) {
  return (static_cast<uint64_t>(type) << 32U) | static_cast<uint64_t>(sub_id);
}

struct OtherAttrs {
 public:
  /* the attr whether is in whitelist. */
  bool CheckAttrIsValid(const std::string &attr) const;

  graphStatus SetAttr(const std::string &attr, const AnyValue &value);

  graphStatus GetAttr(const std::string &attr, AnyValue &value) const;

  const std::unordered_map<std::string, AnyValue> &FastGetAllAttr() const;

  std::unordered_map<std::string, AnyValue> GetAllAttr() const;

  /* the attr whether is saved in other attr group. */
  bool CheckAttrIsExist(const std::string &attr) const;

  bool DeleteSingleAttr(const std::string &attr);

  void DeleteAllAttrs();

  void Swap(OtherAttrs &other);

 private:
  /* whitelist for attr */
  static const std::unordered_set<std::string> valid_attrs_;
  std::unordered_map<std::string, AnyValue> keys_to_attrs_;
};

class AttrStore {
 public:
  class CustomDefinedAttrStore {
   public:
    bool Exists(const std::string &name) const noexcept;
    bool Delete(const std::string &name);
    void Clear();
    void Swap(CustomDefinedAttrStore &other);

    AnyValue *GetOrCreateAnyValue(const std::string &name);
    AnyValue *MutableAnyValue(const std::string &name) const noexcept;
    const AnyValue *GetAnyValue(const std::string &name) const noexcept;

    void GetAllNames(std::set<std::string> &names) const;
    void GetAllAttrs(std::map<std::string, AnyValue> &names_to_attr) const;
    void GetAllAttrsWithFilter(std::map<std::string, AnyValue> &names_to_attr, const AttrNameFilter &attr_filter) const;

   private:
    std::unordered_map<std::string, AnyValue> attrs_;
  };
  static AttrStore Create(const size_t pre_defined_attr_count);

  AttrStore() = default;
  ~AttrStore() {
    ClearAttrsGroups();
  }

  AttrStore(const AttrStore &other);
  AttrStore &operator=(const AttrStore &other);

  AttrStore(AttrStore &&other);
  AttrStore &operator=(AttrStore &&other);

  template<typename T>
  bool Set(const AttrId attr_id, T &&value) const;
  template<typename T>
  bool Set(const AttrId attr_id, const T &value) const;
  template<typename T>
  bool SetByName(const std::string &name, T &&value);
  template<typename T>
  bool SetByName(const std::string &name, const T &value);

  template<typename T>
  const T *Get(const AttrId attr_id) const;
  template<typename T>
  T *MutableGet(const AttrId attr_id);
  template<typename T>
  const T *GetByName(const std::string &name) const;
  template<typename T>
  T *MutableGetByName(const std::string &name);

  AttrId GetIdByName(const std::string &name) const noexcept;
  void SetNameAndId(std::string name, const AttrId id);

  bool Exists(const AttrId attr_id) const noexcept;
  bool Exists(const std::string &name) const noexcept;

  bool Delete(const std::string &name);
  void Clear();

  void Swap(AttrStore &other);
  bool SetAnyValueByName(const std::string &name, const AnyValue &value);

  // unordered版本更好，为了兼容老版本接口，仍然用set和map，不论用哪种数据结构，这都是非常低效的接口
  std::set<std::string> GetAllAttrNames() const;
  std::map<std::string, AnyValue> GetAllAttrs() const;
  std::map<std::string, AnyValue> GetAllAttrsWithFilter(const AttrNameFilter &attr_filter) const;

  AnyValue *MutableAnyValue(const std::string &name) const noexcept;
  AnyValue *GetOrCreateAnyValue(const std::string &name);
  const AnyValue *GetAnyValue(const std::string &name) const noexcept;

  graphStatus SetAttrToOtherGroup(const std::string &attr, const AnyValue &value);
  graphStatus GetAttrFromOtherGroup(const std::string &attr, AnyValue &value) const;
  const std::unordered_map<std::string, AnyValue> &FastGetAllAttrsFromOtherGroup() const;
  std::unordered_map<std::string, AnyValue> GetAllAttrsFromOtherGroup() const;
  bool CheckAttrIsExistInOtherGroup(const std::string &attr) const;
  bool DeleteSingleAttrsInOtherGroup(const std::string &attr);
  void DeleteAllAttrsInOtherGroup();

  template<class T>
  typename std::enable_if<std::is_base_of<AttrGroupsBase, T>::value, T *>::type GetOrCreateAttrsGroup();

  template<typename T, typename... Args>
  typename std::enable_if<std::is_base_of<AttrGroupsBase, T>::value, T *>::type CreateAttrsGroup(Args &&... args);

  template<class T>
  typename std::enable_if<std::is_base_of<AttrGroupsBase, T>::value, bool>::type
  CheckAttrGroupIsExist() const;

  // 属性组删除接口，可用于属性组重置场景，先删除再创建，正常删除返回true，属性组不存在返回false
  template<class T>
  typename std::enable_if<std::is_base_of<AttrGroupsBase, T>::value, bool>::type DeleteAttrsGroup();

  template<class T>
  typename std::enable_if<std::is_base_of<AttrGroupsBase, T>::value, T *>::type GetAttrsGroup() const;

  const AttrGroupsMap &GetAttrsGroupPtr() const;
  AttrGroupsMap &MutableAttrsGroupPtr();

  void ClearAllAttrs();
  void ClearAllAttrsInOtherAttrs();
  bool ClearAttrInOtherAttrs(const std::string &attr_name);

  void ClearAttrsGroups() {
    attrs_groups_ptr_.clear();
  }

 private:
  AnyValue *MutableAnyValue(const AttrId attr_id) const noexcept;
  AnyValue *GetOrCreateAnyValue(const AttrId attr_id) const;
  const AnyValue *GetAnyValue(const AttrId attr_id) const noexcept;
  void CopyAttrStoreAllMembers(const AttrStore &other);

  class PreDefinedAttrStore {
  public:
    bool Exists(const AttrSubId index) const noexcept;
    bool Delete(const AttrSubId index);
    void Clear();
    void Swap(PreDefinedAttrStore &other);

    AnyValue *GetOrCreateAnyValue(const AttrSubId index) const;
    AnyValue *MutableAnyValue(const AttrSubId index) const noexcept;
    const AnyValue *GetAnyValue(const AttrSubId index) const noexcept;

    void Resize(const size_t s);

   private:
    std::vector<AnyValue> attrs_;
  };
  std::unordered_map<std::string, AttrId> names_to_id_;
  // 更好的办法是定义一个虚基类、派生出两个子类，然后保存两个子类的指针：`std::array<std::unique_ptr<SubAttrStore>, kAttrTypeEnd>`
  // 然后根据不同的SubAttr类型，调用对应子类的函数。但是这么做会导致创建AttrStore时，总会带有两次子类实例堆申请的开销，
  // 为了减少堆内存申请，直接将子类平铺在成员变量上。
  PreDefinedAttrStore pre_defined_attrs_;
  CustomDefinedAttrStore general_attrs_;
  OtherAttrs other_attrs_;
  AttrGroupsMap attrs_groups_ptr_;
};

template<typename T>
bool AttrStore::Set(const AttrId attr_id, const T &value) const {
  auto *const v = GetOrCreateAnyValue(attr_id);
  if (v == nullptr) {
    return false;
  }
  (void)v->SetValue(value);
  return true;
}
template<typename T>
bool AttrStore::Set(const AttrId attr_id, T &&value) const {
  auto *const v = GetOrCreateAnyValue(attr_id);
  if (v == nullptr) {
    return false;
  }
  (void)v->SetValue(std::forward<T>(value));
  return true;
}
template<typename T>
bool AttrStore::SetByName(const std::string &name, T &&value) {
  auto *const v = GetOrCreateAnyValue(name);
  if (v == nullptr) {
    return false;
  }
  (void)v->SetValue(std::forward<T>(value));
  return true;
}
template<typename T>
bool AttrStore::SetByName(const std::string &name, const T &value) {
  auto *const v = GetOrCreateAnyValue(name);
  if (v == nullptr) {
    return false;
  }
  (void)v->SetValue(value);
  return true;
}

template<typename T>
const T *AttrStore::Get(const AttrId attr_id) const {
  auto *const v = GetAnyValue(attr_id);
  if (v == nullptr) {
    return nullptr;
  }
  return v->Get<T>();
}
template<typename T>
const T *AttrStore::GetByName(const std::string &name) const {
  auto *const v = GetAnyValue(name);
  if (v == nullptr) {
    return nullptr;
  }
  return v->Get<T>();
}

template<typename T>
T *AttrStore::MutableGet(const AttrId attr_id) {
  auto *const v = MutableAnyValue(attr_id);
  if (v == nullptr) {
    return nullptr;
  }
  return v->MutableGet<T>();
}
template<typename T>
T *AttrStore::MutableGetByName(const std::string &name) {
  auto *const v = MutableAnyValue(name);
  if (v == nullptr) {
    return nullptr;
  }
  return v->MutableGet<T>();
}

template<class T>
typename std::enable_if<std::is_base_of<AttrGroupsBase, T>::value, T *>::type AttrStore::GetOrCreateAttrsGroup() {
  auto attr_group = attrs_groups_ptr_.find(GetTypeId<T>());
  if (attr_group == attrs_groups_ptr_.end()) {
    auto t = std::unique_ptr<T>(new(std::nothrow) T());
    auto raw_ptr = t.get();
    attrs_groups_ptr_.emplace(GetTypeId<T>(), std::move(t));
    return reinterpret_cast<T *>(raw_ptr);
  }
  if (attr_group->second == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<T *>(attr_group->second.get());
}

template<typename T, typename... Args>
typename std::enable_if<std::is_base_of<AttrGroupsBase, T>::value, T *>::type AttrStore::CreateAttrsGroup(Args &&... args) {
  auto attr_group = attrs_groups_ptr_.find(GetTypeId<T>());
  if (attr_group != attrs_groups_ptr_.end()) {
    return nullptr;
  }
  using T_nc = typename std::remove_const<T>::type;
  auto t = std::unique_ptr<T>(new (std::nothrow) T_nc(std::forward<Args>(args)...));
  auto raw_ptr = t.get();
  attrs_groups_ptr_.emplace(GetTypeId<T>(), std::move(t));
  return reinterpret_cast<T *>(raw_ptr);
}

template<class T>
typename std::enable_if<std::is_base_of<AttrGroupsBase, T>::value, T *>::type AttrStore::GetAttrsGroup() const {
  auto attr_group = attrs_groups_ptr_.find(GetTypeId<T>());
  if ((attr_group == attrs_groups_ptr_.end()) || (attr_group->second == nullptr)) {
    return nullptr;
  }
  return reinterpret_cast<T *>(attr_group->second.get());
}

template<class T>
typename std::enable_if<std::is_base_of<AttrGroupsBase, T>::value, bool>::type AttrStore::DeleteAttrsGroup() {
  auto attr_group = attrs_groups_ptr_.find(GetTypeId<T>());
  if ((attr_group == attrs_groups_ptr_.end())) {
    return false;
  }
  attrs_groups_ptr_.erase(attr_group);
  return true;
}

template<class T>
typename std::enable_if<std::is_base_of<AttrGroupsBase, T>::value, bool>::type AttrStore::CheckAttrGroupIsExist() const {
  return attrs_groups_ptr_.find(GetTypeId<T>()) != attrs_groups_ptr_.end();
}

}  // namespace ge

#endif  // EXECUTE_GRAPH_ATTR_STORE_H
