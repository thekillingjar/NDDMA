/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HF55B1FFE_C64C_4671_8A25_A57DDD5D1280
#define HF55B1FFE_C64C_4671_8A25_A57DDD5D1280

#include "easy_graph/graph/node_id.h"
#include "ge_graph_dsl/ge.h"
#include "ge_graph_dsl/op_desc/op_box.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg.h"
#include "graph/ge_attr_value.h"
#include "graph/op_desc.h"
#include "graph/debug/ge_attr_define.h"

GE_NS_BEGIN

struct OpDescCfgBox : OpBox, private OpDescCfg {
  OpDescCfgBox(const OpType &opType);
  OpDescCfgBox &InCnt(int in_cnt);
  OpDescCfgBox &OutCnt(int out_cnt);
  OpDescCfgBox &InNames(std::initializer_list<std::string> names);
  OpDescCfgBox &OutNames(std::initializer_list<std::string> names);
  OpDescCfgBox &ParentNodeIndex(int node_index);
  OpDescCfgBox &TensorDesc(Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                           std::vector<int64_t> shape = {1, 1, 224, 224});
  OpDescCfgBox &Weight(GeTensorPtr &);
  OpDescCfgBox &StreamId(int stream_id);

  template <typename Type>
  OpDescCfgBox &Attr(const std::string &name, Type &&value) {
    auto attrvalue = ge::GeAttrValue::CreateFrom<Type>(std::forward<Type>(value));
    attrs_.emplace(std::make_pair(name, attrvalue));
    return *this;
  }

  template <typename Type>
  OpDescCfgBox &Attr(const std::string &name, Type &value) {
    auto attrvalue = ge::GeAttrValue::CreateFrom<Type>(value);
    attrs_.emplace(std::make_pair(name, attrvalue));
    return *this;
  }

  template <typename Type>
  OpDescCfgBox &InputAttr(const size_t index, const std::string &name, Type &&value) {
    auto attrvalue = ge::GeAttrValue::CreateFrom<Type>(value);
    auto kv = std::make_pair(name, attrvalue);
    auto kvs = input_attrs_.find(index);
    if (kvs == input_attrs_.end()) {
      input_attrs_[index] = {kv};
    } else {
      kvs->second.emplace(kv);
    }
    return *this;
  }

  template <typename Type>
  OpDescCfgBox &InputAttr(const size_t index, const std::string &name, Type &value) {
    auto attrvalue = ge::GeAttrValue::CreateFrom<Type>(value);
    auto kv = std::make_pair(name, attrvalue);
    auto kvs = input_attrs_.find(index);
    if (kvs == input_attrs_.end()) {
      input_attrs_[index] = {kv};
    } else {
      kvs->second.emplace(kv);
    }
    return *this;
  }

  template <typename Type>
  OpDescCfgBox &OutputAttr(const size_t index, const std::string &name, Type &&value) {
    auto attrvalue = ge::GeAttrValue::CreateFrom<Type>(value);
    auto kv = std::make_pair(name, attrvalue);
    auto kvs = output_attrs_.find(index);
    if (kvs == output_attrs_.end()) {
      output_attrs_[index] = {kv};
    } else {
      kvs->second.emplace(kv);
    }
    return *this;
  }

  template <typename Type>
  OpDescCfgBox &OutputAttr(const size_t index, const std::string &name, Type &value) {
    auto attrvalue = ge::GeAttrValue::CreateFrom<Type>(value);
    auto kv = std::make_pair(name, attrvalue);
    auto kvs = output_attrs_.find(index);
    if (kvs == output_attrs_.end()) {
      input_attrs_[index] = {kv};
    } else {
      kvs->second.emplace(kv);
    }
    return *this;
  }

  OpDescCfgBox &Attr(const std::string &name, int32_t value);
  OpDescCfgBox &Attr(const std::string &name, uint32_t value);
  OpDescCfgBox &Attr(const std::string &name, const std::vector<int32_t> &value);
  OpDescCfgBox &Attr(const std::string &name, const std::vector<uint32_t> &value);
  OpDescCfgBox &Attr(const std::string &name, const std::vector<int64_t> &value);
  OpDescCfgBox &Attr(const std::string &name, const std::vector<std::string> &value);
  OpDescCfgBox &Attr(const std::string &name, const char *value);
  OpDescCfgBox &Attr(const std::string &name, const NamedAttrs& named_attrs);
  OpDescCfgBox &Attr(const std::string &name, const std::vector<NamedAttrs>& named_attrs_vec);

  OpDescCfgBox &InputAttr(const size_t index, const std::string &name, int32_t value);
  OpDescCfgBox &InputAttr(const size_t index, const std::string &name, uint32_t value);
  OpDescCfgBox &InputAttr(const size_t index, const std::string &name, const std::vector<int32_t> &value);
  OpDescCfgBox &InputAttr(const size_t index, const std::string &name, const std::vector<uint32_t> &value);
  OpDescCfgBox &InputAttr(const size_t index, const std::string &name, const std::vector<int64_t> &value);
  OpDescCfgBox &InputAttr(const size_t index, const std::string &name, const char *value);

  OpDescCfgBox &OutputAttr(const size_t index, const std::string &name, int32_t value);
  OpDescCfgBox &OutputAttr(const size_t index, const std::string &name, uint32_t value);
  OpDescCfgBox &OutputAttr(const size_t index, const std::string &name, const std::vector<int32_t> &value);
  OpDescCfgBox &OutputAttr(const size_t index, const std::string &name, const std::vector<uint32_t> &value);
  OpDescCfgBox &OutputAttr(const size_t index, const std::string &name, const std::vector<int64_t> &value);
  OpDescCfgBox &OutputAttr(const size_t index, const std::string &name, const char *value);
  OpDescPtr Build(const ::EG_NS::NodeId &id) const override;

 private:
  void UpdateAttrs(OpDescPtr &) const;
  std::map<std::string, GeAttrValue> attrs_;
  std::map<size_t, std::map<std::string, GeAttrValue>> input_attrs_;
  std::map<size_t, std::map<std::string, GeAttrValue>> output_attrs_;
};

#define OP_CFG(optype) ::GE_NS::OpDescCfgBox(optype)
#define OP_DATA(index) ::GE_NS::OpDescCfgBox(DATA).Attr(ATTR_NAME_INDEX, index).InNames({"data"}).OutNames({"out"})

GE_NS_END

#endif /* HF55B1FFE_C64C_4671_8A25_A57DDD5D1280 */
