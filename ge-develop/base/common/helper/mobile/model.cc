/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model.h"
#include <iostream>
#include <functional>

#include "proto/ge_ir_mobile.pb.h"
#include "common/checker.h"

namespace {

void ConvertToMobileShapeDef(
    const ge::proto::ShapeDef& shape, ge::mobile::proto::ShapeDef* mobile_shape)
{
    for (const auto& d: shape.dim()) {
        mobile_shape->add_dim(d);
    }
}

ge::mobile::proto::DataType ConvertToMobileDataType(const ge::proto::DataType data_type)
{
    std::map<ge::proto::DataType, ge::mobile::proto::DataType> m = {
        {ge::proto::DataType::DT_UNDEFINED, ge::mobile::proto::DataType::DT_UNDEFINED},
        {ge::proto::DataType::DT_FLOAT, ge::mobile::proto::DataType::DT_FLOAT},
        {ge::proto::DataType::DT_FLOAT16, ge::mobile::proto::DataType::DT_FLOAT16},
        {ge::proto::DataType::DT_INT8, ge::mobile::proto::DataType::DT_INT8},
        {ge::proto::DataType::DT_UINT8, ge::mobile::proto::DataType::DT_UINT8},
        {ge::proto::DataType::DT_INT16, ge::mobile::proto::DataType::DT_INT16},
        {ge::proto::DataType::DT_UINT16, ge::mobile::proto::DataType::DT_UINT16},
        {ge::proto::DataType::DT_INT32, ge::mobile::proto::DataType::DT_INT32},
        {ge::proto::DataType::DT_INT64, ge::mobile::proto::DataType::DT_INT64},
        {ge::proto::DataType::DT_UINT32, ge::mobile::proto::DataType::DT_UINT32},
        {ge::proto::DataType::DT_UINT64, ge::mobile::proto::DataType::DT_UINT64},
        {ge::proto::DataType::DT_BOOL, ge::mobile::proto::DataType::DT_BOOL},
        {ge::proto::DataType::DT_DOUBLE, ge::mobile::proto::DataType::DT_DOUBLE},
    };
    auto it = m.find(data_type);
    if (it == m.end()) {
        GELOGE(ge::FAILED, "[Mobile] data_type %zu is not support", data_type);
        return ge::mobile::proto::DataType::DT_UNDEFINED;
    }
    return it->second;
}

ge::mobile::proto::AttrDef_ListValue::ListValueType ConvertToMobileListValueType(
    const ge::proto::AttrDef_ListValue::ListValueType list_value_type)
{
    using AttrDefListValue = ge::proto::AttrDef_ListValue;
    using AttrDefListValueType = AttrDefListValue::ListValueType;

    using MobileAttrDefListValue = ge::mobile::proto::AttrDef_ListValue;
    using MobileAttrDefListValueType = MobileAttrDefListValue::ListValueType;

    std::map<AttrDefListValueType, MobileAttrDefListValueType> m = {
        {AttrDefListValue::VT_LIST_NONE, MobileAttrDefListValue::VT_LIST_NONE},
        {AttrDefListValue::VT_LIST_STRING, MobileAttrDefListValue::VT_LIST_STRING},
        {AttrDefListValue::VT_LIST_INT, MobileAttrDefListValue::VT_LIST_INT},
        {AttrDefListValue::VT_LIST_FLOAT, MobileAttrDefListValue::VT_LIST_FLOAT},
        {AttrDefListValue::VT_LIST_BOOL, MobileAttrDefListValue::VT_LIST_BOOL},
        {AttrDefListValue::VT_LIST_BYTES, MobileAttrDefListValue::VT_LIST_BYTES},
        {AttrDefListValue::VT_LIST_TENSOR_DESC, MobileAttrDefListValue::VT_LIST_TENSOR_DESC},
        {AttrDefListValue::VT_LIST_TENSOR, MobileAttrDefListValue::VT_LIST_TENSOR},
        {AttrDefListValue::VT_LIST_GRAPH, MobileAttrDefListValue::VT_LIST_GRAPH},
        {AttrDefListValue::VT_LIST_NAMED_ATTRS, MobileAttrDefListValue::VT_LIST_NAMED_ATTRS},
    };
    auto it = m.find(list_value_type);
    if (it == m.end()) {
        GELOGE(ge::FAILED, "[Mobile] list_value_type %zu is not support", list_value_type);
        return MobileAttrDefListValue::VT_LIST_NONE;
    }
    return it->second;
}

ge::Status ConvertToMobileAttrDef(
    const ge::proto::AttrDef& attr_def, ge::mobile::proto::AttrDef& mobile_attr_def);

ge::Status ConvertToMobileTensorDescriptor(
    const ge::proto::TensorDescriptor& td, ge::mobile::proto::TensorDescriptor* mobile_td)
{
    mobile_td->set_name(td.name());
    mobile_td->set_dtype(ConvertToMobileDataType(td.dtype()));
    if (td.shape().dim().size() > 0) {
        GE_ASSERT_TRUE(mobile_td->dtype() != ge::mobile::proto::DataType::DT_UNDEFINED,
            "[Mobile] dtype is not support.");
    } else {
        GELOGD("[Mobile] desc shape is null, should not check dtype.");
    }
    ConvertToMobileShapeDef(td.shape(), mobile_td->mutable_shape());
    mobile_td->set_layout(td.layout());

    mobile_td->set_has_out_attr(td.has_out_attr());
    mobile_td->set_size(td.size());
    mobile_td->set_weight_size(td.weight_size());
    mobile_td->set_reuse_input(td.reuse_input());
    mobile_td->set_output_tensor(td.output_tensor());
    mobile_td->set_device_type(td.device_type());
    mobile_td->set_input_tensor(td.input_tensor());
    mobile_td->set_real_dim_cnt(td.real_dim_cnt());
    mobile_td->set_reuse_input_index(td.reuse_input_index());
    mobile_td->set_data_offset(td.data_offset());
    mobile_td->set_cmps_size(td.cmps_size());
    mobile_td->set_cmps_tab(td.cmps_tab());
    mobile_td->set_cmps_tab_offset(td.cmps_tab_offset());

    for (const auto& attr: td.attr()) {
        ge::mobile::proto::AttrDef mobile_attr_def;
        GE_ASSERT_TRUE(ConvertToMobileAttrDef(attr.second, mobile_attr_def) == ge::SUCCESS,
            "[Mobile] convert to mobile attr def failed.");
        (void)mobile_td->mutable_attr()->insert({attr.first, mobile_attr_def});
    }
    return ge::SUCCESS;
}

ge::Status ConvertToMobileTensorDef(
    const ge::proto::TensorDef& t, ge::mobile::proto::TensorDef* mobile_t)
{
    GE_ASSERT_TRUE(
        ConvertToMobileTensorDescriptor(t.desc(), mobile_t->mutable_desc()) == ge::SUCCESS,
        "[Mobile] convert to mobile tensor desc failed.");
    mobile_t->set_data(t.data());
    return ge::SUCCESS;
}

void ConvertToMobileOpDefHelper(
    const ge::proto::OpDef& op, ge::mobile::proto::OpDef* mobile_op)
{
    // repeated string input_name
    for (const auto& i_name: op.input_name()) {
        mobile_op->add_input_name(i_name);
    }
    // repeated string src_name
    for (const auto& s_name: op.src_name()) {
        mobile_op->add_src_name(s_name);
    }
    // repeated string dst_name
    for (const auto& d_name: op.dst_name()) {
        mobile_op->add_dst_name(d_name);
    }
        // repeated int64 src_index
    for (const auto& s_idx: op.src_index()) {
        mobile_op->add_src_index(s_idx);
    }
    // repeated int64 dst_index
    for (const auto& d_idx: op.dst_index()) {
        mobile_op->add_dst_index(d_idx);
    }
    // repeated int64 input_i
    for (const auto& i_i: op.input_i()) {
        mobile_op->add_input_i(i_i);
    }
    // repeated int64 output_i
    if (op.type() == "NetOutput") {
        for (const auto& i_i: op.input_i()) {
            mobile_op->add_output_i(i_i);
        }
    } else {
        for (const auto& o_i: op.output_i()) {
            mobile_op->add_output_i(o_i);
        }
    }
    // repeated int64 workspace
    for (const auto& w_space: op.workspace()) {
        mobile_op->add_workspace(w_space);
    }
    // repeated int64 workspace_bytes
    for (const auto& w_space_b: op.workspace_bytes()) {
        mobile_op->add_workspace_bytes(w_space_b);
    }
    // repeated bool is_input_const
    for (const auto& i_input_const: op.is_input_const()) {
        mobile_op->add_is_input_const(i_input_const);
    }
}

ge::Status ConvertToMobileOpDef(
    const ge::proto::OpDef& op, ge::mobile::proto::OpDef* mobile_op)
{
    mobile_op->set_name(op.name());
    mobile_op->set_type(op.type());

    // repeated string input
    for (const auto& i: op.input()) {
        mobile_op->add_input(i);
    }

    for (const auto& attr: op.attr()) {
        ge::mobile::proto::AttrDef mobile_attr_def;
        GE_ASSERT_TRUE(ConvertToMobileAttrDef(attr.second, mobile_attr_def) == ge::SUCCESS,
            "[Mobile] convert to mobile attr def failed.");
        (void)mobile_op->mutable_attr()->insert({attr.first, mobile_attr_def});
    }

    mobile_op->set_has_out_attr(op.has_out_attr());
    mobile_op->set_id(op.id());
    mobile_op->set_stream_id(op.stream_id());

    ConvertToMobileOpDefHelper(op, mobile_op);

    // repeated TensorDescriptor input_desc
    for (const auto& i_desc: op.input_desc()) {
        GE_ASSERT_TRUE(
            ConvertToMobileTensorDescriptor(i_desc, mobile_op->add_input_desc()) == ge::SUCCESS,
            "[Mobile] convert to mobile tensor desc failed.");
    }

    // output desc to output desc
    for (const auto& o_desc: op.output_desc()) {
        GE_ASSERT_TRUE(
            ConvertToMobileTensorDescriptor(o_desc, mobile_op->add_output_desc()) == ge::SUCCESS,
            "[Mobile] convert to mobile tensor desc failed.");
    }
    return ge::SUCCESS;
}

ge::Status ConvertToMobileGraphDef(
    const ge::proto::GraphDef& g, ge::mobile::proto::GraphDef* mobile_g)
{
    mobile_g->set_name(g.name());
    // repeated string input
    for (const auto& i: g.input()) {
        mobile_g->add_input(i);
    }
    // repeated string output
    for (const auto& o: g.output()) {
        mobile_g->add_output(o);
    }

    // repeated OpDef op
    for (const auto& op: g.op()) {
        GE_ASSERT_TRUE(ConvertToMobileOpDef(op, mobile_g->add_op()) == ge::SUCCESS,
            "[Mobile] convert to mobile op def failed.");
    }

    for (const auto& attr: g.attr()) {
        ge::mobile::proto::AttrDef mobile_attr_def;
        GE_ASSERT_TRUE(ConvertToMobileAttrDef(attr.second, mobile_attr_def) == ge::SUCCESS,
            "[Mobile] convert to mobile attr def failed.");
        (void)mobile_g->mutable_attr()->insert({attr.first, mobile_attr_def});
    }
    return ge::SUCCESS;
}

ge::Status ConvertToMobileNamedAttrs(
    const ge::proto::NamedAttrs& na, ge::mobile::proto::NamedAttrs* mobile_na)
{
    mobile_na->set_name(na.name());
    for (const auto& attr: na.attr()) {
        ge::mobile::proto::AttrDef mobile_attr_def;
        GE_ASSERT_TRUE(ConvertToMobileAttrDef(attr.second, mobile_attr_def) == ge::SUCCESS,
            "[Mobile] convert to mobile attr def failed.");
        (void)mobile_na->mutable_attr()->insert({attr.first, mobile_attr_def});
    }
    return ge::SUCCESS;
}

ge::Status ConvertToMobileAttrDefList(
    const ge::proto::AttrDef& attr_def, ge::mobile::proto::AttrDef& mobile_attr_def)
{
    const auto& attr_def_list = attr_def.list();
    auto mobile_attr_def_list = mobile_attr_def.mutable_list();
    // repeated bytes s
    for (const auto& attr_def_list_s: attr_def_list.s()) {
        mobile_attr_def_list->add_s(attr_def_list_s);
    }
    // repeated bytes i
    for (const auto& attr_def_list_i: attr_def_list.i()) {
        mobile_attr_def_list->add_i(attr_def_list_i);
    }
    // repeated bytes f
    for (const auto& attr_def_list_f: attr_def_list.f()) {
        mobile_attr_def_list->add_f(attr_def_list_f);
    }
    // repeated bytes b
    for (const auto& attr_def_list_b: attr_def_list.b()) {
        mobile_attr_def_list->add_b(attr_def_list_b);
    }
    // repeated bytes bt
    for (const auto& attr_def_list_bt: attr_def_list.bt()) {
        mobile_attr_def_list->add_bt(attr_def_list_bt);
    }
    // repeated TensorDescriptor td
    for (const auto& attr_def_list_td: attr_def_list.td()) {
        GE_ASSERT_TRUE(ConvertToMobileTensorDescriptor(
            attr_def_list_td, mobile_attr_def_list->add_tf()) == ge::SUCCESS,
            "[Mobile] convert to mobile tensor desc failed.");
    }
    // repeated TensorDef t
    for (const auto& attr_def_list_t: attr_def_list.t()) {
        GE_ASSERT_TRUE(ConvertToMobileTensorDef(
            attr_def_list_t, mobile_attr_def_list->add_t()) == ge::SUCCESS,
            "[Mobile] convert to mobile tensor def failed.");
    }
    // repeated GraphDef g
    for (const auto& attr_def_list_g: attr_def_list.g()) {
        GE_ASSERT_TRUE(ConvertToMobileGraphDef(
            attr_def_list_g, mobile_attr_def_list->add_g()) == ge::SUCCESS,
            "[Mobile] convert to mobile graph def failed.");
    }
    // repeated NamedAttrs na
    for (const auto& attr_def_list_na: attr_def_list.na()) {
        GE_ASSERT_TRUE(ConvertToMobileNamedAttrs(
            attr_def_list_na, mobile_attr_def_list->add_na()) == ge::SUCCESS,
            "[Mobile] convert to mobile named attrs failed.");
    }
    // AttrDefListValue val_type
    mobile_attr_def_list->set_val_type(
        ConvertToMobileListValueType(attr_def_list.val_type()));
    return ge::SUCCESS;
}

bool ConvertToMobileAttrDefBasic(
    const ge::proto::AttrDef& attr_def, ge::mobile::proto::AttrDef& mobile_attr_def)
{
    // bytes s
    if (attr_def.value_case() == ge::proto::AttrDef::ValueCase::kS) {
        mobile_attr_def.set_s(attr_def.s());
        return true;
    }
    // int64 i
    if (attr_def.value_case() == ge::proto::AttrDef::ValueCase::kI) {
        mobile_attr_def.set_i(attr_def.i());
        return true;
    }
    // float f
    if (attr_def.value_case() == ge::proto::AttrDef::ValueCase::kF) {
        mobile_attr_def.set_f(attr_def.f());
        return true;
    }
    // bool b
    if (attr_def.value_case() == ge::proto::AttrDef::ValueCase::kB) {
        mobile_attr_def.set_b(attr_def.b());
        return true;
    }
    // bytes bt
    if (attr_def.value_case() == ge::proto::AttrDef::ValueCase::kBt) {
        mobile_attr_def.set_bt(attr_def.bt());
        return true;
    }
    return false;
}

bool ConvertToMobileAttrDefListList(
    const ge::proto::AttrDef& attr_def, ge::mobile::proto::AttrDef& mobile_attr_def)
{
    // ListListInt list_list_int
    if (attr_def.value_case() == ge::proto::AttrDef::ValueCase::kListListInt) {
        for (const auto& list_list_i: attr_def.list_list_int().list_list_i()) {
            auto* mobile_list_list_i = mobile_attr_def.mutable_list_list_int()->add_list_list_i();
            for (const auto& list_i: list_list_i.list_i()) {
                mobile_list_list_i->add_list_i(list_i);
            }
        }
        return true;
    }
    // ListListFloat list_list_float
    if (attr_def.value_case() == ge::proto::AttrDef::ValueCase::kListListFloat) {
        for (const auto& list_list_f: attr_def.list_list_float().list_list_f()) {
            auto* mobile_list_list_f = mobile_attr_def.mutable_list_list_float()->add_list_list_f();
            for (const auto& list_f: list_list_f.list_f()) {
                mobile_list_list_f->add_list_f(list_f);
            }
        }
        return true;
    }
    return false;
}

ge::Status ConvertToMobileAttrDef(
    const ge::proto::AttrDef& attr_def, ge::mobile::proto::AttrDef& mobile_attr_def)
{
    // basic value
    if (ConvertToMobileAttrDefBasic(attr_def, mobile_attr_def)) {
        return ge::SUCCESS;
    }

    // ListValue list
    if (attr_def.value_case() == ge::proto::AttrDef::ValueCase::kList) {
        GE_ASSERT_TRUE(ConvertToMobileAttrDefList(attr_def, mobile_attr_def) == ge::SUCCESS,
            "[Mobile] convert to mobile attr def list failed.");
        return ge::SUCCESS;
    }

    // NamedAttrs func
    if (attr_def.value_case() == ge::proto::AttrDef::ValueCase::kFunc) {
        GE_ASSERT_TRUE(ConvertToMobileNamedAttrs(attr_def.func(), mobile_attr_def.mutable_func()) == ge::SUCCESS,
            "[Mobile] convert to mobile named attrs failed.");
        return ge::SUCCESS;
    }

    // TensorDescriptor td
    if (attr_def.value_case() == ge::proto::AttrDef::ValueCase::kTd) {
        GE_ASSERT_TRUE(
            ConvertToMobileTensorDescriptor(attr_def.td(), mobile_attr_def.mutable_td()) == ge::SUCCESS,
            "[Mobile] convert to mobile tensor desc failed.");
        return ge::SUCCESS;
    }

    // TensorDef t
    if (attr_def.value_case() == ge::proto::AttrDef::ValueCase::kT) {
        GE_ASSERT_TRUE(
            ConvertToMobileTensorDef(attr_def.t(), mobile_attr_def.mutable_t()) == ge::SUCCESS,
            "[Mobile] convert to mobile tensor def failed.");
        return ge::SUCCESS;
    }

    // GraphDef g
    if (attr_def.value_case() == ge::proto::AttrDef::ValueCase::kG) {
        GE_ASSERT_TRUE(ConvertToMobileGraphDef(attr_def.g(), mobile_attr_def.mutable_g()) == ge::SUCCESS,
            "[Mobile] convert to mobile graph def failed.");
        return ge::SUCCESS;
    }

    // list list value
    if (ConvertToMobileAttrDefListList(attr_def, mobile_attr_def)) {
        return ge::SUCCESS;
    }
    return ge::SUCCESS;
}

} // namespace

namespace ge {

ge::Status MobileModel::ConvertToMobileModelDef(
    const ge::proto::ModelDef& model_def,
    ge::mobile::proto::ModelDef& mobile_model_def)
{
    // name/version/custom_version
    mobile_model_def.set_name(model_def.name());
    mobile_model_def.set_version(model_def.version());
    mobile_model_def.set_custom_version(model_def.custom_version());
    GELOGI("[Mobile] name: %s  version: %d  custom_version: %s",
        mobile_model_def.name().c_str(), mobile_model_def.version(),
        mobile_model_def.custom_version().c_str());

    // repeated GraphDef graph
    for (const auto& graph: model_def.graph()) {
        GE_ASSERT_TRUE(
            ConvertToMobileGraphDef(graph, mobile_model_def.add_graph()) == ge::SUCCESS,
            "[Mobile] convert to mobile graph def failed.");
    }

    // map<string, AttrDef> attr
    GELOGI("[Mobile] attr map: ");
    for (const auto& attr: model_def.attr()) {
        GELOGI("[Mobile]    attr name: %s", attr.first.c_str());
        ge::mobile::proto::AttrDef mobile_attr_def;
        GE_ASSERT_TRUE(ConvertToMobileAttrDef(attr.second, mobile_attr_def) == ge::SUCCESS,
            "[Mobile] convert to mobile attr def failed.");
        (void)mobile_model_def.mutable_attr()->insert({attr.first, mobile_attr_def});
    }
    return ge::SUCCESS;
}

} // namespace ge
