/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensor_engine/tbe_op_info.h"
#include <cmath>
#include "inc/te_fusion_log.h"
#include "inc/te_fusion_check.h"
#include "inc/te_fusion_util_constants.h"

namespace te {
using namespace fusion;
/**
 * @brief: operator==
 * @param [in] rObject   TbeAttrValue object need to compare
 * @return [out] bool    succ or fail for compare
 */
bool TbeAttrValue::operator==(TbeAttrValue& rObject)
{
    bool isNotEqual = (
        int8_value_        != rObject.int8_value_   ||
        uint8_value_       != rObject.uint8_value_  ||
        int16_value_       != rObject.int16_value_  ||
        uint16_value_      != rObject.uint16_value_ ||
        int32_value_       != rObject.int32_value_  ||
        uint32_value_      != rObject.uint32_value_ ||
        int64_value_       != rObject.int64_value_  ||
        uint64_value_      != rObject.uint64_value_ ||
        fabs(float_value_ - rObject.float_value_) > FLOAT_ESP    ||
        fabs(double_value_ - rObject.double_value_) > DOUBLE_ESP ||
        bool_value_        != rObject.bool_value_   ||
        str_value_         != rObject.str_value_    ||
        list_int8_value_   != rObject.list_int8_value_   ||
        list_uint8_value_  != rObject.list_uint8_value_  ||
        list_int16_value_  != rObject.list_int16_value_  ||
        list_uint16_value_ != rObject.list_uint16_value_ ||
        list_int32_value_  != rObject.list_int32_value_  ||
        list_uint32_value_ != rObject.list_uint32_value_ ||
        list_int64_value_  != rObject.list_int64_value_  ||
        list_uint64_value_ != rObject.list_uint64_value_ ||
        list_bool_value_   != rObject.list_bool_value_   ||
        list_str_value_    != rObject.list_str_value_    ||
        list_list_int64_value_    != rObject.list_list_int64_value_    ||
        list_list_float_value_    != rObject.list_list_float_value_    ||
        dtype_             != rObject.dtype_             ||
        stype_             != rObject.stype_);
    if (isNotEqual) {
        return false;
    }
    if (list_float_value_.size() != rObject.list_float_value_.size() ||
        list_double_value_.size() != rObject.list_double_value_.size()) {
        return false;
    }
    for (size_t idx = 0; idx < list_float_value_.size(); ++idx) {
        if (fabs(list_float_value_[idx] - rObject.list_float_value_[idx]) > FLOAT_ESP) {
            return false;
        }
    }
    for (size_t idx = 0; idx < list_double_value_.size(); ++idx) {
        if (fabs(list_double_value_[idx] - rObject.list_double_value_[idx]) > DOUBLE_ESP) {
            return false;
        }
    }
    return true;
}

/**
 * @brief: operator==
 * @param [in] rObject   TbeOpTensor object need to compare
 * @return [out] bool    succ or fail for compare
 */
bool TbeOpTensor::operator==(TbeOpTensor& rObject)
{
    bool isNotEqual = (
        shape_         != rObject.shape_    ||
        format_        != rObject.format_   ||
        sub_format_    != rObject.sub_format_   ||
        origin_shape_  != rObject.origin_shape_    ||
        origin_format_ != rObject.origin_format_   ||
        addr_type_     != rObject.addr_type_       ||
        valid_shape_   != rObject.valid_shape_     ||
        slice_offset_  != rObject.slice_offset_    ||
        use_L1_workspace_ != rObject.use_L1_workspace_ ||
        L1_fusion_type_ != rObject.L1_fusion_type_ ||
        addr_offset_   != rObject.addr_offset_ ||
        dtype_         != rObject.dtype_    ||
        stype_         != rObject.stype_    ||
        is_const_      != rObject.is_const_ ||
        shapeRange_    != rObject.shapeRange_ ||
        originShapeRange_    != rObject.originShapeRange_ ||
        !(const_value_ == rObject.const_value_));
    if (isNotEqual) {
        return false;
    }
    return true;
}

/**
 * @brief: operator==
 * @param [in] rObject   TbeOpParam object need to compare
 * @return [out] bool    succ or fail for compare
 */
bool TbeOpParam::operator==(TbeOpParam& rObject)
{
    if (type_ != rObject.type_) {
        return false;
    }
    if (tensors_.size() != rObject.tensors_.size()) {
        return false;
    }
    for (size_t idx = 0; idx < tensors_.size(); ++idx) {
        if (!(tensors_[idx] == rObject.tensors_[idx])) {
            return false;
        }
    }
    return true;
}

/**
 * @brief: operator==
 * @param [in] rObject   TbeOpInfo object need to compare
 * @return [out] bool    succ or fail for compare
 */
bool TbeOpInfo::operator==(TbeOpInfo& rObject)
{
    bool isNotEqual = (
        op_name_           != rObject.op_name_           ||
        op_module_name_    != rObject.op_module_name_    ||
        op_file_name_      != rObject.op_file_name_      ||
        op_func_name_      != rObject.op_func_name_      ||
        op_type_           != rObject.op_type_           ||
        op_L1_space_       != rObject.op_L1_space_       ||
        kernel_name_       != rObject.kernel_name_       ||
        pattern_           != rObject.pattern_           ||
        op_check_supported_name_  != rObject.op_check_supported_name_ ||
        rangeLimitType_    != rObject.rangeLimitType_ ||
        ubFusionSpaceSize_ != rObject.ubFusionSpaceSize_);
    if (isNotEqual) {
        return false;
    }
    if (inputs_.size()      != rObject.inputs_.size()  ||
        outputs_.size()     != rObject.outputs_.size() ||
        attr_values_.size() != rObject.attr_values_.size()) {
        return false;
    }
    for (size_t idx = 0; idx < inputs_.size(); ++idx) {
        if (!(inputs_[idx] == rObject.inputs_[idx])) {
            return false;
        }
    }
    for (size_t idx = 0; idx < outputs_.size(); ++idx) {
        if (!(outputs_[idx] == rObject.outputs_[idx])) {
            return false;
        }
    }
    for (size_t idx = 0; idx < attr_values_.size(); ++idx) {
        if (!(attr_values_[idx] == rObject.attr_values_[idx])) {
            return false;
        }
    }
    return true;
}

/**
 * @brief: normalize function name
 * @param [in] funcName           op function name
 * @return [out] bool             succ or fail for normalizing
 */
void TbeOpInfo::NormalizeFuncName(std::string& funcName) const
{
    std::string nameTmp = "";
    bool subHead = false;
    for (string::iterator iter = funcName.begin(); iter != funcName.end(); ++iter) {
        if (islower(*iter)) {
            subHead = false;
        }
        // sample: Abc2Def -> Abc2_Def, Abc2DEf -> Abc2D_Ef
        if (isdigit(*iter)) {
            subHead = true;
        }
        if (isupper(*iter) && iter != funcName.begin()) {
            // sample: AbcDef -> Abc_Def, AbcDEF -> Abc_DEF
            if (!subHead) {
                nameTmp.insert(nameTmp.end(), '_');
                subHead = true;
            } else {
                string::iterator iterNext = iter + 1;
                // sample: AbcDEf -> Abc_D_Ef, Abc2DEF -> Abc2DEF, ABC2DEF -> ABC2DEF
                if (iterNext != funcName.end()) {
                    if (islower(*iterNext)) {
                        nameTmp.insert(nameTmp.end(), '_');
                    }
                }
            }
        }
        nameTmp.insert(nameTmp.end(), *iter);
    }
    transform(nameTmp.begin(), nameTmp.end(), nameTmp.begin(), ::tolower);
    funcName = nameTmp;
}

/**
 * @brief: get module name
 * @param [in] moduleName         op module name
 * @return [out] bool             succ or fail for geting module name
 */
bool TbeOpInfo::GetModuleName(std::string& moduleName) const
{
    moduleName = op_module_name_;

    // using default module name
    if (moduleName == "") {
        moduleName = "topi/cce/";
    } else if (moduleName == ".") {
        // used for op file in python env path
        moduleName = "";
    } else {
        // using custom module name
        int32_t moduleLen = moduleName.length();
        TE_FUSION_CHECK(moduleLen <= 0, {
            REPORT_TE_INNER_ERROR("Module name length should be greater than 0; it's currently [%d].", moduleLen);
            return false;
        });
        if (moduleName.at(moduleLen - 1) != '/') {
            moduleName += "/";
        }
    }
    if (op_file_name_ != "") {
        moduleName += op_file_name_;
    } else {
        std::string funcName = op_type_;
        NormalizeFuncName(funcName);
        moduleName += funcName;
    }
    return true;
}

/**
 * @brief: get op function name
 * @param [in] funcName           op function name
 * @return [out] void
 */
void TbeOpInfo::GetFuncName(std::string& funcName) const
{
    if (op_func_name_ != "") {
        funcName = op_func_name_;
    } else {
        funcName = op_type_;
        NormalizeFuncName(funcName);
    }
}
}
