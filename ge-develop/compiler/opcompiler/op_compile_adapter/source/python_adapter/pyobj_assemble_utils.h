/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_PYTHON_ADAPTER_PYOBJ_ASSEMBLE_UTILS_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_PYTHON_ADAPTER_PYOBJ_ASSEMBLE_UTILS_H_

#include <map>
#include <string>
#include <Python.h>
#include "tensor_engine/tbe_op_info.h"

namespace te {
namespace fusion {
std::string PyObjectToStr(PyObject *&pyObj);
/*
  * @brief: convert op total parameters from class to PyObject
  * @param [in] opinfo: op total parameter set
  * @param [out] pArgs: op all PyObject parameter set
  * @return bool: convert op parameter ok or not
  */

bool SetRangeToPyObj(const bool &hasSet, std::vector<std::pair<int64_t, int64_t>> &shapeRange,
                     const std::vector<int64_t> &shapes, const std::string &rangeType, PyObject *pyTensor);

bool GetPyObjOfShape(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo);

bool GetPyObjOfOriShape(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo);

bool SetStringInfoToPyObj(const std::string &info, const std::string &key, PyObject *&pyTensor);

bool GetPyObjOfFormat(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo);

bool GetPyObjOfCurSubFormat(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo);

bool GetPyObjOfOriFormat(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo);

bool GetPyObjOfDtype(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo);

bool GetPyObjOfAddrType(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo);

bool GetPyObjOfCAxisValue(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo);

bool GetPyObjOfValidShape(const TbeOpTensor &paraTensor, PyObject *&pyTensor, const bool &isInput, const TbeOpInfo &opinfo);

bool GetPyObjOfSliceOffset(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo);

bool GetPyObjOfL1Info(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo);

bool GetPyObjOfTotalShape(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo);

bool GetPyObjSplitIndex(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo);

bool GetPyObjOfIsFirstLayer(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo);

bool GetPyObjOfShapeRange(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo);

bool GetPyObjOfOriginalShapeRange(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo);

bool GetPyObjOfValueRange(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo);

bool GetPyObjOfConstValue(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo);

bool GetPyObjAtomicAtrr(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &, const TbeOpInfo &opinfo);

bool GetPyTensor(const TbeOpTensor &paraTensor, PyObject *&pyTensor, bool &isInput, const TbeOpInfo &opinfo);

bool AddOptTensorArgs(const std::vector<TbeOpTensor> &tensors, PyObject *&pArgs, int32_t &argsIdx, bool &isInput,
                      const TbeOpInfo &opinfo);

bool AddDynTensorArgs(const std::vector<TbeOpTensor> &tensors, PyObject *&pArgs, int32_t &argsIdx, bool &isInput,
                      const TbeOpInfo &opinfo);

bool AddReqTensorArgs(const std::vector<TbeOpTensor> &tensors, PyObject *&pArgs, int32_t &argsIdx, bool &isInput,
                      const TbeOpInfo &opinfo);

bool SimpleAttrForFp32(const TbeAttrValue &attr, PyObject *&pyAttr);

bool SimpleAttrForDouble(const TbeAttrValue &attr, PyObject *&pyAttr);

bool SimpleAttrForBool(const TbeAttrValue &attr, PyObject *&pyAttr);

bool SimpleAttrForString(const TbeAttrValue &attr, PyObject *&pyAttr);

bool SimpleAttrForInt8(const TbeAttrValue &attr, PyObject *&pyAttr);

bool SimpleAttrForUint8(const TbeAttrValue &attr, PyObject *&pyAttr);

bool SimpleAttrForInt16(const TbeAttrValue &attr, PyObject *&pyAttr);

bool SimpleAttrForUint16(const TbeAttrValue &attr, PyObject *&pyAttr);

bool SimpleAttrForInt32(const TbeAttrValue &attr, PyObject *&pyAttr);

bool SimpleAttrForUint32(const TbeAttrValue &attr, PyObject *&pyAttr);

bool SimpleAttrForInt64(const TbeAttrValue &attr, PyObject *&pyAttr);

bool SimpleAttrForUint64(const TbeAttrValue &attr, PyObject *&pyAttr);

bool GetPyAttrSimple(const TbeAttrValue &attr, PyObject *&pyAttr);

bool ComplexAttrForListFp32(const TbeAttrValue &attr, PyObject *&pyAttr);

bool ComplexAttrForListDouble(const TbeAttrValue &attr, PyObject *&pyAttr);

bool ComplexAttrForListBool(const TbeAttrValue &attr, PyObject *&pyAttr);

bool ComplexAttrForListString(const TbeAttrValue &attr, PyObject *&pyAttr);

bool ComplexAttrForListInt64(const TbeAttrValue &attr, PyObject *&pyAttr);

bool GetPyAttrComplex(const TbeAttrValue &attr, PyObject *&pyAttr);

bool PyObjectToPyFullAttr(const TbeAttrValue &attr, PyObject *&pyAttr, PyObject *&pyFullAttr,
                          std::vector<std::string> &variableAttrs);

bool AddAttrArgs(const std::vector<TbeAttrValue> &attrs, PyObject *&pArgs, int32_t &argsIdx, bool isSingleOpBuild,
                 std::vector<std::string> &variableAttrs);

bool AssembleInputsAndOutputs(const TbeOpInfo &opinfo, PyObject *&pyInputs, PyObject *&pyOutputs);

bool AssembleOpArgs(const TbeOpInfo &opinfo, const std::string &kernelName, PyObject *&pyInputs, PyObject *&pyOutputs,
                    PyObject *&pyAttrs, bool isSingleOpBuild = false);

bool AssembleAttrs(const TbeOpInfo &opInfo, const std::string &kernelName, PyObject *&pyAttrs,
                   bool isSingleOpBuild = false);

bool AssembleOpPrivateAttrs(const TbeOpInfo &opInfo, PyObject *&pyPrivateAttrs, bool isSingleOpBuild);

bool AssemleStringDict(const std::map<std::string, std::string> &infoMap, PyObject *&pyInfoDict);

bool GetPyAttr(const TbeAttrValue &attr, PyObject *&pyAttr);

bool GetPyConstValue(const TbeOpTensor &paraTensor, PyObject *&pyTensor);

bool AddTensorArgs(const std::vector<TbeOpParam> &puts, PyObject *&pArgs, int32_t &argsIdx, bool &isInput,
                   const TbeOpInfo &opinfo);
}
}
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_PYTHON_ADAPTER_PYOBJ_ASSEMBLE_UTILS_H_