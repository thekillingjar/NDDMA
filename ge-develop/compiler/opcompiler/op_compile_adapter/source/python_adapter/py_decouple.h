/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_PYTHON_ADAPTER_PY_DECOUPLE_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_PYTHON_ADAPTER_PY_DECOUPLE_H_

#include <Python.h>
#include <string>
#include <mutex>

namespace te {
namespace fusion {
using std::string;

class HandleManager {
public:

    using TEPyObjectCallObject       = PyObject* (*)(PyObject*, PyObject*);
    using TEPyRunSimpleString        = PyObject* (*)(const char*);
    using TEPyDictNew                = PyObject* (*)();
    using TEPyObjectStr              = PyObject* (*)(PyObject*);
    using TEPyDictSetItemString      = int (*)(PyObject*, const char*, PyObject*);
    using TEPyArgParse               = int (*)(PyObject*, const char*, ...);
    using TEPyArgParseTuple          = int (*)(PyObject*, const char*, ...);
    using TEPyBuildValue             = PyObject* (*)(const char*, ...);
    using TEPyObjectCallFunction     = PyObject* (*)(PyObject*, const char*, ...);
    using TEPyObjectCallMethodSizeT  = PyObject* (*)(PyObject*, const char*, const char*, ...);
    using TEPyDictGetItem            = PyObject* (*)(PyObject*, PyObject*);
    using TEPyDictGetItemString      = PyObject* (*)(PyObject*, const char*);
    using TEPyDictKeys               = PyObject* (*)(PyObject*);
    using TEPyErrFetch               = void (*)(PyObject**, PyObject**, PyObject**);
    using TEPyErrNormalizeException  = void (*)(PyObject**, PyObject**, PyObject**);
    using TEPyErrPrint               = void (*)();
    using TEPyEvalInitThreads        = void (*)();
    using TEPyEvalRestoreThread      = void (*)(PyThreadState*);
    using TEPyEvalSaveThread         = PyThreadState* (*)();
    using TEPyEvalThreadsInitialized = int (*)();
    using TEPyFinalize               = void (*)();
    using TEPyFloatFromDouble        = PyObject* (*)(double);
    using TEPyGILStateCheck          = int (*)();
    using TEPyGILStateEnsure         = PyGILState_STATE (*)();
    using TEPyGILStateRelease        = void (*)(PyGILState_STATE);
    using TEPyImportImportModule     = PyObject* (*)(const char*);
    using TEPyInitialize             = void (*)();
    using TEPyListGetItem            = PyObject* (*)(PyObject*, Py_ssize_t);
    using TEPyIsInitialized          = int (*)();
    using TEPyListNew                = PyObject* (*)(Py_ssize_t);
    using TEPyListSetItem            = int (*)(PyObject*, Py_ssize_t, PyObject*);
    using TEPyListSize               = Py_ssize_t (*)(PyObject*);
    using TEPyLongFromLong           = PyObject* (*)(long);
    using TEPyObjectCall             = PyObject* (*)(PyObject*, PyObject*, PyObject*);
    using TEPyObjectGetAttrString    = PyObject* (*)(PyObject*, const char*);
    using TEPyObjectHasAttrString    = int (*)(PyObject*, const char*);
    using TEPyObjectIsTrue           = int (*)(PyObject*);
    using TEPyTupleGetItem           = PyObject* (*)(PyObject*, Py_ssize_t);
    using TEPyTupleNew               = PyObject* (*)(Py_ssize_t);
    using TEPyTupleSetItem           = int (*)(PyObject*, Py_ssize_t, PyObject*);
    using TEPyTupleSize              = Py_ssize_t (*)(PyObject*);
    using TEPyUnicodeAsUTF8          = const char* (*)(PyObject*);
    using TEPyUnicodeFromString      = PyObject* (*)(const char*);
    using TEPyDealloc                = void (*)(PyObject *);
    using TEPyObjPtr                 = PyObject *;

    TEPyObjectGetAttrString    TE_PyObject_GetAttrString;
    TEPyObjectHasAttrString    TE_PyObject_HasAttrString;
    TEPyObjectIsTrue           TE_PyObject_IsTrue;
    TEPyTupleGetItem           TE_PyTuple_GetItem;
    TEPyTupleNew               TE_PyTuple_New;
    TEPyTupleSetItem           TE_PyTuple_SetItem;
    TEPyTupleSize              TE_PyTuple_Size;
    TEPyUnicodeAsUTF8          TE_PyUnicode_AsUTF8;
    TEPyUnicodeFromString      TE_PyUnicode_FromString;
    TEPyGILStateRelease        TE_PyGILState_Release;
    TEPyDictSetItemString      TE_PyDict_SetItemString;
    TEPyRunSimpleString        TE_PyRun_SimpleString;
    TEPyObjectCallObject       TE_PyObject_CallObject;
    TEPyDictNew                TE_PyDict_New;
    TEPyObjectStr              TE_PyObject_Str;
    TEPyArgParse               _PyArg_Parse;
    TEPyArgParseTuple          _PyArg_ParseTuple;
    TEPyBuildValue             _Py_BuildValue;
    TEPyObjectCallFunction     _PyObject_CallFunction;
    TEPyObjectCallMethodSizeT  TE_PyObject_CallMethod_SizeT;
    TEPyDictGetItem            TE_PyDict_GetItem;
    TEPyDictGetItemString      TE_PyDict_GetItemString;
    TEPyDictKeys               TE_PyDict_Keys;
    TEPyErrFetch               TE_PyErr_Fetch;
    TEPyErrNormalizeException  TE_PyErr_NormalizeException;
    TEPyErrPrint               TE_PyErr_Print;
    TEPyEvalInitThreads        TE_PyEval_InitThreads;
    TEPyEvalRestoreThread      TE_PyEval_RestoreThread;
    TEPyEvalSaveThread         TE_PyEval_SaveThread;
    TEPyEvalThreadsInitialized TE_PyEval_ThreadsInitialized;
    TEPyFinalize               TE_Py_Finalize;
    TEPyFloatFromDouble        TE_PyFloat_FromDouble;
    TEPyGILStateCheck          TE_PyGILState_Check;
    TEPyGILStateEnsure         TE_PyGILState_Ensure;
    TEPyImportImportModule     TE_PyImport_ImportModule;
    TEPyInitialize             TE_Py_Initialize;
    TEPyListGetItem            TE_PyList_GetItem;
    TEPyIsInitialized          TE_Py_IsInitialized;
    TEPyListNew                TE_PyList_New;
    TEPyListSetItem            TE_PyList_SetItem;
    TEPyListSize               TE_PyList_Size;
    TEPyLongFromLong           TE_PyLong_FromLong;
    TEPyObjectCall             TE_PyObject_Call;
    TEPyDealloc                TE_Py_Dealloc;

    HandleManager(const HandleManager &) = delete;
    HandleManager &operator=(const HandleManager &) = delete;
    static HandleManager& Instance();

    bool IsPyEnvInitBeforeTbe() const;

    bool Initialize();

    bool Finalize();

    PyObject* get_py_true() {
        return TE_py_true;
    }

    PyObject* get_py_false() {
        return TE_py_false;
    }

    PyObject* get_py_none() {
        return TE_py_none;
    }

private:
    HandleManager();
    ~HandleManager();

    bool isInit_ = false;
    void* pyHandle = nullptr;
    PyObject* TE_py_true = nullptr;
    PyObject* TE_py_false = nullptr;
    PyObject* TE_py_none = nullptr;
    PyThreadState *pyThreadState_ = nullptr;
    bool pyEnvStatusBeforeTbe_ = false;  // record the python init state before te_fusion execution
    mutable std::mutex pyhandle_mutex_;
    bool CheckPythonVersion(FILE *fp, char* line, bool flag) const;
    bool LaunchDynamicLib();
    bool LoadFuncs(bool isDynamic);
    bool HandleClose() const;
    bool LoadDynLibFromPyCfg(std::string &pythonLib, void *handle) const;
    bool CheckCommandValid(string &command, char *line) const;
};
}
}
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_PYTHON_ADAPTER_PY_DECOUPLE_H_
