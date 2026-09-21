/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include "stub_op_desc.h"
#include "stub_continuous_vector.h"

using std::vector;
using std::string;

#define GE_FUNC_DEV_VISIBILITY __attribute__((visibility("hidden")))
#define GE_FUNC_HOST_VISIBILITY __attribute__((visibility("hidden")))

#ifndef __STUB_ADAPTER_FOR_CONTEXT
#define __STUB_ADAPTER_FOR_CONTEXT

typedef void (*FreeCallback)(void *);
typedef struct {
  union {
    void *pointer{nullptr};
    unsigned char inplace[sizeof(void *)];
  } data;
  FreeCallback deleter{nullptr};
} AsyncAnyValue;

namespace gert {

class Shape {
public:
    static constexpr size_t kMaxDimNum = 8;
    static constexpr int64_t kInvalidDimValue = 0;

public:
    Shape() : dim_num_(0), dims_{0} {}
    /**
     * 通过dims值构造shape，例如：Shape({8,3,224,224})创建一个Shape实例，有4个维度，每个维度的值分别是8,3,224,224
     * @param dims shape的所有dim值
     */
    Shape(std::initializer_list<int64_t> args) : dim_num_(0), dims_{0} {
        if (args.size() > kMaxDimNum) {
            return;
        }
        dim_num_ = args.size();
        size_t i = 0;
        for (auto arg : args) {
            dims_[i++] = arg;
        }
    }
    /**
     * 获取dim num
     * @return
     */
    size_t GetDimNum() const {
        return dim_num_;
    }

    void SetDimNum(size_t dim_num) {
        this->dim_num_ = dim_num;
    }

    /**
     * 获取dim值
     * @param idx dim的index，调用者需要保证index合法
     * @return dim值，在idx超出MaxDimNum时，返回`kInvalidDimValue`
     */
    int64_t GetDim(size_t idx) const {
        if (idx >= kMaxDimNum) {
            return kInvalidDimValue;
        }
        return dims_[idx];
    }

    /**
     * 设置dim值
     * @param idx dim的index，调用者需要保证index合法
     * @return
     */
    void SetDim(size_t idx, int64_t dim_value) {
        if (idx >= kMaxDimNum) {
            return;
        }
        dims_[idx] = dim_value;
        this->dim_num_ = (this->dim_num_ < idx) ? idx : this->dim_num_;
    }

    /**
     * 向后扩展一个dim值，如果扩展的dim数量超出Shape的最大限制，那么本函数不做任何事情
     * @param 扩展的dim值
     * @return this引用
     */
    Shape& AppendDim(int64_t value) {
        if (this->dim_num_ >= kMaxDimNum) {
        return *this;
        }
        dims_[this->dim_num_++] = value;
        return *this;
    }

protected:
    size_t dim_num_;
    int64_t dims_[kMaxDimNum];
};

class StorageShape {
public:
    StorageShape() = default;
    StorageShape(Shape storage_shape) : storage_shape_(storage_shape) {}
    const Shape &GetStorageShape() const { return storage_shape_; }
    const Shape &GetOriginShape() const { return storage_shape_; }

protected:
    Shape storage_shape_;
};

class TensorData {
public:
    TensorData() = default;
    TensorData(void* addr) : addr_(addr) {}
    TensorData(void* addr, size_t size) : addr_(addr), size_(size) {}
    ~TensorData() {}
    uint32_t SetAddr(void* addr) {
        addr_ = addr;
        return 0;
    }
    void* GetAddr() const {
        return addr_;
    }
    size_t GetSize() const {
        return size_;
    }
public:
    void* addr_{nullptr};
    size_t size_{3};
};

class Tensor {
public:
    Tensor() = default;
    Tensor(void* addr) : tensor_data_(addr), shape_(){}
    Tensor(void* addr, size_t size) : tensor_data_(addr, size), shape_(){}
    void SetVale(void* addr) {
        tensor_data_.SetAddr(addr);
    }
    const TensorData &GetTensorData() const { return tensor_data_; }
      ge::DataType GetDataType() const { return data_type_; }
    const Shape &GetStorageShape() const { return shape_; }

public:
    TensorData tensor_data_;
    Shape shape_;
    ge::DataType data_type_{ge::DataType::DT_FLOAT};
};

class StorageFormat {
public:
    StorageFormat() = default;

    StorageFormat(ge::Format storage_format) : storage_format_(storage_format), origin_format_(storage_format) {}

    ge::Format GetStorageFormat() const {
        return storage_format_;
    }

    void SetStorageFormat(ge::Format storage_format) {
        storage_format_ = storage_format;
    }

    ge::Format GetOriginFormat() const {
        return origin_format_;
    }

    void SetOriginFormat(ge::Format origin_format) {
        origin_format_ = origin_format;
    }

protected:
    ge::Format storage_format_{};
    ge::Format origin_format_{};
};

class GertTensorDesc {
public:
    GertTensorDesc() = default;
    GertTensorDesc(StorageFormat storageFormat, ge::DataType data_type) : storage_format_(storageFormat), data_type_(data_type) {}
    StorageFormat GetFormat() const { return storage_format_; }
    ge::DataType GetDataType() const { return data_type_; }

protected:

    StorageFormat storage_format_{};
    ge::DataType data_type_{};
};

template<typename T>
class TypedContinuousVector {
 public:
  size_t GetSize() const {
    return 2;
  }

  void SetSize(const size_t size) {
    size_ = size;
  }

  /**
   * 获取首个元素的指针地址，[GetData(), GetData() + GetSize()) 中的数据即为当前容器中保存的数据
   * @return 首个元素的指针地址
   */
  const T *GetData() const {
    return static_cast<const T *>(elements);
  }

private:
  size_t size_;;
  T elements[8];
};

class RuntimeAttrs {
 public:
  /**
   * 获取string类型属性
   * @param index 属性index
   * @return 指向属性值的指针
   */
  const char *GetStr(const size_t index) const {
      return str_;
  }

  const int64_t* GetInt(const size_t index) const {
      return &factor_;
  }

  const float* GetFloat(const size_t index) const {
      return &floatValue_;
  }

  size_t GetAttrNum() const {
    return attrNum_;
  }

  const TypedContinuousVector<int64_t> *GetListInt(const size_t index) const {
    return reinterpret_cast<const TypedContinuousVector<int64_t> *>(vectorInt.GetData());
  }

  const TypedContinuousVector<float> *GetListFloat(const size_t index) const {
    return reinterpret_cast<const TypedContinuousVector<float> *>(vectorFloat.GetData());
  }

  const bool *GetBool(const size_t index) const {
    return &bool_factor_;
  }

  const gert::ContinuousVectorVector *GetListListInt(const size_t index) const {
    return &vectorVectorFactor;
  }

 private:
    TypedContinuousVector<int64_t> vectorInt;
    TypedContinuousVector<float> vectorFloat;
    const char *str_ = "HWC";
    int64_t factor_{255};
    float floatValue_{10};
    size_t attrNum_{6};
    bool bool_factor_{true};
    gert::ContinuousVectorVector vectorVectorFactor;
};

class Chain {
public:
    using Deleter = void (*)(void *);
    Chain();
    ~Chain();

    void FreeResource() {
        if (any_value_.deleter != nullptr) {
            any_value_.deleter(any_value_.data.pointer);
        }
    }

    template<typename T>
    static void DefaultArrayDeleter(T *data) {
        delete[] data;
    }

    template<typename T>
    static void DefaultDeleter(T *data) {
        delete data;
    }

    void Set(void * const data, const Chain::Deleter deleter) {
        FreeResource();
        any_value_.data.pointer = data;
        any_value_.deleter = deleter;
    }

    template<typename T, typename PureT = typename std::remove_extent<T>::type,
           typename std::enable_if<std::is_array<T>::value, int>::type = 0>
    void SetWithDefaultDeleter(PureT *data) {
        Set(data, reinterpret_cast<FreeCallback>(DefaultArrayDeleter<PureT>));
    }

    void SetWithDefaultDeleter(void* data);

    uint8_t* tmp;
    uint8_t* data_;
    AsyncAnyValue any_value_;
};

class KernelContext {
public:
    KernelContext();
    ~KernelContext();
    size_t GetInputNum() const {
        return 2UL;
    }

    Chain* GetOutput(size_t i);

    template<typename T>
    const T GetInputValue(size_t i) const {
        return reinterpret_cast<T>(1UL);
    }
    template<typename T>
    T* GetOutputPointer(size_t i) {
        return reinterpret_cast<T*>(0);
    }
    
    template<typename T>
    T *GetInputPointer(size_t i) {
        return reinterpret_cast<T*>(0);
    }

    Chain* chainArray_[4];


protected:

};

//template<>
//gert::ContinuousVector* KernelContext::GetOutputPointer<gert::ContinuousVector>(size_t i);

class DvppContext {
public:
    DvppContext() = default;
    ~DvppContext() = default;
    void* GetStream() {
        return (void*)1;
    }
    void SetType(string type) { type_ = type; } // 1, Type OK
    void SetNodeNum(uint32_t nodeNum) { nodeNum_ = nodeNum; } // 1, Type OK
    string GetNodeType() { return type_; }

    StorageShape* GetInputShape(uint32_t idx) { return inputShape[idx]; }

    Tensor* GetInputTensor(uint32_t idx) { if(idx + 1 > (inputTensor.size())) { return nullptr; } return inputTensor[idx]; }
    Tensor* GetOutputTensor(uint32_t idx) { return outputTensor[idx]; }
    GertTensorDesc* GetInputDesc(uint32_t idx) { return inputDescs[idx]; }

    GertTensorDesc* GetOutputDesc(uint32_t idx) { return outputDescs[idx]; }
    StorageShape* GetOutputShape(uint32_t idx) { return outputShape[idx]; }

    void AddInputShape(StorageShape* storageShape) { inputShape.push_back(storageShape); }
    void AddOutputShape(StorageShape* storageShape) { outputShape.push_back(storageShape); }
    void AddInputDesc(GertTensorDesc* gertDesc) { inputDescs.push_back(gertDesc); }
    void AddOutputDesc(GertTensorDesc* gertDesc) { outputDescs.push_back(gertDesc); }
    void AddInputTensor(Tensor* tensor) { inputTensor.push_back(tensor); }
    void AddOutputTensor(Tensor* tensor) { outputTensor.push_back(tensor); }
    uint32_t GetComputeNodeInputNum() { return nodeNum_; }
    uint32_t GetComputeNodeOutputNum() { return nodeNum_; }

    const RuntimeAttrs *GetAttrs() const
    {
        return &runtimeAttrs;
    }

    RuntimeAttrs runtimeAttrs;
protected:
    string type_;
    int32_t nodeNum_{0};
    vector<GertTensorDesc*> inputDescs, outputDescs;
    vector<Tensor*> inputTensor, outputTensor;
    vector<StorageShape*> inputShape, outputShape;
};

}

namespace gert {
class GertTensorData {
 public:
  GertTensorData() {
  }

  void *GetAddr() {
    return ptr_;
  }

 private:
  void *ptr_{};
};
}

namespace ge {
using char_t = char;

class Node {
public:
    Node();
    ~Node();
};
}
#endif