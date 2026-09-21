# GE IR实现教程
GE IR（Intermediate Representation）用于从用户模型表达翻译到可执行代码，也是对一个算子计算语义的规格描述。本文将对GE IR的定义及相关的开发工作做介绍。
## 实现IR的交付件

在runtime2下，第一、第二类算子的IR实现需要包含如下几部分：

* DataType推导（必选）
* Shape推导（必选）
* ShapeRange推导（可选）
* TIling功能（必选）
* 声明数据依赖（可选）

## 算子分类

根据算子在shape推导和下发逻辑的区分，将算子分为四类。

| 分类 | 特点                                                         | 可选交付件要求                            |
| ---- | ------------------------------------------------------------ |------------------------------------|
| 一类 | 根据输入shape可以推导出输出shape。                           | 1.InferShpeRange(按需)               |
| 二类 | 依赖输入tensor value才能推导出输出shape。 如Reshape算子，依赖shape输入的value推导输出shape。此类又称为值依赖算子。 | 1.InferShpeRange(按需)<br />2.声明数据依赖 |
| 三类 | 编译时无法推导输出shape，只能推导输出shape range。执行完才能得出输出shape。 如Where算子。<br />在下发时需要按照输出shape range来申请最大输出内存，因此InferShpeRange是必选交付件 | 1.InferShpeRange<br />2.声明数据依赖（按需） |
| 四类 | 无法推导出shape。也不知道输出shape range。如 aicpu融合算子、某些资源类算子。此类算子通常调用tf原生算子实现，因此无IR交付件需要。 | NA                                 |

## 注册

以`Cast`算子为例，通过如下方式注册算子实现：

```C++
#include "register/op_impl_registry.h"  // 包含IMPL_OP需要的头文件

// 实现`InferShape`函数
ge::graphStatus InferShapeForCast(InferShapeContext *context) {
  // TODO
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(Cast)          // 使用 `IMPL_OP` 注册算子实现，该宏接受一个参数，参数即为需要注册的算子类型
    .InferShape(InferShapeForCast);   // 注册 InferShape 函数
```

## 实现 InferDataType

### 1.IR表达推导逻辑

当前支持基于原型的声明式datatype推导，在原型定义上使用泛化的类型，来定义输入/输出的datatype映射关系。

以Add算子为例，原型定义时，如下声明datatype推导规则：

```c++
REG_OP(Add)
    .INPUT(x1, "T")
    .INPUT(x2, "T")
    .OUTPUT(y, "T")
    .OP_END_FACTORY_REG(Add)
```

在x1、x2、y后面的字符串"T"代表了该输入/输出的datatype类型，该声明表达了两层含义：

1. Add算子的两个输入必须拥有相同的数据类型
2. Add算子的输出与输入的类型一致



有时算子对输入输出的数据类型有一定的要求，例如Add算子仅支持对实数做计算，而不支持DT_RESOURCE等复杂的资源类数据，此时可以使用`DATATYPE`来限制类型T的范围：

```c++
REG_OP(Add)
    .INPUT(x1, "T")
    .INPUT(x2, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType::NumberType())  // NumberType包含INT,FLOAT等数据类型，详情在"graph/types.h"中查看
    .OP_END_FACTORY_REG(Add)
```



当输入为DYNAMIC_INPUT时，会出现两种可能：

* 该DYNAMIC_INPUT实例化[^1]后所有输入的datatype类型一致
* 该DYNAMIC_INPUT实例化后的输入中，互相之间没有datatype的约束关系



DATATYPE宏配合T语法使用，表示该数据类型的有效范围，其数据类型只能为TensorType或ListTensorType。若使用了T语法，但未定义DATATYPE，则表示只做数据类型映射，放弃数据类型校验。

如下为常见datatype推导算子原型定义示例。

#### 1.1.输出由输入推导

以StridedSlice算子为例。

输出y的datatype和输入x一致，因此他们使用同一个符号T。T的数据类型可选范围为TensorType::BasicType()，由DATATYPE宏声明。

begin、end、strides输入的数据类型一致，都使用TIndex符号，其数据类型范围为TensorType::IndexNumberType()。

```c++
REG_OP(StridedSlice)
    .INPUT(x, "T")
    .INPUT(begin, "TIndex")
    .INPUT(end, "TIndex")
    .INPUT(strides, "TIndex")
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
    .DATATYPE(T, TensorType::BasicType())
    .DATATYPE(TIndex, TensorType::IndexNumberType())
    .OUTPUT(y, "T")
    .OP_END_FACTORY_REG(StridedSlice)
```

##### 1.1.1 ListTensorType的使用

还可以使用ListTensorType来描述某个符号对应的取值范围，此时该符号所对应的输入/输出应该是动态的。且表示该动态输入/输出的实例datatype取值范围一样，但datatype不必一样。如IdentityN。

```c++
REG_OP(IdentityN)
    .DYNAMIC_INPUT(x, "T")
    .DYNAMIC_OUTPUT(y, "T")
    .DATATYPE(T, ListTensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(IdentityN)
```

再举一个动态输入但不使用ListTensorType的例子，ConcatV2。输出y与输入x的datatype一致，且input_x0...input_xi的datatype都一致，此时T还是使用TensorType表示。

```c++
REG_OP(ConcatV2)
    .DYNAMIC_INPUT(x, "T")
    .INPUT(concat_dim, TensorType::IndexNumberType())
    .OUTPUT(y, "T")
    .ATTR(N, Int, 1)
    .DATATYPE(T, TensorType::BasicType())
    .OP_END_FACTORY_REG(ConcatV2)
```

> WARNING：如果一个输入的DataType是由一个可选输入决定的，那么在可选输入不存在时，该输出的DataType将无法被推导，因此需要考虑可选输入不存在的场景。一般来说，这类算子建议写自定义推导规则（见自定义推导章节）

#### 1.2.输出由属性推导

以Cast算子为例，`Cast`算子将输入Tensor的数据类型转换为属性指定的`dst_type`。属于属性指定输出datatype的推导类型。Cast算子的原型要定义为：

```c++
REG_OP(Cast)
    .INPUT(x, "T")
    .OUTPUT(y, "dst_type")
    .REQUIRED_ATTR(dst_type, Int)
    .DATATYPE(T, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BF16}))
    .DATATYPE(dst_type, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BF16}))
    .OP_END_FACTORY_REG(Cast)
```

如上的IR表示，输出y的datatype由dst_type符号映射到属性dst_type上。其中dst_type的数据类型范围由DATATYPE宏来限定。

#### 1.3.输出为固定datatype

以Equal算子为例，其输出固定为DT_BOOL。那么指定输出符号的TensorType里只有一个值DT_BOOL即可。

```c++
REG_OP(Equal)
    .INPUT(x1, "TIn")
    .INPUT(x2, "TIn")
    .OUTPUT(y, "TOut")
    .DATATYPE(TIn, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8,DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64,DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32})
    .DATATYPE(TOut, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(Equal)
```

#### 1.4. 类型提升规则（todo）

### 2.自定义推导函数

对于自定义推导，无法在IR上表达其映射关系，请注册实现infer data type接口，若算子自注册了infer data type接口，则优先调用该接口。

```c++
ge::graphStatus InferDataTypeForFoo(gert::InferDataTypeContext* context) {
    // TODO 实现自定义推导规则
    if (context->GetInputDataType(0) == DT_INT4) {
        context->SetOutputDataType(0, DT_INT32);
    }
}

IMPL_OP(Foo).InferDataType(InferDataTypeForFoo);// 注册
```

## 实现 InferShape

以`Cast`算子为例，`Cast`算子将输入Tensor的数据类型转换为属性指定的`dst_type`，`Cast`算子的原型为：

```c++
REG_OP(Cast)
    .INPUT(x, "T")
    .OUTPUT(y, "dst_type")
    .REQUIRED_ATTR(dst_type, Int)
    .DATATYPE(T, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BF16}))
    .DATATYPE(dst_type, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BF16}))
    .OP_END_FACTORY_REG(Cast)
```

从`Cast`算子的功能可以看出，其输出Shape等于输入Shape，InferShape函数实现为：

```c++
ge::graphStatus InferShapeForCast(InferShapeContext *context) {
  // 获取第0个输入和输出的指针
  const gert::Shape *shape = context->GetInputShape(0);
  gert::Shape *output_shape = context->GetOutputShape(0); 
  if (shape == nullptr || output_shape == nullptr) {
    // 防御式编程，不应该出现的场景，打印错误并返回失败
  GELOGE(ge::PARAM_INVALID,
         "Failed to infer shape for node %s, type %s, nullptr found",
         context->GetNodeName(),
         context->GetNodeType());
    return ge::GRAPH_FAILED;
  }

  *output_shape = *shape;  // 将输入shape赋值到输出shape
  
  return ge::GRAPH_SUCCESS;
}
```

InferShape函数的原型是确定的，其接受一个`InferShapeContext`作为输入，从此context上，可以获取到输入、输出的shape指针等内容。InferShape成功后，返回ge::GRAPH_SUCCESS，其他返回值被认为推导失败。推导失败后，执行过程结束退出。

从上述代码可以看出，输入shape为const的，因此InferShape时，输入shape是只读、不允许修改的。

`InferShapeContext` public继承自`ExtendedKernelContext`，因此`ExtendedKernelContext`中提供的方法如获取算子type、name、属性等接口均可以在`InferShapeContext`实例中调用。`InferShapeContext`和`ExtendedKernelContext`的详细接口参考《runtime2.0 API specification》。

> WARNING: 为了效率考虑，调用InferShape函数时，框架不会为输出shape做初始化，因此，在InferShape函数中，可以认为输出是**未初始化**的状态。如果在InferShape时，希望通过Append方式操作输出Shape，需要先将输出Shape的DimNum清零，以防止出现未定义行为。

## 实现InferShapeRange

三类算子InferShapeRange是必选交付件。

一类、二类算子，其InferShapeFunc满足如下条件时，可以使用框架自动推导机制，无需提供自定义InferShapeRange。

1. 单输入单输出算子，InferShapeFunc满足单调性。单调性是指，在给定区间内，给定输入x的值，输出y的值是唯一的。若不满足单调性，自动推导结果将是不准确的。 
2. 多输入多输出算子，因自动推导框架需要选取输入shape range的min和max值来灌入InferShape函数，对于多输入的算子，不同输入的不同shape组合都需要合法。

因此算子开发者需要根据算子InferShape的数学语义，判断是否需要提供自定义的InferShapeRange函数。若因算子未提供InferShapeRange函数，导致推导结果错误，只能在后续执行时报错。

### 1.自定义InferShapeRange

以Where算子为例，其IR定义如下所示。

```c++
/**
* @brief Returns locations of nonzero / true values in a tensor. \n

* @par Inputs:
* Including:
* @li x: A Tensor. Must be one of the following types:
  DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_QINT8,
  DT_QUINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_QINT32,
  DT_INT64, DT_UINT64, DT_BOOL, DT_COMPLEX64, DT_COMPLEX128 \n

* @par Outputs:
* @li y: A Tensor of type DT_INT64. \n

* @attention Constraints:
* Where runs on the Ascend AI CPU, which delivers poor performance.\n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Where.
*/

REG_OP(Where)
    .INPUT(x, TensorType({BasicType(), DT_BOOL}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(Where)
```

Where算子的语义为，返回一个Tensor中非0/为True的坐标。

Where的输出y_shape将是[-1, Rank(x_shape)]。其中第一个轴取值依赖于非0值的个数，第二个轴取值取决于x_shape的维度。

例如：输入x是一个4维Tensor，输入shape为[4, 5, 6, 7]。输出shape为[-1, 4]。

那么y_shape的第一个轴取值，最小为0，最大是输入x中value的个数（即x_shape_size，x shape各轴乘积）。

InferShapeRange的实现为：

```c++
ge::graphStatus InferShapeRangeWhere(gert::InferShapeRangeContext *context) {
  // 获取输入和输出的shape range
  auto x_shape_range = context->GetInputShapeRange(0U);
  auto output_shape_range = context->GetOutputShapeRange(0U);
  OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape_range);
  OPS_CHECK_NULL_WITH_CONTEXT(context, output_shape_range);

  // 获取输入x shape range的max shape
  auto x_max_shape = x_shape_range->GetMax();
  // 获取输入x的rank
  auto x_shape_dimnum = x_max_shape->GetDimNum();
  int64_t output_shape_max = 1;
  // 输出y的rank为2
  size_t output_shape_dim_num = 2U;

  for (size_t i = 0U; i < x_shape_dimnum; i++) {
    // 若输入x的最大range unknown, 则输出y的shape最大range也unknown
    if (x_max_shape->GetDim(i) < 0) {
      output_shape_max = -1;
      break;
    }
    // 若输入x的最大shape range是known的，输出y的shape最大range将是x的最大shape size
    output_shape_max *= x_max_shape->GetDim(i);
  }

  // 设置输出range
  output_shape_range->GetMax()->SetDimNum(output_shape_dim_num);
  output_shape_range->GetMax()->SetDim(0, output_shape_max);
  output_shape_range->GetMax()->SetDim(1, x_shape_dimnum);
  output_shape_range->GetMin()->SetDimNum(output_shape_dim_num);
  output_shape_range->GetMin()->SetDim(0, 0);
  output_shape_range->GetMin()->SetDim(1, x_shape_dimnum);

  return ge::GRAPH_SUCCESS;
}
```

### 2.框架自动推导机制

- 若算子输入上有shape range，代入shape range的min/max值，调用静态infer func。对推导出的output shape逐个轴比对，若dim不同，表示为动态轴。min/max推导得出的输出shape的作为输出shape range的min/max。
- 若算子输入不带shape range，只有动态轴的信息。跳过shape range推导。

## 实现 Tiling

Tiling函数分为两步：TilingParse与Tiling，TilingParse中，将算子编译后产生的Json反序列化，解析为自定义的CompileInfo数据结构。由于反序列化时间较长，TilingParse仅在加载阶段执行一遍，其结果CompileInfo会被缓存，在执行时传给Tiling函数使用。

以TransData算子为例，TilingParse与Tiling的函数原型与注册方式如下所示：

```c++
#include "register/op_impl_registry.h"  // 包含IMPL_OP需要的头文件

// TransData的CompileInfo定义
struct TransDataCompileInfo {
  int64_t ub_size;
  int64_t block_dim;
  int64_t group;
  int64_t vnc_fp32_flag;
};

// TilingParse原型，输入为KernelContext
ge::graphStatus TilingPrepareForTransdata(TilingParseContext *context) {
  return ge::GRAPH_SUCCESS;
}

// Tiling原型，输入为TilingContext
ge::graphStatus TilingForTransData(TilingContext *context) {
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(TransData)
    .InferShape(InferShapeForTransData)
    .Tiling(TilingForTransData)                                     // 注册Tiling函数
    .TilingParse<TransDataCompileInfo>(TilingPrepareForTransdata);  // 注册TilingParse函数
```

由于每个算子的CompileInfo的类型不同，在调用`TilingParse`注册时，将`CompileInfo`的类作为模板参数传入，`CompileInfo`类需要提供默认构造函数。

### 实现 TilingParse

TilingParse接受一个编译后json字符串作为输入，对此json做反序列化、解析，并将解析结果写到其输出上，以TransData的Tiling为例：

```c++
ge::graphStatus TilingPrepareForTransdata(TilingParseContext *context) {
  // 获取编译后的json字符串
  const char *json_str = context->GetCompiledJson();
  if (compile_info == nullptr || json_str == nullptr) {
      GELOGE(ge::PARAM_INVALID,
         "Failed to parse compiled info for node %s, type %s, nullptr found",
         context->GetNodeName(),
         context->GetNodeType());
    return ge::GRAPH_FAILED;
  }

  // TilingPrepare仅有一个输出，即为CompileInfo实例。
  // CompileInfo实例已经由框架根据TilingParse注册时声明的类型创建好，此处直接获取即可，不需要自行申请
  TransDataCompileInfo *compile_info = context->GetCompiledInfo<TransDataCompileInfo>();

  // 使用nlohmann库反序列化json
  std::unique_ptr<nlohmann::json> parsed_object_cinfo(new nlohmann::json(nlohmann::json::parse(json_str)));
  if (parsed_object_cinfo == nullptr) {
    return ge::GRAPH_FAILED;
  }
  
  // 解析json，并将解析后的结果写入到CompileInfo实例
  const nlohmann::json &allVars = (*parsed_object_cinfo)["vars"];
  if (!GetCompileValue(allVars, "ub_size", compile_info->ub_size)) {
    GELOGE(ge::PARAM_INVALID, "Failed to find ub_size in json %s", json_str);
    return ge::PARAM_INVALID;
  }
	// 更多的解析逻辑。。。

  return ge::GRAPH_SUCCESS;
}
```

###  实现 Tiling 函数

Tiling函数的参数是`TilingContext`，此类继承自`ExtendedKernelContext`，因此该context从`ExtendedKernelContext`继承了如下能力：

* 该算子的Ir属性
* 该算子所有输入、输出的原始Format、运行时Format
* 该算子所有输入、输出的运行时DataType
* 该算子输入的实例化信息（关于实例化定义，参考InferShape中的描述）

除此之外，TilingContext提供了Tiling函数额外需要的数据：

* 该算子所有输入、输出的原始shape、运行时shape
* CompileInfo实例

Tiling函数需要产生的输出有：

* tiling-key：默认为0
* block-dim：默认为0
* atomic-clean-flag：默认为true
* tiling-data：默认为空
* workspaces大小：默认workspace数量为0个

若Tiling函数没有对输出赋值，那么输出将使用上述默认值。

由于Tiling逻辑较为复杂，我们仅举例在Tiling中操作数据结构的方法：

```c++
struct TransDataMode1010Param {
  int64_t tiling_mode;
  int64_t ub_offset;
  int64_t used_core_cnt;
  int64_t core_step_in;
  int64_t core_step_out;
  // 更多TilingData的参数
};

ge::graphStatus TilingForTransData(TilingContext *context) {
  // 获取输入输出的shape，此shape包含原始shape与运行时shape，详细API参考StorageShape定义
  const StorageShape *in_shape = context->GetInputShape(0);
  const StorageShape *out_shape = context->GetOutputShape(0);
  // 获取CompileInfo实例，此实例由TilingParse写入
  auto compile_info = reinterpret_cast<const TransDataCompileInfo *>(context->GetCompileInfo());
  
  // 获取输入输出的TensorDesc，TensorDesc中包含格式信息、DataType信息
  const CompileTimeTensorDesc *src_td = context->GetInputDesc(0);
  const CompileTimeTensorDesc *dst_td = context->GetOutputDesc(0);
  if (in_shape == nullptr || out_shape == nullptr || src_td == nullptr || dst_td == nullptr ||
      compile_info == nullptr) {
    GELOGE(ge::PARAM_INVALID,
           "Failed to tiling for node %s, type %s, nullptr found",
           context->GetNodeName(),
           context->GetNodeType());
    return ge::FAILED;
  }
  ge::Format src_format = src_td->GetStorageFormat();
  ge::Format dst_format = dst_td->GetStorageFormat();

  // 一般来说，Tiling业务逻辑较为复杂，此处仅举例接口的使用方法
  ASSERT_SUCCESS(context->SetBlockDim(32));  // 设置BlockDim为32
  ASSERT_SUCCESS(context->SetTilingKey(10)); // 设置TilingKey为10
  ASSERT_SUCCESS(context->SetNeedAtomic(false)); // 设置不需要做Atomic清零
  
  size_t *workspaces_size = context->GetWorkspaceSizes(2);  // 需要两块workspace
  ASSERT_NOT_NULL(workspaces_size);
  workspaces_size[0] = 1024;  // 第1块workspace大小为1024
  workspaces_size[1] = 2048;  // 第2块workspace大小为2048
  
  // 设置 tiling data
  auto tiling_data = context->GetTilingData<TransDataMode1010Param>();
  if (tiling_data == nullptr) {
    return ge::FAILED;
  }
  tiling_data->tiling_mode = 10;
  tiling_data->ub_offset = 20;
  // ... 更多赋值tiling_data的操作
  return ge::GRAPH_SUCCESS;
}
```

与TilingParse类似，Tiling函数输出所需的内存已经由框架预申请好。

框架会根据`OpDesc`中属性`op_para_size`来申请对应的TilingData内存，对于AtomicClean算子，此属性打在AtomicClean的宿主节点的OpDesc上，属性名为`atomic_op_para_size`，此属性在算子编译时生成，并由对应引擎设置在`OpDesc`上。

当前workspace的最大个数支持16个，后续会根据TaskDef中的描述按需创建workspace size。

### Append方式操作TilingData

TilingData有两种操作方式，上一节中介绍了固定数据结构方式，这种方式的特点是，TilingData的内存排布是确定的，因此可以将TilingData抽象为一个结构体`Foo`，通过`GetTilingData<Foo>()`获取对应的类实例后，Tiling函数直接将数据写到实例中。另一种方式为Append方式，Append方式的特点是，TilingData的内存排布不确定，无法抽象为一个特定的结构体，此时可以将TilingData当做一个二进制码流，通过不断地Append操作，向码流后面追加数据，本节对Append方式的写法做介绍。

```c++
struct Foo {
  int64_t a;
  int32_t b;
  int32_t c;
};

ge::graphStatus ExampleAppendTilingData(TilingContext *context) {
  TilingData *tiling_data = context->GetRawTilingData();  // Append方式写入，获取RawTilingData
  ASSERT_NOT_NULL(tiling_data);
  
  // 向TilingData的尾部添加一个int64_t类型的数据、一个int32_t的数据，值为10，两次Append后，TilingData的长度为8+4=12
  // Append接口通过数据类型计算添加的数据的长度，因此在不确定C++的字面值常量默认类型时， 可以通过static_cast方式明确指定Append的数据类型
  tiling_data->Append(static_cast<int64_t>(10));
  tiling_data->Append(static_cast<int32_t>(10));
  
  // Append方式同样支持结构体，相对于分次Append结构体中的每个成员变量，一次Append整个结构体编码更简单，也会有更好的性能
  tiling_data->Append<Foo>({10, 20, 30});
  Foo foo{100, 200, 300};
  tiling_data->Append(foo);
  
  return ge::GRAPH_SUCCESS;
}
```



> **WARNING**：固定数据结构、Append两种操作方式是互斥的，在一个Tiling函数中，只可以选择其中一种方式写入TilingData。在一次Tiling的过程中，同时调用了context->GetTilingData<Foo>()和context->GetRawTilingData()两种接口基本上意味着产生了bug。
>
> 固定数据结构、Append两种操作方式互斥的原因在于tiling-data-size的计算，在调用context->GetTilingData<Foo>()时，框架会自动将tiling-data-size设置为sizeof(Foo)，而调用context->GetRawTilingData()时，框架认为处于Append模式，不会修改tiling-data-size，而是在每次Append时，做tiling-data-size的累加动作。假想一种场景：先调用GetRawTilingData，并填入一个int64数据，此时tiling-data-size变成8，此时再调用context->GetTilingData<int32_t>()，此时tiling-data-size将被改为sizeof(int32_t)，第一步Append的int64的数据被覆盖

## 获取属性

在InferShape、Tiling时，可以通过context实例获取算子IR属性值，所谓IR属性，是指在IR注册时定义的属性，以`TransData`算子为例：
```c++
REG_OP(TransData)
    .INPUT(src, TensorType::BasicType())
    .OUTPUT(dst, TensorType::BasicType())
    .REQUIRED_ATTR(src_format, String)
    .REQUIRED_ATTR(dst_format, String)
    .ATTR(group, Int, 1)
    .OP_END_FACTORY_REG(TransData)
```

IR定义中声明了`src_format`、`dst_format`、`group`三个属性，可以通过如下方式获取算子属性：

```c++
ge::graphStatus ExampleGetTransDataAttr(TilingContext *context) {
  // 获取所有属性
  const RuntimeAttrs *attrs = context->GetAttrs();
  ASSERT_NOT_NULL(attrs);
  
  // 按照在IR定义中的顺序，使用index获取属性，index从0开始计数
  const char *src_format = attrs->GetAttrPointer<char>(0);  // 获取src_format，src_format是IR中第一个属性，因此index为0
  const char *dst_format = attrs->GetAttrPointer<char>(1);  // 获取dst_format，dst_format是IR中第二个属性，因此index为1
  const int64_t group = attrs->GetAttrPointer<int64_t>(2);  // 获取group，group是IR中第三个属性，因此index为2
  
  return ge::GRAPH_SUCCESS;
}
```

### 私有属性

部分算子在InferShape或Tiling时，可能需要获取一些不在IR中定义的、额外的属性值。之所以需要获取这类属性值，可能出于两种目的：

* IR定义不完整，IR定义无法完整描述本算子的计算逻辑，因此需要额外的属性描述
* 避免重复计算，有一些计算较为耗时，而且不会变化，因此可以在编译时一次计算完

与IR属性这类公开定义的属性相对应，我们称这一类属性为**私有属性**。按照当前的lowering规则，仅IR属性会被保留下来，因此私有属性会被丢弃。如果希望在执行时也可以拿到私有属性，通过如下方式，在IMPL_OP时声明：

```c++
// 为Foo算子注册两个私有属性，分别为`Bar1`和`Bar2`
IMPL_OP(Foo)
    .PrivateAttr("Bar1")
    .PrivateAttr("Bar2")
```

私有属性的获取接口与IR属性一致，其中index参数顺着IR属性的结束index顺延。假设例子中`Foo`的原型如下所示：

```c++
REG_OP(Foo)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(ir_attr1, String)
    .REQUIRED_ATTR(ir_attr2, String)
    .ATTR(ir_attr3, Int, 1)
    .OP_END_FACTORY_REG(Foo)
```

从Foo的原型上看，该算子有3个原型属性，因此其IR属性的index范围是[0,2]，那么`Foo`算子的两个私有属性index范围是[3,4]，假设两个私有属性的数据类型均为`int64_t`，那么在InferShape时可以通过如下方式获取：

```C++
ge::graphStatus InferShapeForFoo(InferShapeContext *context) {
  // 获取所有属性
  const RuntimeAttrs *attrs = context->GetAttrs();
  ASSERT_NOT_NULL(attrs);

  const int64_t attr1 = attrs->GetInt(3);  // 获取Bar1属性，Bar1是继3个IR属性后的第一个私有属性，因此index为3
  const int64_t attr2 = attrs->GetInt(4);  // 获取Bar2属性，Bar2是继3个IR属性后的第二个私有属性，因此index为4
  
  // 一些其他处理。。。

  return ge::GRAPH_SUCCESS;
}
```

> **WARNING:** 
>
> 1. 虽然IMPL_OP本身允许同一个算子的不同实现被定义在不同的文件中，但是私有属性的声明必须在同一个文件中声明完成，否则我们无法确认私有属性的index
> 2. 私有属性一旦注册，那么必须在Node上可以获取成功，否则lowering报错，程序退出。为了保证私有属性的index是正确的的，框架不可以对无法获得的私有属性做跳过处理。

## 可选/动态输入

Runtime2.0中，通过index而不是字符串name来索引输入输出，这种改动是为了减少执行时的字符串匹配，提升执行效率。由index匹配带来的一个问题是，对于带有OPTIONAL、DYNAMIC类型输入的算子，可能出现实例化[^1]后，单纯通过index无法索引到具体输入的问题，以`DynamicRNNV3`算子为例：

```c++
REG_OP(DynamicRNNV3)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(w, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(b, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(seq_length, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(init_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(init_c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wci, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wcf, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wco, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(mask, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(real_mask, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(project, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
		// 省略一些OUTPUT定义
    .ATTR(cell_type, String, "LSTM")
		// 省略一些属性定义
    .OP_END_FACTORY_REG(DynamicRNNV3)
```

由于`DynamicRNNV3`算子有连续的多个optional输入，这导致`init_h`及其后面的输入的实例化后index都是不确定的，对于这种类型的算子，可以通过IR定义的Index获取对应的Shape等数据，以InferShape为例：

```c++
ge::graphStatus InferShapeForDynamicRNNV3(InferShapeContext *context) {
  // 对于前两个输入，不受到optioanl或dynamic的影响，可以按照常规方法获取输入shape
  auto x_shape = context->GetInputShape(0);
  auto w_shape = context->GetInputShape(1);
  if (x_shape == nullptr || w_shape == nullptr) {
    return ge::GRAPH_FAILED;
  }

  int64_t state_size = 0;
  // 在IR原型上，project是第11个输入(从0开始计数)
  constexpr int64_t kProjectInputIndex = 11;

  // 受到前面optional输入影响的，project实例化后输入的index是不确定的，通过`GetOptionalInputShape`来获取对应的输入shape，
  // `GetOptionalInputShape`的入参为`ir_index`，即原型上对应的index
  auto project_shape = context->GetOptionalInputShape(kProjectInputIndex);
  if (project_shape != nullptr) {
    if (project_shape->GetDimNum() < 2) {
      return ge::GRAPH_FAILED;
    }
    state_size = project_shape->GetDim(1);
  }
  // 更多的infershape逻辑。。。
  return ge::GRAPH_SUCCESS;
}
```

对于dynamic类型的输入，实例化后的输入可能是一到多个，对于此类输入，获取方式为：

```c++
// ir_index：此输入在IR定义中的index，从0开始计数
// relative_index：该输入实例化后的相对index，从0开始计数，例如某个DYNAMIC_INPUT实例化了3个，要取第二个，那么relatvie_index = 1
auto shape = context->GetDynamicInputShape(ir_index, relative_index);
```

对于输入较多的算子，执行时通过index获取输入的情况会显得不够友好，最好能够在IR注册时，自动生成对应的宏或常量，例如project输入的index被预定义为`op::DynamicRNNV3::kIrInputProject`，此特性后续完成，敬请期待。

本节举例的获取optional、dynamic输入的方式，在InferShape、Tiling函数中均可以调用。

## 数据依赖

一般来说，具备输入Shape后，算子可以通过InferShape推导出输出Shape。然而部分算子在InferShape时，需要依赖某个输入的具体值才可以进行，这类算子被称为“数据依赖算子”，对应的输入被称为“数据依赖输入”。以`Reshape` 算子为例，其原型为：

```c++
REG_OP(Reshape)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
        DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
        DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Reshape)
```

`Reshape`算子依据`shape`输入的描述，对`x`输入的shape做调整（关于Reshape算子的功能本文不做赘述），因此`Reshape`算子依赖`shape`输入的值，以Reshape算子为例，数据依赖算子在注册时，需要声明其数据依赖输入：

```c++
IMPL_OP(Reshape)
    .InferShape(InferShapeForReshape)
    .InputsDataDependency({1});        // 声明 `ReShape` 算子的第1个IR输入为数据依赖输入，该index从零开始计算
```

根据第1个输入（shape输入）的值，Reshape算子将第0个输入（x输入）的shape做变换，并输出到其第0个输出（y输出）上。Reshape的InferShape实现为：

```
ge::graphStatus InferShapeForReshape(InferShapeContext *context) {
  const gert::Shape *x_shape = context->GetInputShape(0);        // 获取第0个输入的shape
  const gert::Tensor *shape_tensor = context->GetInputTensor(1); // 获取第1个输入的tensor
  gert::Shape *output_shape = context->GetOutputShape(0);
  if (x_shape == nullptr || shape_tensor == nullptr || output_shape == nullptr) {
    // 防御式编程，不应该出现的场景，打印错误并返回失败
    return ge::GRAPH_FAILED;
  }

  auto reshape_size = static_cast<int32_t>(shape_tensor->GetShapeSize());
  if (reshape_size < 1) {
   // 防御式编程，不应该出现的场景，打印错误并返回失败
    return ge::GRAPH_FAILED;
  }

  // 根据原型信息，Reshape的shape输入支持INT32与INT64两类，根据不同的类型进入对应的模板函数中做真正的shape变换操作
  if (shape_tensor->GetDataType() == ge::DT_INT32) {
    int32_t *reshape_data = shape_tensor->GetData<int32_t>();
    return ReshapeInferShapeImpl<int32_t>(reshape_data, *x_shape, *output_shape, reshape_size);
  } else {
    int64_t *reshape_data = shape_tensor->GetData<int64_t>();
    return ReshapeInferShapeImpl<int64_t>(reshape_data, *x_shape, *output_shape, reshape_size);
  }
}
```

获取到reshape的输入数据后，最终调用核心的`ReshapeInferShapeImpl`逻辑，完成reshape的InferShape逻辑。`ReshapeInferShapeImpl`具体的业务强相关，此处不做更多赘述。

在编译时，总是可以调用GetInputTensor获取一个输入的Tensor，但是在其数据依赖边对应输入不是const时，tensor->GetData返回结果为空指针。

> **WARNING**：为了执行效率考虑，runtime2.0的基础API不做RTTI(Runtime Type Identification)，这意味着调用者需要自行保证获取数据类型的正确性。例如：
>
> * 只有声明过数据依赖的输入，才可以在执行时InferShape时获取其对应的Tensor数据。若对一个未声明数据依赖的输入获取Tensor数据，那么行为是未定义的。
> * 从tensor中获取tensor_data时(GetData<int32_t>或GetData<int64_t>)，使用者需要保证获取的数据类型是正确的的，否则行为是未定义的。
>
> 为了定位方便，runtime2.0未来会在O0编译选项时、或通过额外的编译选项使能RTTI，以提供debug时的类型检查能力。打开RTTI后，类型不匹配的Get操作会导致返回空指针或抛出异常。

[^1]: 实例化：IR定义的算子像一个类定义，其定义了算子的原型。基于原型创建计算图上对应节点的过程，我们称为算子实例化。计算图上的结点，我们称为“算子的实例”，算子实例上的输入输出，我们称为”实例化后的输入输出“
