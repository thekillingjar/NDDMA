# atc 模型转换实践指南：静态 shape、动态多档位与动态 shape

## 1 引言

本文面向应用开发者，围绕两个核心问题展开：

* **输入尺寸是否会变化**
* **变化是否可以提前枚举**

基于这两个维度，给出在 Ascend 推理场景下使用 **atc** 转换模型的可落地实践方案。文中不区分前端框架，适用于 atc 支持的所有模型格式（如 ONNX、TensorFlow PB、Caffe 等）。

在 Ascend 推理场景中，shape 的选择会直接影响编译器优化程度、运行时调度方式以及最终的性能稳定性。合理选择静态 shape、动态多档位或动态 shape，并结合 atc 的能力特性，是获得稳定吞吐和低时延的关键。

本文默认读者已了解通过 atc 进行模型转换，并基于 **aclmdl** 接口完成模型加载与推理的完整流程。

---

## 2 模型转换与执行的整体流程

在展开具体策略之前，先从整体流程上统一 atc 与模型执行阶段的基本概念。

用户通过 **atc** 命令将模型转换为一个 `.om`（Offline Model）文件，再通过 **aclmdl** 系列接口加载并执行该模型。在 GE（Graph Engine）视角下，这两个阶段分别称为 **编译（compile）** 和 **执行（execute）**。

* **编译阶段**
  GE 读取用户在 atc 中指定的模型文件（如 ONNX 或 PB），对计算图进行分析与优化，生成可在 NPU 上执行的二进制模型文件（`.om`）。

* **执行阶段**
  GE 通过 aclmdl 接口加载 `.om` 文件，将其部署到 NPU 设备上，并执行后续的推理任务。

需要明确的是，GE 采用的是**编译期与执行期职责清晰分离**的模型：

* 编译阶段耗时较长，但通常只需执行一次，用于生成 `.om`；
* 执行阶段不再进行结构性图优化，推理开销较小，`.om` 加载后可被反复执行。

这一特性决定了 **shape 信息在编译期的重要性**。

## 3 静态 shape、动态 shape 与性能特性

### 静态 shape

**静态 shape** 指模型在多次执行过程中，所有 tensor（输入、输出以及中间 tensor）的维度完全固定，任何维度都不允许发生变化。

在该模式下，编译阶段可以进行最充分的优化，并可使能执行期的 **下沉调度**。下沉调度的具体机制可参考官方说明：
[https://www.hiascend.com/developer/techArticles/20240715-1](https://www.hiascend.com/developer/techArticles/20240715-1)

在工程实践中，静态 shape 通常能够获得最佳的推理性能和稳定性。

---

### 动态 shape

**动态 shape** 指模型在多次执行过程中，输入或中间 tensor 的维度可能发生变化。

其优势在于灵活性，但代价也较为明显：

* 编译期可用的优化显著减少；
* 无法使能下沉调度；
* 推理性能和时延稳定性通常较差。

因此，在性能敏感的推理场景中，应尽量避免使用完全动态 shape。

---

### 动态多档位（推荐的折中方案）

考虑到静态 shape 带来的显著性能优势，atc 提供了 **动态多档位** 能力，用于应对 **shape 变化有限且可以枚举** 的场景。

动态多档位的本质是：
在模型转换阶段，**一次性为模型指定多个确定的静态 shape 档位**。运行时根据实际输入选择匹配的档位执行，但每个档位在编译期均按静态 shape 处理。

例如，若模型仅 batch 维度可变，且可能取值为：

* `[1, 3, 224, 224]`
* `[8, 3, 224, 224]`
* `[16, 3, 224, 224]`

则可以将这三种 batch 作为三个档位同时传递给 atc。

在启用动态多档位后：

* 模型在执行层面仍表现为“动态”；
* 编译器对每个档位均可进行静态 shape 优化；
* 推理性能通常可与单一静态 shape 基本持平。

需要注意的是，动态多档位在带来性能收益的同时，也会引入额外代价：

* **模型内存占用以最大档位为准**
  即使当前执行的是最小档位，模型整体内存占用仍等同于最大档位。例如，若 batch 最大档位为 1024，即使执行 batch=1，内存占用仍按 1024 计算。
* **编译时间随档位数量线性增长**
  一般情况下，N 个档位的编译时间约为单一静态 shape 的 N 倍。

## 4 atc 中 shape 相关参数配置方式概览

本章从 **atc 参数配置角度**，说明静态 shape、完全动态 shape 与动态多档位三种策略在 atc 中的具体表达方式。

### 静态 shape 的参数配置方式

在静态 shape 策略下，模型在编译期需要 **完全确定所有输入 tensor 的维度**。atc 转换时，用户需要为每个输入显式指定一个确定的 shape。

例如：

```shell
--input_shape="input_0_0:16,32,208,208;input_1_0:16,64,208,208"
```

上述配置中，所有输入维度在编译期已完全确定，模型以静态 shape 方式进行编译。

涉及的配置项：

* **`--input_shape`**
  [https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/atctool/atlasatcparam_16_0016.html](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/atctool/atlasatcparam_16_0016.html)

---

### 动态 shape 的参数配置方式

动态 shape 指在编译期 **无法完全确定输入或中间 tensor 的维度**，shape 需要在运行期才能确定。

在 atc 中，动态 shape 仍通过 `--input_shape` 进行配置，使用 `-1` 表示对应维度为动态。

例如：

```shell
--input_shape="input_0_0:-1,32,208,208;input_1_0:-1,64,208,208"
```

上述配置表示 batch 维度在编译期不可确定，模型将以动态 shape 方式编译。

**动态 shape 与静态 shape 使用相同的配置项**，差异仅体现在是否存在不可确定的维度。

---

### 动态多档位的参数配置方式

动态多档位用于处理 **shape 变化有限且可以提前枚举** 的场景。

在 atc 中，动态多档位的配置核心并不是“让模型支持运行期动态 shape”，而是：

> **在编译期一次性枚举所有可能出现的确定 shape 档位。**

每个档位在编译阶段均按静态 shape 处理，运行时根据输入 shape 命中对应档位执行。

动态多档位配置通常具备以下特征：

* shape 的所有变化在编译期已枚举完成；
* 编译产物中包含多个静态 shape 的执行路径；
* 未命中任何档位的输入 shape 会导致执行失败。

从语义上看，动态多档位仍属于 **“编译期 shape 确定”** 的范畴，只是确定的 shape 不止一个。

#### 示例：动态 batch 的多档位配置

若模型仅 batch 维度可变，可按如下方式配置：

```shell
$ atc \
  --input_shape="input_0_0:-1,32,208,208;input_1_0:-1,64,208,208" \
  --dynamic_batch_size="1,8,16"
```

其中：

* `--input_shape` 中的 `-1` 表示 batch 维度为动态；
* `--dynamic_batch_size` 枚举了运行期可能出现的所有 batch 档位。

该配置表示模型执行时，仅允许出现以下输入 shape：

* `[1, 3, 224, 224]`
* `[8, 3, 224, 224]`
* `[16, 3, 224, 224]`

#### 动态多档位相关配置项

* **动态 batch（`--dynamic_batch_size`）**
  [https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/atctool/atlasatcparam_16_0018.html](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/atctool/atlasatcparam_16_0018.html)

* **动态 image size（`--dynamic_image_size`）**
  [https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/atctool/atlasatcparam_16_0019.html](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/atctool/atlasatcparam_16_0019.html)

* **任意维度动态（`--dynamic_dims`）**
  [https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/atctool/atlasatcparam_16_0020.html](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/atctool/atlasatcparam_16_0020.html)

其中，`--dynamic_dims` 的配置能力可以覆盖动态 batch 与动态 image size，但配置相对更为复杂。
