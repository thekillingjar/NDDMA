# aclopLoad<a name="ZH-CN_TOPIC_0000001265081858"></a>

## 产品支持情况<a name="section8178181118225"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p48327011813"><a name="p48327011813"></a><a name="p48327011813"></a><span id="ph583230201815"><a name="ph583230201815"></a><a name="ph583230201815"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p7948163910184"><a name="p7948163910184"></a><a name="p7948163910184"></a>√</p>
</td>
</tr>
<tr id="row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><span id="ph1483216010188"><a name="ph1483216010188"></a><a name="ph1483216010188"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p19948143911820"><a name="p19948143911820"></a><a name="p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section73552454261"></a>

从内存中加载单算子模型数据（单算子模型数据是指“单算子编译成\*.om文件后，再将om文件读取到内存中”的数据），由用户管理内存。

## 函数原型<a name="section195491952142612"></a>

```
aclError aclopLoad(const void *model,  size_t modelSize)
```

## 参数说明<a name="section155499553266"></a>

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p0124125211120"><a name="p0124125211120"></a><a name="p0124125211120"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1412319523115"><a name="p1412319523115"></a><a name="p1412319523115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1712295213116"><a name="p1712295213116"></a><a name="p1712295213116"></a>单算子模型数据的内存地址指针。</p>
</td>
</tr>
<tr id="row7909131293411"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1912212521619"><a name="p1912212521619"></a><a name="p1912212521619"></a>modelSize</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p171219521613"><a name="p171219521613"></a><a name="p171219521613"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p212018526117"><a name="p212018526117"></a><a name="p212018526117"></a>内存中的模型数据长度，单位Byte。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1435713587268"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

## 约束说明<a name="section69698505514"></a>

-   动态Shape算子场景、昇腾虚拟化实例场景下，模型文件加载环境中的算子库版本需与模型文件编译环境的版本一致，否则在加载算子时会报错。

    可通过$\{INSTALL\_DIR\}/opp/version.info文件中的version字段查看算子库版本。

    $\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。以root用户安装为例，则安装后文件存储路径为：/usr/local/Ascend/cann。

-   在加载前，请先根据单算子om文件的大小评估内存空间是否足够，内存空间不足，会导致应用程序异常。

    <a name="zh-cn_topic_0000001312400585_table18590323205417"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001312400585_row55902023135416"><th class="cellrowborder" valign="top" width="30.459999999999997%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001312400585_p185901023155419"><a name="zh-cn_topic_0000001312400585_p185901023155419"></a><a name="zh-cn_topic_0000001312400585_p185901023155419"></a>型号</p>
    </th>
    <th class="cellrowborder" valign="top" width="69.54%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001312400585_p65901223195413"><a name="zh-cn_topic_0000001312400585_p65901223195413"></a><a name="zh-cn_topic_0000001312400585_p65901223195413"></a>一个进程内正在执行的算子的最大个数上限</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001312400585_row1559182313540"><td class="cellrowborder" valign="top" width="30.459999999999997%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001312400585_p0591162319548"><a name="zh-cn_topic_0000001312400585_p0591162319548"></a><a name="zh-cn_topic_0000001312400585_p0591162319548"></a><span id="zh-cn_topic_0000001312400585_ph159132305415"><a name="zh-cn_topic_0000001312400585_ph159132305415"></a><a name="zh-cn_topic_0000001312400585_ph159132305415"></a><term id="zh-cn_topic_0000001312400585_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312400585_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312400585_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312400585_zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001312400585_zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001312400585_zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
    <p id="zh-cn_topic_0000001312400585_p19591123115414"><a name="zh-cn_topic_0000001312400585_p19591123115414"></a><a name="zh-cn_topic_0000001312400585_p19591123115414"></a><span id="zh-cn_topic_0000001312400585_ph6591923125419"><a name="zh-cn_topic_0000001312400585_ph6591923125419"></a><a name="zh-cn_topic_0000001312400585_ph6591923125419"></a><term id="zh-cn_topic_0000001312400585_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312400585_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312400585_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312400585_zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001312400585_zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001312400585_zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="69.54%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001312400585_p17591112315542"><a name="zh-cn_topic_0000001312400585_p17591112315542"></a><a name="zh-cn_topic_0000001312400585_p17591112315542"></a>2000000</p>
    </td>
    </tr>
    </tbody>
    </table>

