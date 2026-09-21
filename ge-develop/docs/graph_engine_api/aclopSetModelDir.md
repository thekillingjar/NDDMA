# aclopSetModelDir<a name="ZH-CN_TOPIC_0000001312400585"></a>

## 产品支持情况<a name="section15254644421"></a>

<a name="zh-cn_topic_0000002219420921_table14931115524110"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002219420921_row1993118556414"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002219420921_p29315553419"><a name="zh-cn_topic_0000002219420921_p29315553419"></a><a name="zh-cn_topic_0000002219420921_p29315553419"></a><span id="zh-cn_topic_0000002219420921_ph59311455164119"><a name="zh-cn_topic_0000002219420921_ph59311455164119"></a><a name="zh-cn_topic_0000002219420921_ph59311455164119"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002219420921_p59313557417"><a name="zh-cn_topic_0000002219420921_p59313557417"></a><a name="zh-cn_topic_0000002219420921_p59313557417"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002219420921_row1693117553411"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002219420921_p1493195513412"><a name="zh-cn_topic_0000002219420921_p1493195513412"></a><a name="zh-cn_topic_0000002219420921_p1493195513412"></a><span id="zh-cn_topic_0000002219420921_ph1093110555418"><a name="zh-cn_topic_0000002219420921_ph1093110555418"></a><a name="zh-cn_topic_0000002219420921_ph1093110555418"></a><term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002219420921_p20931175524111"><a name="zh-cn_topic_0000002219420921_p20931175524111"></a><a name="zh-cn_topic_0000002219420921_p20931175524111"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002219420921_row199312559416"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002219420921_p0931555144119"><a name="zh-cn_topic_0000002219420921_p0931555144119"></a><a name="zh-cn_topic_0000002219420921_p0931555144119"></a><span id="zh-cn_topic_0000002219420921_ph1693115559411"><a name="zh-cn_topic_0000002219420921_ph1693115559411"></a><a name="zh-cn_topic_0000002219420921_ph1693115559411"></a><term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002219420921_p129321955154117"><a name="zh-cn_topic_0000002219420921_p129321955154117"></a><a name="zh-cn_topic_0000002219420921_p129321955154117"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section338171020171"></a>

设置加载模型文件的目录，该模型文件是由单算子编译成的\*.om文件。

## 函数原型<a name="section326531811177"></a>

```
aclError aclopSetModelDir(const char *modelDir)
```

## 参数说明<a name="section1176530151716"></a>

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p0124125211120"><a name="p0124125211120"></a><a name="p0124125211120"></a><strong id="b781210203207"><a name="b781210203207"></a><a name="b781210203207"></a>modelDir</strong></p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1412319523115"><a name="p1412319523115"></a><a name="p1412319523115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1712295213116"><a name="p1712295213116"></a><a name="p1712295213116"></a>模型文件路径的指针。</p>
<p id="p7597171914317"><a name="p7597171914317"></a><a name="p7597171914317"></a>此处可设置多级目录，但系统最多从多级目录的最后一级开始，读取三级目录下的模型文件。</p>
<p id="p45075519443"><a name="p45075519443"></a><a name="p45075519443"></a>例如，将modelDir设置为"dir0/dir1"，dir1下的目录层级为<span class="filepath" id="filepath52121514184820"><a name="filepath52121514184820"></a><a name="filepath52121514184820"></a>“dir2/dir3/dir4”</span>，这时系统只支持读取<span class="filepath" id="filepath1980310199484"><a name="filepath1980310199484"></a><a name="filepath1980310199484"></a>“dir1”</span>、<span class="filepath" id="filepath1111911251487"><a name="filepath1111911251487"></a><a name="filepath1111911251487"></a>“dir1/dir2”</span>、<span class="filepath" id="filepath3175162984815"><a name="filepath3175162984815"></a><a name="filepath3175162984815"></a>“dir1/dir2/dir3”</span>目录下的模型文件。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section181533516175"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

## 约束说明<a name="section9589202317549"></a>

-   动态Shape算子场景、昇腾虚拟化实例场景下，模型文件加载环境中的算子库版本需与模型文件编译环境的版本一致，否则在加载算子时会报错。

    可通过$\{INSTALL\_DIR\}/opp/version.info文件中的version字段查看算子库版本。

    $\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。以root用户安装为例，则安装后文件存储路径为：/usr/local/Ascend/cann。

-   在加载前，请先根据单算子om文件的大小评估内存空间是否足够，内存空间不足，会导致应用程序异常。

    <a name="table18590323205417"></a>
    <table><thead align="left"><tr id="row55902023135416"><th class="cellrowborder" valign="top" width="30.459999999999997%" id="mcps1.1.3.1.1"><p id="p185901023155419"><a name="p185901023155419"></a><a name="p185901023155419"></a>型号</p>
    </th>
    <th class="cellrowborder" valign="top" width="69.54%" id="mcps1.1.3.1.2"><p id="p65901223195413"><a name="p65901223195413"></a><a name="p65901223195413"></a>一个进程内正在执行的算子的最大个数上限</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1559182313540"><td class="cellrowborder" valign="top" width="30.459999999999997%" headers="mcps1.1.3.1.1 "><p id="p0591162319548"><a name="p0591162319548"></a><a name="p0591162319548"></a><span id="ph159132305415"><a name="ph159132305415"></a><a name="ph159132305415"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
    <p id="p19591123115414"><a name="p19591123115414"></a><a name="p19591123115414"></a><span id="ph6591923125419"><a name="ph6591923125419"></a><a name="ph6591923125419"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="69.54%" headers="mcps1.1.3.1.2 "><p id="p17591112315542"><a name="p17591112315542"></a><a name="p17591112315542"></a>2000000</p>
    </td>
    </tr>
    </tbody>
    </table>

