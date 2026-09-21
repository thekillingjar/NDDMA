# aclgrphCalibration<a name="ZH-CN_TOPIC_0000001312725957"></a>

## 产品支持情况<a name="section197451857688"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p48327011813"><a name="p48327011813"></a><a name="p48327011813"></a><span id="ph583230201815"><a name="ph583230201815"></a><a name="ph583230201815"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1770312587308"><a name="p1770312587308"></a><a name="p1770312587308"></a>√</p>
</td>
</tr>
<tr id="row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><span id="ph1483216010188"><a name="ph1483216010188"></a><a name="ph1483216010188"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1970545813305"><a name="p1970545813305"></a><a name="p1970545813305"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 头文件/库文件<a name="section710017525388"></a>

-   头文件：\#include <amct/acl\_graph\_calibration.h\>
-   库文件：libamctacl.so

## 功能说明<a name="section61382201"></a>

将非量化Graph自动修改为量化后的Graph。

## 函数原型<a name="section15568903"></a>

```
graphStatus aclgrphCalibration(ge::Graph &graph, const std::map<ge::AscendString, ge::AscendString> &quantizeConfigs)
```

## 参数说明<a name="section5902405"></a>

<a name="table19987085"></a>
<table><thead align="left"><tr id="row32738149"><th class="cellrowborder" valign="top" width="12.25%" id="mcps1.1.4.1.1"><p id="p34544438"><a name="p34544438"></a><a name="p34544438"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.64%" id="mcps1.1.4.1.2"><p id="p46636132"><a name="p46636132"></a><a name="p46636132"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="80.11%" id="mcps1.1.4.1.3"><p id="p19430358"><a name="p19430358"></a><a name="p19430358"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row30355189"><td class="cellrowborder" valign="top" width="12.25%" headers="mcps1.1.4.1.1 "><p id="p42851240"><a name="p42851240"></a><a name="p42851240"></a>graph</p>
</td>
<td class="cellrowborder" valign="top" width="7.64%" headers="mcps1.1.4.1.2 "><p id="p48398424"><a name="p48398424"></a><a name="p48398424"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.11%" headers="mcps1.1.4.1.3 "><p id="p29997509121"><a name="p29997509121"></a><a name="p29997509121"></a>待修改的用户原始Graph。</p>
</td>
</tr>
<tr id="row50297679"><td class="cellrowborder" valign="top" width="12.25%" headers="mcps1.1.4.1.1 "><p id="p47580213"><a name="p47580213"></a><a name="p47580213"></a>quantizeConfigs</p>
</td>
<td class="cellrowborder" valign="top" width="7.64%" headers="mcps1.1.4.1.2 "><p id="p28792051"><a name="p28792051"></a><a name="p28792051"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.11%" headers="mcps1.1.4.1.3 "><p id="p146821413175313"><a name="p146821413175313"></a><a name="p146821413175313"></a>map表，key为参数类型，value为参数值，均为字符串格式，描述执行接口所需要的配置选项。具体配置参数如下所示:</p>
<a name="ul191832111128"></a><a name="ul191832111128"></a><ul id="ul191832111128"><li>INPUT_DATA_DIR：必填，用于计算量化因子的数据bin文件路径，建议传入不少于一个batch的数据。如果模型有多个输入，则输入数据文件以英文逗号分隔。</li><li>INPUT_SHAPE：必填，输入数据的shape。例如："input_name1:n1,c1,h1,w1;input_name2:n2,c2,h2,w2"。指定的节点必须放在双引号中，节点中间使用英文分号分隔。input_name必须是Graph中的节点名称。</li><li>SOC_VERSION：必填，进行训练后量化校准推理时，所使用的芯片名称。<p id="p125911356174714"><a name="p125911356174714"></a><a name="p125911356174714"></a><span id="ph98439118480"><a name="ph98439118480"></a><a name="ph98439118480"></a><em id="zh-cn_topic_0000001312483385_i79331727136"><a name="zh-cn_topic_0000001312483385_i79331727136"></a><a name="zh-cn_topic_0000001312483385_i79331727136"></a>&lt;soc_version&gt;</em></span>查询方法为：</p>
<a name="ul38588187503"></a><a name="ul38588187503"></a><ul id="ul38588187503"><li>针对如下<span id="zh-cn_topic_0000001265392790_ph20833205312295"><a name="zh-cn_topic_0000001265392790_ph20833205312295"></a><a name="zh-cn_topic_0000001265392790_ph20833205312295"></a>产品</span>：在安装<span id="zh-cn_topic_0000001265392790_ph196874123168"><a name="zh-cn_topic_0000001265392790_ph196874123168"></a><a name="zh-cn_topic_0000001265392790_ph196874123168"></a>昇腾AI处理器</span>的服务器执行<strong id="zh-cn_topic_0000001265392790_b17687612191618"><a name="zh-cn_topic_0000001265392790_b17687612191618"></a><a name="zh-cn_topic_0000001265392790_b17687612191618"></a>npu-smi info</strong>命令进行查询，获取<strong id="zh-cn_topic_0000001265392790_b10161437131915"><a name="zh-cn_topic_0000001265392790_b10161437131915"></a><a name="zh-cn_topic_0000001265392790_b10161437131915"></a>Name</strong>信息。实际配置值为AscendName，例如<strong id="zh-cn_topic_0000001265392790_b16284944181920"><a name="zh-cn_topic_0000001265392790_b16284944181920"></a><a name="zh-cn_topic_0000001265392790_b16284944181920"></a>Name</strong>取值为<em id="zh-cn_topic_0000001265392790_i1478775919179"><a name="zh-cn_topic_0000001265392790_i1478775919179"></a><a name="zh-cn_topic_0000001265392790_i1478775919179"></a>xxxyy</em>，实际配置值为Ascend<em id="zh-cn_topic_0000001265392790_i1678775901719"><a name="zh-cn_topic_0000001265392790_i1678775901719"></a><a name="zh-cn_topic_0000001265392790_i1678775901719"></a>xxxyy</em>。<p id="zh-cn_topic_0000001265392790_p3529538154519"><a name="zh-cn_topic_0000001265392790_p3529538154519"></a><a name="zh-cn_topic_0000001265392790_p3529538154519"></a><span id="zh-cn_topic_0000001265392790_ph1483216010188"><a name="zh-cn_topic_0000001265392790_ph1483216010188"></a><a name="zh-cn_topic_0000001265392790_ph1483216010188"></a><term id="zh-cn_topic_0000001265392790_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001265392790_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001265392790_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001265392790_zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001265392790_zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001265392790_zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</li><li>针对如下<span id="zh-cn_topic_0000001265392790_ph13446185695413"><a name="zh-cn_topic_0000001265392790_ph13446185695413"></a><a name="zh-cn_topic_0000001265392790_ph13446185695413"></a>产品</span>，在安装<span id="zh-cn_topic_0000001265392790_ph17911124171120"><a name="zh-cn_topic_0000001265392790_ph17911124171120"></a><a name="zh-cn_topic_0000001265392790_ph17911124171120"></a>昇腾AI处理器</span>的服务器执行<strong id="zh-cn_topic_0000001265392790_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_b206066255591"><a name="zh-cn_topic_0000001265392790_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_b206066255591"></a><a name="zh-cn_topic_0000001265392790_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_b206066255591"></a>npu-smi info -t board -i </strong><em id="zh-cn_topic_0000001265392790_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_i16609202515915"><a name="zh-cn_topic_0000001265392790_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_i16609202515915"></a><a name="zh-cn_topic_0000001265392790_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_i16609202515915"></a>id</em><strong id="zh-cn_topic_0000001265392790_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_b14358631175910"><a name="zh-cn_topic_0000001265392790_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_b14358631175910"></a><a name="zh-cn_topic_0000001265392790_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_b14358631175910"></a> -c </strong><em id="zh-cn_topic_0000001265392790_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_i16269732165915"><a name="zh-cn_topic_0000001265392790_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_i16269732165915"></a><a name="zh-cn_topic_0000001265392790_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_i16269732165915"></a>chip_id</em>命令进行查询，获取<strong id="zh-cn_topic_0000001265392790_b11257114917192"><a name="zh-cn_topic_0000001265392790_b11257114917192"></a><a name="zh-cn_topic_0000001265392790_b11257114917192"></a>Chip Name</strong>和<strong id="zh-cn_topic_0000001265392790_b72671651121916"><a name="zh-cn_topic_0000001265392790_b72671651121916"></a><a name="zh-cn_topic_0000001265392790_b72671651121916"></a>NPU Name</strong>信息，实际配置值为Chip Name_NPU Name。例如<strong id="zh-cn_topic_0000001265392790_b13136111611203"><a name="zh-cn_topic_0000001265392790_b13136111611203"></a><a name="zh-cn_topic_0000001265392790_b13136111611203"></a>Chip Name</strong>取值为Ascend<em id="zh-cn_topic_0000001265392790_i68701996189"><a name="zh-cn_topic_0000001265392790_i68701996189"></a><a name="zh-cn_topic_0000001265392790_i68701996189"></a>xxx</em>，<strong id="zh-cn_topic_0000001265392790_b51347352112"><a name="zh-cn_topic_0000001265392790_b51347352112"></a><a name="zh-cn_topic_0000001265392790_b51347352112"></a>NPU Name</strong>取值为1234，实际配置值为Ascend<em id="zh-cn_topic_0000001265392790_i82901912141813"><a name="zh-cn_topic_0000001265392790_i82901912141813"></a><a name="zh-cn_topic_0000001265392790_i82901912141813"></a>xxx</em><em id="zh-cn_topic_0000001265392790_i154501458102213"><a name="zh-cn_topic_0000001265392790_i154501458102213"></a><a name="zh-cn_topic_0000001265392790_i154501458102213"></a>_</em>1234。其中：<a name="zh-cn_topic_0000001265392790_ul2747601334"></a><a name="zh-cn_topic_0000001265392790_ul2747601334"></a><ul id="zh-cn_topic_0000001265392790_ul2747601334"><li>id：设备id，通过<strong id="zh-cn_topic_0000001265392790_b83171930133314"><a name="zh-cn_topic_0000001265392790_b83171930133314"></a><a name="zh-cn_topic_0000001265392790_b83171930133314"></a>npu-smi info -l</strong>命令查出的NPU ID即为设备id。</li><li>chip_id：芯片id，通过<strong id="zh-cn_topic_0000001265392790_b18888204343317"><a name="zh-cn_topic_0000001265392790_b18888204343317"></a><a name="zh-cn_topic_0000001265392790_b18888204343317"></a>npu-smi info -m</strong>命令查出的Chip ID即为芯片id。</li></ul>
<p id="zh-cn_topic_0000001265392790_p12136131554410"><a name="zh-cn_topic_0000001265392790_p12136131554410"></a><a name="zh-cn_topic_0000001265392790_p12136131554410"></a><span id="zh-cn_topic_0000001265392790_ph13754548217"><a name="zh-cn_topic_0000001265392790_ph13754548217"></a><a name="zh-cn_topic_0000001265392790_ph13754548217"></a><term id="zh-cn_topic_0000001265392790_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001265392790_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001265392790_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001265392790_zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001265392790_zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001265392790_zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</li></ul>
</li><li>INPUT_FORMAT：选填，输入数据的排布格式，支持"NCHW"、"NHWC"、"ND"。</li><li>INPUT_FP16_NODES：选填，指定输入数据类型为FP16的输入节点名称。</li><li>CONFIG_FILE：选填，用于配置高级选项的配置文件路径。</li><li>LOG_LEVEL：选填，设置训练后量化时的日志等级，该参数只控制训练后量化过程中显示的日志级别，默认显示info级别：<a name="ul1976183491120"></a><a name="ul1976183491120"></a><ul id="ul1976183491120"><li>debug：输出debug/info/warning/error/event级别的日志信息。</li><li>info：输出info/warning/error/event级别的日志信息。</li><li>warning：输出warning/error/event级别的日志信息。</li><li>error：输出error/event级别的日志信息。</li></ul>
<div class="p" id="p20171194115274"><a name="p20171194115274"></a><a name="p20171194115274"></a>此外，训练后量化过程中的日志落盘信息由<strong id="b138391955163119"><a name="b138391955163119"></a><a name="b138391955163119"></a>AMCT_LOG_DUMP</strong>环境变量进行控制：<a name="ul1530013259251"></a><a name="ul1530013259251"></a><ul id="ul1530013259251"><li><strong id="b41411750162513"><a name="b41411750162513"></a><a name="b41411750162513"></a>export AMCT_LOG_DUMP=1</strong>：日志落盘到当前路径的<span class="filepath" id="filepath15761201472014"><a name="filepath15761201472014"></a><a name="filepath15761201472014"></a>“amct_log_{timestamp}/amct_acl.log”</span>文件中，<strong id="b48891224200"><a name="b48891224200"></a><a name="b48891224200"></a>不保存</strong>量化因子record文件和graph文件。</li><li><strong id="b15687732182619"><a name="b15687732182619"></a><a name="b15687732182619"></a>export AMCT_LOG_DUMP=2</strong>：日志落盘到当前路径的<span class="filepath" id="filepath455853591813"><a name="filepath455853591813"></a><a name="filepath455853591813"></a>“amct_log_{timestamp}/amct_acl.log”</span>文件中，同时在<span class="filepath" id="filepath106789215197"><a name="filepath106789215197"></a><a name="filepath106789215197"></a>“amct_log_{timestamp}”</span>目录下保存量化因子record文件<em id="i5239113105412"><a name="i5239113105412"></a><a name="i5239113105412"></a>。</em></li><li><strong id="b575182352714"><a name="b575182352714"></a><a name="b575182352714"></a>export AMCT_LOG_DUMP=3</strong>：日志落盘到当前路径的<span class="filepath" id="filepath147461850151918"><a name="filepath147461850151918"></a><a name="filepath147461850151918"></a>“amct_log_{timestamp}/amct_acl.log”</span>文件中，同时在<span class="filepath" id="filepath8288132482017"><a name="filepath8288132482017"></a><a name="filepath8288132482017"></a>“amct_log_{timestamp}”</span>目录下保存量化因子record文件和包含量化过程中各阶段图描述信息的graph文件<em id="i8880123653216"><a name="i8880123653216"></a><a name="i8880123653216"></a>。</em></li></ul>
</div>
<p id="p733554915616"><a name="p733554915616"></a><a name="p733554915616"></a>为防止日志文件、record文件、graph文件持续落盘导致磁盘被写满，请及时清理这些文件。</p>
<p id="p103351491561"><a name="p103351491561"></a><a name="p103351491561"></a>如果用户配置了<strong id="b1833574925616"><a name="b1833574925616"></a><a name="b1833574925616"></a>ASCEND_WORK_PATH</strong>环境变量，则上述日志、量化因子record文件和graph文件存储到该环境变量指定的路径下，例如ASCEND_WORK_PATH=/home/test，则存储路径为：/home/test/amct_acl/amct_log_<em id="i933574919567"><a name="i933574919567"></a><a name="i933574919567"></a>{pid}</em>_<em id="i7335649145615"><a name="i7335649145615"></a><a name="i7335649145615"></a>时间戳</em>。其中，amct_acl模型转换过程中会自动创建，<em id="i143351349125610"><a name="i143351349125610"></a><a name="i143351349125610"></a>{pid}</em>为进程号。</p>
<div class="note" id="note132641528162618"><a name="note132641528162618"></a><a name="note132641528162618"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1827233152617"><a name="p1827233152617"></a><a name="p1827233152617"></a>上述日志文件、record文件、graph文件重新执行量化时会被覆盖，请用户自行进行保存。此外，由于生成的日志文件大小和所要量化模型层数有关，请用户确保服务器有足够空间：</p>
<p id="p15827733122618"><a name="p15827733122618"></a><a name="p15827733122618"></a>以量化resnet101模型为例，日志级别设置为INFO，日志文件大小为12KB左右，中间临时文件大小为260MB左右；日志级别设置为DEBUG，日志文件大小为390KB左右，中间临时文件大小为430MB左右。</p>
</div></div>
</li><li>OUT_NODES：选填，用户Graph的输出节点名。</li><li>DEVICE_ID：选填，指定设备ID，默认为0。<a name="ul208514172278"></a><a name="ul208514172278"></a><ul id="ul208514172278"><li><span id="ph109861212587"><a name="ph109861212587"></a><a name="ph109861212587"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_1"><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_1"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811_1"><a name="zh-cn_topic_0000001312391781_term184716139811_1"></a><a name="zh-cn_topic_0000001312391781_term184716139811_1"></a>Atlas A2 推理系列产品</term></span>，支持该选项。</li><li><span id="ph19489154814101"><a name="ph19489154814101"></a><a name="ph19489154814101"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_1"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115_1"><a name="zh-cn_topic_0000001312391781_term131434243115_1"></a><a name="zh-cn_topic_0000001312391781_term131434243115_1"></a>Atlas A3 推理系列产品</term></span>，支持该选项。</li></ul>
</li><li>infer_aicore_num：选填，进行训练后量化校准推理时，使用的AI Core核数，查询方法请参见<a href="aclgrphBuildInitialize支持的配置参数.md">aclgrphBuildInitialize支持的配置参数</a> &gt; AICORE_NUM</li><li>IP_ADDR：指定NCS所在服务器的IP地址。<a name="ul115254351271"></a><a name="ul115254351271"></a><ul id="ul115254351271"><li><span id="ph206688291413"><a name="ph206688291413"></a><a name="ph206688291413"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_2"><a name="zh-cn_topic_0000001312391781_term11962195213215_2"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_2"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811_2"><a name="zh-cn_topic_0000001312391781_term184716139811_2"></a><a name="zh-cn_topic_0000001312391781_term184716139811_2"></a>Atlas A2 推理系列产品</term></span>，不支持该选项。</li><li><span id="ph1687615551112"><a name="ph1687615551112"></a><a name="ph1687615551112"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_2"><a name="zh-cn_topic_0000001312391781_term1253731311225_2"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_2"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115_2"><a name="zh-cn_topic_0000001312391781_term131434243115_2"></a><a name="zh-cn_topic_0000001312391781_term131434243115_2"></a>Atlas A3 推理系列产品</term></span>，不支持该选项。</li></ul>
</li><li>PORT：指定NCS所在服务器端口。<a name="ul1832102917"></a><a name="ul1832102917"></a><ul id="ul1832102917"><li><span id="ph3170192421810"><a name="ph3170192421810"></a><a name="ph3170192421810"></a><term id="zh-cn_topic_0000001312391781_term11962195213215_3"><a name="zh-cn_topic_0000001312391781_term11962195213215_3"></a><a name="zh-cn_topic_0000001312391781_term11962195213215_3"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811_3"><a name="zh-cn_topic_0000001312391781_term184716139811_3"></a><a name="zh-cn_topic_0000001312391781_term184716139811_3"></a>Atlas A2 推理系列产品</term></span>，不支持该选项。</li><li><span id="ph151847153126"><a name="ph151847153126"></a><a name="ph151847153126"></a><term id="zh-cn_topic_0000001312391781_term1253731311225_3"><a name="zh-cn_topic_0000001312391781_term1253731311225_3"></a><a name="zh-cn_topic_0000001312391781_term1253731311225_3"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115_3"><a name="zh-cn_topic_0000001312391781_term131434243115_3"></a><a name="zh-cn_topic_0000001312391781_term131434243115_3"></a>Atlas A3 推理系列产品</term></span>，不支持该选项。</li></ul>
</li><li>INSERT_OP_CONF：插入算子的配置文件路径与文件名，例如Aipp预处理算子。<p id="zh-cn_topic_0000001265073918_p14219233615"><a name="zh-cn_topic_0000001265073918_p14219233615"></a><a name="zh-cn_topic_0000001265073918_p14219233615"></a>配置文件路径和文件名：支持大小写字母（a-z，A-Z）、数字（0-9）、下划线（_）、中划线（-）、句点（.）、中文字符；文件后缀不局限于.cfg格式，但是配置文件中的内容需要满足prototxt格式。</p>
<p id="p55634143114"><a name="p55634143114"></a><a name="p55634143114"></a>配置文件的内容示例如下：</p>
<pre class="screen" id="screen14556028182514"><a name="screen14556028182514"></a><a name="screen14556028182514"></a>aipp_op {
   aipp_mode:static
   input_format:YUV420SP_U8
   csc_switch:true
   var_reci_chn_0:0.00392157
   var_reci_chn_1:0.00392157
   var_reci_chn_2:0.00392157
}</pre>
<p id="p1039512691010"><a name="p1039512691010"></a><a name="p1039512691010"></a>关于配置文件详细配置以及参数说明，请参见<span id="ph1739546201010"><a name="ph1739546201010"></a><a name="ph1739546201010"></a>《ATC离线模型编译工具用户指南》</span>。</p>
</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section53121653"></a>

<a name="table60309786"></a>
<table><thead align="left"><tr id="row33911763"><th class="cellrowborder" valign="top" width="12.57%" id="mcps1.1.4.1.1"><p id="p62498284"><a name="p62498284"></a><a name="p62498284"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.12%" id="mcps1.1.4.1.2"><p id="p29196278"><a name="p29196278"></a><a name="p29196278"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="77.31%" id="mcps1.1.4.1.3"><p id="p16088345"><a name="p16088345"></a><a name="p16088345"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row28087548"><td class="cellrowborder" valign="top" width="12.57%" headers="mcps1.1.4.1.1 "><p id="p60498887"><a name="p60498887"></a><a name="p60498887"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="10.12%" headers="mcps1.1.4.1.2 "><p id="p1462819"><a name="p1462819"></a><a name="p1462819"></a>graphStatus</p>
</td>
<td class="cellrowborder" valign="top" width="77.31%" headers="mcps1.1.4.1.3 "><p id="p285891713014"><a name="p285891713014"></a><a name="p285891713014"></a>GRAPH_SUCCESS(0)：成功。</p>
<p id="p990892"><a name="p990892"></a><a name="p990892"></a>其他：失败。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section8332830"></a>

-   对于已经被插入量化算子的Graph不支持进行量化。
-   使用配置文件中的**calibration**训练后量化功能时，只支持**带NPU设备**的安装场景，详细介绍请参见《CANN 软件安装指南》手册搭建对应产品环境。

