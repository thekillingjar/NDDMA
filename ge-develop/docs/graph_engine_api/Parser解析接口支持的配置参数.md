# Parser解析接口支持的配置参数<a name="ZH-CN_TOPIC_0000001265085954"></a>

**表 1**  Parser解析接口支持的配置参数

<a name="table102241012358"></a>
<table><thead align="left"><tr id="row12247013356"><th class="cellrowborder" valign="top" width="20.76%" id="mcps1.2.3.1.1"><p id="p359016311297"><a name="p359016311297"></a><a name="p359016311297"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="79.24%" id="mcps1.2.3.1.2"><p id="p106113382919"><a name="p106113382919"></a><a name="p106113382919"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1745512525131"><td class="cellrowborder" valign="top" width="20.76%" headers="mcps1.2.3.1.1 "><p id="p460110503285"><a name="p460110503285"></a><a name="p460110503285"></a>INPUT_FP16_NODES</p>
</td>
<td class="cellrowborder" valign="top" width="79.24%" headers="mcps1.2.3.1.2 "><p id="p1660145012819"><a name="p1660145012819"></a><a name="p1660145012819"></a>指定输入数据类型为FP16的输入节点名称。</p>
<p id="p260175018281"><a name="p260175018281"></a><a name="p260175018281"></a>例如："node_name1;node_name2"，指定的节点必须放在双引号中，节点中间使用英文分号分隔。</p>
<p id="p145481755132014"><a name="p145481755132014"></a><a name="p145481755132014"></a>配置示例：</p>
<pre class="screen" id="screen14548255142015"><a name="screen14548255142015"></a><a name="screen14548255142015"></a>{ge::AscendString(ge::ir_option::<strong id="b8549455182019"><a name="b8549455182019"></a><a name="b8549455182019"></a>INPUT_FP16_NODES</strong>), ge::AscendString("input1;input2")},</pre>
</td>
</tr>
<tr id="row18231143131610"><td class="cellrowborder" valign="top" width="20.76%" headers="mcps1.2.3.1.1 "><p id="p146851143142019"><a name="p146851143142019"></a><a name="p146851143142019"></a>IS_INPUT_ADJUST_HW_LAYOUT</p>
</td>
<td class="cellrowborder" valign="top" width="79.24%" headers="mcps1.2.3.1.2 "><p id="p478574510165"><a name="p478574510165"></a><a name="p478574510165"></a>用于指定网络输入数据类型是否为FP16，数据格式是否为NC1HWC0。</p>
<p id="p157851845101615"><a name="p157851845101615"></a><a name="p157851845101615"></a>该参数需要与INPUT_FP16_NODES配合使用。若IS_INPUT_ADJUST_HW_LAYOUT参数设置为true，对应INPUT_FP16_NODES节点的输入数据类型为FP16，输入数据格式为NC1HWC0。</p>
<p id="p678504521613"><a name="p678504521613"></a><a name="p678504521613"></a>取值范围为false或true，默认值为false。</p>
<p id="p12884142719189"><a name="p12884142719189"></a><a name="p12884142719189"></a>配置示例：</p>
<pre class="screen" id="screen1254184711918"><a name="screen1254184711918"></a><a name="screen1254184711918"></a>{ge::AscendString(ge::ir_option::<strong id="b1898835841915"><a name="b1898835841915"></a><a name="b1898835841915"></a>INPUT_FP16_NODES</strong>), ge::AscendString("input1;input2")},
{ge::AscendString(ge::ir_option::<strong id="b11900020162015"><a name="b11900020162015"></a><a name="b11900020162015"></a>IS_INPUT_ADJUST_HW_LAYOUT</strong>), ge::AscendString("true,true")},</pre>
</td>
</tr>
<tr id="row77691937052"><td class="cellrowborder" valign="top" width="20.76%" headers="mcps1.2.3.1.1 "><p id="p20770437159"><a name="p20770437159"></a><a name="p20770437159"></a>OUTPUT</p>
</td>
<td class="cellrowborder" valign="top" width="79.24%" headers="mcps1.2.3.1.2 "><p id="p2077013370520"><a name="p2077013370520"></a><a name="p2077013370520"></a>指定转图后计算图名称。</p>
<p id="p78761242217"><a name="p78761242217"></a><a name="p78761242217"></a>配置示例：</p>
<pre class="screen" id="screen787611418210"><a name="screen787611418210"></a><a name="screen787611418210"></a>{ge::AscendString(ge::ir_option::<strong id="b18359216215"><a name="b18359216215"></a><a name="b18359216215"></a>OUTPUT</strong>), ge::AscendString("newIssue")},</pre>
</td>
</tr>
<tr id="row671216572257"><td class="cellrowborder" valign="top" width="20.76%" headers="mcps1.2.3.1.1 "><p id="p799416432611"><a name="p799416432611"></a><a name="p799416432611"></a>IS_OUTPUT_ADJUST_HW_LAYOUT</p>
</td>
<td class="cellrowborder" valign="top" width="79.24%" headers="mcps1.2.3.1.2 "><p id="p69111216122617"><a name="p69111216122617"></a><a name="p69111216122617"></a>用于指定网络输出的数据类型是否为FP16，数据格式是否为NC1HWC0。</p>
<p id="p1691119168268"><a name="p1691119168268"></a><a name="p1691119168268"></a>该参数需要与OUT_NODES配合使用。</p>
<p id="p189111716152614"><a name="p189111716152614"></a><a name="p189111716152614"></a>若IS_OUTPUT_ADJUST_HW_LAYOUT参数设置为true，对应OUT_NODES中输出节点的输出数据类型为FP16，数据格式为NC1HWC0。</p>
<p id="p991111619264"><a name="p991111619264"></a><a name="p991111619264"></a>取值：false或true，默认为false。</p>
<p id="p132905161296"><a name="p132905161296"></a><a name="p132905161296"></a>配置示例：</p>
<pre class="screen" id="screen17290111613299"><a name="screen17290111613299"></a><a name="screen17290111613299"></a>{ge::AscendString(ge::ir_option::<strong id="b945217595292"><a name="b945217595292"></a><a name="b945217595292"></a>OUT_NODES</strong>), ge::AscendString("add_input:0")},
{ge::AscendString(ge::ir_option::<strong id="b10934171173011"><a name="b10934171173011"></a><a name="b10934171173011"></a>IS_OUTPUT_ADJUST_HW_LAYOUT</strong>), ge::AscendString("true")},</pre>
</td>
</tr>
<tr id="row3525152710286"><td class="cellrowborder" valign="top" width="20.76%" headers="mcps1.2.3.1.1 "><p id="p1560135042814"><a name="p1560135042814"></a><a name="p1560135042814"></a>OUT_NODES</p>
</td>
<td class="cellrowborder" valign="top" width="79.24%" headers="mcps1.2.3.1.2 "><p id="p0601850172820"><a name="p0601850172820"></a><a name="p0601850172820"></a>指定某层输出节点（算子）作为网络模型的输出或指定网络模型输出的名称。</p>
<p id="p186017508283"><a name="p186017508283"></a><a name="p186017508283"></a>如果不指定输出节点（算子名称），则模型的输出默认为最后一层的算子信息，如果指定，则以指定的为准。</p>
<p id="p5601135072814"><a name="p5601135072814"></a><a name="p5601135072814"></a>某些情况下，用户想要查看某层算子参数是否合适，则需要将该层算子的参数输出，即可以在模型编译时通过该参数指定输出某层算子，模型编译后，在相应.om模型文件最后即可以看到指定输出算子的参数信息，如果通过.om模型文件无法查看，则可以将.om模型文件转换成JSON格式后查看。</p>
<p id="p6874165815315"><a name="p6874165815315"></a><a name="p6874165815315"></a>支持三种格式：</p>
<a name="ul1376462243319"></a><a name="ul1376462243319"></a><ul id="ul1376462243319"><li><strong id="b5764722103319"><a name="b5764722103319"></a><a name="b5764722103319"></a>格式1</strong>："node_name1:0;node_name1:1;node_name2:0"。<p id="p12764192218333"><a name="p12764192218333"></a><a name="p12764192218333"></a>网络模型中的节点（node_name）名称，指定的输出节点必须放在双引号中，节点中间使用英文分号分隔。node_name必须是模型编译前的网络模型中的节点名称，冒号后的数字表示第几个输出，例如node_name1:0，表示节点名称为node_name1的第1个输出。</p>
<div class="p" id="p1876452212334"><a name="p1876452212334"></a><a name="p1876452212334"></a><strong id="b127641522163314"><a name="b127641522163314"></a><a name="b127641522163314"></a>示例：</strong><pre class="screen" id="screen87653222332"><a name="screen87653222332"></a><a name="screen87653222332"></a>{ge::AscendString(ge::ir_option::<strong id="b1576592211332"><a name="b1576592211332"></a><a name="b1576592211332"></a>OUT_NODES</strong>), ge::AscendString("add_input:0")},</pre>
</div>
</li></ul>
<a name="ul2050524123220"></a><a name="ul2050524123220"></a><ul id="ul2050524123220"><li><strong id="b18702103223219"><a name="b18702103223219"></a><a name="b18702103223219"></a>格式2</strong>："topname1;topname2"。(仅支持<strong id="b11673151963414"><a name="b11673151963414"></a><a name="b11673151963414"></a>CAFFE</strong>)<p id="p99991049193214"><a name="p99991049193214"></a><a name="p99991049193214"></a>某层输出节点的topname，指定的输出节点必须放在双引号中，节点中间使用英文分号分隔。topname必须是模型编译前的caffe网络模型中的layer的某个top名称；若几个layer有相同的topname，最后一个layer为准。</p>
<div class="p" id="p10852259154017"><a name="p10852259154017"></a><a name="p10852259154017"></a><strong id="b12884455124017"><a name="b12884455124017"></a><a name="b12884455124017"></a>示例：</strong><pre class="screen" id="screen16852105914406"><a name="screen16852105914406"></a><a name="screen16852105914406"></a>{ge::AscendString(ge::ir_option::<strong id="b988465514401"><a name="b988465514401"></a><a name="b988465514401"></a>OUT_NODES</strong>), ge::AscendString("res5c_branch2c")},</pre>
</div>
</li><li><strong id="b102922294427"><a name="b102922294427"></a><a name="b102922294427"></a>格式3</strong>："output1;output2;output3"（仅支持<strong id="b54911031144117"><a name="b54911031144117"></a><a name="b54911031144117"></a>ONNX</strong>）<p id="p2184133454115"><a name="p2184133454115"></a><a name="p2184133454115"></a>指定网络模型输出的名称（output的name），指定的output的name必须放在双引号中，多个output的name中间使用英文分号分隔。output必须是网络模型的输出。</p>
<div class="p" id="p47912112547"><a name="p47912112547"></a><a name="p47912112547"></a><strong id="b1756073504215"><a name="b1756073504215"></a><a name="b1756073504215"></a>示例：</strong><pre class="screen" id="screen178841155154020"><a name="screen178841155154020"></a><a name="screen178841155154020"></a>{ge::AscendString(ge::ir_option::<strong id="b1358254916414"><a name="b1358254916414"></a><a name="b1358254916414"></a>OUT_NODES</strong>), ge::AscendString("output1")},</pre>
</div>
</li></ul>
</td>
</tr>
<tr id="row8956519203612"><td class="cellrowborder" valign="top" width="20.76%" headers="mcps1.2.3.1.1 "><p id="p32211821153616"><a name="p32211821153616"></a><a name="p32211821153616"></a>ENABLE_SCOPE_FUSION_PASSES</p>
</td>
<td class="cellrowborder" valign="top" width="79.24%" headers="mcps1.2.3.1.2 "><p id="p1224193420455"><a name="p1224193420455"></a><a name="p1224193420455"></a>指定编译时需要生效的融合规则列表。</p>
<p id="p17224183415451"><a name="p17224183415451"></a><a name="p17224183415451"></a>融合规则分类如下：</p>
<a name="ul10224143464519"></a><a name="ul10224143464519"></a><ul id="ul10224143464519"><li><p id="p12224153474517"><a name="p12224153474517"></a><a name="p12224153474517"></a>内置融合规则：</p>
<a name="ul922412340452"></a><a name="ul922412340452"></a><ul id="ul922412340452"><li><p id="p1422473474510"><a name="p1422473474510"></a><a name="p1422473474510"></a>General：各网络通用的scope融合规则；默认生效，不支持用户指定失效。</p>
</li><li><p id="p822473411452"><a name="p822473411452"></a><a name="p822473411452"></a>Non-General：特定网络适用的scope融合规则；默认不生效，用户可以通过ENABLE_SCOPE_FUSION_PASSES参数指定生效的融合规则列表。</p>
</li></ul>
</li></ul>
<a name="ul182242345450"></a><a name="ul182242345450"></a><ul id="ul182242345450"><li><p id="p19224134164514"><a name="p19224134164514"></a><a name="p19224134164514"></a>用户自定义的融合规则：</p>
<a name="ul19224123416456"></a><a name="ul19224123416456"></a><ul id="ul19224123416456"><li><p id="p162240346451"><a name="p162240346451"></a><a name="p162240346451"></a>General：加载后默认生效，暂不提供用户指定失效的功能。</p>
</li><li><p id="p7224113413453"><a name="p7224113413453"></a><a name="p7224113413453"></a>Non-General：加载后默认不生效，用户可以通过ENABLE_SCOPE_FUSION_PASSES参数指定生效的融合规则列表。</p>
</li></ul>
</li></ul>
<p id="p12991151451315"><a name="p12991151451315"></a><a name="p12991151451315"></a>指定的融合规则必须放在双引号中，规则中间使用英文逗号分隔。</p>
<p id="p17698869466"><a name="p17698869466"></a><a name="p17698869466"></a><strong id="b1890414554533"><a name="b1890414554533"></a><a name="b1890414554533"></a>配置示例</strong>：</p>
<pre class="screen" id="screen13698106164612"><a name="screen13698106164612"></a><a name="screen13698106164612"></a>{ge::AscendString(ge::ir_option::<strong id="b9524556165911"><a name="b9524556165911"></a><a name="b9524556165911"></a>ENABLE_SCOPE_FUSION_PASSES</strong>), ge::AscendString("ScopePass1,ScopePass2")},</pre>
<div class="note" id="note185311145202814"><a name="note185311145202814"></a><a name="note185311145202814"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p55311645142813"><a name="p55311645142813"></a><a name="p55311645142813"></a>仅aclgrphParseTensorFlow支持该参数。</p>
</div></div>
</td>
</tr>
<tr id="row1064017524538"><td class="cellrowborder" valign="top" width="20.76%" headers="mcps1.2.3.1.1 "><p id="p1564017523532"><a name="p1564017523532"></a><a name="p1564017523532"></a>INPUT_DATA_NAMES</p>
</td>
<td class="cellrowborder" valign="top" width="79.24%" headers="mcps1.2.3.1.2 "><p id="p7640552185317"><a name="p7640552185317"></a><a name="p7640552185317"></a>指定模型文件输入节点的name和index属性的映射关系。系统按照输入name的顺序，设置对应输入节点的index属性。</p>
<p id="p8827579555"><a name="p8827579555"></a><a name="p8827579555"></a><strong id="b1130125225310"><a name="b1130125225310"></a><a name="b1130125225310"></a>配置示例</strong>：</p>
<pre class="screen" id="screen188316579550"><a name="screen188316579550"></a><a name="screen188316579550"></a>{ge::AscendString(ge::ir_option::<strong id="b189081214105614"><a name="b189081214105614"></a><a name="b189081214105614"></a>INPUT_DATA_NAMES</strong>), ge::AscendString("Placeholder,Placeholder_1")},</pre>
</td>
</tr>
<tr id="row34889448218"><td class="cellrowborder" valign="top" width="20.76%" headers="mcps1.2.3.1.1 "><p id="p5994201282915"><a name="p5994201282915"></a><a name="p5994201282915"></a>INPUT_SHAPE</p>
</td>
<td class="cellrowborder" valign="top" width="79.24%" headers="mcps1.2.3.1.2 "><p id="p191195313445"><a name="p191195313445"></a><a name="p191195313445"></a>模型输入的shape信息。</p>
<p id="p4818943515"><a name="p4818943515"></a><a name="p4818943515"></a><strong id="b1997853603511"><a name="b1997853603511"></a><a name="b1997853603511"></a>参数取值</strong>：</p>
<a name="ul18348151983611"></a><a name="ul18348151983611"></a><ul id="ul18348151983611"><li>静态shape。<a name="ul1912311449369"></a><a name="ul1912311449369"></a><ul id="ul1912311449369"><li>若模型为单个输入，则shape信息为"input_name:n,c,h,w"；指定的节点必须放在双引号中。</li><li>若模型有多个输入，则shape信息为"input_name1:n1,c1,h1,w1;input_name2:n2,c2,h2,w2"；不同输入之间使用英文分号分隔，input_name必须是转换前的网络模型中的节点名称。</li></ul>
</li><li>若原始模型中输入数据的某个或某些维度值不固定，当前支持通过设置shape范围的方式转换模型。<a name="zh-cn_topic_0000001265392746_ul1918983152412"></a><a name="zh-cn_topic_0000001265392746_ul1918983152412"></a><ul id="zh-cn_topic_0000001265392746_ul1918983152412"><li>设置shape范围。<p id="p1173134904516"><a name="p1173134904516"></a><a name="p1173134904516"></a>设置INPUT_SHAPE参数时，可将对应维度的值设置为范围。</p>
<a name="ul183092116455"></a><a name="ul183092116455"></a><ul id="ul183092116455"><li>支持按照name设置："input_name1:n1,c1,h1,w1;input_name2:n2,c2,h2,w2"，例如："input_name1:8~20,3,5,-1;input_name2:5,3~9,10,-1"。指定的节点必须放在双引号中，节点中间使用英文分号分隔。input_name必须是转换前的网络模型中的节点名称。如果用户知道data节点的name，推荐按照name设置。</li><li>支持按照index设置："n1,c1,h1,w1;n2,c2,h2,w2"，例如："8~20,3,5,-1;5,3~9,10,-1"。可以不指定节点名，节点按照索引顺序排列，节点中间使用英文分号分隔。按照index设置shape范围时，data节点需要设置属性index，说明是第几个输入，index从0开始。</li></ul>
<p id="zh-cn_topic_0000001265392746_p79798472412"><a name="zh-cn_topic_0000001265392746_p79798472412"></a><a name="zh-cn_topic_0000001265392746_p79798472412"></a>如果用户不想指定维度的范围或具体取值，则可以将其设置为-1，表示此维度可以使用&gt;=0的任意取值，该场景下取值上限为int64数据类型表达范围，但受限于host和device侧物理内存的大小，用户可以通过增大内存来支持。</p>
</li></ul>
</li><li>shape为标量。<a name="ul13148101105515"></a><a name="ul13148101105515"></a><ul id="ul13148101105515"><li>非动态分档场景：<p id="p5673173219453"><a name="p5673173219453"></a><a name="p5673173219453"></a>shape为标量的输入，可选配置，例如模型有两个输入，input_name1为标量，即shape为"[]"形式，input_name2输入shape为[n2,c2,h2,w2]，则shape信息为"<strong id="b1568561055114"><a name="b1568561055114"></a><a name="b1568561055114"></a>input_name1:</strong>;input_name2:n2,c2,h2,w2"，指定的节点必须放在双引号中，不同输入之间使用英文分号分隔，input_name必须是转换前的网络模型中的节点名称；标量的输入如果配置，则配置为空。</p>
</li></ul>
</li></ul>
<p id="p36438454352"><a name="p36438454352"></a><a name="p36438454352"></a><strong id="b2082541350"><a name="b2082541350"></a><a name="b2082541350"></a>配置示例：</strong></p>
<a name="ul75371434611"></a><a name="ul75371434611"></a><ul id="ul75371434611"><li>静态shape，例如某网络的输入shape信息，输入1<strong id="b47771081914"><a name="b47771081914"></a><a name="b47771081914"></a>：</strong>input_0_0 [16,32,208,208]，输入2：input_1_0 [16,64,208,208]，则INPUT_SHAPE的配置信息为：<pre class="screen" id="screen1485218281822"><a name="screen1485218281822"></a><a name="screen1485218281822"></a>{ge::AscendString(ge::ir_option::INPUT_SHAPE), ge::AscendString("input_0_0:16,32,208,208;input_1_0:16,64,208,208")},</pre>
</li><li>设置shape范围的示例：<pre class="screen" id="screen231111178582"><a name="screen231111178582"></a><a name="screen231111178582"></a>{ge::AscendString(ge::ir_option::INPUT_SHAPE), ge::AscendString("input_0_0:1<font style="font-size:8pt" Face="Courier New" >~</font>10,32,208,208;input_1_0:16,64,100<font style="font-size:8pt" Face="Courier New" >~</font>208,100<font style="font-size:8pt" Face="Courier New" >~</font>208")},</pre>
</li><li>shape为标量<a name="ul954852952520"></a><a name="ul954852952520"></a><ul id="ul954852952520"><li>非动态分档场景<p id="p20574194712282"><a name="p20574194712282"></a><a name="p20574194712282"></a>shape为标量的输入，可选配置。例如模型有两个输入，<strong id="b15823193315273"><a name="b15823193315273"></a><a name="b15823193315273"></a>input_name1</strong>为标量，input_name2输入shape为[16,32,208,208]，配置示例为：</p>
<pre class="screen" id="screen4404181322810"><a name="screen4404181322810"></a><a name="screen4404181322810"></a>{ge::AscendString(ge::ir_option::INPUT_SHAPE), ge::AscendString("<strong id="b9833353619"><a name="b9833353619"></a><a name="b9833353619"></a>input_name1:</strong>;input_name2:16,32,208,208")},</pre>
<p id="p4526182514289"><a name="p4526182514289"></a><a name="p4526182514289"></a>上述示例中的<strong id="b1610241016434"><a name="b1610241016434"></a><a name="b1610241016434"></a>input_name1</strong>为可选配置<strong id="b8102110114310"><a name="b8102110114310"></a><a name="b8102110114310"></a>。</strong></p>
</li></ul>
</li></ul>
<div class="note" id="note57257346598"><a name="note57257346598"></a><a name="note57257346598"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1491313313481"><a name="p1491313313481"></a><a name="p1491313313481"></a>INPUT_SHAPE为可选设置。如果不设置，系统直接读取对应Data节点的shape信息，如果设置，以此处设置的为准，同时刷新对应Data节点的shape信息。</p>
</div></div>
</td>
</tr>
</tbody>
</table>

