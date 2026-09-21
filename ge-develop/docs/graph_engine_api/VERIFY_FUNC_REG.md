# VERIFY\_FUNC\_REG<a name="ZH-CN_TOPIC_0000002484311528"></a>

## 头文件<a name="section995734502317"></a>

\#include <graph/operator\_reg.h\>

## 功能说明<a name="zh-cn_topic_0000001312403753_section212607105720"></a>

注册算子的Verify函数。

## 函数原型<a name="zh-cn_topic_0000001312403753_section129451113125413"></a>

```
VERIFY_FUNC_REG(op_name, x)
```

## 参数说明<a name="zh-cn_topic_0000001312403753_section1843519321017"></a>

<a name="zh-cn_topic_0000001312403753_table18350576"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001312403753_row29692563"><th class="cellrowborder" valign="top" width="25.3%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001312403753_p56287378"><a name="zh-cn_topic_0000001312403753_p56287378"></a><a name="zh-cn_topic_0000001312403753_p56287378"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="22.84%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001312403753_p62983758"><a name="zh-cn_topic_0000001312403753_p62983758"></a><a name="zh-cn_topic_0000001312403753_p62983758"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="51.85999999999999%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001312403753_p47160602"><a name="zh-cn_topic_0000001312403753_p47160602"></a><a name="zh-cn_topic_0000001312403753_p47160602"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001312403753_row61912443"><td class="cellrowborder" valign="top" width="25.3%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312403753_p9757144353"><a name="zh-cn_topic_0000001312403753_p9757144353"></a><a name="zh-cn_topic_0000001312403753_p9757144353"></a>op_name</p>
</td>
<td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312403753_p13947197203216"><a name="zh-cn_topic_0000001312403753_p13947197203216"></a><a name="zh-cn_topic_0000001312403753_p13947197203216"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="51.85999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312403753_p1476526254"><a name="zh-cn_topic_0000001312403753_p1476526254"></a><a name="zh-cn_topic_0000001312403753_p1476526254"></a>算子类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312403753_row168464453311"><td class="cellrowborder" valign="top" width="25.3%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312403753_p147558420513"><a name="zh-cn_topic_0000001312403753_p147558420513"></a><a name="zh-cn_topic_0000001312403753_p147558420513"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312403753_p209461672321"><a name="zh-cn_topic_0000001312403753_p209461672321"></a><a name="zh-cn_topic_0000001312403753_p209461672321"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="51.85999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312403753_p47631661757"><a name="zh-cn_topic_0000001312403753_p47631661757"></a><a name="zh-cn_topic_0000001312403753_p47631661757"></a>Verify函数名，和<a href="IMPLEMT_VERIFIER.md">IMPLEMT_VERIFIER</a>的Verify函数名保持一致。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001312403753_section8491941407"></a>

无

## 约束说明<a name="zh-cn_topic_0000001312403753_section65498832"></a>

无

