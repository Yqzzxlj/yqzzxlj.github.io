---
title: markdown syntax
date: 2020-03-03 20:17:35
tags: test
---

```js
function add(n1, n2) {
  const a1 = n1.split('').reverse();
  const a2 = n2.split('').reverse();
  const result = [];
  for (let i = 0, l = Math.max(a1.length, a2.length); i < l; i++) {
    result[i] = (result[i] || 0) + parseInt(a1[i] || 0) + parseInt(a2[i] || 0);
    while (result[i] >= 10) {
      result[i] -= 10;
      result[i + 1] = (result[i + 1] || 0) + 1;
    }
  }
  return result.reverse().join('');
}
```

#一级标题
##二级标题
###三级标题
####四级标题
#####五级标题
######六级标题

**加粗**
*倾斜*
***加粗倾斜***
~~加删除线~~

>引用的内容（可以嵌套）


分割线
---
***

![图片 alt](http://file.koolearn.com/20161207/14810957953513.png "网上随便找的")

[超链接名](超链接地址 "超链接title")

* 列表内容
- 列表内容
+ 列表内容

1. 列表内容
2. 列表内容
3. 列表内容

表头|表头|表头
---|:--:|---:
内容|内容|内容
内容|内容|内容

```
第二行分割表头和内容
-有一个就行
文字默认居左
--两边加:表示文字居中
--右边加:表示文字居右
原生的语法两边都要用|包起来，此处原作者省略。
```

单行代码:代码之间分别用一个反引号包起来
`代码内容`

代码块:代码之间分别用三个反引号包起来，且两边的反引号独占一行

 ```flow
 st=>start: 开始 
 op=>operation: My Operation 
 cond=>condition: Yes or No? 
 e=>end: 结束 

 st->op->cond 
 cond(yes)->e 
 cond(no)->op 
 ```


```flow
st=>start: 开始框
op=>operation: 处理框
cond=>condition: 判断框(是或否?)
sub1=>subroutine: 子流程
io=>inputoutput: 输入输出框
e=>end: 结束框
st->op->cond
cond(yes)->io->e
cond(no)->sub1(right)->op
```
