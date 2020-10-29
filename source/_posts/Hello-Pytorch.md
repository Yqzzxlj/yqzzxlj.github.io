---
title: Hello Pytorch
date: 2020-03-06 15:53:40
categories: 
    - 机器学习
    - pytorch学习笔记
top_img: 
toc:
---

## 安装Pytorch

跟大佬学习用anaconda，创建虚拟环境，然后参照pytorch官网教程安装。
遇到了torch和numpy版本冲突的问题，卸载numpy安一个合适的版本

<!--more-->
## 简单使用Pytorch

`asdf`

```python3
// tensor 的使用
torch.tensor(data, requires_grad=True)
// 与numpy的array之间相互转换
torch.from_numpy(np_array)
torch.numpy(tensor)
```

更多详细API见[官方](https://pytorch.org/docs/stable/torch.html)

``` python
// 激励函数的使用
import torch.nn.functional as F
F.relu(tensor)
```

更多详细API见[官方](https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions)



