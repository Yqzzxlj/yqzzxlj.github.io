---
title: 自编码、DQN、GAN网络学习
date: 2020-03-10 15:54:30
tags: pytorch
categories: pytorch学习笔记
---

[代码出处](https://github.com/MorvanZhou/PyTorch-Tutorial/tree/master/tutorial-contents)

这次学习了AutoEncode、DQN、GAN，均为经典网络。

## AutoEncode

AutoEncode 可分为编码部分和解码部分，可以理解为先降维后还原来压缩数据的一种方式。
具体代码见上方链接。

## DQN

深度强化学习。Q-learning 的深度网络模型。根据reward做出选择。
具体代码见上方链接。

## GAN

对抗生成网络。Generator 会根据随机数来生成有意义的数据 , Discriminator 会学习如何判断哪些是真实数据 , 哪些是生成数据, 然后将学习的经验反向传递给 Generator, 让 Generator 能根据随机数生成更像真实数据的数据.
具体代码见上方链接。

## GPU加速

在网络，数据等变量后加上cuda函数

## dropout

减少过拟合，和传统机器学习中正则化有类似的作用

## 批标准化

Batch Normalization (BN) 被添加在每一个全连接和激励函数之间.使数据分布在激励函数的敏感作用区域。