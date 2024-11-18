#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Master_thesis
@File    ：gantt_chart.py
@IDE     ：PyCharm 
@Author  ：lyon
@Date    ：12/12/2023 22:02 
@Des     ：
"""
from matplotlib import pyplot as plt


def gantt_chart(Machines):
    plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei,Times New Roman
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    M = ['red', 'blue', 'yellow', 'orange', 'green', 'palegoldenrod', 'purple', 'pink', 'Thistle', 'Magenta',
         'SlateBlue', 'RoyalBlue', 'Cyan', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod', 'mediumslateblue',
         'navajowhite', 'navy', 'sandybrown', 'moccasin']
    Job_text = ['J' + str(i + 1) for i in range(100)]
    Machine_text = ['M' + str(i + 1) for i in range(50)]

    for i in range(len(Machines)):
        for j in range(len(Machines[i].start)):
            if Machines[i].finish[j] - Machines[i].start[j] != 0:
                plt.barh(i, width=Machines[i].finish[j] - Machines[i].start[j],
                         height=0.8, left=Machines[i].start[j],
                         color=M[Machines[i]._on[j]],
                         edgecolor='black')
                plt.text(x=Machines[i].start[j] + (Machines[i].finish[j] - Machines[i].start[j]) / 2 - 0.1,
                         y=i,
                         s=Job_text[Machines[i]._on[j]],
                         fontsize=12)
    plt.show()
