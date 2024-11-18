#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：FJSP-Obj_AGV
@File    ：load_txt.py
@IDE     ：PyCharm 
@Author  ：lyon
@Date    ：08/10/2023 09:01 
@Des     ：
"""


def load_txt(path):
    array = []
    with open(path) as f:
        data = f.readlines()
    for line in data:
        line = line.strip("\n")  # 去除末尾的换行符
        data_split = list(filter(None, line.split(' ')))
        temp = list(map(int, data_split))
        array.append(temp)

    J_num, M_num, AGV_num = array[0][0], array[0][1], array[0][2]
    Op_num = []
    for i in range(1, len(array)):
        Op_num.append(array[i][0])

    PT = []

    for i in range(J_num):
        Job_i = []
        for j in range(Op_num[i]):
            Job_i.append([] * M_num)
        PT.append(Job_i)

    for idx, job in enumerate(array):
        if idx == 0:
            continue
        for Op in range(job[0]):
            machine_num = job[Op * 4 + 1]  # 兼容机器的数量
            machine = job[2 + Op * 4: 2 + Op * 4 + machine_num]
            PT[idx - 1][Op].append([x for x in machine])
            PT[idx - 1][Op].append(machine_num * [job[4 + Op * 4]])

    Op_dic = dict(enumerate(Op_num))
    O_num = sum(Op_num)
    arrive_time = [0 for i in range(J_num)]
    T_ijave = []  # 每个工件平均工序加工时间之和
    for i in range(J_num):
        Tad = []
        for j in range(Op_dic[i]):
            Tad.append(sum(PT[i][j][1]) / len(PT[i][j][1]))
        T_ijave.append(sum(Tad))
    due_time = [int(T_ijave[i] * 1.5) for i in range(J_num)]
    return PT, M_num, Op_dic, O_num, J_num, AGV_num, arrive_time, due_time


def load_travel_time(path):
    TT = []
    with open(path) as f:
        data = f.readlines()
    for line in data:
        line = line.strip("\n")  # 去除末尾的换行符
        data_split = line.split(' ')
        temp = list(map(int, data_split))
        TT.append(temp)
    return TT


if __name__ == '__main__':
    # PT, M_num, Op_dic, O_num, J_num, AGV_num = load_txt('./fjsp5.txt')
    PT, M_num, Op_dic, O_num, J_num, AGV_num ,arrive_time, due_time= load_txt('Dataset/DeroussiNorre/fjsp5.txt')
    TT = load_travel_time('Dataset/DeroussiNorre/travel_time.txt')
    print(PT)
    print(arrive_time)
    print(due_time)
    print(Op_dic)
    print(O_num)