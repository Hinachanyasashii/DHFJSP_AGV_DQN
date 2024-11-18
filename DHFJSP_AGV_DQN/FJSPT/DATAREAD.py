import numpy as np


# 读取文件数据
def DataReadDHFJSP(Filepath):
    data = []
    enter = '\n'
    # for example "../DATASET/J10M5O5.txt"
    with open(Filepath, "r", encoding='utf-8') as f1:
        for line in f1:
            temp = line.split(' ')
            l = len(temp)
            for i in range(l):
                if temp[i] != enter:
                    data.append(int(temp[i]))
    N = data[0]
    F = data[1]
    TM = data[2]
    H = np.zeros(N, dtype=int)
    NM = []

    # read number of selectable machine for each operation
    p = 5  # index in date array
    for f in range(F):
        for j in range(N):
            H[j] = data[p]
            p = p + 2
            temp = []
            for o in range(int(H[j])):
                temp.append(data[p])
                NM1 = data[p]
                p = p + 1
                for k in range(int(NM1)):
                    p = p + 1
                    p = p + 1
                p = p + 1
            p = p + 1
            NM.append(temp)

    SH = int(np.sum(H))
    opmax = int(max(H))
    M = np.zeros(shape=(N, opmax, TM), dtype=int)
    time = np.zeros(shape=(F, N, opmax, TM))

    p = 5  # index in date array
    for f in range(F):
        for j in range(N):
            H[j] = data[p]
            p = p + 2
            for o in range(int(H[j])):
                NM1 = data[p]
                p = p + 1
                for k in range(int(NM1)):
                    M[j][o][k] = data[p]
                    p = p + 1
                    t = int(M[j][o][k])
                    time[f][j][o][t - 1] = data[p]
                    p = p + 1
                p = p + 1
            p = p + 1

    f1.close()
    # convert benchmark to variable like number of job N, number of machine M, number of operation of each job H
    # number of seletable machine of each operation NM, total operation number SH,
    # the processing time of each operation on each selectable machine time
    ProF = np.zeros((N, F))
    for f in range(F):
        for i in range(N):
            toTime = 0
            for j in range(int(H[i])):
                averT = 0
                NM1 = int(NM[i][j])
                for k in range(NM1):
                    mc = M[i][j][k] - 1
                    averT = averT + time[f][i][j][mc]
                averT = averT / NM1
                toTime = toTime + averT
            ProF[i][f] = toTime
    for i in range(N):
        tot = 0
        for f in range(F):
            tot = tot + ProF[i][f]
        for f in range(F):
            ProF[i][f] = ProF[i][f] / tot
    Op_dic = {}
    for idx, value in enumerate(H):
        Op_dic[idx] = value
    O_num = sum(H)
    arrive_time = [0 for i in range(N)]
    due_time = [100 for i in range(N)]

    return N, F, TM, H, SH, NM, M, time, ProF, arrive_time, due_time, Op_dic, O_num


def process_data(M, time, F, N):
    factory_list = [[] for i in range(F)]
    for j in range(N):
        job_list = []
        for i in range(F):
            operation_list = []
            for k in range(len(M[0])):
                # M 有问题
                filtered_M = [id + 1 for id, x in enumerate(time[i][j][k]) if x != 0]  # 去除M[j][k]中的0元素
                filtered_time = [x for x in time[i][j][k] if x != 0]  # 去除time[i][j][k]中的0元素
                if filtered_M and filtered_time:  # 只要filtered_M或filtered_time不为空，就添加
                    combined_list = [filtered_M, filtered_time]
                    operation_list.append(combined_list)
            factory_list[i].append(operation_list)

    return factory_list  # 每个工厂的每个工件可加工机器索引及其加工时间


if __name__ == '__main__':
    path = '10J2F.txt'
    N, F, TM, H, SH, NM, M, time, ProF, arrive_time, due_time, Op_dic, O_num = DataReadDHFJSP(path)
    # N工件数，F工厂数，TM机器数，H工件工序列表，SH总工序，NM每个工件的工序兼容机器数，M可加工机器序号，time加工时间，ProF，
    # arrive到达时间全为0，due截至时间全为100，Op_dic工序字典，O_num工序总数
    # 调用函数并打印结果
    PT = process_data(M, time, F, N)
