"""
@Project ：FJSP-Obj_AGV
@File    ：FJSPT_Env.py
@IDE     ：PyCharm 
@Author  ：lyon
@Date    ：08/10/2023 14:12 
@Des     ：
"""
import random

import numpy as np
import load_txt
from object_FJSP import Obj_Job, Obj_Machine, Obj_AGV, Obj_Factory


class FJSPT_Env:
    def __init__(self, F_num, J_num, M_num, O_num, AGV_num, Op_dic, PT_f, TT, arrive_time, due_time):
        self.F_num = F_num
        self.J_num = J_num  # 工件数
        self.M_num = M_num  # 机器数
        self.O_num = O_num  # 工序总数
        self.AGV_num = AGV_num  # AGV数
        self.Op_dic = Op_dic  # 工件对应的工序数，字典类型
        self.TT = TT  # travel time，地图
        self.AGV_l = [0 for _ in range(AGV_num)]  # 各AGV的实际使用率
        self.Jobs = []  # 工件集
        self.PT_f = PT_f
        self.C_max = 0
        self.TEC = 0
        self.omega = 1

        for i in range(J_num):
            AT_i = arrive_time[i]
            DT_i = due_time[i]
            factory_id = -1
            job = Obj_Job(i, self.Op_dic[i], AT_i, DT_i, factory_id)
            self.Jobs.append(job)

        self.Machines = []  # 机器集
        for f in range(F_num):
            m_list = []
            for i in range(M_num):
                machine = Obj_Machine(i + 1)
                m_list.append(machine)
            self.Machines.append(m_list)

        self.AGVs = []  # AGV集
        for f in range(F_num):
            a_list = []
            for i in range(AGV_num):
                AGV = Obj_AGV(i)
                a_list.append(AGV)
            self.AGVs.append(a_list)

        self.Factories = []
        for i in range(F_num):
            factory = Obj_Factory(i)
            self.Factories.append(factory)

    # 机器平均使用率
    def states(self):
        # 1 AGV平均利用率 UA_ave
        UA_f = []
        for f in range(self.F_num):
            UA = []
            for a in self.AGVs[f]:
                UA.append(a.UA_l)
            UA_f.append(UA)
        UA_ave_f = []
        for f in range(self.F_num):
            UA_ave_f.append(sum(UA_f[f]) / self.AGV_num)
        # 2 AGV利用率标准差 UA_std
        UA_std_f = []
        for f in range(self.F_num):
            tmp = 0
            for u in UA_f[f]:
                tmp += np.square(u - UA_ave_f[f])
            UA_std_f.append(np.sqrt(tmp / self.AGV_num))

        # 1 机器平均利用率 UJ_ave
        UJ_f = []
        for f in range(self.F_num):
            UJ = []
            for job in self.Jobs:
                UJ.append(job.UJ_i)
            UJ_f.append(UJ)
        UJ_ave_f = []
        for f in range(self.F_num):
            UJ_ave_f.append(sum(UJ_f[f]) / self.M_num)

        # 2 机器利用率标准差 UJ_std
        UJ_std_f = []
        for f in range(self.F_num):
            tmp = 0
            for u in UJ_f[f]:
                tmp += np.square(u - UJ_ave_f[f])
            UJ_std_f.append(np.sqrt(tmp / self.M_num))

        # 3 平均工序完成率 CRO_ave
        NO = 0
        for job in self.Jobs:
            NO += job.cur_Op
        CRO_ave = NO / self.O_num

        # 4 平均工件完成率(将每个工件中工序完成率平均一下) CRJ_ave
        CRJs = []
        for job in self.Jobs:
            CRJs.append(job.CRJ)
        CRJ_ave = sum(CRJs) / self.J_num

        # 5 工件工序完成率标准差 CRJ_std
        tmp = 0
        for u in CRJs:
            tmp += np.square(u - CRJ_ave)
        CRJ_std = np.sqrt(tmp / self.J_num)
        # 6 Estimated tardiness rate Tard_e 预计延误率
        # T_cur:平均最大完工时间
        T_cur1 = []
        for f in range(self.F_num):
            T_cur1.append(sum([self.Machines[f][i].CT_k for i in range(self.M_num)]) / self.M_num)
        T_cur = sum(T_cur1) / len(T_cur1)
        N_tard, N_left = 0, 0
        for job in self.Jobs:
            if job.NO_i > job.cur_Op:
                N_left += job.NO_i - job.cur_Op
                T_left = 0
                for j in range(job.cur_Op + 1, job.NO_i):
                    #  fixme 有问题
                    T_left += 0
                    if T_left + T_cur > job.DT_i:
                        N_tard += job.NO_i - j + 1
        try:
            Tard_e = N_tard / N_left
        except:
            Tard_e = 9999
        # 7 Actual tardiness rate Tard_a
        N_tard, N_left = 0, 0
        for job in self.Jobs:
            if job.NO_i > job.cur_Op:
                N_left += job.NO_i - job.cur_Op
                try:
                    if job.CT_i > job.DT_i:
                        N_tard += job.NO_i - j
                except:
                    pass
        try:
            Tard_a = N_tard / N_left
        except:
            Tard_a = 9999
        return [sum(UJ_ave_f) / len(UJ_ave_f), sum(UJ_std_f) / len(UJ_std_f), CRO_ave, CRJ_ave, CRJ_std,
                sum(UA_ave_f) / len(UA_ave_f), sum(UA_std_f) / len(UA_std_f), Tard_e, Tard_a]

    def scheduling(self, action: [Obj_Job, Obj_Machine, Obj_AGV, Obj_Factory]):
        job, machine, AGV, fac = action[0], action[1], action[2], action[3]
        # todo
        job.next_pos = machine.id  # todo 重新走一遍调度流程
        _Mj = self.PT_f[fac.id][job.id][job.cur_Op][0].index(machine.id)
        PT_job = self.PT_f[fac.id][job.id][job.cur_Op][1][_Mj]

        if job.cur_pos != job.next_pos:  # 当前工件位置和工件下一工序加工位置不同，使用AGV的情况
            # 小车开始搬运时间 = max(工件上一工序结束加工时间，小车结束时间 + 小车从结束位置前往工件处的时间)
            unload_time = self.TT[AGV.cur_pos][job.cur_pos]
            load_time = self.TT[job.cur_pos][job.next_pos]
            unload_start_time = AGV.transport_start_time(job, unload_time, load_time)

            start_time = machine.job_start_time(PT_job, unload_start_time + unload_time + load_time)
            self.AGVs[fac.id][AGV.id]._update(unload_start_time, unload_time, load_time, job)
        else:
            # next_machine可能和now_machine是同一个，这个时候就不用AGV参与
            start_time = machine.job_start_time(PT_job, job.CT_i)
        end_time = start_time + PT_job
        self.Jobs[job.id]._update(start_time, end_time, machine.id)
        self.Machines[fac.id][machine.id - 1]._update(start_time, end_time, job.id)
        self.Factories[fac.id]._update(self.Jobs[job.id])

    def reward(self, Ta_t, Te_t, Ta_t1, Te_t1, U_t, U_t1):
        # 奖励函数用于判断在状态s下选择动作a的好坏
        CT_i = []
        for job in self.Jobs:
            CT_i.append(job.CT_i)
        r_1 = self.C_max - max(CT_i)
        self.C_max = max(CT_i)
        # todo 定义能耗相关的参数，r_1和r_2怎么归一化呢，怎么得到最大值和最小值呢，r_normal = (r - r_min) / (r_max - r_min)
        TEC_i = 0
        r_2 = self.TEC - TEC_i
        self.TEC = TEC_i
        r_t = self.omega * r_1 + (1 - self.omega) * r_2
        return r_t


# PT, M_num, Op_dic, O_num, J_num, AGV_num, arrive_time, due_time = load_txt.load_txt('Dataset/DeroussiNorre/fjsp5.txt')
# TT = load_txt.load_travel_time('Dataset/DeroussiNorre/travel_time.txt')
# env = FJSPT_Env(F_num,J_num, M_num, O_num, AGV_num, Op_dic, PT, TT, arrive_time, due_time)
