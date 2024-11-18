"""
@Project ：FJSP-Obj_AGV
@File    ：action_space.py
@IDE     ：PyCharm 
@Author  ：lyon
@Date    ：08/10/2023 14:48 
@Des     ：
"""
import random

import numpy as np

from FJSPT import load_txt
from FJSPT.DATAREAD import process_data, DataReadDHFJSP
from FJSPT.object_FJSP import Obj_Job, Obj_AGV, Obj_Machine, Obj_Factory
from FJSPT_Env import FJSPT_Env


def dispatch_rule(a: int, env: FJSPT_Env) -> [Obj_Job, Obj_Machine, Obj_AGV,Obj_Factory]:
    """

    :param a: 动作序号，a[0]工件编号，a[1]机器编号，a[2]AGV编号
    :param env: 调度环境
    :return: 选择的job, machine, AGV
    """
    # if a == 0:
    #     return custom_rule1(env)
    # elif a == 1:
    #     return custom_rule2(env)
    # elif a == 2:
    #     return custom_rule3(env)
    # elif a == 3:
    #     return custom_rule4(env)
    # elif a == 4:
    #     return custom_rule5(env)
    # elif a == 5:
    #     return custom_rule6(env)

    # 构建动作空间，每个动作由工件、机器、AGV、工厂调度规则组成
    methods = []
    for i in [SPTSO, LPTSO, FOPNR, MOPNR, SPTJ, LPTJ, ]:
        for j in [LUM, HUM, SPTM, LPTM, ]:
            for k in [HUV, LUV, MinCTV, MaxCTV]:
                for l in [SPTF, LPTF, MinNJF, MaxNJF, ]:
                    methods.append([i, j, k, l])

    CDR1 = [SPTJ, MinCTM, MinCTV, MinNJF]
    CDR2 = [SPTJ, SPTM, MinCTV, MinNJF]
    CDR3 = [SPTR, MinCTM, MinCTV, MinNJF]
    CDR4 = [SPTR, SPTM, MinCTV, MinNJF]
    CDR5 = [MOPNR, MinCTM, MinCTV, MinNJF]
    CDR6 = [MOPNR, SPTM, MinCTV, MinNJF]
    rules = methods[a]  # 网络选择的动作a
    # rules = CDR6
    # 先选工件，再选工厂，再选机器，再选AGV
    job = rules[0](env)
    if job.factory_id == -1:
        factory = rules[3](env)
        job.factory_id = factory.id
    else:
        factory = env.Factories[job.factory_id]
    machine = rules[1](env, job, factory)

    AGV = rules[2](env, job, factory)

    # try:
    #     rules = methods[a - 6]
    #     job = rules[0](env)
    #     if job.factory_id == -1:
    #         factory = rules[3](env)
    #     else:
    #         factory = env.Factories[job.factory_id]
    #     machine = rules[1](env, job, factory)
    #     AGV = rules[2](env, job)
    # except:
    #     print("动作有问题！")

    return job, machine, AGV, factory


def SPTF(env: FJSPT_Env)-> Obj_Factory:
    """
    最短加工时间，工厂，返回选择的工厂类生成的实例
    :return:
    """
    Factories = env.Factories
    ct_f = []
    for f in Factories:
        ct_f.append(f.CT_f)
    return Factories[np.argmin(ct_f)]


def LPTF(env: FJSPT_Env)-> Obj_Factory:
    """
    最短加工时间，工厂，返回选择的工厂类生成的实例
    :return:
    """
    Factories = env.Factories
    ct_f = []
    for f in Factories:
        ct_f.append(f.CT_f)
    return Factories[np.argmax(ct_f)]


def MinNJF(env: FJSPT_Env)-> Obj_Factory:
    """
    选择工件数最少的工件 Minimum number of jobs
    :param env:
    :return:
    """
    Factories = env.Factories
    J_num_f = []
    for f in Factories:
        J_num_f.append(f.J_num_f)
    return Factories[np.argmin(J_num_f)]


def MaxNJF(env: FJSPT_Env)-> Obj_Factory:
    """
    选择工件数最少的工件 Minimum number of jobs
    :param env:
    :return:
    """
    Factories = env.Factories
    J_num_f = []
    for f in Factories:
        J_num_f.append(f.J_num_f)
    return Factories[np.argmax(J_num_f)]


def custom_rule1(env: FJSPT_Env) -> [Obj_Job, Obj_Machine, Obj_AGV,Obj_Factory]:
    """
    选择离交货期近的工件，最早空闲的机器，利用率最低的AGV
    :param env:
    :return:
    """
    Machines = env.Machines
    Jobs = env.Jobs
    AGVs = env.AGVs
    Factories = env.Factories
    factory = Factories[0]

    # T_cur:平均最大完工时间
    T_cur = sum([Machines[i].CT_k for i in range(env.M_num)]) / env.M_num
    # Tard_Jobs:预计不能按期完成的工件
    Tard_Jobs = [job for job in Jobs if job.cur_Op < job.NO_i and job.DT_i < T_cur]
    UC_Jobs = [job for job in Jobs if job.cur_Op < job.NO_i]  # Uncompleted Job
    if Tard_Jobs == []:
        Job_i = UC_Jobs[np.argmax([(job.DT_i - T_cur) / (job.NO_i - job.cur_Op) for job in UC_Jobs])]
    else:
        T_ijave = []
        for job in Tard_Jobs:
            Tad = []
            for j in range(job.cur_Op, job.NO_i):
                Tad.append(sum(env.PT_f[factory.id][job.id][j][1]) / len(env.PT_f[factory.id][job.id][j][1]))
            T_ijave.append(T_cur + sum(Tad) - job.DT_i)
            # todo 离Due Time近的工件, 看看最后是argmax还是argmin
        Job_i = Tard_Jobs[np.argmax(T_ijave)]
    CT_i = Job_i.CT_i
    cur_Op = Job_i.cur_Op
    Machines_ST = []
    compatible_Mj = env.PT_f[factory.id][Job_i.id][Job_i.cur_Op][0]
    compatible_machines = []
    for i in compatible_Mj:
        compatible_machines.append(env.Machines[i - 1])
    for id, machine in enumerate(compatible_machines):
        Machines_ST.append(max(machine.CT_k, CT_i + env.PT_f[factory.id][Job_i.id][Job_i.cur_Op][1][id]))
    Machine_k = compatible_machines[np.argmin(Machines_ST)]
    utili = []
    for AGV in AGVs:
        utili.append(AGV.UA_l)
    AGV_l = env.AGVs[np.argmin(utili)]
    return Job_i, Machine_k, AGV_l, factory


def custom_rule2(env: FJSPT_Env) -> [Obj_Job, Obj_Machine, Obj_AGV, Obj_Factory]:
    Machines = env.Machines
    Jobs = env.Jobs
    AGVs = env.AGVs
    Factories = env.Factories
    factory = Factories[0]
    # T_cur:平均最大完工时间
    T_cur = sum([Machines[i].CT_k for i in range(env.M_num)]) / env.M_num
    # Tard_Jobs:预计不能按期完成的工件
    Tard_Jobs = [job for job in Jobs if job.cur_Op < job.NO_i and job.DT_i < T_cur]
    UC_Jobs = [job for job in Jobs if job.cur_Op < job.NO_i]  # Uncompleted Job
    T_ijave = []
    for i in range(env.J_num):
        Tad = []
        for j in range(Jobs[i].cur_Op, Jobs[i].NO_i):
            Tad.append(sum(env.PT_f[factory.id][Jobs[i].id][j][1]) / len(env.PT_f[factory.id][Jobs[i].id][j][1]))
        T_ijave.append(sum(Tad))
    if Tard_Jobs == []:
        Job_i = UC_Jobs[np.argmin([(job.DT_i - T_cur) / T_ijave[job.id] for job in UC_Jobs])]
    else:
        # todo argmax和argmin
        Job_i = Tard_Jobs[np.argmax([T_cur + T_ijave[job.id] - job.DT_i for job in Tard_Jobs])]
    # todo 机器和AGV的选择方式和custom rule 1一样
    CT_i = Job_i.CT_i
    cur_Op = Job_i.cur_Op
    Machines_ST = []
    compatible_Mj = env.PT_f[factory.id][Job_i.id][Job_i.cur_Op][0]

    compatible_machines = []
    for i in compatible_Mj:
        compatible_machines.append(env.Machines[i - 1])
    for id, machine in enumerate(compatible_machines):
        Machines_ST.append(max(machine.CT_k, env.PT_f[factory.id][Job_i.id][Job_i.cur_Op][1][id]))
    Machine_k = compatible_machines[np.argmin(Machines_ST)]
    utili = []
    for AGV in AGVs:
        utili.append(AGV.UA_l)
    AGV_l = env.AGVs[np.argmin(utili)]
    return Job_i, Machine_k, AGV_l,factory


def custom_rule3(env: FJSPT_Env) -> [Obj_Job, Obj_Machine, Obj_AGV,Obj_Factory]:
    Machines = env.Machines
    Jobs = env.Jobs
    AGVs = env.AGVs
    Factories = env.Factories
    factory = Factories[0]
    # T_cur:平均最大完工时间
    T_cur = sum([Machines[i].CT_k for i in range(env.M_num)]) / env.M_num
    UC_Jobs = [job for job in Jobs if job.cur_Op < job.NO_i]  # Uncompleted Job
    T_ijave = []
    for job in UC_Jobs:
        Tad = []
        for j in range(job.cur_Op, job.NO_i):
            Tad.append(sum(env.PT_f[factory.id][job.id][j][1]) / len(env.PT_f[factory.id][job.id][j][1]))
        T_ijave.append(T_cur + sum(Tad) - job.DT_i)
    Job_i = UC_Jobs[np.argmax(T_ijave)]
    cur_Op = Job_i.cur_Op
    if random.random() < 0.5:
        compatible_Mj = env.PT_f[factory.id][Job_i.id][Job_i.cur_Op][0]

        compatible_machines = []
        for i in compatible_Mj:
            compatible_machines.append(env.Machines[i - 1])
        UM_k = []
        for machine in compatible_machines:
            UM_k.append(machine.UM_k)
        Machine_k = compatible_machines[np.argmin(UM_k)]  # 选择机器利用率最低的机器
    else:
        compatible_Mj = env.PT_f[factory.id][Job_i.id][Job_i.cur_Op][0]
        compatible_machines = []
        for i in compatible_Mj:
            compatible_machines.append(env.Machines[i - 1])
        MT = []  # Machining Time
        for machine in compatible_machines:
            MT.append(sum(machine.end_time) - sum(machine.start_time))
        Machine_k = compatible_machines[np.argmin(MT)]  # 选择机器加工时间最少的机器
    # todo AGV的选择方式
    utili = []
    for AGV in AGVs:
        utili.append(AGV.UA_l)
    AGV_l = env.AGVs[np.argmin(utili)]
    return Job_i, Machine_k, AGV_l,factory


def custom_rule4(env: FJSPT_Env) -> [Obj_Job, Obj_Machine, Obj_AGV,Obj_Factory]:
    Factories = env.Factories
    factory = Factories[0]
    Machines = env.Machines
    Jobs = env.Jobs
    AGVs = env.AGVs
    UC_Jobs = [job for job in Jobs if job.cur_Op < job.NO_i]  # Uncompleted Job
    Job_i = random.choice(UC_Jobs)
    CT_i = Job_i.CT_i
    cur_Op = Job_i.cur_Op
    Machines_ST = []
    compatible_Mj = env.PT_f[factory.id][Job_i.id][Job_i.cur_Op][0]

    compatible_machines = []
    for i in compatible_Mj:
        compatible_machines.append(env.Machines[i - 1])
    for id, machine in enumerate(compatible_machines):
        Machines_ST.append(max(machine.CT_k, CT_i + env.PT_f[factory.id][Job_i.id][Job_i.cur_Op][1][id]))

    Machine_k = compatible_machines[np.argmin(Machines_ST)]  # 选择最早的可用机器
    # todo AGV的选择方式
    utili = []
    for AGV in AGVs:
        utili.append(AGV.UA_l)
    AGV_l = env.AGVs[np.argmin(utili)]
    return Job_i, Machine_k, AGV_l,factory


def custom_rule5(env: FJSPT_Env) -> [Obj_Job, Obj_Machine, Obj_AGV]:
    Factories = env.Factories
    factory = Factories[0]
    Machines = env.Machines
    Jobs = env.Jobs
    AGVs = env.AGVs
    # T_cur:平均最大完工时间
    T_cur = sum([Machines[i].CT_k for i in range(env.M_num)]) / env.M_num
    # Tard_Jobs:预计不能按期完成的工件
    Tard_Jobs = [job for job in Jobs if job.cur_Op < job.NO_i and job.DT_i < T_cur]
    UC_Jobs = [job for job in Jobs if job.cur_Op < job.NO_i]  # Uncompleted Job
    if Tard_Jobs == []:
        Job_i = UC_Jobs[np.argmin([job.CRJ * (job.DT_i - T_cur) for job in UC_Jobs])]
    else:
        T_ijave = []
        for job in Tard_Jobs:
            Tad = []
            for j in range(job.cur_Op, job.NO_i):
                Tad.append(sum(env.PT_f[factory.id][job.id][j][1]) / len(env.PT_f[factory.id][job.id][j][1]))

            T_ijave.append(1 / (job.CRJ + 1) * (T_cur + sum(Tad) - job.DT_i))
            # todo 离Due Time近的工件, 看看最后是argmax还是argmin
        Job_i = Tard_Jobs[np.argmax(T_ijave)]
    CT_i = Job_i.CT_i
    cur_Op = Job_i.cur_Op
    Machines_ST = []
    compatible_Mj = env.PT_f[factory.id][Job_i.id][Job_i.cur_Op][0]
    compatible_machines = []
    for i in compatible_Mj:
        compatible_machines.append(env.Machines[i - 1])
    for id, machine in enumerate(compatible_machines):
        Machines_ST.append(max(machine.CT_k, CT_i + env.PT_f[factory.id][Job_i.id][Job_i.cur_Op][1][id]))

    Machine_k = compatible_machines[np.argmin(Machines_ST)]  # 选择最早的可用机器
    # todo AGV的选择方式
    utili = []
    for AGV in AGVs:
        utili.append(AGV.UA_l)
    AGV_l = env.AGVs[np.argmin(utili)]
    return Job_i, Machine_k, AGV_l,factory


def custom_rule6(env: FJSPT_Env) -> [Obj_Job, Obj_Machine, Obj_AGV]:
    Factories = env.Factories
    factory = Factories[0]
    Machines = env.Machines
    Jobs = env.Jobs
    AGVs = env.AGVs
    # T_cur:平均最大完工时间
    T_cur = sum([Machines[i].CT_k for i in range(env.M_num)]) / env.M_num
    UC_Jobs = [job for job in Jobs if job.cur_Op < job.NO_i]  # Uncompleted Job
    T_ijave = []
    for job in UC_Jobs:
        Tad = []
        for j in range(job.cur_Op, job.NO_i):
            Tad.append(sum(env.PT_f[factory.id][job.id][j][1]) / len(env.PT_f[factory.id][job.id][j][1]))
        T_ijave.append(T_cur + sum(Tad) - job.DT_i)
    Job_i = UC_Jobs[np.argmax(T_ijave)]
    CT_i = Job_i.CT_i
    cur_Op = Job_i.cur_Op
    Machines_ST = []
    compatible_Mj = env.PT_f[factory.id][Job_i.id][Job_i.cur_Op][0]
    compatible_machines = []
    for i in compatible_Mj:
        compatible_machines.append(env.Machines[i - 1])
    for id, machine in enumerate(compatible_machines):
        Machines_ST.append(max(machine.CT_k, CT_i + env.PT_f[factory.id][Job_i.id][Job_i.cur_Op][1][id]))
    Machine_k = compatible_machines[np.argmin(Machines_ST)]  # 选择最早的可用机器
    # todo AGV的选择方式
    utili = []
    for AGV in AGVs:
        utili.append(AGV.UA_l)
    AGV_l = env.AGVs[np.argmin(utili)]
    return Job_i, Machine_k, AGV_l,factory


# ###################################################工件调度规则###################################################
# 随机选择工件
def random_job(env: FJSPT_Env) -> Obj_Job:
    job_id = random.randint(0, env.J_num - 1)
    while env.Jobs[job_id].cur_Op >= env.Jobs[job_id].NO_i:
        job_id = random.randint(0, env.J_num - 1)
    return env.Jobs[job_id]


# Shortest  Processing Time of Job 优先安排处理时间最短的作业，使得处理时间短的作业尽早开始
def SPTJ(env: FJSPT_Env) -> Obj_Job:
    Jobs = []
    for j in env.Jobs:
        if j.cur_Op < j.NO_i:
            Jobs.append(j)
    PT_f = []

    for job in Jobs:
        pt = []
        if job.factory_id == -1:
            for f in range(env.F_num):
                pt.append(min(env.PT_f[f][job.id][job.cur_Op][1]))  # 其实max和min都一样，因为数据集中不同机器上的加工时间一样
        else:
            for f in range(env.F_num):
                if job.factory_id == f:
                    pt.append(min(env.PT_f[f][job.id][job.cur_Op][1]))  # 其实max和min都一样，因为数据集中不同机器上的加工时间一样
                else:
                    pt.append(99999)
        PT_f.append(pt)
    PT_f = np.asarray(PT_f, dtype=object)
    job_index = np.argmin(PT_f) // env.F_num
    return Jobs[job_index]


# Longest Processing Time of Job 优先安排处理时间最长的作业，使得处理时间长的作业尽早开始，从而最大程度地减少系统的平均加工时间。
def LPTJ(env: FJSPT_Env) -> Obj_Job:
    Jobs = []
    for j in env.Jobs:
        if j.cur_Op < j.NO_i:
            Jobs.append(j)
    PT_f = []

    for job in Jobs:
        pt = []
        if job.factory_id == -1:
            for f in range(env.F_num):
                pt.append(max(env.PT_f[f][job.id][job.cur_Op][1]))  # 其实max和min都一样，因为数据集中不同机器上的加工时间一样
        else:
            for f in range(env.F_num):
                if job.factory_id == f:
                    pt.append(max(env.PT_f[f][job.id][job.cur_Op][1]))  # 其实max和min都一样，因为数据集中不同机器上的加工时间一样
                else:
                    pt.append(-99999)
        PT_f.append(pt)
    PT_f = np.asarray(PT_f, dtype=object)
    job_index = np.argmax(PT_f) // env.F_num
    return Jobs[job_index]


# Shortest Processing Time Remaining 选择剩余工序 平均加工时间和 最短的工件
def SPTR(env: FJSPT_Env) -> Obj_Job:
    Jobs = []
    for j in env.Jobs:
        if j.cur_Op < j.NO_i:
            Jobs.append(j)
    RPT_f = []
    for j in Jobs:
        RPT_j = []
        rpt = 0
        if j.factory_id == -1:
            for f in range(env.F_num):
                cur_Op = j.cur_Op
                if j.cur_Op < j.NO_i:
                    while cur_Op < j.NO_i:
                        rpt += sum(env.PT_f[f][j.id][j.cur_Op][1]) / len(env.PT_f[f][j.id][j.cur_Op])
                        cur_Op += 1
                else:
                    rpt = 99999
                RPT_j.append(rpt)
        else:
            for f in range(env.F_num):
                if f == j.factory_id:
                    cur_Op = j.cur_Op
                    rpt = 0
                    if j.cur_Op < j.NO_i:
                        while cur_Op < j.NO_i:
                            rpt += sum(env.PT_f[f][j.id][j.cur_Op][1]) / len(env.PT_f[f][j.id][j.cur_Op])
                            cur_Op += 1
                    else:
                        rpt = 99999
                else:
                    rpt = 99999
                RPT_j.append(rpt)
        RPT_f.append(RPT_j)
    RPT_f = np.asarray(RPT_f, dtype=object)
    index_j = np.argmin(RPT_f)// env.F_num
    return Jobs[index_j]


# Longest Processing Time Remaining 选择剩余工序 平均加工时间和 最长的工件
# MWKR(Most Work Remaining)规则，即“最大未完成任务”规则。 它是优先选择余下加工时间最长的任务。
def LPTR(env: FJSPT_Env) -> Obj_Job:
    Jobs = []
    for j in env.Jobs:
        if j.cur_Op < j.NO_i:
            Jobs.append(j)
    RPT_f = []
    for j in Jobs:
        RPT_j = []
        rpt = 0
        if j.factory_id == -1:
            for f in range(env.F_num):
                cur_Op = j.cur_Op
                if j.cur_Op < j.NO_i:
                    while cur_Op < j.NO_i:
                        rpt += sum(env.PT_f[f][j.id][j.cur_Op][1]) / len(env.PT_f[f][j.id][j.cur_Op])
                        cur_Op += 1
                else:
                    rpt = -99999
                RPT_j.append(rpt)
        else:
            for f in range(env.F_num):
                if f == j.factory_id:
                    cur_Op = j.cur_Op
                    rpt = 0
                    if j.cur_Op < j.NO_i:
                        while cur_Op < j.NO_i:
                            rpt += sum(env.PT_f[f][j.id][j.cur_Op][1]) / len(env.PT_f[f][j.id][j.cur_Op])
                            cur_Op += 1
                    else:
                        rpt = -99999
                else:
                    rpt = -99999
                RPT_j.append(rpt)
        RPT_f.append(RPT_j)
    RPT_f = np.asarray(RPT_f, dtype=object)
    index_j = np.argmax(RPT_f) // env.F_num
    return Jobs[index_j]


# Fewest Operations Remaining 选择剩余工序数最少的工件
def FOPNR(env: FJSPT_Env) -> Obj_Job:
    Jobs = []
    for j in env.Jobs:
        if j.cur_Op < j.NO_i:
            Jobs.append(j)
    Remain_O_num = []
    for j in Jobs:
        if j.cur_Op < j.NO_i:
            Remain_O_num.append(j.NO_i - j.cur_Op)
        else:
            Remain_O_num.append(0)
    return Jobs[np.argmin(Remain_O_num)]


# Most Operations Remaining 选择剩余工序数最多的工件
def MOPNR(env: FJSPT_Env) -> Obj_Job:
    Jobs = []
    for j in env.Jobs:
        if j.cur_Op < j.NO_i:
            Jobs.append(j)
    Remain_O_num = []
    for j in Jobs:
        if j.cur_Op < j.NO_i:
            Remain_O_num.append(j.NO_i - j.cur_Op)
        else:
            Remain_O_num.append(0)
    return Jobs[np.argmax(Remain_O_num)]


# Shortest Processing Time of Subsequent Operation, 选择下一个加工时间最短的工序
def SPTSO(env: FJSPT_Env) -> Obj_Job:
    Jobs = []
    for j in env.Jobs:
        if j.cur_Op < j.NO_i:
            Jobs.append(j)
    NPT_f = []
    for j in Jobs:
        NPT_j = []
        if j.cur_Op < j.NO_i - 1:
            if j.factory_id == -1:
                for f in range(env.F_num):
                    NPT_j.append(min(env.PT_f[f][j.id][j.cur_Op + 1][1]))
            else:
                for f in range(env.F_num):
                    if f == j.factory_id:
                        NPT_j.append(min(env.PT_f[f][j.id][j.cur_Op + 1][1]))
                    else:
                        NPT_j.append(99999)

        else:
            NPT_j.append(99999)
        NPT_f.append(NPT_j)
    NPT_f = np.asarray(NPT_f, dtype=object)
    index_job = np.argmin(NPT_f) // env.F_num
    return Jobs[index_job]


# Longest Processing Time of Subsequent Operation, 选择下一个加工时间最长的工序
def LPTSO(env: FJSPT_Env) -> Obj_Job:
    Jobs = []
    for j in env.Jobs:
        if j.cur_Op < j.NO_i:
            Jobs.append(j)
    NPT_f = []
    for j in Jobs:
        NPT_j = []
        if j.cur_Op < j.NO_i - 1:
            if j.factory_id == -1:
                for f in range(env.F_num):
                    # print(j.cur_Op + 1)
                    # if j.cur_Op + 1 == 5:
                    #     a = 0
                    NPT_j.append(max(env.PT_f[f][j.id][j.cur_Op + 1][1]))
            else:
                for f in range(env.F_num):
                    if f == j.factory_id:
                        NPT_j.append(max(env.PT_f[f][j.id][j.cur_Op + 1][1]))
                    else:
                        NPT_j.append(-99999)

        else:
            NPT_j.append(-99999)
        NPT_f.append(NPT_j)

    NPT_f = np.asarray(NPT_f, dtype=object)
    index_job = np.argmax(NPT_f) // env.F_num
    return Jobs[index_job]


# ###################################################机器调度规则###################################################
# 在选中工件的可选机器集中选择相应的机器
def random_machine(env: FJSPT_Env, job: Obj_Job, fac: Obj_Factory) -> Obj_Machine:
    cur_Op = job.cur_Op
    # compatible_machines = job.PT_i[cur_Op][0]
    compatible_machines = env.PT_f[fac.id][job.id][job.cur_Op + 1][0]
    machine_id = random.choice(compatible_machines)
    return env.Machines[machine_id - 1]


# Lowest Utilization Machine，选择利用率最低的机器
def LUM(env: FJSPT_Env, job: Obj_Job, fac: Obj_Factory) -> Obj_Machine:
    compatible_Mj = env.PT_f[fac.id][job.id][job.cur_Op][0]
    compatible_machines = []
    for i in compatible_Mj:
        compatible_machines.append(env.Machines[fac.id][i - 1])
    UM_k = []
    for machine in compatible_machines:
        UM_k.append(machine.UM_k)
    return compatible_machines[np.argmin(UM_k)]


# Highest Utilization Machine，选择利用率最高的机器
def HUM(env: FJSPT_Env, job: Obj_Job, fac: Obj_Factory) -> Obj_Machine:
    cur_Op = job.cur_Op
    compatible_Mj = env.PT_f[fac.id][job.id][job.cur_Op][0]
    compatible_machines = []
    for i in compatible_Mj:
        compatible_machines.append(env.Machines[fac.id][i - 1])
    UM_k = []
    for machine in compatible_machines:
        UM_k.append(machine.UM_k)
    return compatible_machines[np.argmax(UM_k)]


# Shortest Processing Time of Machine，选择加工时间最短的机器
def SPTM(env: FJSPT_Env, job: Obj_Job, fac: Obj_Factory) -> Obj_Machine:
    cur_Op = job.cur_Op
    compatible_Mj = env.PT_f[fac.id][job.id][cur_Op][0]
    compatible_machines = []
    for i in compatible_Mj:
        compatible_machines.append(env.Machines[fac.id][i - 1])
    PT = env.PT_f[fac.id][job.id][cur_Op][1]
    return compatible_machines[np.argmin(PT)]


# Longest Processing Time of Machine，选择加工时间最长的机器
def LPTM(env: FJSPT_Env, job: Obj_Job, fac: Obj_Factory) -> Obj_Machine:
    cur_Op = job.cur_Op
    compatible_Mj = env.PT_f[fac.id][job.id][job.cur_Op][0]
    compatible_machines = []
    for i in compatible_Mj:
        compatible_machines.append(env.Machines[fac.id][i - 1])
    PT = env.PT_f[fac.id][job.id][job.cur_Op][1]
    return compatible_machines[np.argmax(PT)]

def MinCTM(env: FJSPT_Env, job: Obj_Job, fac: Obj_Factory) -> Obj_Machine:
    compatible_Mj = env.PT_f[fac.id][job.id][job.cur_Op][0]
    compatible_machines = []
    for i in compatible_Mj:
        compatible_machines.append(env.Machines[fac.id][i - 1])
    CT_k = []
    for machine in compatible_machines:
        CT_k.append(machine.CT_k)
    return compatible_machines[np.argmin(CT_k)]



# ###################################################AGV调度规则###################################################
# 随机选择AGV
def random_AGV(env: FJSPT_Env, job: Obj_Job, fac: Obj_Factory) -> Obj_AGV:
    AGV_num = env.AGV_num
    return env.AGVs[random.randint(0, AGV_num - 1)]


# Shortest Path，将任务分配给距离目标最近的AGV，以最小化行驶距离和时间。适用于减少能源消耗和提高效率的情况。
# 或者说是STT shortest travel time，选择距离目的地最近的AGV执行运输任务
def STT(env: FJSPT_Env, job: Obj_Job, fac: Obj_Factory) -> Obj_AGV:
    job_pos = job.cur_pos
    travel_matrix = env.TT
    AGVs = env.AGVs
    travel_time = []
    for AGV in AGVs:
        AGV_pos = AGV[fac.id].cur_pos
        travel_time.append(travel_matrix[AGV_pos][job_pos])
    return env.AGVs[fac.id][np.argmin(travel_time)]


# LUV, lowest utilization of vehicle， 选择利用率最低的AGV执行运输任务
def LUV(env: FJSPT_Env, job: Obj_Job, fac: Obj_Factory) -> Obj_AGV:
    AGVs = env.AGVs[fac.id]
    utili = []
    for AGV in AGVs:
        utili.append(AGV.UA_l)
    return env.AGVs[fac.id][np.argmin(utili)]


# HUV, highest utilization of vehicle， 选择利用率最高的AGV执行运输任务
def HUV(env: FJSPT_Env, job: Obj_Job, fac: Obj_Factory) -> Obj_AGV:
    AGVs = env.AGVs[fac.id]
    utili = []
    for AGV in AGVs:
        utili.append(AGV.UA_l)
    return env.AGVs[fac.id][np.argmax(utili)]\

# MaxCTV, Maximum CT， 选择利用率最高的AGV执行运输任务
def MaxCTV(env: FJSPT_Env, job: Obj_Job, fac: Obj_Factory) -> Obj_AGV:
    AGVs = env.AGVs[fac.id]
    CT = []
    for AGV in AGVs:
        CT.append(AGV.CT_l)
    return env.AGVs[fac.id][np.argmax(CT)]


# MaxCTV, Minimum CT， 选择利用率最高的AGV执行运输任务
def MinCTV(env: FJSPT_Env, job: Obj_Job, fac: Obj_Factory) -> Obj_AGV:
    AGVs = env.AGVs[fac.id]
    CT = []
    for AGV in AGVs:
        CT.append(AGV.CT_l)
    return env.AGVs[fac.id][np.argmin(CT)]


# LIV, Lowest Idle time of vehicle， 选择空闲时间最长的AGV执行运输任务-->和AGV利用率差不多
def LIV(env: FJSPT_Env, job: Obj_Job, fac: Obj_Factory) -> Obj_AGV:
    AGVs = env.AGVs
    idle_time = []
    for AGV in AGVs:
        idle_time.append(sum(AGV[fac.id].load_end_time) - sum(AGV[fac.id].unload_start_time))
    return env.AGVs[fac.id][np.argmax(idle_time)]


# Earliest idle Vehicle  选择最早空闲的AGV
def EIV(env: FJSPT_Env, job: Obj_Job, fac: Obj_Factory) -> Obj_AGV:
    AGVs = env.AGVs
    CT_l = []
    for AGV in AGVs:
        CT_l.append(AGV[fac.id].CT_l)
    return env.AGVs[fac.id][np.argmin(CT_l)]


if __name__ == '__main__':
    a = 0
    path = '10J2F.txt'
    J_num, F_num, M_num, H, SH, NM, M, times, ProF, arrive_time, due_time, Op_dic, Op_num = DataReadDHFJSP(path)
    PT = process_data(M, times, F_num, J_num)
    AGV_num = 2
    TT = load_txt.load_travel_time('Dataset/DeroussiNorre/travel_time.txt')
    env = FJSPT_Env(2, J_num, M_num, Op_num, AGV_num, Op_dic, PT, TT, arrive_time, due_time)
    while a < 64:
        job, machine, AGV, factory = dispatch_rule(a, env)  # 修改这里的调用方式
        a += 1
        print(job.id, machine.id, AGV.id)

