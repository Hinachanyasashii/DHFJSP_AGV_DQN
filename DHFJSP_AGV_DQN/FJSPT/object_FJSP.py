"""
@Project ：FJSP-Obj_AGV
@File    ：object_FJSP.py
@IDE     ：PyCharm 
@Author  ：lyon
@Date    ：08/10/2023 14:26 
@Des     ：
"""

import time

import matplotlib.pyplot as plt


def gantt_chart(Machines, AGVs, save_picture, num=None):
    [F_num, J_num, M_num, AGV_num, f] = num
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    M = ['#0984e3', '#00cec9', '#ffeaa7', '#81ecec', '#6c5ce7', '#fd79a8',
         '#74b9ff', '#a29bfe', '#e17055', '#fab1a0', '#55efc4', '#fdcb6e', '#00b894',
         '#6ab04c', '#f7dc6f', '#3498db', '#8e44ad', '#2ecc71',
         '#2980b9', '#c0392b', '#9b59b6', '#e74c3c', '#1abc9c', '#34495e', '#f1c40f', '#7f8c8d',
         '#16a085', '#d35400', '#27ae60', '#f39c12', '#8e44ad', '#c0392b', '#bdc3c7',
         '#7f8c8d', '#1abc9c', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
         '#2980b9', '#c0392b', '#9b59b6', '#e74c3c', '#1abc9c', '#34495e', '#f1c40f', '#7f8c8d',
         '#16a085', '#d35400', '#27ae60', '#f39c12', '#8e44ad', '#c0392b', '#bdc3c7',
         '#7f8c8d', '#1abc9c', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
         '#2980b9', '#c0392b', '#9b59b6', '#e74c3c', '#1abc9c', '#34495e', '#f1c40f', '#7f8c8d',
         '#16a085', '#d35400', '#27ae60', '#f39c12', '#8e44ad', '#c0392b', '#bdc3c7',
         '#7f8c8d', '#1abc9c', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
         '#2980b9', '#c0392b', '#9b59b6', '#e74c3c', '#1abc9c', '#34495e', '#f1c40f', '#7f8c8d',
         '#16a085', '#d35400', '#27ae60', '#f39c12', '#8e44ad', '#c0392b', '#bdc3c7',
         '#7f8c8d', '#1abc9c', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
         '#2980b9', '#c0392b', '#9b59b6', '#e74c3c', '#1abc9c', '#34495e', '#f1c40f', '#7f8c8d',
         '#16a085', '#d35400', '#27ae60', '#f39c12', '#8e44ad', '#c0392b', '#bdc3c7',
         '#7f8c8d', '#1abc9c', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
         '#2980b9', '#c0392b', '#9b59b6', '#e74c3c', '#1abc9c', '#34495e', '#f1c40f', '#7f8c8d',
         '#16a085', '#d35400', '#27ae60', '#f39c12', '#8e44ad', '#c0392b', '#bdc3c7',
         '#7f8c8d', '#1abc9c', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
         '#2980b9', '#c0392b', '#9b59b6', '#e74c3c', '#1abc9c', '#34495e', '#f1c40f', '#7f8c8d',
         '#16a085', '#d35400', '#27ae60', '#f39c12', '#8e44ad', '#c0392b', '#bdc3c7',
         '#7f8c8d', '#1abc9c', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
         '#2980b9', '#c0392b', '#9b59b6', '#e74c3c', '#1abc9c', '#34495e', '#f1c40f', '#7f8c8d',
         '#16a085', '#d35400', '#27ae60', '#f39c12', '#8e44ad', '#c0392b', '#bdc3c7',
         '#7f8c8d', '#1abc9c', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
         '#2980b9', '#c0392b', '#9b59b6', '#e74c3c', '#1abc9c', '#34495e', '#f1c40f', '#7f8c8d',
         '#16a085', '#d35400', '#27ae60', '#f39c12', '#8e44ad', '#c0392b', '#bdc3c7',
         '#7f8c8d', '#1abc9c', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
         '#2980b9', '#c0392b', '#9b59b6', '#e74c3c', '#1abc9c', '#34495e', '#f1c40f', '#7f8c8d',
         '#16a085', '#d35400', '#27ae60', '#f39c12', '#8e44ad', '#c0392b', '#bdc3c7',
         '#7f8c8d', '#1abc9c', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
         '#2980b9', '#c0392b', '#9b59b6', '#e74c3c', '#1abc9c', '#34495e', '#f1c40f', '#7f8c8d',
         '#16a085', '#d35400', '#27ae60', '#f39c12', '#8e44ad', '#c0392b', '#bdc3c7',
         '#7f8c8d', '#1abc9c', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
         '#2980b9', '#c0392b', '#9b59b6', '#e74c3c', '#1abc9c', '#34495e', '#f1c40f', '#7f8c8d',
         '#16a085', '#d35400', '#27ae60', '#f39c12', '#8e44ad', '#c0392b', '#bdc3c7',
         '#7f8c8d', '#1abc9c', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
         '#2980b9', '#c0392b', '#9b59b6', '#e74c3c', '#1abc9c', '#34495e', '#f1c40f', '#7f8c8d',
         '#16a085', '#d35400', '#27ae60', '#f39c12', '#8e44ad', '#c0392b', '#bdc3c7',
         '#7f8c8d', '#1abc9c', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff']

    Job_text = ['J' + str(i + 1) for i in range(J_num)]
    AGV_text = ['A' + str(i + 1) for i in range(AGV_num)]
    Machine_text = ['M' + str(i + 1) for i in range(M_num)]
    plt.subplots_adjust(top=0.6)  # 调整上边距

    for i in range(len(Machines)):
        for j in range(len(Machines[i].start_time)):
            if Machines[i].end_time[j] - Machines[i].start_time[j] != 0:
                plt.barh(i + 1, width=Machines[i].end_time[j] - Machines[i].start_time[j],
                         height=0.8, left=Machines[i].start_time[j],
                         color=M[Machines[i]._on[j]],
                         edgecolor='black', linewidth=0.5)

    for i in range(len(AGVs)):
        for j in range(len(AGVs[i].load_start_time)):
            if AGVs[i].load_end_time[j] - AGVs[i].load_start_time[j] != 0:
                # AGV负载时候的甘特图
                plt.barh(len(Machines) + i + 1, width=AGVs[i].load_end_time[j] - AGVs[i].load_start_time[j],
                         height=0.8, left=AGVs[i].load_start_time[j],
                         color=M[AGVs[i]._on[j]],
                         edgecolor='black', linewidth=0.5)
                # AGV空载时候的甘特图
                plt.barh(len(Machines) + i + 1, width=AGVs[i].load_start_time[j] - AGVs[i].unload_start_time[j],
                         height=0.8, left=AGVs[i].unload_start_time[j],
                         color="white",
                         edgecolor='black', linewidth=0.5)

    y_ticks = ["0"] + Machine_text[: len(Machines)] + AGV_text[: len(AGVs)]
    y = range(len(y_ticks))
    plt.yticks(y, y_ticks)

    makespans = []
    for m in Machines:
        makespans.append(m.CT_k)
    title = "F{}J{}M{}AGV{}f{}-makespan{}".format(F_num, J_num, M_num, AGV_num, f, max(makespans))
    # plt.axvline(x=max(makespans), color='r')
    plt.suptitle(title)
    plt.xlim(0, None)

    existing_jobs = sorted(
        set(Machines[i]._on[j] for i in range(len(Machines)) for j in range(len(Machines[i].start_time))))

    # 创建图例文本
    existing_jobs_text = [Job_text[j] for j in existing_jobs]
    # 使用已经存在的颜色创建图例标记
    handles = [plt.Rectangle((0, 0), 1, 1, color=M[j]) for j in existing_jobs]
    # 添加图例，并指定图例标记
    plt.legend(handles, existing_jobs_text, loc='upper center', bbox_to_anchor=(0.5, 1.75), ncol=5)

    if save_picture:
        plt.savefig("Result/Revision/gantt/" + title + "_" + time.strftime('%Y%m%d_%H%M', time.localtime()) + ".png")
    plt.show()






class Obj_Job:
    def __init__(self, id, NO_i, AT_i, DT_i,factory_id):
        self.factory_id=factory_id
        self.id = id
        self.start_time = []
        self.end_time = []
        self.cur_Op = 0  # 加工到第几个工序
        self.assign_for = []
        self.cur_pos = 0  # 当前工件所在位置，0表示在LU
        self.next_pos = 0  # 当前工件所在位置，0表示在LU
        self.NO_i = NO_i  # 工件i工序的数量
        self.CT_i = AT_i  # 工件i的最大完工时间，初始的时候等于工件到达时间
        self.CRJ = 0  # 工件完成率
        self.UJ_i = 0
        self.traj = [0]  # 工件运动轨迹
        self.AT_i = AT_i  # 工件到达时间
        self.DT_i = DT_i  # 交货期

    def _update(self, start, end, cur_pos):
        self.start_time.append(start)
        self.end_time.append(end)
        self.cur_pos = cur_pos

        self.CT_i = max(self.end_time)
        self.UJ_i = (sum(self.end_time) - sum(self.start_time)) / self.CT_i
        self.cur_Op += 1  # 加工到第几个工序
        self.CRJ = self.cur_Op / self.NO_i
        self.traj.append(self.cur_pos)

    def idle_time(self):
        idle = []
        try:
            if self.start_time[0] != 0:
                idle.append([0, self.start_time[0]])
            K = [[self.end_time[i], self.start_time[i + 1]] for i in range(len(self.end_time)) if
                 self.start_time[i + 1] - self.end_time[i] > 0]
            idle.extend(K)
        except:
            pass
        return idle


class Obj_Machine:
    def __init__(self, id):
        self.id = id
        self.start_time = []
        self.end_time = []
        self._on = []  # 在加工哪个工件
        self.CT_k = 0
        self.UM_k = 0
        self.job_buffer = []  # 缓冲区
        self.buffer_capacity = 10

    def add_to_buffer(self, job):
        if len(self.job_buffer) < self.buffer_capacity:
            self.job_buffer.append(job)

    def remove_from_buffer(self, idx):
        if len(self.job_buffer) > 0:
            return self.job_buffer.pop(idx)

    def _update(self, start, end, job_id):
        self.start_time.append(start)
        self.end_time.append(end)
        self.start_time.sort()
        self.end_time.sort()
        self._on.insert(self.start_time.index(start), job_id)

        self.CT_k = max(self.end_time)
        self.UM_k = (sum(self.end_time) - sum(self.start_time)) / self.CT_k

    def job_start_time(self, PT, AT):
        """

        :param PT: 加工时间
        :param AT: 工件被AGV运输到达机器上的时间
        :return: 工件在该机器上加工最早开始的时间
        """
        start = max(AT, self.CT_k)
        Gaps = []
        if self.start_time:
            if self.start_time[0] > 0 and self.start_time[0] > AT:
                Gaps.append([0, self.start_time[0]])
            if len(self.start_time) > 1:
                Gaps.extend([[self.end_time[i], self.start_time[i + 1]] for i in range(0, len(self.start_time) - 1) if
                             self.start_time[i + 1] - self.end_time[i] > 0 and self.start_time[i + 1] > AT])
        if Gaps:
            for gap in Gaps:
                if gap[0] >= AT and gap[1] - gap[0] >= PT:
                    return gap[0]
                elif gap[0] < AT and gap[1] - AT >= PT:
                    return AT
        return start

    def idle_time(self):
        idle = []
        try:
            if self.start_time[0] != 0:
                idle.append([0, self.start_time[0]])
            K = [[self.end_time[i], self.start_time[i + 1]] for i in range(len(self.end_time)) if
                 self.start_time[i + 1] - self.end_time[i] > 0]
            idle.extend(K)
        except:
            pass
        return idle


class Obj_AGV:
    def __init__(self, id):
        self.id = id
        self.current_job = None  # 当前装载的工件，None表示没有装载任何工件
        # 小车开始、结束时间和小车所在位置
        self.unload_start_time = []
        self.load_start_time = []
        self.load_end_time = []
        self.start_location = []
        self.end_location = []
        self.agv_process_record = []
        self._on = []
        self.cur_pos = 0
        self.CT_l = 0  # 该AGV上的最大完工时间
        self.UA_l = 0
        self.traj = []  # 开始在LU

    def _update(self, unload_start_time, unload_time, load_time, job: Obj_Job):
        self.unload_start_time.append(unload_start_time)
        self.load_start_time.append(unload_start_time + unload_time)
        self.load_end_time.append(unload_start_time + unload_time + load_time)
        self.load_end_time.sort()
        self.load_start_time.sort()
        self.unload_start_time.sort()
        self._on.insert(self.load_start_time.index(unload_start_time + unload_time), job.id)
        # 记录小车的搬运情况
        self.agv_process_record.append([unload_time, unload_start_time + unload_time + load_time,
                                        job.cur_pos, job.next_pos, [job.id, job.cur_Op]])
        # 更新小车的位置
        self.cur_pos = job.next_pos
        self.traj.append(job.cur_pos)
        self.traj.append(job.next_pos)

        self.CT_l = max(self.load_end_time)
        self.UA_l = (sum(self.load_end_time) - sum(self.unload_start_time)) / self.CT_l

    def transport_start_time(self, job: Obj_Job, unload_time, load_time):
        start = max(job.CT_i - unload_time, self.CT_l)
        # Gaps = []
        # if len(self.unload_start_time) > 1:
        #     Gaps.extend([[self.load_end_time[i], self.unload_start_time[i + 1]] for i in
        #                  range(0, len(self.unload_start_time) - 1)
        #                  if self.load_end_time[i + 1] - self.load_end_time[i] > 0 and self.unload_start_time[
        #                      i + 1] > job.CT_i])
        # if Gaps:
        #     for gap in Gaps:
        #         if gap[0] >= job.CT_i and gap[1] - gap[0] >= TT:
        #             # fixme TT应该还要加上 AGV回到原来位置的时间
        #             return gap[0]
        #         elif gap[0] < job.CT_i and gap[1] - job.CT_i > TT:
        #             return job.CT_i
        return start

    def idle_time(self):
        idle = []
        try:
            if self.load_start_time[0] != 0:
                idle.append([0, self.load_start_time[0]])
            K = [[self.load_end_time[i], self.load_start_time[i + 1]] for i in range(len(self.load_end_time)) if
                 self.load_start_time[i + 1] - self.load_end_time[i] > 0]
            idle.extend(K)
        except:
            pass
        return idle



class Obj_Factory:
    def __init__(self, id):
        self.id = id
        self._on = []  # 在加工哪个工件
        self.CT_f = 0
        self.J_num_f = 0  # 工厂f中工件的个数

    # todo 定义工厂相关的函数，属性
    def _update(self, job):
        self._on.append(job)
        ct = []
        for j in self._on:
            ct.append(j.CT_i)
        self.CT_f = max(ct)
        self.J_num_f += 1
