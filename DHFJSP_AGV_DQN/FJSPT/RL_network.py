"""
@Project ：Master_thesis
@File    ：RL_network.py
@IDE     ：PyCharm 
@Author  ：lyon
@Date    ：10/10/2023 10:25 
@Des     ：
"""

import random
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import savgol_filter

from DATAREAD import DataReadDHFJSP, process_data
from FJSPT_Env import FJSPT_Env
from action_space import dispatch_rule
import load_txt
from object_FJSP import Obj_Job, Obj_AGV, Obj_Machine, gantt_chart, Obj_Factory


class DQN:
    def __init__(self, input_size, output_size, episodes, e_greedy_decrement, learning_rate):
        self.Hid_Size = 30
        self.input_size = input_size
        self.output_size = output_size
        self.episodes = episodes

        # Define Q-network model
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.Hid_Size),
            nn.ReLU(),
            nn.Linear(self.Hid_Size, self.Hid_Size),
            nn.ReLU(),
            nn.Linear(self.Hid_Size, self.Hid_Size),
            nn.ReLU(),
            nn.Linear(self.Hid_Size, self.Hid_Size),
            nn.ReLU(),
            nn.Linear(self.Hid_Size, self.Hid_Size),
            nn.ReLU(),
            nn.Linear(self.Hid_Size, self.output_size)
        )

        # Target network(DDQN)
        self.target_model = nn.Sequential(
            nn.Linear(self.input_size, self.Hid_Size),
            nn.ReLU(),
            nn.Linear(self.Hid_Size, self.Hid_Size),
            nn.ReLU(),
            nn.Linear(self.Hid_Size, self.Hid_Size),
            nn.ReLU(),
            nn.Linear(self.Hid_Size, self.Hid_Size),
            nn.ReLU(),
            nn.Linear(self.Hid_Size, self.Hid_Size),
            nn.ReLU(),
            nn.Linear(self.Hid_Size, self.output_size)
        )
        self.replace_target()  # 初始化时同步目标网络的权重

        # Q-network Parameters
        self.gamma = 0.95  # Discount factor for future rewards
        self.global_step = 0
        self.update_target_steps = 50  # Update target network steps

        # Agent Parameters
        self.e_greedy = 1
        self.e_greedy_decrement = e_greedy_decrement

        # Replay Buffer
        self.buffer = deque(maxlen=2000)
        self.batch_size = 64

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def replace_target(self):  # 更新目标网络参数
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self):
        if self.global_step % self.update_target_steps == 0:
            self.replace_target()

        if len(self.buffer) < self.batch_size:
            return  # 如果缓冲区内的样本数量小于 batch_size，则返回，不执行更新

        minibatch = random.sample(self.buffer, self.batch_size)
        # Convert minibatch data to PyTorch tensors
        state_batch = torch.tensor(np.array([sample[0].reshape(9) for sample in minibatch]), dtype=torch.float)
        action_batch = torch.tensor(np.array([sample[1] for sample in minibatch]), dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(np.array([sample[2] for sample in minibatch]), dtype=torch.float)
        next_state_batch = torch.tensor(np.array([sample[3].reshape(9) for sample in minibatch]), dtype=torch.float)
        done_batch = torch.tensor(np.array([sample[4] for sample in minibatch]), dtype=torch.bool)

        q_values = self.model(state_batch).gather(1, action_batch)

        # Double DQN: main model selects action, target model evaluates Q-value of that action
        next_action_batch = self.model(next_state_batch).argmax(dim=1).unsqueeze(1)
        next_q_values = self.target_model(next_state_batch).gather(1, next_action_batch).squeeze()
        expected_q_values = reward_batch + (~done_batch) * self.gamma * next_q_values

        loss = self.loss_func(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.global_step += 1

    def select_action(self, e, obs) -> (Obj_Job, Obj_Machine, Obj_AGV, Obj_Factory):
        if random.random() < self.e_greedy:
            a_t = random.randint(0, self.output_size - 1)
        else:
            a_t = np.argmax(self.model(torch.tensor(obs, dtype=torch.float32)).detach().numpy())
        self.e_greedy = max(0.01, self.e_greedy - 1.1 * e / self.episodes)
        return a_t

    def _append(self, exp):
        self.buffer.append(exp)

    def main(self, F_num, J_num, M_num, Op_num, AGV_num, Op_dic, PT_f, TT, output_size, arrive_time, due_time):
        k = 0
        TR = []  # 记录每个训练回合（episode）的奖励
        min_total_reward = 100  # 初始最小奖励
        train_st = time.time()
        print(f'current instance: {J_num}J{F_num}F{AGV_num}A')
        for e in range(self.episodes):
            ep_st = time.time()
            Total_reward = 0
            obs = [0 for _ in range(input_size)]
            obs = np.expand_dims(obs, 0)
            done = False
            env = FJSPT_Env(F_num, J_num, M_num, Op_num, AGV_num, Op_dic, PT_f, TT, arrive_time, due_time)
            at_cnt = [0 for _ in range(output_size)]  # 每个episode动作计数
            for i in range(Op_num):
                k += 1
                at = self.select_action(e, obs)
                at_cnt[at] += 1
                at_trans = dispatch_rule(at, env)
                # print('这是第', i + 1, '道工序->>', '执行action:', at, ' ', '将工件', at_trans[0].id, '安排到机器',
                #       at_trans[1].id, '使用AGV',  at_trans[2].id, '进行搬运')
                env.scheduling(at_trans)
                obs_t = env.states()
                if i == Op_num - 1:
                    done = True
                obs_t = np.expand_dims(obs_t, 0)
                # todo 定义奖励函数 r_t = env.reward(obs[0][6], obs[0][5], obs_t[0][6], obs_t[0][5], obs[0][0], obs_t[0][0])
                r_t = env.reward(1, 1, 1, 1, 1, 1)
                self._append((obs, at, r_t, obs_t, done))
                if k > self.batch_size:
                    self.replay()
                Total_reward += r_t
                obs = obs_t
            Jobs = env.Jobs
            End = []
            for Ji in range(len(Jobs)):
                End.append(Jobs[Ji].CT_i)  # 记录每个工件的完工时间
            ep_et = time.time()
            if (e + 1) % 50 == 0:
                print(f'Episode {e+1}\t reward: {Total_reward:.2f}\t  training time: {ep_et - ep_st:.2f}')
                # print('-----------------------开始第', e + 1, '次训练------------------------------')
                # print('<<<<<<<<<-------------reward:', Total_reward, '--------------->>>>>>>>>>')
            TR.append(Total_reward)
            # if -Total_reward < 110:
            #     plt.bar([i for i in range(len(at_cnt))], at_cnt)
            # plt.show()

            # 如果当前episode的总奖励是最小的生成甘特图
            if -Total_reward < min_total_reward:
                for f in range(env.F_num):
                    gantt_chart(env.Machines[f], env.AGVs[f], False, [F_num, J_num, M_num, AGV_num, f])
                min_total_reward = -Total_reward
                # torch.save(d.model.state_dict(),
                #            "Result/pth/Q_network_" + "F{}J{}M{}A{}.pth".format(F_num, J_num, M_num, AGV_num))
            # gantt_chart(env.Machines, env.AGVs, False)

        # 平滑奖励曲线（所有episode）
        filter_TR = savgol_filter(TR, 99, 1, mode="nearest")  # Savitzky-Golay滤波器
        plt.plot(TR)
        plt.plot(filter_TR)
        plt.axhline(y=max(TR), color='r')
        plt.savefig("Result/Revision/reward/" + "reward_{}_F{}J{}M{}A{}"\
                    .format(-max(TR), env.F_num, env.J_num, env.M_num, env.AGV_num) +
                    time.strftime('%Y%m%d_%H%M', time.localtime()) + ".png")
        plt.show()
        print("最优的奖励为: ", max(TR))
        print("e-greedy: ", self.e_greedy)

        # 将 TR 转换为 DataFrame，每个值为一行
        df = pd.DataFrame([abs(x) for x in TR], columns=[f'{J_num}J{F_num}F{AGV_num}A'])
        # 保存到 Excel 文件
        df.to_excel(f"Result/{J_num}J{F_num}F{AGV_num}A.xlsx", index=False)  # `index=False` 去掉行索引
        # return Total_reward
        train_et = time.time()
        print((train_et-train_st)/self.episodes)

        return -max(TR)


# Initialize DQN
input_size = 9  # Modify this according to your input dimension
output_size = 384  # 工件调度规则个数 * 机器调度规则个数 * AGV调度规则个数
episodes = 10000
e_greedy_decrement = 1e-6
learning_rate = 1e-3
d = DQN(input_size, output_size, episodes, e_greedy_decrement, learning_rate)

# Run main function
path = 'Dataset/lirui/10J2F.txt'
J_num, F_num, M_num, H, SH, NM, M, times, ProF, arrive_time, due_time, Op_dic, Op_num = DataReadDHFJSP(path)
PT_f = process_data(M, times, F_num, J_num)
# 将数据导出
# for idf, f in enumerate(PT_f):
#     tt1 = []
#     tt2 = []
#     for id, j in enumerate(f):
#         print("工件号：{}".format(id))
#         for o in j:
#             t1 = ""
#             t2 = ""
#             for i in list(map(int, o[0])):
#                 t1 += str(i) + ","
#             for i in list(map(int, o[1])):
#                 t2 += str(i) + ","
#             tt1.append(t1)
#             tt2.append(t2)
#     tt = []
#     for i in range(len(tt1)):
#         tt.append([tt1[i][:-1], tt2[i][:-1]])
#     data2 = pd.DataFrame(tt, columns=['A', 'B'])
#     data2.to_csv(str(idf) + ".csv")

AGV_num = 2
TT = load_txt.load_travel_time('Dataset/DeroussiNorre/travel_time.txt')
# TT = load_txt.load_travel_time('Dataset/DeroussiNorre/自定义数据集运输时间.txt')
makespan = d.main(F_num, J_num, M_num, Op_num, AGV_num, Op_dic, PT_f, TT, output_size, arrive_time, due_time)

# torch.save(d.model.state_dict(), "Result/pth/Q_network_" + "F{}J{}M{}A{}".format(F_num, J_num, M_num, AGV_num) +
#            "_{}.pth".format(time.strftime('%m%d_%H%M', time.localtime())))
