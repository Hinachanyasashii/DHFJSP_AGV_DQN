import random
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import savgol_filter

from FJSPT.DATAREAD import DataReadDHFJSP, process_data
from FJSPT.FJSPT_Env import FJSPT_Env
from FJSPT.action_space import dispatch_rule
from FJSPT import load_txt
from FJSPT.object_FJSP import Obj_Job, Obj_AGV, Obj_Machine, gantt_chart, Obj_Factory


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

        self.target_model = self.model

        # Q-network Parameters
        self.gamma = 0.95  # Discount factor for future rewards
        self.global_step = 0
        self.update_target_steps = 50  # Update target network steps

        # Agent Parameters
        self.e_greedy = 0.3
        self.e_greedy_decrement = e_greedy_decrement

        # Replay Buffer
        self.buffer = deque(maxlen=2000)
        self.batch_size = 64

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def replace_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self):
        if self.global_step % self.update_target_steps == 0:
            self.replace_target()

        minibatch = random.sample(self.buffer, self.batch_size)
        # Convert minibatch data to PyTorch tensors
        state_batch = torch.tensor(np.array([sample[0].reshape(9) for sample in minibatch]), dtype=torch.float)
        action_batch = torch.tensor(np.array([sample[1] for sample in minibatch]), dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(np.array([sample[2] for sample in minibatch]), dtype=torch.float)
        next_state_batch = torch.tensor(np.array([sample[3].reshape(9) for sample in minibatch]), dtype=torch.float)
        done_batch = torch.tensor(np.array([sample[4] for sample in minibatch]), dtype=torch.bool)

        q_values = self.model(state_batch).gather(1, action_batch)
        next_q_values = self.target_model(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (~done_batch) * self.gamma * next_q_values

        loss = self.loss_func(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.global_step += 1

    def select_action(self, e, obs) -> (Obj_Job, Obj_Machine, Obj_AGV, Obj_Factory):
        """
        测试阶段，贪婪策略
        :param e:
        :param obs:
        :return:
        """
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
        TR = []
        min_total_reward = 400
        for e in range(self.episodes):
            Total_reward = 0
            obs = [0 for _ in range(input_size)]
            obs = np.expand_dims(obs, 0)
            done = False
            env = FJSPT_Env(F_num, J_num, M_num, Op_num, AGV_num, Op_dic, PT_f, TT, arrive_time, due_time)
            at_cnt = [0 for _ in range(output_size)]
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
                End.append(Jobs[Ji].CT_i)
            if (e + 1) % 50 == 0:
                print('-----------------------开始第', e + 1, '次训练------------------------------')
                print('<<<<<<<<<-----------------reward:', Total_reward, '------------------->>>>>>>>>>')
            TR.append(Total_reward)
            # if -Total_reward < 110:
            #     plt.bar([i for i in range(len(at_cnt))], at_cnt)
            plt.show()
            if -Total_reward < min_total_reward:
                for f in range(env.F_num):
                    gantt_chart(env.Machines[f], env.AGVs[f], True, [F_num, J_num, M_num, AGV_num, f])
                min_total_reward = -Total_reward
            # gantt_chart(env.Machines, env.AGVs, False)
            # print()
        filter_TR = savgol_filter(TR, 99, 1, mode="nearest")  # Savitzky-Golay滤波器
        plt.plot(TR)
        plt.plot(filter_TR)
        plt.axhline(y=max(TR), color='r')
        plt.savefig("Result/DeroussiNorre/" + "reward_{}_F{}J{}M{}A{}"\
                    .format(-max(TR), env.F_num, env.J_num, env.M_num, env.AGV_num) +
                    time.strftime('%Y%m%d_%H%M', time.localtime()) + ".png")
        plt.show()
        print("最优的奖励为: ", max(TR))
        print("e-greedy: ", self.e_greedy)
        # return Total_reward
        return -max(TR)


# Initialize DQN
input_size = 9  # Modify this according to your input dimension
output_size = 384  # 工件调度规则个数 * 机器调度规则个数 * AGV调度规则个数 + 6
episodes = 10000
e_greedy_decrement = 1e-6
learning_rate = 1e-3
d = DQN(input_size, output_size, episodes, e_greedy_decrement, learning_rate)

# Run main function
path = './Dataset/lirui/20J3F.txt'  # 算例
J_num, F_num, M_num, H, SH, NM, M, times, ProF, arrive_time, due_time, Op_dic, Op_num = DataReadDHFJSP(path)
PT_f = process_data(M, times, F_num, J_num)
AGV_num = 5

TT = load_txt.load_travel_time('Dataset/DeroussiNorre/travel_time.txt')  # 地图

d.model.load_state_dict(torch.load("Result/pth/Q_network_F3J20M5A2_0314_1456.pth"))

makespan = d.main(F_num, J_num, M_num, Op_num, AGV_num, Op_dic, PT_f, TT, output_size, arrive_time, due_time)

# torch.save(d.model.state_dict(), "Result/pth/Q_network_" + "F{}J{}M{}A{}".format(F_num, J_num, M_num, AGV_num) +
#            "_{}.pth".format(time.strftime('%m%d_%H%M', time.localtime())))
