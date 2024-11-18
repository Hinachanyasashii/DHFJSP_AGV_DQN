import os

import numpy as np


def instance_generator(n_j, n_f):

    n_m = 5
    op_per_job = 5
    n_op = n_j * op_per_job  # 总工序数
    op_per_mch_min = 1  # 工件兼容机器数
    op_per_mch_max = n_m
    low = 5  # 加工时间上下界
    high = 20

    dataset_op_pt = []
    for i in range(n_f):

        op_use_mch = np.random.randint(low=op_per_mch_min, high=op_per_mch_max + 1, size=n_op)  # 兼容机器数，左含右不含
        op_pt = np.random.randint(low=low, high=high + 1, size=(n_op, n_m))  # 加工时间矩阵

        # 随机设置不兼容机器加工时间为0
        for row in range(op_pt.shape[0]):
            mch_num = int(op_use_mch[row])
            if mch_num < n_m:
                inf_pos = np.random.choice(np.arange(0, n_m), n_m - mch_num, replace=False)  # 随机选择n_m - mch_num个机器索引
                op_pt[row][inf_pos] = 0
        dataset_op_pt.append(op_pt)

    return dataset_op_pt


def matrix_to_text(n_j, n_f, data):

    n_m = 5
    op_per_job = 5
    text = [f'{n_j} {n_f} {n_m}']
    # data = instance_generator(n_j, n_f)

    for i in range(n_f):
        op_idx = 0
        op_pt = data[i]
        for j in range(n_j):
            text.append(f'{i+1} {j+1} {n_m}')
            for x in range(op_per_job):
                line = f'{x+1}'
                use_mch = np.where(op_pt[op_idx] != 0)[0]  # np.where返回一个包含数组的元组，[0]取元组的第一个元素，即机器索引的数组
                line = line + ' ' + str(use_mch.shape[0])  # 提取每个工序可使用机器数
                for k in use_mch:
                    line = line + ' ' + str(k + 1) + ' ' + str(op_pt[op_idx][k])  # 提取机器索引和加工时间
                text.append(line)
                op_idx += 1
            text.append("")
    return text


def generate_data_to_files(n_j, n_f, cover_data_flag):

    directory = f'./Dataset/'
    filename = f'{n_j}J{n_f}F.txt'
    full_path = os.path.join(directory, filename)

    if not os.path.exists(full_path) or cover_data_flag:
        dataset_op_pt = instance_generator(n_j, n_f)
        text = matrix_to_text(n_j, n_f, dataset_op_pt)

        doc = open(full_path, 'w')
        for i in range(len(text)):
            print(text[i], file=doc)
        doc.close()
    else:
        print("the data already exists...")


# def generate_data_to_files(n_j, n_f):
#
#     path = f'./Dataset/'
#     filename = f'{n_j}J{n_f}F'
#
#     dataset_op_pt = instance_generator(n_j, n_f)
#     text = matrix_to_text(n_j, n_f, dataset_op_pt)
#
#     doc = open(path + filename + '.txt', 'w')
#     for i in range(len(text)):
#         print(text[i], file=doc)
#     doc.close()


if __name__ == '__main__':
    job = 80
    factory = 6
    generate_data_to_files(job, factory, False)

    # dataset_op_pt = instance_generator(job, factory)
    # print(dataset_op_pt)
    # print(matrix_to_text(job, factory))
