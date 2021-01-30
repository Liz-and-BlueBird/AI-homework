import pygame
import random
import os
from PIL import Image
from collections import namedtuple
import torch
import torchvision
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Gomoku:
    def __init__(self):
        self.space_size = 60  # 棋盘周围的边距
        self.cell_size = 40  # 方格大小
        self.cell_num = 15  # 每行方格数目+1，可落子个数
        self.board_size = self.cell_size * (self.cell_num - 1) + self.space_size * 2  # 棋盘大小15x15，方格数14x14
        self.chess_arr = []  # 存储棋子坐标(x，y，flag)，x：0..14，y：0..14，flag：1黑子（先手，玩家），2白子（后手，电脑）
        self.game_state = 0  # 游戏状态：0正常进行，1黑子胜，2白子胜
        self.screen = None  # 声明一个变量，用于init函数中存储Surface类型值
        self.computer = None  # 电脑

    def draw(self):
        """
        绘制棋盘
        绘制棋子
        产生胜负时绘制胜负信息
        """

        def draw_board():
            """
            绘制棋盘
            """
            for x in range(0, self.cell_size * self.cell_num, self.cell_size):
                pygame.draw.line(self.screen, (200, 200, 200), (x + self.space_size, self.space_size),
                                 (x + self.space_size, self.cell_size * (self.cell_num - 1) + self.space_size), 1)
            for y in range(0, self.cell_size * self.cell_num, self.cell_size):
                pygame.draw.line(self.screen, (200, 200, 200), (self.space_size, y + self.space_size),
                                 (self.cell_size * (self.cell_num - 1) + self.space_size, y + self.space_size), 1)

        def draw_chess():
            """
            绘制棋子
            """
            for x, y, chess_type in self.chess_arr:
                chess_color = (30, 30, 30) if chess_type == 1 else (225, 225, 225)
                pygame.draw.circle(self.screen, chess_color,
                                   (x * self.cell_size + self.space_size, y * self.cell_size + self.space_size),
                                   16,  # 半径：16
                                   16)

        def draw_winner():
            """
            产生胜负时绘制胜负信息
            """
            if self.game_state > 0:
                my_font = pygame.font.Font(None, 60)
                win_text = '%s win' % ('black' if self.game_state == 1 else 'white')
                text_Image = my_font.render(win_text, True, (210, 210, 0))
                self.screen.blit(text_Image, (260, 320))

        draw_board()
        draw_chess()
        draw_winner()
        # 更新显示
        pygame.display.update()

    def check_state(self):
        """
        判断是否产生胜负
        :return: 游戏状态：0继续，1黑子胜，2白子胜
        """
        # 对落下的每个棋子
        for x, y, flag in self.chess_arr:
            # 依次判断左右对角线、竖直和水平方向
            for m, n in [(1, 1), (-1, 1), (0, 1), (1, 0)]:
                # 保存方向上落子情况，例如：0，0，1，1，（1），1，1，0，0
                check_arr = []
                for i in range(-4, 5):
                    if (x + m * i, y + n * i, flag) in self.chess_arr:
                        check_arr.append(1)
                    else:
                        check_arr.append(0)
                # 形成五子连线时，紧接着的5个元素为1
                for i in range(0, 5):
                    if sum(check_arr[i:i + 5]) == 5:  # 和为5
                        return flag  # 返回获胜方
        return 0  # 游戏继续

    def save_current_image(self, name):
        """
        截图，用于识别棋局落子情况
        :param name: player or computer
        """
        # 不存在则创建tmp目录
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        pygame.image.save(self.screen, 'tmp\\%s.png' % name)  # 保存为name.png

    def save_dataset_image(self, set_name):
        """
        保存图片数据
        :param set_name: train，val，test
        """

        def check_dir():
            """
            检查路径
            :return: 返回最大索引
            """
            if not os.path.exists(set_name):
                os.makedirs(set_name)
            if not os.path.exists(os.path.join(set_name, 'image')):
                os.makedirs(os.path.join(set_name, 'image'))
            if not os.path.exists(os.path.join(set_name, 'label')):
                os.makedirs(os.path.join(set_name, 'label'))
            return len(os.listdir(os.path.join(set_name, 'label')))

        def save_image():
            """
            截图，用于识别棋局落子情况
            """
            pygame.image.save(self.screen, '%s\\image\\%d.png' % (set_name, idx))  # 保存为idx.png

        def save_label():
            """
            保存棋局落子情况
            """
            # 构造标签
            label = np.zeros((15, 15), dtype=float)  # 15x15的二维数组，float类型
            for chess in self.chess_arr:
                label[chess[1]][chess[0]] = chess[2]  # 落子位置标注棋子类型
            # 写入文件
            np.save('%s\\label\\%d' % (set_name, idx), label)  # 保存为x.npz

        idx = check_dir() + 1
        save_image()
        save_label()

    def get_dataset(self, image_count=10, chess_count=100, set_name='train'):
        """
        电脑之间进行图形游戏，中途进行截图
        :param image_count: 保存图片数
        :param chess_count: 黑子最大个数，用于控制棋盘上棋子数目
        :param set_name: train，val，test
        """
        self.load_computer()  # 载入电脑
        count = 0  # 记录黑子个数
        self.init()  # 窗口初始化
        while True:
            # 事件监听
            for event in pygame.event.get():
                # 退出
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            # 黑子
            if self.game_state == 0:  # 可以落子
                self.chess_arr.append(self.computer(mode=0, flag=1, chess_arr=self.chess_arr))
                self.draw()
                # 棋盘满
                if len(self.chess_arr) == 225:
                    count = 0
                    self.restart()
                    continue
                count += 1  # 黑子个数增1
                if count == random.randint(1, chess_count):  # 黑子个数控制在1..chess_count间
                    if self.game_state == 0:  # 黑子未获胜
                        self.save_dataset_image(set_name)
                        image_count -= 1
                        if image_count == 0:  # 到达指定数目，退出程序
                            pygame.quit()
                            exit()
                    count = 0
                    self.restart()
                    continue
                if count > chess_count:  # 黑子个数越界，重置游戏
                    count = 0
                    self.restart()
                    continue
            else:  # 白子胜，重置游戏
                count = 0
                self.restart()
                continue
            # 白子
            if self.game_state == 0:  # 可以落子
                self.chess_arr.append(self.computer(mode=0, flag=2, chess_arr=self.chess_arr))
                self.draw()
                # 棋盘满
                if len(self.chess_arr) == 225:
                    count = 0
                    self.restart()
                    continue
            else:  # 黑子胜，重置游戏
                count = 0
                self.restart()
                continue

    def set_arr(self, arr):
        """
        设置落子信息
        :param arr: 保存有落子信息的列表
        """
        self.chess_arr = list(arr)

    def show_arr(self):
        """
        单纯绘制棋盘显示棋子信息
        :return: True, 显示下一个
        """
        self.init()
        # 游戏状态非正常进行
        self.game_state = -1
        # 监听事件
        while True:
            for event in pygame.event.get():
                # 退出
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                # 按下空格键显示下一个
                if (event.type == pygame.KEYDOWN) and (event.key == pygame.K_SPACE):
                    return True

    def init(self):
        """
        图形窗口初始化
        """
        pygame.init()  # 初始化
        pygame.display.set_caption('五子棋')  # 窗口说明：五子棋
        self.screen = pygame.display.set_mode((self.board_size, self.board_size))  # 窗口大小：680x680
        self.screen.fill((0, 0, 150))  # 蓝色填充窗口

        self.draw()  # 绘制

    def restart(self):
        """
        重置游戏
        """
        self.chess_arr = []  # 清空落子
        self.game_state = 0  # 游戏状态为正常进行
        self.init()  # 窗口重新初始化

    def player(self):
        """
        玩家落子
        :return: 落子坐标(x, y)，若落子位置非法，返回(-1, -1)
        """
        # 获取鼠标点击坐标
        x, y = (round(((i - self.space_size) * 1. / self.cell_size))
                for i in pygame.mouse.get_pos())
        # 加入棋子坐标数组中
        if (x >= 0) and (x < self.cell_num) and (y >= 0) and (y < self.cell_num) and (
                (x, y, 1) not in self.chess_arr) and ((x, y, 2) not in self.chess_arr):
            self.chess_arr.append((x, y, 1))
            # 检查棋局状态
            self.game_state = self.check_state()
            # 返回落子坐标
            return x, y
        return -1, -1

    def load_computer(self):
        """
        载入电脑
        """
        self.computer = Computer()

    def game(self, mode=0):
        """
        进行游戏
        :param mode: 对局方式
                         0: 传入棋子信息, 电脑随机落子
                         1: 电脑通过识别图像获取棋子信息, 随机落子
                         2: 博弈搜索, MIN-MAX
                         3: 神经网络识别棋局
                         4: DQN识别棋局
                         ...
        """
        self.load_computer()  # 载入电脑
        self.init()  # 窗口初始化
        # 获取棋盘image
        if mode > 0:
            self.save_current_image('computer')
        # 监听事件
        while True:
            for event in pygame.event.get():
                # 退出
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                # 落子
                if (self.game_state == 0) and (event.type == pygame.MOUSEBUTTONUP):
                    # 玩家落子
                    x, y = self.player()
                    # 落子位置非法
                    if x == -1:
                        continue
                    # 绘制
                    self.draw()
                    # 棋盘满
                    if len(self.chess_arr) == 225:
                        # 游戏状态非正常进行
                        self.game_state = -1
                        continue
                    # 获取棋盘image
                    if mode > 0:
                        self.save_current_image('player')
                    # 若未分胜负，则电脑紧接着落子
                    if self.game_state == 0:
                        if mode > 0:  # 电脑通过识别图像获取棋子信息
                            self.chess_arr.append(self.computer(mode=mode, flag=2))
                        else:  # 传入棋子信息
                            self.chess_arr.append(self.computer(mode=mode, flag=2, chess_arr=self.chess_arr))
                        # 检查棋局状态
                        self.game_state = self.check_state()
                        # 绘制
                        self.draw()
                        # 棋盘满
                        if len(self.chess_arr) == 225:
                            # 游戏状态非正常进行
                            self.game_state = -1
                            continue
                        # 获取棋盘image
                        if mode > 0:
                            self.save_current_image('computer')
                # 监听空格按键用于重置游戏
                if (event.type == pygame.KEYDOWN) and (event.key == pygame.K_SPACE):
                    self.restart()


class Computer:
    def __init__(self, CNN_net=None, SelectCNN_net=None, DQN_net=None):
        # 识别棋子信息的网络
        self.CNN_model = CNNModel()
        # 传入网络
        if CNN_net is not None:
            self.CNN_model.net = CNN_net
        # CNN模型文件存在则加载
        elif os.path.exists('CNN.pt'):
            self.CNN_model.load_net()

        # 判断棋局状态的网络
        self.selectCNN_model = SelectCNNModel()
        # 传入网络
        if SelectCNN_net is not None:
            self.selectCNN_model.net = SelectCNN_net
        # SelectCNN模型文件存在则加载
        elif os.path.exists('SelectCNN.pt'):
            self.selectCNN_model.load_net()

        # 判断落子位置的网络
        self.DQN_model = DQNModel()
        # 传入网络
        if DQN_net is not None:
            self.DQN_model.net = DQN_net
        # DQN模型文件存在则加载
        elif os.path.exists('DQN.pt'):
            self.DQN_model.load_net()

        self.chess_arr = None  # 保存棋子信息

    def __call__(self, mode=0, flag=2, chess_arr=None):
        """
        :param mode: 对局方式
                         0: 传入棋子信息, 电脑随机落子
                         1: 电脑通过识别图像获取棋子信息, 随机落子
                         2: 博弈搜索, MIN-MAX
                         3: 神经网络识别棋局
                         4: DQN识别棋局
                         ...
        :param flag: 1 黑子 或 2 白子
        :param chess_arr: 棋子信息
        :return: 落子坐标 (x, y)
        """
        if chess_arr is None:
            self.load_chess()  # 从图像中加载棋子信息
        else:
            self.chess_arr = list(chess_arr)  # 调用者传入棋子信息

        if mode <= 1:  # 随机落子
            x, y = self.get_random_xy()
            while self.check_xy(x, y):
                x, y = self.get_random_xy()
            return x, y, flag
        elif mode == 2:  # 博弈搜索, MIN-MAX
            x, y = self.get_minmax_xy(flag)
            return x, y, flag
        elif mode == 3:  # 神经网络识别棋局
            x, y = self.get_selectCNN_xy(flag)
            return x, y, flag
        elif mode == 4:  # DQN识别棋局
            x, y = self.get_DQN_xy(flag)
            return x, y, flag

    def load_chess(self, path='tmp\\player.png'):
        """
        图像识别棋子信息
        :param path: image路径
        """
        self.chess_arr = self.CNN_model.image_to_chess(path)

    def check_state(self, chess_arr=None):
        """
        判断是否产生胜负
        :param chess_arr: 是否传入棋子信息
        :return: 游戏状态：0继续，1黑子胜，2白子胜
        """
        # numpy数组形式传入
        if isinstance(chess_arr, np.ndarray):
            chess_arr = [(idx % 15, int(idx / 15), 1) for idx in np.where(chess_arr.flatten() == 1)[0]] + [
                (idx % 15, int(idx / 15), 2) for idx in np.where(chess_arr.flatten() == 2)[0]]
        # 未传入
        if chess_arr is None:
            chess_arr = self.chess_arr
        # 对落下的每个棋子
        for x, y, flag in chess_arr:
            # 依次判断竖直、水平、左右对角线方向
            for m, n in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
                # 保存方向上落子情况，例如：0，0，1，1，（1），1，1，0，0
                check_arr = []
                for i in range(-4, 5):
                    if (x + m * i, y + n * i, flag) in chess_arr:
                        check_arr.append(1)
                    else:
                        check_arr.append(0)
                # 形成五子连线时，紧接着的5个元素为1
                for i in range(5):
                    if sum(check_arr[i:i + 5]) == 5:  # 和为5
                        return flag  # 返回获胜方
        return 0  # 游戏继续

    def check_xy(self, x, y):
        """
        检查该坐标是否已落子
        :return: True: 已落子 False: 未落子
        """
        if ((x, y, 1) in self.chess_arr) or ((x, y, 2) in self.chess_arr):
            return True
        return False

    def get_player_xy(self):
        """
        根据player.png和computer.png获取玩家落子位置, ..->computer.png->player.png(玩家落子)->电脑识别
        :return: (x, y)
        """
        chess = set(self.CNN_model.image_to_chess('tmp\\player.png')) - set(
            self.CNN_model.image_to_chess('tmp\\computer.png'))
        return chess.pop()

    @staticmethod
    def optimize(init_input, flag=2):
        """
        对输入到神经网络的数据做一些调整
        正反馈: flag = 1: 0 -> 0, 1 -> 1, 2 -> -1 flag = 2: 0 -> 0, 1 -> -1, 2 -> 1
        偏移: 在标签值的基础上加上正态分布随机小量
        维度扩张: 扩张batch和channel维
        :param init_input: 初始棋盘数据 (15, 15) 0, 1, 2
        :param flag: 黑子或白子
        :return: 神经网络的input (1, 1, 15, 15)
        """
        # 保留原始数据
        net_input = copy.deepcopy(init_input)
        # 正反馈
        for i in range(15):
            for j in range(15):
                if flag == 1:
                    if net_input[i][j] == 2:
                        net_input[j][j] = -1
                if flag == 2:
                    if net_input[i][j] == 1:
                        net_input[i][j] = -1
                    if net_input[i][j] == 2:
                        net_input[i][j] = 1
        # 偏移
        net_input += torch.normal(mean=0, std=0.01, size=net_input.shape).to(net_input.device)
        # 维度扩张
        net_input = net_input.unsqueeze(0).unsqueeze(0)
        # 返回输入
        return net_input

    def eve(self, opponent, epochs):
        """
        电脑与电脑竞争
        :param opponent: 对手
        :param epochs: 对局数
        :return: 己方胜利次数
        """
        # 己方胜利局数
        win = 0
        # 对局
        for epoch in range(epochs):
            count = 0  # 统计棋盘上棋子数
            self.chess_arr = list()  # 棋子清空
            while True:
                self.chess_arr.append(self.get_selectCNN_xy(flag=1) + (1,))  # 己方执黑子
                count += 1
                if count == 225:  # 第225个棋子必为黑子
                    break
                if self.check_state() != 0:  # 己方胜利
                    win += 1
                    break
                opponent.chess_arr = self.chess_arr  # 对手棋盘信息更新
                self.chess_arr.append(opponent.get_selectCNN_xy(flag=2) + (2,))  # 对手执白子
                count += 1
                if self.check_state() != 0:  # 对手胜利
                    break
        return win

    @staticmethod
    def get_random_xy():
        """
        获取随机坐标
        :return: (x, y)
        """
        # 随机选取坐标
        x = random.randint(0, 14)  # 0..14
        y = random.randint(0, 14)
        return x, y

    def get_minmax_xy(self, depth=4, flag=2):
        """
        极大极小搜索确定落子位置
        当前待落子方为MAX
        :param depth: 搜索深度
        :param flag: 1 黑子 或 2 白子
        :return: (x, y)
        """

        def chess_sort(current):
            """
            获取可落子坐标并根据距已落子坐标的质心远近排序
            :param current: 当前chess_board (15, 15)
            :return: 排序后的可落子坐标列表
            """
            # 存储可落子坐标
            chess_avail = []
            # 质心坐标, 已落子个数
            p_x, p_y, count = 0, 0, 0
            # 统计
            for l_x in range(15):
                for l_y in range(15):
                    if current[l_y][l_x] != 0:
                        p_x += l_x
                        p_y += l_y
                        count += 1
                    if current[l_y][l_x] == 0:
                        chess_avail.append((l_x, l_y))
            # 质心x坐标
            p_x = int(p_x / count)
            # 质心y坐标
            p_y = int(p_y / count)
            # 距质心远近排序
            chess_avail.sort(key=lambda point: abs(point[0] - p_x) + abs(point[1] - p_y))

            return chess_avail

        def evaluate(current, state=0):
            """
            到达子节点或产生胜负后进行评估
            评估函数与双方在落子位置周围棋子个数有关
            :param current: 子节点chess_board (15, 15)
            :param state: 棋局状态, 传入非零值表示已判断棋局状态, 且产生胜负
            :return: 评估值
            """
            # 判断棋局状态
            if state == 0:
                state = self.check_state(current)
            if state == 0:  # 正常进行
                value = [0, 0]  # 黑子, 白子的评估值
                for v_x in range(15):
                    for v_y in range(15):
                        if current[v_y][v_x] == 0:  # 未落子
                            continue
                        for m, n in [(0, 1), (1, 0), (1, 1), (-1, 1)]:  # 竖直、水平、左对角线、右对角线
                            arr = []  # 存储方向上位置信息, 个数 5..9
                            for i in range(-4, 5):
                                # 棋盘内
                                if ((v_x + m * i) >= 0) and ((v_x + m * i) <= 14) and ((v_y + n * i) >= 0) and (
                                        (v_y + n * i) <= 14):
                                    arr.append(current[v_y + n * i][v_x + m * i])
                            # 评估函数进行评估
                            for i in range(len(arr) - 4):  # 每次取连续5个位置
                                # 存在对手棋子
                                if (current[v_y][v_x] % 2 + 1) in arr[i:i + 5]:
                                    # 虽然己方已无可能5子连线, 但是这里也阻止了对方5子连线
                                    # 统计对方棋子个数, 越多则越说明抑制程度越大
                                    value[current[v_y][v_x] - 1] += arr[i:i + 5].count(current[v_y][v_x])
                                # 不存在对手棋子
                                else:
                                    # 己方有可能5子连线, 属于有利位置
                                    # 己方棋子个数平方累加, 实现构成5子连线优先级大于封锁对手行动
                                    value[current[v_y][v_x] - 1] += (arr[i:i + 5].count(current[v_y][v_x]) ** 2)
                # 返回 MAX - MIN
                return value[0] - value[1] if flag == 1 else value[1] - value[0]
            elif state == flag:  # MAX胜
                return float('inf')  # 评估值为无穷大
            else:  # MIN胜
                return float('-inf')  # 评估值为无穷小

        def search(current, level, parent_value):
            """
            搜索过程
            :param current: 当前chess_board (15, 15)
            :param level: 当前层数, 1..depth
            :param parent_value: 父节点所在层的value
            :return: 评估值, 搜索入口返回最后的落子坐标
            """
            t_x, t_y = 0, 0  # (x, y)

            # 判断棋局状态
            state = self.check_state(current)
            if state != 0:  # 产生胜负
                return evaluate(current, state)

            if level > depth:  # 到达子节点
                return evaluate(current)  # 返回评估值

            if level % 2 == 1:  # MAX层
                value = float('-inf')  # 极大值
            else:  # MIN层
                value = float('inf')  # 极小值

            for s_x, s_y in chess_sort(current):  # 对可落子坐标排序, 缩短搜索时间
                # 落子
                current[s_y][s_x] = level % 2 + 1
                # 获取评估值
                rtv = search(current, level + 1, value)
                # 收回棋子
                current[s_y][s_x] = 0

                # MAX层取极大值
                if level % 2 == 1 and rtv > value:
                    # 更新
                    value = rtv
                    # 记录下坐标
                    t_x, t_y = s_x, s_y
                    # beta剪枝
                    if value >= parent_value:
                        if level == 1:  # 搜索入口返回最后的落子坐标
                            return t_x, t_y
                        return value

                # MIN层取极小值
                if level % 2 == 0 and rtv < value:
                    # 更新
                    value = rtv
                    # 记录下坐标
                    t_x, t_y = s_x, s_y
                    # alpha剪枝
                    if value <= parent_value:
                        if level == 1:  # 搜索入口返回最后的落子坐标
                            return t_x, t_y
                        return value

            # 搜索入口返回最后的落子坐标
            if level == 1:
                return t_x, t_y
            # 返回评估值
            return value

        def check_four(current, self_flag):
            """
            检查是否存在4子连线
            :param current: 当前chess_board (15, 15)
            :param self_flag: 棋子颜色
            :return: 存在4子连线则返回一端坐标, 否则返回(-1, -1)
            """
            # 对落下的每个棋子
            for t_x, t_y, t_flag in self.chess_arr:
                # 对方棋子
                if t_flag != self_flag:
                    continue
                # 依次判断竖直、水平、左右对角线方向
                for m, n in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
                    # 保存方向上落子情况，例如：(1), 1, 1, 1
                    check_arr = []
                    for i in range(4):
                        if (t_x + m * i, t_y + n * i, t_flag) in self.chess_arr:
                            check_arr.append(1)
                        else:
                            check_arr.append(0)
                    # 形成4子连线时, 4个元素和为1
                    if sum(check_arr) == 4:
                        # 前端
                        if ((t_x - m) >= 0) and ((t_x - m) <= 14) and ((t_y - n) >= 0) and ((t_y - n) <= 14) and (
                                current[t_y - n][t_x - m] == 0):
                            return t_x - m, t_y - n
                        # 后端
                        if ((t_x + 4 * m) >= 0) and ((t_x + 4 * m) <= 14) and ((t_y + 4 * n) >= 0) and (
                                (t_y + 4 * n) <= 14) and (current[t_y + 4 * n][t_x + 4 * m] == 0):
                            return t_x + 4 * m, t_y + 4 * n
            return -1, -1  # 不存在

        # 棋盘上无子, 随机落子
        if len(self.chess_arr) == 0:
            return self.get_random_xy()

        # 从图像中加载棋子信息
        # self.load_chess()
        # 15x15的二维数组, int类型
        chess_board = np.zeros((15, 15), dtype=int)
        # 标注落子位置, 棋子类型
        for chess in self.chess_arr:
            chess_board[chess[1]][chess[0]] = chess[2]

        # 先检查己方是否存在4子连线
        x, y = check_four(chess_board, flag)
        if x != -1:
            # 返回落子坐标
            return x, y

        # 再检查对方是否存在4子连线
        x, y = check_four(chess_board, flag % 2 + 1)
        if x != -1:
            # 返回落子坐标
            return x, y

        # 从第1层开始博弈搜索, MAX, MIN, MAX...
        x, y = search(chess_board, 1, float('inf'))

        # 返回落子坐标
        return x, y

    def get_selectCNN_xy(self, flag=2):
        """
        通过网络识别棋局
        :param flag: 1 落黑子 或 2 落白子
        :return: (x, y)
        """

        # gpu or cpu
        device = list(self.selectCNN_model.net.parameters())[0].device

        # FloatTensor (15, 15)
        chess_board = torch.zeros(15, 15)
        # 标注落子位置, 棋子类型
        for chess in self.chess_arr:
            chess_board[chess[1]][chess[0]] = chess[2]

        # 送入网络, 压缩batch维
        chess = self.selectCNN_model.net(self.optimize(chess_board, flag).to(device)).squeeze(0)  # (225)
        # 输入非零位置置无穷小
        for loc in range(chess.shape[0]):
            if chess_board[int(loc / 15)][loc % 15] != 0:
                chess[loc] = float('-inf')

        # 取剩余可落子位置中值最大一个输出
        loc = chess.cpu().argmax(dim=0).item()

        # 返回 (x, y)
        return loc % 15, int(loc / 15)

    def get_DQN_xy(self, flag=2):
        """
        DQN获取落子位置
        :param flag: 1 落黑子 或 2 落白子
        :return: (x, y)
        """

        # FloatTensor (15, 15)
        chess_board = torch.zeros(15, 15)
        # 标注落子位置, 棋子类型
        for chess in self.chess_arr:
            chess_board[chess[1]][chess[0]] = chess[2]

        # gpu or cpu
        device = list(self.DQN_model.net.parameters())[0].device

        # 送入网络
        chess = self.DQN_model.net(self.optimize(chess_board, flag).to(device))  # (1, 225)
        # 压缩batch维
        chess = chess.squeeze(0)  # (225)
        # 输入非零位置置无穷小
        for loc in range(chess.shape[0]):
            if chess_board[int(loc / 15)][loc % 15] != 0:
                chess[loc] = float('-inf')

        # 取剩余可落子位置中值最大一个输出
        loc = chess.cpu().argmax(dim=0).item()

        # 返回 (x, y)
        return loc % 15, int(loc / 15)


class ImageDataset(Data.Dataset):
    def __init__(self, data_dir):
        # 目录: train，val，test
        self.data_dir = data_dir
        # 对PIL图片进行转换
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop((600, 600)),  # 中心裁剪, (680, 680) -> (600, 600)
            torchvision.transforms.ToTensor()  # 转换成FloatTensor, range [0.0, 1.0], shape (C x H x W)
        ])

    def __getitem__(self, idx):
        # 0..99..(索引) -> 1..100..(文件名)
        idx += 1
        # 图片, (channel 3, height 600, width 600)
        image = self.transforms(Image.open(os.path.join(self.data_dir, 'image', '%d.png' % idx)))
        # 标签, (225)
        label = torch.tensor(np.load(os.path.join(self.data_dir, 'label', '%d.npy' % idx)).flatten(), dtype=torch.float)
        # 返回(image, label)
        return image, label

    def __len__(self):
        # 返回样本数
        return len(os.listdir(os.path.join(self.data_dir, 'label')))


class GlobalCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_size = None

    def forward(self, y_hat, y):
        """
        :param y_hat: (batch_size, 225, 3)
        :param y: (batch_size, 225)
        :return: 损失大小
        """
        # 获取批量大小
        self.batch_size = y_hat.shape[0]
        # 损失值
        loss = torch.zeros(1).to(y_hat.device)
        # 逐样本累加loss
        for i in range(self.batch_size):
            loss += F.cross_entropy(y_hat[i], y[i].to(torch.int64))
        return loss


class CNN(nn.Module):
    """
    图像识别, 获取棋盘上棋子信息
    """

    def __init__(self):
        super().__init__()
        # 卷积层, 输入通道 RGB 3, 输出通道 6, 卷积核 (20, 20), 步长 (20, 20)
        # image (600, 600), 每个棋子半径16, 占据(40, 40)的方块
        # 方块(40, 40) -> (2, 2) image(600, 600) -> (30, 30)
        # (3, 600, 600) -> (6, 30, 30)
        self.conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=20, stride=20)
        # 平均池化(local), 方块(2, 2) -> (1, 1), image(30, 30) -> (15, 15)
        # 每一个点(方块)储存棋子信息
        # (6, 30, 30) -> (6, 15, 15)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # 线性层
        # (225, 6) -> (225, 3)
        self.fc = nn.Linear(in_features=6, out_features=3)

    def forward(self, image):
        # 卷积, 线性激活, 池化, 压缩, 展平
        # (batch_size, 3, 600, 600) -> (batch_size, 6, 225)
        x = self.pool(F.relu(self.conv(image))).view(-1, 6, 225)
        # 线性连接
        # (batch_size, 6, 225) -> (batch_size, 225, 6) -> (batch_size, 225, 3)
        # 3: 类别
        return self.fc(x.permute(0, 2, 1))


class CNNModel:
    def __init__(self):
        self.net = CNN()  # 实例化网络

    def load_net(self, path='CNN.pt'):
        """
        加载模型网络
        :param path: 模型路径
        """
        # 加载模型, 导入网络参数
        self.net.load_state_dict(torch.load(path)['net'])

    @staticmethod
    def open_image(path):
        """
        打开图片
        :param path: 图片路径
        :return: 可以输入网络的数据 (1, 3, 600, 600)
        """
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop((600, 600)),
            torchvision.transforms.ToTensor()
        ])
        return transforms(Image.open(path)).unsqueeze(0)

    def image_to_chess(self, path):
        """
        完成图片到棋子信息的转换
        :param path: 图片路径
        :return: 保存有棋子信息的列表
        """
        # 打开图片
        image = self.open_image(path)
        # 网络识别
        label_hat = self.net(image).argmax(dim=2).squeeze(0).numpy()
        # 转换为棋子信息列表
        chess_arr = [(idx % 15, int(idx / 15), 1) for idx in np.where(label_hat == 1)[0]] + [
            (idx % 15, int(idx / 15), 2) for idx in np.where(label_hat == 2)[0]]
        return chess_arr

    def test(self, path=None):
        """
        在测试集上测试或单张图片测试
        """

        def set_test():
            """
            在测试集上测试
            """
            # 实例化一个五子棋对象
            gomoku = Gomoku()
            # 样本个数
            count = len(os.listdir('test\\image'))
            # 索引顺序遍历
            for idx in range(1, count + 1):
                # 更新棋子信息
                gomoku.set_arr(self.image_to_chess('test\\image\\%d.png' % idx))
                # 打印对应样本名
                print('当前对应样本:test\\image\\%d.png' % idx)
                # 图形显示
                if gomoku.show_arr():
                    continue
                else:
                    break

        def image_test():
            """
            单张图片测试
            """
            # 实例化一个五子棋对象
            gomoku = Gomoku()
            # 更新棋子信息
            gomoku.set_arr(self.image_to_chess(path))
            # 打印对应t图片名
            print('当前对应图片:%s' % path)
            # 图形显示
            gomoku.show_arr()

        if path is None:
            set_test()
        else:
            image_test()


class SelectCNN(nn.Module):
    """
    根据当前棋子信息, 选择下一步落子位置
    """

    def __init__(self, requires_grad=False):
        super().__init__()
        # 归一化
        self.isn = nn.InstanceNorm2d(num_features=1)
        # 卷积层
        # 输入通道 1 输出通道 3 卷积核 3 x 3 偏移量 False
        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, bias=False)
        # 最大池化
        # 核 (2, 2) 步长 1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        # 线性层
        # (1, 432) -> (1, 225)
        self.fc = nn.Linear(in_features=432, out_features=225)

        # 网络适应度
        self.apt = 0

        # 是否需要计算梯度
        for parameter in self.parameters():
            parameter.requires_grad = requires_grad

    def forward(self, chess_board):
        """
        :param chess_board: (1, 1, 15, 15)
        :return: (225)
        """
        # 归一化
        x = self.isn(chess_board)
        # 卷积 线性激活 池化 维度变换
        # (1, 1, 15, 15) -> (1, 432)
        x = self.pool(F.relu(self.conv(x))).view(1, -1)
        # 全连接
        # (1, 432) -> (1, 225)
        return self.fc(x)


class SelectCNNModel:
    def __init__(self):
        self.net = SelectCNN()  # 实例化网络

    def load_net(self, path='SelectCNN.pt'):
        """
        加载模型网络
        :param path: 模型路径
        """
        # 加载模型, 导入网络参数
        self.net.load_state_dict(torch.load(path)['net'])


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
        self.bn = nn.BatchNorm2d(num_features=3)
        self.fc = nn.Linear(in_features=507, out_features=225)

    def forward(self, x):
        """
        :param x: 批量棋盘数据 (batch_size, 1, 15, 15)
        :return: 批量棋盘Q-value (batch_size, 225)
        """
        x = F.relu(self.bn(self.conv(x)))
        return self.fc(x.view(x.shape[0], -1))


class DQNModel:
    def __init__(self):
        self.net = DQN()  # 实例化网络

    def load_net(self, path='DQN.pt'):
        """
        加载模型网络
        :param path: 模型路径
        """
        # 加载模型, 导入网络参数
        self.net.load_state_dict(torch.load(path)['net'])


class ReplayMemory:
    """
    经验池
    """

    def __init__(self, capacity=10000):
        """
        :param capacity: 池容量
        """
        self.capacity = capacity
        # 经验池
        self.memory = []
        # 指针, 指向经验池中可以存储的位置
        self.position = 0

    def push(self, *args):
        """
        记忆一次行动
        :param args: 一次行动, (状态, 动作, 奖励, 下一状态, 是否到达终点, 电脑执子颜色)
                      状态为一个15 x 15的numpy数组, 记录棋盘信息
                      动作为选择落子的坐标 idx = x + 15 * y
        """
        # 组织行动信息
        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'end', 'flag'))

        # 入池
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        # 指针向后挪一位
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        采样
        :param batch_size: 批量大小
        :return: 批量数据
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        :return: 池中数据个数
        """
        return len(self.memory)


if __name__ == '__main__':
    # 实例化五子棋
    s = Gomoku()
    # 开始游戏
    # mode: 对局方式
    # 0: 传入棋子信息, 电脑随机落子
    # 1: 电脑通过识别图像获取棋子信息, 随机落子
    # 2: 博弈搜索, MIN-MAX
    # 3: 神经网络识别棋局
    # 4: DQN识别棋局
    s.game(mode=2)
    # 空格键可以重置游戏
