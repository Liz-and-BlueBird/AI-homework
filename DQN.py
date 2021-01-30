from Class import *
import time
import math


def optimize(policy_net, target_net, memory, optimizer, batch_size):
    """
    网络权重优化
    :param policy_net: 策略网络
    :param target_net: 目标网络
    :param memory: 经验池
    :param optimizer: 优化器
    :param batch_size: 批量大小
    :return: 损失大小
    """
    # 样本个数不够
    if len(memory) < batch_size:
        return None

    # gamma参数
    GAMMA = 0.99

    # cpu or gpu
    device = list(policy_net.parameters())[0].device

    # 批量样本
    batch = memory.sample(batch_size)

    # 旧状态 (batch_size, 1, 15, 15)
    state = torch.cat([Computer.optimize(torch.Tensor(i.state), i.flag) for i in batch], dim=0).to(device)
    # 采取的动作 (batch_size, 1)
    action = torch.LongTensor([i.action for i in batch]).unsqueeze(1).to(device)
    # 新状态
    next_state = torch.cat([Computer.optimize(torch.Tensor(i.next_state), i.flag) for i in batch], dim=0).to(device)
    # 奖励 (batch_size, 1)
    reward = torch.Tensor([i.reward for i in batch]).unsqueeze(1).to(device)
    # 是否结束
    end = torch.Tensor([i.end for i in batch]).unsqueeze(1).to(device)

    # 输入Q值 (batch_size, 1)
    q = torch.gather(input=policy_net(state), dim=1, index=action)
    # 获取期待Q值 (batch_size, 1)
    with torch.no_grad():
        out = target_net(next_state)
        # 输出对应输入非零位置置无穷小
        for i in range(len(batch)):
            for loc in range(state.shape[-1] ** 2):
                if batch[i].state[loc // 15][loc % 15] != 0:
                    out[i][loc] = float('-inf')
        # 到达终点奖励
        reward_end = torch.where(end == 0, end, reward)
        # 期待Q值 (batch_size, 1)
        except_q = torch.where(torch.gt(reward_end, torch.zeros(batch_size, 1).to(device)), reward_end,
                               torch.max(input=out, dim=1)[0].unsqueeze(1) * GAMMA + reward)

    # 损失函数
    loss = F.smooth_l1_loss(input=q, target=except_q)

    # 清空梯度
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 将梯度限制在-1到1之间
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    # 更新权重
    optimizer.step()

    # 返回损失
    return loss


def train(batch_size=256, epochs=50, period=5):
    """
    DQN学习过程
    :param batch_size: 批量大小
    :param epochs: 对局数
    :param period: 拷贝周期
    """

    def get_chess_np(chess_arr_):
        """
        构造numpy数组存储棋盘信息
        :param chess_arr_: 棋子信息
        :return: (15, 15)
        """
        # 15x15的二维数组
        chess_board = np.zeros((15, 15))
        # 标注落子位置, 棋子类型
        for chess in chess_arr_:
            chess_board[chess[1]][chess[0]] = chess[2]
        return chess_board

    def policy():
        """
        做决策
        :return: x, y
        """
        # epsilon, 随轮数增加指数衰减
        eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (epoch_count + epoch + 1) / EPS_DECAY)

        with torch.no_grad():
            # 存储未占用位置
            empty = []
            # 获得所有行动Q值
            out = policy_net(Computer.optimize(torch.Tensor(state).to(device), flag).to(device)).squeeze(0)
            # 输出对应输入非零位置置无穷小
            for loc in range(225):
                if state[loc // 15][loc % 15] != 0:
                    out[loc] = float('-inf')
                else:
                    empty.append(loc)
            # 最大Q值坐标索引
            idx = out.argmax(dim=0).item()
            # 随机坐标索引
            random_idx = random.sample(empty, k=1)[0]

            # 以eps的概率返回随机坐标
            if random.random() <= eps:
                return random_idx % 15, random_idx // 15
            else:
                return idx % 15, idx // 15

    def get_reward():
        """
        获取奖励
        :return: 奖励值
        """
        # 奖励由两部分组成: 扩大优势(60%), 缩小劣势(40%)
        rtv = [0] * 2

        # 扩大优势
        together = np.zeros([4, 4])  # 第0维 0: 水平 1: 竖直 2: 左上往右下 3: 左下往右上 第1维 0: left 1: right 2 3: 是否出现相反棋子
        # 缩小劣势
        opponent = np.zeros([4, 4])
        # 统计落子周围相同和不同棋子个数
        for i in range(1, 5):
            # 水平左
            if (together[0][2] == 0) and ((x - i) >= 0):
                if state[y][x - i] == flag:
                    together[0][0] += 1
                else:
                    together[0][2] = 1
            if (opponent[0][2] == 0) and ((x - i) >= 0):
                if state[y][x - i] == (flag % 2 + 1):
                    opponent[0][0] += 1
                else:
                    opponent[0][2] = 1
            # 水平右
            if (together[0][3] == 0) and ((x + i) <= 14):
                if state[y][x + i] == flag:
                    together[0][1] += 1
                else:
                    together[0][3] = 1
            if (opponent[0][3] == 0) and ((x + i) <= 14):
                if state[y][x + i] == (flag % 2 + 1):
                    opponent[0][1] += 1
                else:
                    opponent[0][3] = 1
            # 竖直上
            if (together[1][2] == 0) and ((y - i) >= 0):
                if state[y - i][x] == flag:
                    together[1][0] += 1
                else:
                    together[1][2] = 1
            if (opponent[1][2] == 0) and ((y - i) >= 0):
                if state[y - i][x] == (flag % 2 + 1):
                    opponent[1][0] += 1
                else:
                    opponent[1][2] = 1
            # 竖直下
            if (together[1][3] == 0) and ((y + i) <= 14):
                if state[y + i][x] == flag:
                    together[1][1] += 1
                else:
                    together[1][3] = 1
            if (opponent[1][3] == 0) and ((y + i) <= 14):
                if state[y + i][x] == (flag % 2 + 1):
                    opponent[1][1] += 1
                else:
                    opponent[1][3] = 1
            # 左上
            if (together[2][2] == 0) and ((x - i) >= 0) and ((y - i) >= 0):
                if state[y - i][x - i] == flag:
                    together[2][0] += 1
                else:
                    together[2][2] = 1
            if (opponent[2][2] == 0) and ((x - i) >= 0) and ((y - i) >= 0):
                if state[y - i][x - i] == (flag % 2 + 1):
                    opponent[2][0] += 1
                else:
                    opponent[2][2] = 1
            # 右下
            if (together[2][3] == 0) and ((x + i) <= 14) and ((y + i) <= 14):
                if state[y + i][x + i] == flag:
                    together[2][1] += 1
                else:
                    together[2][3] = 1
            if (opponent[2][3] == 0) and ((x + i) <= 14) and ((y + i) <= 14):
                if state[y + i][x + i] == (flag % 2 + 1):
                    opponent[2][1] += 1
                else:
                    opponent[2][3] = 1
            # 左下
            if (together[3][2] == 0) and ((x - i) >= 0) and ((y + i) <= 14):
                if state[y + i][x - i] == flag:
                    together[3][0] += 1
                else:
                    together[3][2] = 1
            if (opponent[3][2] == 0) and ((x - i) >= 0) and ((y + i) <= 14):
                if state[y + i][x - i] == (flag % 2 + 1):
                    opponent[3][0] += 1
                else:
                    opponent[3][2] = 1
            # 右上
            if (together[3][3] == 0) and ((x + i) <= 14) and ((y - i) >= 0):
                if state[y - i][x - i] == flag:
                    together[3][1] += 1
                else:
                    together[3][3] = 1
            if (opponent[3][3] == 0) and ((x + i) <= 14) and ((y - i) >= 0):
                if state[y - i][x - i] == (flag % 2 + 1):
                    opponent[3][1] += 1
                else:
                    opponent[3][3] = 1
        # 获取reward
        for i in range(4):
            # 寻找是否扩大了最大共线棋子个数
            if together[i][0] + together[i][1] + 1 > max_line[flag - 1]:
                # 更新为新的最大共线棋子个数
                max_line[flag - 1] = together[i][0] + together[i][1] + 1
                # 5以上统一记录为5
                if max_line[flag - 1] > 5:
                    max_line[flag - 1] = 5
                # reward的第一部分更新
                rtv[0] = max_line[flag - 1] * 100
            # 寻找是否抑制了共线可能性
            if opponent[i][0] + opponent[i][1] + 1 >= 3:
                # reward的第二部分更新
                rtv[1] = 3 * 100

        # return 0.6 * rtv[0] + 0.4 * rtv[1]
        return 1000 if max_line[flag - 1] == 5 else 0

    # epsilon-贪婪
    EPS_START = 0.1  # 初始值
    EPS_END = 0.01  # 末值
    EPS_DECAY = 100  # 衰减幅度

    # 有英伟达GPU则使用gpu, 否则cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training on", device)

    # 策略网络, 用于选择行动
    policy_net = DQN().to(device)
    # 目标网络, 用于计算Q值
    target_net = copy.deepcopy(policy_net)
    # 经验池
    memory = ReplayMemory()
    # 优化器
    optimizer = torch.optim.RMSprop(policy_net.parameters())

    # 对局总轮数
    epoch_count = 0

    # 在原有模型的基础上继续学习
    if os.path.exists('DQN.pt'):
        model = torch.load('DQN.pt')
        policy_net.load_state_dict(model['net'])
        target_net.load_state_dict(policy_net.state_dict())
        optimizer.load_state_dict(model['optimizer'])
        epoch_count += model['epoch']

    # 强化学习过程
    for epoch in range(epochs):
        # 损失值 权重更新次数 一轮开始时间
        loss, opt_count, start = 0., 0, time.time()

        # 初始化棋盘
        chess_arr = list()
        # 棋盘上棋子个数
        count = 0
        # 棋子颜色, 模型每落一次子变换一次, 先手黑子
        flag = 1
        # 共线的棋子最大个数
        max_line = [1, 1]
        # 初始状态
        state = get_chess_np(chess_arr)
        # 对手决策保存
        o_x, o_y = -1, -1

        # 对局
        while True:
            # 落子
            if o_x == -1:
                x, y = policy()
            else:
                x, y = o_x, o_y
            # 更新棋局
            chess_arr.append((x, y, flag))
            # 落子数加1
            count += 1
            # 奖励
            reward = get_reward()
            # 换手
            flag = flag % 2 + 1

            # 己方胜利或者棋盘满
            if (max_line[flag % 2] == 5) or (count == 225):
                # 状态标记为结束
                memory.push(state, x + 15 * y, state, reward, 1, flag % 2 + 1)
                break

            # 保存己方要做决策的状态
            init_state = state
            # 状态更新, 对手要做决策的状态
            state = get_chess_np(chess_arr)
            # 对手做决策
            o_x, o_y = policy()
            # 记忆, 下一状态预测了对手的预测, 在模型未更新的情况下
            memory.push(init_state, x + 15 * y, get_chess_np(chess_arr + [(o_x, o_y, flag)]), reward, 0, flag % 2 + 1)

        # 权重更新
        opt = optimize(policy_net, target_net, memory, optimizer, batch_size)
        if opt is not None:
            # 更新次数加1
            opt_count += 1
            # 记录loss
            loss += opt

        # 参数拷贝
        if (epoch_count + epoch + 1) % period == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 打印信息
        print('epoch %d, loss %.2f, time %.2f sec' % (
            epoch_count + epoch + 1, loss / opt_count if opt_count != 0 else 0, time.time() - start))

    # 保存模型
    torch.save({
        'net': policy_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch_count + epochs,
    }, 'DQN.pt')


if __name__ == '__main__':
    # 训练入口
    train()
