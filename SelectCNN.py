from Class import *
import time
import math


def adapt(parents, epochs=5):
    """
    适应度函数, eve, 计算胜率作为适应度
    :param parents: 群体
    :param epochs: eve轮数
    :return: 适应度
    """
    # 加载环境
    env = SelectCNNModel()
    env.load_net()

    # gpu or cpu
    env.net.to(list(parents[0].parameters())[0].device)

    # 作为对手
    opponent = Computer(SelectCNN_net=env.net)

    # 群体中个体数
    number = len(parents)

    # eve
    for i in range(number):
        win = Computer(SelectCNN_net=parents[i]).eve(opponent, epochs)
        # 适应度 = 获胜轮数 / 对战轮数
        parents[i].apt = win / epochs


def mutation(parents):
    """
    突变
    :param parents: 父群体
    :return: 突变产生的新群体
    """
    # 群体中个体数
    number = len(parents)
    # 复制生成新个体
    children = copy.deepcopy(parents)
    # 新个体突变
    for i in range(number):
        for parameter in children[i].parameters():
            # + (1 - 适应度)开根 * 正态分布
            parameter.add_(math.sqrt(1 - children[i].apt) * torch.normal(mean=0, std=1, size=parameter.shape).to(
                list(parents[0].parameters())[0].device))

    return children


def optimizer(group, q_value, epochs=1):
    """
    q-竞争优化
    :param group: 竞争的群体
    :param q_value: q-竞争的q值
    :param epochs: eve轮数
    :return: 经过淘汰后的群体
    """

    def random_opponent(ecp):
        """
        获取随机q-value个对手
        :param ecp: 排除
        :return: q-value个对手
        """
        opponents = []
        count = 0
        while count != q_value:
            x = random.randint(0, number - 1)
            if (x != ecp) and (x not in opponents):
                opponents.append(x)
                count += 1
        return opponents

    # 群体中个体数
    number = len(group)
    # 个体 -> 电脑
    computers = [Computer(SelectCNN_net=group[i]) for i in range(number)]
    # 记录获胜轮数
    wins = {}
    # eve
    for i in range(number):
        win = 0
        for opponent in random_opponent(i):
            win += computers[i].eve(computers[opponent], epochs)
        # 获胜轮数
        wins[group[i]] = win

    # 胜率由高到低取前 number / 2 个
    return [i[0] for i in sorted(wins.items(), key=lambda j: j[1])][0:int(number / 2)]


def train(number=5, q_value=5, epochs=20, period=5):
    """
    训练
    进化规划
    :param number: 群体包含个体个数
    :param q_value: q-竞争的q值
    :param epochs: 迭代轮数
    :param period: 周期, 环境更新
    """

    # 有英伟达GPU则使用gpu,否则cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training on", device)

    # 生成初始群体, 包含5个个体
    parents = [SelectCNN().to(device) for _ in range(number)]

    # 创建环境
    if not os.path.exists('SelectCNN.pt'):
        # 随机
        torch.save({'net': SelectCNN().state_dict()}, 'SelectCNN.pt')

    # 计算适应度
    adapt(parents)

    # 进化迭代
    for epoch in range(epochs):
        # 开始时间
        start = time.time()

        # 突变形成新个体
        children = mutation(parents)
        # 竞争获得下一代
        parents = optimizer(parents + children, q_value)

        # 计算适应度
        adapt(parents)

        # 打印进化信息
        print('generation %d, adaptability' % (epoch + 1), end='')
        for i in range(number):
            print(' %.2f' % parents[i].apt, end='')
        print(', time %.1f sec' % (time.time() - start))

        # 更新环境, 即最优个体
        if epoch % period == (period - 1):
            # 最优个体
            best = max(parents, key=lambda parent: parent.apt)

            # 保存模型
            torch.save({'net': best.state_dict()}, 'SelectCNN.pt')

            # 打印最优个体信息
            print('period %d, best adaptability %.2f' % ((epoch + 1) / period, best.apt))


if __name__ == '__main__':
    # 训练入口
    train()
