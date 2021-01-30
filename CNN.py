from Class import *
import time


def load_batch_iter(set_name, batch_size, shuffle=True):
    """
    加载批量数据, image (batch_size, channel 3, height 600, width 600), label (batch_size, 225)
    :param set_name: train, val, test
    :param batch_size: 批量大小
    :param shuffle: 是否搅乱顺序
    :return: 批量迭代器
    """
    return Data.DataLoader(ImageDataset(set_name), batch_size=batch_size, shuffle=shuffle)


def equal(y_hat, y):
    """
    样本为单位进行比较, 得出识别正确的样本数
    :param y_hat: 模型识别的标签结果 (batch_size, 225)
    :param y: 标签 (batch_size, 225)
    :return: batch_size内识别正确的样本数
    """
    # 获取批量大小
    batch_size = y_hat.shape[0]
    # 识别正确的样本数
    correct = 0
    # 逐样本比较
    for i in range(batch_size):
        correct += int(torch.equal(y_hat[i], y[i]))
    return correct


def evaluate(data_iter, net, device=None):
    """
    在验证集上验证模型
    :param data_iter: 数据集
    :param net: 神经网络
    :param device: cpu or gpu
    :return: 模型在验证集上的准确度
    """
    # 没有指定device则使用net的device
    if device is None:
        device = list(net.parameters())[0].device
    # 准确数, 样本数
    acc_sum, sample_count = 0., 0
    # 不计算梯度
    with torch.no_grad():
        # 进入评估模式
        net.eval()
        for X, y in data_iter:
            # 准确数累加
            acc_sum += equal(net(X.to(device)).argmax(dim=2).float(), y.to(device))
            # 样本数累加
            sample_count += y.shape[0]
        # 回到训练模式
        net.train()
    return acc_sum / sample_count


def train(train_iter, val_iter, net, loss, optimizer, device, epochs):
    """
    训练网络
    :param train_iter: 训练集
    :param val_iter: 验证集
    :param net: 神经网络
    :param loss: 损失函数
    :param optimizer: 优化器
    :param device: cpu or gpu
    :param epochs: 迭代次数
    """
    # 网络转换到cpu或gpu
    net = net.to(device)
    print("training on", device)
    # 训练过程
    for epoch in range(epochs):
        # 损失和, 准确识别样本数, 已训练批量数, 已训练样本数, 一次迭代的开始时间
        loss_sum, acc_sum, batch_count, sample_count, start = 0., 0., 0, 0, time.time()
        for X, y in train_iter:
            # use cpu or gpu
            X = X.to(device)
            y = y.to(device)
            # 前向传播
            y_hat = net(X)
            # 计算损失
            l = loss(y_hat, y)
            # 梯度清零
            optimizer.zero_grad()
            # 反向求导
            l.backward()
            # 一次优化
            optimizer.step()
            # 损失累加
            loss_sum += l.cpu().item()
            # 准确数累加
            acc_sum += equal(y_hat.argmax(dim=2).float(), y)
            # 样本数累加
            sample_count += y.shape[0]
            # 批量数+1
            batch_count += 1
        # 在验证集上验证获取验证准确度
        val_acc = evaluate(val_iter, net)
        # 迭代轮数, 训练平均损失(批量平均), 训练准确度(样本平均), 验证准确度(验证样本平均), 一次迭代耗时
        print('epoch %d, loss %.4f, train acc %.2f, val acc %.2f, time %.1f sec'
              % (epoch + 1, loss_sum / batch_count, acc_sum / sample_count, val_acc, time.time() - start))


def get_CNN_model(epochs=10, batch_size=8, lr=0.05):
    """
    获取模型
    :param epochs: 训练轮数
    :param batch_size: 批量大小
    :param lr: 学习率
    :return: 模型
    """
    train_iter = load_batch_iter('train', batch_size)  # 训练集
    val_iter = load_batch_iter('val', batch_size)  # 验证集

    # 有英伟达GPU则使用gpu,否则cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 网络
    net = CNN()
    # 自定义损失函数
    loss = GlobalCrossEntropy()
    # Adam优化
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)

    # 训练
    train(train_iter, val_iter, net, loss, optimizer, device, epochs)

    # 保存模型
    torch.save({'net': net.state_dict(), 'optimizer': optimizer.state_dict()}, 'CNN.pt')

    # 返回训练好的网络
    return net


if __name__ == '__main__':
    # 训练入口
    get_CNN_model()
