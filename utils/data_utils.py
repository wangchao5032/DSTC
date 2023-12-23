
import numpy as np
import torch
from utils import constants
from funcy import merge

def argsort(seq):
    """
    sort by length in reverse order
    ---
    seq (list[array[int32]])
    """
    # enumerate是将一个可遍历的数据对象组合为一个索引序
    # sort是根据存储的数据长度按大到小排序，并只取出enumerate分配的索引
    return [x for x,y in sorted(enumerate(seq),
                                key = lambda x: len(x[1]),
                                reverse=True)]

def pad_array(a, max_length, PAD=constants.PAD):
    """
    a (array[int32])
    """
    return np.concatenate((a, [PAD]*(max_length - len(a))))

def pad_arrays(a):
    # 获取所有轨迹数据的最长值
    max_length = max(map(len, a))
    # 处理每个轨迹数据，让轨迹长度都=最长长度，使用PAD来填充
    a = [pad_array(a[i], max_length) for i in range(len(a))]
    a = np.stack(a).astype(np.int)
    return torch.LongTensor(a)

def pad_arrays_pair(src, trg):
    """
    src (list[array[int32]])
    trg (list[array[int32]])
    The length of the list equals to the batch size
    """
    assert len(src) == len(trg), "source and target should have the same length"
    # 首先分配索引，然后根据长度进行排序，返回排序后的索引序列
    idx = argsort(src)  # idx是根据轨迹长度排序后的编号（0-63，和轨迹的原始编号无关）
    # 让轨迹按长度从大到小组成数据集
    src = list(np.array(src)[idx])  # 重新排序后的轨迹，例如src[0]是这batch条轨迹中最长的一个
    trg = list(np.array(trg)[idx])
    # 并且获取每个轨迹的长度
    lengths = list(map(len, src))
    lengths = torch.LongTensor(lengths)
    # 处理每个轨迹数据，让长度相同，不够的用PAD填充
    src = pad_arrays(src)
    trg = pad_arrays(trg)
    # (batch, seq_len) => (seq_len, batch)
    # 输出转置，一列是一个轨迹序列，并且输出[LEN1,LEN2,LEN3,LEN4...]每个轨迹的长度
    return src.t().contiguous(), lengths.view(1, -1), trg.t().contiguous()


def pad_arrays_pair_keep_invp(src, trg, old_idx):
    '''
    :param src: (list[array[int32]])
    :param trg: (list[array[int32]])
    :param old_idx: 输入的batch条轨迹的原始序号
    :return:  The length of the list equals to the batch size
    '''
    assert len(src) == len(trg), "source and target should have the same length"
    idx = argsort(src)  # idx是根据轨迹长度排序后的编号（0-63，和轨迹的原始编号无关）
    src = list(np.array(src)[idx])  # 重新排序后的轨迹，例如src[0]是这batch条轨迹中最长的一个
    trg = list(np.array(trg)[idx])
    lengths = list(map(len, src))
    lengths = torch.LongTensor(lengths)
    src = pad_arrays(src)
    trg = pad_arrays(trg)
    # 获得轨迹的原始序号
    l = []
    for i in range(len(idx)):
        l.append(old_idx[idx[i]])  # 第i长的轨迹，是输入的第idx[i]个数据，它的原始轨迹序号是old_idx[idx[i]]
    # (batch, seq_len) => (seq_len, batch)
    l = torch.LongTensor(l)
    return src.t().contiguous(), lengths.view(1, -1), trg.t().contiguous(), l

def pad_arrays_pair_OnlyTrg(trg):
    """  
    src (list[array[int32]])
    trg (list[array[int32]])
    The length of the list equals to the batch size
    """
    idx = argsort(trg)

    trg = list(np.array(trg)[idx])
    lengths = list(map(len, trg))
    lengths = torch.LongTensor(lengths)

    trg = pad_arrays(trg)
    # (batch, seq_len) => (seq_len, batch)
    return lengths.view(1, -1), trg.t().contiguous()

def invpermute(p):
    """
    inverse permutation
    """
    p = np.asarray(p)
    # 创建相同shape的数组
    invp = np.empty_like(p)
    # p[i]是排序位置，i是本来位置，invp是一个反映射
    for i in range(p.size):
        invp[p[i]] = i
    return invp

def pad_arrays_keep_invp(src, label):
    """
    Pad arrays and return inverse permutation

    Input:
    src (list[array[int32]])
    ---
    Output:
    src (seq_len, batch)
    lengths (1, batch)
    invp (batch,): inverse permutation, src.t()[invp] gets original order
    """
    # 对轨迹数据长度进行排序，并返回按从大到小的轨迹idx，idx就是srcdata中轨迹的索引
    idx = argsort(src)
    # 将src按照长度从大到小排列
    src = list(np.array(src)[idx])
    # 针对每个轨迹获得长度列表，每个元素对应src中该位置的轨迹长度
    lengths = list(map(len, src))
    lengths = torch.LongTensor(lengths)
    # 填充较短的轨迹，一系列操作与训练时差不多，也就是处理数据的操作一样
    src = pad_arrays(src)
    # 反映射的索引，src可以根据此回到原来的排序前的src
    invp = torch.LongTensor(invpermute(idx))
    return src.t().contiguous(), lengths.view(1, -1), invp, label

class DataLoader():
    """
    srcfile: source file name
    trgfile: target file name
    batch: batch size
    validate: if validate = True return batch orderly otherwise return
        batch randomly
    """
    def __init__(self, srcfile, trgfile, batch, bucketsize, validate=False):
        self.srcfile = srcfile
        self.trgfile = trgfile
        self.batch = batch
        self.validate = validate
        self.bucketsize = bucketsize

    def insert(self, s, t):
        # 遍历事先设置好的桶
        for i in range(len(self.bucketsize)):
            # 将噪音轨迹和原始轨迹长度放入第一个满足小于指定桶的长度阈值的桶
            if len(s) <= self.bucketsize[i][0] and len(t) <= self.bucketsize[i][1]:
                self.srcdata[i].append(np.array(s, dtype=np.int32))
                self.trgdata[i].append(np.array(t, dtype=np.int32))
                return 1
        return 0

    def load(self, max_num_line=0):
        # 1*1,里面会放入一个个序列数组 [[]]
        # 根据桶的数量，每个桶都放了满足条件的轨迹序列[[],[],[]]为一个桶
        self.srcdata = [[] for _ in range(len(self.bucketsize))]
        self.trgdata = [[] for _ in range(len(self.bucketsize))]
        srcstream, trgstream = open(self.srcfile, 'r'), open(self.trgfile, 'r')
        num_line = 0
        # 两两匹配取出数据，每个trg会对应一个src轨迹
        for (s, t) in zip(srcstream, trgstream):
            # 轨迹序列是由” “划分，所以可以直接split取出
            s = [int(x) for x in s.split()]
            # 同上，并且给轨迹加上起始标志
            t = [constants.BOS] + [int(x) for x in t.split()] + [constants.EOS]
            # s, t是一个轨迹的序列点数组
            # 返回1则插入成功，否则0
            num_line += self.insert(s, t)
            # 最多max_num_line行数据
            if num_line >= max_num_line and max_num_line > 0: break
            if num_line % 100000 == 0:
                print("Read line {}".format(num_line))
        ## if vliadate is True we merge all buckets into one
        if self.validate == True:
            # [[][][][]]k*1，直接压缩, 也就是a[0] = b[0][0]
            self.srcdata = np.array(merge(*self.srcdata), dtype=object)
            self.trgdata = np.array(merge(*self.trgdata), dtype=object)
            self.start = 0
            self.size = len(self.srcdata)
        else:
            # shuffle
            for idx, (src, trg) in enumerate(zip(self.srcdata, self.trgdata)):
                data_l = len(src)
                shuffle = np.random.permutation(np.arange(data_l))
                self.srcdata[idx] = np.array(src, dtype=object)[shuffle]
                self.trgdata[idx] = np.array(trg, dtype=object)[shuffle]
            self.srcdata = list(map(np.array, self.srcdata))
            self.trgdata = list(map(np.array, self.trgdata))
            # 记录了每个桶含多少数据
            self.allocation = list(map(len, self.srcdata))
            # 记录了每个桶被选择的概率
            self.p = np.array(self.allocation) / sum(self.allocation)
        srcstream.close(), trgstream.close()

    def shuffle(self):
        self.index = 0
        for idx, (src, trg) in enumerate(zip(self.srcdata, self.trgdata)):
            data_l = len(src)
            shuffle = np.random.permutation(np.arange(data_l))
            self.srcdata[idx] = src[shuffle]
            self.trgdata[idx] = trg[shuffle]

    def getbatch_loader(self):
        if self.validate == True:
            src = self.srcdata[self.start:self.start+self.batch]
            trg = self.trgdata[self.start:self.start+self.batch]
            ## update `start` for next batch
            self.start += self.batch
            if self.start >= self.size:
                self.start = 0
            return pad_arrays_pair(src, trg)
        else:
            ## select bucket
            sample = np.random.multinomial(1, self.p)
            # 取出一个非零的，根据上code也就是随机选取一个桶
            bucket = np.nonzero(sample)[0][0]
            ## select data from the bucket
            # batch也就是实现设定好的batchsize，对指定的桶，根据里面的数量取出batch个数据，可能会重复，如果指定不重复需要引入参数false，那么需要轨迹数据大于batchsize可能报错
            # idx = np.random.choice(len(self.srcdata[bucket]), self.batch)  # 挑选的batch条轨迹数据的原始编号
            src = self.srcdata[bucket]
            trg = self.trgdata[bucket]
            if (self.index + 1) * self.batch < len(src):
                idx = np.arange(self.index * self.batch, (self.index + 1) * self.batch)
            else:
                idx = np.concatenate((np.arange(self.index * self.batch, len(src)),
                                     np.arange(0, (self.index + 1) * self.batch - len(src))), axis=0)
            self.index+=1

            # 取出指定索引的数据
            return pad_arrays_pair(src[idx], trg[idx])

    def getbatch_keep_invp(self):
        ## select bucket
        sample = np.random.multinomial(1, self.p)  # 从8份中随机挑选一个，如第2个
        bucket = np.nonzero(sample)[0][0]  # 随机挑选的一个的序号
        ## select data from the bucket
        idx = np.random.choice(len(self.srcdata[bucket]), self.batch)  # 挑选的batch条轨迹数据的原始编号
        return pad_arrays_pair_keep_invp(self.srcdata[bucket][idx], self.trgdata[bucket][idx], idx)

class DataOrderScaner():
    def __init__(self, srcfile, labelfile, batch):
        self.srcfile = srcfile
        self.labelfile = labelfile
        self.batch = batch
        self.srcdata = []
        self.labels = []
        self.start = 0
    def load(self, max_num_line=0):
        num_line = 0
        with open(self.srcfile, 'r') as srcstream:
            with open(self.labelfile, 'r') as labelstream:
                # 加载该文件，并按行读取每个轨迹
                for (s, label) in zip(srcstream, labelstream):
                    # 对一条轨迹按点拆分成序列
                    s = [int(x) for x in s.split()]
                    label = int(label)
                    self.srcdata.append(np.array(s, dtype=np.int32))
                    self.labels.append(label)
                    num_line += 1
                    if max_num_line > 0 and num_line >= max_num_line:
                        break
        self.size = len(self.srcdata)
        self.shuffle = np.random.permutation(np.arange(self.size))
        self.shuffle_invp = np.random.permutation(np.arange(self.size))
        for idx, to in enumerate(self.shuffle):
            self.shuffle_invp[to] = idx
        self.srcdata = np.array(self.srcdata, dtype=object)[self.shuffle]
        self.labels = np.array(self.labels, dtype=object)[self.shuffle]
        self.start = 0
    def getbatch_scaner(self, need_neghbor=False):
        """
        Output:
        src (seq_len, batch)
        lengths (1, batch)
        invp (batch,): inverse permutation, src.t()[invp] gets original order
        """
        # 当获取的batch大于当前大小返回
        if self.start >= self.size:
            if need_neghbor:
                return (None, None, None, None), (None, None, None, None)
            return None, None, None, None
        # 每次获取batch个数据
        src = self.srcdata[self.start:self.start+self.batch]
        labels = self.labels[self.start:self.start+self.batch]
        if need_neghbor:
            index = self.data_neighbors_before_shuffle[self.shuffle[self.start:self.start+self.batch]]
            index.astype('int64')
            neighbor = self.srcdata[self.shuffle_invp[index]]
            neighbor = neighbor.reshape(neighbor.shape[0] * neighbor.shape[1], -1).squeeze()
            self.start += self.batch
            return pad_arrays_keep_invp(src, labels), pad_arrays_keep_invp(neighbor, labels)
        self.start += self.batch
        
        return pad_arrays_keep_invp(src, labels)

    def get_data_num(self):
        return len(self.srcdata)

    def set_neighbor_data(self, cluster_data_neighbors):
        self.data_neighbors_before_shuffle = np.array(list(map(cluster_data_neighbors.get, range(len(cluster_data_neighbors)))))