import math
import numpy as np

def isArea(start, end, timestamp):
    if start < end:
        return start <= timestamp <= end
    else:
        return timestamp >= start or timestamp <= end
    
def calAngle(pos, total_pos):
    # return 1 / 2 - ((pos + 1) / total_pos - 1 / total_pos) + 2
    return 1/2 - 2 * (pos / total_pos)
    
def cfe(region, timestamp):
    total_pos = 86400 // region.timestep
    # pos 指示属于第几块区间
    if region.mintime > timestamp:
        pos = (86400 - region.mintime + timestamp) // region.timestep
    else:
        pos = (timestamp - region.mintime) // region.timestep
    start = region.mintime + pos * region.timecellsize # 区间开始
    if start > 86400:
        start -= 86400
    end = start + region.timecellsize
    if end > 86400:
        end -= 86400
    angle = calAngle(pos, total_pos)
    # return np.array(math.cos(math.pi * angle), math.sin(math.pi * angle))
    return math.pi * angle
    # l_fuzzy_l = start - region.timefuzzysize # 左模糊边界-左
    # l_fuzzy_r = start + region.timefuzzysize # 左模糊边界-右
    # if l_fuzzy_r > 86400:
    #     l_fuzzy_r -= 86400
    # if l_fuzzy_l < 0:
    #     l_fuzzy_l += 86400
    # r_fuzzy_l = end - region.timefuzzysize # 右模糊边界-左
    # r_fuzzy_r = end + region.timefuzzysize # 右模糊边界-右
    # if r_fuzzy_r > 86400:
    #     r_fuzzy_r -= 86400
    # if r_fuzzy_l < 0:
    #     r_fuzzy_l += 86400
    # 
    # # 确定是否处于模糊界，然后按照模糊度算坐标，否则坐标明确
    # angle = calAngle(pos, total_pos)
    # interval = np.array(math.cos(math.pi * angle), math.sin(math.pi * angle))
    # if isArea(l_fuzzy_l, l_fuzzy_r, timestamp):
    #     l_pos = pos - 1
    #     if l_pos < 0:
    #         l_pos = total_pos - 1
    #     l_angle = calAngle(l_pos, total_pos)
    #     l_interval = np.array(math.cos(math.pi * l_angle), math.sin(math.pi * l_angle))
    #     # 计算模糊度，分左区间和当前区间
    #     # 左区间
    #     l_fuzzy_degree = l_fuzzy_r - timestamp
    #     if l_fuzzy_degree < 0:
    #         l_fuzzy_degree += 86400
    #     # 当前区间
    #     fuzzy_degree = timestamp - l_fuzzy_l
    #     if fuzzy_degree < 0:
    #         fuzzy_degree += 86400
    #     # 归一化
    #     l_fuzzy_degree /= (region.timefuzzysize * 2)
    #     fuzzy_degree /= (region.timefuzzysize * 2)
    #     return l_fuzzy_degree * l_interval + fuzzy_degree * interval
    # elif isArea(l_fuzzy_r, r_fuzzy_l, timestamp):
    #     # 坐标明确，是该区间的极坐标
    #     return interval
    # elif isArea(r_fuzzy_l, r_fuzzy_r, timestamp):
    #     r_pos = pos + 1
    #     if r_pos >= total_pos:
    #         r_pos = 0
    #     r_angle = calAngle(r_pos, total_pos)
    #     r_interval = np.array(math.cos(math.pi * r_angle), math.sin(math.pi * r_angle))
    #     # 计算模糊度，分右区间和当前区间
    #     # 右区间
    #     r_fuzzy_degree = timestamp - r_fuzzy_l
    #     if r_fuzzy_degree < 0:
    #         r_fuzzy_degree += 86400
    #     # 当前区间
    #     fuzzy_degree = r_fuzzy_r - timestamp
    #     if fuzzy_degree < 0:
    #         fuzzy_degree += 86400
    #     # 归一化
    #     r_fuzzy_degree /= (region.timefuzzysize * 2)
    #     fuzzy_degree /= (region.timefuzzysize * 2)
    #     return r_fuzzy_degree * r_interval + fuzzy_degree * interval
