import numpy as np
import matplotlib.pyplot as plt
from typing import AnyStr, List

def draw_trace(fn: List[AnyStr]):
    lines_list = list()
    for fni in fn:
        with open(fni, "rt") as f:
            lines = f.readlines()
            lines_list.append(lines)

    s1 = 'avg(len(game))='
    for lines in lines_list:
        vals = [np.float32(l[len(s1):len(s1) + 5]) for l in lines if l.startswith(s1)]
        x = range(len(vals))
        plt.plot(x, vals)

    plt.show()

    s2="avg(game reward)="
    for lines in lines_list:
        vals1 = [np.float32(l[l.find(s2)+len(s2):]) for l in lines if l.find(s2)>=0]
        x1 = range(len(vals1))
        plt.plot(x1, vals1)
    plt.show()

    s3 = "avg(game_figures)="
    a1 = False
    for lines in lines_list:
        vals2 = [np.float32(l[l.find(s3)+len(s3):l.find(s3)+len(s3)+5]) for l in lines if l.find(s3)>=0]
        if vals2:
            a1 = True
            x2 = range(len(vals2))
            plt.plot(x2, vals2)
    if a1:
        plt.show()

    pass

def main():
    draw_trace(['/tmp/tetris/v018/trace_190522_05.txt','/tmp/tetris/v018/trace_190601_16.txt'])


if __name__=="__main__":
    main()
