import psutil
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    #开启一个窗口，num设置子图数量，这里如果在add_subplot里写了子图数量，num设置多少就没影响了
    #figsize设置窗口大小，dpi设置分辨率
    fig = plt.figure(num=2, figsize=(15, 8),dpi=80)
    #使用add_subplot在窗口加子图，其本质就是添加坐标系
    #三个参数分别为：行数，列数，本子图是所有子图中的第几个，最后一个参数设置错了子图可能发生重叠
    ax1 = fig.add_subplot(2,1,1)  
    ax2 = fig.add_subplot(2,1,2)
    ax1.set_xlim(0, 30)
    #绘制曲线 
    ax1.plot(np.arange(0,1,0.1),range(0,10,1),color='g')
    #同理，在同一个坐标系ax1上绘图，可以在ax1坐标系上画两条曲线，实现跟上一段代码一样的效果
    ax1.plot(np.arange(0,1,0.1),range(0,20,2),color='b')
    #在第二个子图上画图
    ax2.plot(np.arange(0,1,0.1),range(0,20,2),color='r')
    plt.show()