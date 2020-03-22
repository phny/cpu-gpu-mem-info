import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil
import pynvml
import time
import datetime


class CpuGpuMemoryInfo(object):

    def __init__(self, save_location):
        # 采样次数
        self.sample_nums = 0

        # 程序启动时间
        self.start_time = datetime.datetime.fromtimestamp(time.time())

        # 保存位置
        self.save_location = save_location

        #cpu, gpu个数
        self.cpu_nums = psutil.cpu_count()
        pynvml.nvmlInit()

        # 获取显卡的大小
        self.gpu_nums = pynvml.nvmlDeviceGetCount()
        _handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        _info = pynvml.nvmlDeviceGetMemoryInfo(_handle)
        self.total_gpu_mem_size = _info.total / 1024 / 1024
        
        # gpu, cpu list
        self.cpu_list = [[] for i in range(self.cpu_nums)]
        self.gpu_list = [[] for i in range(self.gpu_nums)]
        self.mem_list = []

        # 添加各个子图
        self.fig = plt.figure(num=3, figsize=(20, 12),dpi=80)
        self.cpu_subplot = self.fig.add_subplot(3,1,1)  
        self.gpu_subplot = self.fig.add_subplot(3,1,2)
        self.mem_subplot = self.fig.add_subplot(3,1,3)

    def monitor(self):
        print("start monitor")
        while True:
            # 获取cpu信息
            cpu_used_rate_list = psutil.cpu_percent(interval=1, percpu=True)
            for ind, per_cpu_use_rate in enumerate(cpu_used_rate_list):
                self.cpu_list[ind].append(per_cpu_use_rate)
            # 清除原图像
            self.cpu_subplot.cla()
            #设置cpu坐标轴属性
            self.cpu_subplot.set_title("CPU INFO")
            self.cpu_subplot.set_xlabel("Time")
            self.cpu_subplot.set_ylabel("Use Rate")
            self.cpu_subplot.grid()
            # 画出每个cpu的折线
            for ind, cpu in enumerate(self.cpu_list):
                #x, y = self.take_sample_plots(cpu, 100)
                self.cpu_subplot.plot(cpu, label='cpu-' + str(ind))
                # self.cpu_subplot.plot(x, y, label='cpu-' + str(ind))
            # 坐标自动调整
            self.cpu_subplot.autoscale()
            self.cpu_subplot.set_ylim(0, 100)
            # 设置图例位置,loc可以为[upper, lower, left, right, center]
            self.cpu_subplot.legend(loc='best', shadow=True)

            # 获取显存信息
            for gpu_ind in range(self.gpu_nums):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_ind)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.gpu_list[gpu_ind].append(info.used / 1024 / 1024)
            #　清除原图像
            self.gpu_subplot.cla()
            # 设置坐标轴属性
            self.gpu_subplot.set_title("GPU INFO")
            self.gpu_subplot.set_xlabel("Time")
            self.gpu_subplot.set_ylabel("Used(Mib)")
            self.gpu_subplot.grid()
             # 画出每个gpu的折线
            for ind, gpu in enumerate(self.gpu_list):
                self.gpu_subplot.plot(gpu, label='gpu-' + str(ind))
            # gpu坐标自动调整
            self.gpu_subplot.autoscale()
            self.gpu_subplot.set_ylim(0, self.total_gpu_mem_size)
            # 设置图例位置,loc可以为[upper, lower, left, right, center]
            self.gpu_subplot.legend(loc='best', shadow=True)

            # 获取内存信息
            mem_info = psutil.virtual_memory()
            self.mem_list.append(mem_info.used / 1024 / 1024)
            # 清除原图像
            self.mem_subplot.cla()
            self.mem_subplot.set_title("Memory Info")
            self.mem_subplot.set_xlabel("Time")
            self.mem_subplot.set_ylabel("Memory(Mib)")
            self.mem_subplot.grid()
            self.mem_subplot.plot(self.mem_list, label='memory')
            self.mem_subplot.autoscale()
            self.mem_subplot.set_ylim(0, mem_info.total / 1024 / 1024)
            # 设置图例位置,loc可以为[upper, lower, left, right, center]
            self.mem_subplot.legend(loc='best', shadow=True)

            # 输出采样的次数， 定时保存图像
            self.sample_nums += 1
            if (self.sample_nums % 10 == 0 and self.sample_nums != 0):
                plt.savefig(self.save_location)
                print("save success, sample count = %d" %  self.sample_nums)

            # 采样间隔
            plt.pause(0.5)


    def run(self):
        try:
            while True:
                # 打开交互模式
                plt.ion()
                # 进行画图
                self.monitor()
                # 关闭交互模式
                plt.ioff()
        except KeyboardInterrupt:
            plt.savefig(self.save_location)
            print("save to %s" % self.save_location)

    # def take_sample_plots(self, src_list, n):
    #     if len(src_list) <= n:
    #         date_plot_list = [str(datetime.datetime.fromtimestamp(self.start_time.timestamp() + i)) for i in src_list]
    #         return date_plot_list, src_list
    #     else:
    #         sample_step = len(src_list) / n
    #         if (n <= 0):
    #             print("n is: %d, invalid" % n)
    #         sample_plot_list = [item for ind, item in enumerate(src_list) if ind % sample_step == 0]
    #         date_plot_list = [self.start_time.timestamp() + i for i in sample_plot_list]
    #         date_plot_list = [str(datetime.datetime.fromtimestamp(i)) for i in date_plot_list]
    #         src_list = [i for ind, i in enumerate(src_list) if ind % sample_step == 0 ]

    #         if len(src_list) != len(date_plot_list):
    #             print("error")
    #             exit(-1)
    #         return date_plot_list, src_list


if __name__ == "__main__":
    monitor = CpuGpuMemoryInfo("./out.png")
    monitor.run()