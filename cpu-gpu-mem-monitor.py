import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil
import pynvml
import time
import datetime
import pandas as pd
import ast
import numpy as np


class CpuGpuMemoryInfo(object):

    def __init__(self, save_location):
        # 采样次数
        self.sample_nums = 0

        # 采样间隔, 默认设置为5秒
        self.interval = 5
        
        # data dict
        self.data_dict = {}

        # 程序启动时间
        self.start_time = datetime.datetime.fromtimestamp(time.time())
        self.time_plot_list = []

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
        """
        资源监控
        """
        print("start monitor")
        while True:
            # 时间散点+1
            self.time_plot_list.append(self.sample_nums)
            # 输出采样的次数， 定时保存图像
            self.sample_nums += 1

            # 获取cpu信息
            cpu_used_rate_list = psutil.cpu_percent(interval=0.0, percpu=True)
            for cpu_ind, per_cpu_use_rate in enumerate(cpu_used_rate_list):
                self.cpu_list[cpu_ind].append(per_cpu_use_rate)
            # 清除原图像
            self.cpu_subplot.cla()
            #设置cpu坐标轴属性
            self.cpu_subplot.set_title("CPU INFO")
            self.cpu_subplot.set_xlabel("Time")
            self.cpu_subplot.set_ylabel("Use Rate")
            self.cpu_subplot.grid()
            # 画出每个cpu的折线
            for ind, cpu in enumerate(self.cpu_list):
                self.cpu_subplot.scatter(self.time_plot_list, cpu, label='cpu-' + str(ind))
                self.data_dict["cpu-" + str(ind)] = [self.time_plot_list, cpu]
            # 坐标自动调整
            self.cpu_subplot.autoscale()
            self.cpu_subplot.set_ylim(0, 100)
            # 设置图例位置,loc可以为[upper, lower, left, right, center]
            self.cpu_subplot.legend(loc='right', shadow=True)

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
                self.data_dict['gpu-'+str(ind)] = [self.time_plot_list, gpu]

            # gpu坐标自动调整
            self.gpu_subplot.autoscale()
            self.gpu_subplot.set_ylim(0, self.total_gpu_mem_size)
            # 设置图例位置,loc可以为[upper, lower, left, right, center]
            self.gpu_subplot.legend(loc='right', shadow=True)

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
            self.data_dict['mem'] = [self.time_plot_list, self.mem_list]
            self.mem_subplot.autoscale()
            self.mem_subplot.set_ylim(0, mem_info.total / 1024 / 1024)
            # 设置图例位置,loc可以为[upper, lower, left, right, center]
            self.mem_subplot.legend(loc='right', shadow=True)

            # 输出采样次数
            if (self.sample_nums % 10 == 0 and self.sample_nums != 0):
                plt.savefig(self.save_location)
                print("save figure success, sample count = {}, now time = {}".\
                     format(self.sample_nums, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                # 保存数据
                self.save_data(self.data_dict)

            # 采样间隔
            plt.pause(self.interval)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    def take_sample_plots(self, x_list, y_list, n):
        """
        样本点采样
        """
        # 判断ｘ, y 坐标数据的长度是否相等
        if (len(x_list) != len(y_list)):
            print("x list length not equal to y list ")
            exit(-1)
        if (len(x_list) < n and len(y_list) < n):
            return x_list, y_list

        # 采样步长
        sample_step = len(x_list) / n

        # 按照步长采样
        x_list = [item for ind, item in enumerate(x_list) if ind % sample_step == 0 ]
        y_list = [item for ind, item in enumerate(y_list) if ind % sample_step == 0 ]
        print(x_list)
        print(y_list)

        return x_list, y_list 

    def data_fitting(self, x_list, y_list):
        """
        对数据进行拟合
        """
        y = np.array(y_list)
        
        # 用3次多项式拟合
        f1 = np.polyfit(x_list, y, 3)
        p1 = np.poly1d(f1)
        # print(p1)
        
        # 也可使用yvals=np.polyval(f1, x)
        yvals = p1(x_list)  #拟合y值

        return x_list, yvals


    def save_data(self, dict):
        """
        将数据保存到文件, 方便后续画图分析
        """
        file_name = "cpu-gpu-mem-info.csv"
        pd.DataFrame.from_dict(dict, orient='index').T.to_csv(file_name, index=False)
        print("save data to file: {} success, now time = {}".format(file_name, 
                            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    def load_data(self):
        """
        从文件读取数据
        """
        file_name = "dbscan.csv"
        file = pd.read_csv(file_name)
        df = pd.DataFrame(file)
        data_dict = df.to_dict()
        return data_dict
    
    def draw_cpu_image(self):
        """
        cpu图像
        """
        # 清除原图像
        self.cpu_subplot.cla()
        #设置cpu坐标轴属性
        self.cpu_subplot.set_title("CPU INFO")
        self.cpu_subplot.set_xlabel("Time")
        self.cpu_subplot.set_ylabel("Use Rate")
        self.cpu_subplot.grid()
        
        data_dict = self.load_data()

        # 从数据中获取cpu个数
        cpu_count = 0
        for i in data_dict.keys():
            if (str(i).startswith("cpu")):
                cpu_count += 1

        cpu_mean = 0

        for i in range(cpu_count):
            x_list = ast.literal_eval(data_dict["cpu-" + str(i)][0])
            x_list = [item * self.interval for item in x_list]

            y_list = ast.literal_eval(data_dict["cpu-" + str(i)][1])
            # print("max cpu-" + str(i) + " = %d" % max(y_list))
            # print("min cpu-" + str(i) + " = %d" % min(y_list))
            # print("avg cpu-" + str(i) + " = %d" % np.mean(y_list))
            cpu_mean += np.mean(y_list)

            x_list_new, y_list_new = self.data_fitting(x_list, y_list)

           #  self.cpu_subplot.plot(x_list, y_list, 's', label='original values')
            self.cpu_subplot.plot(x_list_new, y_list_new, label='cpu-' + str(i))
        print("total mean cpu = {}".format(cpu_mean / cpu_count))
        # 坐标自动调整
        self.cpu_subplot.set_xlim(0, x_list[len(x_list) - 1])
        self.cpu_subplot.set_ylim(0, 100)
        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        # self.cpu_subplot.legend(loc='right', shadow=True)
        print("finished draw cpu image")

    def draw_gpu_image(self):
        """
        gpu图像
        """
        #　清除原图像
        self.gpu_subplot.cla()
        # 设置坐标轴属性
        self.gpu_subplot.set_title("GPU INFO")
        self.gpu_subplot.set_xlabel("Time")
        self.gpu_subplot.set_ylabel("Used(Mib)")
        self.gpu_subplot.grid()

        data_dict = self.load_data()

        # 从数据中获取cpu个数
        gpu_count = 0
        for i in data_dict.keys():
            if (str(i).startswith("gpu")):
                gpu_count += 1

        for i in range(gpu_count):
            x_list = ast.literal_eval(data_dict["gpu-" + str(i)][0])
            x_list = [item * self.interval for item in x_list]

            y_list = ast.literal_eval(data_dict["gpu-" + str(i)][1])
            # print("max gpu-" + str(i) + " = %d" % max(y_list))
            # print("min gpu-" + str(i) + " = %d" % min(y_list))
            # print("avg gpu-" + str(i) + " = %d" % np.mean(y_list))
            x_list_new, y_list_new = self.data_fitting(x_list, y_list)

            self.gpu_subplot.plot(x_list, y_list, label='gpu-' + str(i))
            # self.gpu_subplot.plot(x_list_new, y_list_new, label='gpu-' + str(i))
        gpu_0 = ast.literal_eval(data_dict["gpu-0"][1])
        print("gpu-0 max = {}".format(max(gpu_0)))
        print("gpu-0 avg = {}".format(np.mean(gpu_0)))
        print("gpu-0 min = {}".format(min(gpu_0)))
        print("finished draw gpu image")


        self.gpu_subplot.set_xlim(0, x_list[len(x_list) - 1])
        self.gpu_subplot.set_ylim(0, 12288)
        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        self.gpu_subplot.legend(loc='right', shadow=True)

    def draw_mem_image(self):
        """
        memory图像
        """
        # 清除原图像
        self.mem_subplot.cla()
        self.mem_subplot.set_title("Memory Info")
        self.mem_subplot.set_xlabel("Time")
        self.mem_subplot.set_ylabel("Memory(Mib)")
        self.mem_subplot.grid()
        data_dict = self.load_data()
        x_list = ast.literal_eval(data_dict["mem"][0])
        x_list = [item * self.interval for item in x_list]
        y_list = ast.literal_eval(data_dict["mem"][1])

        print("min memory = %d" % min(y_list))
        print("max memory = %d" % max(y_list))
        print("avg memory = %d" % np.mean(y_list))

        self.mem_subplot.plot(x_list, y_list, label='memory')
        self.mem_subplot.set_xlim(0, x_list[len(x_list) - 1])
        self.mem_subplot.set_ylim(0, 512 * 1024)
        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        self.mem_subplot.legend(loc='right', shadow=True)
        print("finished draw memory image")


    def save_image(self):
        """
        保存图像
        """
        plt.savefig(self.save_location)

    def run(self):
        """
        程序入口
        """
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

    def draw(self):
        self.draw_cpu_image()
        self.draw_gpu_image()
        self.draw_mem_image()
        self.save_image()


if __name__ == "__main__":
    monitor = CpuGpuMemoryInfo("./test.png")
    # monitor.run()
    monitor.draw()
    # data_dict = monitor.load_data()
    # print(ast.literal_eval(data_dict["mem"][1]))
    # print(len(ast.literal_eval(data_dict["mem"][1])))
    