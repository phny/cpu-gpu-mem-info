import os
import ast
import pandas as pd
import datetime
import time
import pynvml
import psutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import json


class CpuGpuMemoryInfo(object):
    def __init__(self, save_dir, pid=None, mode="auto"):
        # 启动模式(暂时未使用)
        self.mode = mode

        # 采样次数
        self.sample_nums = 0

        # 目标程序的pid
        self.pid = pid

        # 运行时间
        self.duration = 0

        # 采样间隔, 默认设置为5秒
        self.interval = 3

        # data dict, 记录cpu,gpu,memory数据
        self.data_dict = {}

        # 程序启动时间戳
        self.start_timestamp = int(time.time())
        self.time_plot_list = []

        # 保存位置
        self.save_dir = save_dir
        if (not os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir)

        #cpu个数
        self.cpu_nums = psutil.cpu_count()
       
        pynvml.nvmlInit()
        # 获取gpu个数
        self.gpu_nums = pynvml.nvmlDeviceGetCount()
        _handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        _info = pynvml.nvmlDeviceGetMemoryInfo(_handle)
         # 获取显卡的大小
        self.total_gpu_mem_size = _info.total / 1024 / 1024

        # 获取内存大小
        _mem_handle = psutil.virtual_memory()
        self.total_mem_size = _mem_handle.total / 1024 / 1024

        # gpu, cpu list
        self.cpu_list = [[] for i in range(self.cpu_nums)]
        self.gpu_list = [[] for i in range(self.gpu_nums)]
        self.mem_list = []

        # 添加各个子图
        self.fig = plt.figure(num=3, figsize=(20, 12), dpi=80)
        self.cpu_subplot = self.fig.add_subplot(3, 1, 1)
        self.gpu_subplot = self.fig.add_subplot(3, 1, 2)
        self.mem_subplot = self.fig.add_subplot(3, 1, 3)

    def monitor(self, args):
        """
        资源监控
        """
        print("start monitor")
        while True:
            # 时间散点
            now_time = int(time.time())
            self.time_plot_list.append(now_time - self.start_timestamp)
            # 输出采样的次数， 定时保存图像
            self.sample_nums += 1

            # 获取cpu信息
            cpu_used_rate_list = psutil.cpu_percent(interval=1, percpu=True)
            for cpu_ind, per_cpu_use_rate in enumerate(cpu_used_rate_list):
                self.cpu_list[cpu_ind].append(per_cpu_use_rate)
            # 清除原图像
            self.cpu_subplot.cla()
            # 设置cpu坐标轴属性
            self.cpu_subplot.set_title("CPU INFO")
            self.cpu_subplot.set_xlabel("Time")
            self.cpu_subplot.set_ylabel("Use Rate")
            self.cpu_subplot.grid()
            # 画出每个cpu的折线
            for ind, cpu in enumerate(self.cpu_list):
                # 画出原始cpu折线图
                self.cpu_subplot.plot(self.time_plot_list, cpu, label='cpu-' + str(ind))
                self.data_dict["cpu-" + str(ind)] = [self.time_plot_list, cpu]
            # 坐标自动调整
            self.cpu_subplot.autoscale()
            self.cpu_subplot.set_ylim(0, 110)
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
                self.gpu_subplot.plot(self.time_plot_list, gpu, label='gpu-' + str(ind))
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
            self.mem_subplot.plot(self.time_plot_list, self.mem_list, label='memory')
            self.data_dict['mem'] = [self.time_plot_list, self.mem_list]
            self.mem_subplot.autoscale()
            self.mem_subplot.set_ylim(0, mem_info.total / 1024 / 1024)
            # 设置图例位置,loc可以为[upper, lower, left, right, center]
            self.mem_subplot.legend(loc='right', shadow=True)

            # 输出采样次数
            if (self.sample_nums % 5 == 0 and self.sample_nums != 0):
                # 保存图像
                # plt.savefig(os.path.join(self.save_dir, "monitor-result.png"), bbox_inches = 'tight')
                # print("save figure success, sample count = {}, now time = {}".
                #       format(self.sample_nums, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                
                # 保存数据: ./monitor-result.csv
                self.save_data(self.data_dict, self.save_dir)

                # 检查目标进程是否已经退出
                try:
                    psutil.Process(self.pid)
                except Exception as e:
                    # 进程退出, 画图保存， 并结束监控
                    self.draw('./monitor-result.csv', fitting=True)
                    self.draw('./monitor-result.csv', fitting=False)
                    print("pid = {} not found, finished monitor now， total run time: {} s". \
                            format(self.pid, time.time() - self.start_timestamp))
                    print("finished monitor")
                    exit(1)
                finally:
                    # 进程未退出，画图保存, 继续监控
                    self.draw('./monitor-result.csv', fitting=True)
                    self.draw('./monitor-result.csv', fitting=False)

            # 暂停 interval 秒
            plt.pause(self.interval)
            print("sample nums: {}, now time: {}".format(self.sample_nums, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

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
        yvals = p1(x_list)  # 拟合y值

        return x_list, yvals

    def save_data(self, dict, save_dir):
        """
        将数据保存到文件, 方便后续画图分析
        """
        file_name = "monitor-result.csv"
        pd.DataFrame.from_dict(dict, orient='index').T.to_csv(
            os.path.join(save_dir, file_name), index=False)
        print("save data to file: {} success, now time = {}".format(file_name,
                                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    def load_data(self, data_path):
        """
        从文件读取数据
        """
        file = pd.read_csv(data_path)
        df = pd.DataFrame(file)
        data_dict = df.to_dict()
        return data_dict

    def draw_cpu_image(self, data_path, fitting):
        """
        cpu图像
        """
        # 清除原图像
        self.cpu_subplot.cla()
        # 设置cpu坐标轴属性
        self.cpu_subplot.set_title("CPU INFO")
        self.cpu_subplot.set_xlabel("Time")
        self.cpu_subplot.set_ylabel("Use Rate")
        self.cpu_subplot.grid()

        data_dict = self.load_data(data_path)

        # 从数据中获取cpu个数
        cpu_count = 0
        for i in data_dict.keys():
            if (str(i).startswith("cpu")):
                cpu_count += 1

        mean_cpu = 0
        max_cpu = 0
        min_cpu = 100

        for i in range(cpu_count):
            x_list = ast.literal_eval(data_dict["cpu-" + str(i)][0])
            # x_list = [item * self.interval for item in x_list]

            y_list = ast.literal_eval(data_dict["cpu-" + str(i)][1])

            # min， max， mean
            mean_cpu += np.mean(y_list)
            if (max(y_list) > max_cpu):
                max_cpu = max(y_list)
            if (min(y_list) < min_cpu):
                min_cpu = min(y_list)
            if fitting:
                # 对cpu的数据进行3次多项式拟合，去掉其中不合理的峰值
                x_list_new, y_list_new = self.data_fitting(x_list, y_list)
            else:
                # 不拟合
                x_list_new = x_list
                y_list_new = y_list

            # self.cpu_subplot.plot(x_list, y_list, 's', label='original values')
            self.cpu_subplot.plot(x_list_new, y_list_new,
                                  label='cpu-' + str(i))
        print("min cpu = {}%".format(min_cpu))
        print("max cpu = {}%".format(max_cpu))
        print("total mean cpu = {}%".format(mean_cpu / cpu_count))
        # 坐标自动调整
        self.cpu_subplot.set_xlim(0, x_list[len(x_list) - 1])
        # 设置成110，当cpu 100%的时候显示好看点
        self.cpu_subplot.set_ylim(0, 110)
        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        # self.cpu_subplot.legend(loc='right', shadow=True) # 去掉， 因为当cpu为64个的时候， 长度超过图表限制了
        print("finished draw cpu image")

    def draw_gpu_image(self, data_path, fitting):
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

        data_dict = self.load_data(data_path)

        # 从数据中获取cpu个数
        gpu_count = 0
        for i in data_dict.keys():
            if (str(i).startswith("gpu")):
                gpu_count += 1

        # 记录最大最小平均的gpu
        mean_gpu = 0
        max_gpu = 0
        min_gpu = 100000000

        for i in range(gpu_count):
            x_list = ast.literal_eval(data_dict["gpu-" + str(i)][0])
            y_list = ast.literal_eval(data_dict["gpu-" + str(i)][1])
            
            # gpu数据拟合
            # x_list_new, y_list_new = self.data_fitting(x_list, y_list)
            # 记录gpu的最大最小以及平均值
            mean_gpu += np.mean(y_list)
            if (max(y_list) > max_gpu):
                max_gpu = max(y_list)
            if (min(y_list) < min_gpu):
                min_gpu = min(y_list)
            # 画gpu图像
            self.gpu_subplot.plot(x_list, y_list, label='gpu-' + str(i))
            # self.gpu_subplot.plot(x_list_new, y_list_new, label='gpu-' + str(i))
        print("max gpu = {}".format(max_gpu))
        print("avg gpu = {}".format(mean_gpu / gpu_count))
        print("min gpu = {}".format(min_gpu))
        print("finished draw gpu image")
        # 设置gpu坐标轴范围
        self.gpu_subplot.set_xlim(0, x_list[len(x_list) - 1])
        if (self.mode == "manual"):
            f = open("./config.json", "r")
            d = json.load(f)
            self.total_gpu_mem_size = d["gpu"]["max_gpu_size"]
        self.gpu_subplot.set_ylim(0, self.total_gpu_mem_size)
        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        self.gpu_subplot.legend(loc='right', shadow=True)

    def draw_mem_image(self, data_path, fitting):
        """
        memory图像
        """
        # 清除原图像
        self.mem_subplot.cla()
        self.mem_subplot.set_title("Memory Info")
        self.mem_subplot.set_xlabel("Time")
        self.mem_subplot.set_ylabel("Memory(Mib)")
        self.mem_subplot.grid()
        data_dict = self.load_data(data_path)
        x_list = ast.literal_eval(data_dict["mem"][0])
        # x_list = [item * self.interval for item in x_list]
        y_list = ast.literal_eval(data_dict["mem"][1])

        print("min memory = %d" % min(y_list))
        print("max memory = %d" % max(y_list))
        print("avg memory = %d" % np.mean(y_list))

        self.mem_subplot.plot(x_list, y_list, label='memory')
        self.mem_subplot.set_xlim(0, x_list[len(x_list) - 1])
        if (self.mode == "manual"):
            # 从配置文件获取total_mem_size
            f = open("./config.json", "r")
            d = json.load(f)
            self.total_mem_size = d["memory"]["max_mem_size"]
        self.mem_subplot.set_ylim(0, self.total_mem_size)
        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        self.mem_subplot.legend(loc='right', shadow=True)
        print("finished draw memory image")

    def save_image(self, fitting):
        """
        保存图像
        """
        if fitting:
            plt.savefig(os.path.join(self.save_dir, "monitor-result-fitting.png"), bbox_inches = 'tight')
        else:
            plt.savefig(os.path.join(self.save_dir, "monitor-result.png"), bbox_inches = 'tight')

    def run(self, args):
        """
        程序入口
        """
        try:
            while True:
                # 打开交互模式
                plt.ion()
                # 进行画图
                self.monitor(args)
                # 关闭交互模式
                plt.ioff()
        except KeyboardInterrupt:
            plt.savefig(os.path.join(self.save_dir, "monitor-result-fitting.png"), bbox_inches = 'tight')
            print("save to %s" % self.save_dir)
            print("total run time: {} s".format(time.time() - self.start_timestamp))

    def draw(self, data_path, fitting=False):
        """画出cpu，gpu，memory的图像并保存
        """
        self.draw_cpu_image(data_path, fitting)
        self.draw_gpu_image(data_path, fitting)
        self.draw_mem_image(data_path, fitting)
        self.save_image(fitting)


def main():
    """
    根据提供的进程pid开启监控系统的cpu,gpu,memory， 进程结束则程序画图保存退出
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_dir", metavar="monitor result save dir", type=str,
                        required=False, dest="save_dir", help="with dir name: ./monitor-result", default="./monitor-result")
    parser.add_argument("-p", "--pid", metavar="process pid", type=int, dest="pid", help="目前程序的PID，可以不提供", 
                        required=False, default=1)
    
    args = parser.parse_args()

    monitor = CpuGpuMemoryInfo(args.save_dir, args.pid)
    monitor.run(args)


if __name__ == "__main__":
    # python3.6 cpu-gpu-mem-monitor.py --save_dir=./out --pid=target_pid
    main()
