from cpu_gpu_mem_monitor import CpuGpuMemoryInfo

import json
import argparse
import matplotlib.pyplot as plt
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


def main():
    """
    根据提供的进程pid开启监控系统的cpu,gpu,memory， 进程结束则程序画图保存退出
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_dir", metavar="monitor result save dir", type=str,
                        required=False, dest="save_dir", help="with dir name: ./monitor-result", default="./monitor-result")
    parser.add_argument("-d", "--data_path", metavar="input data path", type=str, dest="data_path", help="用于画图的数据",
                        required=False, default=1)

    args = parser.parse_args()

    monitor = CpuGpuMemoryInfo(args.save_dir, pid=None, mode="manual")
    monitor.manual_draw(args)


if __name__ == "__main__":
    main()
