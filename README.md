## 一个用于检测主机 cpu， gpu，内存的使用情况的简单的脚本

## 运行

```shell
# 自动监控模式：
python3.6 cpu-gpu-mem-monitor.py --save_location=./out/result.png --pid=1234

# 手动对数据进行画图
# 手动画图需要在config.json中填写max_gpu_size以及max_mem_size的值（因为从数据中无法得知显存以及内存的最大值）
python3.6 draw_manual.py --save_dir=./out --data_path=./data.csv
```
