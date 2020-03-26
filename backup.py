# coding:utf-8

import os
import sys
import psutil as p
import commands
import argparse


def count_process_mem_gpu(processName):
    cmd = 'ps -aux | grep ' + processName + \
        '| grep -v grep ' + '| grep -v ' + sys.argv[0]
    mem_status, r = commands.getstatusoutput(cmd)

    r_list = filter(None, r.split(' '))

    mem = 0.0
    gpu = 0.0

    # flag to mean whther the process if over
    finished_flag = False

    if mem_status != 0:
        finished_flag = True
        return (mem, gpu, finished_flag)

    if len(r_list) >= 7:
        tmp_mem = r_list[5]
        if tmp_mem.endswith('g'):
            mem = float(tmp_mem[:-1]) * 1024
        elif tmp_mem.endswith('m'):
            mem = float(tmp_mem[:-1])
        else:
            mem = float(tmp_mem) / 1024
    cmd_gpu = 'nvidia-smi | grep ' + processName
    gpu_status, r_gpu = commands.getstatusoutput(cmd_gpu)
    gpu_list = filter(None, r_gpu.split(' '))
    if len(gpu_list) >= 6:
        gpu = gpu + float(gpu_list[5][:-3])
    return (mem, gpu, finished_flag)


def check_process(processName):
    cmd = 'ps -aux | grep ' + processName + \
        '| grep -v grep' + '| grep -v ' + sys.argv[0]
    status, out = commands.getstatusoutput(cmd)
    if status != 0:
        # process not exist
        return False
    return True


def mem_info_show_mode(processName, location, save_flag, mem_list, gpu_list):
    import matplotlib.pyplot as plt
    check_result = check_process(processName)
    if not check_result:
        print('Process: [%s] is not exist!!!' % processName)
        sys.exit(0)
    fig, ax = plt.subplots(figsize=(12, 8), ncols=1, nrows=2)
    plt.grid(True)
    mem_y = []
    gpu_y = []

    labels = ['max mem', 'min mem', 'avg mem', 'max gpu', 'min gpu', 'avg gpu']
    time = 0.0
    x = []
    try:
        while True:
            result = count_process_mem_gpu(processName)
            if result[2] == False:
                # process is running
                mem_list.append(round(result[0], 2))
                gpu_list.append(round(result[1], 2))

                ax[0].cla()
                ax[0].grid()
                ax[0].set_xlabel('time(second)')
                ax[0].set_ylabel('memory(Mib)')
                ax[0].set_ylim(0, max([max(mem_list), max(gpu_list)]) * 1.5)
                ax[0].set_xlim(0, time + 10)

                x.append(time)
                mem_y.append(round(result[0], 2))
                gpu_y.append(round(result[1], 2))

                ax[0].plot(x, mem_y, label='memory', color='red')
                ax[0].plot(gpu_y, label='gpu', color='blue')
                ax[0].legend(loc='lower right')

                # per 0.5 second get mem info
                ax[1].cla()
                ax[1].set_ylim(0, max(mem_list) + 50)
                ax[1].bar([1, 2, 3, 4, 5, 6], [max(mem_list), min(mem_list), sum(mem_list) / len(mem_list), max(gpu_list), min(
                    gpu_list), sum(gpu_list) / len(gpu_list)], 0.2, label='memory', color='#87CEFA', align='center', alpha=0.8)
                ax[1].set_xticks([1, 2, 3, 4, 5, 6])
                ax[1].set_xticklabels(labels)
                ax[1].set_ylabel('memory(Mib)')
                ax[1].set_ylim(0, max([max(mem_list), max(gpu_list)]) * 1.5)
                ax[1].text(1, max(mem_list), '')
                ax[1].text(1, max(mem_list), max(mem_list), ha='center')
                ax[1].text(2, min(mem_list), '')
                ax[1].text(2, min(mem_list), min(mem_list), ha='center')
                ax[1].text(3, sum(mem_list) / len(mem_list), '')
                ax[1].text(3, sum(mem_list) / len(mem_list),
                           round(sum(mem_list) / len(mem_list), 2), ha='center')
                ax[1].text(4, max(gpu_list), '')
                ax[1].text(4, max(gpu_list), max(gpu_list), ha='center')
                ax[1].text(5, min(gpu_list), '')
                ax[1].text(5, min(gpu_list), min(gpu_list), ha='center')
                ax[1].text(6, sum(gpu_list) / len(gpu_list), '')
                ax[1].text(6, sum(gpu_list) / len(gpu_list),
                           round(sum(gpu_list) / len(gpu_list), 2), ha='center')
                plt.pause(0.5)

                time = time + 0.5
            else:
                # process is finished
                print('process ' + processName + ' is finished, now exit')
                if save_flag != 'no' and location != 'no':
                    ax[0].set_xlim(0, time + 10)
                    ax[0].set_ylim(
                        0, max([max(mem_list), max(gpu_list)]) * 1.5)
                    ax[1].set_ylim(
                        0, max([max(mem_list), max(gpu_list)]) * 1.5)
                    plt.savefig(location)
                    print('Result has store in: %s' % location)
                    return
                else:
                    print(
                        'you have not supply the parameter save_flag and location, so the result is abort!!!')
                    return
    except Exception as err:
        print(err)


def mem_info_agg_mode(processName, location, save_flag, mem_list, gpu_list):
    import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # check the process name
    check_result = check_process(processName)
    if not check_result:
        print('Process: [%s] is not exist!!!' % processName)
        sys.exit(0)
    fig, ax = plt.subplots(figsize=(12, 8), ncols=1, nrows=2)
    plt.grid(True)
    mem_y = []
    gpu_y = []

    labels = ['max mem', 'min mem', 'avg mem', 'max gpu', 'min gpu', 'avg gpu']

    time = 0.0
    x = []

    try:
        while True:
            result = count_process_mem_gpu(processName)
            if result[2] == False:
                mem_list.append(round(result[0], 2))
                gpu_list.append(round(result[1], 2))

                ax[0].cla()
                ax[0].grid()
                ax[0].set_xlabel('time(second)')
                ax[0].set_ylabel('memory(Mib)')
                ax[0].set_xlim(0, time + 10)
                ax[0].set_ylim(0, max([max(mem_list), max(gpu_list)]) * 1.5)

                x.append(time)

                mem_y.append(round(result[0], 2))
                gpu_y.append(round(result[1], 2))

                ax[0].plot(x, mem_y, label='memory', color='red')
                ax[0].plot(gpu_y, label='gpu', color='blue')
                ax[0].legend(loc='lower right')

                # per 0.5 second get mem info
                ax[1].cla()
                ax[1].set_ylim(0, max(mem_list) + 50)
                ax[1].bar([1, 2, 3, 4, 5, 6], [max(mem_list), min(mem_list), sum(mem_list) / len(mem_list), max(gpu_list), min(
                    gpu_list), sum(gpu_list) / len(gpu_list)], 0.2, label='memory', color='#87CEFA', align='center', alpha=0.8)
                ax[1].set_xticks([1, 2, 3, 4, 5, 6])
                ax[1].set_xticklabels(labels)
                ax[1].set_ylabel('memory(Mib)')
                ax[1].set_ylim(0, max([max(mem_list), max(gpu_list)]) * 1.5)
                ax[1].text(1, max(mem_list), '')
                ax[1].text(1, max(mem_list), max(mem_list), ha='center')
                ax[1].text(2, min(mem_list), '')
                ax[1].text(2, min(mem_list), min(mem_list), ha='center')
                ax[1].text(3, sum(mem_list) / len(mem_list), '')
                ax[1].text(3, sum(mem_list) / len(mem_list),
                           round(sum(mem_list) / len(mem_list), 2), ha='center')
                ax[1].text(4, max(gpu_list), '')
                ax[1].text(4, max(gpu_list), max(gpu_list), ha='center')
                ax[1].text(5, min(gpu_list), '')
                ax[1].text(5, min(gpu_list), min(gpu_list), ha='center')
                ax[1].text(6, sum(gpu_list) / len(gpu_list), '')
                ax[1].text(6, sum(gpu_list) / len(gpu_list),
                           round(sum(gpu_list) / len(gpu_list), 2), ha='center')
                plt.pause(0.5)

                time = time + 0.5
            else:
                if save_flag != 'no' and location != 'no':
                    print('process ' + processName + ' is finished, now exit')
                    # save result
                    ax[0].set_xlim(0, time + 10)
                    ax[0].set_ylim(
                        0, max([max(mem_list), max(gpu_list)]) * 1.5)
                    ax[1].set_ylim(
                        0, max([max(mem_list), max(gpu_list)]) * 1.5)
                    plt.savefig(location)
                    print('Result has store in: %s' % location)
                    return
                else:
                    print(
                        'you have not supply the parameter save_flag and location, so the result is abort!!!')
                    return
    except Exception as err:
        print(err)


def print_summary_info(mem_list, gpu_list):
    print('Statistics summary...')
    print('avg gpu memory: [%s Mib]' % (str(round(sum(
        gpu_list) / len(gpu_list), 2))) if len(gpu_list) > 0 else 'avg gpu memory: [0 Mib]')
    print('max gpu memory: [%s Mib]' %
          str(max(gpu_list if len(gpu_list) > 0 else '0')))
    print('min gpu memory: [%s Mib]' %
          str(min(gpu_list if len(gpu_list) > 0 else '0')))
    print('avg mem memory: [%s Mib]' % (str(round(sum(
        mem_list) / len(mem_list), 2))) if len(mem_list) > 0 else 'avg mem memory: [0 Mib]')
    print('max mem memory: [%s Mib]' %
          str(max(mem_list if len(mem_list) > 0 else '0')))
    print('min mem memory: [%s Mib]' %
          str(min(mem_list if len(mem_list) > 0 else '0')))


def main():
    mem_list = list()
    gpu_list = list()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--process', help='A process you want to see its resource used', required=True)
    parser.add_argument(
        '-s', '--save', help='save the result in some place, you must use -l/--location to specify the location to store the result', default='no')
    parser.add_argument(
        '-l', '--location', help='A location save the result, example: /home/test/test.png', default='no')
    parser.add_argument(
        '--show', help='show the picture or not, default is show', default='yes')
    args = parser.parse_args()

    if args.location != 'no' and args.save != 'no' and args.show == 'no':
        # Agg mdoe
        mem_info_agg_mode(args.process, args.location,
                          args.save, mem_list, gpu_list)
        print_summary_info(mem_list, gpu_list)
    else:
        # show mode
        mem_info_show_mode(args.process, args.location,
                           args.save, mem_list, gpu_list)
        print_summary_info(mem_list, gpu_list)


if __name__ == '__main__':
    main()
