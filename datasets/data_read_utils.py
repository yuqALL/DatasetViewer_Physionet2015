import os
from sys import exit
import logging
import numpy as np
from collections import defaultdict
import copy
import csv
import wfdb

log = logging.getLogger(__name__)
log.setLevel('DEBUG')

r'''
本文件仅负责数据的读取
'''

all_sensor_name = ['I', 'II', 'III', 'V', 'aVL', 'aVR', 'aVF', 'RESP', 'PLETH', 'MCL', 'ABP']
important_sig = ['I', 'II', 'III', 'V', 'aVL', 'aVR', 'aVF', 'MCL']
other_sig = ['RESP', 'PLETH', 'ABP']
all_alarm_id = {'Ventricular_Tachycardia': [1, 0, 0, 0, 0], 'Tachycardia': [0, 1, 0, 0, 0],
                'Ventricular_Flutter_Fib': [0, 0, 1, 0, 0], 'Bradycardia': [0, 0, 0, 1, 0],
                'Asystole': [0, 0, 0, 0, 1]}
all_alarm_num = {'Ventricular_Tachycardia': 0, 'Tachycardia': 0,
                 'Ventricular_Flutter_Fib': 0, 'Bradycardia': 0,
                 'Asystole': 0}


def read_interest_area(sig, start, end):
    sig = sig[start:end]
    # if np.any(np.isnan(sig)):
    #     print(sig)
    return sig


def stat_data_use_file(filename, data_path):
    fileList = []
    with open(filename, 'r') as f:
        for line in f:
            fileList.append(os.path.join(data_path, line.strip('\n')))
    positive = 0
    negative = 0
    for s in fileList:
        record = load_record(s)
        if record.comments[1] == 'False alarm':
            positive += 1
        else:
            negative += 1

    return positive, negative, positive / negative


def resample_sig(cnt, point=1000):
    from scipy import signal
    cnt_resample = signal.resample(cnt, point, axis=0)
    return cnt_resample


def fill_nan(signal):
    """Solution provided by Divakar."""

    mask = np.isnan(signal)
    idx = np.where(~mask.T, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = signal[idx.T, np.arange(idx.shape[0])[None, :]]
    out = np.nan_to_num(out)
    return out


def load_file_list(path, shuffle=True, seed=20200426):
    file_list = os.listdir(path)
    file_list = sorted(os.path.join(path, f[:-4]) for f in file_list if
                       os.path.isfile(os.path.join(path, f)) and f.endswith(".mat"))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(file_list)
    return file_list


def load_file(filename, data_path):
    content = []
    with open(filename, 'r') as f:
        for line in f:
            content.append(os.path.join(data_path, line.strip('\n')))
    return content


def load_record_id(filename):
    content = []
    with open(filename, 'r') as f:
        for line in f:
            content.append(line.strip('\n'))
    return content


def load_label_path(path):
    file_list = os.listdir(path)
    complete_label_set = set()
    uncomplete_label_set = set()
    for f in file_list:
        with open(os.path.join(path, f), 'r') as csvfile:
            row = csv.reader(csvfile, delimiter=',')
            header = next(row)[0]
            if header == 'not complete':
                uncomplete_label_set.add(f[:-4])
            else:
                complete_label_set.add(f[:-4])
    return complete_label_set, uncomplete_label_set


def cross_val_files(data_folder, n_split):
    cv_datafiles = []
    for i in range(1, n_split + 1):
        data_path = defaultdict(list)
        data_path['train'] = load_file(data_folder + str(i) + '_fold_train_files_list.txt', data_folder)
        data_path['test'] = load_file(data_folder + str(i) + '_fold_test_files_list.txt', data_folder)
        cv_datafiles.append(data_path)
    return cv_datafiles


def load_record(filename):
    record = wfdb.rdrecord(filename)
    return record


def load_record_extra_info(filename):
    record = load_record(filename)
    fs = int(record.fs)
    sensor = record.sig_name
    event_classes = record.comments
    event_id = all_alarm_id[event_classes[0]]
    extra = event_id
    return extra


def get_chan_ind(sensor):
    chan_inds = [all_sensor_name.index(s) for s in sensor]
    return chan_inds


def load_full_sig(filename, fillnan=True):
    record = load_record(filename)
    fs = int(record.fs)
    continuous_signal = record.p_signal
    length = len(continuous_signal)
    cnt = np.full((length, len(all_sensor_name)), np.nan, dtype='float32')
    chan_inds = get_chan_ind(record.sig_name)
    cnt[:, chan_inds] = continuous_signal[:fs * length]
    if fillnan:
        cnt = fill_nan(cnt)
    event_classes = record.comments
    label = 0
    if event_classes[1] == 'True alarm':
        label = 1
    return cnt, label


def load_short_sig(filename, length=15, fillnan=True, gnorm=False):
    record = load_record(filename)
    fs = int(record.fs)
    cnt = np.full((fs * length, len(all_sensor_name)), np.nan, dtype='float32')
    continuous_signal = record.p_signal
    chan_inds = get_chan_ind(record.sig_name)
    cnt[:, chan_inds] = continuous_signal[(300 - length) * fs:300 * fs, :]

    if fillnan:
        cnt = fill_nan(cnt)
    if gnorm:
        tmp = np.full((fs * 10, len(all_sensor_name)), 0, dtype='float32')
        tmp[:, chan_inds] = continuous_signal[(290 - length) * fs:(300 - length) * fs, :]
        # minv = np.nanmin(tmp, axis=0)
        # minv = np.nan_to_num(minv)
        # maxv = np.nanmax(tmp, axis=0)
        # maxv = np.nan_to_num(maxv)
        # print(tmp.shape)
        minv = np.percentile(tmp, 5, axis=0)
        maxv = np.percentile(tmp, 95, axis=0)
        minv = np.nan_to_num(minv)
        maxv = np.nan_to_num(maxv)
        if isinstance(maxv, np.ndarray):
            t = [1 / v if v else 1 for v in (maxv - minv)]
        else:
            t = 1 / (maxv - minv) if maxv - minv else 1.0
        cnt -= minv
        cnt *= t

    event_classes = record.comments
    label = 0
    if event_classes[1] == 'True alarm':
        label = 1
    return np.array(cnt), label


def check_valid_channel(sig):
    sig = np.nan_to_num(sig)
    result = np.any(np.greater(abs(sig), 0), axis=0)
    #  or np.all((abs(sig) < 1e-12), axis=0) np.all(np.isnan(sig), axis=0) or
    return result


def te_check_valid_channel():
    sig = [[np.nan, np.nan],
           [0.0, 0.0], [1, 0.00000000000000000001], [np.nan, 0.0], [1, 0]]
    sig = np.array(sig)
    print(check_valid_channel(sig))


def load_short_need_sig(filename, length=15, fillnan=True, gnorm=False):
    record = load_record(filename)
    fs = int(record.fs)
    cnt = np.full((fs * length, 2), np.nan, dtype='float32')
    continuous_signal = record.p_signal
    i = 0
    if gnorm:
        tmp = np.full((fs * 10, 2), 0, dtype='float32')
    for j, s in enumerate(record.sig_name):
        if s in important_sig:
            cnt[:, i] = continuous_signal[(300 - length) * fs:300 * fs, j]
            if gnorm:
                tmp[:, i] = continuous_signal[(290 - length) * fs:(300 - length) * fs, j]
            i += 1
        if i == 2:
            break
    if i != 2:
        print(filename)
    if fillnan:
        cnt = fill_nan(cnt)
    if gnorm:
        # minv = np.nanmin(tmp, axis=0)
        # minv = np.nan_to_num(minv)
        # maxv = np.nanmax(tmp, axis=0)
        # maxv = np.nan_to_num(maxv)

        minv = np.percentile(tmp, 5, axis=0)
        maxv = np.percentile(tmp, 95, axis=0)
        minv = np.nan_to_num(minv)
        maxv = np.nan_to_num(maxv)
        if isinstance(maxv, np.ndarray):
            t = [1 / v if v else 1 for v in (maxv - minv)]
        else:
            t = 1 / (maxv - minv) if maxv - minv else 1.0
        cnt -= minv
        cnt *= t
    event_classes = record.comments
    label = 0
    if event_classes[1] == 'True alarm':
        label = 1
    return cnt, label


def load_short_all_sig(filename, length=15, fillnan=True, gnorm=False):
    record = load_record(filename)
    fs = int(record.fs)
    cnt = np.full((fs * length, len(record.sig_name)), np.nan, dtype='float32')
    continuous_signal = record.p_signal
    i = 0
    if gnorm:
        tmp = np.full((fs * 10, len(record.sig_name)), 0, dtype='float32')
    for j, s in enumerate(record.sig_name):
        cnt[:, i] = continuous_signal[(300 - length) * fs:300 * fs, j]
        if gnorm:
            tmp[:, i] = continuous_signal[(290 - length) * fs:(300 - length) * fs, j]
        i += 1

    if fillnan:
        cnt = fill_nan(cnt)
    if gnorm:
        # minv = np.nanmin(tmp, axis=0)
        # minv = np.nan_to_num(minv)
        # maxv = np.nanmax(tmp, axis=0)
        # maxv = np.nan_to_num(maxv)

        minv = np.percentile(tmp, 5, axis=0)
        maxv = np.percentile(tmp, 95, axis=0)
        minv = np.nan_to_num(minv)
        maxv = np.nan_to_num(maxv)
        if isinstance(maxv, np.ndarray):
            t = [1 / v if v else 1 for v in (maxv - minv)]
        else:
            t = 1 / (maxv - minv) if maxv - minv else 1.0
        cnt -= minv
        cnt *= t
    event_classes = record.comments
    label = 0
    if event_classes[1] == 'True alarm':
        label = 1
    return cnt, label


def minmax_scale(cnt):
    minv = np.nanmin(cnt, axis=0)
    maxv = np.nanmax(cnt, axis=0)
    if isinstance(maxv, np.ndarray):
        t = [1 / v if v else 1 for v in (maxv - minv)]
    else:
        t = 1 / (maxv - minv) if maxv - minv else 1.0
    cnt -= minv
    cnt *= t
    return cnt


def sig_flip(cnt):
    cnt *= -1
    return cnt


def minmax_scale_zero(cnt):
    minv = np.nanmin(cnt, axis=0)
    maxv = np.nanmax(cnt, axis=0)
    if isinstance(maxv, np.ndarray):
        t = [1 / v if v else 1 for v in (maxv - minv)]
    else:
        t = 1 / (maxv - minv) if maxv - minv else 1.0
    cnt -= minv
    cnt *= t
    cnt = (cnt - 0.5) * 2
    return cnt


def wgn(cnt, snr):
    Ps = np.sum(abs(cnt) ** 2, axis=0) / len(cnt)
    Pn = Ps / (10 ** (snr / 10))
    Pn = np.reshape(Pn, (1, len(Pn)))
    noise = np.random.randn(*cnt.shape) * np.sqrt(Pn)
    cnt += noise
    return cnt


def sin_gaussion_noise(cnt, fz=50, factor='default', sigma='default'):
    # cnt = copy.deepcopy(sig)
    continer = [[i] * cnt.shape[1] for i in range(cnt.shape[0])]
    f = 0.1 * (np.nanmax(cnt, axis=0) - np.nanmin(cnt, axis=0))
    if factor == 'default':
        sin_noise = f * np.sin(np.array(continer) * fz * np.pi * 2 / 250 + np.random.randn(*cnt.shape) * np.pi)
    else:
        sin_noise = factor * np.sin(np.array(continer) * fz * np.pi * 2)
    if sigma == 'default':
        gauss_noise = f * np.random.randn(*cnt.shape)
    else:
        gauss_noise = sigma * np.random.randn(*cnt.shape)
    cnt += sin_noise + gauss_noise
    return cnt


def gaussion_noise_bak(sig, sigma='default'):
    cnt = copy.deepcopy(sig)
    if sigma == 'default':
        sigma = 0.1 * (np.nanmax(cnt, axis=0) - np.nanmin(cnt, axis=0))
        noise = sigma * np.random.randn(*cnt.shape)
    else:
        noise = sigma * np.random.randn(*cnt.shape)
    cnt += noise
    return cnt


def gaussion_noise(sig, sigma='default', drop_noise=0.8):
    cnt = copy.deepcopy(sig)
    n, c = cnt.shape
    zeros = int(n * drop_noise)
    ones = n - zeros
    factor = np.array([0] * zeros + [1] * ones)

    if sigma == 'default':
        sigma = 0.1 * (np.nanmax(cnt, axis=0) - np.nanmin(cnt, axis=0))
        noise = sigma * np.random.randn(*cnt.shape)
    else:
        noise = sigma * np.random.randn(*cnt.shape)
    for i in range(c):
        np.random.shuffle(factor)
        noise[:, i] = factor * noise[:, i]
    cnt += noise
    return cnt


def amp_noise(sig):
    cnt = copy.deepcopy(sig)
    A = np.random.randn() + 0.5
    cnt = A * cnt
    return cnt


def sin_noise(sig, fz=50, factor='default'):
    cnt = copy.deepcopy(sig)
    continer = [[i] * cnt.shape[1] for i in range(cnt.shape[0])]
    if factor == 'default':
        factor = 0.5 * (np.nanmax(cnt, axis=0) - np.nanmin(cnt, axis=0))
        noise = factor * np.sin(np.array(continer) * fz * np.pi * 2)
    else:
        noise = factor * np.sin(np.array(continer) * fz * np.pi * 2)

    cnt += noise
    return cnt


def load_long_sig(filename, length=45, fillnan=True):
    record = load_record(filename)
    fs = int(record.fs)
    cnt = np.full((fs * length, len(all_sensor_name)), np.nan, dtype='float32')
    continuous_signal = record.p_signal
    chan_inds = get_chan_ind(record.sig_name)
    cnt[:, chan_inds] = continuous_signal[(330 - length) * fs:fs * 330]
    if fillnan:
        cnt = fill_nan(cnt)
    event_classes = record.comments
    label = 0
    if event_classes[1] == 'True alarm':
        label = 1
    return cnt, label


def sub_window_split(cnt, dilation=125, window=2500, start_point=0, mmscale=False, add_noise_prob=0.0,
                     add_amp_noise=False):
    result_cnt = []
    ch = cnt.shape[1]
    for i in range(start_point, cnt.shape[0], dilation):
        if i + window > cnt.shape[0]:
            break
        info = np.zeros((window, ch), dtype='float32')
        info[:] = cnt[i:i + window]
        if mmscale:
            minmax_scale(info)
        else:
            if add_amp_noise and np.random.uniform() > 0.5:
                amp_noise(info)
        if add_noise_prob > 0 and add_noise_prob > np.random.uniform():
            # p = np.random.uniform()
            # if p > gnoise:
            #     gaussion_noise(info)
            # else:
            #     sin_noise(info, 50)
            gaussion_noise(info)

        result_cnt.append(info)
    return np.array(result_cnt)


def filter_sensors(load_sensor_names=None):
    if load_sensor_names is None:
        return list(range(len(all_sensor_name)))
    chan_inds = get_chan_ind(load_sensor_names)
    return chan_inds


def stat_diff_alarm_nums(filelist):
    alarm_nums = defaultdict(list)
    for f in filelist:
        record = load_record(f)
        comments = record.comments
        if comments[0] not in alarm_nums:
            alarm_nums[comments[0]] = [0, 0]
        if comments[1] == 'True alarm':
            alarm_nums[comments[0]][1] += 1
        else:
            alarm_nums[comments[0]][0] += 1
    print(alarm_nums)


def chect_dataset():
    train_datapaths = ["1_fold_train_files_list.txt", "2_fold_train_files_list.txt", "3_fold_train_files_list.txt",
                       "4_fold_train_files_list.txt", "5_fold_train_files_list.txt"]
    test_datapaths = ["1_fold_test_files_list.txt", "2_fold_test_files_list.txt", "3_fold_test_files_list.txt",
                      "4_fold_test_files_list.txt", "5_fold_test_files_list.txt"]
    for i in range(5):
        train_files = load_file('../data/training/' + train_datapaths[i], '../data/training/')
        test_files = load_file('../data/training/' + test_datapaths[i], '../data/training/')
        print(len(train_files), len(test_files))
        files = set(train_files)
        for f in test_files:
            if f in files:
                print(i + 1, f)
        # files.update(set(test_files))
        # print(len(files))


if __name__ == "__main__":
    data_folder = '../data/training/'
    pid = 'a103l'
    record = load_record(data_folder + pid)
    exit(0)

    import wfdb

    # cross_val_files(data_folder, n_split=5)
    chect_dataset()

    # cross_val_dataset_balance(data_folder)
    # exit(0)
    # split_dataset_balance(data_folder)

    test_file_list = load_file(data_folder + "1_fold_test_files_list.txt", data_folder)
    stat_diff_alarm_nums(test_file_list)
    test_file_list = load_file(data_folder + "2_fold_test_files_list.txt", data_folder)
    stat_diff_alarm_nums(test_file_list)
    test_file_list = load_file(data_folder + "3_fold_test_files_list.txt", data_folder)
    stat_diff_alarm_nums(test_file_list)

    train_file_list = load_file(data_folder + "1_fold_train_files_list.txt", data_folder)
    stat_diff_alarm_nums(train_file_list)

    all_file_list = load_file(data_folder + "RECORDS", data_folder)
    stat_diff_alarm_nums(all_file_list)

    # test_check_valid_channel()
    #
    root_path = '../data/training/'
    # print(stat_data(root_path))
    pid = 'a103l'
    # split_dataset(root_path)
    # split_dataset_balance(root_path)

    exit(0)
