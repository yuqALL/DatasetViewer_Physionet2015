import logging
import re
from sklearn import preprocessing
import os.path

import numpy as np
import wfdb
import joblib
import shutil
import random
from collections import defaultdict
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import ShuffleSplit, train_test_split
from numpy.random import RandomState

from .data_read_utils import *

log = logging.getLogger(__name__)
log.setLevel('DEBUG')


def get_balanced_batches(
        n_trials, rng, shuffle, n_batches=None, batch_size=None
):
    assert batch_size is not None or n_batches is not None
    if n_batches is None:
        n_batches = int(np.round(n_trials / float(batch_size)))

    if n_batches > 0:
        min_batch_size = n_trials // n_batches
        n_batches_with_extra_trial = n_trials % n_batches
    else:
        n_batches = 1
        min_batch_size = n_trials
        n_batches_with_extra_trial = 0
    assert n_batches_with_extra_trial < n_batches

    all_inds = np.array(range(n_trials))

    if shuffle:
        rng.shuffle(all_inds)
    i_start_trial = 0
    i_stop_trial = 0
    batches = []
    for i_batch in range(n_batches):
        i_stop_trial += min_batch_size
        if i_batch < n_batches_with_extra_trial:
            i_stop_trial += 1
        batch_inds = all_inds[range(i_start_trial, i_stop_trial)]
        batches.append(batch_inds)
        i_start_trial = i_stop_trial
    assert i_start_trial == n_trials
    for b in batches:
        yield b
    # return batches


def stat_true_false(file_list, alarm_dict):
    true_list = []
    false_list = []
    for i, f in enumerate(file_list):
        if alarm_dict[f] == 0:
            true_list.append(i)
        elif alarm_dict[f] == 1:
            false_list.append(i)
        else:
            raise Exception("False Label Found!!!")
    return np.array(true_list), np.array(false_list)


def get_alarm_balanced_batches(data_list, alarm_dict, rng, shuffle, batch_size=None):
    true_list, false_list = stat_true_false(data_list, alarm_dict)

    p_trials = len(false_list)
    n_trials = len(true_list)

    if batch_size > min(p_trials, n_trials):
        batch_size = min(p_trials, n_trials)

    trials = max(p_trials, n_trials)
    less = abs(p_trials - n_trials)
    less_list = np.random.randint(0, trials - less, less)

    n_batches = int(np.round(n_trials / float(batch_size)))
    m_batch_size = trials // n_batches

    n_batches_with_extra_trial = trials % n_batches

    if p_trials < n_trials:
        less_inds = false_list[less_list]
        false_list = np.concatenate((false_list, less_inds), axis=0)
    else:
        less_inds = true_list[less_list]
        true_list = np.concatenate((true_list, less_inds), axis=0)

    if shuffle:
        rng.shuffle(false_list)
        rng.shuffle(true_list)

    i_start_trial = 0
    i_stop_trial = 0
    batches = []
    for i_batch in range(n_batches):
        i_stop_trial += m_batch_size
        if i_batch < n_batches_with_extra_trial:
            i_stop_trial += 1
        batch_inds = np.concatenate(
            (true_list[range(i_start_trial, i_stop_trial)],
             false_list[range(i_start_trial, i_stop_trial)]), axis=0)
        if shuffle:
            rng.shuffle(batch_inds)
        batches.append(batch_inds)
        i_start_trial = i_stop_trial
    if i_start_trial != trials:
        print(i_start_trial, trials, n_batches_with_extra_trial)
        raise Exception("ERROR load Dataset!!!")
    assert i_start_trial == trials
    return batches


class DatasetLoader:

    def __init__(self, opt):
        self.opt = opt
        self.rng = RandomState(20200426)
        self.files = load_file(self.opt.train_files, self.opt.data_folder)
        self.testfiles = load_file(self.opt.test_files, self.opt.data_folder)
        self.alarm_type = opt.alarm_type
        self.need_files = []
        self.need_test_files = []
        self.need_files_label = []
        self.need_sig_data = dict()
        self.need_sig_extra = dict()
        self.need_sig_label = dict()

        self.leave_out_files = defaultdict(list)

    def _filter_sensors_data(self, extra, sensor_ind):
        row_index = []
        for i, e in enumerate(extra):
            valid_record = True
            for ind in sensor_ind:
                if e[0, ind] == 1:
                    valid_record = False
                    break
            if valid_record:
                row_index.append(i)
        return row_index

    def get_extro_len(self):
        res = 0
        for k, v in self.need_sig_extra.items():
            res = len(self.need_sig_extra[k])
            break
        # if self.opt.use_minmax_scale:
        #     return res + len(self.opt.load_sensor_names) * 2
        return res

    def clear_load_data(self):
        self.need_sig_data.clear()
        self.need_sig_extra.clear()
        self.need_sig_label.clear()

    def load_need_data(self, files, clear=False):
        if not len(files):
            return
        if clear:
            self.need_sig_data.clear()
            self.need_sig_extra.clear()
            self.need_sig_label.clear()
        wanted_chan_inds = filter_sensors(self.opt.load_sensor_names)
        for f in files:
            if f not in self.need_sig_label:
                if self.opt.load_important_sig:
                    cnt, label = load_short_need_sig(f, length=self.opt.load_sig_length,
                                                     gnorm=self.opt.use_global_minmax)
                    self.need_sig_data[f] = deepcopy(cnt)
                if self.opt.load_all_sig:
                    cnt, label = load_short_all_sig(f, length=self.opt.load_sig_length,
                                                    gnorm=self.opt.use_global_minmax)
                    self.need_sig_data[f] = deepcopy(cnt)
                else:
                    cnt, label = load_short_sig(f, length=self.opt.load_sig_length, gnorm=self.opt.use_global_minmax)
                    self.need_sig_data[f] = cnt[:, wanted_chan_inds]
                self.need_sig_label[f] = label
                self.need_sig_extra[f] = load_record_extra_info(f)
                # if self.opt.use_minmax_scale:
                #     cnt,coefficient, b = minmax_scale(self.need_sig_data[f])
                #     self.need_sig_extra[f] += coefficient + b
                # if self.opt.use_gaussian_noise:
                #     self.need_sig_data[f] = wgn(self.need_sig_data[f], self.opt.gaussian_snr)
                #     self.need_sig_extra[f].append(self.opt.gaussian_snr)
        return

    def _load_data(self, files, check=True, test_mode=False):
        if len(files) == 0:
            return None, None, None
        all_sig = []
        all_sig_extra = []
        all_sig_label = []
        for f in files:
            if self.opt.split:
                cnt, extra, label_cnt = self.need_sig_data[f], self.need_sig_extra[f], self.need_sig_label[f]
                valid = check_valid_channel(cnt)

                result_cnt = sub_window_split(cnt, dilation=self.opt.dilation, window=self.opt.window_size,
                                              mmscale=self.opt.use_minmax_scale,
                                              add_noise_prob=self.opt.add_noise_prob if not test_mode else 0,
                                              add_amp_noise=self.opt.use_amplitude_noise if not test_mode else False)
                n = len(result_cnt)
                result_extra = [extra[:] for _ in range(n)]
                result_label = [label_cnt for _ in range(n)]
                if self.opt.load_important_sig and check:
                    if not test_mode:
                        if np.all(valid):
                            result_cnt = np.concatenate((result_cnt[:, :, 0], result_cnt[:, :, 1]), axis=0)[:, :,
                                         np.newaxis]
                            result_extra = [extra[:] for _ in range(2 * n)]
                            result_label = [label_cnt for _ in range(2 * n)]
                        elif valid[0]:
                            result_cnt = result_cnt[:, :, 0][:, :, np.newaxis]
                        elif valid[1]:
                            result_cnt = result_cnt[:, :, 1][:, :, np.newaxis]
                        else:
                            continue
                    else:
                        result_cnt = np.concatenate((result_cnt[:, :, 0], result_cnt[:, :, 1]), axis=0)[:, :,
                                     np.newaxis]
                        result_extra = [extra[:] for _ in range(2 * n)]
                        result_label = [label_cnt for _ in range(2 * n)]
                elif self.opt.load_all_sig and check:
                    k = result_cnt.shape[2]

                    result_cnt = np.concatenate((np.split(result_cnt, result_cnt.shape[2], axis=2)), axis=0).squeeze()[
                                 :, :,
                                 np.newaxis]
                    result_extra = [extra[:] for _ in range(k * n)]
                    result_label = [label_cnt for _ in range(k * n)]

            else:
                result_cnt, result_extra, result_label = [self.need_sig_data[f]], [self.need_sig_extra[f]], \
                                                         [self.need_sig_label[f]]

            all_sig.append(result_cnt)
            all_sig_extra.append(result_extra)
            all_sig_label.append(result_label)
        if len(all_sig) == 0:
            print(files)
            return None, None, None

        sig = np.concatenate(np.array(all_sig), axis=0)
        extra = np.concatenate(np.array(all_sig_extra), axis=0)
        label = np.concatenate(np.array(all_sig_label), axis=0)
        sig = np.transpose(sig, (0, 2, 1))
        return sig, extra, label

    def get_people(self, filepath, alarm_type='all', check=True):
        if filepath not in self.need_sig_label:
            return None, None, None

        # event_classes = load_record(filepath).comments
        # if alarm_type != 'all' and event_classes[0] != alarm_type:
        #     return None, None, None
        return self._load_data([filepath], check=check, test_mode=True)

    def get_single_data(self, filepath, alarm_type='all', check=True):
        if filepath not in self.need_sig_label:
            return None, None, None

        event_classes = load_record(filepath).comments
        if alarm_type != 'all' and event_classes[0] != alarm_type:
            return None, None, None

        if self.opt.split:
            cnt, extra, label_cnt = self.need_sig_data[filepath], self.need_sig_extra[filepath], self.need_sig_label[
                filepath]
            valid = check_valid_channel(cnt)

            result_cnt = sub_window_split(cnt, dilation=self.opt.dilation, window=self.opt.window_size,
                                          mmscale=self.opt.use_minmax_scale,
                                          add_noise_prob=0,
                                          add_amp_noise=False)
            n = len(result_cnt)
            result_extra = [extra[:] for _ in range(n)]
            result_label = [label_cnt for _ in range(n)]
            if self.opt.load_important_sig and check:
                if np.all(valid):
                    result_cnt = np.concatenate((result_cnt[:, :, 0], result_cnt[:, :, 1]), axis=0)[:, :,
                                 np.newaxis]
                    result_extra = [extra[:] for _ in range(2 * n)]
                    result_label = [label_cnt for _ in range(2 * n)]
                elif valid[0]:
                    result_cnt = result_cnt[:, :, 0][:, :, np.newaxis]
                elif valid[1]:
                    result_cnt = result_cnt[:, :, 1][:, :, np.newaxis]
                else:
                    return None, None, None
        else:
            result_cnt, result_extra, result_label = self.need_sig_data[filepath], self.need_sig_extra[filepath], \
                                                     self.need_sig_label[
                                                         filepath]

        if isinstance(result_cnt, list):
            sig = []
            for cnt in result_cnt:
                sig += [np.transpose(cnt, (0, 2, 1))]
        else:
            sig = np.transpose(result_cnt, (0, 2, 1))
        return sig, result_extra, result_label

    def get_batches(self, dataset, shuffle, setname):
        if setname == "test":
            for ind, people in enumerate(dataset):
                batch_X, batch_extra, batch_y = self._load_data([people], test_mode=True)

                # batch_extra = batch_extra if self.opt.use_extra else None
                # add empty fourth dimension if necessary

                if batch_X.ndim == 2:
                    batch_X = np.array([batch_X])
                    batch_y = np.array([batch_y])
                    batch_extra = np.array([batch_extra]) if self.opt.use_extra else None
                yield (batch_X, batch_extra, batch_y)
        else:
            n_trials = len(dataset)
            # from guppy import hpy
            # h = hpy()
            people_batches = get_alarm_balanced_batches(dataset, self.need_sig_label, batch_size=self.opt.load_file_batch,
                                                        rng=self.rng, shuffle=shuffle)
            # people_batches = get_balanced_batches(
            #     n_trials, batch_size=self.opt.batch_size, rng=self.rng, shuffle=shuffle
            # )
            for batch_inds in people_batches:
                batch = dataset[batch_inds]
                sig, extra, label = self._load_data(batch)
                data_batches = get_balanced_batches(
                    len(label), batch_size=self.opt.batch_size, rng=self.rng, shuffle=True
                )
                for b_ind in data_batches:
                    batch_X = sig[b_ind]
                    batch_extra = extra[b_ind]
                    # batch_extra = extra[b_ind] if self.opt.use_extra else None
                    batch_y = label[b_ind]

                    # print(h.heap())
                    yield (batch_X, batch_extra, batch_y)
    def mp_load(self, dataset, people_batches, test_mode=False):
        import multiprocessing as mp
        pool = mp.Pool(4)
        it = pool.imap_unordered(self._load_data, [dataset[batch] for batch in people_batches])
        return it

    def get_batches_bak(self, dataset, shuffle, setname):
        if not isinstance(dataset, np.ndarray):
            dataset = np.array(dataset)
        if setname == "test":
            test_mode = True
        else:
            test_mode = False

        people_batches = get_alarm_balanced_batches(dataset, self.need_sig_label, batch_size=self.opt.load_file_batch,
                                                    rng=self.rng, shuffle=shuffle)
        sig_generator = self.mp_load(dataset, people_batches, test_mode)
        for sig, extra, label in sig_generator:
            data_batches = get_balanced_batches(
                len(label), batch_size=self.opt.batch_size, rng=self.rng, shuffle=True
            )
            for b_ind in data_batches:
                batch_X = sig[b_ind]
                batch_extra = extra[b_ind]
                batch_y = label[b_ind]
                yield (batch_X, batch_extra, batch_y)

    def _filter_files(self, files, alarm_type='all', explict_sensor=False, need_long_record=False):

        need_files = []
        for f in files:
            if need_long_record and f.endswith('s'):
                continue
            record = wfdb.rdrecord(f)
            if alarm_type != 'all' and record.comments[0] != alarm_type:
                continue
            # sensors = set(record.sig_name)
            sensors = {k: v for v, k in enumerate(record.sig_name)}
            check = check_valid_channel(record.p_signal[-20 * 250:])
            have_all_need_sensor = True
            if explict_sensor:
                for s in self.opt.load_sensor_names:
                    if s not in sensors or not check[sensors[s]]:
                        have_all_need_sensor = False
                        break
            if self.opt.load_important_sig:
                have_all_need_sensor = False
                for s, ind in sensors.items():
                    if s in important_sig and check[ind]:
                        have_all_need_sensor = True
                        break

            if have_all_need_sensor:
                need_files.append(f)
        need_files = np.array(need_files)
        return need_files

    def filter_files(self, alarm_type='all', explict_sensor=False, need_long_record=False):
        self.clear_load_data()
        self.alarm_type = alarm_type
        data_path = defaultdict(np.ndarray)
        self.need_files = self._filter_files(self.files, alarm_type, explict_sensor, need_long_record)
        self.load_need_data(self.need_files)
        self.need_test_files = self._filter_files(self.testfiles, alarm_type, explict_sensor, need_long_record)
        self.load_need_data(self.need_test_files)

        data_path['train'] = self.need_files
        data_path['test'] = self.need_test_files
        self.leave_out_files = deepcopy(data_path)
        return

    def filter_test_files(self, alarm_type='all', explict_sensor=False, need_long_record=False):
        self.alarm_type = alarm_type
        self.need_test_files = self._filter_files(self.testfiles, alarm_type, explict_sensor, need_long_record)
        self.load_need_data(self.testfiles, clear=True)
        return

    def cross_val_split(self):
        cv_datafiles = []
        for i in range(1, self.opt.n_split + 1):
            data_path = defaultdict(list)
            data_path['train'] = load_file(self.opt.data_folder + str(i) + '_fold_train_files_list.txt',
                                           self.opt.data_folder)
            data_path['test'] = load_file(self.opt.data_folder + str(i) + '_fold_test_files_list.txt',
                                          self.opt.data_folder)

            cv_datafiles.append(data_path)
        self.cv_data_files = deepcopy(cv_datafiles)
        return

    def leave_out_split(self):
        if not len(self.need_files_label):
            return

        data_path = defaultdict(list)
        data_path['train'] = self.need_files
        data_path['test'] = self.need_test_files
        self.leave_out_files = deepcopy(data_path)
        return
