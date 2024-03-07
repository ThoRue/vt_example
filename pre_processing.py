# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:05:18 2022

@author: rueland
"""


import numpy as np
from collections import Counter
import os
from sys import stdout


def cut_data(ca, spikes, signal_data, trial_table, limit_start=3, rew_fallback=True, flash=True, buffer=True, stop='reward', conversion_factor=1/1024*64):
    """
    Cut out section of data according to place or location of events

    Input:
        ca              :   raw calcium activity
        spikes          :   raw spike probabilities
        signal_data     :   other datatypes mapped to the signal 
        limit_start     :   integer of start of trials in cm
        stop            :   either integer of end of trials in cm or string
                            specifying where to stop e.g. "reward"

    Output:
        ca_cut          :   same organisation as the input file, but cut
        spikes_cut      :   same organisation as the input file, but cut
        signal_data     :   same organisation as the input file, but cut

    """

    place_ttable = trial_table[trial_table['marked_for_place_conversion'] == 1].copy(deep=True)
    
    new_ca = []
    new_spikes = []
    new_signal_data = {}
    
    # get earliest reward spot to avoid cutting soem trials after reward
    if stop == 'reward':
        
        for trial in np.arange(len(place_ttable)):
            # find reward place
            if flash is False and rew_fallback is True:
                rew_place = int(int(place_ttable.iloc[trial].end_place) - 30/conversion_factor)
            else:
                rew_place = place_ttable.iloc[trial].reward_place[0]
            # find first occurence of reward place in place vector
            ind = np.argmin(np.abs(np.array(signal_data['place'][trial]) - rew_place))
            # restrict ca and spikes
            new_ca.append([x[:ind] for x in ca[trial]])
            new_spikes.append([x[:ind] for x in spikes[trial]])
            # restrict all signal datas
            for key in signal_data:
                if key not in new_signal_data:
                    new_signal_data[key] = []
                new_signal_data[key].append(signal_data[key][trial][:ind])
            
            
            
    new_ca_start = []
    new_spikes_start = []
    new_signal_data_start = {}
            
    if type(limit_start) == int and buffer is True:
        for trial in np.arange(len(place_ttable)):
            # find reward place
            temp_lim = int(int(place_ttable.iloc[trial].start_place) + limit_start/conversion_factor)
            # find last occurence of start buffer place in place vector
            ind = len(signal_data['place'][trial]) - np.argmin(np.abs(np.array(signal_data['place'][trial]) - temp_lim)[::-1])
            # restrict ca and spikes
            new_ca_start.append([x[ind:] for x in new_ca[trial]])
            new_spikes_start.append([x[ind:] for x in new_spikes[trial]])
            
            # restrict all signal datas
            for key in new_signal_data:
                if key not in new_signal_data_start:
                    new_signal_data_start[key] = []
                new_signal_data_start[key].append(new_signal_data[key][trial][ind:])
                

    return new_ca_start, new_spikes_start, new_signal_data_start


def create_signal_data(vr_data, signal, ttable):
    # cleaning place vectors from -1s
    clean_place, clean_time = clean_place_vec(vr_data['place'], vr_data['time'])
    
    # correct time vectors for wrong start timing
    clean_time = [x-np.min(x) for x in clean_time]
    
    signal_data = {}
    
    # create time vector for signal
    sig_time = []
    for t, trial in enumerate(signal):
        # sig_time.append(np.arange(0, max(clean_time[t]), 1/fs))
        sig_time.append(np.linspace(0, max(clean_time[t]), len(signal[t][0])))
        
    signal_data['time'] = sig_time
        
    # get place for each timepoint in signal
    sig_place = []
    for t, trial in enumerate(sig_time):
        trial_place = []
        for timepoint in trial:
            trial_place.append(clean_place[t][np.argmin(abs(clean_time[t]-timepoint))])
        sig_place.append(trial_place)
    
    signal_data['place'] = sig_place
    
    # create a seq vector for each trial
    seq_vec = []
    for t, trial in enumerate(sig_place):
        temp_vec = np.zeros(len(trial))
        for f, fp in enumerate(ttable.iloc[t].flash_place):
            temp_vec[(np.array(trial) > fp) & (np.array(trial) < (fp + 160))] = int(ttable.iloc[t].stimID[f])+1
        seq_vec.append(temp_vec)
        
    signal_data['seq_vec'] = seq_vec
    
    return signal_data


def make_traces_positive(FcellCorr):
    """
    Tries to circumenvent the problem of negative traces bing created from
    strange motion correction
    Check if global minimum off all traces recorded is smaller than zero, and
    if this is the case, adds this minimum value to all tracecs to lift them
    up above 0
    """
    min_fl = np.min([np.min([np.min(y) for y in x]) for x in FcellCorr])
    if min_fl < 0:
        new_FcellCorr = []
        for t, trial in enumerate(FcellCorr):
            new_FcellCorr.append([])
            for r, roi in enumerate(trial):
                new_FcellCorr[t].append(roi-min_fl)
    else:
        new_FcellCorr = FcellCorr

    return new_FcellCorr


def calculate_dff_ws(FcellCorr, norm_type='median'):

    # concatenate signals into one signal per session and roi
    len_list = []
    temp_sig = [[] for x in FcellCorr[0]]
    for t, trial in enumerate(FcellCorr):
        for r, roi in enumerate(trial):
            if r == 0:
                len_list.append(len(roi))
            temp_sig[r].append(roi)

    conc_sig = [np.concatenate(np.array(x, dtype='object')) for x in temp_sig]

    # normalize each roi to its mean/median per session
    norm_sig = []
    session_std = []
    for roi in conc_sig:
        if norm_type == 'median':
            if np.median(roi) != 0:
                norm_sig.append((roi-np.median(roi))/np.median(roi))
            else:
                norm_sig.append(np.zeros(len(roi)))
                
        elif norm_type == 'mean':
            if np.mean(roi) != 0:
                norm_sig.append((roi-np.mean(roi))/np.mean(roi))
            else:
                norm_sig.append(np.zeros(len(roi)))
        session_std.append(np.nanstd(norm_sig[-1]))

    # seperate session into trials again
    signal_dff = []
    old_len = 0
    for t, tr_len in enumerate(len_list):
        signal_dff.append([])
        for r, roi in enumerate(norm_sig):
            if t == 0:
                signal_dff[t].append(roi[:tr_len])
            else:
                signal_dff[t].append(roi[old_len:old_len+tr_len])
        old_len = old_len+tr_len


    return signal_dff, session_std


def clean_place_vec(place_vec_all, time_vec_all):
    clean_place = []
    clean_time = []

    for t, place_trial in enumerate(place_vec_all):
        good_inds = np.where(place_trial != -1)[0]
        new_place = place_trial[good_inds]
        new_time = np.array(time_vec_all[t])[good_inds]
        clean_place.append(new_place)
        clean_time.append(new_time)

    return clean_place, clean_time


def continuate_place_vec(place_vec_all, time_vec_all, timestep=0.001):

    # define arbitrary timestep in which i want to sample e.g. 0.01 sec
    current_time = 0
    res_place_vec_all = []
    res_time_vec_all = []
    for p, place_vec in enumerate(place_vec_all):
        res_time_vec_all.append([])
        res_place_vec_all.append([])
        for t, timepoint in enumerate(time_vec_all[p]):
            while current_time < timepoint:
                res_place_vec_all[p].append(place_vec[t])
                res_time_vec_all[p].append(current_time)
                current_time = current_time+timestep
                
        current_time = 0
        stdout.write("\r"+"Trial "+str(p)+" of "+str(len(place_vec_all)-1)+" resampled.")
        stdout.flush()

    # calculate speed for new place vector
    res_speed_vec_all = []
    for t, trial in enumerate(res_place_vec_all):
        res_speed_vec_all.append(np.append(np.diff(np.array(trial))/timestep, [0]))

    return res_place_vec_all, res_speed_vec_all, res_time_vec_all


def clean_and_resample_place(vr_data):
    # cleaning place vectors from -1s
    clean_place, clean_time = clean_place_vec(vr_data['place'], vr_data['time'])

   # Convert place vector into continous equally sampled signal
    (res_place_vec_all,
     res_speed_vec_all,
     res_time_vec_all) = continuate_place_vec(clean_place,
                                               clean_time,
                                               timestep=0.001)


    vr_data['res_place'] = res_place_vec_all
    vr_data['res_time'] = res_time_vec_all
    
    
    return vr_data


def excluded_trials(vr_data, settings, trial_table, threshold_leeway=3, conversion_factor=1/1024*64, use_def_len=False, flash=True):

    # generate empty list to store faulty trial_ids in
    excluded_list = []
    reason_dict = {}

    # find irregularities in place vector data
    start_of_trials = np.array([round(np.min(np.array(x)[np.where(np.array(x) >= 0)])) for x in vr_data['place']])
    mod_start = Counter(start_of_trials).most_common(1)[0][0]
    thres_start = threshold_leeway/conversion_factor    
    
    end_of_trials = np.array([round(np.max(x)) for x in vr_data['place']])
    len_of_trials = end_of_trials-start_of_trials
    mod_len = Counter(len_of_trials).most_common(1)[0][0]
    thres_len = threshold_leeway/conversion_factor
    exl_len = np.arange(0, len(len_of_trials))[(len_of_trials > mod_len+thres_len) | (len_of_trials < mod_len-thres_len)]
    exl_start = np.arange(0, len(start_of_trials))[(start_of_trials > mod_start+thres_start) | (start_of_trials < mod_start-thres_start)]

    for x in exl_len:
        excluded_list.append(x)
        if x not in reason_dict:
            reason_dict[x] = []
        reason_dict[x].append('wrong corridor length')
    for x in exl_start:
        excluded_list.append(x)
        if x not in reason_dict:
            reason_dict[x] = []
        reason_dict[x].append('wrong start position')
    # find irregularities in the flash vector data
    # check if flash vector is existing or if its empty -> hinting at non flash
    # corridor
    if 'flash' in vr_data.keys() and flash is True:
        if use_def_len == False:
            lengths =[len([int(y) for y in x if y.isdigit()]) if type(x)==str else 0 for x in trial_table['stimID'][trial_table.marked_for_analysis == 1].values]
            des_len = max(lengths)
            if des_len == 0:
                des_len = 3
        else:
            des_len = 3
        exl_flash = np.where(np.array([len(x) for x in vr_data['flash']]) < des_len)[0]
        for entry in exl_flash:
            excluded_list.append(entry)
            if entry not in reason_dict:
                reason_dict[entry] = []
            reason_dict[entry].append('wrong amount of flashes')

        # map flash and reward vector to place to find errors
        flash_place = []
        for t, trial in enumerate(vr_data['flash']):
            temp = vr_data['place'][t][trial-1]*conversion_factor
            temp = np.array([int(x) for x in temp])
            flash_place.append(temp)

        mod_flash = []
        for column in np.arange(np.max([len(x) for x in flash_place])):
            temp_col = []
            for trial in flash_place:
                if len(trial) > column:
                    temp_col.append(trial[column])
            mod_flash.append(Counter(temp_col).most_common(1)[0][0])
        thres_flash = threshold_leeway/conversion_factor

        exl_flash_place = []
        for i, pv in enumerate(flash_place):
            if len(pv) < 3:
                exl_flash_place.append(i)
            elif (pv[0] > mod_flash[0]+thres_flash or pv[0] < mod_flash[0]-thres_flash or
                  pv[1] > mod_flash[1]+thres_flash or pv[1] < mod_flash[1]-thres_flash or
                  pv[2] > mod_flash[2]+thres_flash or pv[2] < mod_flash[2]-thres_flash):

                exl_flash_place.append(i)

        for x in exl_flash_place:
            excluded_list.append(x)
            if x not in reason_dict:
                reason_dict[x] = []
            reason_dict[x].append('wrong location of flashes')

    # find irregularities in the reward vector data
    if 'reward' in vr_data.keys() and flash is True:
        exl_rew = np.where(np.array([len(x) for x in vr_data['reward']]) < 1)[0]
        for entry in exl_rew:
            excluded_list.append(entry)
            if entry not in reason_dict:
                reason_dict[entry] = []
            reason_dict[entry].append('wrong amount of rewards')

        reward_place = []
        for t, trial in enumerate(vr_data['reward']):
            temp = vr_data['place'][t][trial-1]*conversion_factor
            temp = np.array([int(x) for x in temp])
            reward_place.append(temp)

        mod_rew = Counter([int(x) for x in reward_place if len(x) == 1]).most_common(1)[0][0]
        thres_rew = threshold_leeway/conversion_factor

        for t, trial in enumerate(reward_place):
            if len(trial) > 0:
                if trial[0] > mod_rew+thres_rew or trial[0] < mod_rew-thres_rew:
                    excluded_list.append(t)
                    if t not in reason_dict:
                        reason_dict[t] = []
                    reason_dict[t].append('wrong location of reward')

    # find all unique entries in the excluded list and find out which trials are left
    excluded_list_cont = [0 if x in excluded_list else 1 for x in np.arange(len(trial_table))]
    reason_list_cont = [reason_dict[x] if x in excluded_list else [] for x in np.arange(len(trial_table))]
    
    complete_excluded = []
    complete_reason = []
    cur_ind = 0
    for state in (trial_table.marked_for_analysis == 1).values:
        if state == True:
            complete_excluded.append(excluded_list_cont[cur_ind])
            complete_reason.append(reason_list_cont[cur_ind])
            cur_ind = cur_ind+1
        else:
            complete_excluded.append(0)
            complete_reason.append([])
    
    trial_table['marked_for_place_conversion'] = complete_excluded
    trial_table['reason_for_place_exclusion'] = complete_reason

    return trial_table


def extend_trial_table(trial_table, vr_data, FcellCorr, conversion_factor=1/1024*64):
    
    additional_columns =  ['time_len_s_vr',
                           'time_len_s_ca',
                           'place_len_ticks',
                           'place_len_cm',
                           'reward_tick',
                           'reward_place',
                           'reward_place_cm',
                           'start_place',
                           'start_place_cm',
                           'end_place',
                           'end_place_cm',
                           'flash_tick',
                           'flash_place',
                           'flash_place_cm',
                           'teleport_tick',
                           'teleport_place',
                           'teleport_place_cm']
    
    per_trial_data = []
    for t_ind in np.arange(len(trial_table[trial_table.marked_for_analysis == 1])):
        time_len_s_vr = vr_data['time'][t_ind][-1]
        time_len_s_ca = len(FcellCorr[t_ind][0])/30
        place_len_ticks = np.max(vr_data['place'][t_ind]) - np.min(vr_data['place'][t_ind][vr_data['place'][t_ind] != -1])
        place_len_cm = place_len_ticks*conversion_factor
        reward_tick = vr_data['reward'][t_ind]
        reward_place = vr_data['place'][t_ind][vr_data['reward'][t_ind]-1]
        reward_place_cm = reward_place*conversion_factor
        start_place = np.min(vr_data['place'][t_ind][vr_data['place'][t_ind] > -1])
        start_place_cm = start_place*conversion_factor
        end_place = np.max(vr_data['place'][t_ind][vr_data['place'][t_ind] > -1])
        end_place_cm = end_place*conversion_factor
        flash_tick = vr_data['flash'][t_ind]
        flash_place = vr_data['place'][t_ind][vr_data['flash'][t_ind]-2]
        flash_place_cm = vr_data['place'][t_ind][vr_data['flash'][t_ind]-2]*conversion_factor
        teleport_tick = vr_data['teleport'][t_ind]
        teleport_place = vr_data['place'][t_ind][vr_data['teleport'][t_ind]-1]
        teleport_place_cm = teleport_place*conversion_factor
        per_trial_data.append([time_len_s_vr,
                               time_len_s_ca,
                               place_len_ticks,
                               place_len_cm,
                               reward_tick,
                               reward_place,
                               reward_place_cm,
                               start_place,
                               start_place_cm,
                               end_place,
                               end_place_cm,
                               flash_tick,
                               flash_place,
                               flash_place_cm,
                               teleport_tick, 
                               teleport_place,
                               teleport_place_cm])
        
    complete_data = []
    cur_ind = 0
    for state in (trial_table.marked_for_analysis == 1).values:
        if state == True:
            complete_data.append(per_trial_data[cur_ind])
            cur_ind = cur_ind+1
        else:
            complete_data.append([[] for x in np.arange(len(per_trial_data[0]))])

    for c, column in enumerate(additional_columns):
        trial_table[column] = [x[c] for x in complete_data]
        
    trial_table['stimID'] = ['012' if type(x) == float else x for x in trial_table['stimID'].values]
    
        
    return trial_table


def apply_exclusion(trial_table, vr_data, Fcell, FcellNeu, FcellCorr, spikes, settings, save=True, pupil_data={}):
    """
    Restrict data to trials marked for place conversion in trial table

    Input:
        trial_table     : table of all trials and specific parameters
        vr_data         : dict of vr_data containing place, time etc.
        Fcell           : list of list of numpy arrays containing the per cell and trial raw floruescence
        FcellNeu        : list of list of numpy arrays containing the per cell and trial neuropil floruescence
        FcellCorr       : list of list of numpy arrays containing the per cell and trial neuropil corrected floruescence
        spikes          : list of list of numpy arrays containing the per cell and trial spike probabilities
        
    Output:
        trial_table     : table of all trials and specific parameters
        vr_data_ex      : same as input but trials not marked for place conversion are excluded
        Fcell_ex        : same as input but trials not marked for place conversion are excluded
        FcellNeu_ex     : same as input but trials not marked for place conversion are excluded
        FcellCorr_ex    : same as input but trials not marked for place conversion are excluded
        spikes_ex       : same as input but trials not marked for place conversion are excluded

    """

    analysis_list = np.where(trial_table.marked_for_place_conversion[trial_table.marked_for_analysis == 1].values)[0]

    # exclude trials from vr_data
    vr_data_ex = {}
    for key in vr_data:
        if type(vr_data[key]) == list:
            vr_data_ex[key] = [x for i, x in enumerate(vr_data[key]) if i in analysis_list]
        elif type(vr_data[key]) == np.ndarray:
            vr_data_ex[key] = vr_data[key][np.array(analysis_list)]

    if len(pupil_data) > 0:
        pupil_data_ex = {}
        for key in pupil_data:
            if type(pupil_data[key]) == list:
                pupil_data_ex[key] = [x for i, x in enumerate(pupil_data[key]) if i in analysis_list]
            elif type(pupil_data[key]) == np.ndarray:
                pupil_data_ex[key] = pupil_data[key][np.array(analysis_list)]

    # exclude trials from calcium data
    Fcell_ex = [x for i,x in enumerate(Fcell) if i in analysis_list]
    FcellNeu_ex = [x for i,x in enumerate(FcellNeu) if i in analysis_list]
    FcellCorr_ex = [x for i,x in enumerate(FcellCorr) if i in analysis_list]
    spikes_ex = [x for i,x in enumerate(spikes) if i in analysis_list]
    
    # save trial table
    if save is True:
        trial_table.to_csv(os.path.join(settings['general']['analysis_path'], 'trial_table_anaylsis_'+'_'.join([settings['experiment']['task'], settings['experiment']['MouseID'], settings['experiment']['date']])+'.csv'))

    if len(pupil_data) > 0:
        return trial_table, vr_data_ex, Fcell_ex, FcellNeu_ex, FcellCorr_ex, spikes_ex, pupil_data_ex
    else:
        return trial_table, vr_data_ex, Fcell_ex, FcellNeu_ex, FcellCorr_ex, spikes_ex


def split_super_session(settings, trial_table, vr_data, Fcell, FcellNeu, FcellCorr, spikes, pupil_data={}):
    
    split_data = {}
    
    for exp in trial_table.exp.unique():
        split_data[exp] = {}
        inds = trial_table.reset_index()[trial_table.reset_index().exp == exp].index
        split_data[exp]['vr_data'] = {}
        for key in vr_data:
            split_data[exp]['vr_data'][key] = [x for i,x in enumerate(vr_data[key]) if i in inds]
            
        split_data[exp]['pupil_data'] = {}
        if len(pupil_data) > 0:
            for key in pupil_data:
                split_data[exp]['pupil_data'][key] = [x for i,x in enumerate(pupil_data[key]) if i in inds]     

                
        split_data[exp]['Fcell'] = Fcell[inds]
        split_data[exp]['FcellNeu'] = FcellNeu[inds]
        split_data[exp]['FcellCorr'] = FcellCorr[inds]
        split_data[exp]['spikes'] = spikes[inds]
        split_data[exp]['trial_table'] = trial_table.iloc[inds]
        
        
    return split_data

        
def convolve_spikes(spikes_ex, convolve_kernel='gaussian', ops=[]):
    
    if convolve_kernel == 'gaussian':
        from scipy import signal
        kernel = signal.gaussian(30, std=3)
        mode = 'same'
        
    elif convolve_kernel == 'decay':
        import math
        kernel = []
        for x in np.arange(100):
            kernel.append(math.e**(-x/10))
        mode = 'full'
    
    elif convolve_kernel == 'decay_long':
        import math
        kernel = []
        for x in np.arange(100):
            kernel.append(math.e**(-x/20))
        mode = 'full'    
        
    elif convolve_kernel == 'exp':
        
        fWidth = ops['tau'] * ops['fs']
        fLength = int(np.ceil(fWidth * 5))
        kernel = exponential_filter(fWidth, fLength)
        # kernel = np.concatenate((np.zeros(len(kernel)), kernel))
        mode = 'full'
        
        
    convolved_spikes = [[] for x in np.arange(len(spikes_ex))]
    for c in np.arange(len(spikes_ex[0])):
        temp_cell = np.concatenate([x[c] for x in spikes_ex])
        temp_convolved = np.convolve(temp_cell, kernel, mode)
        temp_convolved = temp_convolved[:len(temp_cell)]
        if len(temp_cell) < len(kernel):
            print('warning')
        for t, trial_size in enumerate([len(x[0]) for x in spikes_ex]):
            if t == 0:
                convolved_spikes[t].append(np.array(temp_convolved[:trial_size]))
                last_size = trial_size
            else:
                convolved_spikes[t].append(np.array(temp_convolved[last_size:last_size+trial_size]))
                last_size = last_size + trial_size

    return convolved_spikes



        
def convolve_spikes_cellwise(spikes_mapped, convolve_kernel='gaussian', ops=[]):
    
    if convolve_kernel == 'gaussian':
        from scipy import signal
        kernel = signal.gaussian(30, std=3)
        mode = 'same'
        
    elif convolve_kernel == 'decay':
        import math
        kernel = []
        for x in np.arange(100):
            kernel.append(math.e**(-x/10))
        mode = 'full'
    
    elif convolve_kernel == 'decay_long':
        import math
        kernel = []
        for x in np.arange(100):
            kernel.append(math.e**(-x/20))
        mode = 'full'    
        
    elif convolve_kernel == 'exp':
        
        fWidth = ops['tau'] * ops['fs']
        fLength = int(np.ceil(fWidth * 5))
        kernel = exponential_filter(fWidth, fLength)
        # kernel = np.concatenate((np.zeros(len(kernel)), kernel))
        mode = 'full'
        
        
    convolved_spikes = [[] for x in np.arange(len(spikes_mapped))]
    for c, cell in enumerate(spikes_mapped):
        temp_cell = np.concatenate(cell)
        temp_convolved = np.convolve(temp_cell, kernel, mode)
        temp_convolved = temp_convolved[:len(temp_cell)]
        if len(temp_cell) < len(kernel):
            print('warning')
        for t, trial_size in enumerate([len(x) for x in cell]):
            if t == 0:
                convolved_spikes[c].append(np.array(temp_convolved[:trial_size]))
                last_size = trial_size
            else:
                convolved_spikes[c].append(np.array(temp_convolved[last_size:last_size+trial_size]))
                last_size = last_size + trial_size

    return convolved_spikes



def exponential_filter(decay_time, kernel_length):
    # Generate a 1D exponential filter kernel with specified decay time.
    # Decay time indicates after how many samples 1/e (37%) of the intial value is reached

    # Compute the decay factor from the decay time
    alpha = np.exp(-1 / float(decay_time))

    # Generate the kernel values
    kernel = np.zeros(kernel_length)
    for t in range(kernel_length):
        kernel[t] = alpha ** (t)

    # Normalize the kernel to have unit energy
    kernel = kernel / np.sum(kernel)

    # Adjust the kernel length if necessary to ensure that it sums to 1
    kernel_sum = np.sum(kernel)
    if abs(kernel_sum - 1) > np.finfo(float).eps:
        diff = 1 - kernel_sum
        kernel[-1] = kernel[-1] + diff

    return kernel   