# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:31:59 2022

@author: rueland
"""

import pandas as pd
import numpy as np
import os
import datetime
import h5py
from scipy import interpolate


def load_dataset_super_session(folder, 
                               extraction='cellpose', 
                               iscell_type='automatic', 
                               complete=False, 
                               load_pupil=False):
    # load settings
    settings_path = [os.path.join(folder, x) for x in os.listdir(folder) if 'settings' in x and extraction in x][0]
    settings = np.load(settings_path, allow_pickle=True).item()
    settings['experiment']['task'] = 'SUPER_SESSION'
    
    # load trial table
    trial_table = pd.read_csv(os.path.join(settings['general']['savepath_folder'], 
                                           [x for x in os.listdir(settings['general']['savepath_folder']) if 'trial_table' in x and 'SUPER_SESSION' in x][0]), 
                              index_col=0, dtype={'stimID' : str})
    
    # load corresponding ca_data
    Fcell, FcellNeu, FcellCorr, spikes, file_list, stat, ops, iscell = load_suite2p_super_session(folder, 
                                                                                                  iscell_type=iscell_type,
                                                                                                  extraction=extraction)

    # load corresponding vr_data
    vr_data, trial_table = load_vr_data_super_session(trial_table, file_list)
    
    if load_pupil == True:
        # load corresponding pupil_data
        try:
            pupil_data = load_pupil_super_session(trial_table, file_list, folder, filtered=True)
        except:
            pupil_data = {}
            print('Could not load pupil data')
    else:
        pupil_data = {}
        
    # rearrange axes to fit rest of anaylsis
    Fcell = np.swapaxes(np.array(Fcell, dtype='object'), 0, 1)
    FcellNeu = np.swapaxes(np.array(FcellNeu, dtype='object'), 0, 1)
    FcellCorr = np.swapaxes(np.array(FcellCorr, dtype='object'), 0, 1)
    spikes = np.swapaxes(np.array(spikes, dtype='object'), 0, 1)

    if complete and load_pupil:
        return settings, trial_table, vr_data, Fcell, FcellNeu, FcellCorr, spikes, stat, ops, iscell, pupil_data
    elif complete:
        return settings, trial_table, vr_data, Fcell, FcellNeu, FcellCorr, spikes, stat, ops, iscell
    elif load_pupil==True:
        return settings, trial_table, vr_data, Fcell, FcellNeu, FcellCorr, spikes, pupil_data
    else:
        return settings, trial_table, vr_data, Fcell, FcellNeu, FcellCorr, spikes




def load_suite2p_super_session(folder, iscell_type='automatic', extraction='cellpose'):
    
    complete_folder = os.path.join(folder, 'suite2p_'+extraction, 'plane0')
    
    Fcell = np.load(os.path.join(complete_folder, "F.npy"))
    FcellNeu = np.load(os.path.join(complete_folder, "Fneu.npy"))    
    spikes = np.load(os.path.join(complete_folder, "spks.npy"))
    if iscell_type == 'automatic':
        iscell = np.load(os.path.join(complete_folder, "iscell.npy"))
    elif iscell_type == 'manual':
        iscell = np.load(os.path.join(complete_folder, "iscell_manual.npy"))
    else:
        print('Iscell type is not specified. Iscell could not be loaded')
    stat = np.load(os.path.join(complete_folder, "stat.npy"), allow_pickle=True)
    ops = np.load(os.path.join(complete_folder, "ops.npy"), allow_pickle=True).item()    
    
    spikes_filt = spikes[iscell[:,0].astype(bool)]
    spikes_filt = [np.split(x, np.cumsum(ops['frames_per_file'])[:-1]) for x in spikes_filt]
        
    F_filt = Fcell[iscell[:,0].astype(bool)]
    FNeu_filt = FcellNeu[iscell[:,0].astype(bool)]
    Fc = F_filt - ops['neucoeff'] * FNeu_filt
    
    F_filt = [np.split(x, np.cumsum(ops['frames_per_file'])[:-1]) for x in F_filt]
    FNeu_filt = [np.split(x, np.cumsum(ops['frames_per_file'])[:-1]) for x in FNeu_filt]
    Fc = [np.split(x, np.cumsum(ops['frames_per_file'])[:-1]) for x in Fc]
  
    file_list = ops['filelist']
  
    return F_filt, FNeu_filt, Fc, spikes_filt, file_list, stat, ops, iscell


def load_vr_data_single_session(trial_table):
    
    path_list = trial_table[trial_table['marked_for_analysis'] != 0].vr_file.values
    
    flash_ticks = []
    reward_ticks = []
    reward_modalities = []
    lick_ticks = []
    steps = []
    teleport_ticks = []
    times = []
    for path in path_list:
        cur_vr = pd.read_csv(path, delimiter=';', usecols=['time', 'steps', 'cue_left', 'reward_ID', 'reward', 'sensor_barrier_ID', 'lick'])
        cur_vr_filt = cur_vr[cur_vr.time != '0'].reset_index(drop=True)
        clean_licks = []
        for x in np.arange(len(cur_vr_filt)):
            temp_val = cur_vr_filt.lick.iloc[x]
            try:
                conv_val = float(temp_val)
            except:
                conv_val = temp_val
            clean_licks.append(conv_val)        
        cur_flash_ticks = cur_vr_filt[cur_vr_filt.cue_left != -1].index.values
        cur_reward_tick = cur_vr_filt[cur_vr_filt.reward_ID != -1].index.values
        cur_reward_modality = cur_vr_filt.iloc[cur_reward_tick+1].reward.values
        cur_lick_tick = cur_vr_filt[cur_vr_filt.lick != -1].index.values
        cur_teleport_tick = cur_vr_filt[cur_vr_filt.sensor_barrier_ID != -1].index.values[-1]
        cur_steps = cur_vr_filt.steps.values
        cur_time = [datetime.datetime.strptime(x, '%H:%M:%S.%f') for x in cur_vr_filt.time.values]
        cur_time = np.array([(x - cur_time[0]).total_seconds() for x in cur_time])
        flash_ticks.append(cur_flash_ticks)
        reward_ticks.append(cur_reward_tick)
        reward_modalities.append(cur_reward_modality)
        lick_ticks.append(cur_lick_tick)
        steps.append(cur_steps)
        teleport_ticks.append(cur_teleport_tick)
        times.append(cur_time)
    vr_data = {}
    vr_data['place'] = steps
    vr_data['flash'] = flash_ticks
    vr_data['reward'] = reward_ticks
    vr_data['reward_mod'] = reward_modalities
    vr_data['lick'] = lick_ticks
    vr_data['time'] = times
    vr_data['teleport'] = teleport_ticks
    

    return vr_data


def load_vr_data_super_session(trial_table, file_list):
    path_list = trial_table[trial_table['marked_for_analysis'] != 0].vr_file.values
    
    # match file list with path lists 
    tif_list = [os.path.split(x)[1].split('.tif')[0] for x in file_list]
    vr_list = [os.path.split(x)[1].split('.csv')[0] for x in path_list]
    vr_list = [x.split('tr-')[0] + '{:05d}'.format(int(x.split('tr-')[1]))  for x in vr_list]
    
    # sort_ids = []
    # for tif_id, tif in enumerate(tif_list):
    #     vr_id = np.where(tif == np.array(vr_list))[0][0]
        
    #     sort_ids.append([tif_id, vr_id])
    
    
    tif_exps =  [x.split('_')[2] for x in tif_list]
    vr_exps =  [x.split('_')[2] for x in vr_list]
    
    if tif_exps == vr_exps:
        sorted_vr_path_list = path_list
        sorted_tt = trial_table[trial_table['marked_for_analysis'] != 0].copy(deep=True)
    else:
        print('Failure to match trials')
        return
    
        
    # sorted_vr_path_list = np.array(path_list)[[x[1] for x in sort_ids]]
    
    # sorted_tt = trial_table[trial_table['marked_for_analysis'] != 0].iloc[[x[1] for x in sort_ids]]
    sorted_tt['exp'] = [x.split('\\')[-1].split('_')[2] for x in sorted_vr_path_list]
    
    flash_ticks = []
    reward_ticks = []
    reward_modalities = []
    lick_ticks = []
    steps = []
    teleport_ticks = []
    times = []
    ypos = []
    
    for path in sorted_vr_path_list:
        cur_vr = pd.read_csv(path, delimiter=';', usecols=['time', 'ypos', 'steps', 'cue_left', 'reward_ID', 'reward', 'lick', 'sensor_barrier_ID'])
        cur_vr_filt = cur_vr[cur_vr.time != '0'].reset_index(drop=True)
        
        clean_licks = []
        for x in np.arange(len(cur_vr_filt)):
            temp_val = cur_vr_filt.lick.iloc[x]
            try:
                conv_val = float(temp_val)
            except:
                conv_val = temp_val
            clean_licks.append(conv_val)
        
        cur_vr_filt['lick'] = clean_licks
        cur_flash_ticks = cur_vr_filt[cur_vr_filt.cue_left != -1].index.values
        cur_reward_tick = cur_vr_filt[cur_vr_filt.reward_ID != -1].index.values
        cur_reward_modality = cur_vr_filt.iloc[cur_reward_tick+1].reward.values
        cur_lick_tick = cur_vr_filt[cur_vr_filt.lick != -1].index.values        
        cur_teleport_tick = cur_vr_filt[cur_vr_filt.sensor_barrier_ID != -1].index.values[-1]
        cur_steps = cur_vr_filt.steps.values
        cur_time = [datetime.datetime.strptime(x, '%H:%M:%S.%f') for x in cur_vr_filt.time.values]
        cur_time = np.array([(x - cur_time[0]).total_seconds() for x in cur_time])
        cur_ypos = cur_vr_filt['ypos'].values
        flash_ticks.append(cur_flash_ticks)
        reward_ticks.append(cur_reward_tick)
        reward_modalities.append(cur_reward_modality)
        lick_ticks.append(cur_lick_tick)
        steps.append(cur_steps)
        teleport_ticks.append(cur_teleport_tick)
        times.append(cur_time)
        ypos.append(cur_ypos)

    vr_data = {}
    vr_data['place'] = steps
    vr_data['flash'] = flash_ticks
    vr_data['reward'] = reward_ticks
    vr_data['lick'] = lick_ticks
    vr_data['reward_mod'] = reward_modalities    
    vr_data['time'] = times
    vr_data['teleport'] = teleport_ticks
    vr_data['ypos'] = ypos
    
    return vr_data, sorted_tt


def load_pupil_super_session(trial_table, file_list, folder, filtered=True):
       
       
    path_list = trial_table[trial_table['marked_for_analysis'] != 0].pupil_file.values
    
    # match file list with path lists 
    tif_list = [os.path.split(x)[1].split('.tif')[0] for x in file_list]
    pupil_list = [os.path.split(x)[1].split('.tdms')[0] for x in path_list]
    pupil_list = [x.split('tr-')[0] + '{:05d}'.format(int(x.split('tr-')[1].split('_')[0]))  for x in pupil_list]

 
    tif_exps =  [x.split('_')[2] for x in tif_list]
    pupil_exps =  [x.split('_')[2] for x in pupil_list]
    
    if tif_exps == pupil_exps:
        sorted_pupil_path_list = path_list
        sorted_tt = trial_table[trial_table['marked_for_analysis'] != 0].copy(deep=True)
    else:
        print('Failure to match trials')
        return

    sorted_tt['exp'] = [x.split('\\')[5] for x in sorted_pupil_path_list]
    
    if filtered is True:
        pupil_files = [os.path.join(os.path.join(folder, 'pupil_dlc'), x) for x in os.listdir(os.path.join(folder, 'pupil_dlc')) if 'filtered' in x]
    else:
        pupil_files = [os.path.join(os.path.join(folder, 'pupil_dlc'), x) for x in os.listdir(os.path.join(folder, 'pupil_dlc')) if 'filtered' not in x]
        
        
    comp_pup_files = [os.path.split(x)[1].split('DLC')[0] for x in pupil_files]
    
    # match pupil files and extracted pupil files
    extracted_sorted_pupil_path_list = []
    for i, sorted_file in enumerate(sorted_pupil_path_list):
        try:
            sorted_ending = os.path.split(sorted_file)[1].split('.tdms')[0]
            extracted_sorted_pupil_path_list.append(pupil_files[np.where(np.array(comp_pup_files) == sorted_ending)[0][0]])
        except:
            print('Failure to match pupil trials')
    
    
    x = []
    y = []
    r = []
    for file in extracted_sorted_pupil_path_list:
        temp_df = pd.read_csv(file, header=2)
        x.append((temp_df['x.3'] + temp_df['x.1']).values/2)
        y.append((temp_df['y'] + temp_df['y.2']).values/2)
        r.append(((temp_df['y.2'] - temp_df['y']) + (temp_df['x.1'] - temp_df['x.3'])).values/4)
       
    pupil_data = {}
    pupil_data['x'] = x
    pupil_data['y'] = y
    pupil_data['r'] = r
    
    return pupil_data


def load_pupil_dlc(path_dlc_folder, filtered=True):
    
    if filtered is True:
        pupil_files = [os.path.join(path_dlc_folder, x) for x in os.listdir(path_dlc_folder) if 'filtered' in x]
    else:
        pupil_files = [os.path.join(path_dlc_folder, x) for x in os.listdir(path_dlc_folder) if 'filtered' not in x]
    
    
    complete_df = []
    for file in pupil_files:
        temp_df = pd.read_csv(file, header=2)
        
        exp = os.path.split(file)[-1].split('_')[2]
        mouse = os.path.split(file)[-1].split('_')[3]
        date = os.path.split(file)[-1].split('_')[4]
        
        dlc_df = pd.DataFrame(data=np.arange(len(temp_df)), columns=['ind'])
        dlc_df['x'] = (temp_df['x.3'] + temp_df['x.1'])/2
        dlc_df['y'] = (temp_df['y'] + temp_df['y.2'])/2
        dlc_df['r'] = ((temp_df['y.2'] - temp_df['y']) + (temp_df['x.1'] - temp_df['x.3']))/4
        dlc_df['type'] = ['dlc' for x in np.arange(len(dlc_df))]
        dlc_df['exp'] = [exp for x in np.arange(len(dlc_df))]
        dlc_df['mouse'] = [mouse for x in np.arange(len(dlc_df))]
        dlc_df['date'] = [date for x in np.arange(len(dlc_df))]
                
        if len(complete_df) == 0:
            complete_df = dlc_df
        else:
            complete_df = complete_df.append(dlc_df)
        
    return complete_df


def load_pupil_data_old(path_old, interpolation=True):
    
    # get old results
    f = h5py.File(path_old, 'r')
    trial_ids = [int(f['trial_ids'][x][0]) for x in np.arange(len(f['r']))]
    frame_ids = []
    old_df = []
    old_df_header = ['x', 'y', 'r', 'trial']
    
    for x in np.arange(len(f['r'])):
        temp_data = {}
        for key in ['x', 'y', 'r']:
            temp_data[key] = np.squeeze(f[f[key][x][0]][:])
            if key == 'x':
                frame_ids.append(len(temp_data[key]))
            
        for e in np.arange(len(temp_data['r'])):
            old_df.append([temp_data['x'][e], temp_data['y'][e], temp_data['r'][e], trial_ids[x]])
    
        
    old_df = pd.DataFrame(data=old_df, columns=old_df_header)
    old_df['ind'] = np.arange(len(old_df))       
    old_df['type'] = ['old' for x in np.arange(len(old_df))]   
    
    
    # use old interpolation to clean pupil trace
    
    if interpolation is True:
        # convert to array
        concat_x = old_df.x.copy(deep=True).values
        concat_y = old_df.y.copy(deep=True).values
        concat_r = old_df.r.copy(deep=True).values
        
        # exclude 0s and fill with nan
        concat_x[concat_x == 0] = np.nan
        concat_y[concat_y == 0] = np.nan
        concat_r[concat_r == 0] = np.nan
        
        # exclude to high of changes and replace with nan
        concat_x[np.where(np.abs(np.diff(concat_x)) > 10)[0]+1] = np.nan
        concat_y[np.where(np.abs(np.diff(concat_y)) > 10)[0]+1] = np.nan
        concat_r[np.where(np.abs(np.diff(concat_r)) > 10)[0]+1] = np.nan
        
        # replace nan by lineaer interpolation
        clean_x = fill_nan(concat_x)
        clean_y = fill_nan(concat_y)
        clean_r = fill_nan(concat_r)
    
        old_df_interp = old_df.copy(deep=True)
        old_df_interp.x = clean_x
        old_df_interp.y = clean_y
        old_df_interp.r = clean_r
        old_df_interp['type'] = ['old_interp' for x in np.arange(len(old_df))]

        return old_df_interp

    else:
        return old_df

def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    return B