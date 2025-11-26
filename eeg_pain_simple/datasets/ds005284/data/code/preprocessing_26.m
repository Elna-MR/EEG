%% 26ByBiosemi
clear;
clc;
eeglab

behav_data = tdfread('H:\BIDS\BIDS_26_ByBiosemi\participants.tsv');%read the information of participants,change your path
sub = behav_data.participant_id;
event_marker = {'condition 54'};
input_path = 'H:\BIDS\BIDS_26_ByBiosemi\';% change your path
save_path1 = 'H:\BIDS\BIDS_26_ByBiosemi\derivatives\session_merged_data\';% change your path
save_path2 = 'H:\BIDS\BIDS_26_ByBiosemi\derivatives\filter\';% change your path
info=readmatrix('H:\BIDS\CSV\participants_singletrial_26ByBiosemi.csv');% table contains rating and intensity information,change your path

for subi = 1:size(sub, 1)
    % merge data
    input_path_1 = [input_path sub(subi, :) '\eeg\'];
    filename1= [input_path_1,sub(subi, :) '_task-26ByBiosemi_eeg.bdf'];
    EEG = pop_biosig(filename1);
    % modify the electrodes' name
    EEG = pop_select( EEG, 'channel',{'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16','A17','A18','A19','A20','A21','A22','A23','A24','A25','A26','A27','A28','A29','A30','A31','A32','B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12','B13','B14','B15','B16','B17','B18','B19','B20','B21','B22','B23','B24','B25','B26','B27','B28','B29','B30','B31','B32'});
    % channel location
    EEG=pop_chanedit(EEG, 'changefield',{1 'labels' 'FP1'},'changefield',{2 'labels' 'AF7'},'changefield',{3 'labels' 'AF3'},'changefield',{4 'labels' 'F1'},'changefield',{5 'labels' 'F3'},'changefield',{6 'labels' 'F5'},'changefield',{7 'labels' 'F7'},'changefield',{8 'labels' 'FT7'},'changefield',{9 'labels' 'FC5'},'changefield',{10 'labels' 'FC3'},'changefield',{11 'labels' 'FC1'},'changefield',{12 'labels' 'C1'},'changefield',{13 'labels' 'C3'},'changefield',{14 'labels' 'C5'},'changefield',{15 'labels' 'T7'},'changefield',{16 'labels' 'TP7'},'changefield',{17 'labels' 'CP5'},'changefield',{18 'labels' 'CP3'},'changefield',{19 'labels' 'CP3'},'changefield',{20 'labels' 'CP1'},'changefield',{21 'labels' 'P1'},'changefield',{22 'labels' 'P3'},'changefield',{23 'labels' 'P5'},'changefield',{24 'labels' 'P7'},'changefield',{19 'labels' 'CP1'},'changefield',{20 'labels' 'P1'},'changefield',{21 'labels' 'P3'},'changefield',{22 'labels' 'P5'},'changefield',{23 'labels' 'P7'},'changefield',{24 'labels' 'P9'},'changefield',{25 'labels' 'PO7'},'changefield',{26 'labels' 'PO3'},'changefield',{27 'labels' 'O1'},'changefield',{28 'labels' 'LZ'},'changefield',{29 'labels' 'OZ'},'changefield',{30 'labels' 'POZ'},'changefield',{31 'labels' 'PZ'},'changefield',{32 'labels' 'CPZ'},'changefield',{33 'labels' 'FPZ'},'changefield',{34 'labels' 'FP2'},'changefield',{35 'labels' 'AF8'},'changefield',{36 'labels' 'AF4'},'changefield',{37 'labels' 'AFZ'},'changefield',{38 'labels' 'FZ'},'changefield',{39 'labels' 'F2'},'changefield',{40 'labels' 'F4'},'changefield',{41 'labels' 'F6'},'changefield',{42 'labels' 'F8'},'changefield',{43 'labels' 'FT8'},'changefield',{44 'labels' 'FC6'},'changefield',{45 'labels' 'fc4'},'changefield',{45 'labels' 'FC4'},'changefield',{46 'labels' 'FC2'},'changefield',{47 'labels' 'FCZ'},'changefield',{48 'labels' 'CZ'},'changefield',{49 'labels' 'C2'},'changefield',{50 'labels' 'C4'},'changefield',{51 'labels' 'C6'},'changefield',{52 'labels' 'T8'},'changefield',{53 'labels' 'TP8'},'changefield',{54 'labels' 'CP6'},'changefield',{55 'labels' 'CP4'},'changefield',{56 'labels' 'CP2'},'changefield',{57 'labels' 'P2'},'changefield',{58 'labels' 'P4'},'changefield',{59 'labels' 'P6'},'changefield',{60 'labels' 'P8'},'changefield',{61 'labels' 'P10'},'changefield',{62 'labels' 'PO8'},'changefield',{63 'labels' 'PO4'},'changefield',{64 'labels' 'O2'},'lookup',...
        'F:\matlab2019a\toolbox\eeglab\plugins\dipfit\standard_BESA\standard-10-5-cap385.elp');% change your path
    % save
    EEG = pop_saveset( EEG, 'filename',[sub(subi, :) '_26ByBiosemi.set'],'filepath',save_path1);
    
    % resample
    EEG = pop_resample( EEG, 1000);
    
    % filter
    EEG = pop_eegfiltnew(EEG, 'locutoff',1);
    EEG = pop_eegfiltnew(EEG, 'hicutoff',100);
    EEG= pop_eegfiltnew(EEG, 'locutoff',48,'hicutoff',52,'revfilt',1); % notch filtering
    EEG = pop_epoch( EEG, event_marker, [-1  2], 'newname', 'Merged datasets epochs', 'epochinfo', 'yes'); % epoching
    EEG = pop_rmbase( EEG, [-1000 0]);
    
    % add rating and power
    trilnum=length(EEG.event);
    for e = 1:length(EEG.event)
        EEG.event(e).rating = info(e+subi*trilnum-trilnum,4) ;
        EEG.event(e).laser_power = info(e+subi*trilnum-trilnum,3) ;
    end
    % save
    EEG = pop_saveset(EEG, 'filename', [sub(subi, :) '_26ByBiosemi.set'], 'filepath', save_path2);
end

%% Manually deleting bad trials and interpolating bad channels

%% ICA
clear;
clc;
eeglab

behav_data = tdfread('G:\BIDS\BIDS_26_ByBiosemi\participants.tsv');%read the information of `participants,change your path
sub = behav_data.participant_id;
input_path = 'G:\BIDS\BIDS_26_ByBiosemi\derivatives\no_ica';% change your path
save_path = 'G:\BIDS\BIDS_26_ByBiosemi\derivatives\ica';% change your path

for subi = 1:size(sub, 1)
    file_name = [sub(subi, :) '_26ByBiosemi.set'];
    EEG = pop_loadset('filename', file_name, 'filepath', input_path);
    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on');%run ICA
    EEG = pop_saveset(EEG, 'filename',[sub(subi,:) '_26ByBiosemi.set'],'filepath',save_path);
end

%% Manually deleting bad components

%% filter and rereference 
clear;
clc;
eeglab

behav_data = tdfread('G:\bids_ALL_result\BIDS_26_ByBiosemi\participants.tsv');% change your path
sub = behav_data.participant_id;
input_path = 'G:\bids_ALL_result\BIDS_26_ByBiosemi\derivatives\after_ica\';% change your path
save_path1 = 'G:\bids_ALL_result\BIDS_26_ByBiosemi\derivatives\refilter_30\';% change your path
save_path2 = 'G:\bids_ALL_result\BIDS_26_ByBiosemi\derivatives\rerefer\';% change your path
for subi = 1:size(sub, 1)
     file_name = [sub(subi, :) '_26ByBiosemi.set'];
     EEG = pop_loadset('filename', file_name, 'filepath', input_path);
     % filter
     EEG = pop_eegfiltnew(EEG, 'hicutoff',30);
     %save
     EEG = pop_saveset(EEG, 'filename',file_name,'filepath',save_path1); 
     % rereference
    EEG = pop_reref(EEG, []);
     %save
     EEG = pop_saveset(EEG, 'filename', file_name, 'filepath', save_path2);
end