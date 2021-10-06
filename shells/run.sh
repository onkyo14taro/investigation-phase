#!/bin/bash

dir_project="$(dirname $(cd $(dirname $0); pwd))"
cd "${dir_project}/scripts"
file_main="main.py"


############################################################
### Configuration
############################################################
readonly BATCH_SIZE=32
readonly TRAINING_STEP_UNIT=50000     # Steps for one training (because there is a limit to the amount of continuous execution time)
readonly OPTIONS=(--gpus 1 --spec_augment --dropout_last 0.2 --weight_decay 1e-5 --label_smooth 0.1 --batch_size ${BATCH_SIZE})


############################################################
### Functions
############################################################
function calc_epochs () {
    dataset_size=${1:?}
    target_steps=${2:?}
    steps_per_epoch=$(($dataset_size / $BATCH_SIZE))
    total_epochs=$(($target_steps / $steps_per_epoch))
    if [ $(($target_steps % $steps_per_epoch)) -ne 0 ]; then
        total_epochs=$(($total_epochs + 1))
    fi
    echo $total_epochs
}

function calc_max_epochs () {
    dataset_size=${1:?}
    training_time=${2:?}
    total_epochs=$(calc_epochs $dataset_size $TRAINING_STEP_UNIT)
    total_epochs=$(($total_epochs * $training_time))
    echo $total_epochs
}

function acoustic_scenes () {
    dataset_size=6122
    python $file_main --task_name ACOUSTIC_SCENES \
        --max_epochs 2 \
        --earlystopping_patience $(calc_epochs $dataset_size 20000) \
        ${OPTIONS[@]} ${@:2}
}

function bird_audio () {
    dataset_size=27690
    python $file_main --task_name BIRD_AUDIO \
        --max_epochs $(calc_max_epochs $dataset_size $1) \
        --earlystopping_patience $(calc_epochs $dataset_size 50000) \
        ${OPTIONS[@]} ${@:2}
}

function emotion () {
    dataset_size=5146
    python $file_main --task_name EMOTION \
        --max_epochs $(calc_max_epochs $dataset_size $1) \
        --earlystopping_patience $(calc_epochs $dataset_size 20000) \
        ${OPTIONS[@]} ${@:2}
}

function speech_commands () {
    dataset_size=84843
    python $file_main --task_name SPEECH_COMMANDS \
        --max_epochs $(calc_max_epochs $dataset_size $1) \
        --earlystopping_patience $(calc_epochs $dataset_size 20000) \
        ${OPTIONS[@]} ${@:2}
}

function speaker_vox () {
    dataset_size=128086
    python $file_main --task_name SPEAKER_VOX \
        --max_epochs $(calc_max_epochs $dataset_size $1) \
        --earlystopping_patience $(calc_epochs $dataset_size 20000) \
        ${OPTIONS[@]} ${@:2}
}

function language () {
    dataset_size=148654
    python $file_main --task_name LANGUAGE \
        --max_epochs $(calc_max_epochs $dataset_size $1) \
        --earlystopping_patience $(calc_epochs $dataset_size 50000) \
        ${OPTIONS[@]} ${@:2}
}

function music_inst () {
    dataset_size=289205
    python $file_main --task_name MUSIC_INST \
        --max_epochs $(calc_max_epochs $dataset_size $1) \
        --earlystopping_patience $(calc_epochs $dataset_size 50000) \
        ${OPTIONS[@]} ${@:2}
}

function music_pitch () {
    dataset_size=289205
    python $file_main --task_name MUSIC_PITCH \
        --max_epochs $(calc_max_epochs $dataset_size $1) \
        --earlystopping_patience $(calc_epochs $dataset_size 20000) \
        ${OPTIONS[@]} ${@:2}
}


################################################################################
################################################################################
################################################################################
################################################################################

# If you want to resume the training from the checkpoint, increase the number just after `{task_name}` and run this script again.
# e.g.,
# 1. Execute       `acoustic_scenes 1 --version 0 --n_bins 40 --features power`
# 2. Then, execute `acoustic_scenes 2 --version 0 --n_bins 40 --features power`


# acoustic_scenes 1 --version 0 --n_bins 40 --features power
# acoustic_scenes 1 --version 0 --n_bins 40 --features power inst_freq
# acoustic_scenes 1 --version 0 --n_bins 40 --features power inst_freq --phase_feat_attn_power
# acoustic_scenes 1 --version 0 --n_bins 40 --features power grp_dly
# acoustic_scenes 1 --version 0 --n_bins 40 --features power grp_dly --phase_feat_attn_power
# acoustic_scenes 1 --version 0 --n_bins 40 --features power phase_phasor
# acoustic_scenes 1 --version 0 --n_bins 40 --features power phase_phasor --phase_feat_attn_power
# acoustic_scenes 1 --version 0 --n_bins 40 --features power inst_freq_rot
# acoustic_scenes 1 --version 0 --n_bins 40 --features power inst_freq_rot --phase_feat_attn_power
# acoustic_scenes 1 --version 0 --n_bins 40 --features power grp_dly_rot
# acoustic_scenes 1 --version 0 --n_bins 40 --features power grp_dly_rot --phase_feat_attn_power
# acoustic_scenes 1 --version 0 --n_bins 40 --features power phase_phasor_rot
# acoustic_scenes 1 --version 0 --n_bins 40 --features power phase_phasor_rot --phase_feat_attn_power


# bird_audio 1 --version 0 --n_bins 40 --features power
# bird_audio 1 --version 0 --n_bins 40 --features power inst_freq
# bird_audio 1 --version 0 --n_bins 40 --features power inst_freq --phase_feat_attn_power
# bird_audio 1 --version 0 --n_bins 40 --features power grp_dly
# bird_audio 1 --version 0 --n_bins 40 --features power grp_dly --phase_feat_attn_power
# bird_audio 1 --version 0 --n_bins 40 --features power phase_phasor
# bird_audio 1 --version 0 --n_bins 40 --features power phase_phasor --phase_feat_attn_power
# bird_audio 1 --version 0 --n_bins 40 --features power inst_freq_rot
# bird_audio 1 --version 0 --n_bins 40 --features power inst_freq_rot --phase_feat_attn_power
# bird_audio 1 --version 0 --n_bins 40 --features power grp_dly_rot
# bird_audio 1 --version 0 --n_bins 40 --features power grp_dly_rot --phase_feat_attn_power
# bird_audio 1 --version 0 --n_bins 40 --features power phase_phasor_rot
# bird_audio 1 --version 0 --n_bins 40 --features power phase_phasor_rot --phase_feat_attn_power


# emotion 1 --version 0 --n_bins 40 --features power
# emotion 1 --version 0 --n_bins 40 --features power inst_freq
# emotion 1 --version 0 --n_bins 40 --features power inst_freq --phase_feat_attn_power
# emotion 1 --version 0 --n_bins 40 --features power grp_dly
# emotion 1 --version 0 --n_bins 40 --features power grp_dly --phase_feat_attn_power
# emotion 1 --version 0 --n_bins 40 --features power phase_phasor
# emotion 1 --version 0 --n_bins 40 --features power phase_phasor --phase_feat_attn_power
# emotion 1 --version 0 --n_bins 40 --features power inst_freq_rot
# emotion 1 --version 0 --n_bins 40 --features power inst_freq_rot --phase_feat_attn_power
# emotion 1 --version 0 --n_bins 40 --features power grp_dly_rot
# emotion 1 --version 0 --n_bins 40 --features power grp_dly_rot --phase_feat_attn_power
# emotion 1 --version 0 --n_bins 40 --features power phase_phasor_rot
# emotion 1 --version 0 --n_bins 40 --features power phase_phasor_rot --phase_feat_attn_power


# speaker_vox 1 --version 0 --n_bins 40 --features power
# speaker_vox 1 --version 0 --n_bins 40 --features power inst_freq
# speaker_vox 1 --version 0 --n_bins 40 --features power inst_freq --phase_feat_attn_power
# speaker_vox 1 --version 0 --n_bins 40 --features power grp_dly
# speaker_vox 1 --version 0 --n_bins 40 --features power grp_dly --phase_feat_attn_power
# speaker_vox 1 --version 0 --n_bins 40 --features power phase_phasor
# speaker_vox 1 --version 0 --n_bins 40 --features power phase_phasor --phase_feat_attn_power
# speaker_vox 1 --version 0 --n_bins 40 --features power inst_freq_rot
# speaker_vox 1 --version 0 --n_bins 40 --features power inst_freq_rot --phase_feat_attn_power
# speaker_vox 1 --version 0 --n_bins 40 --features power grp_dly_rot
# speaker_vox 1 --version 0 --n_bins 40 --features power grp_dly_rot --phase_feat_attn_power
# speaker_vox 1 --version 0 --n_bins 40 --features power phase_phasor_rot
# speaker_vox 1 --version 0 --n_bins 40 --features power phase_phasor_rot --phase_feat_attn_power


# music_inst 1 --version 0 --n_bins 40 --features power
# music_inst 1 --version 0 --n_bins 40 --features power inst_freq
# music_inst 1 --version 0 --n_bins 40 --features power inst_freq --phase_feat_attn_power
# music_inst 1 --version 0 --n_bins 40 --features power grp_dly
# music_inst 1 --version 0 --n_bins 40 --features power grp_dly --phase_feat_attn_power
# music_inst 1 --version 0 --n_bins 40 --features power phase_phasor
# music_inst 1 --version 0 --n_bins 40 --features power phase_phasor --phase_feat_attn_power
# music_inst 1 --version 0 --n_bins 40 --features power inst_freq_rot
# music_inst 1 --version 0 --n_bins 40 --features power inst_freq_rot --phase_feat_attn_power
# music_inst 1 --version 0 --n_bins 40 --features power grp_dly_rot
# music_inst 1 --version 0 --n_bins 40 --features power grp_dly_rot --phase_feat_attn_power
# music_inst 1 --version 0 --n_bins 40 --features power phase_phasor_rot
# music_inst 1 --version 0 --n_bins 40 --features power phase_phasor_rot --phase_feat_attn_power


# music_pitch 1 --version 0 --n_bins 40 --features power
# music_pitch 1 --version 0 --n_bins 40 --features power inst_freq
# music_pitch 1 --version 0 --n_bins 40 --features power inst_freq --phase_feat_attn_power
# music_pitch 1 --version 0 --n_bins 40 --features power grp_dly
# music_pitch 1 --version 0 --n_bins 40 --features power grp_dly --phase_feat_attn_power
# music_pitch 1 --version 0 --n_bins 40 --features power phase_phasor
# music_pitch 1 --version 0 --n_bins 40 --features power phase_phasor --phase_feat_attn_power
# music_pitch 1 --version 0 --n_bins 40 --features power inst_freq_rot
# music_pitch 1 --version 0 --n_bins 40 --features power inst_freq_rot --phase_feat_attn_power
# music_pitch 1 --version 0 --n_bins 40 --features power grp_dly_rot
# music_pitch 1 --version 0 --n_bins 40 --features power grp_dly_rot --phase_feat_attn_power
# music_pitch 1 --version 0 --n_bins 40 --features power phase_phasor_rot
# music_pitch 1 --version 0 --n_bins 40 --features power phase_phasor_rot --phase_feat_attn_power


# speech_commands 1 --version 0 --n_bins 40 --features power
# speech_commands 1 --version 0 --n_bins 40 --features power inst_freq
# speech_commands 1 --version 0 --n_bins 40 --features power inst_freq --phase_feat_attn_power
# speech_commands 1 --version 0 --n_bins 40 --features power grp_dly
# speech_commands 1 --version 0 --n_bins 40 --features power grp_dly --phase_feat_attn_power
# speech_commands 1 --version 0 --n_bins 40 --features power phase_phasor
# speech_commands 1 --version 0 --n_bins 40 --features power phase_phasor --phase_feat_attn_power
# speech_commands 1 --version 0 --n_bins 40 --features power inst_freq_rot
# speech_commands 1 --version 0 --n_bins 40 --features power inst_freq_rot --phase_feat_attn_power
# speech_commands 1 --version 0 --n_bins 40 --features power grp_dly_rot
# speech_commands 1 --version 0 --n_bins 40 --features power grp_dly_rot --phase_feat_attn_power
# speech_commands 1 --version 0 --n_bins 40 --features power phase_phasor_rot
# speech_commands 1 --version 0 --n_bins 40 --features power phase_phasor_rot --phase_feat_attn_power


# language 1 --version 0 --n_bins 40 --features power
# language 1 --version 0 --n_bins 40 --features power inst_freq
# language 1 --version 0 --n_bins 40 --features power inst_freq --phase_feat_attn_power
# language 1 --version 0 --n_bins 40 --features power grp_dly
# language 1 --version 0 --n_bins 40 --features power grp_dly --phase_feat_attn_power
# language 1 --version 0 --n_bins 40 --features power phase_phasor
# language 1 --version 0 --n_bins 40 --features power phase_phasor --phase_feat_attn_power
# language 1 --version 0 --n_bins 40 --features power inst_freq_rot
# language 1 --version 0 --n_bins 40 --features power inst_freq_rot --phase_feat_attn_power
# language 1 --version 0 --n_bins 40 --features power grp_dly_rot
# language 1 --version 0 --n_bins 40 --features power grp_dly_rot --phase_feat_attn_power
# language 1 --version 0 --n_bins 40 --features power phase_phasor_rot
# language 1 --version 0 --n_bins 40 --features power phase_phasor_rot --phase_feat_attn_power
