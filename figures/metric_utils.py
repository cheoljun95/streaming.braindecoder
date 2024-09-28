import error_rate as metrics
import numpy as np
from tabulate import tabulate
import os
import tqdm
import matplotlib.pyplot as plt
from results import *

def remove_repeated_chars(word):
    """ Remove repeated characters (3+) in a word. """
    new_word = ""
    last_char = ""
    repeat_count = 0
    for char in word:
        if char == last_char:
            repeat_count += 1
        else:
            last_char = char
            repeat_count = 1
        if repeat_count <= 2:
            new_word += char
    return new_word

def calculate_error_rates_for_sequences(gt_sequences, pred_sequences, compute_character_error_rates, parcellation_num):
    """ Calculate error rates for sequences of labels."""
    return metrics.error_rate_of_sequences(
        act_sequences=gt_sequences,
        pred_sequences=pred_sequences,
        use_average_token_len=True,
        parcellation_num=parcellation_num,
        compute_character_error_rates=compute_character_error_rates
    )

def calculate_error_rates_and_summarize(comparison_sets, results, N=10, max_pred_words=9, results_summary={}, g2p_names=[]):
    """
    Calculate the summary of error rates and outliers for a set of experiments.

    Parameters:
    - comparison_sets: list, a list of tuples containing the ground truth and predicted labels to compare.
    - results: dict, experiment data.
    - N: int, number of trials per pseudoblock.
    - max_pred_words: int, maximum predicted words to consider.
    - d: int, number of decimal places to round the results to for visualization.
    - results_summary: dict, a dictionary to store the summary of error rates and outliers. Default is an empty dictionary.

    Returns:
    - A dictionary containing the summary of error rates and outliers.
    """
    
    cache_dir = "decoder_results/PERs"

    # Iterate over experiments
    for experiment in results:
        modality_results = results[experiment]
        for modality in modality_results:
            for gt_label, pred_label in comparison_sets:
                if gt_label not in modality_results[modality] or pred_label not in modality_results[modality]:
                    continue

                wers, cers, raw_wers, raw_cers, raw_pers, word_accuracy, ld_list, ins_list, dels_list, subs_list, len_diffs, pers = [], [], [], [], [], [], [], [], [], [], [], []
                ld_list_chars, ins_list_chars, dels_list_chars, subs_list_chars = [], [], [], []
                source_text, pred_text = [], []
                total_files = len(modality_results[modality][gt_label])
                outlier_count = 0
                comparison_key = f"{experiment}_{modality} {gt_label}_vs_{pred_label}"
                print(comparison_key)

                # Pre-compute the phonemes for the ground truth and predicted labels
                if f"{comparison_key}" in g2p_names:
                    gt_phonemes, pred_phonemes = [], []
                    cache_file_gt = os.path.join(cache_dir, f"{comparison_key}_gt.npz")
                    cache_file_pred = os.path.join(cache_dir, f"{comparison_key}_pred.npz")
                    gt_phonemes = np.load(cache_file_gt)
                    pred_phonemes = np.load(cache_file_pred)
                    gt_phonemes = list(gt_phonemes[gt_phonemes.files[0]])
                    pred_phonemes = list(pred_phonemes[pred_phonemes.files[0]])

                # Iterate over the files in blocks of N and get the error rates
                for i in range(0, total_files, N):
                    end_index = min(i + N, total_files)
                    block_file_ids = list(modality_results[modality][gt_label])[i:end_index]
                    gt_block = [str(modality_results[modality][gt_label][file_id]) for file_id in block_file_ids]
                    if f"{comparison_key}" in g2p_names:

                        gt_phoneme_block = [gt_phonemes[i] for i in range(i, end_index)]
                        pred_phoneme_block = [pred_phonemes[i] for i in range(i, end_index)]

                    pred_block = []
                    for file_id in block_file_ids:
                        decoded_text = str(modality_results[modality][pred_label][file_id])
                        decoded_text_filtered = ' '.join([remove_repeated_chars(word)  for word in decoded_text.split(' ')[:max_pred_words]])
                        if len(decoded_text_filtered) < len(decoded_text):  # Check if modification occurred
                            outlier_count += 1  # Increment if modification was made
                        pred_block.append(decoded_text_filtered)
                        pred_text.append(decoded_text_filtered)
                        source = str(modality_results[modality][gt_label][file_id])
                        source_text.append(source)
                    
                    # Calculate error rates for each block
                    wer_block = calculate_error_rates_for_sequences(gt_block, pred_block, False, N)['error_rates']
                    cer_block = calculate_error_rates_for_sequences(gt_block, pred_block, True, N)['error_rates']
                    wers.extend(wer_block)
                    cers.extend(cer_block)
                    if f"{comparison_key}" in g2p_names:
                        per_block = calculate_error_rates_for_sequences(gt_phoneme_block, pred_phoneme_block, False, N)['error_rates']
                        pers.extend(per_block)
                    
                    # Calculate raw WERs with N=1 for each file
                    for gt, pred in zip(gt_block, pred_block):
                        raw_wer = calculate_error_rates_for_sequences([gt], [pred], False, 1)['error_rates'][0]
                        raw_wers.append(raw_wer)
                        raw_cer = calculate_error_rates_for_sequences([gt], [pred], True, 1)['error_rates'][0]
                        raw_cers.append(raw_cer)
                        if f"{comparison_key}" in g2p_names:
                            raw_per = calculate_error_rates_for_sequences([gt_phonemes[i]], [pred_phonemes[i]], False, 1)['error_rates'][0]
                            raw_pers.append(raw_per)

                        # Calculate word accuracy as whether the predicted word is in the ground truth
                        gt_words = gt.split()
                        pred_words = pred.split()
                        accuracy_list = [1.0 if word in gt_words else 0.0 for word in pred_words]
                        word_accuracy += accuracy_list
                        len_diffs.append(np.abs(len(gt_words) - len(pred_words)))

                        # Calculate the LD inclduing insertions, deletions, and substitutions
                        ld, ins, dels, subs = levenshtein_distance_stats(gt, pred, mode='word')
                        ld_list.append(ld)
                        ins_list.append(ins)
                        dels_list.append(dels)
                        subs_list.append(subs)
                        ld, ins, dels, subs = levenshtein_distance_stats(gt, pred, mode='character')
                        ld_list_chars.append(ld)
                        ins_list_chars.append(ins)
                        dels_list_chars.append(dels)
                        subs_list_chars.append(subs)

                # Store the results in the summary dictionary
                results_summary[comparison_key] = {
                    'wers': wers,
                    'cers': cers,
                    'raw_wers': raw_wers,
                    'raw_cers': raw_cers,
                    'raw_pers': raw_pers,
                    'outlier_count': outlier_count,
                    'outlier_percentage': (outlier_count / total_files) * 100,
                    'word_accuracy': word_accuracy,
                    'ld_norm': ld_list,
                    'ins_norm': ins_list,
                    'dels_norm': dels_list,
                    'subs_norm': subs_list,
                    'len_diffs': len_diffs,
                    'ld_chars': ld_list_chars,
                    'ins_chars': ins_list_chars,
                    'dels_chars': dels_list_chars,
                    'subs_chars': subs_list_chars,
                    'source_text': source_text,
                    'pred_text': pred_text,
                    'pers': pers if f"{comparison_key}" in g2p_names else []
                }
    return results_summary

def calculate_speech_metrics_and_summarize(comparison_sets, results, sr=16000, detect_names=[], vad_threshold=0.1, results_summary={}, max_pred_words=9):
    """
    Calculate the summary of speech metrics for a set of experiments.

    Parameters:
    - comparison_sets: list, a list of tuples containing the ground truth and predicted labels to compare.
    - results: dict, experiment data.
    - d: int, number of decimal places to round the results to for visualization.
    - results_summary: dict, a dictionary to store the summary of speech metrics. Default is an empty dictionary.

    Returns:
    - A dictionary containing the summary of speech metrics
    """

    # Iterate over experiments
    cached_results_summary = {}
    results_summary = {} 
    for experiment in results:
        if 'ctc' in experiment:
            continue
        modality_results = results[experiment]
        for modality in modality_results:
            for gt_label, pred_label in comparison_sets:
                if pred_label not in modality_results[modality] or 'tts' in modality:
                    continue
                onsets, offsets, timecourses, text_onsets, text_offsets, text_timecourses, wpms = [], [], [], [], [], [], []
                comparison_key = f"{experiment}_{modality} {gt_label}_vs_{pred_label}"
                if 's24' in comparison_key or 'ablat' in comparison_key or 'chance' in comparison_key or "bictc" in comparison_key or "lm" in comparison_key or "rnnt" in comparison_key:
                    continue
                if comparison_key in detect_names:
                    detection_onsets, detection_offsets, detection_wpm, detection_text_onsets, detection_from_cue, detection_text_offsets = [], [], [], [], [], []
                    predicted_starts, predicted_ends = [], []
                print(comparison_key)

                # Compute the VAD metrics (onsets, offsets, timecourses). Adjust for go-cue timing
                if 'chance' not in comparison_key and 'ema' not in comparison_key and 'mea' not in comparison_key and 'emg' not in comparison_key:
                    for i, file_id in enumerate(modality_results[modality][pred_label]):
                        pred_wav = modality_results[modality][pred_label][file_id]
                        if 'tm1k' in comparison_key:
                            cue_time = 1.0
                        elif 'phrases' in comparison_key:
                            cue_time = 0.5
                        if 'realtime' in comparison_key:
                            # Obtain the cue time from the real-time blocks
                            event_time = modality_results[modality]['event_time'][file_id]
                            scheduled_event_time = modality_results[modality]['event_time_scheduler'][file_id]
                            cue_time = event_time - scheduled_event_time
                        if 'metzger' in comparison_key:
                            cue_time = 0.0
                        pred_wav = pred_wav[int(cue_time*sr):]
                        onsets_, offsets_, timecourse = vad_onset_offset(pred_wav, sr, threshold=vad_threshold)

                        # Where there are no onsets, offsets, or timecourses, set to maximum value
                        onsets_ = [len(pred_wav) / sr] if len(onsets_) == 0 else onsets_
                        offsets_ = [len(pred_wav) / sr] if len(offsets_) == 0 else offsets_
                        timecourse = np.zeros_like(pred_wav) if len(timecourse) == 0 else timecourse
                        onsets.append(onsets_)
                        offsets.append(offsets_)
                        timecourses.append(timecourse)

                        # detection onsets and offsets
                        if comparison_key in detect_names:
                            detected_attempt = modality_results[modality]['predicted_start'][file_id]
                            detected_attempt_end = modality_results[modality]['predicted_end'][file_id]
                            cue_block_time = modality_results[modality]['event_time'][file_id]
                            detection_offset_from_cue = detected_attempt - cue_block_time
                            detected_onsets_ = onsets_ - detection_offset_from_cue
                            detected_offsets_ = offsets_ - detection_offset_from_cue
                            detection_onsets.append(detected_onsets_)
                            detection_offsets.append(detected_offsets_)
                            detection_from_cue.append(detection_offset_from_cue)

                            predicted_starts.append(detected_attempt)
                            predicted_ends.append(detected_attempt_end)

                        # Compute words per minute
                        decoded_text_label = pred_label.replace('pred_audio', 'asr_transcript') if 'realtime' not in comparison_key else pred_label.replace('pred_audio', 'asr_transcript')
                        decoded_text = str(modality_results[modality][decoded_text_label][file_id])
                        decoded_text_filtered = ' '.join([remove_repeated_chars(word) for word in decoded_text.split(' ')[:max_pred_words]])
                        N = len(decoded_text_filtered.split(' '))
                        T = offsets_[-1]
                        if 'metzger' in comparison_key: # Here we should append the full length of the neural signal fed into the model + 150ms for the post-processing
                            if 'tm1k' in comparison_key:
                                neural_offset = 7.5 + 0.150
                            elif 'phrases' in comparison_key:
                                neural_offset = 4.62 + 0.150
                            T += neural_offset
                        WPM = N / (T / 60)
                        wpms.append(WPM)

                        # detection WPM
                        if comparison_key in detect_names:
                            T_detection = T - detection_offset_from_cue
                            WPM_detection = N / (T_detection / 60)
                            detection_wpm.append(WPM_detection)

                        # Get multimodal synchronization and text latency metrics
                        if 'streaming' in comparison_key:
                            if 'tm1k' in comparison_key:
                                cue_time = 1.0
                            elif 'phrases' in comparison_key:
                                cue_time = 0.5
                            realtime = True if 'realtime' in comparison_key else False
                            text_onset, text_offset, text_timecourse = text_onset_offset(modality_results[modality]['text_emissions_causal'][file_id], cue_time, realtime=realtime)
                            text_onsets.append(text_onset)
                            text_offsets.append(text_offset)
                            text_timecourses.append(text_timecourse)
                            if comparison_key in detect_names:
                                detection_text_onset = text_onset - detection_offset_from_cue
                                detection_text_onsets.append(detection_text_onset)
                                detection_text_offset = text_offset - detection_offset_from_cue
                                detection_text_offsets.append(detection_text_offset)

                # Store the results in the summary dictionary
                results_summary[comparison_key] = {
                    'onsets': onsets,
                    'offsets': offsets,
                    'timecourses': timecourses,
                }
                if 'streaming' in comparison_key:
                    results_summary[comparison_key]['text_onsets'] = text_onsets
                    results_summary[comparison_key]['text_timecourses'] = text_timecourses
                    results_summary[comparison_key]['text_offsets'] = text_timecourses
                if gt_label in modality_results[modality] and compute_mcd:
                    results_summary[comparison_key]['mcds'] = mcds
                if comparison_key in detect_names:
                    results_summary[comparison_key]['detection_onsets'] = detection_onsets
                    results_summary[comparison_key]['detection_offsets'] = detection_offsets
                    results_summary[comparison_key]['detection_wpm'] = detection_wpm
                    results_summary[comparison_key]['detection_text_onsets'] = detection_text_onsets
                    results_summary[comparison_key]['detection_text_offsets'] = detection_text_offsets
                    results_summary[comparison_key]['detection_from_cue'] = detection_from_cue
                    results_summary[comparison_key]['predicted_starts'] = predicted_starts
                    results_summary[comparison_key]['predicted_ends'] = predicted_ends
                results_summary[comparison_key]['wpms'] = wpms
    cached_results_summary.update(results_summary)
    return cached_results_summary


def vad_onset_offset(signal, fs, frame_size_ms=20, hop_size_ms=10, threshold=0.1):
    """
    Perform Voice Activity Detection (VAD) on a given audio signal and return the onsets and offsets of speech segments.
    
    Parameters:
    - signal: The audio signal array.
    - fs: The sampling frequency of the audio signal.
    - frame_size_ms: The frame size in milliseconds for energy calculation.
    - hop_size_ms: The hop size in milliseconds between consecutive frames.
    - threshold: The energy threshold for detecting speech.
    
    Returns:
    - onsets: The onset times of speech segments.
    - offsets: The offset times of speech segments.
    - time_course: A binary array indicating the speech segments.
    """
    frame_size = int(frame_size_ms * 0.001 * fs)
    hop_size = int(hop_size_ms * 0.001 * fs)
    
    # Calculate short-term energy
    energy = np.array([np.sum(signal[i:i+frame_size]**2) for i in range(0, len(signal) - frame_size + 1, hop_size)])
    if len(energy) == 0 or np.max(energy) == 0:
        return [], [], []
    energy_normalized = energy / np.max(energy)
    
    # Detect speech frames
    speech_frames = energy_normalized > threshold
    
    # Initialize lists to hold onsets and offsets
    onsets, offsets = [], []
    
    # Flag to indicate if we are in a speech segment
    in_speech = False
    for i, is_speech in enumerate(speech_frames):
        if is_speech and not in_speech:
            onsets.append(i * hop_size / fs)
            in_speech = True
        elif not is_speech and in_speech:
            offsets.append((i * hop_size + frame_size) / fs)
            in_speech = False
            
    # Check if the last segment is speech
    if in_speech:
        offsets.append(len(signal) / fs)
        
    # Generate a time_course array
    time_course = np.zeros_like(signal, dtype=int)
    for onset, offset in zip(onsets, offsets):
        start_sample = int(onset * fs)
        end_sample = int(offset * fs)
        time_course[start_sample:end_sample] = 1
    
    return onsets, offsets, time_course


def text_onset_offset(text_emissions, go_cue_time, buffer_size=16, neural_sr=200, realtime=False):
    """Compute the onset and offset times, and timecourse of text segments during text-decoding.

    Parameters:
    - text_emissions: list of text predictions at each sample.
    - go_cue_time: the time when the go cue was given.
    - buffer_size: size of the buffer used in decoding.
    - neural_sr: neural sampling rate.
    - realtime: flag indicating if data is real-time.

    Returns:
    - onset: time when the first non-empty text emission occurs.
    - offset: last time point at which the text predictions change.
    - timecourses: list of tuples with time points and corresponding text emissions.
    """
    try:
        # Initialize onset and offset
        onset = None
        offset = None

        # Compute onset time as the first non-empty text emission
        for sample_index, entry in enumerate(text_emissions):
            if not realtime:
                if entry.strip():
                    onset = sample_index * buffer_size / neural_sr - go_cue_time
                    break
            else:
                if entry[0].strip():
                    onset = int(entry[2]) / neural_sr - go_cue_time
                    break
        if onset is None:
            # If no non-empty emission, onset is at the end
            onset = len(text_emissions) * buffer_size / neural_sr - go_cue_time

        # Compute timecourses for every change in text emission
        timecourses = []
        last_text = None
        for sample_index, entry in enumerate(text_emissions):
            current_text = entry if not realtime else entry[0]
            if last_text is None or current_text != last_text:
                time_point = (
                    sample_index * buffer_size / neural_sr - go_cue_time
                    if not realtime
                    else int(entry[2]) / neural_sr - go_cue_time
                )
                timecourses.append((time_point, current_text))
                offset = time_point  # Update offset to the time of the current change
            last_text = current_text

        if offset is None:
            # If no changes, set offset to onset
            offset = onset

    except Exception as e:
        print("Error in text_onset_offset:", e)
        print("text_emissions:", text_emissions)
        print("go_cue_time:", go_cue_time)
        return None, None, None

    return onset, offset, timecourses


def levenshtein_distance_stats(s1, s2, mode='character'):
    """ Compute the Levenshtein distance between two strings and return the number of insertions, deletions, and substitutions. """
    # Split the strings into words or characters based on the mode
    if mode == 'word':
        s1, s2 = s1.split(), s2.split()

    rows = len(s1) + 1
    cols = len(s2) + 1
    distance = [[0 for _ in range(cols)] for _ in range(rows)]

    # Initialize the distance matrix
    for i in range(1, rows):
        distance[i][0] = i
    for j in range(1, cols):
        distance[0][j] = j

    for col in range(1, cols):
        for row in range(1, rows):
            if s1[row - 1] == s2[col - 1]:
                cost = 0
            else:
                cost = 1
            distance[row][col] = min(distance[row - 1][col] + 1,      # deletion
                                     distance[row][col - 1] + 1,      # insertion
                                     distance[row - 1][col - 1] + cost) # substitution

    # Backtrack to find I, D, S
    insertions, deletions, substitutions = 0, 0, 0
    row, col = rows - 1, cols - 1
    while row > 0 or col > 0:
        if row > 0 and col > 0 and distance[row][col] == distance[row-1][col-1] + cost:
            if cost:
                substitutions += 1
            row, col = row-1, col-1
        elif row > 0 and distance[row][col] == distance[row-1][col] + 1:
            deletions += 1
            row -= 1
        elif col > 0 and distance[row][col] == distance[row][col-1] + 1:
            insertions += 1
            col -= 1
        if row > 0 and col > 0:
            cost = 0 if s1[row-1] == s2[col-1] else 1

    return distance[-1][-1], insertions, deletions, substitutions

def get_bootstrapped_accuracies(trial_accuracies, num_repeats=None, median=True):
    """
    Computes accuracies using a bootstrapping method.
    """

    trial_accuracies = np.asarray(trial_accuracies)
    num_trials = len(trial_accuracies)

    if num_repeats is None:
        num_repeats = num_trials

    bootstrapped_accuracies = np.empty(shape=[num_repeats], dtype=float)

    for i in range(num_repeats):
        np.random.seed(i)
        cur_trial_samples = np.random.choice(trial_accuracies, size=num_trials, replace=True)
        if median:
            bootstrapped_accuracies[i] = np.median(cur_trial_samples)
        else:
            bootstrapped_accuracies[i] = np.mean(cur_trial_samples)

    return bootstrapped_accuracies