# -*- coding: utf-8 -*-
"""
A module implementing evaluation metrics.

:Author: David Moses
:Copyright: Copyright (c) 2023, David Moses, All rights reserved.
"""

import re
import math
import string
import itertools
import collections
import numpy as np
import editdistance

def error_rate(ref, hyp, exclude=(), remove_punc=True, lower_case=True,
               return_distance=False):
    """
    Computes the error rate between two sequences using Levenshtein distance.

    This function can either be used to return the edit distance or the error
    rate (the latter is the edit distance divided by the length of the
    reference sequence).

    If the length of either the reference or hypothesis sequence is 0, this
    function will immediately return 1.0.

    Parameters
    ----------
    ref : list
        The reference sequence.
    hyp : list
        The hypothesis sequence.
    exclude : list or tuple
        All elements in the reference and hypothesis sequences that match
        the value of any element in this exclusion sequence will be
        excluded from the error rate calculation.
    remove_punc : bool
        Specifies whether or not to remove punctuation from the starts or
        ends of each token. Any punctuation within a token is preserved.
    lower_case : bool
        Specifies whether or not to enforce all tokens to be lowercase.
    return_distance : bool
        Specifies whether or not to return the edit distance (an integer)
        instead of the error rate.

    Returns
    -------
    float or int
        Either the error rate (as a float) or the edit distance (as an int).
    """

    # Immediately returns an error rate of 1.0 if either the reference or the
    # hypothesis sequence is empty
    if len(ref) == 0 or len(hyp) == 0:
        return 1.

    # Ensures that the provided sequences are lists of strings
    ref = [str(t) for t in ref]
    hyp = [str(t) for t in hyp]

    # If exclusion values have been provided, the reference and hypothesis
    # sequences are re-created without the exclusion elements
    if len(exclude) != 0:
        ref = [e for e in ref if e not in exclude]
        hyp = [e for e in hyp if e not in exclude]

    # Forces all tokens to be lowercase (if desired)
    if lower_case:
        ref = [e.lower() for e in ref]
        hyp = [e.lower() for e in hyp]

    # Removes punctuation from the starts and ends of each token (if desired)
    if remove_punc:
        punctuation_tuple = tuple(i for i in string.punctuation)
        def _removePunc(_seq):
            for i in range(len(_seq)):
                while _seq[i].startswith(punctuation_tuple):
                    _seq[i] = _seq[i][1:]
                while _seq[i].endswith(punctuation_tuple):
                    _seq[i] = _seq[i][:-1]
        _removePunc(ref)
        _removePunc(hyp)

    # Computes the edit distance between the two sequences
    distance = editdistance.eval(ref, hyp)

    # Returns either the edit distance or the error rate
    if return_distance:
        return distance
    else:
        return float(distance) / len(ref)

def error_rate_of_sequences(
        act_sequences, pred_sequences, alphabetic_only=True,
        use_average_token_len=False, delimiter=None,
        compute_character_error_rates=False, parcellation_num=None, **kwargs
):
    """
    Computes the token error rate between two sequences.

    Parameters
    ----------
    act_sequences : list
        A list of tokens or sequences representing the actual (reference)
        labels (as strings).
    pred_sequences : list
        A list of tokens or sequences representing the predicted labels (as
        strings).
    alphabetic_only : bool
        Specifies whether or not to format the sequences by removing all
        non-alphabetic characters from the tokens.
    use_average_token_len : bool
        If this is `True`, then the `return_distance` argument for calls to
        the `error_rate` will be set equal to `True` so that edit distances
        are computed. Afterwards, the edit distance associated with each
        sequence prediction will then be divided by the average number of
        tokens per sequence (across all sequences). This can be used to
        decrease the impact that shorter sequences have on the overall error
        rate. Note that if this argument is `True`, this function will return
        error rates (if `return_distance` is specified as a keyword argument
        to this function, it will not be used).
    delimiter : str or None
        The delimiter to use in the string `split` method when splitting each
        sequence from a string into a list. If this is `None`, the `split`
        method will be used with its default behavior. This is not used if
        `compute_character_error_rates` evaluates to `True`.
    compute_character_error_rates : bool
        Specifies whether or not to compute character error rates (instead of
        token error rates).
    parcellation_num : int or None
        If this is not `None`, then the sequences will be parcelled into
        subsets, each containing no more than `parcellation_num` sequences.
        Parcellation will be performed in order (e.g. the first
        `parcellation_num` sequences will be in the first subset, and so
        forth). Then, an error rate will be computed for each subset using
        recursive calls to this function (using relevant arguments to this
        function call). If `use_average_token_len` evaluates to `True`, the
        error rate used for each subset will be equal to the mean error rate
        across the sequences in that subset; otherwise, it will be equal to the
        overall error rate. After computing error rates for each subset, these
        error rates will be used to compute the overall error-rate mean and
        standard deviation.
    **kwargs
        Additional keyword arguments are accepted and passed to the
        `error_rate` function (during all calls to the function).

    Returns
    -------
    dict
        If `parcellation_num` is `None`, the returned dictionary will contain
        the following items:
        - "overall_error_rate" : float
            The mean token error rate between the two aggregate sequences.
            If `parcellation_num` is not `None`, this item will not be present
            in the returned dictionary.
        - "error_rates" : ndarray, float or int, (num_sequences,)
            The token error rate (or edit distance) between each sequence pair,
            as a 1D float (or int) array with length equal to the number of
            sequences. If there is a mismatch between the number of actual and
            predicted sequences, only the first `N` sequences are compared,
            where `N` is the smaller of the two lengths.
        - "mean_error_rate" : float
            The mean token error rate across all of the sequence pairs.
        - "stdev_error_rate" : float
            The standard deviation of the error rate across all of the sequence
            pairs.
        - "act_labels" : list
            The actual tokens (as a list of strings).
        - "pred_labels" : list
            The predicted tokens (as a list of strings).
        Otherwise, if `parcellation_num` is not `None`, the returned dictionary
        will contain the following items:
        - "error_rates" : ndarray, float or int, (num_sequences,)
            Error rates (or edit distances) across subsets.
        - "mean_error_rate" : float
            The mean error rate across subsets.
        - "stdev_error_rate" : float
            The standard deviation of the error rate across subsets.
        - "subset_results" : list
            A list in which each element corresponds to one of the subsets and
            is a dictionary of results returned by the recursive calls to this
            function for the corresponding subset.
    """

    # If a parcellation number was provided, separate steps are taken to
    # perform the parcellation and the analyses
    if parcellation_num is not None:

        # Performs the parcellation
        act_sequences_splits  = collections.defaultdict(list)
        pred_sequences_splits = collections.defaultdict(list)

        i = 0
        for cur_act_sequence, cur_pred_sequence in zip(
                act_sequences, pred_sequences
        ):
            if len(act_sequences_splits[i]) == parcellation_num:
                i += 1
            act_sequences_splits[i].append(cur_act_sequence)
            pred_sequences_splits[i].append(cur_pred_sequence)

        # Computes the results for each subset
        subset_results = [
            error_rate_of_sequences(
                act_sequences_splits[i], pred_sequences_splits[i],
                alphabetic_only=alphabetic_only,
                use_average_token_len=use_average_token_len,
                delimiter=delimiter,
                compute_character_error_rates=compute_character_error_rates,
                parcellation_num=None, **kwargs
            ) for i in sorted(act_sequences_splits.keys())
        ]

        # Computes the error rates per subset
        summary_key = (
            'mean_error_rate' if use_average_token_len
            else 'overall_error_rate'
        )
        error_rates = np.array([r[summary_key] for r in subset_results])

        # Returns the results
        return {
            'error_rates'      : error_rates,
            'mean_error_rate'  : np.mean(error_rates),
            'stdev_error_rate' : np.std(error_rates),
            'subset_results'   : subset_results
        }

    # Defines a local function to convert each sequence from a string into a
    # list
    def _split(_s):
        if compute_character_error_rates:
            return list(_s)
        elif delimiter is not None:
            return _s.split(delimiter)
        else:
            return _s.split()

    # Splits the strings in each sequence into token lists
    sequences = {
        k: [_split(s) for s in seq]
        for k, seq in zip(('act', 'pred'), (act_sequences, pred_sequences))
    }

    # Removes all non-letter characters from the tokens (if desired)
    if alphabetic_only:
        pattern = re.compile('[^a-zA-Z]')
        sequences = {
            key: [[pattern.sub('', w) for w in i] for i in cur_sequences]
            for key, cur_sequences in sequences.items()
        }

    # If re-scaling of the edit distances into error rates using the average
    # token length across all sequences is desired, the `return_distance`
    # keyword argument is forced to be `True`
    if use_average_token_len:
        kwargs['return_distance'] = True

    # Computes the error rate for each sequence pair
    error_rates = np.array([
        error_rate(ref=cur_act, hyp=cur_pred, **kwargs)
        for cur_act, cur_pred in zip(sequences['act'], sequences['pred'])
    ])

    # Splits the sequences into individual token lists
    labels = {
        key: list(itertools.chain(*cur_sequences))
        for key, cur_sequences in sequences.items()
    }

    # Computes the overall error rate
    overall_error_rate = error_rate(
        ref=labels['act'], hyp=labels['pred'], **kwargs
    )

    # If re-scaling of the edit distances into error rates using the average
    # token length across all sequences is desired, the computed distances
    # will then be re-scaled by the average token length
    if use_average_token_len:
        average_token_len = np.mean([len(s) for s in sequences['act']])
        error_rates = error_rates / average_token_len
        overall_error_rate /= (average_token_len * len(sequences['act']))

    # Returns the error rate between the two sequences and the labels
    return {
        'overall_error_rate' : overall_error_rate,
        'mean_error_rate'    : np.mean(error_rates),
        'stdev_error_rate'   : np.std(error_rates),
        'error_rates'        : error_rates,
        'act_labels'         : labels['act'],
        'pred_labels'        : labels['pred']
    }