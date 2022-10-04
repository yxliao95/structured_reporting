import itertools
import json
import logging
import os
import subprocess
import sys
import re
import ast
import time
from tkinter import N
import numpy as np
from math import isnan
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Event

import hydra
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

# pylint: disable=import-error,wrong-import-order
from common_utils.common_utils import check_and_create_dirs, check_and_remove_dirs
from common_utils.coref_utils import ConllToken, auto_append_value_to_list, auto_extend_value_to_list
from common_utils.file_checker import FileChecker

logger = logging.getLogger()
module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")

FILE_CHECKER = FileChecker()
START_EVENT = Event()

# Utils


class ReportInfo:
    def __init__(self, doc_id, section_name) -> None:
        self.doc_id = doc_id
        self.section_name = section_name
        self.coref_mention_num = 0
        self.coref_group_num = 0
        # It would be like: {'muc': 0.0, 'bcub': 0.0, 'ceafe': 0.0}
        self.mention_recall_dict: dict[str, float] = {}
        self.mention_precision_dict: dict[str, float] = {}
        self.mention_f1_dict: dict[str, float] = {}

        self.coref_recall_dict: dict[str, float] = {}
        self.coref_precision_dict: dict[str, float] = {}
        self.coref_f1_dict: dict[str, float] = {}

        self.conll_f1_avg = 0
        self.conll_recall_avg = 0
        self.conll_precision_avg = 0

    def add_conll_f1score(self, metric_name: str, metric_value: float):
        """ It would be like: {'muc': 0.0, 'bcub': 0.0, 'ceafe': 0.0} """
        self.coref_f1_dict[metric_name] = metric_value

    def calculate_avg(self):
        self.conll_f1_avg = np.round(np.mean([i for _, i in self.coref_f1_dict.items()]), decimals=3)
        self.conll_recall_avg = np.round(np.mean([i for _, i in self.coref_recall_dict.items()]), decimals=3)
        self.conll_precision_avg = np.round(np.mean([i for _, i in self.coref_precision_dict.items()]), decimals=3)


def align_to_spacy(config, model_cfg, df_spacy, df_pred):

    spacy_col_cfg = config.name_style.spacy.column_name
    coref_col_cfg = model_cfg.target_column

    # Merge and align two df. Keep the index for later use
    df_spacy[config.df_col.spacy_index] = df_spacy.index
    df_pred[config.df_col.coref_index] = df_pred.index
    # Columns with NaNs are will be float dtype such as coref_group and coref_group_conll
    df_aligned = df_spacy.merge(df_pred, how="outer", left_index=True, right_on=coref_col_cfg.spacy_index).reset_index().drop(columns=["index"])

    # Token level alignment. Assuming spacy has token A B C, corenlp has token A' B' C'
    checkpoint_token_index = 0

    one2many_flag = False
    many2one_flag = False
    many2many_flag = False
    left_str = ""
    right_str = ""

    model2spacy_tok_indices: list[list[int]] = [-1] * len(df_spacy)

    coref_index_appearance_count_dict: dict[int, int] = {}  # {coref token index: number of appearances}

    for _idx, _itemSeries in df_aligned.iterrows():
        spacy_index = _itemSeries.get(config.df_col.spacy_index)
        spacy_index = int(spacy_index) if not isnan(spacy_index) else spacy_index
        coref_index = _itemSeries.get(config.df_col.coref_index)
        coref_index = int(coref_index) if not isnan(coref_index) else coref_index

        spacy_tokenStr = _itemSeries.get(spacy_col_cfg.token)
        spacy_tokenStr_isnan = False if isinstance(spacy_tokenStr, str) else isnan(spacy_tokenStr)
        spacy_tokenStr = str(spacy_tokenStr)
        coref_tokenStr = _itemSeries.get(coref_col_cfg.token)
        coref_tokenStr_isnan = False if isinstance(coref_tokenStr, str) else isnan(coref_tokenStr)
        coref_tokenStr = str(coref_tokenStr)

        prev_spacy_tokenStr = str(df_aligned.loc[checkpoint_token_index, spacy_col_cfg.token])
        prev_coref_tokenStr = df_aligned.loc[checkpoint_token_index, coref_col_cfg.token]
        prev_coref_tokenStr_isnan = False if isinstance(prev_coref_tokenStr, str) else isnan(prev_coref_tokenStr)
        prev_coref_tokenStr = str(prev_coref_tokenStr)

        prev_coref_index = df_aligned.loc[checkpoint_token_index, config.df_col.coref_index]
        prev_coref_index = int(prev_coref_index) if not isnan(prev_coref_index) else prev_coref_index
        try:
            # TokA = TokA'
            if spacy_tokenStr == coref_tokenStr:
                logger.debug("%s) Identical token: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                checkpoint_token_index = _idx
                # Token-level operation
                auto_append_value_to_list(model2spacy_tok_indices, spacy_index, coref_index)
                coref_index_appearance_count_dict[coref_index] = coref_index_appearance_count_dict.get(coref_index, 0) + 1
                # In case tokA != tokA', tokB = tokB'
                one2many_flag, many2one_flag, many2many_flag = False, False, False
                left_str, right_str = "", ""
            else:
                # One2many: TokA = TokA'+TokB'+TokC', TokA contains TokA'
                # Or, Many2many: TokA+TokC = TokA'+TokB', TokA contains TokA'. For example: df_aligned
                # .h    .       |   .h.s    .       |   .h      .       (one2many)
                # .h    h.s.    |   .h.s    h.s.    |   .h      h.      (many2many)
                # .s    NaN     |   .       NaN     |   .s      s.      (many2many)
                # .     NaN     |                   |   .       NaN     (many2many)
                if coref_tokenStr in spacy_tokenStr and not spacy_tokenStr_isnan and not coref_tokenStr_isnan:
                    if not one2many_flag:
                        logger.debug("%s) One2many token start: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                        auto_append_value_to_list(model2spacy_tok_indices, spacy_index, coref_index)
                        coref_index_appearance_count_dict[coref_index] = coref_index_appearance_count_dict.get(coref_index, 0) + 1
                        checkpoint_token_index = _idx
                        left_str += spacy_tokenStr
                        right_str += coref_tokenStr
                        one2many_flag = True
                    else:
                        logger.debug("%s) One2many token: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                        auto_append_value_to_list(model2spacy_tok_indices, spacy_index, coref_index)
                        coref_index_appearance_count_dict[coref_index] = coref_index_appearance_count_dict.get(coref_index, 0) + 1
                        right_str += coref_tokenStr

                # Many2one: TokA+TokB+TokC = TokA', TokA of TokA+TokB+TokC, TokA is subset_of TokA',
                # Or, Many2many: TokA+TokB = TokA'+TokC', TokA is subset_of TokA'
                # Or Many2one -> One2many
                # '     's  (Many2one)
                # s.    .   (One2many)
                elif spacy_tokenStr in coref_tokenStr and not spacy_tokenStr_isnan and not coref_tokenStr_isnan:
                    logger.debug("%s) Many2one token start: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                    auto_append_value_to_list(model2spacy_tok_indices, spacy_index, coref_index)
                    coref_index_appearance_count_dict[coref_index] = coref_index_appearance_count_dict.get(coref_index, 0) + 1
                    checkpoint_token_index = _idx
                    left_str += spacy_tokenStr
                    right_str += coref_tokenStr
                    many2one_flag = True

                # Many2one: TokB,TokC of TokA+TokB+TokC
                elif isnan(coref_index) and spacy_tokenStr in prev_coref_tokenStr and not many2many_flag:
                    logger.debug("%s) Many2one token: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                    auto_append_value_to_list(model2spacy_tok_indices, spacy_index, prev_coref_index)
                    coref_index_appearance_count_dict[prev_coref_index] = coref_index_appearance_count_dict.get(prev_coref_index, 0) + 1
                    left_str += spacy_tokenStr

                # TokA exist, TokA' not exist, not many2one, and not many2many
                # elif isnan(coref_index) and spacy_tokenStr not in prev_coref_tokenStr and not many2many_token_flag:
                elif isnan(coref_index) and spacy_tokenStr not in prev_coref_tokenStr and not many2many_flag:
                    logger.debug("%s) Empty coref token: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                    one2many_flag, many2one_flag, many2many_flag = False, False, False
                    left_str, right_str = "", ""

                # Many2many: rest of the tokens.
                # Three example of (spacy_tok  coref_tok) in row 2-4
                # .h    .       |   .h.s    .       |   .h      .
                # .h    h.s.    |   .h.s    h.s.    |   .h      h.      (many2many)
                # .s    NaN     |   .       NaN     |   .s      s.      (many2many)
                # .     NaN     |                   |   .       NaN     (many2many)
                else:
                    logger.debug("%s) Change to many2many token : %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                    many2many_flag = True

                    if one2many_flag:
                        # If currTok == prevTok, get the tok from checkpoint and update with curr row info
                        if spacy_tokenStr == prev_spacy_tokenStr:
                            auto_append_value_to_list(model2spacy_tok_indices, spacy_index, coref_index)
                            coref_index_appearance_count_dict[coref_index] = coref_index_appearance_count_dict.get(coref_index, 0) + 1
                            checkpoint_token_index = _idx
                            right_str += coref_tokenStr

                        elif spacy_tokenStr != prev_spacy_tokenStr:
                            # Just like a new token
                            if not isnan(coref_index):
                                auto_append_value_to_list(model2spacy_tok_indices, spacy_index, coref_index)
                                coref_index_appearance_count_dict[coref_index] = coref_index_appearance_count_dict.get(coref_index, 0) + 1
                                checkpoint_token_index = _idx
                                left_str += spacy_tokenStr
                                right_str += coref_tokenStr

                            # If is NaN, use the info from checkpoint row (rather than checkpoint tok, because checkpoint tok might have other rows' info)
                            else:
                                auto_append_value_to_list(model2spacy_tok_indices, spacy_index, prev_coref_index)
                                coref_index_appearance_count_dict[prev_coref_index] = coref_index_appearance_count_dict.get(prev_coref_index, 0) + 1
                                left_str += spacy_tokenStr

                    # .     .h      (many2one)
                    # h.s.  .s.     (would go to one2many, not here)
                    elif many2one_flag:
                        auto_append_value_to_list(model2spacy_tok_indices, spacy_index, coref_index)
                        coref_index_appearance_count_dict[coref_index] = coref_index_appearance_count_dict.get(coref_index, 0) + 1
                        checkpoint_token_index = _idx
                        left_str += spacy_tokenStr
                        right_str += coref_tokenStr

                    # prevTok = prevTok', TokA != TokA' (neither contains nor subset_of).
                    # Just simply create a spacy tokenObj
                    else:
                        logger.debug("%s) Unequal token: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                        checkpoint_token_index = _idx
                        auto_append_value_to_list(model2spacy_tok_indices, spacy_index, coref_index)
                        coref_index_appearance_count_dict[coref_index] = coref_index_appearance_count_dict.get(coref_index, 0) + 1
                        one2many_flag, many2one_flag, many2many_flag = False, False, False
                        left_str, right_str = "", ""

                if True in [one2many_flag, many2one_flag, many2many_flag] and left_str.replace(" ", "") == right_str.replace(" ", ""):
                    logger.debug("%s) Multi-alignment end", _idx)
                    one2many_flag, many2one_flag, many2many_flag = False, False, False
                    left_str, right_str = "", ""

        except Exception:
            logger.error(traceback.format_exc())
            raise

    # model2spacy_tok_indices might be: [[1], [2], [3, 4, 5], -1, [6], [6], [6], -1, [7, 8], [8], [8]]
    # which is [eq, eq, one2many, NaN, many2one, many2one, many2one, NaN, many2many, many2many, many2many]
    return model2spacy_tok_indices, coref_index_appearance_count_dict


def align_spacy_to_ground_truth(gt_token_list, spacy_toekn_list, df_spacy=None, spacy_cfg=None, df_pred=None, model_cfg=None) -> list[list[int]]:
    """Return:
        spacy2gt_tok_indices: A list of tokens in which each element is a list of indices indicating the corresponding 
            indices in spacy.
        empty_token_with_conll_label_dict: In very few case, when spacy_token is `\n`, its conll_label could exist. 
            But this token would be discard, thus we need to move its conll label forward or backward to valid gt token.
    """
    empty_token_idx_with_conll_label_dict: dict[int, str] = {}  # e.g. {1110: "(252"} means that token at `1110` has conll label `(252`

    curr_gt_token_pointer = 0
    curr_spacy_token_pointer = 0

    # Align spacy tokens to ground-truth tokens
    spacy2gt_tok_indices: list[list[int]] = [-1] * len(gt_token_list)
    default_nan_list = ["#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
                        "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null"]

    right_str = ""
    while curr_gt_token_pointer < len(gt_token_list) and curr_spacy_token_pointer < len(spacy_toekn_list):
        curr_gt_token = str(gt_token_list[curr_gt_token_pointer])
        curr_spacy_token = str(spacy_toekn_list[curr_spacy_token_pointer])

        if curr_spacy_token.strip() == "" and curr_gt_token.strip() == "":
            logger.debug("%s,%s) curr_gt_token, curr_spacy_token are empty: %s | %s", curr_gt_token_pointer, curr_spacy_token_pointer, curr_gt_token, curr_spacy_token)
            auto_append_value_to_list(spacy2gt_tok_indices, curr_gt_token_pointer, curr_spacy_token_pointer)
            curr_spacy_token_pointer += 1
            curr_gt_token_pointer += 1
            continue
        else:
            if curr_spacy_token.strip() == "":
                logger.debug("%s,%s) curr_spacy_token is empty in csv: %s", curr_gt_token_pointer, curr_spacy_token_pointer, curr_spacy_token.encode())
                if df_pred is not None:
                    # In very few case, when spacy_token is \n, its conll_label could exist.
                    # To fix this, we should find out its corresponidng label, than track that token index.
                    conll_label_str = str(df_pred.loc[curr_spacy_token_pointer, model_cfg.target_column.coref_group_conll])
                    conll_label_list = ast.literal_eval(conll_label_str) if conll_label_str != "-1" else []
                    for conll_label in conll_label_list:
                        empty_token_idx_with_conll_label_dict[curr_spacy_token_pointer] = conll_label
                curr_spacy_token_pointer += 1
                continue

            if curr_gt_token.strip() in ["", "nan"]:
                logger.debug("%s,%s) curr_gt_token is empty: %s", curr_gt_token_pointer, curr_spacy_token_pointer, curr_gt_token.encode())
                curr_gt_token_pointer += 1
                continue

        if curr_gt_token == curr_spacy_token:
            logger.debug("%s,%s) curr_gt_token, curr_spacy_token: %s | %s", curr_gt_token_pointer, curr_spacy_token_pointer, curr_gt_token, curr_spacy_token)
            auto_append_value_to_list(spacy2gt_tok_indices, curr_gt_token_pointer, curr_spacy_token_pointer)
            curr_gt_token_pointer += 1
            curr_spacy_token_pointer += 1
        else:
            if curr_spacy_token in curr_gt_token:
                logger.debug("%s,%s) curr_gt_token contains curr_spacy_token: %s | %s", curr_gt_token_pointer, curr_spacy_token_pointer, curr_gt_token, curr_spacy_token)
                auto_append_value_to_list(spacy2gt_tok_indices, curr_gt_token_pointer, curr_spacy_token_pointer)
                right_str += curr_spacy_token
            else:
                # In most of the case, this is because the `curr_spacy_token` is nan in fast_coref_joint output, but the token in spacy model output is correct.
                if df_spacy is not None:
                    curr_spacy_token = str(df_spacy.loc[curr_spacy_token_pointer, spacy_cfg.target_column.token])
                    if curr_gt_token == curr_spacy_token:
                        logger.debug("%s,%s) [Using spacy token] curr_gt_token, curr_spacy_token: %s | %s", curr_gt_token_pointer, curr_spacy_token_pointer, curr_gt_token, curr_spacy_token)
                        auto_append_value_to_list(spacy2gt_tok_indices, curr_gt_token_pointer, curr_spacy_token_pointer)
                        right_str += curr_spacy_token
                    elif curr_spacy_token in curr_gt_token:
                        logger.debug("%s,%s) [Using spacy token] curr_gt_token contains curr_spacy_token: %s | %s", curr_gt_token_pointer, curr_spacy_token_pointer, curr_gt_token, curr_spacy_token)
                        auto_append_value_to_list(spacy2gt_tok_indices, curr_gt_token_pointer, curr_spacy_token_pointer)
                        right_str += curr_spacy_token
                    else:
                        raise Exception(
                            f"[Using spacy token] Error occured as curr_spacy_token: {curr_spacy_token.encode()} NOT in curr_gt_token: {curr_gt_token.encode()}. Details: gt_index={curr_gt_token_pointer}, spacy_index={curr_spacy_token_pointer}")
                else:
                    raise Exception(
                        f"Error occured as curr_spacy_token: {curr_spacy_token.encode()} NOT in curr_gt_token: {curr_gt_token.encode()}. Details: gt_index={curr_gt_token_pointer}, spacy_index={curr_spacy_token_pointer}")

            if curr_gt_token == right_str:
                logger.debug("%s,%s) curr_gt_token, right_str: %s | %s", curr_gt_token_pointer, curr_spacy_token_pointer, curr_gt_token, right_str)
                curr_spacy_token_pointer += 1
                curr_gt_token_pointer += 1
                right_str = ""
            else:
                curr_spacy_token_pointer += 1

    assert spacy_toekn_list[curr_spacy_token_pointer].strip() == ""
    assert len(spacy2gt_tok_indices) == len(gt_token_list)
    return spacy2gt_tok_indices, empty_token_idx_with_conll_label_dict


def convert_non_spacy_token_csv_to_conll_format(config, model_cfg, section_name, doc_id, coref_index_appearance_count_dict: dict[int, int], model2spacy_tok_indices: list[list[int]], spacy2gt_tok_indices: list[list[int]], df_pred: pd.DataFrame, df_spacy: pd.DataFrame) -> list[list[ConllToken]]:
    """Args:
    model2spacy_tok_indices: The index of this list is the index of spacy token, the value of each item is the corresponding index of coref (model) token.
    spacy2gt_tok_indices: The index of this list is the index of ground-truth token, the value of each item is the corresponding index of spacy (model) token.
    coref_index_appearance_count_dict: The key is the index of coref (model) token, the value is the number of appearance of this index in model2spacy_tok_indices
    Return:
        list[list[ConllToken]]: The first dimension is sentences. The second dimension is tokens.
    """
    sentence_list: list[list[ConllToken]] = []
    token_list: list[ConllToken] = []
    sentence_id = 0
    model_col_cfg = model_cfg.target_column
    spacy_col_cfg = config.input.spacy.target_column
    prev_sentence_id = 0

    count_coref_index_appear_at_gt_dict: dict[int, int] = {}
    for spacy_indices in spacy2gt_tok_indices:
        sentence_id = int(df_spacy.loc[spacy_indices[0], spacy_col_cfg.sentence_group])
        conllToken = ConllToken(doc_id, sentence_id, spacy_indices[0], "")
        conll_labels_from_single_or_multi_row = []

        # In some cases, multiple spacy tokens are pointing to the same coref_model tokens.
        # We need to modify the conll labels properly, otherwise the conll scoring script will crash
        # Such as (1 should not be followed by another (1 on next line.
        # But we also consider that the above multiple spacy tokens are pointed by one gt token,
        # which means that we should keep only one conll label row from multiple same coref_model tokens
        # and dont modify it.
        model2gt_curr_tok_indices = []
        coref_index_first_time_appear_at_gt_dict: dict[int, bool] = {}
        for spacy_index in spacy_indices:
            if model2spacy_tok_indices[spacy_index] == -1:
                conllToken.tokenStr += ""
                continue
            else:
                conllToken.tokenStr += df_spacy.loc[spacy_index, spacy_col_cfg.token]
                model2gt_curr_tok_indices.extend(model2spacy_tok_indices[spacy_index])
                for model_index in model2spacy_tok_indices[spacy_index]:
                    at_gt_appearance_count = count_coref_index_appear_at_gt_dict.get(model_index, 0)
                    count_coref_index_appear_at_gt_dict[model_index] = at_gt_appearance_count + 1
                    # First time appear
                    if at_gt_appearance_count == 0:
                        coref_index_first_time_appear_at_gt_dict[model_index] = True

        for model_index in set(model2gt_curr_tok_indices):
            conll_label_str = str(df_pred.loc[model_index, model_col_cfg.coref_group_conll])
            conll_label_list = ast.literal_eval(conll_label_str)
            first_time_appear = coref_index_first_time_appear_at_gt_dict.get(model_index, False)

            if conll_label_str != "-1":
                if first_time_appear:
                    # Meet max appearance times at the first time it appear, then keep the conll label
                    if count_coref_index_appear_at_gt_dict[model_index] == coref_index_appearance_count_dict[model_index]:
                        conll_labels_from_single_or_multi_row.extend(conll_label_list)
                    # Otherwise adapt it to multiple line
                    else:
                        adapted_conll_list = adapt_oneline_conll_list_to_multiline(conll_label_list, "multi_line_begin")
                        conll_labels_from_single_or_multi_row.extend(adapted_conll_list)
                else:
                    # Meet max appearance times
                    if count_coref_index_appear_at_gt_dict[model_index] == coref_index_appearance_count_dict[model_index]:
                        adapted_conll_list = adapt_oneline_conll_list_to_multiline(conll_label_list, "multi_line_end")
                        conll_labels_from_single_or_multi_row.extend(adapted_conll_list)

        # In some special cases, even single (non-aggregrated) token (clinical-263, No.443 tok) will have conll labels like `(0|(0)`
        # which is caused by the voing algorithm and is correct in programming (but it is wrong as a prediction result).
        # We want to consider this as normal and do not remove any labels.
        gt_map_coref_tokens = []
        for spacy_index in spacy_indices:
            if model2spacy_tok_indices[spacy_index] != -1:
                gt_map_coref_tokens.extend(model2spacy_tok_indices[spacy_index])
        if len(gt_map_coref_tokens) > 1:
            aggregrated_conll_labels_for_one_row: list[str] = remove_duplicated_conll_label2(conll_labels_from_single_or_multi_row)
            conllToken.add_coref_label("|".join(aggregrated_conll_labels_for_one_row))
        else:
            conllToken.add_coref_label("|".join(conll_labels_from_single_or_multi_row))

        if sentence_id == prev_sentence_id:
            token_list.append(conllToken)
        else:
            sentence_list.append(token_list)
            token_list: list[ConllToken] = []
            token_list.append(conllToken)
            prev_sentence_id = sentence_id
    sentence_list.append(token_list)

    return sentence_list


def adapt_oneline_conll_list_to_multiline(conll_label_list: list[str], begin_or_end: str = None) -> list[str]:
    """ For example: Three spacy tokens are pointing to the same tokens of coref model. 
    TokA: (1) | (2 | 3) ->  (1 | (2
    TokB: (1) | (2 | 3) ->
    TokC: (1) | (2 | 3) ->  1) | 3)
    """
    new_conll_label_list: list[str] = []
    if begin_or_end == "multi_line_begin":
        for conll_label in conll_label_list:
            if "(" in conll_label and ")" in conll_label:
                new_conll_label_list.append(conll_label.replace(")", ""))
            elif "(" in conll_label:
                new_conll_label_list.append(conll_label)
    elif begin_or_end == "multi_line_end":
        for conll_label in conll_label_list:
            if "(" in conll_label and ")" in conll_label:
                new_conll_label_list.append(conll_label.replace("(", ""))
            elif ")" in conll_label:
                new_conll_label_list.append(conll_label)
    else:
        raise Exception("Should specify begin or end.")
    return new_conll_label_list


def merge_conll_label(conll_label_list: list, extra_label: str) -> list[str]:
    """ Notice that extra_label do not have closed label like (1) """
    if extra_label in conll_label_list:
        return conll_label_list

    new_label_list = []
    extra_label_val = extra_label.replace("(", "").replace(")", "")
    matched_and_merged = False
    for conll_label in conll_label_list:
        conll_label_val = conll_label.replace("(", "").replace(")", "")
        if conll_label_val == extra_label_val:
            new_label_list.append(f"({conll_label_val})")
        else:
            new_label_list.append(conll_label)
    if matched_and_merged:
        new_label_list.append(extra_label)
    return new_label_list


def convert_spacy_token_csv_to_conll_format(config, model_cfg, section_name, doc_id, spacy2gt_tok_indices: list[list[int]], df_voted: pd.DataFrame, target_token_index_and_conll_label_dict) -> list[list[ConllToken]]:
    """
    Return:
        list[list[ConllToken]]: The first dimension is sentences. The second dimension is tokens.
    """
    sentence_list: list[list[ConllToken]] = []
    token_list: list[ConllToken] = []
    sentence_id = 0
    column_cfg = model_cfg.target_column
    prev_sentence_id = 0
    for indices in spacy2gt_tok_indices:
        sentence_id = int(df_voted.loc[indices[0], column_cfg.sentence_group])
        conllToken = ConllToken(doc_id, sentence_id, indices[0], "")
        conll_labels_from_single_or_multi_row = []
        for index in indices:
            conllToken.tokenStr += df_voted.loc[index, column_cfg.token]
            conll_label_str = str(df_voted.loc[index, column_cfg.coref_group_conll])

            # Special case when this token should have extra conll label came from the skipped spacy token
            extra_conll_label = ""
            if index in target_token_index_and_conll_label_dict:
                extra_conll_label = target_token_index_and_conll_label_dict[index]

            if conll_label_str != "-1" and extra_conll_label != "":
                conll_labels_from_single_or_multi_row.extend(merge_conll_label(ast.literal_eval(conll_label_str), extra_conll_label))
            elif conll_label_str == "-1" and extra_conll_label != "":
                conll_labels_from_single_or_multi_row.extend(extra_conll_label)
            elif conll_label_str != "-1" and extra_conll_label == "":
                conll_labels_from_single_or_multi_row.extend(ast.literal_eval(conll_label_str))

        # In some special case, even single (non-aggregrated) token (clinical-263, No.443 tok) will have conll labels like `(0|(0)`
        # which is caused by the voing algorithm and is correct in programming (but it is wrong as a prediction result).
        # We want to consider this as normal and do not remove any labels.
        if len(indices) > 1:
            aggregrated_conll_labels_for_one_row: list[str] = remove_duplicated_conll_label2(conll_labels_from_single_or_multi_row)
            conllToken.add_coref_label("|".join(aggregrated_conll_labels_for_one_row))
        else:
            conllToken.add_coref_label("|".join(conll_labels_from_single_or_multi_row))

        if sentence_id == prev_sentence_id:
            token_list.append(conllToken)
        else:
            sentence_list.append(token_list)
            token_list: list[ConllToken] = []
            token_list.append(conllToken)
            prev_sentence_id = sentence_id
    sentence_list.append(token_list)

    return sentence_list


def remove_duplicated_conll_label2(input_label_list: list[str]) -> list[str]:
    """ Arg: A list of conll labels from multiple token rows.

    Return:
        ["(1", "1)"] will become ["(1)"],
        ["(1)", "(1)"] will become ["(1)"],
        ["1)", "(1)"] will remain,
        ["(1", "(1", "1)"] will become ["(1", "(1)"]
        others will remain the same
    """
    input_label_list_edit = input_label_list.copy()
    aggregrated_label_list = []

    for curr_index, curr_label in enumerate(input_label_list):
        if input_label_list_edit[curr_index] == -1:
            continue
        label_value = curr_label.replace("(", "").replace(")", "")

        if "(" in curr_label and ")" not in curr_label:
            target_label = f"{label_value})"
        elif "(" not in curr_label and ")" in curr_label:
            target_label = f"({label_value}"
        elif "(" in curr_label and ")" in curr_label:
            target_label = f"({label_value})"

        try:
            target_index = input_label_list_edit.index(target_label)
            input_label_list_edit[target_index] = -1
            input_label_list_edit[curr_index] = -1
            aggregrated_label_list.append(f"({label_value})")

        except ValueError:
            aggregrated_label_list.append(curr_label)
            input_label_list_edit[curr_index] = -1
            continue

    aggregrated_label_list.extend(list(itertools.filterfalse(lambda x: x == -1, input_label_list_edit)))
    return aggregrated_label_list


# Deprecated. Will cause error on model: rb, fj
def remove_duplicated_conll_label(label_list: list[str]) -> list[str]:
    """ Arg: A list of conll labels from multiple token rows.

    Return:
        ["(1", "1)"] will become ["(1)"],
        ["(1)", "(1)"] will become ["(1)"],
        ["1)", "(1)"] will remain,
        ["(1", "(1", "1)")] will become ["(1", "(1)"]
        others will remain the same
    """
    numberStr_list = [i.replace("(", "").replace(")", "") for i in label_list]
    numberStr_set = set(numberStr_list)

    added_numberStr_set = set()
    new_label_list = []
    if len(numberStr_list) != len(set(numberStr_set)):
        for id_str in numberStr_set:
            left_trigger, right_trigger, both_trigger = False, False, False
            id_index = 0
            if numberStr_list.count(id_str) > 1:
                while True:
                    try:
                        id_index = numberStr_list.index(id_str, id_index)
                    except ValueError:
                        break
                    id_label = label_list[id_index]
                    if "(" in id_label and ")" not in id_label:
                        left_trigger = True
                    if "(" not in id_label and ")" in id_label:
                        right_trigger = True
                    if "(" in id_label and ")" in id_label:
                        both_trigger = True
                    id_index += 1
            if left_trigger and right_trigger:
                new_label_list.append(f"({id_str})")
                added_numberStr_set.add(id_str)
            elif not left_trigger and not right_trigger and both_trigger:
                new_label_list.append(f"({id_str})")
                added_numberStr_set.add(id_str)

    # Find the rest of the labels that are not duplicated.
    new_label_list.extend([labelStr for labelStr in label_list for numberStr in numberStr_set.difference(added_numberStr_set) if int(numberStr) == int(labelStr.replace("(", "").replace(")", ""))])

    return new_label_list


def convert_gt_csv_to_conll_format(config, doc_id, df_gt: pd.DataFrame) -> list[list[ConllToken]]:
    """
    Return:
        list[list[ConllToken]]: The first dimension is sentences. The second dimension is tokens.
    """
    column_cfg = config.input.ground_truth.target_column

    sentence_list: list[list[ConllToken]] = []
    token_list: list[ConllToken] = []
    sentence_list.append(token_list)

    sentence_id = 0
    prev_sentence_id = 0
    for _index, _data_series in df_gt.iterrows():
        sentence_id = int(_data_series.get(column_cfg.sentence_group))
        conllToken = ConllToken(doc_id, sentence_id, _index, str(_data_series.get(column_cfg.token)))
        conllToken.add_coref_label(str(_data_series.get(column_cfg.coref_group_conll)))

        if sentence_id == prev_sentence_id:
            token_list.append(conllToken)
        else:
            token_list: list[ConllToken] = []
            token_list.append(conllToken)
            sentence_list.append(token_list)
            prev_sentence_id = sentence_id

    return sentence_list


def write_conll_file(temp_output_dir, doc_id, sentence_list: list[list[ConllToken]]) -> str:
    BEGIN = f"#begin document ({doc_id}); part 0\n"
    SENTENCE_SEPARATOR = "\n"
    END = "#end document\n"

    out_file_path = os.path.join(temp_output_dir, f"{doc_id}.conll")

    with open(out_file_path, "w", encoding="UTF-8") as out:
        out.write(BEGIN)
        for sent in sentence_list:
            # Skip empty sentence
            if len(sent) == 1 and sent[0].tokenStr == "":
                continue
            for tok in sent:
                out.write(tok.get_conll_str() + "\n")
            out.write(SENTENCE_SEPARATOR)
        out.write(END)

    return out_file_path


def invoke_conll_script(scorer_path: str, use_which_metric: str, groundtruth_file_path: str, predicted_file_path: str) -> tuple[str, str]:
    """Args:
        scorer_path: The path of the CoNLL scorer script: scorer.pl
        use_which_metric: muc, bclub, ceafe
        groundtruth_file_path: The path of the file serve as a ground truth file
        predicted_file_path: The path of the file serve as a predicted output

    Returns:
        out: The standard output of the script.
        err: The error message if the script is failed. Empty if no error.
    """
    command = [scorer_path, use_which_metric, groundtruth_file_path, predicted_file_path, "none"]

    result = subprocess.run(command, capture_output=True, check=True)
    out = result.stdout.decode('utf-8')
    err = result.stderr.decode('utf-8')
    if err:
        err += f" Error command: {command}"
    return out, err


def resolve_conll_script_output(output_str) -> tuple[float, float, float, float, float, float]:
    """Args:
        output_str: The output of the CoNLL scorer script: scorer.pl. It only support single metric output, i.e. muc, bcub, ceafe, ceafm
    Returns:
        The percentage float value extracted from the script output. The ``%`` symble is omitted.
    """
    regexPattern = r"(\d*\.?\d*)%"
    scores = [float(i) for i in re.findall(regexPattern, output_str)]
    mention_recall = scores[0]
    mention_precision = scores[1]
    mention_f1 = scores[2]
    coref_recall = scores[3]
    coref_precision = scores[4]
    coref_f1 = scores[5]
    return mention_recall, mention_precision, mention_f1, coref_recall, coref_precision, coref_f1


def find_cloest_index(spacy2gt_tok_indices, empty_token_idx_with_conll_label_dict) -> dict[int, str]:
    """ A closed conll label `(1)` will be dropped. Only unclosed label like `1)` is remained. """
    flatten_indices = [spacy_index for spacy_indices in spacy2gt_tok_indices for spacy_index in spacy_indices]
    flatten_indices = sorted(flatten_indices)

    target_token_index_and_conll_label_dict: dict[int, str] = {}
    for tok_idx, label in empty_token_idx_with_conll_label_dict.items():
        if "(" in label and ")" not in label:
            # Search forward
            while tok_idx < flatten_indices[-1]:
                if tok_idx in flatten_indices:
                    target_token_index_and_conll_label_dict[tok_idx] = label
                    break
                tok_idx += 1
        if "(" not in label and ")" in label:
            # Search backward
            while tok_idx >= 0:
                if tok_idx in flatten_indices:
                    target_token_index_and_conll_label_dict[tok_idx] = label
                    break
                tok_idx -= 1
    return target_token_index_and_conll_label_dict

#######


def batch_processing(config, model_cfg, section_name, file_name, input_dir, reportInfo: ReportInfo) -> ReportInfo:
    """ Compute the CoNLL coreference scores """
    START_EVENT.wait()
    # if file_name != "clinical-373.csv":
    #     return None

    gt_cfg = config.input.ground_truth
    scorer_cfg = config.scorer
    spacy_cfg = config.input.spacy

    try:
        doc_id = file_name.replace(".csv", "")

        input_file_path = os.path.join(input_dir, file_name)
        df_pred = pd.read_csv(input_file_path, index_col=0, na_filter=False)
        gt_file_path = os.path.join(gt_cfg.csv_dir, file_name)
        df_gt = pd.read_csv(gt_file_path, index_col=0, na_filter=False)
        spacy_file_path = os.path.join(spacy_cfg.csv_dir, section_name, file_name)
        df_spacy = pd.read_csv(spacy_file_path, index_col=0, na_filter=False)

        # Some of the i2b2 raw files are utf-8 start with DOM, but we didn't remove the DOM character, thus we fix it here.
        df_gt.iloc[0] = df_gt.iloc[0].apply(lambda x: x.replace("\ufeff", "").replace("\xef\xbb\xbf", "") if isinstance(x, str) else x)
        df_pred.iloc[0] = df_pred.iloc[0].apply(lambda x: x.replace("\ufeff", "").replace("\xef\xbb\xbf", "") if isinstance(x, str) else x)
        df_spacy.iloc[0] = df_spacy.iloc[0].apply(lambda x: x.replace("\ufeff", "").replace("\xef\xbb\xbf", "") if isinstance(x, str) else x)

        # Generate conll format predicted files
        if model_cfg.align_to_spacy:

            # Algin to spacy first, then align spacy to gt
            model2spacy_tok_indices, coref_index_appearance_count_dict = align_to_spacy(config, model_cfg, df_spacy, df_pred)

            gt_token_list = df_gt.loc[:, gt_cfg.target_column.token_for_alignment].tolist()
            spacy_token_list = df_spacy.loc[:, spacy_cfg.target_column.token].tolist()
            spacy2gt_tok_indices, _ = align_spacy_to_ground_truth(gt_token_list, spacy_token_list)
            sentence_list_pred = convert_non_spacy_token_csv_to_conll_format(config, model_cfg, section_name, doc_id, coref_index_appearance_count_dict, model2spacy_tok_indices,
                                                                             spacy2gt_tok_indices, df_pred, df_spacy)
        else:
            # Directly align to ground-truth
            gt_token_list = df_gt.loc[:, gt_cfg.target_column.token_for_alignment].tolist()
            spacy_token_list = df_pred.loc[:, model_cfg.target_column.token].tolist()
            # Some token has conll label but does not exist in gt, that is what `empty_token_idx_with_conll_label_dict` is used for.
            spacy2gt_tok_indices, empty_token_idx_with_conll_label_dict = align_spacy_to_ground_truth(
                gt_token_list, spacy_token_list, df_spacy=df_spacy, spacy_cfg=spacy_cfg, df_pred=df_pred, model_cfg=model_cfg)
            target_token_index_and_conll_label_dict = find_cloest_index(spacy2gt_tok_indices, empty_token_idx_with_conll_label_dict)
            sentence_list_pred = convert_spacy_token_csv_to_conll_format(config, model_cfg, section_name, doc_id, spacy2gt_tok_indices, df_pred, target_token_index_and_conll_label_dict)

        temp_dir_for_conll_pred = os.path.join(config.output.base_dir, model_cfg.name, config.output.base_temp_dir_name, "predict", section_name)
        check_and_create_dirs(temp_dir_for_conll_pred)
        pred_conll_path = write_conll_file(temp_dir_for_conll_pred, doc_id, sentence_list_pred)

        # Generate conll format ground-truth files
        sentence_list_gt = convert_gt_csv_to_conll_format(config, doc_id, df_gt)

        temp_gt_dir = os.path.join(config.output.base_dir, model_cfg.name, config.output.base_temp_dir_name, "ground_truth", section_name)
        check_and_create_dirs(temp_gt_dir)
        gt_conll_path = write_conll_file(temp_gt_dir, doc_id, sentence_list_gt)

        for scorer_metric in scorer_cfg.metrics:
            out, err = invoke_conll_script(scorer_cfg.path, scorer_metric, gt_conll_path, pred_conll_path)
            if err:
                logger.error("Error occur when invoking conll scorer script. Error msg: %s", err)
                reportInfo.mention_f1_dict[scorer_metric] = 0.0
                reportInfo.mention_recall_dict[scorer_metric] = 0.0
                reportInfo.mention_precision_dict[scorer_metric] = 0.0
                reportInfo.coref_f1_dict[scorer_metric] = 0.0
                reportInfo.coref_recall_dict[scorer_metric] = 0.0
                reportInfo.coref_precision_dict[scorer_metric] = 0.0
            else:
                mention_recall, mention_precision, mention_f1, coref_recall, coref_precision, coref_f1 = resolve_conll_script_output(out)
                reportInfo.mention_f1_dict[scorer_metric] = mention_f1
                reportInfo.mention_recall_dict[scorer_metric] = mention_recall
                reportInfo.mention_precision_dict[scorer_metric] = mention_precision
                reportInfo.coref_f1_dict[scorer_metric] = coref_f1
                reportInfo.coref_recall_dict[scorer_metric] = coref_recall
                reportInfo.coref_precision_dict[scorer_metric] = coref_precision
    except Exception:
        logger.error("Error occured in file: %s, section: %s", file_name, section_name)
        logger.error(traceback.format_exc())
        raise

    reportInfo.calculate_avg()
    return reportInfo


@hydra.main(version_base=None, config_path=config_path, config_name="statistic")
def main(config):
    # print(OmegaConf.to_yaml(config))

    check_and_remove_dirs(config.output.base_dir, config.clear_history)

    scoring_col_cfg = config.name_style.scoring.column_name
    model_cfg_list = [config.input.source.models.get(model_code) for model_code in config.input.source.use]

    for model_cfg in model_cfg_list:
        logger.info("Processing model output: %s", model_cfg.name)
        output_dir = os.path.join(config.output.base_dir, model_cfg.name)
        check_and_create_dirs(output_dir)
        startTime = time.time()
        for section_name in config.input.section:
            logger.info("--- Processing section: %s", section_name)
            df_scoring = pd.DataFrame(columns=[scoring_col_cfg.doc_id,
                                               scoring_col_cfg.conll_f1_avg, scoring_col_cfg.conll_recall_avg, scoring_col_cfg.conll_precision_avg,
                                               scoring_col_cfg.muc_f1, scoring_col_cfg.muc_recall, scoring_col_cfg.muc_precision,
                                               scoring_col_cfg.bcub_f1, scoring_col_cfg.bcub_recall, scoring_col_cfg.bcub_precision,
                                               scoring_col_cfg.ceafe_f1, scoring_col_cfg.ceafe_recall, scoring_col_cfg.ceafe_precision,
                                               scoring_col_cfg.mention_f1, scoring_col_cfg.mention_recall, scoring_col_cfg.mention_precision])
            input_dir = os.path.join(model_cfg.dir, section_name)

            tasks = []
            with ProcessPoolExecutor(max_workers=config.thread.workers) as executor:
                for file_name in FILE_CHECKER.filter(os.listdir(input_dir)):
                    doc_id = file_name.replace(".csv", "")
                    reportInfo = ReportInfo(doc_id, section_name)
                    tasks.append(executor.submit(batch_processing, config, model_cfg, section_name, file_name, input_dir, reportInfo))

                # Start multiprocessing
                START_EVENT.set()

                # Receive results from multiprocessing.
                for future in tqdm(as_completed(tasks), total=len(tasks)):
                    reportInfo: ReportInfo = future.result()
                    if reportInfo is None:
                        continue
                    # The muc, bcub and ceafe values at mention level are equal.
                    df_scoring.loc[len(df_scoring)] = [reportInfo.doc_id,
                                                       reportInfo.conll_f1_avg, reportInfo.conll_recall_avg, reportInfo.conll_precision_avg,
                                                       reportInfo.coref_f1_dict["muc"], reportInfo.coref_recall_dict["muc"], reportInfo.coref_precision_dict["muc"],
                                                       reportInfo.coref_f1_dict["bcub"], reportInfo.coref_recall_dict["bcub"], reportInfo.coref_precision_dict["bcub"],
                                                       reportInfo.coref_f1_dict["ceafe"], reportInfo.coref_recall_dict["ceafe"], reportInfo.coref_precision_dict["ceafe"],
                                                       reportInfo.mention_f1_dict["muc"], reportInfo.mention_recall_dict["muc"], reportInfo.mention_precision_dict["muc"]]

            # Overall statistic
            df_scoring.loc[len(df_scoring)] = df_scoring.mean(axis=0, numeric_only=True)
            df_scoring.loc[len(df_scoring)-1, scoring_col_cfg.doc_id] = "All document"
            df_scoring.to_csv(os.path.join(output_dir, f"conll_scores_{section_name}.csv"))

        # Log runtime information
        check_and_create_dirs(config.output.base_dir)
        with open(os.path.join(output_dir, config.output.log_file_name), "w", encoding="UTF-8") as f:
            log_out = {
                "Source": {
                    "Model name": model_cfg.name,
                    "Input data dir": model_cfg.dir,
                    "Data sections": str(config.input.section)
                },
                "Time cost": f"{time.time() - startTime:.2f}s"
            }
            f.write(json.dumps(log_out, indent=2))
            f.write("\n\n")


if __name__ == "__main__":
    sys.argv.append("+statistic/coref_scoring@_global_=i2b2")
    main()  # pylint: disable=no-value-for-parameter
