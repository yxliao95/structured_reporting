import ast
from collections import Counter
import logging
import os
import random
import re
import shutil
import pandas as pd
import numpy as np
from natsort import natsorted

logger = logging.getLogger()


class ConllToken(object):
    def __init__(self, docId, sentenceId, tokenId, tokenStr):
        self.docId = docId
        self.sentenceId = sentenceId
        self.tokenId = tokenId
        self.tokenStr = tokenStr
        self.corefLabel = ""

    def add_coref_label(self, label):
        if not self.corefLabel:
            self.corefLabel = label
        else:
            self.corefLabel = f"{self.corefLabel}|{label}"

    def get_conll_str(self):
        # IMPORTANT! Any tokens that trigger regex: \((\d+) or (\d+)\) will also
        # trigger "conll/reference-coreference-scorers" unexpectedly,
        # which will either cause execution error or wrong metric score.
        # See coref/wrong_conll_scorer_example for details.
        tok_str = self.tokenStr
        if re.search(r"\(?[^A-Za-z]+\)?", tok_str):
            tok_str = tok_str.replace("(", "[").replace(")", "]")
        if tok_str.strip() == "":
            tok_str = ""
        if self.corefLabel:
            return f"{self.docId}\t0\t{self.tokenId}\t{tok_str}\t" + "_\t" * 8 + self.corefLabel
        return f"{self.docId}\t0\t{self.tokenId}\t{tok_str}\t" + "_\t" * 7 + "_"

    def __str__(self) -> str:
        return f"{self.tokenStr}({self.sentenceId}:{self.tokenId})|[{self.corefLabel}]"

    __repr__ = __str__


def get_data_split(doc_files_shuffled: list, data_split_name: list, data_split_num: list):
    """ Split the dataset. The output is a list of dict:

    """
    data_split: list[dict] = []
    curr_index = 0
    for _idx, _output_name_prefix in enumerate(data_split_name):
        next_index = curr_index + data_split_num[_idx]
        curr_split = {"output_name_prefix": _output_name_prefix, "file_list": doc_files_shuffled[curr_index:next_index]}
        data_split.append(curr_split)
        curr_index = next_index
    logger.debug("Dataset is split into %s with proportion %s", data_split_name, data_split_num)
    return data_split


def shuffle_list(file_list, seed=42):
    """ Sort the list by file name numerically, then shuffle the list.

    Args:
        seed: 0 to use random seed to shuffle the dataset; -1 to disable shuffle.
    """
    file_list = natsorted(file_list)  # Sort by file name numerically
    if seed != -1:
        if seed == 0:
            random.Random().shuffle(file_list)
        else:
            random.Random(seed).shuffle(file_list)
    return file_list


def split_and_shuffle_list(doc_files, testset_size, fold_id: int, seed=42):
    """ Include three steps: Sort the list by file name numerically.
    Move the test set to the end of the list.
    Shuffle the train set and test set separately.

    Args:
        doc_files: Source file name list.
        testset_size: The size of the test set.
        fold_id: The index of the current cross-validation fold.
        seed: Random seed for shuffling.
    """
    doc_files = natsorted(doc_files)  # Sort by file name numerically
    doc_files_test = doc_files[fold_id * testset_size: (fold_id + 1) * testset_size]
    # The actual test split size is typicall larger than proportion value, causing the last test split size less than expected.
    if len(doc_files_test) < testset_size:
        doc_files_test = doc_files[-testset_size - 1: -1]
    doc_files_train = [i for i in doc_files if i not in doc_files_test]
    # Shuffle train/test files
    doc_files_train = shuffle_list(doc_files_train, seed)
    doc_files_test = shuffle_list(doc_files_test, seed)
    # Concat the shuffled train/test files
    return [*doc_files_train, *doc_files_test]


def get_porportion_and_name(split_config, doc_files):
    """ Get the actual numerical values for the dataset split and corresponding output name prefixes """
    # Compute the actual numerical values for the dataset split
    if split_config.proportion:
        data_split_proportion = [int(i) for i in split_config.proportion.split(",")]  # [7, 4, 1]
        data_split_proportion_norm = [i / sum(data_split_proportion) for i in data_split_proportion]  # [0.7, 0.4, 0.1]
        data_split_num = [int(i * len(doc_files)) for i in data_split_proportion_norm]  # [247.33..., 141.33..., 35.33...]
        data_split_num[-1] = len(doc_files) - sum(data_split_num[0:-1])  # [247, 141, 36]
        # The output names of the dataset split
        data_split_name = [i for i in split_config.output_name_prefix.split(",")]
    else:
        data_split_name = [split_config.output_name_prefix]
        data_split_num = [len(doc_files)]
    assert len(data_split_name) == len(data_split_num)
    return data_split_name, data_split_num


def check_and_make_dir(_dir, raiseExceptionIfExist=False, errMsg=""):
    if not os.path.exists(_dir):
        os.makedirs(_dir)
        logger.debug("Created directory: %s", _dir)
    else:
        logger.debug("The directory already exists: %s", _dir)
        if raiseExceptionIfExist:
            raise Exception(f"The directory {_dir} already exists. {errMsg}")


def remove_all(_dir):
    if os.path.exists(_dir):
        shutil.rmtree(_dir)


def get_file_name_prefix(file_path, suffix):
    """ Extract the basename from base and remove the ``suffix``.
    e.g.: ../../../clinical-1.txt => clinical-1
    """
    return os.path.basename(file_path).rstrip(suffix)

# Migrated


def remove_tag_from_list(text_list):
    """ Remove ``@[tag]`` from the text of the list, and return a new list """
    new_list = []
    for text in text_list:
        regexPattern = r"@\[.+\]"
        res = re.sub(regexPattern, "", text)
        new_list.append(res.strip())
    return new_list


def load_data(file_path, section_name_cfg):
    """Load data from ``input_path``.
    ``section_name_cfg`` is the config load from ``config/name_style/mimic_cxr_section.yaml``
    """
    df = pd.read_json(file_path, orient="records", lines=True)
    df = df.sort_values(by=[section_name_cfg.PID, section_name_cfg.SID])
    pid_list = df.loc[:, section_name_cfg.PID].to_list()
    sid_list = df.loc[:, section_name_cfg.SID].to_list()
    findings_list = remove_tag_from_list(df.loc[:, section_name_cfg.FINDINGS].to_list())
    impression_list = remove_tag_from_list(df.loc[:, section_name_cfg.IMPRESSION].to_list())
    pfi_list = remove_tag_from_list(df.loc[:, section_name_cfg.PFI].to_list())
    fai_list = remove_tag_from_list(df.loc[:, section_name_cfg.FAI].to_list())
    return len(sid_list), pid_list, sid_list, findings_list, impression_list, pfi_list, fai_list


def load_data_bySection(input_path, target_section_cfg, section_name_cfg):
    """
    Return:
        data_size: int
        pid_list: list[str]
        sid_list: list[str]
        section_list: list[tuple]; the tuple is like (section_name:str, section_text_list:list)
    """
    data_size, pid_list, sid_list, findings_list, impression_list, pfi_list, fai_list = load_data(input_path, section_name_cfg)

    section_list = []
    if target_section_cfg.findings:
        section_list.append((section_name_cfg.FINDINGS, findings_list))
    if target_section_cfg.impression:
        section_list.append((section_name_cfg.IMPRESSION, impression_list))
    if target_section_cfg.provisional_findings_impression:
        section_list.append((section_name_cfg.PFI, pfi_list))
    if target_section_cfg.findings_and_impression:
        section_list.append((section_name_cfg.FAI, fai_list))

    return data_size, pid_list, sid_list, section_list
# Migrated


def auto_append_value_to_list(target_list, target_index, target_value):
    """ Append target_value into target_list[target_index], where the initialized value is -1
    and should be changed to a list when a target_value is provided. """
    if target_list[target_index] == -1:
        target_list[target_index] = [target_value]
    else:
        target_list[target_index].append(target_value)


def auto_extend_value_to_list(target_list, target_index, target_value_list):
    """ Extend target_value into target_list[target_index], where the initialized value is -1
    and should be changed to a list when a target_value is provided. """
    if target_list[target_index] == -1:
        target_list[target_index] = target_value_list
    else:
        target_list[target_index].extend(target_value_list)

# For statistic


def resolve_mention_and_group_num(df: pd.DataFrame, conll_colName: str, omit_singleton=True) -> tuple[int, int]:
    """Args:
        df: The dataframe resolved from csv file.
        conll_colName: The name of the column with conll format elements.
        omit_singleton: Omit singleton mention and the corresponding coref group.

    Return:
        The number of coreference mentions and coreference groups.
    """
    corefGroup_counter = Counter()
    for conll_corefGroup_list_str in df[~df.loc[:, conll_colName].isin(["-1", -1.0, np.nan])].loc[:, conll_colName].to_list():
        if isinstance(conll_corefGroup_list_str, list):
            conll_corefGroup_str_list = conll_corefGroup_list_str
        else:
            conll_corefGroup_str_list = ast.literal_eval(conll_corefGroup_list_str)
        for conll_corefGroup_str in conll_corefGroup_str_list:
            result = re.search(r"(\d+)\)", conll_corefGroup_str)  # An coref mention always end with "number)"
            if result:
                corefGroup_counter.update([int(result.group(1))])
    if omit_singleton:
        non_singletone_counter: list[tuple] = list(filter(lambda item: item[1] > 1, corefGroup_counter.items()))
        coref_mention_num = sum([v for k, v in non_singletone_counter])
        coref_group_num = len([k for k, v in non_singletone_counter])
        return coref_mention_num, coref_group_num
