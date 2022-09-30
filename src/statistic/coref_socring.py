from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fileinput import filename
import json
import logging
import os
import sys
import time
import re
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Event
import numpy as np
import traceback

import hydra
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

# pylint: disable=import-error,wrong-import-order
from common_utils.common_utils import check_and_create_dirs, check_and_remove_dirs
from common_utils.coref_utils import ConllToken, auto_append_value_to_list
from common_utils.file_checker import FileChecker

logger = logging.getLogger()
module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")

FILE_CHECKER = FileChecker()
START_EVENT = Event()

# Utils


class ReportInfo:
    def __init__(self, sid, section_name) -> None:
        self.sid = sid
        self.section_name = section_name
        self.coref_mention_num = 0
        self.coref_group_num = 0
        self.conll_f1score: dict[str, float] = {}

    def set_conll_f1score(self, conll_f1score_dict: dict[str, float]):
        """ It would be like: {'muc': 0.0, 'bcub': 0.0, 'ceafe': 0.0} """
        self.conll_f1score = conll_f1score_dict

    def set_coref_mention_num(self, coref_mention_num: int):
        self.coref_mention_num = coref_mention_num

    def set_coref_group_num(self, coref_group_num: int):
        self.coref_group_num = coref_group_num


def resolve_singleton_list(df: pd.DataFrame, conll_colName: str) -> list[int]:
    """Args:
        df: The dataframe resolved from csv file.
        conll_colName: The name of the column with conll format elements.

    Return:
        A list of the coref group ids to which the singletons belong.
    """
    corefGroup_counter = Counter()
    for conll_corefGroup_list_str in df[~df.loc[:, conll_colName].isin(["-1", -1.0, np.nan])].loc[:, conll_colName].to_list():
        for conll_corefGroup_str in ast.literal_eval(conll_corefGroup_list_str):
            result = re.search(r"(\d+)\)", conll_corefGroup_str)  # An coref mention always end with "number)"
            if result:
                corefGroup_counter.update([int(result.group(1))])
    singleton_corefGroupId_checklist = [k for k, v in list(filter(lambda item: item[1] == 1, corefGroup_counter.items()))]
    return singleton_corefGroupId_checklist


def convert_to_conll_format(config, df: pd.DataFrame,  conll_colName: str, doc_id: str, sectionName: str, singleton_corefGroupId_checklist: list[int], remove_singleton=True) -> list[list[ConllToken]]:
    """Args:
        df: The dataframe resolved from csv file.
        conll_colName: The name of the column with conll format elements.
        singleton_corefGroupId_checklist: A list of the coref group ids to which the singletons belong.
        remove_singleton: True by defalut.

    Return:
        list[list[ConllToken]]: The first dimension is sentences. The second dimension is tokens.
    """
    sentence_list: list[list[ConllToken]] = []
    sentence_id = 0
    while True:
        token_list: list[ConllToken] = []
        df_sentence = df[df.loc[:, config.name_style.spacy.column_name.sentence_group] == sentence_id].reset_index()
        if df_sentence.empty:
            break
        for _idx, data in df_sentence.iterrows():
            # Skip all whitespces like "\n" and " ".
            if str(data[config.name_style.spacy.column_name.token]).strip() == "":
                continue
            conllToken = ConllToken(doc_id, sentence_id, _idx, data[config.name_style.spacy.column_name.token])
            conll_corefGroup_list_str = data[conll_colName]
            if isinstance(conll_corefGroup_list_str, str) and conll_corefGroup_list_str != "-1":
                if remove_singleton:
                    conll_corefGroup_list = ast.literal_eval(conll_corefGroup_list_str)

                    del_indices = []
                    for _idx, conll_corefGroup_str in enumerate(conll_corefGroup_list):
                        result = re.search(r"\d+", conll_corefGroup_str)
                        # This cell contain coreference mention singleton, we want to ignore this singleton but keep other coref mention.
                        if result and int(result.group(0)) in singleton_corefGroupId_checklist:
                            del_indices.append(_idx)
                    # Delete them in reverse order to avoid error
                    for _idx in sorted(del_indices, reverse=True):
                        del conll_corefGroup_list[_idx]

                    if conll_corefGroup_list:
                        conllToken.add_coref_mark("|".join(conll_corefGroup_list))
                else:
                    conllToken.add_coref_mark("|".join(ast.literal_eval(conll_corefGroup_list_str)))
            token_list.append(conllToken)
        sentence_list.append(token_list)
        sentence_id += 1
    return sentence_list


def write_conll_file(temp_output_dir, doc_id, sentence_list: list[list[ConllToken]]):
    BEGIN = f"#begin document ({doc_id}); part 0\n"
    SENTENCE_SEPARATOR = "\n"
    END = "#end document\n"

    with open(os.path.join(temp_output_dir, f"{doc_id}.conll"), "w", encoding="UTF-8") as out:
        out.write(BEGIN)
        for sent in sentence_list:
            # Skip empty sentence
            if len(sent) == 1 and sent[0].tokenStr == "":
                continue
            for tok in sent:
                out.write(tok.get_conll_str() + "\n")
            out.write(SENTENCE_SEPARATOR)
        out.write(END)
#######


def batch_processing(config, model_cfg, section_name, file_name, input_dir, temp_output_dir):
    """ Compute the CoNLL coreference scores """
    START_EVENT.wait()
    # if file_name != "clinical-542.csv":
    #     return
    # logger.info("Processing: %s", file_name)
    voting_column_cfg = config.name_style.voting.column_name
    try:
        doc_id = file_name.replace(".csv", "")
        reportInfo = ReportInfo(doc_id, section_name)

        input_file_path = os.path.join(input_dir, file_name)
        df_voted = pd.read_csv(input_file_path, index_col=0, na_filter=False)
        gt_file_path = os.path.join(config.input.gt_dir, file_name)
        df_gt = pd.read_csv(gt_file_path, index_col=0, na_filter=False)
        # Some of the i2b2 raw files are utf-8 start with DOM, but we didn't remove the DOM character, thus we fix it here.
        df_gt.iloc[0] = df_gt.iloc[0].apply(lambda x: x.replace("\ufeff", "").replace("\xef\xbb\xbf", "") if isinstance(x, str) else x)
        df_voted.iloc[0] = df_voted.iloc[0].apply(lambda x: x.replace("\ufeff", "").replace("\xef\xbb\xbf", "") if isinstance(x, str) else x)

        # Align to ground-truth
        gt_token_list = df_gt.loc[:, "[gt]original_token"].tolist()
        spacy_toekn_list = df_voted.loc[:, "[sp]token"].tolist()

        curr_gt_token_pointer = 0
        curr_spacy_token_pointer = 0

        output_sentences_indices: list[list[int]] = [-1] * len(gt_token_list)

        left_str = ""
        right_str = ""
        # We didn't correctly save the following possible nan value in the previous process. Thus, we need to resolve the nan value alignment here.
        nan_list = ["#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null"]
        while curr_gt_token_pointer < len(gt_token_list) and curr_spacy_token_pointer < len(spacy_toekn_list):
            curr_gt_token = str(gt_token_list[curr_gt_token_pointer])
            curr_spacy_token = str(spacy_toekn_list[curr_spacy_token_pointer])

            if curr_spacy_token.strip() == "":
                logger.debug("%s,%s) curr_spacy_token is empty in csv: %s", curr_gt_token_pointer, curr_spacy_token_pointer, curr_spacy_token.encode())
                curr_spacy_token_pointer += 1
                continue

            if curr_gt_token.strip() in ["", "nan"]:
                logger.debug("%s,%s) curr_spacy_token is empty: %s", curr_gt_token_pointer, curr_spacy_token_pointer, curr_gt_token.encode())
                curr_gt_token_pointer += 1
                continue

            if curr_gt_token == curr_spacy_token:
                logger.debug("%s,%s) curr_gt_token, curr_spacy_token: %s | %s", curr_gt_token_pointer, curr_spacy_token_pointer, curr_gt_token, curr_spacy_token)
                auto_append_value_to_list(output_sentences_indices, curr_gt_token_pointer, [curr_spacy_token_pointer])
                curr_gt_token_pointer += 1
                curr_spacy_token_pointer += 1
            else:
                if curr_spacy_token in curr_gt_token:
                    logger.debug("%s,%s) curr_gt_token contains curr_spacy_token: %s | %s", curr_gt_token_pointer, curr_spacy_token_pointer, curr_gt_token, curr_spacy_token)
                    right_str += curr_spacy_token
                    curr_spacy_token_pointer += 1
                else:
                    if curr_spacy_token == "nan":
                        nan_value = nan_list[[i in curr_gt_token for i in nan_list].index(True)]
                        logger.info("Replace unrecognized curr_spacy_token: %s with %s`", curr_spacy_token, nan_value)
                        curr_spacy_token = nan_value
                        right_str += curr_spacy_token
                        curr_spacy_token_pointer += 1
                    else:
                        raise Exception(
                            f"Error occured as curr_spacy_token: {curr_spacy_token.encode()} NOT in curr_gt_token: {curr_gt_token.encode()}. Details: {file_name}, gt_index={curr_gt_token_pointer}, spacy_index={curr_spacy_token_pointer}")

                if curr_gt_token == right_str:
                    logger.debug("%s,%s) curr_gt_token, right_str: %s | %s", curr_gt_token_pointer, curr_spacy_token_pointer, curr_gt_token, right_str)
                    auto_append_value_to_list(output_sentences_indices, curr_gt_token_pointer, [curr_spacy_token_pointer])
                    curr_gt_token_pointer += 1
                    right_str = ""

        assert spacy_toekn_list[curr_spacy_token_pointer].strip() == ""

        # # Remove singleton and generate temp files for CoNLL scripts
        # conll_colName = model_cfg.conll_column_name
        # # Get the ids of coref groups that only have singleton.
        # singleton_corefGroupId_checklist = resolve_singleton_list(df_voted, conll_colName)
        # # Ignore singleton when converting. For
        # sentence_list = convert_to_conll_format(config, df_voted,  conll_colName, doc_id, section_name, singleton_corefGroupId_checklist)
        # # Write file.
        # write_conll_file(temp_output_dir, doc_id, sentence_list)

    except Exception:
        logger.error(traceback.format_exc())
        raise

    return None


@hydra.main(version_base=None, config_path=config_path, config_name="statistic")
def main(config):
    print(OmegaConf.to_yaml(config))

    check_and_remove_dirs(config.output.base_dir, config.clear_history)

    model_cfg_list = [config.input.source.models.get(model_code) for model_code in config.input.source.use]

    for model_cfg in model_cfg_list:
        logger.debug("Processing model output: %s", model_cfg.name)
        for section_name in config.input.section:
            logger.debug("Processing section: %s", section_name)

            input_dir = os.path.join(model_cfg.dir, section_name)
            temp_output_dir = os.path.join(config.output.temp_dir, model_cfg.name, section_name)
            check_and_create_dirs(temp_output_dir)

            tasks = []
            with ProcessPoolExecutor(max_workers=config.thread.workers) as executor:
                for file_name in FILE_CHECKER.filter(os.listdir(input_dir)):
                    tasks.append(executor.submit(batch_processing, config, model_cfg, section_name, file_name, input_dir, temp_output_dir))

                # Start multiprocessing
                START_EVENT.set()

                # Receive results from multiprocessing.
                for future in tqdm(as_completed(tasks), total=len(tasks)):
                    future.result()


if __name__ == "__main__":
    sys.argv.append("+statistic/coref_scoring@_global_=i2b2")
    main()  # pylint: disable=no-value-for-parameter
