import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
from multiprocessing import Event
import os
import re
import shutil
import sys
import time
import subprocess
import traceback

import hydra
import pandas as pd
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
# pylint: disable=import-error
from common_utils.common_utils import check_and_create_dirs, check_and_remove_dirs
from common_utils.coref_utils import ConllToken
from common_utils.file_checker import FileChecker

logger = logging.getLogger()
module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")

FILE_CHECKER = FileChecker()


# utils

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
        for conll_corefGroup_str in ast.literal_eval(conll_corefGroup_list_str):
            result = re.search(r"(\d+)\)", conll_corefGroup_str)  # An coref mention always end with "number)"
            if result:
                corefGroup_counter.update([int(result.group(1))])
    if omit_singleton:
        non_singletone_counter: list[tuple] = list(filter(lambda item: item[1] > 1, corefGroup_counter.items()))
        coref_mention_num = sum([v for k, v in non_singletone_counter])
        coref_group_num = len([k for k, v in non_singletone_counter])
        return coref_mention_num, coref_group_num


def resolve_singleton_list(df: pd.DataFrame, conll_colName: str) -> list[int]:
    """Args:
        df: The dataframe resolved from csv file.
        conll_colName: The name of the column with conll format elements.

    Return:
        A list of the coref group ids which the singletons belong to.
    """
    corefGroup_counter = Counter()
    for conll_corefGroup_list_str in df[~df.loc[:, conll_colName].isin(["-1", -1.0, np.nan])].loc[:, conll_colName].to_list():
        for conll_corefGroup_str in ast.literal_eval(conll_corefGroup_list_str):
            result = re.search(r"(\d+)\)", conll_corefGroup_str)  # An coref mention always end with "number)"
            if result:
                corefGroup_counter.update([int(result.group(1))])
    singleton_corefGroupId_checklist = [k for k, v in list(filter(lambda item: item[1] == 1, corefGroup_counter.items()))]
    return singleton_corefGroupId_checklist


def convert_to_conll_format(config, df: pd.DataFrame,  conll_colName: str, sid: str, sectionName: str, singleton_corefGroupId_checklist: list[int], remove_singleton=True) -> list[list[ConllToken]]:
    """Args:
        df: The dataframe resolved from csv file.
        conll_colName: The name of the column with conll format elements.
        singleton_corefGroupId_checklist: A list of the coref group ids which the singletons belong to.
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
            conllToken = ConllToken(sid+"_"+sectionName, sentence_id, _idx, data[config.name_style.spacy.column_name.token])
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


def write_conll_file(temp_output_dir, sid, model_prefix, sentence_list: list[list[ConllToken]]):
    BEGIN = f"#begin document ({sid}); part 0\n"
    SENTENCE_SEPARATOR = "\n"
    END = "#end document\n"

    with open(os.path.join(temp_output_dir, f"{sid}-{model_prefix}.conll"), "w", encoding="UTF-8") as out:
        out.write(BEGIN)
        for sent in sentence_list:
            # Skip empty sentence
            if len(sent) == 1 and sent[0].tokenStr == "":
                continue
            for tok in sent:
                out.write(tok.get_conll_str() + "\n")
            out.write(SENTENCE_SEPARATOR)
        out.write(END)


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


def get_source_model_info(scorer_cfg):
    """Returns:
        coref_group_conll_colName_list: A list of the target column name mentioned in statistic.yaml
        coref_model_prefix_name_list: A list of the prefix of the coref model mentioned in statistic.yaml.
    """
    coref_group_conll_colName_list = []
    coref_model_prefix_name_list = []
    for model_keyName in scorer_cfg.source.in_use:
        modelOutput_cfg = scorer_cfg.source.models.get(model_keyName)
        coref_model_prefix_name_list.append(modelOutput_cfg.column_prefix)
        coref_group_conll_colName_list.append(modelOutput_cfg.column_name)
    return coref_group_conll_colName_list, coref_model_prefix_name_list


def get_pairwise_model_name_for_column(source_model_names):
    """ It would be like '[ml]-[rb]' """
    candidate_coref_model_prefix_list = source_model_names.copy()
    pairwise_model_colName_list: list[str] = []
    while len(candidate_coref_model_prefix_list) > 0:
        pair1 = candidate_coref_model_prefix_list.pop(0)
        for pair2 in candidate_coref_model_prefix_list:
            pairwise_model_colName_list.append(f"{pair1}-{pair2}")
    return pairwise_model_colName_list
###


class ReportInfo:
    def __init__(self, sid, section_name) -> None:
        self.sid = sid
        self.section_name = section_name
        self.coref_mention_num = {}
        self.coref_group_num = {}
        self.conll_f1score = {}

    def set_conll_f1score(self, conll_f1score_dict: dict[str, dict[str, float]]):
        """ It would be like: {'[ml]-[rb]': {'muc': 0.0, 'bcub': 0.0, 'ceafe': 0.0}, ...} """
        self.conll_f1score = conll_f1score_dict

    def set_coref_mention_num(self, coref_mention_num: dict[str, int]):
        """ It would be like: {'[ml]': 0, '[rb]': 0, '[fj]': 0} """
        self.coref_mention_num = coref_mention_num

    def set_coref_group_num(self, coref_group_num: dict[str, int]):
        """ It would be like: {'[ml]': 0, '[rb]': 0, '[fj]': 0} """
        self.coref_group_num = coref_group_num


class SectionOverallInfo:
    def __init__(self, config, section_name: str) -> None:
        self.config = config
        self.section_name = section_name
        self.notEmpty_file_num = 0
        self.hasCoref_file_num = 0
        self.df = None
        self._column_suffix_list1 = ["mention_num", "group_num"]
        self._column_suffix_list2 = config.coref.scorer.metrics  # [muc, bcub, ceafe]
        self.initDataFrame(config.coref.scorer)

    def initDataFrame(self, scorer_cfg):
        _, coref_model_prefix_name_list = get_source_model_info(scorer_cfg)
        pairwise_model_colName_list = get_pairwise_model_name_for_column(coref_model_prefix_name_list)
        coref_mention_and_group_columns = [n+suffix for n in coref_model_prefix_name_list for suffix in self._column_suffix_list1]
        conll_f1score_columns = [n+suffix for n in pairwise_model_colName_list for suffix in self._column_suffix_list2]
        self.df = pd.DataFrame(columns=coref_mention_and_group_columns+conll_f1score_columns)

    def update(self, reportInfo: ReportInfo):
        sid = reportInfo.sid
        self.df.loc[sid] = ""
        for source_model_prefix_name, mention_num in reportInfo.coref_mention_num.items():
            self.df.loc[sid][source_model_prefix_name+"mention_num"] = int(mention_num)

        for source_model_prefix_name, group_num in reportInfo.coref_group_num.items():
            self.df.loc[sid][source_model_prefix_name+"group_num"] = int(group_num)

        for pairwise_model_name, metrics_score_dict in reportInfo.conll_f1score.items():
            for metric_name in self._column_suffix_list2:
                self.df.loc[sid][pairwise_model_name+metric_name] = float(metrics_score_dict[metric_name])

    def get_overall_output(self):
        out = {
            "Section name": self.section_name,
            "Number of reports (sections) that are not empty": self.notEmpty_file_num,
        }
        out["Details"] = {}
        for column_type1_name in self.df.filter(regex="|".join(self._column_suffix_list1), axis=1).columns.to_list():
            c = Counter(self.df.loc[:, column_type1_name].tolist())
            c = sorted(c.items(), key=lambda counterItem: counterItem[0], reverse=False)
            out["Details"][column_type1_name] = {
                "Sum": self.df.loc[:, column_type1_name].sum(),
                "Mean": self.df.loc[:, column_type1_name].mean(),
                "Median": self.df.loc[:, column_type1_name].median(),
                "Counter": dict([[f"Number of files that {column_type1_name}={k}", v] for k, v in c]),
            }
        for column_type2_name in self.df.filter(regex="|".join(self._column_suffix_list2), axis=1).columns.to_list():
            out["Details"][column_type2_name] = {
                "Avg F1 score": self.df.loc[:, column_type2_name].mean(),
            }

        return json.dumps(out, indent=2)


START_EVENT = Event()


def statistic(config, input_file_path, section_name, sid, temp_output_dir) -> ReportInfo:
    START_EVENT.wait()

    scorer_cfg = config.coref.scorer

    df = pd.read_csv(input_file_path, index_col=0)
    reportInfo = ReportInfo(sid, section_name)

    # Compute the CoNLL coreference scores
    # Read from config and decide which model outputs to use.
    coref_group_conll_colName_list, coref_model_prefix_name_list = get_source_model_info(scorer_cfg)
    # Get the keys for the ``conll_f1score_dict``
    pairwise_model_colName_list = get_pairwise_model_name_for_column(coref_model_prefix_name_list)

    # Remove singleton and generate temp files for CoNLL scripts
    for conll_colName, model_prefix in zip(coref_group_conll_colName_list, coref_model_prefix_name_list):
        # Get singleton
        singleton_corefGroupId_checklist = resolve_singleton_list(df, conll_colName)
        # Ignore singleton when converting
        sentence_list = convert_to_conll_format(config, df,  conll_colName, sid, section_name, singleton_corefGroupId_checklist)
        # Write file.
        write_conll_file(temp_output_dir, sid, model_prefix, sentence_list)

    # Invoke CoNLL script
    idx = 0
    conll_f1score_dict: dict[str, dict[str, float]] = {}
    candidate_conll_file_path_for_pairwise_comparision = []
    for model_prefix in coref_model_prefix_name_list:
        candidate_conll_file_path_for_pairwise_comparision.append(os.path.join(temp_output_dir, f"{sid}-{model_prefix}.conll"))

    while len(candidate_conll_file_path_for_pairwise_comparision) > 0:
        serve_as_groundtruth = candidate_conll_file_path_for_pairwise_comparision.pop(0)
        for serve_as_predict in candidate_conll_file_path_for_pairwise_comparision:
            conll_f1score_dict[pairwise_model_colName_list[idx]] = {}
            for scorer_metric in scorer_cfg.metrics:
                try:
                    out, err = invoke_conll_script(scorer_cfg.path, scorer_metric, serve_as_groundtruth, serve_as_predict)
                    if err:
                        logger.error("Error occur when invoking conll scorer script. Error msg: %s", err)
                        conll_f1score_dict[pairwise_model_colName_list[idx]][scorer_metric] = 0.0
                    else:
                        _, _, _, _, _, coref_f1 = resolve_conll_script_output(out)
                        conll_f1score_dict[pairwise_model_colName_list[idx]][scorer_metric] = coref_f1
                except subprocess.CalledProcessError as e:
                    logger.error("Failed when running conll script. sid: %s, section name: %s. Error msg: %s", sid, section_name, e)
                    logger.error(traceback.format_exc())
            idx += 1

    reportInfo.set_conll_f1score(conll_f1score_dict)

    # Statistic for the number of coreference mentions and coref groups
    coref_mention_num_dict, coref_group_num_dict = {}, {}
    for conll_colName, model_prefix in zip(coref_group_conll_colName_list, coref_model_prefix_name_list):
        coref_mention_num, coref_group_num = resolve_mention_and_group_num(df, conll_colName)
        coref_mention_num_dict[model_prefix] = coref_mention_num
        coref_group_num_dict[model_prefix] = coref_group_num

    reportInfo.set_coref_mention_num(coref_mention_num_dict)
    reportInfo.set_coref_group_num(coref_group_num_dict)

    # Statistic of coreference mention overlaps

    # Remove temp files
    # for model_prefix in coref_model_prefix_name_list:
    #     os.remove(os.path.join(temp_output_dir, f"{sid}-{model_prefix}.conll"))

    return sid, reportInfo


def main_process(config):
    input_dir = config.coref.input_dir
    for section_entry in os.scandir(input_dir):
        if section_entry.is_dir():
            # Write the statistical result individually. One file each section.
            sectionOverallInfo = SectionOverallInfo(config, section_entry.name)
            logger.info("Analysing section: %s", sectionOverallInfo.section_name)

            temp_output_dir = os.path.join(config.coref.scorer.temp_data_dir, section_entry.name)
            check_and_create_dirs(temp_output_dir)

            # Construct multiprocessing
            tasks = []
            with ProcessPoolExecutor(max_workers=config.thread.workers) as executor:
                # counter = 0
                for report_entry in tqdm(os.scandir(section_entry.path)):
                    if FILE_CHECKER.ignore(report_entry.path):
                        continue
                    # if counter >= 2:
                    #     break
                    # else:
                    #     counter += 1
                    sectionOverallInfo.notEmpty_file_num += 1
                    sid = report_entry.name.rstrip(".csv")
                    if sid != "s52683488":
                        continue
                    tasks.append(executor.submit(statistic, config, report_entry.path, section_entry.name, sid, temp_output_dir))

                # Start multiprocessing
                START_EVENT.set()

                # Receive results from multiprocessing.
                for future in tqdm(as_completed(tasks), total=len(tasks)):
                    sid, reportInfo = future.result()
                    sectionOverallInfo.update(reportInfo)

                # Statistical details
                detail_csv_file = os.path.join(config.coref.output_dir, section_entry.name+"_details.csv")
                sectionOverallInfo.df.to_csv(detail_csv_file)
                # Overall statistics
                with open(os.path.join(config.coref.output_dir, "overall"), "a", encoding="UTF-8") as f:
                    f.write(f"{sectionOverallInfo.get_overall_output()}\n")

                START_EVENT.clear()


@hydra.main(version_base=None, config_path=config_path, config_name="statistic")
def main(config):
    print(OmegaConf.to_yaml(config))

    # output_base_dir = config.coref.output_dir
    # check_and_remove_dirs(output_base_dir, config.coref.clear_history)
    # check_and_create_dirs(output_base_dir)

    # main_process(config)

    # # check_and_remove_dir(config.coref.scorer.temp_data_dir, do_remove=True)
    # logger.info("Done. Please find the statistical output in: %s", output_base_dir)


if __name__ == "__main__":
    sys.argv.append("statistic@_global_=coref_statistic")
    main()  # pylint: disable=no-value-for-parameter
