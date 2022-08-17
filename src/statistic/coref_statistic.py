import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import os
import shutil
import time

import hydra
import pandas as pd
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from common_utils.file_checker import FileChecker

logger = logging.getLogger()

pkg_dir = os.path.dirname(__file__)
src = os.path.dirname(pkg_dir)
config_path = os.path.join(src, "config")

FILE_CHECKER = FileChecker()


def statistic(config, input_file_path, sid):
    """ Read the csv files and return the sid (filename) and the number of coref groups. """
    df = pd.read_csv(input_file_path)
    df_coref_group = df[~df.loc[:, config.name_style.corenlp.column_name.coref_group].isin(["-1", -1.0, np.nan])]
    coref_group_set = set()
    for list_str in df_coref_group.loc[:, config.name_style.corenlp.column_name.coref_group].to_list():
        coref_group_set.update(ast.literal_eval(list_str))
    coerf_group_num = len(coref_group_set)
    return sid, coerf_group_num


class StatisticOutput:
    def __init__(self, section_name: str) -> None:
        self.section_name = section_name
        self.notEmpty_file_num = 0
        self.hasCoref_file_num = 0
        self._corefGroup_list_ = []

    def add_coerf_group_num(self, coerf_group_num):
        self._corefGroup_list_.append(coerf_group_num)
        self.hasCoref_file_num += 1 if coerf_group_num > 0 else 0

    def get_final_output(self):
        c = Counter(self._corefGroup_list_)
        c = sorted(c.items(), key=lambda pair: pair[0], reverse=False)
        out = {
            "Section name": self.section_name,
            "Number of reports (sections) that are not empty": self.notEmpty_file_num,
            "Number of reports (sections) that have coreference mentions": self.hasCoref_file_num,
            "Percentage": f"{self.hasCoref_file_num / self.notEmpty_file_num:.2f}",
            "Details": dict([[f"Number of files that have {k} coref group(s)", v] for k, v in c])
        }
        return json.dumps(out, indent=2)


def main_process(config):
    input_dir = config.coref.target_dir
    for section_entry in os.scandir(input_dir):
        # Write the statistical result individually. One file each section.
        out = StatisticOutput(section_entry.name)
        logger.info("Analysing section: %s", out.section_name)
        tasks = []
        # Construct and start multiprocessing
        with ProcessPoolExecutor(max_workers=config.thread.workers) as executor:
            for report_entry in tqdm(os.scandir(section_entry.path)):
                if FILE_CHECKER.ignore(report_entry.path):
                    continue
                out.notEmpty_file_num += 1
                sid = report_entry.name.rstrip(config.coref_data_preprocessing.mimic_cxr.input.suffix)
                tasks.append(executor.submit(statistic, config, report_entry.path, sid))
                if out.notEmpty_file_num % 100 == 0:
                    time.sleep(0.01)
        # Receive results from multiprocessing. Individual statistic.
        for future in tqdm(as_completed(tasks)):
            sid, coerf_group_num = future.result()
            out.add_coerf_group_num(coerf_group_num)
            with open(os.path.join(config.coref.output_dir, f"{out.section_name}"), "a", encoding="UTF-8") as f:
                f.write(f"{sid} {coerf_group_num}\n")
        # Overall statistic
        with open(os.path.join(config.coref.output_dir, "overall"), "a", encoding="UTF-8") as f:
            f.write(f"{out.get_final_output()}\n")


@hydra.main(version_base=None, config_path=config_path, config_name="statistic")
def main(config):
    logger.debug(OmegaConf.to_yaml(config))

    output_base_dir = config.coref.output_dir
    if config.coref.clear_history and os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    main_process(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
