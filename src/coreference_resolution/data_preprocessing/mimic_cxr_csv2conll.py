import json
import logging
import os
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Event
from collections import Counter, defaultdict

import hydra
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

# pylint: disable=import-error,wrong-import-order
from common_utils.file_checker import FileChecker
from common_utils.common_utils import check_and_remove_dirs
from common_utils.coref_utils import ConllToken, check_and_make_dir, get_data_split, get_file_name_prefix, get_porportion_and_name, remove_all, resolve_mention_and_group_num, shuffle_list

FILE_CHECKER = FileChecker()
START_EVENT = Event()

logger = logging.getLogger()
pkg_path = os.path.dirname(__file__)
coref_module_path = os.path.dirname(pkg_path)
config_path = os.path.join(os.path.dirname(coref_module_path), "config")


def batch_processing(input_cfg, temp_cfg, section_name, input_file_path) -> int:
    """ All whitespces like "\n", "\n " and " " are skipped. 
    Return:
        True if this doc has at least one coref group. Otherwise False
    """
    START_EVENT.wait()

    doc_id = get_file_name_prefix(input_file_path, input_cfg.suffix)
    BEGIN = f"#begin document ({doc_id}_{section_name}); part 0\n"
    SENTENCE_SEPARATOR = "\n"
    END = "#end document\n"
    output_file_path = os.path.join(temp_cfg.base_dir, section_name, f"{doc_id}.conll")

    # Resolve CSV file
    sentenc_list: list[list[ConllToken]] = []
    df = pd.read_csv(input_file_path, index_col=0, na_filter=False)
    _, coref_group_num = resolve_mention_and_group_num(df, input_cfg.column_name.coref_group_conll)

    # Write .conll file only if doc has at least one coref group
    if coref_group_num > 0:
        sentence_id = 0
        while True:
            token_list: list[ConllToken] = []
            df_sentence = df[df.loc[:, input_cfg.column_name.sentence_group] == sentence_id].reset_index()
            if df_sentence.empty:
                break
            for _idx, data in df_sentence.iterrows():
                # Skip all whitespces like "\n", "\n " and " ".
                if str(data[input_cfg.column_name.token]).strip() == "":
                    continue
                conllToken = ConllToken(doc_id+"_"+section_name, sentence_id, _idx, data[input_cfg.column_name.token])
                coref_col_cell = data[input_cfg.column_name.coref_group_conll]
                if isinstance(coref_col_cell, str) and coref_col_cell != "-1":
                    conllToken.add_coref_label("|".join(ast.literal_eval(coref_col_cell)))
                token_list.append(conllToken)
            sentenc_list.append(token_list)
            sentence_id += 1

        with open(output_file_path, "w", encoding="UTF-8") as out:
            out.write(BEGIN)
            for sent in sentenc_list:
                # Skip empty sentence
                if len(sent) == 1 and sent[0].tokenStr == "":
                    continue
                for tok in sent:
                    out.write(tok.get_conll_str() + "\n")
                out.write(SENTENCE_SEPARATOR)
            out.write(END)

    return doc_id, coref_group_num


def convert_to_individual_conll(config, input_cfg, temp_cfg, section_name, section_docs_dir) -> tuple[Counter, dict]:

    section_temp_conll_dir = os.path.join(temp_cfg.base_dir, section_name)
    check_and_make_dir(section_temp_conll_dir)
    logger.debug("Generating individual conll files to: %s", section_temp_conll_dir)

    doc_files = FILE_CHECKER.filter(os.listdir(section_docs_dir))
    doc_coref_counter = Counter()
    with ProcessPoolExecutor(max_workers=config.thread.workers) as executor:
        all_task = []
        for file_name in tqdm(doc_files):
            # if len(all_task) > 100:
            #     break
            input_file_path = os.path.join(section_docs_dir, file_name)
            all_task.append(executor.submit(batch_processing, input_cfg, temp_cfg, section_name, input_file_path))

        # Notify tasks to start
        START_EVENT.set()

        corefGroupNum_docId_dict = defaultdict(list)
        if all_task:
            for future in tqdm(as_completed(all_task), total=len(all_task)):
                doc_id, coref_group_num = future.result()
                doc_coref_counter.update([coref_group_num])
                corefGroupNum_docId_dict[coref_group_num].append(doc_id)
            logger.info("Done.")
        else:
            logger.info("All empty. Skipped.")

        executor.shutdown(wait=True, cancel_futures=False)
        START_EVENT.clear()

    return doc_coref_counter, corefGroupNum_docId_dict


def prepare_conll(config, input_cfg, temp_cfg):

    # Process each sections
    log_out = []
    for section_name in input_cfg.section:
        logger.info("Processing section: [%s]", section_name)
        docs_dir = os.path.join(input_cfg.base_dir, section_name)
        doc_coref_counter, corefGroupNum_docId_dict = convert_to_individual_conll(config, input_cfg, temp_cfg, section_name, docs_dir)

        dict_file = os.path.join(temp_cfg.base_dir, f"{section_name}{temp_cfg.detail_file_suffix}")
        with open(dict_file, "w", encoding="UTF-8") as f:
            f.write(json.dumps(corefGroupNum_docId_dict))

        log_out.append({
            "Section": section_name,
            "File counter": dict([[f"coref_group_num ({group_num})", f"{doc_num} docs"] for group_num, doc_num in sorted(doc_coref_counter.items(), key=lambda x: x[0])]),
        })

    return log_out


def get_actural_doc_ids(file_path) -> dict[str, list[str]]:
    """ Get the actual doc (all) ids for each corefGroupNum.
    Return:
        {"0": ["sid1, sid2, ...."], "1": [], ....}
    """
    with open(file_path, "r", encoding="UTF-8") as f:
        groupNum_allDocId_dict: dict[str, list[str]] = ast.literal_eval(f.readline())
    return groupNum_allDocId_dict


def copy_and_paste_conll(input_conll_file, output_conll_file):
    with open(input_conll_file, "r", encoding="UTF-8") as f_in, open(output_conll_file, "a", encoding="UTF-8") as f_out:
        f_out.write("".join(f_in.readlines()))
        f_out.write("\n")


def aggregrate_conll(config) -> defaultdict[str, int]:
    """ Args:
    data_distribution: {'findings': {0: 141182, 1: 12772, 2: 1645, 3: 295, ...}, ...}
    """
    log_out = {}

    split_cfg_list = [config.data_split.get(mode_name) for mode_name in config.data_split.activate]
    for split_cfg in split_cfg_list:
        check_and_remove_dirs(os.path.join(config.output.base_dir, config.output.conll_dir_name, split_cfg.dir_name), True)
        log_out[split_cfg.dir_name] = {}
        # For the train and dev split
        logger.info("Processing [%s] split", split_cfg.dir_name)
        #  split_cfg such as train_dev_2k
        if split_cfg.get("sample_detail", None):
            data_split_name_list = [i.strip() for i in split_cfg.output_name_prefix.split(",")]
            for sectionName_sampleNum_dict in split_cfg.sample_detail:
                for section_name, sample_num_str in sectionName_sampleNum_dict.items():
                    log_out[split_cfg.dir_name][section_name] = {}

                    # Get the actual doc (all) ids for each corefGroupNum. {"0": ["sid1, sid2, ...."], "1": [], ....}
                    groupNum_allDocId_dict: dict[str, list[str]] = get_actural_doc_ids(os.path.join(config.temp_pred.base_dir, f"{section_name}{config.temp_pred.detail_file_suffix}"))

                    # Get the actual split number according the config.
                    data_split_proportion = [int(i) for i in split_cfg.proportion.split(",")]  # [8, 2]
                    data_split_proportion_norm = [i / sum(data_split_proportion) for i in data_split_proportion]  # [0.8, 0.2]

                    # sample_num_str = "{1: 544, 2: 544, 3: 295, 4: 83, 5: 25, 6: 6, 7: 2, 8: 1}"
                    for groupNum, sampleDocNum in ast.literal_eval(sample_num_str).items():
                        log_out[split_cfg.dir_name][section_name][groupNum] = defaultdict(int)
                        log_out[split_cfg.dir_name][section_name][groupNum]["expect_all"] = sampleDocNum

                        data_split_num_list = [int(i * sampleDocNum) for i in data_split_proportion_norm]  # [247.33..., 141.33...]
                        if sampleDocNum == 1:  # Assign to the train set first.
                            data_split_num_list[0] = 1
                        data_split_num_list[-1] = sampleDocNum - sum(data_split_num_list[0:-1])  # [247, 141] (train, dev)

                        # Get the acutal doc ids. Remove the doc_ids that used in testset. Then shuffle.
                        docId_list = groupNum_allDocId_dict[str(groupNum)]
                        docId_testset_list = [i.rstrip(".csv") for i in FILE_CHECKER.filter(os.listdir(os.path.join(split_cfg.test_docs_dir, section_name)))]
                        docId_list_exclude = [x for x in docId_list if x not in docId_testset_list]
                        docId_list_shuffle = shuffle_list(docId_list_exclude, config.shuffle_seed)
                        logger.debug("len(docId_list): before removing: %s, after removing %s", len(docId_list), len(docId_list_exclude))

                        for split_name, split_num in zip(data_split_name_list, data_split_num_list):

                            output_dir = os.path.join(config.output.base_dir, config.output.conll_dir_name, split_cfg.dir_name)
                            check_and_make_dir(output_dir)
                            output_conll_file = os.path.join(output_dir, f"{split_name}{config.output.suffix}")

                            # Aggregrate one by one
                            for doc_id in docId_list_shuffle[0:split_num]:
                                input_conll_file = os.path.join(config.temp_pred.base_dir, section_name, f"{doc_id}{config.output.suffix}")
                                copy_and_paste_conll(input_conll_file, output_conll_file)
                                log_out[split_cfg.dir_name][section_name][groupNum][split_name] += 1
                                log_out[split_cfg.dir_name][section_name][groupNum]["actual_all"] += 1

                            docId_list_shuffle = docId_list_shuffle[split_num:]

            # Extra logging info
            temp_dict = {}
            for split_name in data_split_name_list:
                temp_dict[split_name] = sum([val_dict[split_name] for _, gNum_dict in log_out[split_cfg.dir_name].items() for _, val_dict in gNum_dict.items()])
            for split_name in data_split_name_list:
                log_out[split_cfg.dir_name][f"{split_name} (actual all)"] = temp_dict[split_name]

        # split_cfg such as train_manual_100_1
        elif split_cfg.get("proportion", None):
            # Get the actual split number according the config.
            data_split_proportion = [int(i) for i in split_cfg.proportion.split(",")]  # [8, 2]
            data_split_proportion_norm = [i / sum(data_split_proportion) for i in data_split_proportion]  # [0.8, 0.2]
            # Split to train and dev
            data_split_name_list = [i.strip() for i in split_cfg.output_name_prefix.split(",")]

            for section_entry in os.scandir(split_cfg.target_doc_dir):
                if section_entry.is_file():
                    continue
                section_name = section_entry.name
                log_out[split_cfg.dir_name][section_name] = defaultdict(int)

                # Get the actual doc (all) ids for each corefGroupNum. {"0": ["sid1, sid2, ...."], "1": [], ....}. Then remove the keys.
                groupNum_allDocId_dict: dict[str, list[str]] = get_actural_doc_ids(os.path.join(split_cfg.source_dir, f"{section_name}{config.temp_pred.detail_file_suffix}"))
                allDocId_list = [docId for groupNum, docIds in groupNum_allDocId_dict.items() if groupNum != "0" for docId in docIds]
                # Remove docs that have 0_coref
                docId_trainset_list = [i.rstrip(".csv") for i in FILE_CHECKER.filter(os.listdir(os.path.join(split_cfg.target_doc_dir, section_name)))]
                docId_list = [x for x in docId_trainset_list if x in allDocId_list]
                docId_list_shuffle = shuffle_list(docId_list, config.shuffle_seed)

                log_out[split_cfg.dir_name][section_name]["expect_all"] = len(docId_trainset_list)

                # Get doc num for train and dev splits
                sampleDocNum = len(docId_list)
                data_split_num_list = [int(i * sampleDocNum) for i in data_split_proportion_norm]  # [247.33..., 141.33...]
                if sampleDocNum == 1:  # Assign to the train set first.
                    data_split_num_list[0] = 1
                data_split_num_list[-1] = sampleDocNum - sum(data_split_num_list[0:-1])  # [247, 141] (train, dev)

                for split_name, split_num in zip(data_split_name_list, data_split_num_list):
                    output_dir = os.path.join(config.output.base_dir, config.output.conll_dir_name, split_cfg.dir_name)
                    check_and_make_dir(output_dir)
                    output_conll_file = os.path.join(output_dir, f"{split_name}{config.output.suffix}")

                    # Aggregrate one by one
                    for doc_id in docId_list_shuffle[0:split_num]:
                        input_conll_file = os.path.join(split_cfg.source_dir, section_name, f"{doc_id}{config.output.suffix}")
                        copy_and_paste_conll(input_conll_file, output_conll_file)
                        log_out[split_cfg.dir_name][section_name][split_name] += 1
                        log_out[split_cfg.dir_name][section_name]["actual_all"] += 1
                    docId_list_shuffle = docId_list_shuffle[split_num:]

        # split_cfg such as test_gt test_pred
        else:
            split_name = "test"
            for section_entry in os.scandir(split_cfg.target_doc_dir):
                if section_entry.is_file():
                    continue
                section_name = section_entry.name

                # Get the actual doc (all) ids for each corefGroupNum. {"0": ["sid1, sid2, ...."], "1": [], ....}. Then remove the keys.
                groupNum_allDocId_dict: dict[str, list[str]] = get_actural_doc_ids(os.path.join(split_cfg.source_dir, f"{section_name}{config.temp_pred.detail_file_suffix}"))
                # Remove docs that have 0_coref
                allDocId_list = [docId for groupNum, docIds in groupNum_allDocId_dict.items() if groupNum != "0" for docId in docIds]
                docId_testset_list = [i.rstrip(".csv") for i in FILE_CHECKER.filter(os.listdir(os.path.join(split_cfg.target_doc_dir, section_name)))]
                docId_list = [x for x in docId_testset_list if x in allDocId_list]

                # docId_list = FILE_CHECKER.filter(os.listdir(section_entry.path))
                # docId_list = [i.rstrip(".conll") for i in docId_list]
                log_out[split_cfg.dir_name][section_name] = {"expect_all": len(docId_list), split_name: len(docId_list)}

                output_dir = os.path.join(config.output.base_dir, config.output.conll_dir_name, split_cfg.dir_name)
                check_and_make_dir(output_dir)
                output_conll_file = os.path.join(output_dir, f"{split_name}{config.output.suffix}")

                # Aggregrate one by one
                for doc_id in docId_list:
                    input_conll_file = os.path.join(split_cfg.source_dir, section_name, f"{doc_id}{config.output.suffix}")
                    copy_and_paste_conll(input_conll_file, output_conll_file)

            log_out[split_cfg.dir_name][split_name] = sum([val_dict[split_name] for _, val_dict in log_out[split_cfg.dir_name].items()])

    return log_out
