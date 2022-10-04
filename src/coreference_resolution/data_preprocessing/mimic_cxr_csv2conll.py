import logging
import os
import ast
import shutil

import hydra
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from common_utils.file_checker import FileChecker
from common_utils.coref_utils import ConllToken, check_and_make_dir, get_data_split, get_file_name_prefix, get_porportion_and_name, remove_all, shuffle_list

FILE_CHECKER = FileChecker()

logger = logging.getLogger()
pkg_path = os.path.dirname(__file__)
coref_module_path = os.path.dirname(pkg_path)
config_path = os.path.join(os.path.dirname(coref_module_path), "config")


def convert_each(config, sectionName, input_file_path, output_file_path):
    mimic_cfg = config.coref_data_preprocessing.mimic_cxr
    sid = get_file_name_prefix(input_file_path, mimic_cfg.input.suffix)
    BEGIN = f"#begin document ({sid}); part 0\n"
    SENTENCE_SEPARATOR = "\n"
    END = "#end document\n"

    # Resolve CSV file
    sentenc_list: list[list[ConllToken]] = []
    df = pd.read_csv(input_file_path, index_col=0)
    sentence_id = 0
    while True:
        token_list: list[ConllToken] = []
        df_sentence = df[df.loc[:, config.name_style.spacy.column_name.sentence_group] == sentence_id].reset_index()
        if df_sentence.empty:
            break
        for _idx, data in df_sentence.iterrows():
            # Skip all whitespces like "\n", "\n " and " ".
            if str(data[config.name_style.spacy.column_name.token]).strip() == "":
                continue
            conllToken = ConllToken(sid+"_"+sectionName, sentence_id, _idx, data[config.name_style.spacy.column_name.token])
            coref_col_cell = data[config.name_style.corenlp.column_name.coref_group_conll]
            if isinstance(coref_col_cell, str) and coref_col_cell != "-1":
                conllToken.add_coref_label("|".join(ast.literal_eval(coref_col_cell)))
            token_list.append(conllToken)
        sentenc_list.append(token_list)
        sentence_id += 1

    # Write .conll file
    with open(output_file_path, "a", encoding="UTF-8") as out:
        out.write(BEGIN)
        for sent in sentenc_list:
            # Skip empty sentence
            if len(sent) == 1 and sent[0].tokenStr == "":
                continue
            for tok in sent:
                out.write(tok.get_conll_str() + "\n")
            out.write(SENTENCE_SEPARATOR)
        out.write(END)


def convert_all(config, sectionName, data_split, docs_dir, output_dir):
    mimic_cfg = config.coref_data_preprocessing.mimic_cxr
    for split in data_split:
        logger.info('Converting the files for [%s] set:', split["output_name_prefix"])
        # Output file
        output_file_path = os.path.join(output_dir, split["output_name_prefix"] + mimic_cfg.output.suffix)
        for _file_name in tqdm(split["file_list"]):
            # Input files
            input_file_path = os.path.join(docs_dir, _file_name)
            convert_each(config, sectionName, input_file_path, output_file_path)
        logger.info("Output: %s", output_file_path)


def convert_to_conll(config, sectionName, output_base_dir, docs_dir):
    doc_files = FILE_CHECKER.filter(os.listdir(docs_dir))
    logger.info("Detected %s documents in total.", len(doc_files))

    mimic_cfg = config.coref_data_preprocessing.mimic_cxr
    cfg_list = [mimic_cfg.data_split.if_unsplit] if mimic_cfg.data_split.unsplit else []
    cfg_list.append(mimic_cfg.data_split.if_split) if mimic_cfg.data_split.split else None

    # Generate datasets with correspondingh split configs
    for cfg in cfg_list:
        data_split_name, data_split_num = get_porportion_and_name(cfg, doc_files)
        doc_files_shuffle = shuffle_list(doc_files, mimic_cfg.shuffle_seed)  # Shuffle the file list

        data_split = get_data_split(doc_files_shuffle, data_split_name, data_split_num)  # Split the files

        dir_name_suffix = cfg.dir_name_suffix
        dir_name = sectionName + ("_" + dir_name_suffix if dir_name_suffix else "")
        output_dataset_dir = os.path.join(output_base_dir, dir_name)
        # The history output dir will be deleted and created again.
        if mimic_cfg.clear_history:
            remove_all(output_dataset_dir)

        logger.info("*** Creating an [%s] dataset at: %s", cfg.log_hint, output_dataset_dir)
        output_conll_dir = os.path.join(output_dataset_dir, mimic_cfg.output.root_dir_name)
        check_and_make_dir(output_conll_dir)

        convert_all(config, sectionName, data_split, docs_dir, output_conll_dir)


def invoke(config, temp_output_dir: str):
    section_name = config.name_style.mimic_cxr.section_name
    multiprocessing_cfg = config.coref_data_preprocessing.mimic_cxr.multiprocessing
    mimic_cfg = config.coref_data_preprocessing.mimic_cxr

    # Sections that required to process.
    section_list = []
    if multiprocessing_cfg.target_section.findings:
        section_list.append(section_name.FINDINGS)
    if multiprocessing_cfg.target_section.impression:
        section_list.append(section_name.IMPRESSION)
    if multiprocessing_cfg.target_section.provisional_findings_impression:
        section_list.append(section_name.PFI)
    if multiprocessing_cfg.target_section.findings_and_impression:
        section_list.append(section_name.FAI)

    # Process each sections
    output_base_dir = mimic_cfg.output_dir
    for _sectionName in section_list:
        logger.info("Processing section: [%s]", _sectionName)
        docs_dir = os.path.join(temp_output_dir, _sectionName)
        convert_to_conll(config, _sectionName, output_base_dir, docs_dir)

    # Remove temp dir
    if mimic_cfg.remove_temp and os.path.exists(temp_output_dir):
        logger.debug("Removed the temporary directory: %s", temp_output_dir)
        shutil.rmtree(temp_output_dir)

    return output_base_dir


@hydra.main(version_base=None, config_path=config_path, config_name="coreference_resolution")
def main(config):
    OmegaConf.to_yaml(config)

    invoke(config, config.coref_data_preprocessing.mimic_cxr.temp_dir)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
