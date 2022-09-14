import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pipe
import traceback

import hydra
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
# pylint: disable=import-error,wrong-import-order
from common_utils.coref_utils import remove_tag_from_list
from common_utils.nlp_utils import align, align_byIndex_individually_nestedgruop, align_coref_groups_in_conll_format, getTokenOffset, align_byIndex_individually_withData_noOverlap
from nlp_ensemble.nlp_processor.spacy_process import SpacyProcess
from nlp_ensemble.nlp_processor.corenlp_process import CorenlpUrlProcess, formatCorenlpDocument

logger = logging.getLogger()
pkg_path = os.path.dirname(__file__)
coref_module_path = os.path.dirname(pkg_path)
config_path = os.path.join(os.path.dirname(coref_module_path), "config")


def load_data(file_path, section_name):
    """Load data from ``file_path``. ``section_name`` is the config load from ``config/name_style/mimic_cxr_section.yaml``"""
    df = pd.read_json(file_path, orient="records", lines=True)
    df = df.sort_values(by=[section_name.PID, section_name.SID])
    pid_list = df.loc[:, section_name.PID].to_list()
    sid_list = df.loc[:, section_name.SID].to_list()
    findings_list = remove_tag_from_list(df.loc[:, section_name.FINDINGS].to_list())
    impression_list = remove_tag_from_list(df.loc[:, section_name.IMPRESSION].to_list())
    pfi_list = remove_tag_from_list(df.loc[:, section_name.PFI].to_list())
    fai_list = remove_tag_from_list(df.loc[:, section_name.FAI].to_list())
    return len(sid_list), pid_list, sid_list, findings_list, impression_list, pfi_list, fai_list


def batch_processing(input_text_list: list, input_sid_list: list, input_pid_list: list, sectionName: str, progressId: int, config) -> tuple[int, str, int]:  # pylint: disable=unused-argument
    """ The task of multiprocessing.
    Args:
        input_text_list: A batch of list of the corresponding section text
        input_sid_list: A batch of sid list
        input_pid_list: A batch of pid list
        sectionName: The name of the section to which the input_text_list belongs.
        progressId: Start from 0
        config: hydra config

    Return:
        progressId: The ``progressId`` of this process
        msg: The message passed to the main process.
        num: The number of records processed in this batch.
    """
    batch_data = dict.fromkeys(input_sid_list, None)  # Dict: Key = sid, Value = {df_[name]:DataFrame, ...}
    try:
        logger.debug("Batch Process [%s] started: %s", progressId, input_sid_list)
        # We create three sub processors as below, as each of them take 3s to process a batch of 10 records
        # 1. For Spacy
        spacy_outPipe, spacy_inPipe = Pipe(False)
        spacy_process = SpacyProcess(progressId, spacy_inPipe, input_text_list, input_sid_list)
        spacy_process.start()
        # 3. For CoreNLP
        corenlp_url = config.nlp.corenlp.request_url
        corenlp_outPipe, corenlp_inPipe = Pipe(False)
        corenlp_process = CorenlpUrlProcess(corenlp_url, progressId, corenlp_inPipe, input_text_list, input_sid_list)
        corenlp_process.start()
        # corenlp_dict = corenlp_outPipe.recv()

        # The integration process
        # 1. Spacy
        spacy_output = spacy_outPipe.recv()  # Wait until receive a result, thus no need for p.join()
        spacy_prcess_id = spacy_output["processId"]
        spacy_output_list = spacy_output["data"]
        logger.debug("Batch Process [%s] received spacy [%s]", progressId, spacy_prcess_id)
        time1 = time.time()
        spacy_nametyle = config.name_style.spacy.column_name
        for spacy_output in spacy_output_list:
            sid = spacy_output["sid"]
            doc = spacy_output["doc"]
            df_spacy = pd.DataFrame(
                {
                    spacy_nametyle["token"]: [tok.text for tok in doc],
                    spacy_nametyle["token_offset"]: getTokenOffset(doc.text, doc),
                    spacy_nametyle["sentence_group"]: align(len(doc), doc.sents),
                }
            )
            batch_data[sid] = {"df_spacy": df_spacy}  # Must create a dict here rather than when define the variable.
        time2 = time.time()
        logger.debug(
            "Batch Process [%s] finished processing Spacy [%s], cost: %ss", progressId, spacy_prcess_id, time2-time1
        )

        # 3. CoreNLP
        corenlp_output = corenlp_outPipe.recv()
        corenlp_processId = corenlp_output["processId"]
        corenlp_dict = corenlp_output["data"]
        logger.debug("Batch Process [%s] received CoreNLP [%s]", progressId, corenlp_processId)
        time5 = time.time()
        for sid in input_sid_list:
            corenlp_json = json.loads(corenlp_dict[sid])
            df_base = batch_data[sid]["df_spacy"]
            tokenOffset_base = df_base.loc[:, spacy_nametyle["token_offset"]].tolist()
            referTo_spacy, tokenTotalNum, corefMentionInfo, corefGroupInfo, _, _, _ = formatCorenlpDocument(tokenOffset_base, corenlp_json)
            # Align each item to coreNLP token first, then algin the whole df to spacy df by index mapping (referTo_spacy).
            df_corenlp_rowsNum = tokenTotalNum
            corenlp_nametyle = config.name_style.corenlp.column_name
            df_corenlp = pd.DataFrame(
                {
                    corenlp_nametyle["token"]: [token["originalText"] for sentence in corenlp_json["sentences"] for token in sentence["tokens"]],
                    corenlp_nametyle["coref_mention"]: align_byIndex_individually_withData_noOverlap(df_corenlp_rowsNum, corefMentionInfo),
                    corenlp_nametyle["coref_group"]: align_byIndex_individually_nestedgruop(df_corenlp_rowsNum, corefGroupInfo),
                    corenlp_nametyle["coref_group_conll"]: align_coref_groups_in_conll_format(df_corenlp_rowsNum, corefGroupInfo),
                },
                index=referTo_spacy,
            )
            batch_data[sid]["df_corenlp"] = df_corenlp
            # print(f"sid:{sid}, df_corenlp.shape:{df_corenlp.shape}")
        time6 = time.time()
        logger.debug("Batch Process [%s] finished processing CoreNLP [%s], cost: %ss", progressId, corenlp_processId, time6-time5)

        # Aggregation, and write to the disk
        for _sid, _df in batch_data.items():
            df_all = _df["df_spacy"]
            df_all = df_all.join(_df["df_corenlp"])
            # Output csv for later converting to conll
            csv_temp_dir = os.path.join(config.coref_data_preprocessing.mimic_cxr.temp_dir, sectionName)
            if not os.path.exists(csv_temp_dir):
                os.makedirs(csv_temp_dir)
            # If the section is empty, then skip.
            if input_text_list[input_sid_list.index(_sid)] == "None":
                continue
            df_all.to_csv(os.path.join(csv_temp_dir, f"{_sid}.csv"))
            logger.debug("Batch Process [%s] output: sid:%s, df_shape:%s", progressId, _sid, df_all.shape)
        return progressId, "Done", len(batch_data)
    except Exception:  # pylint: disable=broad-except
        logger.error("Error occured in batch Process [%s]:", progressId)
        logger.error("Keys in batch_data: %s", batch_data.keys())
        logger.error(traceback.format_exc())
        return progressId, "Error occured", 0


def run(config) -> str:
    """ Use multiprocessing to process data. Data are split into batches. 

    Return:
        temp_output_dir
    """
    # Load data
    mimic_cfg = config.coref_data_preprocessing.mimic_cxr
    section_name = config.name_style.mimic_cxr.section_name
    logger.info("Loading data from %s", mimic_cfg.dataset_path)
    data_size, pid_list, sid_list, findings_list, impression_list, pfi_list, fai_list = load_data(mimic_cfg.dataset_path, section_name)

    # We create a batch of records at each time and submit a task
    logger.debug("Main process started.")
    section_list: list[tuple] = []
    multiprocessing_cfg = config.coref_data_preprocessing.mimic_cxr.multiprocessing
    # Sections to be processed
    if multiprocessing_cfg.target_section.findings:
        section_list.append((section_name.FINDINGS, findings_list))
    if multiprocessing_cfg.target_section.impression:
        section_list.append((section_name.IMPRESSION, impression_list))
    if multiprocessing_cfg.target_section.provisional_findings_impression:
        section_list.append((section_name.PFI, pfi_list))
    if multiprocessing_cfg.target_section.findings_and_impression:
        section_list.append((section_name.FAI, fai_list))

    # Loop sections
    for _sectionName, text_list in section_list:
        all_task = []
        # executor = ProcessPoolExecutor(max_workers=multiprocessing_cfg.workers_in_pool)
        # Loop section records with multiprocessing
        with ProcessPoolExecutor(max_workers=multiprocessing_cfg.workers_in_pool) as executor:
            logger.info("Processing section: [%s]", _sectionName)
            for progressId, startIndex in tqdm(enumerate(range(multiprocessing_cfg.data_start_pos, multiprocessing_cfg.data_end_pos, multiprocessing_cfg.batch_size))):
                # Construct batch data
                endIndex = startIndex + multiprocessing_cfg.batch_size if startIndex + multiprocessing_cfg.batch_size < data_size else data_size
                input_text_list = [(text if text else "None") for text in text_list[startIndex:endIndex]]
                input_sid_list = list(sid_list[startIndex:endIndex])
                input_pid_list = list(pid_list[startIndex:endIndex])
                # Submit the task for one batch
                all_task.append(executor.submit(batch_processing, input_text_list, input_sid_list, input_pid_list, _sectionName, progressId, config))
                # We found that the tqdm progress bar will stuck somehow somewhere during this progress, without error raised.
                if progressId % 100 == 0:
                    time.sleep(0.01)
            # When a submitted task finished, the output is received here.
            total_num = len(all_task) * multiprocessing_cfg.batch_size
            total_num = total_num if total_num < data_size else data_size
            with tqdm(total=total_num) as pbar:
                for future in as_completed(all_task):
                    processId, msg, num_processed = future.result()
                    pbar.update(num_processed)
                    logger.debug("Result from batch_processing [%s] : %s", processId, msg)
            executor.shutdown(wait=True, cancel_futures=False)
            logger.info("Done.")
    logger.debug("Main process finished.")
    logger.info("Temp output dir: %s", config.coref_data_preprocessing.mimic_cxr.temp_dir)
    return config.coref_data_preprocessing.mimic_cxr.temp_dir


@hydra.main(version_base=None, config_path=config_path, config_name="coreference_resolution")
def main(config):
    print(OmegaConf.to_yaml(config))
    run(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
