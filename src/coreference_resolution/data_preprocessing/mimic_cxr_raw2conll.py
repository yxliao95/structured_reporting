from traceback import print_list
from omegaconf import OmegaConf
import pandas as pd
import hydra, logging, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, Pool, Pipe, Lock
from IPython.display import display, HTML
import json, time, random, os
import traceback

logger = logging.getLogger()
pkg_path = os.path.dirname(__file__)
coref_module_path = os.path.dirname(pkg_path)
config_path = os.path.join(os.path.dirname(coref_module_path), "config")


def load_data(file_path, section_name):
    """Load data from ``file_path``."""
    df = pd.read_json(file_path, orient="records", lines=True)
    df = df.sort_values(by=[section_name.PID, section_name.SID])
    pid_list = df.loc[:, section_name.PID].to_list()
    sid_list = df.loc[:, section_name.SID].to_list()
    findings_list = df.loc[:, section_name.FINDINGS].to_list()
    impression_list = df.loc[:, section_name.IMPRESSION].to_list()
    pfi_list = df.loc[:, section_name.PFI].to_list()
    fai_list = df.loc[:, section_name.FAI].to_list()
    return len(sid_list), pid_list, sid_list, findings_list, impression_list, pfi_list, fai_list


def batch_processing(input_text_list, input_sid_list, input_pid_list, sectionName, progressId):
    batch_data = dict.fromkeys(input_sid_list, None)  # Dict: Key = sid, Value = {df_[name]:DataFrame, ...}
    try:
        print(sectionName, progressId)
        print(len(input_text_list), input_sid_list, input_pid_list)
        print()
        return progressId, "Done"
    except Exception as err:
        logger.error(f"Error occured in batch Process [{progressId}]:")
        logger.error(f"Keys in batch_data: {batch_data.keys()}")
        logger.error(traceback.format_exc())
        return progressId, "Error occured"


def invoke(config):
    # Load data
    mimic_cfg = config.coref_data_preprocessing.mimic_cxr
    section_name = config.name_style.mimic_cxr.section_name
    logger.info(f"Loading data from {mimic_cfg.dataset_path}")
    data_size, pid_list, sid_list, findings_list, impression_list, pfi_list, fai_list = load_data(
        mimic_cfg.dataset_path, section_name
    )
    # We create a batch of records at each time and submit a task
    multiprocessing_cfg = config.multiprocessing
    section_list: list[tuple] = []
    if multiprocessing_cfg.target_section.findings:
        section_list.append((section_name.FINDINGS, findings_list))
    if multiprocessing_cfg.target_section.impression:
        section_list.append((section_name.IMPRESSION, impression_list))
    if multiprocessing_cfg.target_section.provisional_findings_impression:
        section_list.append((section_name.PFI, pfi_list))
    if multiprocessing_cfg.target_section.findings_and_impression:
        section_list.append((section_name.FAI, fai_list))
    logger.info(f"Main process started")
    all_task = []
    executor = ProcessPoolExecutor(max_workers=multiprocessing_cfg.workers_in_pool)
    for _sectionName, text_list in section_list:
        for progressId, startIndex in enumerate(
            range(multiprocessing_cfg.data_start_pos, multiprocessing_cfg.data_end_pos, multiprocessing_cfg.batch_size)
        ):
            # Construct batch data
            endIndex = (
                startIndex + multiprocessing_cfg.batch_size
                if startIndex + multiprocessing_cfg.batch_size < data_size
                else data_size
            )
            input_text_list = [(text if text else "None") for text in text_list[startIndex:endIndex]]
            input_sid_list = [i for i in sid_list[startIndex:endIndex]]
            input_pid_list = [i for i in pid_list[startIndex:endIndex]]

            # Submit the task for one batch
            all_task.append(
                executor.submit(
                    batch_processing, input_text_list, input_sid_list, input_pid_list, _sectionName, progressId
                )
            )

        # When a submitted task finished, the output is received here.
        for future in as_completed(all_task):
            processId, msg = future.result()
            logger.info(f"Result from batch_processing [{processId}] : {msg}")
    logger.info(f"Main process finished")


@hydra.main(version_base=None, config_path=config_path, config_name="coreference_resolution")
def main(config):
    print(OmegaConf.to_yaml(config))
    invoke(config)


if __name__ == "__main__":
    main()
