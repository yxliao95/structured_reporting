from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
from multiprocessing import Event, Pipe
import os
import time
import traceback

import hydra
import pandas as pd  # import pandas before spacy
import spacy
from tqdm import tqdm
from omegaconf import OmegaConf

# pylint: disable=import-error,wrong-import-order
from common_utils.data_loader_utils import load_mimic_cxr_bySection
from common_utils.common_utils import check_and_create_dirs
from common_utils.nlp_utils import align, getTokenOffset
from nlp_ensemble.nlp_processor.spacy_process import SpacyProcess, init_spacy

logger = logging.getLogger()
module_path = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(os.path.dirname(module_path), "config")
START_EVENT = Event()


def batch_processing(input_text_list: list, input_id_list: list, sectionName: str, progressId: int, config) -> tuple[int, str, int]:
    """ The task of multiprocessing.
    Args:
        input_text_list: A batch of list of the corresponding section text
        input_id_list: A batch of id list
        sectionName: The name of the section to which the input_text_list belongs.
        progressId: Start from 0
        config: hydra config

    Return:
        progressId: The ``progressId`` of this process
        msg: The message passed to the main process.
        num: The number of records processed in this batch.
    """
    START_EVENT.wait()
    batch_data = dict.fromkeys(input_id_list, None)  # Dict: Key = sid, Value = {df_[name]:DataFrame, ...}
    try:
        logger.debug("Batch Process [%s] started: %s", progressId, input_id_list)
        # We create three sub processors as below, as each of them take 3s to process a batch of 10 records
        # 1. For Spacy
        spacy_outPipe, spacy_inPipe = Pipe(False)
        spacy_process = SpacyProcess(progressId, spacy_inPipe, input_text_list, input_id_list)
        spacy_process.start()

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
                    spacy_nametyle["token"]: [str(tok.text) for tok in doc],
                    spacy_nametyle["token_offset"]: [int(i) for i in getTokenOffset(doc.text, doc)],
                    spacy_nametyle["sentence_group"]: [int(i) for i in align(len(doc), doc.sents)],
                    spacy_nametyle["noun_chunk"]: [int(i) for i in align(len(doc), doc.noun_chunks)],
                    spacy_nametyle["lemma"]: [str(tok.lemma_) for tok in doc],
                    spacy_nametyle["pos_core"]: [f"[{tok.pos_}]{spacy.explain(tok.pos_)}" for tok in doc],
                    spacy_nametyle["pos_feature"]: [f"[{tok.tag_}]{spacy.explain(tok.tag_)}" for tok in doc],
                    spacy_nametyle["dependency_tag"]: [f"[{tok.dep_}]{spacy.explain(tok.dep_)}" for tok in doc],
                    spacy_nametyle["dependency_head"]: [f"{tok.head.text}|{tok.head.i}" for tok in doc],
                    spacy_nametyle["dependency_children"]: [[f"{child.text}|{child.i}" for child in tok.children] for tok in doc],
                    spacy_nametyle["morphology"]: [str(tok.morph) for tok in doc],
                    spacy_nametyle["is_alpha"]: [bool(tok.is_alpha) for tok in doc],
                    spacy_nametyle["is_stop"]: [bool(tok.is_stop) for tok in doc],
                    spacy_nametyle["is_pronoun"]: [bool(tok.pos_ == "PRON") for tok in doc],
                    spacy_nametyle["trailing_space"]: [bool(tok.whitespace_) for tok in doc],
                }
            )
            batch_data[sid] = {"df_spacy": df_spacy}  # Must create a dict here rather than when define the variable.
        time2 = time.time()

        logger.debug(
            "Batch Process [%s] finished processing Spacy [%s], cost: %ss", progressId, spacy_prcess_id, time2-time1
        )

        # Write to the disk
        for _id, _df in batch_data.items():
            df_all = _df["df_spacy"]

            # Output csv for later usage
            output_dir = os.path.join(config.spacy.output_dir, sectionName)
            check_and_create_dirs(output_dir)

            df_all.to_csv(os.path.join(output_dir, f"{_id}.csv"))

            logger.debug("Batch Process [%s] output: sid:%s, df_shape:%s", progressId, _id, df_all.shape)
        return progressId, "Done", len(batch_data)

    except Exception:  # pylint: disable=broad-except
        logger.error("Error occured in batch Process [%s]:", progressId)
        logger.error("Keys in batch_data: %s", batch_data.keys())
        logger.error(traceback.format_exc())
        return progressId, "Error occured", 0


def run(config, id_list, section_list):
    batch_process_cfg = config.batch_process
    log_not_empty_records = {}

    for _sectionName, text_list in section_list:
        all_task = []

        with ProcessPoolExecutor(max_workers=config.spacy.multiprocess_workers) as executor:
            logger.info("Processing section: [%s]", _sectionName)
            progressId = 0
            input_text_list, input_id_list = [], []
            not_empty_num = 0

            # Submit tasks
            for currentIdx in tqdm(range(batch_process_cfg.data_start_pos, batch_process_cfg.data_end_pos)):
                # Construct batch data and skip empty record
                if text_list[currentIdx]:
                    input_text_list.append(text_list[currentIdx])
                    input_id_list.append(id_list[currentIdx])
                # Submit task for this batch data
                if len(input_text_list) == batch_process_cfg.batch_size or (currentIdx + 1 == batch_process_cfg.data_end_pos and input_text_list):
                    all_task.append(executor.submit(batch_processing, input_text_list, input_id_list, _sectionName, progressId, config))
                    progressId += 1
                    not_empty_num += len(input_text_list)
                    input_text_list, input_id_list = [], []

            log_not_empty_records[_sectionName] = not_empty_num

            # Notify tasks to start
            START_EVENT.set()

            # When a submitted task finished, the output is received here.
            if all_task:
                with tqdm(total=not_empty_num) as pbar:
                    for future in as_completed(all_task):
                        processId, msg, num_processed = future.result()
                        pbar.update(num_processed)
                        logger.debug("Result from batch_processing [%s] : %s", processId, msg)
                logger.info("Done.")
            else:
                logger.info("All empty. Skipped.")

            executor.shutdown(wait=True, cancel_futures=False)
            START_EVENT.clear()

    return log_not_empty_records


@hydra.main(version_base=None, config_path=config_path, config_name="nlp_ensemble")
def main(config):
    print(OmegaConf.to_yaml(config))

    section_name_cfg = config.name_style.mimic_cxr.section_name
    batch_process_cfg = config.batch_process
    output_section_cfg = config.output.section

    startTime = time.time()

    # Load data
    input_path = config.input.path
    logger.info("Loading mimic-cxr section data from %s", input_path)
    data_size, pid_list, sid_list, section_list = load_mimic_cxr_bySection(input_path, output_section_cfg, section_name_cfg)

    # Init spacy
    logger.info("Initializing spaCy")
    model, enable_component, disable_component = init_spacy(config.nlp_properties.spacy)

    # The main processing method.
    log_not_empty_records = run(config, sid_list, section_list)

    # Log runtime information
    with open(os.path.join(config.spacy.output_dir, config.output.log_file), "w", encoding="UTF-8") as f:
        log_out = {
            "Using": {
                "Library": "spaCy",
                "Model": model,
                "Pipeline enable": str(enable_component),
                "Pipeline disable": str(disable_component)
            },
            "Number of input records": batch_process_cfg.data_end_pos - batch_process_cfg.data_start_pos,
            "Number of not empty records": log_not_empty_records,
            "Time cost": time.time() - startTime
        }
        f.write(json.dumps(log_out, indent=2))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
