from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
from multiprocessing import Event, Pipe
import os
import time
import traceback

import hydra
import pandas as pd  # import pandas before spacy
from tqdm import tqdm
from omegaconf import OmegaConf
from stanza.server import CoreNLPClient

# pylint: disable=import-error,wrong-import-order
from common_utils.ensemble_utils import load_data_bySection
from common_utils.nlp_utils import align_byIndex_individually_nestedgruop, align_byIndex_individually_withData_dictInList, align_coref_groups_in_conll_format, align_byIndex_individually_withData_noOverlap
from nlp_processor.corenlp_process import CorenlpUrlProcess, formatCorenlpDocument

logger = logging.getLogger()
module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")
START_EVENT = Event()

### utils ###


def start_server(client_cfg, server_properties):
    """ Remember to invoke ``client.stop()`` to close the server. """
    prop = {'annotators': server_properties.annotators}
    logger.debug("Server properties: %s", prop)
    if server_properties.get('coref_algorithm', None) is not None:
        prop['coref.algorithm'] = server_properties.coref_algorithm
    client = CoreNLPClient(memory=client_cfg.memory, threads=client_cfg.threads, endpoint=client_cfg.endpoint,
                           be_quiet=client_cfg.be_quiet, timeout=client_cfg.timeout, output_format=client_cfg.outputFormat, properties=prop)
    client.start()
    client.ensure_alive()
    return client


######


def batch_processing(input_text_list: list, input_sid_list: list, sectionName: str, progressId: int, config, coref_server_name) -> tuple[int, str, int]:
    """ The task of multiprocessing.
    Args:
        input_text_list: A batch of list of the corresponding section text
        input_sid_list: A batch of sid list
        sectionName: The name of the section to which the input_text_list belongs.
        progressId: Start from 0
        config: hydra config
        coref_server_name: scoref, ncoref, dcoref. It is used to locate the config.name_style.corenlp.column_name.

    Return:
        progressId: The ``progressId`` of this process
        msg: The message passed to the main process.
        num: The number of records processed in this batch.
    """

    START_EVENT.wait()
    batch_data = dict.fromkeys(input_sid_list, None)  # Dict: Key = sid, Value = {df_[name]:DataFrame, ...}
    try:
        request_url = config.nlp_properties.corenlp.request_url
        spacy_nametyle = config.name_style.spacy.column_name
        corenlp_nametyle = config.name_style.corenlp.column_name

        logger.debug("Batch Process [%s] started: %s", progressId, input_sid_list)
        # 3. For CoreNLP
        corenlp_outPipe, corenlp_inPipe = Pipe(False)
        corenlp_process = CorenlpUrlProcess(request_url, progressId, corenlp_inPipe, input_text_list, input_sid_list)
        corenlp_process.start()

        # 1. Load previous output (mainly the spacy output)
        logger.debug("Loading previous output from disk.")
        for sid in input_sid_list:
            try:
                file_path = os.path.join(config.output.dir, sectionName, f"{sid}.csv")
                df_from_disk = pd.read_csv(file_path)
                batch_data[sid] = {"df_from_disk": df_from_disk}
            except Exception:
                logger.error("Failed when reading the csv file")
                logger.error("Section: %s, coref-component: %s, sid: %s", sectionName, coref_server_name, sid)
                logger.error(traceback.format_exc())
                raise

        # 3. CoreNLP
        corenlp_output = corenlp_outPipe.recv()
        corenlp_processId = corenlp_output["processId"]
        corenlp_dict = corenlp_output["data"]
        logger.debug("Batch Process [%s] received CoreNLP [%s]", progressId, corenlp_processId)
        time5 = time.time()
        for sid in input_sid_list:
            try:
                corenlp_json = json.loads(corenlp_dict[sid])
                df_base = batch_data[sid]["df_from_disk"]
                tokenOffset_base = df_base.loc[:, spacy_nametyle["token_offset"]].tolist()
                referTo_spacy, tokenTotalNum, corefMentionInfo, corefGroupInfo, dependenceInfo, depPlusInfo, depPlusPlusInfo = formatCorenlpDocument(tokenOffset_base, corenlp_json)
                # Align each item to coreNLP token first, then algin the whole df to spacy df by index mapping (referTo_spacy).
                df_corenlp_rowsNum = tokenTotalNum

                df_corenlp = None
                if coref_server_name is None or config.corenlp.default_server_properties == coref_server_name:
                    df_corenlp = pd.DataFrame(
                        {
                            corenlp_nametyle["token"]: [str(token["originalText"]) for sentence in corenlp_json["sentences"] for token in sentence["tokens"]],
                            corenlp_nametyle["pos"]: [str(token["pos"]) for sentence in corenlp_json["sentences"] for token in sentence["tokens"]],
                            corenlp_nametyle["lemma"]: [str(token["lemma"]) for sentence in corenlp_json["sentences"] for token in sentence["tokens"]],
                            corenlp_nametyle["dependency"]: [str(i) for i in align_byIndex_individually_withData_dictInList(df_corenlp_rowsNum, dependenceInfo)],
                            corenlp_nametyle["dependency+"]: [str(i) for i in align_byIndex_individually_withData_dictInList(df_corenlp_rowsNum, depPlusInfo)],
                            corenlp_nametyle["dependency++"]: [str(i) for i in align_byIndex_individually_withData_dictInList(df_corenlp_rowsNum, depPlusPlusInfo)],
                        },
                        index=referTo_spacy,
                    )

                df_corenlp_coref = None
                if config.corenlp.require_coref:
                    df_corenlp_coref = pd.DataFrame(
                        {
                            corenlp_nametyle[coref_server_name + "_mention"]: [str(i) for i in align_byIndex_individually_withData_noOverlap(df_corenlp_rowsNum, corefMentionInfo)],
                            corenlp_nametyle[coref_server_name + "_group"]: [str(i) for i in align_byIndex_individually_nestedgruop(df_corenlp_rowsNum, corefGroupInfo)],
                            corenlp_nametyle[coref_server_name + "_group_conll"]: [str(i) for i in align_coref_groups_in_conll_format(df_corenlp_rowsNum, corefGroupInfo)],
                        },
                        index=referTo_spacy,
                    )

                if df_corenlp is not None and df_corenlp_coref is not None:
                    df_corenlp = df_corenlp.join(df_corenlp_coref)
                elif df_corenlp_coref is not None:
                    df_corenlp = df_corenlp_coref

                batch_data[sid]["df_corenlp"] = df_corenlp
            except Exception:
                logger.error("Failed when processing the CoreNLP server output")
                logger.error("Section: %s, coref-component: %s, sid: %s", sectionName, coref_server_name, sid)
                logger.error("Server output (corenlp_dict[sid]): %s", corenlp_dict[sid])
                logger.error(traceback.format_exc())
                raise

        time6 = time.time()
        logger.debug("Batch Process [%s] finished processing CoreNLP [%s], cost: %ss", progressId, corenlp_processId, time6-time5)

        # Write to the disk
        for _sid, _df in batch_data.items():
            try:
                df_all = _df["df_from_disk"]
                df_all = df_all.join(_df["df_corenlp"])

                # Output csv for later usage
                output_dir = os.path.join(config.output.dir, sectionName)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                df_all.to_csv(os.path.join(output_dir, f"{_sid}.csv"))
                logger.debug("Batch Process [%s] output: sid:%s, df_shape:%s", progressId, _sid, df_all.shape)
            except Exception:
                logger.error("Failed when saving the DataFrame")
                logger.error("Section: %s, coref-component: %s, sid: %s", sectionName, coref_server_name, _sid)
                logger.error("_df list: %s", _df)
                logger.error(traceback.format_exc())
                raise
        return progressId, "Done", len(batch_data)

    except Exception:  # pylint: disable=broad-except
        logger.error("Error occured in batch Process [%s]:", progressId)
        logger.error("Keys in batch_data: %s", batch_data.keys())
        logger.error(traceback.format_exc())
        with open(config.corenlp.unfinished_records_path, "a", encoding="UTF-8") as f:
            f.write(f"{sectionName}-{coref_server_name}: {batch_data.keys()}\n")
        return progressId, "Error occured", 0


def run(config, coref_server_name, sid_list, section_list):
    batch_process_cfg = config.batch_process
    log_not_empty_records = {}

    for _sectionName, text_list in section_list:
        all_task = []

        with ProcessPoolExecutor(max_workers=config.corenlp.multiprocess_workers) as executor:
            logger.info("Processing section: [%s]", _sectionName)
            progressId = 0
            input_text_list, input_sid_list = [], []
            not_empty_num = 0

            # Submit tasks
            for currentIdx in tqdm(range(batch_process_cfg.data_start_pos, batch_process_cfg.data_end_pos)):
                # Construct batch data and skip empty record
                if text_list[currentIdx]:
                    input_text_list.append(text_list[currentIdx])
                    input_sid_list.append(sid_list[currentIdx])
                # Submit task for this batch data
                if len(input_text_list) == batch_process_cfg.batch_size or (currentIdx + 1 == batch_process_cfg.data_end_pos and input_text_list):
                    all_task.append(executor.submit(batch_processing, input_text_list, input_sid_list, _sectionName, progressId, config, coref_server_name))
                    progressId += 1
                    not_empty_num += len(input_text_list)
                    input_text_list, input_sid_list = [], []

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
    nlp_cfg = config.nlp_properties.corenlp

    startTime = time.time()

    # Load data
    input_path = config.input.path
    logger.info("Loading mimic-cxr section data from %s", input_path)
    data_size, pid_list, sid_list, section_list = load_data_bySection(input_path, output_section_cfg, section_name_cfg)

    # Init CoreNLP server config
    properties_list: list[tuple[str, list]] = []  # (coref_name, coref_properties)
    if config.corenlp.require_coref:
        for coref_component_name, is_required in config.corenlp.use_server_properties.items():
            if is_required:
                properties_list.append((coref_component_name, nlp_cfg.server_properties.get(coref_component_name)))
    else:
        properties_list.append((None, nlp_cfg.server_properties.default))

    for coref_component_name, server_properties_cfg in properties_list:
        logger.info("Starting server: %s", coref_component_name)
        client = start_server(nlp_cfg.server, server_properties_cfg)

        # The main processing method.@hydra.main(version_base=None, config_path=config_path, config_name="nlp_ensemble")
        log_not_empty_records = run(config, coref_component_name, sid_list, section_list)

        # Shutdown CoreNLP server
        logger.info("Shutdown server")
        client.stop()

        # Log runtime information
        with open(config.output.log_path, "a", encoding="UTF-8") as f:
            log_out = {
                "Using": {
                    "Library": "CoreNLP",
                    "Properties": OmegaConf.to_object(server_properties_cfg),
                },
                "Number of input records": batch_process_cfg.data_end_pos - batch_process_cfg.data_start_pos,
                "Number of not empty records": log_not_empty_records,
                "Time cost": time.time() - startTime
            }
            f.write(json.dumps(log_out, indent=2))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
