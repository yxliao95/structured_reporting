import json
import logging
import os
import sys
import time
import re
import ast

import hydra
from omegaconf import OmegaConf

# pylint: disable=import-error,wrong-import-order
from common_utils.data_loader_utils import load_i2b2
from common_utils.common_utils import check_and_remove_file, remove_dirs
from nlp_ensemble.nlp_processor.spacy_process import init_spacy
import nlp_ensemble.nlp_menbers.play_spacy as play_spacy
import nlp_ensemble.nlp_menbers.play_corenlp as play_corenlp
import nlp_ensemble.nlp_menbers.play_fastcoref as play_fastcoref

logger = logging.getLogger()
module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")


def run_spacy(config, id_list, section_list):
    batch_process_cfg = config.batch_process
    startTime = time.time()

    # Init spacy
    logger.info("Initializing spaCy")
    model, enable_component, disable_component = init_spacy(config.nlp_properties.spacy)

    # The main processing method.
    log_not_empty_records = play_spacy.run(config, id_list, section_list)

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
            "Time cost": f"{time.time() - startTime:.2f}s"
        }
        f.write(json.dumps(log_out, indent=2))
        f.write("\n\n")


def run_corenlp(config, id_list, section_list):
    batch_process_cfg = config.batch_process
    corenlp_cfg = config.nlp_properties.corenlp
    startTime = time.time()

    # Init CoreNLP server config
    properties_list: list[tuple[str, list]] = []  # (coref_name, coref_properties)
    for coref_server_name, is_required in config.corenlp.use_server_properties.items():
        if is_required:
            properties_list.append((coref_server_name, corenlp_cfg.server_properties.get(coref_server_name)))

    for coref_server_name, server_properties_cfg in properties_list:
        logger.info("Starting server: %s", coref_server_name)
        client = play_corenlp.start_server(corenlp_cfg.server, server_properties_cfg)

        # The main processing method.
        log_not_empty_records = play_corenlp.run(config, coref_server_name, id_list, section_list)

        # Shutdown CoreNLP server
        logger.info("Shutdown server: %s", coref_server_name)
        client.stop()

        # Log runtime information
        with open(os.path.join(config.corenlp.output_dir, config.output.log_file), "a", encoding="UTF-8") as f:
            log_out = {
                "Using": {
                    "Library": "CoreNLP",
                    "Server name": coref_server_name,
                    "Properties": OmegaConf.to_object(server_properties_cfg),
                },
                "Number of input records": batch_process_cfg.data_end_pos - batch_process_cfg.data_start_pos,
                "Number of not empty records": log_not_empty_records,
                "Time cost": f"{time.time() - startTime:.2f}s"
            }
            f.write(json.dumps(log_out, indent=2))
            f.write("\n\n")


def rerun_corenlp_for_unfinished_records(config, id_list, section_list):
    startTime = time.time()

    # Rewrite the setting
    config.corenlp.multiprocess_workers = config.corenlp_for_unfinished_records.multiprocess_workers
    config.nlp_properties.corenlp.server.memory = "4G"
    config.nlp_properties.corenlp.server.threads = 8

    # Read the log file
    with open(config.corenlp_for_unfinished_records.unfinished_records_path, "r", encoding="UTF-8") as f:
        records_log = f.readlines()

    # Remove the log file
    check_and_remove_file(config.corenlp_for_unfinished_records.unfinished_records_path)

    if records_log:
        # Extract from the log file.
        unfinished_record_dict: dict[str, dict[str, list[str]]] = {}  # {server_name: {section_name: [id, ...], ...}, ...}
        for line in records_log:
            res = re.match(r"(.*)-(.*): dict_keys\((.*)\)", line.strip())
            try:
                _section_name = res.group(1)
                _server_name = res.group(2)
                _id_list_str = res.group(3)
                if _server_name not in unfinished_record_dict:
                    unfinished_record_dict[_server_name] = {_section_name: ast.literal_eval(_id_list_str)}
                else:
                    if _section_name not in unfinished_record_dict[_server_name]:
                        unfinished_record_dict[_server_name][_section_name] = ast.literal_eval(_id_list_str)
                    else:
                        unfinished_record_dict[_server_name][_section_name] += ast.literal_eval(_id_list_str)
            except Exception:
                print("Can not correctly resolve %s. The content should be like: all-scoref: dict_keys(['clinical-1', ...]", config.corenlp_for_unfinished_records.unfinished_records_path)
                raise

        corenlp_cfg = config.nlp_properties.corenlp
        for coref_server_name, section_list_dict in unfinished_record_dict.items():
            # Start server
            logger.info("Starting server: %s", coref_server_name)
            client = play_corenlp.start_server(corenlp_cfg.server, corenlp_cfg.server_properties[coref_server_name])

            # Process only one section per time.
            for _sectionName, _sidList in section_list_dict.items():
                new_id_list = _sidList
                # section_list is a list of tuple, this line is using both the key names' and values' indices to find the target values.
                _text_list = [section_list[[sectName for sectName, _ in section_list].index(_sectionName)][1][id_list.index(sid)] for sid in _sidList]
                new_section_list = [(_sectionName, _text_list)]

                # Modify config
                config.batch_process.data_end_pos = len(new_id_list)

                # Run
                log_not_empty_records = play_corenlp.run(config, coref_server_name, new_id_list, new_section_list)

                # Log runtime information
                with open(os.path.join(config.corenlp.output_dir, config.output.log_file), "a", encoding="UTF-8") as f:
                    log_out = {
                        "Using (Re-run)": {
                            "Library": "CoreNLP",
                            "Which server": coref_server_name,
                        },
                        "Which section": _sectionName,
                        "Number of unfinished records re-processed": config.batch_process.data_end_pos - config.batch_process.data_start_pos,
                        "The sid of the unfinished records re-processed": str(new_id_list),
                        "Number of not empty records within": log_not_empty_records,
                        "Time cost": f"{time.time() - startTime:.2f}s"
                    }
                    f.write(json.dumps(log_out, indent=2))
                    f.write("\n\n")

            # Shutdown CoreNLP server
            logger.info("Shutdown server: %s", coref_server_name)
            client.stop()
    else:
        logger.info("No unfinished records detected. Done.")


def run_fastcoref_joint(config):

    use_sections = [_sectionName for _sectionName, _trueORfalse in config.output.section.items() if _trueORfalse]
    startTime = time.time()

    # Init fast-coref-joint
    logger.info("Initializing fast-coref-joint model")
    model, subword_tokenizer, max_segment_len = play_fastcoref.init_coref_model(config)

    # The main processing method.
    processed_record_num_per_section = play_fastcoref.run(config, use_sections, model, subword_tokenizer, max_segment_len)

    # Log runtime information
    with open(os.path.join(config.fastcoref_joint.output_dir, config.output.log_file), "w", encoding="UTF-8") as f:
        log_out = {
            "Using": {
                "Library": "fast-coref",
                "Coref model": os.path.join(config.fastcoref_joint.model_dir, "model.pth"),
                "Document encoder": config.fastcoref_joint.doc_encoder_dir,
                "Base tokenizer": "SpaCy",
                "Subword tokenizer": config.fastcoref_joint.doc_encoder_dir,
                "Max segmentation length": max_segment_len,
            },
            "Number of processed records": processed_record_num_per_section,
            "Time cost": f"{time.time() - startTime:.2f}s"
        }
        f.write(json.dumps(log_out, indent=2))
        f.write("\n\n")


@hydra.main(version_base=None, config_path=config_path, config_name="nlp_ensemble")
def main(config):
    print(OmegaConf.to_yaml(config))

    startTime = time.time()

    if config.clear_history:
        logger.info("Deleted history dirs: %s", config.output.base_dir)
        remove_dirs(config.output.base_dir)

    # Load data
    json_name_cfg = config.name_style.i2b2.json
    input_path = config.input.path
    logger.info("Loading i2b2 data from %s", input_path)
    id_list, section_list = load_i2b2(input_path, json_name_cfg)

    if config.spacy.activate:
        logger.info("*" * 60)
        logger.info("SpaCy activated")
        run_spacy(config, id_list, section_list)

    if config.corenlp.activate:
        logger.info("*" * 60)
        logger.info("CoreNLP activated")
        run_corenlp(config, id_list, section_list)

    # The stop signal is the unexisting of `config.corenlp_for_unfinished_records.unfinished_records_path`
    # (We delete it at the beginning and no such file is newly generated.)
    curr_iter = 0
    while config.corenlp_for_unfinished_records.activate and os.path.exists(config.corenlp_for_unfinished_records.unfinished_records_path):
        if curr_iter == config.corenlp_for_unfinished_records.max_iteration_num:
            logger.warning("CoreNLP re-activated [%s] times and some unexpected errors are remaining. Force stopped.", curr_iter)
            break
        else:
            logger.info("*" * 60)
            logger.info("CoreNLP re-activated for unfinished records, curr_iter_num: %s", curr_iter)
            rerun_corenlp_for_unfinished_records(config, id_list, section_list)
            curr_iter += 1

    if config.fastcoref_joint.activate:
        logger.info("*" * 60)
        logger.info("fast-coref-joint model activated")
        run_fastcoref_joint(config)

    logger.info("*" * 60)
    logger.info("Total time cost: %.2f", time.time() - startTime)
    logger.info("Execution info: %s", config.output.log_file)


if __name__ == "__main__":
    sys.argv.append("nlp_ensemble@_global_=i2b2")
    # It seems that using multiple workers will cause unexpected errors like failure to do dependency parsing sometime.
    sys.argv.append("corenlp.multiprocess_workers=1")
    main()  # pylint: disable=no-value-for-parameter
