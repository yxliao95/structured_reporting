import json
import logging
import os
import time

import hydra
from omegaconf import OmegaConf

# pylint: disable=import-error,wrong-import-order
from common_utils.ensemble_utils import load_data_bySection
from common_utils.common_utils import remove_all
from nlp_processor.spacy_process import init_spacy
import play_spacy
import play_corenlp

logger = logging.getLogger()
module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")


def run_corenlp(config, sid_list, section_list):
    batch_process_cfg = config.batch_process
    corenlp_cfg = config.nlp_properties.corenlp
    startTime = time.time()

    # Init CoreNLP server config
    properties_list: list[tuple[str, list]] = []  # (coref_name, coref_properties)
    if config.corenlp.require_coref:
        for coref_component_name, is_required in config.corenlp.use_coref_pipeline.items():
            if is_required:
                properties_list.append((coref_component_name, corenlp_cfg.server_properties.get(coref_component_name)))
    else:
        properties_list.append((None, corenlp_cfg.server_properties.default))

    for coref_component_name, server_properties_cfg in properties_list:
        logger.info("Starting server: %s", coref_component_name)
        client = play_corenlp.start_server(corenlp_cfg.server, server_properties_cfg)

        # The main processing method.
        log_not_empty_records = play_corenlp.run(config, coref_component_name, sid_list, section_list)

        # Shutdown CoreNLP server
        logger.info("Shutdown server: %s", coref_component_name)
        client.stop()

        # Log runtime information
        with open(config.output.log_path, "a", encoding="UTF-8") as f:
            log_out = {
                "Using": {
                    "Library": "CoreNLP",
                    "Properties": OmegaConf.to_object(server_properties_cfg),
                    "Output": "All annotators' results" if coref_component_name is None or config.corenlp.default_pipeline_provider == coref_component_name else "Only the last coref annotator's results"
                },
                "Number of input records": batch_process_cfg.data_end_pos - batch_process_cfg.data_start_pos,
                "Number of not empty records": log_not_empty_records,
                "Time cost": f"{time.time() - startTime:.2f}s"
            }
            f.write(json.dumps(log_out, indent=2))
            f.write("\n\n")


def run_spacy(config, sid_list, section_list):
    batch_process_cfg = config.batch_process
    startTime = time.time()

    # Init spacy
    logger.info("Initializing spaCy")
    model, enable_component, disable_component = init_spacy(config.nlp_properties.spacy)

    # The main processing method.
    log_not_empty_records = play_spacy.run(config, sid_list, section_list)

    # Log runtime information
    with open(config.output.log_path, "w", encoding="UTF-8") as f:
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


@hydra.main(version_base=None, config_path=config_path, config_name="nlp_ensemble")
def main(config):
    print(OmegaConf.to_yaml(config))

    startTime = time.time()

    if config.clear_history:
        remove_all(config.output.dir)

    # Load data
    section_name_cfg = config.name_style.mimic_cxr.section_name
    output_section_cfg = config.output.section
    input_path = config.input.path
    logger.info("Loading mimic-cxr section data from %s", input_path)
    data_size, pid_list, sid_list, section_list = load_data_bySection(input_path, output_section_cfg, section_name_cfg)

    logger.info("*" * 60)
    logger.info("SpaCy activated")
    run_spacy(config, sid_list, section_list)

    logger.info("*" * 60)
    logger.info("CoreNLP activated")
    run_corenlp(config, sid_list, section_list)

    logger.info("*" * 60)
    logger.info("Total time cost: %.2f", time.time() - startTime)
    logger.info("Execution info: %s", config.output.log_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
