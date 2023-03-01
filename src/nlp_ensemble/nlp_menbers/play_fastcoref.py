import json
import logging
import os
import time
import traceback

import hydra
import pandas as pd  # import pandas before spacy
from tqdm import tqdm
from omegaconf import OmegaConf
import torch
from transformers import AutoModel, AutoTokenizer

# pylint: disable=import-error,wrong-import-order
from common_utils.file_checker import FileChecker
from common_utils.common_utils import check_and_create_dirs
from common_utils.nlp_utils import align_byIndex_individually_nestedgruop, align_coref_groups_in_conll_format
from model.utils import action_sequences_to_clusters
from model.entity_ranking_model import EntityRankingModel
from inference.tokenize_doc import tokenize_and_segment_doc


logger = logging.getLogger()
module_path = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(os.path.dirname(module_path), "config")
FILE_CHECKER = FileChecker()


def init_coref_model(config):
    model_dir = config.fastcoref_joint.model_dir
    doc_encoder_dir = config.fastcoref_joint.doc_encoder_dir

    # Load model checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(os.path.join(model_dir, "model.pth"), map_location=device)

    # Load model training config
    model_cfg = OmegaConf.create(checkpoint["config"])
    # In case ARCCA cannot use offline mode via terminal command, we use the encoder's local path instead of its name to prevent automatic downloading from huggingface.
    model_cfg.model.doc_encoder.transformer.model_str = doc_encoder_dir
    print("Model config %s", OmegaConf.to_yaml(model_cfg))

    # Load fast-coref model
    model = EntityRankingModel(model_cfg.model, model_cfg.trainer)
    model.load_state_dict(checkpoint["model"], strict=False)

    # Load the encoder
    model.mention_proposer.doc_encoder.lm_encoder = (
        AutoModel.from_pretrained(
            pretrained_model_name_or_path=doc_encoder_dir
        )
    )
    model.mention_proposer.doc_encoder.tokenizer = (
        AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=doc_encoder_dir
        )
    )

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    max_segment_len = model_cfg.model.doc_encoder.transformer.max_segment_len
    subword_tokenizer = model.mention_proposer.doc_encoder.tokenizer

    return model, subword_tokenizer, max_segment_len


@torch.no_grad()
def inference(model, tokenized_doc):
    pred_mentions, mention_scores, gt_actions, pred_actions = model(tokenized_doc)
    return pred_mentions, mention_scores, gt_actions, pred_actions


def resolve_output(tokenized_doc, pred_mentions, pred_actions, ignore_singleton=False) -> list[list[int]]:
    """Returns:
        A list of coreference groups. Each group has one or more coref mentions.
        Each of the mentions consist of a list of indices refering to the corresponding token position.
    """
    idx_clusters = action_sequences_to_clusters(pred_actions, pred_mentions)
    subtoken_map = tokenized_doc["subtoken_map"]  # The elements in subtoken_map is the indices of the input token.
    coref_group_list = []
    for idx_cluster in idx_clusters:
        coref_group = []
        # Ignore singleton
        if ignore_singleton and len(idx_cluster) == 1:
            continue
        for (ment_start, ment_end) in idx_cluster:
            coref_group.append(list(range(subtoken_map[ment_start], subtoken_map[ment_end] + 1)))
        coref_group_list.append(coref_group)
    return coref_group_list


def format_input_tok_same_as_traingset(df_base, spacy_nametyle):
    """ 
    Format the input token by using the same approach as creating training sets for fast-coref models 
    For example, we skipped all whitespces like "\n", "\n " and " ".
    """
    sent_tok_2d_list: list[list[str]] = []
    sentence_id = 0
    index_map = []  # map the input token index to the spacy token index.
    curr_spacy_index = 0
    while True:
        token_list2: list[str] = []
        df_sentence = df_base[df_base.loc[:, spacy_nametyle.sentence_group] == sentence_id].reset_index()
        if df_sentence.empty:
            break
        for _idx, data in df_sentence.iterrows():
            # Skip all whitespces like "\n", "\n " and " ".
            curr_spacy_index += 1
            if str(data[spacy_nametyle.token]).strip() == "":
                continue
            conllToken = data[spacy_nametyle.token]
            token_list2.append(conllToken)
            index_map.append(curr_spacy_index-1)

        sent_tok_2d_list.append(token_list2)
        sentence_id += 1
    return sent_tok_2d_list, index_map


def align_to_spacy(tok_indices_in_spacy, label_aligned_to_input, input_tok_list, spacy_tok_list):
    aligned_to_spacy_tok = [-1] * len(spacy_tok_list)
    for idx_in_spacy, label, input_tok in zip(tok_indices_in_spacy, label_aligned_to_input, input_tok_list):
        assert input_tok == spacy_tok_list[idx_in_spacy]
        aligned_to_spacy_tok[idx_in_spacy] = label
    return aligned_to_spacy_tok


def run(config, use_sections: list, model: EntityRankingModel, subword_tokenizer, max_segment_len):
    """
    Args:
        model: The fast-coref model loaded from checkpoint
        use_section: A list of section names to be processed, which are also the folder names.
    """
    spacy_nametyle = config.name_style.spacy.column_name
    fastcoref_joint_nametyle = config.name_style.fastcoref_joint.column_name

    processed_record_num_per_section = {}

    for section_name in use_sections:
        logger.info("Processing section: %s", section_name)
        processed_record_num_per_section[section_name] = {"Succeeded": 0, "Failed": 0}

        # CSV files base dir for each sections
        csv_file_dir = os.path.join(config.spacy.output_dir, section_name)

        for file_entry in tqdm(os.scandir(csv_file_dir)):
            if FILE_CHECKER.ignore(file_entry.name):
                continue

            # Load preprocessed tokens from csv files.
            df_base = pd.read_csv(file_entry.path, index_col=0)
            # For example, we skipped all whitespces like "\n", "\n " and " ". This will affect the model prediction.
            # If use spacy output directly, the f1 is 60. If we strip the input, the f1 is 66.
            sent_tok_2d_list, tok_indices_in_spacy = format_input_tok_same_as_traingset(df_base, spacy_nametyle)

            # Using longformer tokenizer to generate subtokens and form the input data.
            tokenized_doc = tokenize_and_segment_doc(sent_tok_2d_list, subword_tokenizer, max_segment_len=max_segment_len)

            try:
                # Get model output
                pred_mentions, mention_scores, gt_actions, pred_actions = inference(model, tokenized_doc)

                # Resolve model output
                coref_group_list = resolve_output(tokenized_doc, pred_mentions, pred_actions, ignore_singleton=True)

                # To dataframe
                spacy_tok_list = df_base.loc[:, spacy_nametyle.token].to_list()
                spacy_sentGroup_list = df_base.loc[:, spacy_nametyle.sentence_group].to_list()
                input_tok_list = [tok for sent in sent_tok_2d_list for tok in sent]

                coref_group_aligned_to_input_tok = align_byIndex_individually_nestedgruop(len(input_tok_list), coref_group_list)
                coref_group_aligned_to_spacy_tok = align_to_spacy(tok_indices_in_spacy, coref_group_aligned_to_input_tok, input_tok_list, spacy_tok_list)

                coref_group_conll_aligned_to_input_tok = align_coref_groups_in_conll_format(len(input_tok_list), coref_group_list)
                coref_group_conll_aligned_to_spacy_tok = align_to_spacy(tok_indices_in_spacy, coref_group_conll_aligned_to_input_tok, input_tok_list, spacy_tok_list)

                df_fastcoref_joint = pd.DataFrame(
                    {
                        fastcoref_joint_nametyle["token_from_spacy"]: [str(i) for i in spacy_tok_list],
                        fastcoref_joint_nametyle["sentence_group"]: [int(i) for i in spacy_sentGroup_list],
                        fastcoref_joint_nametyle["coref_group"]: [str(i) for i in coref_group_aligned_to_spacy_tok],
                        fastcoref_joint_nametyle["coref_group_conll"]: [str(i) for i in coref_group_conll_aligned_to_spacy_tok],
                    },
                )

                # Overwrite csv
                output_dir = os.path.join(config.fastcoref_joint.output_dir, section_name)
                check_and_create_dirs(output_dir)

                df_fastcoref_joint.to_csv(os.path.join(output_dir, file_entry.name))

                processed_record_num_per_section[section_name]["Succeeded"] += 1

            except Exception:
                # If error occurred, skip this record.
                logger.error("Failed when running the model")
                logger.error("Section: %s, file: %s, path: %s", section_name, file_entry.name, file_entry.path)
                logger.error(traceback.format_exc())
                processed_record_num_per_section[section_name]["Failed"] += 1
                with open(config.fastcoref_joint.unfinished_records_path, "a", encoding="UTF-8") as f:
                    f.write(f"{file_entry.path}\n")

    return processed_record_num_per_section


@hydra.main(version_base=None, config_path=config_path, config_name="nlp_ensemble")
def main(config):
    print(OmegaConf.to_yaml(config))

    use_sections = [_sectionName for _sectionName, _trueORfalse in config.output.section.items() if _trueORfalse]

    startTime = time.time()

    # Init fast-coref-joint
    logger.info("Initializing fast-coref-joint")
    model, subword_tokenizer, max_segment_len = init_coref_model(config)

    # The main processing method.
    processed_record_num_per_section = run(config, use_sections, model, subword_tokenizer, max_segment_len)

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


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
