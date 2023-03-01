""" This file is a modified version of fast-coref/src/data-processing/prrocessing-litbank.py """
import collections
import json
import logging
import re
import sys
import os
import shutil

import hydra
# pylint: disable=import-error
from coref_utils import conll
from data_processing.utils import (
    split_into_segments,
    parse_args,
    normalize_word,
)
from data_processing.process_ontonotes import OntoNotesDocumentState
from common_utils.file_checker import FileChecker
from common_utils.common_utils import check_and_create_dirs, check_and_remove_dirs


logger = logging.getLogger()
pkg_path = os.path.dirname(__file__)
coref_module_path = os.path.dirname(pkg_path)
config_path = os.path.join(os.path.dirname(coref_module_path), "config")

FILE_CHECKER = FileChecker()


class DocumentState(OntoNotesDocumentState):
    def __init__(self, key):
        super().__init__(key)
        self.clusters = collections.defaultdict(list)

    def finalize(self):
        self.final_processing()
        return {
            "doc_key": self.doc_key,
            "sentences": self.segments,
            "clusters": self.merged_clusters,
            "sentence_map": self.sentence_map,
            "subtoken_map": self.subtoken_map,
        }


def get_document(document_lines, tokenizer, segment_len):
    document_state = DocumentState(document_lines[0])
    word_idx = -1
    for line in document_lines[1]:
        row = line.split()
        sentence_end = len(row) == 0
        if not sentence_end:
            assert len(row) >= 12
            if len(row) == 12:
                row.append("-")
            word_idx += 1
            word = normalize_word(row[3])
            subtokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            document_state.tokens.append(word)
            document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]  # pylint: disable=no-member
            for sidx, subtoken in enumerate(subtokens):
                document_state.subtokens.append(subtoken)
                info = None if sidx != 0 else (row + [len(subtokens)])
                document_state.info.append(info)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(word_idx)
        else:
            document_state.sentence_end[-1] = True

    split_into_segments(
        document_state, segment_len, document_state.sentence_end, document_state.token_end,  # pylint: disable=no-member
    )
    document = document_state.finalize()
    return document


def minimize_partition(input_path, output_path, tokenizer, seg_len):
    count = 0
    logger.debug("Minimizing %s", input_path)
    documents = []
    with open(input_path, "r", encoding="UTF-8") as input_file:
        for line in input_file.readlines():
            begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
            if begin_document_match:
                doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
                documents.append((doc_key, []))
            elif line.startswith("#end document"):
                continue
            else:
                documents[-1][1].append(line)
    with open(output_path, "w", encoding="UTF-8") as output_file:
        for document_lines in documents:
            document = get_document(document_lines, tokenizer, seg_len)
            output_file.write(json.dumps(document))
            output_file.write("\n")
            count += 1
    logger.info("Wrote %s documents to %s", count, output_path)
    return f"Wrote {count} documents to {output_path}"


def minimize_split(config, args):
    output_dir_longformer = os.path.join(config.output.base_dir, os.path.basename(args.output_dir))
    check_and_create_dirs(output_dir_longformer)
    output_dir_conll = os.path.join(config.output.base_dir, "conll")
    check_and_create_dirs(output_dir_conll)

    log_msg = []
    for element in config.longformer.source:  # [{'train': 'train_dev_3k'}, {'dev': 'train_dev_3k'}, {'test': 'test_only'}]
        for key, val in element.items():
            # Copy to conll dir
            input_path = os.path.join(args.input_dir, config.data_split.get(val).dir_name, f"{key}{config.output.suffix}")
            shutil.copyfile(input_path, os.path.join(output_dir_conll, f"{key}{config.output.suffix}"))

            # Generate to longformer dir
            output_path = os.path.join(output_dir_longformer, f"{key}.{args.seg_len}.jsonlines")
            msg = minimize_partition(input_path, output_path, args.tokenizer, args.seg_len)
            log_msg.append(msg)
    return log_msg


def invoke(config):
    # Remove the previous sys.argv for Hydra
    while len(sys.argv) > 1:
        sys.argv.pop()
    conll_base_dir = os.path.join(config.output.base_dir, config.output.conll_dir_name)
    sys.argv.append(conll_base_dir)
    args = parse_args()
    log_msg = minimize_split(config, args)
    return log_msg
