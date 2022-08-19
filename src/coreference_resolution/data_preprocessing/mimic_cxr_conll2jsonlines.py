""" This file is a modified version of fast-coref/src/data-processing/prrocessing-litbank.py """
import collections
import json
import logging
import re
import sys
import os

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


def minimize_split(config, args):
    for subdir_entry in os.scandir(args.input_dir):
        # Ignore temp and hidden dirs
        if FILE_CHECKER.ignore(subdir_entry.name) or subdir_entry.path == config.coref_data_preprocessing.mimic_cxr.temp_dir:
            continue
        output_dir = os.path.join(subdir_entry.path, os.path.basename(args.output_dir))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for file_entry in os.scandir(os.path.join(subdir_entry.path, config.coref_data_preprocessing.mimic_cxr.output.root_dir_name)):
            input_path = file_entry.path
            output_path = os.path.join(
                output_dir, f"{file_entry.name.removesuffix(config.coref_data_preprocessing.mimic_cxr.output.suffix)}.{args.seg_len}.jsonlines"
            )
            minimize_partition(input_path, output_path, args.tokenizer, args.seg_len)
    logger.info("Done.")
    return output_dir


def invoke(config, input_dir):
    sys.argv.append(input_dir)
    args = parse_args()
    output_dir = minimize_split(config, args)
    # We dont output the files to ``args.output_dir``. Here, the ``args.output_dir`` folder is empty and safe to delete.
    if os.path.exists(args.output_dir):
        os.removedirs(args.output_dir)

    return output_dir


@hydra.main(version_base=None, config_path=config_path, config_name="coreference_resolution")
def main(config):
    input_dir = "/home/yuxiangliao/PhD/workspace/VSCode_workspace/structured_reporting/output/coref/data/mimic_cxr/"
    invoke(config, input_dir)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter