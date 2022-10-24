import os
import shutil
import logging
import re
import random

from tqdm import tqdm

# pylint: disable=import-error,wrong-import-order
from common_utils.coref_utils import get_data_split, get_file_name_prefix, shuffle_list, split_and_shuffle_list, get_porportion_and_name, check_and_make_dir, remove_all, ConllToken
from common_utils.file_checker import FileChecker

logger = logging.getLogger()
FILE_CHECKER = FileChecker()


def clean_and_split_line(sentence: str, debug_doc="", debug_sent=""):
    """ Clean process including:
    remove multiple space symbols;
    remove the appending sapce symbol;
    remove the appending new line symbol.
    """
    # if re.search(" {2,}", sentence):
    #     logger.debug(f"[{debug_doc}] has multiple space symbols at line {debug_sent}")
    # Remove multiple space symbols. See clinical-68 .txt and .chains line 13 as example.
    sentence = re.sub(r" +", " ", sentence)
    sentence = sentence.rstrip("\n")
    # if sentence == "":
    #     logger.debug(f"[{debug_doc}] is empty at line [{debug_sent}]")
    # elif sentence[-1] == " ":
    #     logger.debug(f"[{debug_doc}] has an appending space symbol at line {debug_sent}")
    # Remove the right most space symbol
    sentence = sentence.strip()
    return sentence.split(" ")


def convert_each(doc_file_path, chain_file_path, output_file_path):
    """ Convert the source doc and chain files into the conll file """
    doc_id = get_file_name_prefix(doc_file_path, ".txt")
    BEGIN = f"#begin document ({doc_id}); part 0\n"
    SENTENCE_SEPARATOR = "\n"
    END = "#end document\n"

    # Resolve doc file
    sentenc_list: list[list[ConllToken]] = []
    with open(doc_file_path, "r") as doc:
        for doc_line_id, doc_line in enumerate(doc.readlines()):
            token_list: list[ConllToken] = []
            for doc_token_id, token_str in enumerate(
                clean_and_split_line(doc_line, debug_doc=doc_id, debug_sent=doc_line_id)
            ):
                token_list.append(ConllToken(doc_id, doc_line_id, doc_token_id, token_str))
            sentenc_list.append(token_list)

    # Resolve chain file (coref cluster)
    with open(chain_file_path, "r") as chain:
        for cluster_id, cluster in enumerate(chain.readlines()):
            for coref in cluster.split("||")[0:-1]:  # Drop the last one, which is the type of the coref
                token_range: list[str, str] = coref.split(" ")[-2:]
                start = token_range[0]
                end = token_range[1]
                if start == end:
                    sentId, tokId = start.split(":")
                    mark = f"({cluster_id})"
                    sentenc_list[int(sentId) - 1][int(tokId)].add_coref_label(mark)
                else:
                    sentId, tokId = start.split(":")
                    startMark = f"({cluster_id}"
                    sentenc_list[int(sentId) - 1][int(tokId)].add_coref_label(startMark)

                    sentId, tokId = end.split(":")
                    endMark = f"{cluster_id})"
                    sentenc_list[int(sentId) - 1][int(tokId)].add_coref_label(endMark)

    # Write .conll file
    with open(output_file_path, "a") as out:
        out.write(BEGIN)
        for sent in sentenc_list:
            # Skip empty sentence
            if len(sent) == 1 and sent[0].tokenStr == "":
                continue
            for tok in sent:
                out.write(tok.get_conll_str() + "\n")
            out.write(SENTENCE_SEPARATOR)
        out.write(END)


def convert_all(config, data_split, docs_dir, chains_dir, output_dir_next):
    for split in data_split:
        logger.info('Converting the files for [%s] set:', split["output_name_prefix"])
        # Output file
        output_file_path = os.path.join(output_dir_next, split["output_name_prefix"] + config.output.suffix)
        for _file_name in tqdm(split["file_list"]):
            # Input files
            doc_file_path = os.path.join(docs_dir, _file_name)
            chain_file_path = os.path.join(chains_dir, _file_name + config.input.chain_suffix)
            convert_each(doc_file_path, chain_file_path, output_file_path)
        logger.info("Output: %s", output_file_path)


def convert_to_conll(config, output_dir, docs_dir, chains_dir):
    # Check that the files are matched.
    doc_files = FILE_CHECKER.filter(os.listdir(docs_dir))
    chain_files = FILE_CHECKER.filter(os.listdir(chains_dir))
    
    assert len(doc_files) == len(chain_files)
    logger.info("Detected %s documents in total.", len(chain_files))

    # Do not split the dataset. All the data will be aggregrated into one file with prefix "all"
    if config.data_split.unsplit:
        split_cfg = config.data_split.if_unsplit
        data_split_name, data_split_num = get_porportion_and_name(split_cfg, doc_files)
        doc_files_shuffle = shuffle_list(doc_files, config.shuffle_seed)  # Shuffle the file list

        data_split = get_data_split(doc_files_shuffle, data_split_name, data_split_num)  # Split the files

        output_dir_next = os.path.join(output_dir, config.data_split.if_unsplit.dir_name)
        check_and_make_dir(output_dir_next)

        logger.info("*** Creating an unsplit dataset at: %s", output_dir_next)
        convert_all(config, data_split, docs_dir, chains_dir, output_dir_next)

    # Split the dataset with shuffling the whole soure files
    if config.data_split.random_shuffle:
        split_cfg = config.data_split.if_random_shuffle
        data_split_name, data_split_num = get_porportion_and_name(split_cfg, doc_files)
        doc_files_shuffle = shuffle_list(doc_files, config.shuffle_seed)  # Shuffle the file list

        data_split = get_data_split(doc_files_shuffle, data_split_name, data_split_num)

        output_dir_next = os.path.join(output_dir, config.data_split.if_random_shuffle.dir_name)
        check_and_make_dir(output_dir_next)

        logger.info("*** Creating a shuffle dataset at: %s", output_dir_next)
        convert_all(config, data_split, docs_dir, chains_dir, output_dir_next)

    # Cross-validation. Split the train/test set first and then shuffle the soure files.
    # Make sure the test set of each fold will not overlap.
    if config.data_split.cross_validation:
        folds = int(config.data_split.if_cross_validation.folds)
        logger.info("*** Creating datasets for cross-validaion with %s folds", folds)
        for fold_id in range(folds):
            split_cfg = config.data_split.if_cross_validation
            data_split_name, data_split_num = get_porportion_and_name(split_cfg, doc_files)
            # Change files order for later spliting
            doc_files_fold_shuffle = split_and_shuffle_list(doc_files, data_split_num[-1], fold_id, config.shuffle_seed)
            data_split = get_data_split(doc_files_fold_shuffle, data_split_name, data_split_num)

            output_dir_next = os.path.join(output_dir, str(fold_id))
            check_and_make_dir(output_dir_next)

            logger.info("Creating the fold-%s dataset at: %s", fold_id, output_dir_next)
            convert_all(config, data_split, docs_dir, chains_dir, output_dir_next)


def aggregrate_files(config, temp_dir):
    """ Copy the dataset files from source dirs to the temp dir. """
    docs_dir = os.path.join(temp_dir, config.input.doc_dir_name)
    chains_dir = os.path.join(temp_dir, config.input.chain_dir_name)
    for _path in [docs_dir, chains_dir]:
        os.makedirs(_path, exist_ok=True)
    train_dir_list = [
        config.paths.beth_train,
        config.paths.partners_train,
        config.paths.beth_test,
        config.paths.partners_test,
    ]
    for _dir in train_dir_list:
        shutil.copytree(os.path.join(_dir, config.input.doc_dir_name), docs_dir, dirs_exist_ok=True)
    chain_dir_list = [
        config.paths.beth_train,
        config.paths.partners_train,
        config.paths.beth_test_ground_truth,
        config.paths.partners_test_ground_truth,
    ]
    for _dir in chain_dir_list:
        shutil.copytree(os.path.join(_dir, config.input.chain_dir_name), chains_dir, dirs_exist_ok=True)
    logger.debug("Copied source files to %s", temp_dir)
    return docs_dir, chains_dir


def invoke(config):
    # Copy files from source folders to temp directory. Only the files under /docs and /chains will be copied
    temp_dir = os.path.join(config.output.base_dir, config.output.temp_dir_name)
    if not config.safe_mode:
        remove_all(temp_dir)
    check_and_make_dir(
        temp_dir,
        raiseExceptionIfExist=True,
        errMsg=f"Please backup your files under this path if necessary and remove {temp_dir}.",
    )
    docs_dir, chains_dir = aggregrate_files(config, temp_dir)

    # Convert source files to .conll files
    conll_dir = os.path.join(config.output.base_dir, config.output.root_dir_name)
    if not config.safe_mode:
        remove_all(conll_dir)
    check_and_make_dir(
        conll_dir,
        raiseExceptionIfExist=True,
        errMsg=f"Please remove {temp_dir} and {conll_dir} before next execution.",
    )
    convert_to_conll(config, conll_dir, docs_dir, chains_dir)

    if os.path.exists(temp_dir):
        logger.debug("Removed the temporary directory: %s", temp_dir)
        shutil.rmtree(temp_dir)

    logger.info("Done.")

    return conll_dir
