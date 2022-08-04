from pydoc import doc
from tkinter import SEPARATOR
import pandas
import os, sys, shutil
import hydra
import logging
from natsort import natsorted
import random
from tqdm import tqdm

from sympy import print_rcode

logger = logging.getLogger()


def shuffle_list(file_list, seed=42):
    """ Sort the list by file name numerically, then shuffle the list. 
    seed: 0 to use random seed to shuffle the dataset; -1 to disable shuffle.
    """
    file_list = natsorted(file_list)  # Sort by file name numerically
    if seed != -1:
        if seed == 0:
            random.Random().shuffle(file_list)
        else:
            random.Random(seed).shuffle(file_list)
    return file_list


def split_and_shuffle_list(doc_files, testset_size, fold_id: int, seed=42):
    """ Include three steps: Sort the list by file name numerically. 
    Move the test set to the end of the list.
    Shuffle the train set and test set separately.

    doc_files: Source file name list.
    testset_size: The size of the test set.
    fold_id: The index of the current cross-validation fold.
    seed: Random seed for shuffling.
    """
    doc_files = natsorted(doc_files)  # Sort by file name numerically
    doc_files_test = doc_files[fold_id * testset_size : (fold_id + 1) * testset_size]
    # The actual test split size is typicall larger than proportion value, causing the last test split size less than expected.
    if len(doc_files_test) < testset_size:
        doc_files_test = doc_files[-testset_size - 1 : -1]
    doc_files_train = [i for i in doc_files if i not in doc_files_test]
    # Shuffle train/test files
    doc_files_train = shuffle_list(doc_files_train, seed)
    doc_files_test = shuffle_list(doc_files_test, seed)
    # Concat the shuffled train/test files
    return [*doc_files_train, *doc_files_test]


def get_data_split(doc_files_shuffled, data_split_name, data_split_num):
    """ Split the dataset. The output is a list of dict:

    output_name_prefix: The names specifed in ${data_split.output_name_prefix} (comma split).
    file_list: The file name in /temp/docs, e.g. clinical-1.txt. The size is computed according to the ${data_split.proportion} (comma split).
    """
    data_split: list[dict] = []
    curr_index = 0
    for _idx, _output_name_prefix in enumerate(data_split_name):
        next_index = curr_index + data_split_num[_idx]
        curr_split = {"output_name_prefix": _output_name_prefix, "file_list": doc_files_shuffled[curr_index:next_index]}
        data_split.append(curr_split)
        curr_index = next_index
    logger.debug(f"Dataset is split into {data_split_name} with proportion {data_split_num}")
    return data_split


def get_porportion_and_name(config, doc_files):
    """ Get the actual numerical values for the dataset split and corresponding output name prefixes """
    # Compute the actual numerical values for the dataset split
    data_split_proportion = [int(i) for i in config.data_split.if_split.proportion.split(",")]  # [7, 4, 1]
    data_split_proportion_norm = [i / sum(data_split_proportion) for i in data_split_proportion]  # [0.7, 0.4, 0.1]
    data_split_num = [int(i * len(doc_files)) for i in data_split_proportion_norm]  # [247.33..., 141.33..., 35.33...]
    data_split_num[-1] = len(doc_files) - sum(data_split_num[0:-1])  # [247, 141, 36]
    # The output names of the dataset split
    data_split_name = [i for i in config.data_split.if_split.output_name_prefix.split(",")]
    assert len(data_split_name) == len(data_split_num)
    return data_split_name, data_split_num


def get_file_name_prefix(txt_file_name):
    """ ../../../clinical-1.txt => clinical-1 """
    return os.path.basename(txt_file_name).rstrip(".txt")


def convert_each(doc_file_path, chain_file_path, output_file_path):
    """ Convert the source doc and chain files into the conll file """
    doc_id = get_file_name_prefix(doc_file_path)
    BEGIN = f"#begin document ({doc_id}); part 0\n"
    SENTENCE_SEPARATOR = "\n"
    END = f"#end document\n"

    class Token(object):
        def __init__(self, docId, sentenceId, tokenId, tokenStr):
            self.docId = docId
            self.sentenceId = sentenceId
            self.tokenId = tokenId
            self.tokenStr = tokenStr.rstrip("\n")
            self.corefMark = ""

        def add_coref_mark(self, mark):
            if not self.corefMark:
                self.corefMark = mark
            else:
                self.corefMark = f"{self.corefMark}|{mark}"

        def get_conll_str(self):
            if self.corefMark:
                return f"{self.docId}\t0\t{self.tokenId}\t{self.tokenStr}\t" + "_\t" * 8 + self.corefMark
            else:
                return f"{self.docId}\t0\t{self.tokenId}\t{self.tokenStr}\t" + "_\t" * 7 + "_"

        def __str__(self) -> str:
            return f"{self.tokenStr}({self.sentenceId}:{self.tokenId})|[{self.corefMark}]"

        __repr__ = __str__

    # Resolve doc file
    sentenc_list: list[list[Token]] = []
    with open(doc_file_path, "r") as doc:
        for doc_line_id, doc_line in enumerate(doc.readlines()):
            token_list: list[Token] = []
            for doc_token_id, token_str in enumerate(doc_line.split(" ")):
                token_list.append(Token(doc_id, doc_line_id, doc_token_id, token_str))
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
                    sentenc_list[int(sentId) - 1][int(tokId)].add_coref_mark(mark)
                else:
                    sentId, tokId = start.split(":")
                    startMark = f"({cluster_id}"
                    sentenc_list[int(sentId) - 1][int(tokId)].add_coref_mark(startMark)

                    sentId, tokId = end.split(":")
                    endMark = f"{cluster_id})"
                    sentenc_list[int(sentId) - 1][int(tokId)].add_coref_mark(endMark)

    # Write .conll file
    with open(output_file_path, "a") as out:
        out.write(BEGIN)
        for sent in sentenc_list:
            for tok in sent:
                out.write(tok.get_conll_str() + "\n")
            out.write(SENTENCE_SEPARATOR)
        out.write(END)


def convert_all(config, data_split, docs_dir, chains_dir, output_dir_next):
    for split in data_split:
        logger.info(f'Converting the files for [{split["output_name_prefix"]}] set:')
        for _file_name in tqdm(split["file_list"]):
            # Input files
            doc_file_path = os.path.join(docs_dir, _file_name)
            chain_file_path = os.path.join(chains_dir, _file_name + config.input.chain_suffix)
            # Output file
            output_file_path = os.path.join(output_dir_next, split["output_name_prefix"] + config.output.suffix)
            convert_each(doc_file_path, chain_file_path, output_file_path)
        logger.info(f"Output: {output_file_path}")


def convert_to_conll(config, output_dir, docs_dir, chains_dir):
    # Check that the files are matched.
    doc_files = os.listdir(docs_dir)
    chain_files = os.listdir(chains_dir)
    assert len(doc_files) == len(chain_files)
    logger.info(f"Detected {len(chain_files)} documents in total.")

    # Do not split the dataset. All the data will be aggregrated into one file with prefix "all"
    if config.data_split.unsplit:
        data_split_name = [config.data_split.if_unsplit.output_name_prefix]
        data_split_num = [len(doc_files)]
        doc_files_shuffle = shuffle_list(doc_files, config.shuffle_seed)  # Shuffle the file list

        data_split = get_data_split(doc_files_shuffle, data_split_name, data_split_num)  # Split the files

        output_dir_next = os.path.join(output_dir, config.data_split.if_unsplit.dir_name)
        check_and_make_dir(output_dir_next)

        logger.info(f"***Creating an unsplit dataset at: {output_dir_next}")
        convert_all(config, data_split, docs_dir, chains_dir, output_dir_next)

    # Split the dataset with shuffling the whole soure files
    if config.data_split.random_shuffle:
        data_split_name, data_split_num = get_porportion_and_name(config, doc_files)
        doc_files_shuffle = shuffle_list(doc_files, config.shuffle_seed)  # Shuffle the file list

        data_split = get_data_split(doc_files_shuffle, data_split_name, data_split_num)

        output_dir_next = os.path.join(output_dir, config.data_split.if_random_shuffle.dir_name)
        check_and_make_dir(output_dir_next)

        logger.info(f"***Creating a shuffle dataset at: {output_dir_next}")
        convert_all(config, data_split, docs_dir, chains_dir, output_dir_next)

    # Cross-validation. Split the train/test set first and then shuffle the soure files.
    # Make sure the test set of each fold will not overlap.
    if config.data_split.cross_validation:
        folds = int(config.data_split.if_cross_validation.folds)
        logger.info(f"***Creating datasets for cross-validaion with {folds} folds")
        for fold_id in range(folds):
            data_split_name, data_split_num = get_porportion_and_name(config, doc_files)
            # Change files order for later spliting
            doc_files_fold_shuffle = split_and_shuffle_list(doc_files, data_split_num[-1], fold_id, config.shuffle_seed)
            data_split = get_data_split(doc_files_fold_shuffle, data_split_name, data_split_num)

            output_dir_next = os.path.join(output_dir, str(fold_id))
            check_and_make_dir(output_dir_next)

            logger.info(f"Creating the fold-{fold_id} dataset at: {output_dir_next}")
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
    logger.debug(f"Copied source files to {temp_dir}")
    return docs_dir, chains_dir


def check_and_make_dir(dir, raiseExceptionIfExist=False, errMsg=""):
    if not os.path.exists(dir):
        os.makedirs(dir)
        logger.debug(f"Created directory: {dir}")
    else:
        logger.debug(f"The directory already exists: {dir}")
        if raiseExceptionIfExist:
            raise Exception(f"The directory {dir} already exists. {errMsg}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    # Copy files from source folders to temp directory. Only the files under /docs and /chains will be copied
    temp_dir = os.path.join(config.dataset_dir, config.output.temp_dir_name)
    check_and_make_dir(
        temp_dir,
        raiseExceptionIfExist=True,
        errMsg=f"Please backup your files under this path if necessary and remove {temp_dir}.",
    )
    docs_dir, chains_dir = aggregrate_files(config, temp_dir)

    # Convert source files to .conll files
    conll_dir = os.path.join(config.output_dir, config.output.root_dir_name)
    check_and_make_dir(
        conll_dir,
        raiseExceptionIfExist=True,
        errMsg=f"Please remove {temp_dir} and {conll_dir} before next execution.",
    )
    convert_to_conll(config, conll_dir, docs_dir, chains_dir)

    if os.path.exists(temp_dir):
        logger.debug(f"Removed the temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
