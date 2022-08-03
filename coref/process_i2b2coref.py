from pydoc import doc
import pandas
import os, sys, shutil
import hydra
import logging
from natsort import natsorted
import random

logger = logging.getLogger()


def shuffle_list(file_list, seed=42):
    """ Shuffle the original list """
    random.Random(seed).shuffle(file_list)


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


def split_dataset(config, doc_files):
    # Compute the actual numerical values for the dataset split
    data_split_proportion = [int(i) for i in config.data_split.proportion.split(",")]  # 7,2,1
    data_split_proportion_norm = [i / sum(data_split_proportion) for i in data_split_proportion]  # 0.7, 0.2, 0.1
    data_split_num = [int(i * len(doc_files)) for i in data_split_proportion_norm]
    data_split_num[-1] = len(doc_files) - sum(data_split_num[0:-1])
    # The output names of the dataset split
    data_split_name = [i for i in config.data_split.output_name_prefix.split(",")]
    assert len(data_split_name) == len(data_split_num)
    return data_split_name, data_split_num


def convert_to_conll(config, output_dir, docs_dir, chains_dir):
    # Check that the files are matched.
    doc_files = os.listdir(docs_dir)
    chain_files = os.listdir(chains_dir)
    assert len(doc_files) == len(chain_files)
    # Shuffle dataset
    doc_files = natsorted(doc_files)  # Sort by file name numerically
    shuffle_list(doc_files, config.shuffle_seed)  # Shuffle the file list
    logger.info(f"Detected {len(chain_files)} documents in total.")
    # Split the dataset
    if config.data_split.require_split:
        data_split_name, data_split_num = split_dataset(config, doc_files)
        data_split = get_data_split(doc_files, data_split_name, data_split_num)
    else:
        # Do not split the dataset. All the data will be aggregrated into one file with prefix "all"
        data_split_name = ["all"]
        data_split_num = [len(doc_files)]
        data_split = get_data_split(doc_files, data_split_name, data_split_num)
    for split in data_split:
        print(split["output_name_prefix"])
        print(len(split["file_list"]))


def aggregrate_files(config, temp_dir):
    """ Copy the dataset files from source dirs to temp dirs. """
    docs_dir = os.path.join(temp_dir, "docs")
    chains_dir = os.path.join(temp_dir, "chains")
    # for _path in [docs_dir, chains_dir]:
    #     os.makedirs(_path, exist_ok=True)
    # train_dir_list = [
    #     config.paths.beth_train,
    #     config.paths.partners_train,
    #     config.paths.beth_test,
    #     config.paths.partners_test,
    # ]
    # for _dir in train_dir_list:
    #     shutil.copytree(os.path.join(_dir, "docs"), docs_dir, dirs_exist_ok=True)
    # chain_dir_list = [
    #     config.paths.beth_train,
    #     config.paths.partners_train,
    #     config.paths.beth_test_ground_truth,
    #     config.paths.partners_test_ground_truth,
    # ]
    # for _dir in chain_dir_list:
    #     shutil.copytree(os.path.join(_dir, "chains"), chains_dir, dirs_exist_ok=True)
    logger.debug(f"Copied source files to {temp_dir}")
    return docs_dir, chains_dir


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    # Copy files from source folders to temp directory. Only the files under /docs and /chains will be copied
    temp_dir = os.path.join(config.dataset_dir, "temp")
    if not os.path.exists(temp_dir):
        logger.info(f"Created a temporary directory: {temp_dir}")
        os.mkdir(temp_dir)
    else:
        logger.warning(f"Temporary directory exists: {temp_dir}, and will be deleted later.")
        # raise Exception(
        #     f"Temporary directory exists, please backup your files under this path and make sure the path {temp_dir} does not exist as it will be removed later."
        # )

    docs_dir, chains_dir = aggregrate_files(config, temp_dir)

    # Convert source files to conll files
    conll_dir = os.path.join(config.dataset_dir, "conll")
    if not os.path.exists(conll_dir):
        logger.info(f"Created the output directory: {conll_dir}")
        os.mkdir(conll_dir)
    else:
        logger.warning(f"Output directory already exists: {conll_dir}")
        # raise Exception(
        #     f"Output directory already exists, please backup your files under this path and make sure the path {temp_dir} does not exist as the files inside might be replaced."
        # )

    convert_to_conll(config, conll_dir, docs_dir, chains_dir)

    # if os.path.exists(temp_dir):
    #     logger.info(f"Removed the temporary directory: {temp_dir}")
    #     shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
