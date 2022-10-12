###
# This pre-processing script aims to aggregrate the i2b2 conll files for later usage
###
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from multiprocessing import Event
import os
import shutil
import sys
import re
import json
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf
import pandas as pd

# pylint: disable=import-error
from common_utils.file_checker import FileChecker
from common_utils.common_utils import check_and_create_dirs, check_and_remove_dirs


FILE_CHECKER = FileChecker()
START_EVENT = Event()
logger = logging.getLogger()
module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")

# Utils


class I2b2Token(object):
    def __init__(self, docId, sentenceId, tokenId_sentencewise, tokenId_docwise, tokenStr):
        self.docId = docId
        self.sentenceId = sentenceId
        self.tokenId_sentencewise = tokenId_sentencewise
        self.tokenId_docwise = tokenId_docwise
        self.tokenStr = tokenStr
        self.tokenStr_for_conll = self._adapt_token_for_conll_(tokenStr)
        self.conll_corefMark = ""
        self.offset = 0

    def add_coref_conllmark(self, conllmark):
        if not self.conll_corefMark:
            self.conll_corefMark = conllmark
        else:
            self.conll_corefMark = f"{self.conll_corefMark}|{conllmark}"

    def _adapt_token_for_conll_(self, tokenStr) -> str:
        # IMPORTANT! Tokens start or end with `(` or `)` will trigger "conll/reference-coreference-scorers" unexpectedly,
        # which will either cause execution error or wrong metric score.
        # See coref/wrong_conll_scorer_example for details.
        tokenStr_for_conll = tokenStr
        if re.search(r"\(?[^A-Za-z]+\)?", tokenStr):
            tokenStr_for_conll = tokenStr.replace("(", "[").replace(")", "]")
        if tokenStr.strip() == "":
            tokenStr_for_conll = ""
        return tokenStr_for_conll

    def get_conll_str(self):
        if self.conll_corefMark:
            return f"{self.docId}\t0\t{self.tokenId_sentencewise}\t{self.tokenStr_for_conll}\t" + "_\t" * 8 + self.conll_corefMark
        return f"{self.docId}\t0\t{self.tokenId_sentencewise}\t{self.tokenStr_for_conll}\t" + "_\t" * 7 + "_"

    def __str__(self) -> str:
        return json.dumps({"docId": self.docId,
                           "sentenceId": self.sentenceId,
                           "tokenId_sentencewise": self.tokenId_sentencewise,
                           "tokenId_docwise": self.tokenId_docwise,
                           "tokenStr": self.tokenStr,
                           "conll_corefMark": self.conll_corefMark,
                           }, indent=2)
    __repr__ = __str__


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


def get_file_name_prefix(file_path, suffix):
    """ Extract the basename from base and remove the ``suffix``.
    e.g.: ../../../clinical-1.txt => clinical-1
    """
    return os.path.basename(file_path).rstrip(suffix)


def output_json(config, output_file_path, doc_id, text):
    with open(output_file_path, "a", encoding="UTF-8") as f:
        column_name = config.name_style.i2b2.json
        formatted_reocrd = {
            column_name.id: doc_id,
            column_name.text: text
        }
        f.write(json.dumps(formatted_reocrd))
        f.write("\n")


def output_csv(config, output_file_path, sentence_list: list[list[I2b2Token]],):
    i2b2_namestyle = config.name_style.i2b2.column_name
    df = pd.DataFrame(
        {
            i2b2_namestyle["original_token"]: [str(i2b2Token.tokenStr) for token_list in sentence_list for i2b2Token in token_list],
            i2b2_namestyle["token_in_conll"]: [str(i2b2Token.tokenStr_for_conll) for token_list in sentence_list for i2b2Token in token_list],
            i2b2_namestyle["sentence_group"]: [int(i2b2Token.sentenceId) for token_list in sentence_list for i2b2Token in token_list],
            i2b2_namestyle["token_id_sentencewise"]: [int(i2b2Token.tokenId_sentencewise) for token_list in sentence_list for i2b2Token in token_list],
            i2b2_namestyle["token_id_docwise"]: [int(i2b2Token.tokenId_docwise) for token_list in sentence_list for i2b2Token in token_list],
            i2b2_namestyle["coref_group_conll"]: [str(i2b2Token.conll_corefMark) for token_list in sentence_list for i2b2Token in token_list],
        },
    )
    df.to_csv(output_file_path)


def output_conll(output_file_path, doc_id, sentence_list: list[list[I2b2Token]], mode="a"):
    """ Write .conll file """
    BEGIN = f"#begin document ({doc_id}); part 0\n"
    SENTENCE_SEPARATOR = "\n"
    END = "#end document\n"
    with open(output_file_path, mode, encoding="UTF-8") as out:
        out.write(BEGIN)
        for sent in sentence_list:
            # Skip empty sentence
            if len(sent) == 1 and sent[0].tokenStr == "":
                continue
            for tok in sent:
                out.write(tok.get_conll_str() + "\n")
            out.write(SENTENCE_SEPARATOR)
        out.write(END)

###


def batch_processing(doc_file_path, chain_file_path) -> tuple[str, str, list[list[I2b2Token]]]:
    """ Resolve a single i2b2 document, including a .txt file and a .chains file. """
    START_EVENT.wait()

    doc_id = get_file_name_prefix(doc_file_path, ".txt")

    # Resolve doc file
    sentence_list: list[list[I2b2Token]] = []
    with open(doc_file_path, "r", encoding="UTF-8-sig") as doc:
        tokenId_docwise = 0
        for sentence_id, doc_line in enumerate(doc.readlines()):
            token_list: list[I2b2Token] = []
            for tokenId_sentencewise, token_str in enumerate(clean_and_split_line(doc_line, debug_doc=doc_id, debug_sent=sentence_id)):
                token_list.append(I2b2Token(doc_id, sentence_id, tokenId_sentencewise, tokenId_docwise, token_str))
                tokenId_docwise += 1
            sentence_list.append(token_list)

    # Resolve chain file (coref cluster)
    with open(chain_file_path, "r", encoding="UTF-8-sig") as chain:
        for cluster_id, cluster in enumerate(chain.readlines()):
            for coref in cluster.split("||")[0:-1]:  # Drop the last one, which is the type of the coref
                token_range: list[str, str] = coref.split(" ")[-2:]
                start = token_range[0]
                end = token_range[1]
                if start == end:
                    sentId, tokId = start.split(":")
                    mark = f"({cluster_id})"
                    sentence_list[int(sentId) - 1][int(tokId)].add_coref_conllmark(mark)
                else:
                    sentId, tokId = start.split(":")
                    startMark = f"({cluster_id}"
                    sentence_list[int(sentId) - 1][int(tokId)].add_coref_conllmark(startMark)

                    sentId, tokId = end.split(":")
                    endMark = f"{cluster_id})"
                    sentence_list[int(sentId) - 1][int(tokId)].add_coref_conllmark(endMark)

    return doc_file_path, doc_id, sentence_list


@hydra.main(version_base=None, config_path=config_path, config_name="data_preprocessing")
def main(config):
    """
    This pre-processing script aims to aggregrate the i2b2 raw files (.txt and .chains) for later usage.
    The output of this script is an i2b2_all.jsonlines file and a ground_truth folder
    """
    print(OmegaConf.to_yaml(config))

    logger.info("Pre-processing i2b2 dataset:")

    if config.clear_history:
        logger.info("Cleaning the history output: %s", config.output_base_dir)
        check_and_remove_dirs(config.output_base_dir, config.clear_history)

    # Copy files from source folders to temp directory. Only the files under /docs and /chains will be copied
    temp_dir = os.path.join(config.temp.dir)
    logger.info("Coping the i2b2 source files to temp dir: %s", temp_dir)
    check_and_create_dirs(temp_dir)
    docs_dir, chains_dir = aggregrate_files(config, temp_dir)

    # Check that the files are matched.
    doc_files = os.listdir(docs_dir)
    chain_files = os.listdir(chains_dir)
    assert len(doc_files) == len(chain_files)

    # Process each files
    all_task = []
    with ProcessPoolExecutor(max_workers=config.multiprocess_workers) as executor:
        # Submit task
        logger.info("Loading the documents:")
        for _file_name in tqdm(doc_files):
            # Input files
            doc_file_path = os.path.join(docs_dir, _file_name)
            chain_file_path = os.path.join(chains_dir, _file_name + config.input.chain_suffix)
            all_task.append(executor.submit(batch_processing, doc_file_path, chain_file_path))

        # Notify tasks to start
        START_EVENT.set()

        # When a submitted task finished, the output is received here.
        logger.info("Resolving and saving the documents")
        if all_task:
            for future in tqdm(as_completed(all_task), total=len(all_task)):
                doc_file_path, doc_id, sentence_list = future.result()

                if config.output.json:
                    check_and_create_dirs(config.json.output_dir)
                    output_file_path = os.path.join(config.json.output_dir, config.json.file_name)
                    # Directly aggregrate the source .txt files
                    with open(doc_file_path, "r", encoding="UTF-8") as f:
                        text = "".join(f.readlines())
                    output_json(config, output_file_path, doc_id, text)
                if config.output.csv:
                    check_and_create_dirs(config.csv.output_dir)
                    output_file_path = os.path.join(config.csv.output_dir, f"{doc_id}.csv")
                    output_csv(config, output_file_path, sentence_list)
                if config.output.conll:
                    # Output aggregrate file.
                    check_and_create_dirs(config.conll.output_base_dir)
                    output_file_path = os.path.join(config.conll.output_base_dir, config.conll.file_name)
                    output_conll(output_file_path, doc_id, sentence_list, mode="a")
                    # Output individual files
                    check_and_create_dirs(config.conll.output_dir)
                    output_singlefile_path = os.path.join(config.conll.output_dir, f"{doc_id}.conll")
                    output_conll(output_singlefile_path, doc_id, sentence_list, mode="w")
        START_EVENT.clear()
    logger.info("Removing temp dir: %s", temp_dir)
    check_and_remove_dirs(temp_dir, True)

    logger.info("Done.")


if __name__ == "__main__":
    sys.argv.append("data_preprocessing@_global_=i2b2")
    main()  # pylint: disable=no-value-for-parameter
