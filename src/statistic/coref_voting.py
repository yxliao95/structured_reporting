import json
import logging
import os
import sys
import time
import re
import ast
import math
from math import isnan

import hydra
import pandas as pd
from omegaconf import OmegaConf

# pylint: disable=import-error,wrong-import-order
from common_utils.common_utils import check_and_remove_dirs
from common_utils.file_checker import FileChecker

logger = logging.getLogger()
module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")

FILE_CHECKER = FileChecker()


# Utils
class SpacyToken:
    """ The token from spaCy """

    def __init__(self, index, tokenStr) -> None:
        self.tokenStr = tokenStr
        self.index = index
        self.token_details = {}
        # For each coref model output, if this token belongs to any coref mention, then adds 1.
        # Having three models mean that the max vote_count is 3.
        self.vote_details = {}

    def vote(self, model_short_name, has_mention: bool):
        self.vote_details[model_short_name] = 1 if has_mention else 0

    def update_vote(self, model_short_name, has_mention: bool):
        """ Update the voting results only if `has_mention` is True """
        if has_mention:
            self.vote_details[model_short_name] = 1

    def inherit_vote(self, last_vote_details: dict):
        self.vote_details = last_vote_details.copy()

    def update_token_details(self, model_short_name, status: str):
        """Args:
        status: ["equal", "contains", "subset_of", "not_found"], 
            e.g. spacyTok is subset_of corefTok => TokA+TokB+TokC = TokA';
            spacyTok contains subset_of corefTok => TokA = TokA'+TokB'+TokC'.
        """
        self.token_details[model_short_name] = {"status": status}

    def __repr__(self) -> str:
        """ Inorder to see the raw text, we convert it to byte format for printing. """
        return f"({self.index}){self.tokenStr.encode('utf-8')}"


class DocClass:
    def __init__(self) -> None:
        self.token_list = []

    def add_token(self, baseToken: SpacyToken):
        self.token_list.insert(baseToken.index, baseToken)

    def get_token_byIndex(self, index) -> SpacyToken:
        return self.token_list[index]

    def auto_get_token_byIndex(self, spacy_index, spacy_token) -> SpacyToken:
        """ Get token from list, if not exist then create one. """
        try:
            return self.token_list[spacy_index]
        except IndexError:
            baseToken = SpacyToken(spacy_index, spacy_token)
            self.token_list.insert(baseToken.index, baseToken)
            return baseToken

    def update_token(self, base_token: SpacyToken):
        pass


class CorefMention:
    def __init__(self) -> None:
        self.token_list = []
        self.index_list = []
        self.aligned_index_list = []
        self.mention_group: CorefMentionGroup = None


class CorefMentionGroup:
    def __init__(self) -> None:
        self.mention_list: list[CorefMention] = []


def isNot_empty_or_NaN(coref_group):
    """ Empty refers to "-1", NaN refers to math.nan """
    if coref_group != "-1" and coref_group != str(math.nan):
        return True
    else:
        return False
###


@hydra.main(version_base=None, config_path=config_path, config_name="statistic")
def main(config):
    print(OmegaConf.to_yaml(config))
    source_cfg = config.input.source
    spacy_name_style_cfg = config.name_style.spacy.column_name
    # Get the info of three coref models
    candidate_coref_model_cfg_list = [(model_short_name, source_cfg.coref_models.get(model_short_name)) for model_short_name in source_cfg.in_use]

    check_and_remove_dirs(config.output_dir, config.clear_history)

    # Loop in sections
    section_name_list = source_cfg.section
    for section_name in section_name_list:
        logger.info("Voting on section: %s", section_name)

        spacy_out_dir = os.path.join(source_cfg.baseline_model.dir, section_name)
        if not os.path.exists(spacy_out_dir):
            logger.error("Could not found the target dir (the baseline for token alignment): %s", spacy_out_dir)

        # Loop all files
        docClass = DocClass()
        candidate_fileName_list = FILE_CHECKER.filter(os.listdir(spacy_out_dir))
        for file_name in candidate_fileName_list:
            if file_name == "clinical-682.csv":
                # Read spacy output as alignment base
                spacy_file_path = os.path.join(spacy_out_dir, file_name)
                df_spacy = pd.read_csv(spacy_file_path)

                # Loop the models
                for model_short_name, coref_model_cfg in candidate_coref_model_cfg_list:
                    coref_column_cfg = coref_model_cfg.target_column

                    # Read coref model output to be aligned
                    coref_model_output_path = os.path.join(coref_model_cfg.dir, section_name, file_name)
                    df_coref = pd.read_csv(coref_model_output_path)

                    # Merge and align two df. Keep the index for later use
                    df_spacy[config.df_col.spacy_index] = df_spacy.index
                    df_coref[config.df_col.coref_index] = df_coref.index
                    df_aligned = df_spacy.merge(df_coref, how="outer", left_index=True, right_on=coref_column_cfg.spacy_index).reset_index().drop(columns=["index"])

                    # Token level alignment. Assuming spacy has token A B C, corenlp has token A' B' C'
                    checkpoint_token_index = 0
                    many2one_spacy_token_len = 0
                    one2many_coref_token_len = 0
                    for _idx, _itemSeries in df_aligned.iterrows():
                        spacy_index = _itemSeries.get(config.df_col.spacy_index)
                        coref_index = _itemSeries.get(config.df_col.coref_index)

                        spacy_token = str(_itemSeries.get(spacy_name_style_cfg.token))
                        coref_token = str(_itemSeries.get(coref_column_cfg.token))

                        coref_group = str(_itemSeries.get(coref_column_cfg.coref_group))  # "-1", "nan", "[1, 11]"
                        coref_group_conll = str(_itemSeries.get(coref_column_cfg.coref_group_conll))  # "-1", "nan", "['(1)', '(11']"
                        assert coref_group != "-1.0" or coref_group != -1.0

                        last_spacy_index = df_aligned.loc[checkpoint_token_index, config.df_col.spacy_index]
                        last_spacy_token = str(df_aligned.loc[checkpoint_token_index, spacy_name_style_cfg.token])
                        last_coref_token = str(df_aligned.loc[checkpoint_token_index, coref_column_cfg.token])

                        # TokA = TokA'
                        if spacy_token == coref_token:
                            logger.debug("%s) Identical token: %s | %s", _idx, spacy_token, coref_token)
                            baseToken = docClass.auto_get_token_byIndex(spacy_index, spacy_token)
                            baseToken.update_token_details(model_short_name, "equal")
                            baseToken.vote(model_short_name, isNot_empty_or_NaN(coref_group))
                            checkpoint_token_index = _idx
                        else:
                            # TokA = TokA'+TokB'+TokC', TokA contains TokA'
                            if coref_token in spacy_token:
                                if one2many_coref_token_len == 0:
                                    checkpoint_token_index = _idx
                                    logger.debug("%s) One2many token start: %s | %s", _idx, spacy_token, coref_token)
                                    baseToken = docClass.auto_get_token_byIndex(spacy_index, spacy_token)
                                    baseToken.update_token_details(model_short_name, "contains")
                                    baseToken.vote(model_short_name, isNot_empty_or_NaN(coref_group))
                                else:
                                    logger.debug("%s) One2many token: %s | %s", _idx, spacy_token, coref_token)
                                    one2many_coref_token_len += len(coref_token)
                                    baseToken = docClass.get_token_byIndex(spacy_index)
                                    # Any mention that found in TokA',TokB',TokC' will vote on TokA. Thus, vote again.
                                    baseToken.update_vote(model_short_name, isNot_empty_or_NaN(coref_group))
                                    if one2many_coref_token_len == len(spacy_token):
                                        one2many_coref_token_len = 0
                                        logger.debug("%s) One2many token end", _idx)

                            # TokA+TokB+TokC = TokA', TokA is subset_of TokA'
                            # TokA of TokA+TokB+TokC
                            if spacy_token in coref_token:
                                checkpoint_token_index = _idx
                                many2one_spacy_token_len += len(spacy_token)
                                logger.debug("%s) Many2one token start: %s | %s", _idx, spacy_token, coref_token)
                                baseToken = docClass.auto_get_token_byIndex(spacy_index, spacy_token)
                                baseToken.update_token_details(model_short_name, "subset_of")
                                baseToken.vote(model_short_name, isNot_empty_or_NaN(coref_group))

                            # TokB,TokC of TokA+TokB+TokC
                            if isnan(coref_index) and spacy_token in last_coref_token:
                                many2one_spacy_token_len += len(spacy_token)
                                logger.debug("%s) Many2one token: %s | %s", _idx, spacy_token, coref_token)
                                baseToken = docClass.auto_get_token_byIndex(spacy_index, spacy_token)
                                baseToken.update_token_details(model_short_name, "subset_of")
                                checkpoint_baseToken = docClass.get_token_byIndex(df_aligned.loc[checkpoint_token_index, config.df_col.spacy_index])
                                # Inherit the voting results from TokA'
                                baseToken.inherit_vote(checkpoint_baseToken.vote_details)

                                if many2one_spacy_token_len == len(last_coref_token):
                                    many2one_spacy_token_len = 0
                                    logger.debug("%s) Many2one token end")

                            # TokA exist, TokA' not exist
                            if isnan(coref_index) and spacy_token not in last_coref_token:
                                logger.debug("%s) Empty coref token: %s | %s", _idx, spacy_token.encode(), coref_token)
                                baseToken = docClass.auto_get_token_byIndex(spacy_index, spacy_token)
                                baseToken.update_token_details(model_short_name, "not_found")
                                baseToken.vote(model_short_name, isNot_empty_or_NaN(coref_group))

                return


if __name__ == "__main__":
    sys.argv.append("+statistic/coref_voting@_global_=i2b2")
    main()  # pylint: disable=no-value-for-parameter
