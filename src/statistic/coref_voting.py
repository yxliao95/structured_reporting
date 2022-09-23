from email.policy import default
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
        self.token_details: dict[str, object] = {}
        self.vote_details: dict[str, int] = {}  # {"ml": 0, "rb": 1, "fj": 1}
        # self.of_which_mentions_details: dict[str, list[MentionClass]] = {}
        self.of_which_mentions_details = {}  # The key (which is model_short_name) would only appear when there is coref result

    def vote(self, model_short_name, has_mention: bool):
        self.vote_details[model_short_name] = 1 if has_mention else 0

    def update_vote(self, model_short_name, has_mention: bool):
        """ Update the voting results only if `has_mention` is True """
        if has_mention:
            self.vote_details[model_short_name] = 1

    def inherit_vote(self, last_vote_details: dict):
        self.vote_details = last_vote_details.copy()

    def inherit_mention(self, model_short_name: str, checkpoint_token: "SpacyToken", coref_tokenStr: str, spacy_tokenStr: str):
        for mentionObj in checkpoint_token.of_which_mentions_details.get(model_short_name, []):
            mentionObj.update_mention(coref_tokenStr, spacy_tokenStr, self)
            self.update_mention_details(model_short_name, mentionObj)

    def update_token_details(self, model_short_name, status: str):
        """Args:
        status: ["equal", "contains", "subset_of", "not_found"], 
            e.g. spacyTok is subset_of corefTok => TokA+TokB+TokC = TokA';
            spacyTok contains subset_of corefTok => TokA = TokA'+TokB'+TokC'.
        """
        self.token_details[model_short_name] = {"status": status}

    def update_mention_details(self, model_short_name, mentionObj: "MentionClass"):
        if model_short_name not in self.of_which_mentions_details:
            self.of_which_mentions_details[model_short_name] = []
        if not mentionObj in self.of_which_mentions_details[model_short_name]:
            self.of_which_mentions_details[model_short_name].append(mentionObj)

    def __repr__(self) -> str:
        return json.dumps({
            "index": self.index,
            "tokenStr": self.tokenStr,
            "token_details": self.token_details,
            "vote_details": self.vote_details,
            "of_which_mentions_details": [{model: [m.__repr__ for m in mentions]} for model, mentions in self.of_which_mentions_details.items()],
        }, indent=2)


class MentionClass:
    increment_id = 0

    def __init__(self, model_short_name) -> None:
        self.id = MentionClass.increment_id
        MentionClass.increment_id += 1
        self.of_which_model = model_short_name
        self.mentionStr = []
        self.mentionStr_inSpacy = []
        self.spacy_tokens: list[SpacyToken] = []
        self.of_which_coref_groups: dict[int, MentionGroupClass] = {}

    def init_mention(self, docClass: "DocClass", mentionStr: str, mentionStr_inSpacy: str, spacyTokenObj: SpacyToken, coref_group_id: int):
        self.mentionStr.append(mentionStr)
        self.mentionStr_inSpacy.append(mentionStr_inSpacy)
        self.spacy_tokens.append(spacyTokenObj)
        mentionGroupClass = docClass.auto_get_mention_group(self.of_which_model, coref_group_id)
        mentionGroupClass.add_mention(self)
        self.of_which_coref_groups[coref_group_id] = mentionGroupClass

    def update_mention(self, mentionStr: str, mentionStr_inSpacy: str, spacyTokenObj: SpacyToken):
        self.mentionStr.append(mentionStr)
        if spacyTokenObj not in self.spacy_tokens:
            self.mentionStr_inSpacy.append(mentionStr_inSpacy)
            self.spacy_tokens.append(spacyTokenObj)

    def __repr__(self) -> str:
        return json.dumps({
            "id": self.id,
            "of_which_model": self.of_which_model,
            "mentionStr": self.mentionStr,
            "mentionStr_inSpacy": self.mentionStr_inSpacy,
            "spacy_tokens": [f"{i.index}){i.tokenStr}" for i in self.spacy_tokens],
            "of_which_coref_groups": [f"{id}){group}" for id, group in self.of_which_coref_groups.items()]
        }, indent=2)


class MentionGroupClass:

    def __init__(self, group_id, model_short_name) -> None:
        self.id = group_id
        self.of_which_model = model_short_name
        self.mention_list: list[MentionClass] = []

    def add_mention(self, mentionObj: MentionClass):
        self.mention_list.append(mentionObj)

    def get_last_mention(self) -> MentionClass:
        return self.mention_list[-1]

    def __repr__(self) -> str:
        return json.dumps({
            "group_id": self.id,
            "of_which_model": self.of_which_model,
            "mention_list": [i.id for i in self.mention_list],
        }, indent=2)


class DocClass:
    def __init__(self) -> None:
        self.token_list = []
        self.mention_groups_details: dict[str, dict[int, MentionGroupClass]] = {}

    def get_token_byIndex(self, index) -> SpacyToken:
        return self.token_list[index]

    def auto_get_token_byIndex(self, spacy_index, spacy_tokenStr) -> SpacyToken:
        """ Get token from list, if not exist then create one. """
        try:
            return self.token_list[spacy_index]
        except IndexError:
            baseTokenObj = SpacyToken(spacy_index, spacy_tokenStr)
            self.token_list.insert(baseTokenObj.index, baseTokenObj)
            return baseTokenObj

    def auto_get_mention_group(self, model_short_name, coref_group_id: int):
        """ Get by model name and coref group id, create one if not exist. """
        mention_groups: dict[int, MentionGroupClass] = self._get_mention_groups_dict_(model_short_name)
        if coref_group_id not in [_group_id for _group_id, _ in mention_groups.items()]:
            mentionGroupClass = MentionGroupClass(coref_group_id, model_short_name)
            mention_groups[coref_group_id] = mentionGroupClass
        return mention_groups[coref_group_id]

    def _get_mention_groups_dict_(self, model_short_name) -> dict[int, MentionGroupClass]:
        if model_short_name not in self.mention_groups_details:
            self.mention_groups_details[model_short_name] = {}
        return self.mention_groups_details[model_short_name]


def isNot_empty_or_NaN(coref_group) -> bool:
    """ Empty refers to "-1", NaN refers to math.nan """
    if coref_group != "-1" and coref_group != str(math.nan):
        return True
    else:
        return False


def resolve_coref_group_conll_item(coref_group_conll: str) -> tuple[list[int], list[str]]:
    """ Arg:
        coref_group_conll: The string of a coref_group_conll column item (cell).

    Return:
        group_id_list: A list of coref group id,
        tok_status_list: A list of token status including ["first", "last", "single"]
    """
    coref_group_conll_list: list[str] = ast.literal_eval(coref_group_conll)
    group_id_list: list[int] = []
    tok_status_list: list[str] = []
    for conll_str in coref_group_conll_list:
        if "(" in conll_str and ")" in conll_str:
            coref_group_id = int(conll_str.replace("(", "").replace(")", ""))
            token_status = "single"
        elif "(" in conll_str:
            coref_group_id = int(conll_str.replace("(", ""))
            token_status = "first"
        elif ")" in conll_str:
            coref_group_id = int(conll_str.replace(")", ""))
            token_status = "last"
        else:
            logger.error("Should not see this.")
        group_id_list.append(coref_group_id)
        tok_status_list.append(token_status)
    return group_id_list, tok_status_list


def get_group_id_and_token_status(coref_group: str, coref_group_conll: str) -> dict[int, str]:
    """ This token belongs to which coref group and in what status.

    The status includes:
        "first" token of the mention,
        "inter"mediate token of the mention,
        "last" token of the mention,
        "single" token mention.

    Returns:
        dict{group_id: token status}, e.g. {"8": "first"}.
        Empty dict if no coref group is found.
    """
    has_coref = isNot_empty_or_NaN(coref_group)
    has_coref_conll = isNot_empty_or_NaN(coref_group_conll)
    out: dict[int, str] = {}
    if has_coref:
        # This token belongs to a mention and a mention group.
        coref_group_list: list[int] = ast.literal_eval(coref_group)
        if has_coref_conll:
            group_id_list, tok_status_list = resolve_coref_group_conll_item(coref_group_conll)
            for group_id in coref_group_list:
                if group_id in group_id_list:
                    out[group_id] = tok_status_list[group_id_list.index(group_id)]  # "first", "last", "single"
                else:
                    out[group_id] = "inter"
        else:
            for group_id in coref_group_list:
                out[group_id] = "inter"
    return out


def update_mention_and_mention_group(group_id_and_token_status_dict: dict, model_short_name: str, docObj: DocClass, coref_tokenStr: str, spacy_tokenStr: str, baseTokenObj: SpacyToken):
    for group_id, tok_status in group_id_and_token_status_dict.items():
        if tok_status == "single" or tok_status == "first":
            mentionObj = MentionClass(model_short_name)
            mentionObj.init_mention(docObj, coref_tokenStr, spacy_tokenStr, baseTokenObj, group_id)
        elif tok_status == "inter" or tok_status == "last":
            mentionGroupClass = docObj.auto_get_mention_group(model_short_name, group_id)
            mentionObj = mentionGroupClass.get_last_mention()
            mentionObj.update_mention(coref_tokenStr, spacy_tokenStr, baseTokenObj)
        baseTokenObj.update_mention_details(model_short_name, mentionObj)
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
        docObj = DocClass()
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
                    mention_id = 0
                    for _idx, _itemSeries in df_aligned.iterrows():
                        spacy_index = _itemSeries.get(config.df_col.spacy_index)
                        coref_index = _itemSeries.get(config.df_col.coref_index)

                        spacy_tokenStr = str(_itemSeries.get(spacy_name_style_cfg.token))
                        coref_tokenStr = str(_itemSeries.get(coref_column_cfg.token))

                        coref_group = str(_itemSeries.get(coref_column_cfg.coref_group))  # "-1", "nan", "[1, 11]"
                        coref_group_conll = str(_itemSeries.get(coref_column_cfg.coref_group_conll))  # "-1", "nan", "['(1)', '(11']"
                        assert coref_group != "-1.0" or coref_group != -1.0

                        group_id_and_token_status_dict: dict[int, str] = get_group_id_and_token_status(coref_group, coref_group_conll)

                        last_spacy_index = df_aligned.loc[checkpoint_token_index, config.df_col.spacy_index]
                        last_spacy_token = str(df_aligned.loc[checkpoint_token_index, spacy_name_style_cfg.token])
                        last_coref_token = str(df_aligned.loc[checkpoint_token_index, coref_column_cfg.token])

                        # TokA = TokA'
                        if spacy_tokenStr == coref_tokenStr:
                            logger.debug("%s) Identical token: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                            checkpoint_token_index = _idx
                            # Token-level
                            baseTokenObj = docObj.auto_get_token_byIndex(spacy_index, spacy_tokenStr)
                            baseTokenObj.update_token_details(model_short_name, "equal")
                            baseTokenObj.vote(model_short_name, isNot_empty_or_NaN(coref_group))
                            # Mention-level and Mention-Group-level
                            update_mention_and_mention_group(group_id_and_token_status_dict, model_short_name, docObj, coref_tokenStr, spacy_tokenStr, baseTokenObj)
                        else:
                            # TokA = TokA'+TokB'+TokC', TokA contains TokA'
                            if coref_tokenStr in spacy_tokenStr:
                                if one2many_coref_token_len == 0:
                                    checkpoint_token_index = _idx
                                    logger.debug("%s) One2many token start: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                                    baseTokenObj = docObj.auto_get_token_byIndex(spacy_index, spacy_tokenStr)
                                    baseTokenObj.update_token_details(model_short_name, "contains")
                                    baseTokenObj.vote(model_short_name, isNot_empty_or_NaN(coref_group))
                                    update_mention_and_mention_group(group_id_and_token_status_dict, model_short_name, docObj, coref_tokenStr, spacy_tokenStr, baseTokenObj)
                                else:
                                    logger.debug("%s) One2many token: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                                    one2many_coref_token_len += len(coref_tokenStr)
                                    baseTokenObj = docObj.get_token_byIndex(spacy_index)  # The same token generated above.
                                    # Any mention that found in TokA',TokB',TokC' will vote on TokA. Thus, vote again.
                                    baseTokenObj.update_vote(model_short_name, isNot_empty_or_NaN(coref_group))
                                    update_mention_and_mention_group(group_id_and_token_status_dict, model_short_name, docObj, coref_tokenStr, spacy_tokenStr, baseTokenObj)
                                    if one2many_coref_token_len == len(spacy_tokenStr):
                                        logger.debug("%s) One2many token end", _idx)
                                        one2many_coref_token_len = 0

                            # TokA+TokB+TokC = TokA', TokA is subset_of TokA'
                            # TokA of TokA+TokB+TokC
                            if spacy_tokenStr in coref_tokenStr:
                                checkpoint_token_index = _idx
                                many2one_spacy_token_len += len(spacy_tokenStr)
                                logger.debug("%s) Many2one token start: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                                baseTokenObj = docObj.auto_get_token_byIndex(spacy_index, spacy_tokenStr)
                                baseTokenObj.update_token_details(model_short_name, "subset_of")
                                baseTokenObj.vote(model_short_name, isNot_empty_or_NaN(coref_group))
                                update_mention_and_mention_group(group_id_and_token_status_dict, model_short_name, docObj, coref_tokenStr, spacy_tokenStr, baseTokenObj)

                            # TokB,TokC of TokA+TokB+TokC
                            if isnan(coref_index) and spacy_tokenStr in last_coref_token:
                                many2one_spacy_token_len += len(spacy_tokenStr)
                                logger.debug("%s) Many2one token: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                                baseTokenObj = docObj.auto_get_token_byIndex(spacy_index, spacy_tokenStr)
                                baseTokenObj.update_token_details(model_short_name, "subset_of")
                                checkpoint_baseTokenObj = docObj.get_token_byIndex(df_aligned.loc[checkpoint_token_index, config.df_col.spacy_index])
                                # Inherit the voting results from TokA'
                                baseTokenObj.inherit_vote(checkpoint_baseTokenObj.vote_details)
                                baseTokenObj.inherit_mention(model_short_name, checkpoint_baseTokenObj, coref_tokenStr, spacy_tokenStr)

                                if many2one_spacy_token_len == len(last_coref_token):
                                    many2one_spacy_token_len = 0
                                    logger.debug("%s) Many2one token end")

                            # TokA exist, TokA' not exist
                            if isnan(coref_index) and spacy_tokenStr not in last_coref_token:
                                logger.debug("%s) Empty coref token: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                                baseTokenObj = docObj.auto_get_token_byIndex(spacy_index, spacy_tokenStr)
                                baseTokenObj.update_token_details(model_short_name, "not_found")
                                baseTokenObj.vote(model_short_name, isNot_empty_or_NaN(coref_group))
                                update_mention_and_mention_group(group_id_and_token_status_dict, model_short_name, docObj, coref_tokenStr, spacy_tokenStr, baseTokenObj)

                for tok in docObj.token_list:
                    print(tok)
                return


if __name__ == "__main__":
    sys.argv.append("+statistic/coref_voting@_global_=i2b2")
    main()  # pylint: disable=no-value-for-parameter
