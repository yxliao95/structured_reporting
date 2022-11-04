from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Event
import json
import logging
import os
import sys
import time
import ast
from datetime import datetime
import math
from math import isnan
import traceback
from tqdm import tqdm

import hydra
import pandas as pd
from omegaconf import OmegaConf

# pylint: disable=import-error,wrong-import-order
from common_utils.common_utils import check_and_create_dirs, check_and_remove_dirs
from common_utils.coref_utils import auto_append_value_to_list
from common_utils.file_checker import FileChecker

logger = logging.getLogger()
module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")

FILE_CHECKER = FileChecker()
START_EVENT = Event()

# Utils


class Votable:
    def __init__(self) -> None:
        self.vote_details: dict[str, int] = {}  # {"ml": 0, "rb": 1, "fj": 1}

    def new_vote(self, model_short_name, vote: bool):
        self.vote_details[model_short_name] = 1 if vote else 0

    def get_vote_count(self):
        return sum([i for _, i in self.vote_details.items()])


class SpacyToken(Votable):
    """ The token from spaCy """

    def __init__(self, index, tokenStr) -> None:
        super().__init__()
        self.tokenStr = tokenStr
        self.index = index
        self.token_details: dict[str, object] = {}
        self.vote_details: dict[str, int] = {}  # {"ml": 0, "rb": 1, "fj": 1}
        self.of_which_mentions_details: dict[str, list[MentionClass]] = {}  # The key (which is model_short_name) would only appear when there is coref result

    def update_vote(self, model_short_name, has_mention: bool):
        """ Update the voting results only if `has_mention` is True """
        if has_mention:
            self.vote_details[model_short_name] = 1

    def get_vote_confidence(self):
        return sum([i for _, i in self.vote_details.items()])/len(self.vote_details)

    def inherit_vote(self, last_vote_details: dict):
        self.vote_details = last_vote_details.copy()

    def inherit_mention(self, model_short_name: str, checkpoint_token: "SpacyToken", coref_tokenStr: str, spacy_tokenStr: str):
        """ This is a newly generated token, copy the info from checkpoint and add this token to the info """
        for mentionObj in checkpoint_token.of_which_mentions_details.get(model_short_name, []):
            mentionObj.update_mention(coref_tokenStr, spacy_tokenStr, self)
            self.update_mention_details(model_short_name, mentionObj)

    def update_token_details(self, model_short_name, status: str):
        """Args:
        status: ["equal", "contains", "subset_of", "not_found", "unequal"],
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
            "of_which_mentions_details": [{model: [tok.tokenStr for mention in mentions for tok in mention.spacy_tokens]} for model, mentions in self.of_which_mentions_details.items()],
        }, indent=2)

        # This method gets called when using == on the object
    def __eq__(self, other: "SpacyToken"):
        if not isinstance(other, SpacyToken):
            return False
        if self.index != other.index or self.tokenStr != other.tokenStr:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(f"{self.index} + {self.tokenStr}")


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

        self.confidence = 0  # Mention selection threshold. 0 <= range <= 1. Avg sum of tokens' vote percentages

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
            "of_which_coref_groups id": [id for id, group in self.of_which_coref_groups.items()]
        }, indent=2)

    # This method gets called when using == on the object
    def __eq__(self, other: "MentionClass"):
        if not isinstance(other, MentionClass):
            return False
        mentionA_tokList = self.spacy_tokens
        mentionB_tokList = other.spacy_tokens
        if len(mentionA_tokList) != len(mentionB_tokList):
            return False
        else:
            for mentionB_tok in mentionB_tokList:
                if mentionB_tok not in mentionA_tokList:
                    return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash("".join([str(tok.__hash__()) for tok in self.spacy_tokens]))


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
            "mention_list id": [i.id for i in self.mention_list],
        }, indent=2)


class MentionPair(Votable):
    def __init__(self, mentionA: MentionClass, mentionB: MentionClass, model_short_name: str) -> None:
        super().__init__()
        super().new_vote(model_short_name, True)
        # Regardless of order.
        self.mention_set = set([mentionA, mentionB])

    def update_vote(self, new_mention_pair: "MentionPair"):
        """ If two dicts have different keys, it would result in two keys. otherwise, the first value will be replaced by the second value """
        self.vote_details.update(new_mention_pair.vote_details)

    def __repr__(self) -> str:
        return json.dumps({
            "mention_set str": [m.mentionStr for m in self.mention_set],
            "mention_set confidence": [m.confidence for m in self.mention_set],
            "mention_list id": self.vote_details,
        }, indent=2)

    # This method gets called when using == on the object
    def __eq__(self, other: "MentionPair"):
        if not isinstance(other, MentionPair):
            return False
        if len(self.mention_set) != len(other.mention_set):
            return False
        if self.mention_set.difference(other.mention_set):
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash("".join([str(mention.__hash__()) for mention in self.mention_set]))


class DocClass:
    def __init__(self) -> None:
        """ The element in token_list is SpacyToken """
        self.filename = ""
        self.token_list: list[SpacyToken] = []
        self.mention_groups_details: dict[str, dict[int, MentionGroupClass]] = {}
        self.valid_mentions: list[MentionClass] = []  # Mentions that consist of valid tokens. (Confidence > 0.66)
        self.invalid_mentions: list[MentionClass] = []
        self.eligible_mention_pairs: list[MentionPair] = []  # Mention pairs that consist of valid mention. Eligible doesn't mean valid, we still need to vote
        self.valid_mention_group: list[set[MentionClass]] = []  # The final result.

    def add_mention_pair_and_update_vote(self, new_mention_pair):
        """ If the new mention_pair exist, update the mention_pair vote_details. Otherwise add to the list. """
        if new_mention_pair in self.eligible_mention_pairs:
            old_mention_pair = self.eligible_mention_pairs[self.eligible_mention_pairs.index(new_mention_pair)]
            old_mention_pair.update_vote(new_mention_pair)
        else:
            self.eligible_mention_pairs.append(new_mention_pair)

    def get_token_byIndex(self, idx) -> SpacyToken:
        return self.token_list[int(idx)]

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


def isNot_empty_or_NaN(item_str) -> bool:
    """ Empty means to "-1", NaN refers to math.nan """
    if item_str not in ["-1.0", "-1", str(math.nan)]:
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


def create_or_update_mention_and_mention_group(group_id_and_token_status_dict: dict, model_short_name: str, docObj: DocClass, coref_tokenStr: str, spacy_tokenStr: str, baseTokenObj: SpacyToken):
    """ Generate or update mention according to the current aligned token """
    for group_id, tok_status in group_id_and_token_status_dict.items():
        # Any tokens that is singular mention token or is a first mention token will generate a new mention.
        if tok_status == "single" or tok_status == "first":
            mentionObj = MentionClass(model_short_name)
            mentionObj.init_mention(docObj, coref_tokenStr, spacy_tokenStr, baseTokenObj, group_id)
        elif tok_status == "inter" or tok_status == "last":
            mentionGroupClass = docObj.auto_get_mention_group(model_short_name, group_id)
            try:
                mentionObj = mentionGroupClass.get_last_mention()
            except IndexError:
                logger.error("Error occured in %s", docObj.filename)
                logger.error(traceback.format_exc())
                raise
            mentionObj.update_mention(coref_tokenStr, spacy_tokenStr, baseTokenObj)
        baseTokenObj.update_mention_details(model_short_name, mentionObj)


def resolve_voting_info(config, df_spacy, section_name, file_name) -> DocClass:
    """ Read a file, do statistic, and put all the necessary infomation into `DocClass` """
    source_cfg = config.input.source
    spacy_name_style_cfg = config.name_style.spacy.column_name
    # Get the info of three coref models
    candidate_coref_model_cfg_list = [(model_short_name, source_cfg.coref_models.get(model_short_name)) for model_short_name in source_cfg.in_use]

    docObj = DocClass()
    docObj.filename = file_name
    # Loop the models
    for model_short_name, coref_model_cfg in candidate_coref_model_cfg_list:
        logger.debug("model_short_name %s", model_short_name)
        coref_column_cfg = coref_model_cfg.target_column

        # Read coref model output to be aligned
        coref_model_output_path = os.path.join(coref_model_cfg.dir, section_name, file_name)
        df_coref = pd.read_csv(coref_model_output_path, index_col=0, na_filter=False)
        # Comment out this line to prevent error when we want to use gt for ensemble majority voting.
        # It should not affect the normal procedure as only gt csv files use [sp]token and [sp]sentence_group
        df_coref = df_coref.rename(columns={"[sp]token": "[gt]token", "[sp]sentence_group": "[gt]sentence_group"})

        # Merge and align two df. Keep the index for later use
        df_spacy[config.df_col.spacy_index] = df_spacy.index
        df_coref[config.df_col.coref_index] = df_coref.index
        # Columns with NaNs are will be float dtype such as coref_group and coref_group_conll
        df_aligned = df_spacy.merge(df_coref, how="outer", left_index=True, right_on=coref_column_cfg.spacy_index).reset_index().drop(columns=["index"])

        # Token level alignment. Assuming spacy has token A B C, corenlp has token A' B' C'
        checkpoint_token_index = 0

        one2many_flag = False
        many2one_flag = False
        many2many_flag = False
        left_str = ""
        right_str = ""
        # with pd.option_context('display.max_rows', 9999, 'display.max_columns', None):  # more options can be specified also
        #     print(df_aligned)
        for _idx, _itemSeries in df_aligned.iterrows():
            spacy_index = _itemSeries.get(config.df_col.spacy_index)
            spacy_index = int(spacy_index) if not isnan(spacy_index) else spacy_index
            coref_index = _itemSeries.get(config.df_col.coref_index)
            coref_index = int(coref_index) if not isnan(coref_index) else coref_index

            spacy_tokenStr = _itemSeries.get(spacy_name_style_cfg.token)
            spacy_tokenStr_isnan = False if isinstance(spacy_tokenStr, str) else isnan(spacy_tokenStr)
            spacy_tokenStr = str(spacy_tokenStr)
            coref_tokenStr = _itemSeries.get(coref_column_cfg.token)
            coref_tokenStr_isnan = False if isinstance(coref_tokenStr, str) else isnan(coref_tokenStr)
            coref_tokenStr = str(coref_tokenStr)

            coref_group = str(_itemSeries.get(coref_column_cfg.coref_group))  # "-1", "nan", "[1, 11]"
            coref_group_conll = str(_itemSeries.get(coref_column_cfg.coref_group_conll))  # "-1", "nan", "['(1)', '(11']"
            group_id_and_token_status_dict: dict[int, str] = get_group_id_and_token_status(coref_group, coref_group_conll)

            prev_spacy_tokenStr = str(df_aligned.loc[checkpoint_token_index, spacy_name_style_cfg.token])
            prev_coref_tokenStr = df_aligned.loc[checkpoint_token_index, coref_column_cfg.token]
            prev_coref_tokenStr = str(prev_coref_tokenStr)
            # TokA = TokA'
            try:
                if spacy_tokenStr == coref_tokenStr:
                    logger.debug("%s) Identical token: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                    checkpoint_token_index = _idx
                    # Token-level operation
                    baseTokenObj = docObj.auto_get_token_byIndex(spacy_index, spacy_tokenStr)
                    baseTokenObj.update_token_details(model_short_name, "equal")
                    baseTokenObj.new_vote(model_short_name, isNot_empty_or_NaN(coref_group))
                    # Mention-level and Mention-Group-level
                    create_or_update_mention_and_mention_group(group_id_and_token_status_dict, model_short_name, docObj, coref_tokenStr, spacy_tokenStr, baseTokenObj)
                    # In case tokA != tokA', tokB = tokB'
                    one2many_flag, many2one_flag, many2many_flag = False, False, False
                    left_str, right_str = "", ""
                else:
                    # One2many: TokA = TokA'+TokB'+TokC', TokA contains TokA'
                    # Or, Many2many: TokA+TokC = TokA'+TokB', TokA contains TokA'. For example: df_aligned
                    # .h    .       |   .h.s    .       |   .h      .       (one2many)
                    # .h    h.s.    |   .h.s    h.s.    |   .h      h.      (many2many)
                    # .s    NaN     |   .       NaN     |   .s      s.      (many2many)
                    # .     NaN     |                   |   .       NaN     (many2many)
                    if coref_tokenStr in spacy_tokenStr and not spacy_tokenStr_isnan and not coref_tokenStr_isnan:
                        if not one2many_flag:
                            logger.debug("%s) One2many token start: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                            baseTokenObj = docObj.auto_get_token_byIndex(spacy_index, spacy_tokenStr)
                            baseTokenObj.update_token_details(model_short_name, "contains")
                            baseTokenObj.new_vote(model_short_name, isNot_empty_or_NaN(coref_group))
                            create_or_update_mention_and_mention_group(group_id_and_token_status_dict, model_short_name, docObj, coref_tokenStr, spacy_tokenStr, baseTokenObj)
                            checkpoint_token_index = _idx
                            left_str += spacy_tokenStr
                            right_str += coref_tokenStr
                            one2many_flag = True
                        else:
                            logger.debug("%s) One2many token: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                            baseTokenObj = docObj.get_token_byIndex(spacy_index)  # The same token generated above.
                            # Any mention that found in TokA',TokB',TokC' will vote on TokA. Thus, vote again.
                            baseTokenObj.update_vote(model_short_name, isNot_empty_or_NaN(coref_group))
                            create_or_update_mention_and_mention_group(group_id_and_token_status_dict, model_short_name, docObj, coref_tokenStr, spacy_tokenStr, baseTokenObj)
                            right_str += coref_tokenStr

                    # Many2one: TokA+TokB+TokC = TokA', TokA is subset_of TokA', TokA of TokA+TokB+TokC
                    # Or, Many2many: TokA+TokB = TokA'+TokC', TokA is subset_of TokA'
                    # Or Many2one -> One2many
                    # '     's  (Many2one)
                    # s.    .   (One2many)
                    elif spacy_tokenStr in coref_tokenStr and not spacy_tokenStr_isnan and not coref_tokenStr_isnan:
                        logger.debug("%s) Many2one token start: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                        baseTokenObj = docObj.auto_get_token_byIndex(spacy_index, spacy_tokenStr)
                        baseTokenObj.update_token_details(model_short_name, "subset_of")
                        baseTokenObj.new_vote(model_short_name, isNot_empty_or_NaN(coref_group))
                        create_or_update_mention_and_mention_group(group_id_and_token_status_dict, model_short_name, docObj, coref_tokenStr, spacy_tokenStr, baseTokenObj)
                        checkpoint_token_index = _idx
                        left_str += spacy_tokenStr
                        right_str += coref_tokenStr
                        many2one_flag = True

                    # Many2one: TokB,TokC of TokA+TokB+TokC
                    elif isnan(coref_index) and spacy_tokenStr in prev_coref_tokenStr and not many2many_flag:
                        logger.debug("%s) Many2one token: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                        baseTokenObj = docObj.auto_get_token_byIndex(spacy_index, spacy_tokenStr)
                        baseTokenObj.update_token_details(model_short_name, "subset_of")
                        checkpoint_baseTokenObj = docObj.get_token_byIndex(df_aligned.loc[checkpoint_token_index, config.df_col.spacy_index])
                        # Copy from previous
                        baseTokenObj.inherit_vote(checkpoint_baseTokenObj.vote_details)
                        baseTokenObj.inherit_mention(model_short_name, checkpoint_baseTokenObj, coref_tokenStr, spacy_tokenStr)
                        left_str += spacy_tokenStr

                    # TokA exist, TokA' not exist, not many2one, and not many2many
                    elif isnan(coref_index) and spacy_tokenStr not in prev_coref_tokenStr and not many2many_flag:
                        logger.debug("%s) Empty coref token: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                        baseTokenObj = docObj.auto_get_token_byIndex(spacy_index, spacy_tokenStr)
                        baseTokenObj.update_token_details(model_short_name, "not_found")
                        baseTokenObj.new_vote(model_short_name, isNot_empty_or_NaN(coref_group))
                        create_or_update_mention_and_mention_group(group_id_and_token_status_dict, model_short_name, docObj, coref_tokenStr, spacy_tokenStr, baseTokenObj)
                        one2many_flag, many2one_flag, many2many_flag = False, False, False
                        left_str, right_str = "", ""

                    # Many2many: rest of the tokens.
                    # Three examples: (spacy_tok  coref_tok) in row 2-4
                    # .h    .       |   .h.s    .       |   .h      .
                    # .h    h.s.    |   .h.s    h.s.    |   .h      h.      (many2many)
                    # .s    NaN     |   .       NaN     |   .s      s.      (many2many)
                    # .     NaN     |                   |   .       NaN     (many2many)
                    else:
                        logger.debug("%s) Change to many2many token : %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                        many2many_flag = True

                        if one2many_flag:
                            # If currTok == prevTok, get the tok from checkpoint and update with curr row info
                            if spacy_tokenStr == prev_spacy_tokenStr:
                                baseTokenObj = docObj.auto_get_token_byIndex(spacy_index, spacy_tokenStr)  # Get the same token generated previously, since they have the same spacy_index
                                baseTokenObj.update_token_details(model_short_name, "unequal")
                                baseTokenObj.update_vote(model_short_name, isNot_empty_or_NaN(coref_group))
                                create_or_update_mention_and_mention_group(group_id_and_token_status_dict, model_short_name, docObj, coref_tokenStr, spacy_tokenStr, baseTokenObj)
                                checkpoint_token_index = _idx
                                right_str += coref_tokenStr

                            elif spacy_tokenStr != prev_spacy_tokenStr:
                                baseTokenObj = docObj.auto_get_token_byIndex(spacy_index, spacy_tokenStr)  # Create a new one
                                baseTokenObj.update_token_details(model_short_name, "unequal")
                                # Just like a new token
                                if not isnan(coref_index):
                                    baseTokenObj.new_vote(model_short_name, isNot_empty_or_NaN(coref_group))
                                    create_or_update_mention_and_mention_group(group_id_and_token_status_dict, model_short_name, docObj, coref_tokenStr, spacy_tokenStr, baseTokenObj)
                                    checkpoint_token_index = _idx
                                    left_str += spacy_tokenStr
                                    right_str += coref_tokenStr

                                # If is NaN, use the info from checkpoint row (rather than checkpoint tok, because checkpoint tok might have other rows' info)
                                else:
                                    prev_coref_group = str(df_aligned.loc[checkpoint_token_index, coref_column_cfg.coref_group])
                                    prev_coref_group_conll = str(df_aligned.loc[checkpoint_token_index, coref_column_cfg.coref_group_conll])
                                    prev_group_id_and_token_status_dict: dict[int, str] = get_group_id_and_token_status(prev_coref_group, prev_coref_group_conll)
                                    # Modify token status: first->inter, single->last
                                    # This operation is fine even there are more than one NaN coref tokens, since "inter" and "last" behave the same (update rather than generate)
                                    for _gid, _tok_status in prev_group_id_and_token_status_dict.items():
                                        if prev_group_id_and_token_status_dict[_gid] == "first":
                                            prev_group_id_and_token_status_dict[_gid] = "inter"
                                        if prev_group_id_and_token_status_dict[_gid] == "single":
                                            prev_group_id_and_token_status_dict[_gid] = "last"
                                    baseTokenObj.new_vote(model_short_name, isNot_empty_or_NaN(prev_coref_group))
                                    create_or_update_mention_and_mention_group(prev_group_id_and_token_status_dict, model_short_name, docObj, coref_tokenStr, spacy_tokenStr, baseTokenObj)
                                    left_str += spacy_tokenStr

                        # .     .h      (many2one)
                        # h.s.  .s.
                        elif many2one_flag:
                            baseTokenObj = docObj.auto_get_token_byIndex(spacy_index, spacy_tokenStr)  # Create a new one
                            baseTokenObj.update_token_details(model_short_name, "unequal")
                            baseTokenObj.new_vote(model_short_name, isNot_empty_or_NaN(coref_group))
                            create_or_update_mention_and_mention_group(group_id_and_token_status_dict, model_short_name, docObj, coref_tokenStr, spacy_tokenStr, baseTokenObj)
                            checkpoint_token_index = _idx
                            left_str += spacy_tokenStr
                            right_str += coref_tokenStr

                        # prevTok = prevTok', TokA != TokA' (neither contains nor subset_of).
                        # Just simply create a spacy tokenObj
                        else:
                            logger.debug("%s) Unequal token: %s | %s", _idx, spacy_tokenStr.encode(), coref_tokenStr.encode())
                            checkpoint_token_index = _idx
                            # Token-level operation
                            baseTokenObj = docObj.auto_get_token_byIndex(spacy_index, spacy_tokenStr)
                            baseTokenObj.update_token_details(model_short_name, "unequal")
                            baseTokenObj.new_vote(model_short_name, isNot_empty_or_NaN(coref_group))
                            # Mention-level and Mention-Group-level
                            create_or_update_mention_and_mention_group(group_id_and_token_status_dict, model_short_name, docObj, coref_tokenStr, spacy_tokenStr, baseTokenObj)
                            # In case tokA != tokA', tokB = tokB'
                            one2many_flag, many2one_flag, many2many_flag = False, False, False
                            left_str, right_str = "", ""

                    if True in [one2many_flag, many2one_flag, many2many_flag] and left_str.replace(" ", "") == right_str.replace(" ", ""):
                        logger.debug("%s) Multi-alignment end", _idx)
                        one2many_flag, many2one_flag, many2many_flag = False, False, False
                        left_str, right_str = "", ""

            except Exception:
                logger.error("Error occured in %s", docObj.filename)
                logger.error(traceback.format_exc())
                raise

    return docObj


def compute_voting_result(config, docObj: DocClass) -> list[set[MentionClass]]:
    """ 1. Get valid mention by votes counting on its tokens
    2. Get eligible mention pairs via linking the valid mentions within a mention group
    3. Get valid mention groups by votes counting on eligible mention pairs and grouping valid mention pairs

    The intermediate results are saved in docObj
    """
    for model_short_name, mention_groups_dict in docObj.mention_groups_details.items():
        for group_id, mentionGroupObj in mention_groups_dict.items():
            # Remove invalid mentions
            valid_mentions_in_this_group: list[MentionClass] = []
            for mentionObj in mentionGroupObj.mention_list:
                # Compute tokens' vote count
                for tokenObj in mentionObj.spacy_tokens:
                    mentionObj.confidence += tokenObj.get_vote_confidence()
                mentionObj.confidence = mentionObj.confidence/len(mentionObj.spacy_tokens)

                if mentionObj.confidence > config.voting.confidence:
                    docObj.valid_mentions.append(mentionObj)
                    valid_mentions_in_this_group.append(mentionObj)
                else:
                    docObj.invalid_mentions.append(mentionObj)

            # Ignore mention groups that have only singleton or has less than two valid mention
            if len(valid_mentions_in_this_group) > 1:
                # Construct mention pairs and add into docObj.
                for curr_idx, curr_mention in enumerate(valid_mentions_in_this_group):
                    next_idx = curr_idx+1
                    while next_idx < len(valid_mentions_in_this_group):
                        next_mention = valid_mentions_in_this_group[next_idx]
                        mention_pair = MentionPair(curr_mention, next_mention, model_short_name)
                        docObj.add_mention_pair_and_update_vote(mention_pair)
                        next_idx += 1

    valid_mention_group: list[set[MentionClass]] = []
    mention_pointer: dict[MentionClass, int] = {}
    i = 0
    for mention_pair in docObj.eligible_mention_pairs:
        if mention_pair.get_vote_count() >= config.voting.mention_pair_threshold:
            i += 1
            pairs = list(mention_pair.mention_set)
            mentionA = pairs[0]
            mentionB = pairs[1]
            if mentionA not in mention_pointer and mentionB not in mention_pointer:
                mention_pointer[mentionA] = len(valid_mention_group)
                mention_pointer[mentionB] = len(valid_mention_group)
                valid_mention_group.append(mention_pair.mention_set.copy())

            elif mentionA in mention_pointer and mentionB not in mention_pointer:
                group_idx = mention_pointer[mentionA]
                mention_pointer[mentionB] = group_idx
                valid_mention_group[group_idx].add(mentionB)

            elif mentionA not in mention_pointer and mentionB in mention_pointer:
                group_idx = mention_pointer[mentionB]
                mention_pointer[mentionA] = group_idx
                valid_mention_group[group_idx].add(mentionA)

            elif mentionA in mention_pointer and mentionB in mention_pointer:
                try:
                    if mention_pointer[mentionA] != mention_pointer[mentionB]:
                        # Merge group A and B
                        sourted_groupId = sorted([mention_pointer[mentionA], mention_pointer[mentionB]])
                        valid_mention_group[sourted_groupId[0]].update(valid_mention_group[sourted_groupId[1]])
                        # Update pointer
                        for mention in valid_mention_group[sourted_groupId[1]]:
                            mention_pointer[mention] = sourted_groupId[0]
                        mention_pointer[mentionA] = sourted_groupId[0]
                        mention_pointer[mentionB] = sourted_groupId[0]
                        # Empty the old group
                        valid_mention_group[sourted_groupId[1]] = None
                    else:
                        group_idx = mention_pointer[mentionA]
                        assert mentionA in valid_mention_group[group_idx] and mentionB in valid_mention_group[group_idx]
                except AssertionError:
                    logger.error("Error occur in %s, at loop %s", docObj.filename, i)
                    logger.error(traceback.format_exc())
                    raise
    # Remove empty group
    valid_mention_group = [group for group in valid_mention_group if group is not None]
    docObj.valid_mention_groups = valid_mention_group

    return valid_mention_group


def get_output_df(config, df_spacy, valid_mention_group, docObj: DocClass) -> pd.DataFrame:
    spacy_nametyle = config.name_style.spacy.column_name
    voting_namestyle = config.name_style.voting.column_name

    doc_tok_len = len(df_spacy)
    tokVote = [-1] * doc_tok_len
    tokVoteCount = [-1] * doc_tok_len
    mention = [-1] * doc_tok_len
    mentionConfidence = [-1] * doc_tok_len
    for tokenObj in docObj.token_list:
        tokVote[tokenObj.index] = [i for _, i in tokenObj.vote_details.items()]
        tokVoteCount[tokenObj.index] = tokenObj.get_vote_count()
        mention[tokenObj.index] = [f"{model_name}:{[mentionObj.id for mentionObj in mentionObj_list]}" for model_name, mentionObj_list in tokenObj.of_which_mentions_details.items()]
        mentionConfidence[tokenObj.index] = [f"{mentionObj.id}:{mentionObj.confidence:.3f}" for _, mentionObj_list in tokenObj.of_which_mentions_details.items() for mentionObj in mentionObj_list]

    mentionPair = [-1] * doc_tok_len
    mentionPairVote = [-1] * doc_tok_len
    mentionPair_voteCount = [-1] * doc_tok_len
    for pair_id, mentionPairObj in enumerate(docObj.eligible_mention_pairs):
        voteCount = mentionPairObj.get_vote_count()
        for mentionObj in mentionPairObj.mention_set:
            for tokenObj in mentionObj.spacy_tokens:
                auto_append_value_to_list(mentionPair, tokenObj.index, pair_id)
                auto_append_value_to_list(mentionPairVote, tokenObj.index, f"{pair_id}:{[i for _, i in mentionPairObj.vote_details.items()]}")
                auto_append_value_to_list(mentionPair_voteCount, tokenObj.index, f"{pair_id}:{voteCount}")

    coref_group = [-1] * doc_tok_len
    coref_group_conll = [-1] * doc_tok_len
    for group_id, mention_set in enumerate(valid_mention_group):
        for mentionObj in mention_set:
            mention_indices = []
            # For coref_group
            for tokenObj in mentionObj.spacy_tokens:
                mention_indices.append(tokenObj.index)
                auto_append_value_to_list(coref_group, tokenObj.index, group_id)
            # For coref_group_conll
            sorted_indices = sorted(mention_indices)
            if len(sorted_indices) == 1:
                auto_append_value_to_list(coref_group_conll, sorted_indices[0], f"({group_id})")
            elif len(sorted_indices) > 1:
                auto_append_value_to_list(coref_group_conll, sorted_indices[0], f"({group_id}")
                auto_append_value_to_list(coref_group_conll, sorted_indices[-1], f"{group_id})")

    df_out = df_spacy.loc[:, [spacy_nametyle.token, spacy_nametyle.sentence_group]]
    df_out[voting_namestyle.coref_group] = coref_group
    df_out[voting_namestyle.coref_group_conll] = coref_group_conll
    df_out[voting_namestyle.tok_vote] = tokVote
    df_out[voting_namestyle.tok_vote_count] = tokVoteCount
    df_out[voting_namestyle.mention] = mention
    df_out[voting_namestyle.mention_confidence] = mentionConfidence
    df_out[voting_namestyle.mention_pair] = mentionPair
    df_out[voting_namestyle.mention_pair_vote] = mentionPairVote
    df_out[voting_namestyle.mention_pair_vote_count] = mentionPair_voteCount

    return df_out


def batch_processing(config, spacy_file_path, section_name, file_name):
    """ Voting on one document """
    # if file_name != "clinical-277.csv":
    #     return None

    START_EVENT.wait()
    logger.debug("Processing file %s", file_name)

    # Read spacy output as alignment base
    df_spacy = pd.read_csv(spacy_file_path, index_col=0, na_filter=False)
    # Some of the i2b2 raw files are utf-8 start with DOM, but we didn't remove the DOM character, thus we fix it here.
    df_spacy.iloc[0] = df_spacy.iloc[0].apply(lambda x: x.replace("\ufeff", "").replace("\xef\xbb\xbf", "") if isinstance(x, str) else x)

    docObj: DocClass = resolve_voting_info(config, df_spacy, section_name, file_name)
    valid_mention_group: list[set[MentionClass]] = compute_voting_result(config, docObj)
    df_out = get_output_df(config, df_spacy, valid_mention_group, docObj)

    output_dir = os.path.join(config.output.files_dir, section_name)
    check_and_create_dirs(output_dir)
    output_file_path = os.path.join(output_dir, file_name)

    df_out.to_csv(output_file_path)

    return f"{file_name} done."


@ hydra.main(version_base=None, config_path=config_path, config_name="coreference_resolution")
def main(config):
    print(OmegaConf.to_yaml(config))
    source_cfg = config.input.source

    # Remove history dir
    check_and_remove_dirs(config.output.files_dir, config.clear_history)

    # Loop in sections
    startTime = time.time()
    section_name_list = source_cfg.section
    for section_name in section_name_list:
        logger.info("Voting on section: %s", section_name)

        spacy_out_dir = os.path.join(source_cfg.baseline_model.dir, section_name)
        if not os.path.exists(spacy_out_dir):
            logger.error("Could not found the target dir (the baseline for token alignment): %s", spacy_out_dir)

        candidate_fileName_list = FILE_CHECKER.filter(os.listdir(spacy_out_dir))

        # Loop all files
        with ProcessPoolExecutor(max_workers=config.thread.workers) as executor:
            all_task = []
            for file_name in tqdm(candidate_fileName_list):
                spacy_file_path = os.path.join(spacy_out_dir, file_name)
                all_task.append(executor.submit(batch_processing, config, spacy_file_path, section_name, file_name))

            # Notify tasks to start
            START_EVENT.set()

            # When a submitted task finished, the output is received here.
            if all_task:
                for future in tqdm(as_completed(all_task), total=len(all_task)):
                    msg = future.result()
                    logger.debug(msg)
                logger.info("Done.")
            else:
                logger.info("All empty. Skipped.")

            executor.shutdown(wait=True, cancel_futures=False)
            START_EVENT.clear()

        # Log runtime information
        check_and_create_dirs(config.output.base_dir)

        with open(os.path.join(config.output.base_dir, config.output.log_file_name), "a", encoding="UTF-8") as f:
            log_out = {
                "Using": {
                    "Voting stategy": config.voting.strategy,
                },
                "Section": section_name,
                "Number of processed records": len(all_task),
                "Time cost": f"{time.time() - startTime:.2f}s",
                "Datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            f.write(json.dumps(log_out, indent=2))
            f.write("\n\n")


if __name__ == "__main__":
    sys.argv.append("+coreference_resolution/coref_voting@_global_=mimic_cxr")
    main()  # pylint: disable=no-value-for-parameter
