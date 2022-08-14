import logging
import random
import time
from multiprocessing import Process

import requests
from pandas import DataFrame
from common_utils.nlp_utils import resolveTokenIndices_byPosition

logger = logging.getLogger()


class CorenlpProcess(Process):
    def __init__(self, request_url, progressId, input_pipe, input_text_list, input_sid_list):
        super().__init__()
        self.request_url = request_url
        self.id = random.randint(100, 999)
        self.processId = progressId
        self.input_pipe = input_pipe
        self.input_text_list = input_text_list
        self.input_sid_list = input_sid_list

    def run(self):
        logger.debug("CoreNLP Process [%s-%s] started.", self.processId, self.id)
        time0 = time.time()
        dataDict = {}  # A dict where key=sid, value=coreNLP output in json string
        for text, sid in zip(self.input_text_list, self.input_sid_list):
            corenlp_out_jsonStr = requests.post(self.request_url, data=text.encode()).text
            dataDict[sid] = corenlp_out_jsonStr
        time1 = time.time()
        data = {"processId": self.id, "data": dataDict}
        logger.debug("CoreNLP Process [%s-%s] finished in %ss", self.processId, self.id, time1-time0)
        self.input_pipe.send(data)


def formatCorenlpDocument(tokenOffset_base: DataFrame, corenlp_json):
    referTo_base = [
        resolveTokenIndices_byPosition(tokenOffset_base, token['characterOffsetBegin'], token['characterOffsetEnd'] - token['characterOffsetBegin'])[0]
        for sentence in corenlp_json['sentences']
        for token in sentence['tokens']
    ]

    tokenTotalNum = 0
    sentenceFirstTokenIndex_offset = [0]
    for sentId, sentence in enumerate(corenlp_json['sentences']):
        tokenNum = len(sentence['tokens'])
        tokenTotalNum += tokenNum
        nextOffset = sentenceFirstTokenIndex_offset[sentId] + tokenNum
        sentenceFirstTokenIndex_offset.append(nextOffset)

    dependency_list = []
    depPlus_list = []
    depPlusPlus_list = []
    for sentId, sentence in enumerate(corenlp_json['sentences']):
        for basicDep in sentence['basicDependencies']:
            depTag = basicDep['dep']
            headTok = basicDep['governorGloss']
            headTokIdx = basicDep['governor'] - 1 + sentenceFirstTokenIndex_offset[sentId]
            headTokIdx_inBase = referTo_base[headTokIdx]
            currentTok = basicDep['dependentGloss']
            currentTokIdx = basicDep['dependent'] - 1 + sentenceFirstTokenIndex_offset[sentId]
            currentTokIdx_inBase = referTo_base[currentTokIdx]
            # Make the root dep token point to itself, just like what spacy did.
            if depTag == "ROOT" and headTok == "ROOT" and basicDep['governor'] == 0:
                headTok = currentTok
                headTokIdx = currentTokIdx
                headTokIdx_inBase = currentTokIdx_inBase
            dependency_list.append({
                'index': currentTokIdx,
                'extra_str': f"{depTag}|{headTok}|{headTokIdx_inBase}",  # We use headTokIdx_inBase as all the rows/tokens will finally align to df_base (df_spacy)
            })
        for depPlus in sentence['enhancedDependencies']:
            depTag = depPlus['dep']
            headTok = depPlus['governorGloss']
            headTokIdx = depPlus['governor'] - 1 + sentenceFirstTokenIndex_offset[sentId]
            headTokIdx_inBase = referTo_base[headTokIdx]
            currentTok = depPlus['dependentGloss']
            currentTokIdx = depPlus['dependent'] - 1 + sentenceFirstTokenIndex_offset[sentId]
            currentTokIdx_inBase = referTo_base[currentTokIdx]
            # Make the root dep token point to itself, just like what spacy did.
            if depTag == "ROOT" and headTok == "ROOT" and depPlus['governor'] == 0:
                headTok = currentTok
                headTokIdx = currentTokIdx
                headTokIdx_inBase = currentTokIdx_inBase
            depPlus_list.append({
                'index': currentTokIdx,
                'extra_str': f"{depTag}|{headTok}|{headTokIdx_inBase}",  # We use headTokIdx_inBase as all the rows/tokens will finally align to df_base (df_spacy)
            })
        for depPlusPlus in sentence['enhancedPlusPlusDependencies']:
            depTag = depPlusPlus['dep']
            headTok = depPlusPlus['governorGloss']
            headTokIdx = depPlusPlus['governor'] - 1 + sentenceFirstTokenIndex_offset[sentId]
            headTokIdx_inBase = referTo_base[headTokIdx]
            currentTok = depPlusPlus['dependentGloss']
            currentTokIdx = depPlusPlus['dependent'] - 1 + sentenceFirstTokenIndex_offset[sentId]
            currentTokIdx_inBase = referTo_base[currentTokIdx]
            # Make the root dep token point to itself, just like what spacy did.
            if depTag == "ROOT" and headTok == "ROOT" and depPlusPlus['governor'] == 0:
                headTok = currentTok
                headTokIdx = currentTokIdx
                headTokIdx_inBase = currentTokIdx_inBase
            depPlusPlus_list.append({
                'index': currentTokIdx,
                'extra_str': f"{depTag}|{headTok}|{headTokIdx_inBase}",  # We use headTokIdx_inBase as all the rows/tokens will finally align to df_base (df_spacy)
            })

    corefGroups = []
    corefMetionGroups_withData = []
    for corefChain in corenlp_json['corefs'].values():
        corefGroup = []
        for mention in corefChain:
            sentenceFirstIndexStart = sentenceFirstTokenIndex_offset[mention['sentNum']-1]
            beginIndex = sentenceFirstIndexStart + mention['startIndex'] - 1
            endIndex = sentenceFirstIndexStart + mention['endIndex'] - 1  # The index of the next token of the target token
            mentionIndices = [i for i in range(beginIndex, endIndex)]
            corefGroup.append(mentionIndices)
            mentionType = mention['type']
            corefMetionGroups_withData.append(
                {"indices": mentionIndices, "extra_str": mentionType, }
            )
        # If flatten, then we can't not recognize the boundaries of the mentions
        # corefGroup_flatten = [indices for mention in corefGroup for indices in mention]
        corefGroups.append(corefGroup)
    return referTo_base, tokenTotalNum, corefMetionGroups_withData, corefGroups, dependency_list, depPlus_list, depPlusPlus_list
