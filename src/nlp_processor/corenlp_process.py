import logging
import random
import time
from multiprocessing import Process

import requests
from pandas import DataFrame
from common_utils.nlp_utils import resolveTokenIndices_byPosition  # pylint: disable=import-error

logger = logging.getLogger()


class CorenlpUrlProcess(Process):
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


def formatCorenlpDocument(tokenOffset_base, corenlp_json):
    referTo_base = [
        resolveTokenIndices_byPosition(tokenOffset_base, token['characterOffsetBegin'], token['characterOffsetEnd'] - token['characterOffsetBegin'])[0]
        for sentence in corenlp_json['sentences']
        for token in sentence['tokens']
    ]

    tokenTotalNum = 0
    sentenceGroups = []
    sentenceFirstTokenIndex_offset = [0]
    for sentId, sentence in enumerate(corenlp_json['sentences']):
        tokenNum = len(sentence['tokens'])
        tokenTotalNum += tokenNum
        nextOffset = sentenceFirstTokenIndex_offset[sentId] + tokenNum
        sentenceGroups.extend([sentId]*tokenNum)
        sentenceFirstTokenIndex_offset.append(nextOffset)

    dependency_list = []
    depPlus_list = []
    depPlusPlus_list = []
    for sentId, sentence in enumerate(corenlp_json['sentences']):
        for basicDep in sentence['basicDependencies']:
            depType = basicDep['dep']
            headToken = basicDep['governorGloss']
            headTokenIdx = basicDep['governor'] - 1 + sentenceFirstTokenIndex_offset[sentId]  # refer to corenlp token index
            headToken_toSpacyIdx = referTo_base[headTokenIdx]  # refer to spacy token index
            currentToken = basicDep['dependentGloss']
            currentTokenIdx = basicDep['dependent'] - 1 + sentenceFirstTokenIndex_offset[sentId]  # refer to corenlp token index
            currentToken_toSpacyIdx = referTo_base[currentTokenIdx]  # refer to spacy token index
            # Make the root dep token point to itself, just like what spacy did.
            if depType == "ROOT" and headToken == "ROOT" and basicDep['governor'] == 0:
                headToken = currentToken
                headTokenIdx = currentTokenIdx
                headToken_toSpacyIdx = currentToken_toSpacyIdx
            dependency_list.append({
                'index': currentTokenIdx,
                'extra_str': f"{depType}|{headToken}|{headTokenIdx}",  # headToken or headToken_toSpacyIdx
            })
        for depPlus in sentence['enhancedDependencies']:
            depType = depPlus['dep']
            headToken = depPlus['governorGloss']
            headTokenIdx = depPlus['governor'] - 1 + sentenceFirstTokenIndex_offset[sentId]
            headToken_toSpacyIdx = referTo_base[headTokenIdx]
            currentToken = depPlus['dependentGloss']
            currentTokenIdx = depPlus['dependent'] - 1 + sentenceFirstTokenIndex_offset[sentId]
            currentToken_toSpacyIdx = referTo_base[currentTokenIdx]
            # Make the root dep token point to itself, just like what spacy did.
            if depType == "ROOT" and headToken == "ROOT" and depPlus['governor'] == 0:
                headToken = currentToken
                headTokenIdx = currentTokenIdx
                headToken_toSpacyIdx = currentToken_toSpacyIdx
            depPlus_list.append({
                'index': currentTokenIdx,
                'extra_str': f"{depType}|{headToken}|{headTokenIdx}",  # We use headTokIdx_inBase as all the rows/tokens will finally align to df_base (df_spacy)
            })
        for depPlusPlus in sentence['enhancedPlusPlusDependencies']:
            depType = depPlusPlus['dep']
            headToken = depPlusPlus['governorGloss']
            headTokenIdx = depPlusPlus['governor'] - 1 + sentenceFirstTokenIndex_offset[sentId]
            headToken_toSpacyIdx = referTo_base[headTokenIdx]
            currentToken = depPlusPlus['dependentGloss']
            currentTokenIdx = depPlusPlus['dependent'] - 1 + sentenceFirstTokenIndex_offset[sentId]
            currentToken_toSpacyIdx = referTo_base[currentTokenIdx]
            # Make the root dep token point to itself, just like what spacy did.
            if depType == "ROOT" and headToken == "ROOT" and depPlusPlus['governor'] == 0:
                headToken = currentToken
                headTokenIdx = currentTokenIdx
                headToken_toSpacyIdx = currentToken_toSpacyIdx
            depPlusPlus_list.append({
                'index': currentTokenIdx,
                'extra_str': f"{depType}|{headToken}|{headTokenIdx}",  # We use headTokIdx_inBase as all the rows/tokens will finally align to df_base (df_spacy)
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
    return referTo_base, tokenTotalNum, sentenceGroups, corefMetionGroups_withData, corefGroups, dependency_list, depPlus_list, depPlusPlus_list
