from operator import itemgetter


def align(tokNum, inputTokenGroups):
    alignment = [-1] * tokNum
    for _id, tokenGroup in enumerate(inputTokenGroups):
        alignment[tokenGroup.start: tokenGroup.end] = [_id] * (tokenGroup.end - tokenGroup.start)
    return alignment


def align_byIndex(tokNum, inputIndexGroups):
    alignment = [-1] * tokNum
    for _id, indexGroup in enumerate(inputIndexGroups):
        alignment[indexGroup[0]: indexGroup[-1] + 1] = [_id] * len(indexGroup)
    return alignment


def align_byIndex_individually(length, inputIndexGroups):
    alignment = [-1] * length
    for _id, indexGroup in enumerate(inputIndexGroups):
        for index in indexGroup:
            if alignment[index] == -1:
                alignment[index] = [_id]
            else:
                alignment[index].append(_id)
    return alignment


def align_byIndex_individually_nestedgruop(length, corefGroups):
    """ For coref group, there are mention group inside the coref group """
    alignment = [-1] * length
    for _id, corefGroup in enumerate(corefGroups):
        for mention in corefGroup:
            for index in mention:
                if alignment[index] == -1:
                    alignment[index] = [_id]
                else:
                    alignment[index].append(_id)
    return alignment


def align_coref_groups_in_conll_format(length, corefGroups) -> list:
    """ The coref groups are organized as CoNLL format. """
    alignment = [-1] * length
    for _id, corefGroup in enumerate(corefGroups):
        for mentionTokens in corefGroup:
            if len(mentionTokens) == 1:
                index = mentionTokens[0]
                if alignment[index] == -1:
                    alignment[index] = [f"({_id})"]
                else:
                    alignment[index].append(f"({_id})")
            else:
                start_index = mentionTokens[0]
                if alignment[start_index] == -1:
                    alignment[start_index] = [f"({_id}"]
                else:
                    alignment[start_index].append(f"({_id}")

                end_index = mentionTokens[-1]
                if alignment[end_index] == -1:
                    alignment[end_index] = [f"{_id})"]
                else:
                    alignment[end_index].append(f"{_id})")
    return alignment


def align_byIndex_individually_withData_noOverlap(tokNum, inputIndexGroups_withData):
    alignment = [-1] * tokNum
    for _id, indexGroup_withData in enumerate(inputIndexGroups_withData):
        indexGroup = indexGroup_withData["indices"]
        extra_str = indexGroup_withData["extra_str"]
        for index in indexGroup:
            alignment[index] = f"{_id}|{extra_str}"
    return alignment


def align_byIndex_individually_withData(tokNum, inputIndexGroups_withData):
    alignment = [-1] * tokNum
    for indexGroup_withData in inputIndexGroups_withData:
        indexGroup = indexGroup_withData["indices"]
        extra_str = indexGroup_withData["extra_str"]
        for index in indexGroup:
            if alignment[index] == -1:
                alignment[index] = [extra_str]
            else:
                alignment[index].append(extra_str)
    return alignment


def align_byIndex_individually_withData_dictInList(tokNum, inputDictList):
    alignment = [-1] * tokNum
    for elementDict in inputDictList:
        index = elementDict['index']
        extra_str = elementDict['extra_str']
        if alignment[index] == -1:
            alignment[index] = [extra_str]
        else:
            alignment[index].append(extra_str)
    return alignment


def getTokenOffset(baseText: str, inputTokens):
    startPos = 0
    offset = []
    for token in inputTokens:
        offsetPos = baseText.find(token.text, startPos, len(baseText))
        offset.append(offsetPos)
        startPos = offsetPos + len(token.text)
    return offset


def resolveTokenIndices_byPosition(tokenOffset_base, startPos, length) -> list:
    """ For example: ".h.s." = "." (offset=0) + "h.s." (offset=1)
    Tok_to_be_aligned: ".h"  => tokenOffset: [0,1], startPos: 0, length: 2 -> return: [0,1]
    Tok_to_be_aligned: ".s." => tokenOffset: [0,1], startPos: 2, length: 3 -> return: [1]
    """
    indicesList = []
    doInsert = False
    posPointer = startPos
    for i, currPos in enumerate(tokenOffset_base):
        nextPos = tokenOffset_base[i + 1] if i + 1 < len(tokenOffset_base) else tokenOffset_base[i] + 99
        if not doInsert and posPointer >= currPos and posPointer < nextPos:
            doInsert = True
            posPointer = startPos + length - 1
        elif doInsert and posPointer < currPos:
            break  # break the loop in advance, othewise will stop when finish the loop.
        if doInsert:
            indicesList.append(i)
    return indicesList


def resolveTokenIndices_byPosition_multiToken(tokenOffset, startPosList, lengthList) -> list:
    idxList_3d = [
        resolveTokenIndices_byPosition(tokenOffset, startPos, length)
        for startPos, length in zip(startPosList, lengthList)
    ]
    idxList_flatten = [idx for idxList in idxList_3d for idx in idxList]
    return idxList_flatten


def trimIndices(_indices, keepNum):
    interval = []
    for _id, current in enumerate(_indices):
        if _id == len(_indices) - 1:
            break
        nextid = _id + 1
        _next = _indices[nextid]
        interval.append(_next - current)
    interval_withIdx = list(enumerate(interval))
    trimed_list = sorted(interval_withIdx, key=itemgetter(1))[0: keepNum - 1]
    idx_remained = set()
    for i in trimed_list:
        idx_remained.add(i[0])
        idx_remained.add(i[0] + 1)
    return [_indices[i] for i in idx_remained]


def replPunc(matchObj):
    if matchObj.string == matchObj.group(0):
        return matchObj.string
    return ""


def findSubString(sourceText, subStr, subStr_tokens, begin):
    sourceText = sourceText.lower()
    startPos = sourceText.find(subStr.lower(), begin)
    if startPos != -1:
        return startPos, len(subStr)
    # Sometimes metamap will rewrite the text, making the subStr differ to the source text.
    # In this case, we use token.
    if subStr_tokens:
        subStr_tokens = [i.lower() for i in subStr_tokens]
        startPos = sourceText.find(subStr_tokens[0], begin)
        assert startPos != -1
        nextStartPos = startPos + len(subStr_tokens[0])
        for token in subStr_tokens[1:]:
            nextStartPos = sourceText.find(token, nextStartPos)
            nextStartPos += len(token)
        assert nextStartPos - startPos > 0
        return startPos, nextStartPos - startPos
    return begin, 0
