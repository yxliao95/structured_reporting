class MetaReport:
    """ Class for raw data """

    def __init__(self):
        self.headingList = []
        self.contentList = []
        self.rawContentList = []
        self.contentAffiliationList = []
        self.finalReportIdx = 0

    def append(self, heading, content, rawContent, contentAffiliation):
        self.headingList.append(heading)
        self.contentList.append(content)
        self.rawContentList.append(rawContent)
        self.contentAffiliationList.append(contentAffiliation)

    def initFinalReportLocation(self):
        self.finalReportIdx = self.size()

    def getFinalReportLocation(self):
        return self.finalReportIdx

    def getUniqueHeadingIndices(self):
        affiliationUnique = set(self.contentAffiliationList)
        uniqueIndices = [self.contentAffiliationList.index(aff) for aff in affiliationUnique]
        uniqueIndices.sort()
        return uniqueIndices

    def getUniqueHeadingIndices_finalReport(self):
        affiliationUnique = set(self.contentAffiliationList[self.finalReportIdx :])
        uniqueIndices = [self.contentAffiliationList.index(aff) for aff in affiliationUnique]
        uniqueIndices.sort()
        return uniqueIndices

    def getHeading(self, idx):
        return self.headingList[idx]

    def getContent(self, idx):
        return self.contentList[idx]

    def getRawContent(self, idx):
        return self.rawContentList[idx]

    def getContentAffiliation(self, idx):
        return self.contentAffiliationList[idx]

    def size(self):
        return len(self.headingList)

    def __repr__(self):
        out = ""
        for i in range(self.size()):
            out += f"{self.headingList[i]}[CAFF:{self.contentAffiliationList[i]}]:{self.contentList[i]}[RAW:{self.rawContentList[i]}]"
            if i != self.size() - 1:
                out += "\n"
        return out


class StructuredReport:
    """ Class to store processed data

    Args: 
        heading: original heading
        headingAffiliation: WET_READ, PROCEDURE_INFO, CLINICAL_INFO, COMPARISON, FINDINGS, IMPERSSION
        content: 
        isComplete: Whether all the relevant content extracted.
        potentialMix: Whether the content contain irrelavent section's content. Require manual check.
    """

    def __init__(self):
        self.headingList = []
        self.headingAffiliationList = []
        self.contentList = []
        self.iterPointer = 0

    def append(self, heading, headingAffiliation, content):
        self.headingList.append(heading)
        self.headingAffiliationList.append(headingAffiliation)
        self.contentList.append(content)

    def getByIndex(self, idx) -> tuple:
        return (self.headingList[idx], self.headingAffiliationList[idx], self.contentList[idx])

    def delByIndex(self, idx):
        del self.headingList[idx]
        del self.headingAffiliationList[idx]
        del self.contentList[idx]

    def __iter__(self):
        self.iterPointer = 0
        return self

    def __next__(self) -> tuple:
        i = self.iterPointer
        self.iterPointer += 1
        try:
            return self.getByIndex(i)
        except IndexError:
            raise StopIteration

    def size(self):
        return len(self.headingList)

    def __repr__(self):
        out = ""
        for i in range(self.size()):
            out += f"{self.headingList[i]}[HAFF:{self.headingAffiliationList[i]}]:{self.contentList[i]}"
            if i != self.size() - 1:
                out += "\n"
        return out
