import re
# pylint: disable=import-error
from data_preprocessing.utils_and_rules.data_holder_class import MetaReport, StructuredReport
from data_preprocessing.utils_and_rules.rule_prerequisites import PrerequisiteResources, ignoreHeading, isProcedureInfo_Findings_Heading, mapHeading
from data_preprocessing.utils_and_rules.utils import extract, formatHeading, hasLineFeed, isEmptyRow, isLineSeparator


def finalAddendum_identifyRule(contentRow: str, metaReport: MetaReport, structuredReport: StructuredReport):
    contentRow = contentRow.strip()  # Remove whitespace and \n
    heading = ""
    content = ""
    contentAffiliation = "FINAL ADDENDUM"
    if "FINAL ADDENDUM" == contentRow:
        heading = "FINAL ADDENDUM"
    metaReport.append(heading=heading, content=content, rawContent=contentRow, contentAffiliation=contentAffiliation)


def noTitleSection_identifyRule(contentRow: str, metaReport: MetaReport, structuredReport: StructuredReport):
    heading = ""
    content = ""
    contentAffiliation = metaReport.contentAffiliationList[-1] if metaReport.size() > 0 else ""
    if contentRow.startswith("WET "):
        m = re.match(r"(WET (READ|___)(:| VERSION #(\d|___)))(.*)", contentRow)
        if m:
            heading, content = m.group(1, 5)
        contentAffiliation = contentAffiliation if heading == "" else formatHeading(heading)
    elif contentRow.startswith("PROVISIONAL "):
        heading = "PROVISIONAL FINDINGS IMPRESSION (PFI)"
        content = contentRow[38:]
        contentAffiliation = heading
    elif contentRow.startswith("CLINICAL INFORMATION & "):
        heading = "CLINICAL INFORMATION & QUESTIONS TO BE ANSWERED"
        content = contentRow[47:]
        contentAffiliation = heading
    assert contentAffiliation != ""
    metaReport.append(heading=heading, content=content, rawContent=contentRow, contentAffiliation=contentAffiliation)


def finalReport_identifyRule(contentRow: str, metaReport: MetaReport, structuredReport: StructuredReport):
    contentRow = contentRow.strip()  # Remove whitespace and \n
    if "FINAL REPORT" == contentRow:
        return
    heading, content = extract(contentRow)
    # the affiliation will inherit the heading from previous
    contentAffiliation = (
        metaReport.contentAffiliationList[-1] if metaReport.size() > metaReport.getFinalReportLocation() else "UNKNOWN"
    )
    heading_affiliation_map = PrerequisiteResources.get_heading_affiliation_map()
    if ignoreHeading(heading):
        heading = ""
        content = ""
    elif heading != "":
        contentAffiliation = heading
    elif contentRow.isupper() and contentRow in heading_affiliation_map:
        heading = contentRow
        contentAffiliation = heading
    assert contentAffiliation != ""
    metaReport.append(heading=heading, content=content, rawContent=contentRow, contentAffiliation=contentAffiliation)


def nonFinalReportSection_concatenateRule(metaReport: MetaReport, structuredReport: StructuredReport):
    uniqueIndices = metaReport.getUniqueHeadingIndices()
    for indices_idx, currHeading_idx in enumerate(uniqueIndices):
        heading = metaReport.getContentAffiliation(currHeading_idx)
        try:
            nextHeading_idx = uniqueIndices[indices_idx + 1]
        except IndexError:
            nextHeading_idx = metaReport.size()
        content = metaReport.getContent(currHeading_idx)
        content += "." if content else ""
        for j in range(currHeading_idx + 1, nextHeading_idx):
            contentRow = metaReport.getRawContent(j)
            if isEmptyRow(contentRow):
                contentRow = "\n"
            if isLineSeparator(contentRow):
                continue
            if hasLineFeed(content) and contentRow == "\n":
                continue
            content += " " + contentRow
        content = content.strip()
        structuredReport.append(heading=heading, headingAffiliation=mapHeading(heading), content=content)


def finalReportSection_concatenateRule(metaReport: MetaReport, structuredReport: StructuredReport):
    uniqueIndices = metaReport.getUniqueHeadingIndices_finalReport()
    for indices_idx, currHeading_idx in enumerate(uniqueIndices):
        heading = metaReport.getContentAffiliation(currHeading_idx)
        try:
            nextHeading_idx = uniqueIndices[indices_idx + 1]
        except IndexError:
            nextHeading_idx = metaReport.size()
        content = (
            metaReport.getRawContent(currHeading_idx)
            if heading == "UNKNOWN"
            else metaReport.getContent(currHeading_idx)
        )
        for j in range(currHeading_idx + 1, nextHeading_idx):
            contentRow = metaReport.getRawContent(j)
            if isEmptyRow(contentRow):
                contentRow = "\n"
            if hasLineFeed(content) and contentRow == "\n":
                continue
            content += " " + contentRow
        content = content.strip()
        if isProcedureInfo_Findings_Heading(heading):
            # Add an extra section for those that used prcedure_info content as headings
            structuredReport.append(
                heading="FROM_FINDINGS_HEADING", headingAffiliation="PROCEDURE_INFO", content=heading
            )
            headingAffiliation = "FINDINGS"
        else:
            headingAffiliation = mapHeading(heading)
        structuredReport.append(heading=heading, headingAffiliation=headingAffiliation, content=content)
