###
# This pre-processing script aims to identify and extract the sections (findings, impression, etc.) content from raw mimic-cxr reports.
###
import logging
import os
import sys
import time
import json
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf
# pylint: disable=import-error
from common_utils.file_checker import FileChecker
from common_utils.common_utils import check_and_create_dirs
from data_preprocessing.utils_and_rules.data_holder_class import MetaReport, StructuredReport
from data_preprocessing.utils_and_rules.rule_prerequisites import check_and_get_manual_record, eqToPredefinedTechniqueContent
from data_preprocessing.utils_and_rules.rules import finalAddendum_identifyRule, finalReport_identifyRule
from data_preprocessing.utils_and_rules.rules import finalReportSection_concatenateRule, noTitleSection_identifyRule, nonFinalReportSection_concatenateRule
from data_preprocessing.utils_and_rules.utils import isComparison, isTechnique


logger = logging.getLogger()
FILE_CHECKER = FileChecker()

module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")


def read_raw_data(config):
    """Read from the raw files"""
    dataList = []
    patientCounter = 0
    studyCounter = 0
    logger.info("Loading the raw files.")
    start0 = time.time()
    with tqdm(total=227835) as pbar:
        for dirPath, dirNames, fileNames in os.walk(os.path.join(config.data_dir, "files")):
            patientCounter += 1
            _, pid = os.path.split(dirPath)
            if fileNames:
                for fileName in fileNames:
                    if not FILE_CHECKER.ignore(fileName):
                        studyCounter += 1
                        sid = fileName.removesuffix(".txt")
                        filePath = os.path.join(dirPath, fileName)
                        with open(filePath, "r", encoding="UTF-8") as f:
                            contentList = f.readlines()
                            dataList.append({"pid": pid, "sid": sid, "contentList": contentList})
                        pbar.update(1)
    logger.info("Time cost: %.2fs. Loaded %s files: %s (patients) | %s (studies). ", time.time()-start0, studyCounter, patientCounter, studyCounter)
    return dataList


def resolve(dataList):
    """The raw data will be converted into MetaReport and StructuredReport and appended to dataList."""
    logger.info("Resolving the reports.")
    start0 = time.time()
    for reportItem in tqdm(dataList):
        metaReport = MetaReport()
        reportItem["metaReport"] = metaReport
        structuredReport = StructuredReport()
        reportItem["structuredReport"] = structuredReport
        handler = finalReport_identifyRule
        enterFinalReport = False
        for contentRow in reportItem["contentList"]:
            contentRow = contentRow.strip()
            if not enterFinalReport and contentRow == "FINAL ADDENDUM":
                handler = finalAddendum_identifyRule
            elif not enterFinalReport and contentRow.startswith(("WET ", "PROVISIONAL ", "CLINICAL INFORMATION & ")):
                handler = noTitleSection_identifyRule
            elif contentRow == "FINAL REPORT":
                # Once it turn on, it will remain on untill next report begin.
                # Make sure the [contentRow.startswith("WET READ")] in the final report part will come to this fork.
                enterFinalReport = True
                handler = finalReport_identifyRule
                metaReport.initFinalReportLocation()
                nonFinalReportSection_concatenateRule(metaReport, structuredReport)
            handler(contentRow, metaReport, structuredReport)
        finalReportSection_concatenateRule(metaReport, structuredReport)

    logger.info("Time cost: %.2fs", time.time()-start0)
    return dataList


def convert(config, dataList):
    """1. Perform the final step for the identification of section content.
    2. If ``config.use_artefact`` is True, then the corresponding records will be replaced by the manually preprocessed records.
    """
    logger.info("Converting the output data.")
    start0 = time.time()
    output_list = []
    for dataItem in tqdm(dataList):
        structuredReport = dataItem["structuredReport"]
        assert dataItem["sid"]
        data_item = {
            "PID": dataItem["pid"],
            "SID": dataItem["sid"],
            "FINDINGS": "",
            "IMPRESSION": "",
            "PFI": "",
            "FAI": "",
            "CLINICAL_INFO": "",
            "PROCEDURE_INFO": "",
            "COMPARISON": "",
            "ADDENDUM": "",
            "WET_READ": "",
            "UNKNOWN": "",
        }
        for (h, hAff, c) in structuredReport:
            if hAff in ["FINDINGS", "IMPRESSION"]:
                data_item[hAff] += c + "\n"
            else:
                data_item[hAff] += f"@[{h}]\n{c}\n"

        # Extract findings and impression from no heading section (that was misallocated to other heading)
        if (
            data_item["FINDINGS"] == ""
            and data_item["IMPRESSION"] == ""
            and data_item["PFI"] == ""
            and data_item["FAI"] == ""
        ):
            try:
                (h, hAff, finalContent) = structuredReport.getByIndex(-1)
                finalContentList = finalContent.split("\n")
                if len(finalContentList) > 1:
                    data_item[hAff] = f"@[{h}]\n{finalContentList[0]}\n"
                    data_item["FAI"] += "@[from_no_heading_section]\n"
                    for idx, singleLine in enumerate(finalContentList[1:]):
                        singleLine = singleLine.strip()
                        if isTechnique(singleLine) or eqToPredefinedTechniqueContent(singleLine):
                            data_item["PROCEDURE_INFO"] += f"\n@[no_heading_specified]\n{singleLine}"
                            if len(singleLine) > 200:
                                data_item["FAI"] += f"{singleLine}\n"
                        elif isComparison(singleLine):
                            data_item["COMPARISON"] += f"\n@[no_heading_specified]\n{singleLine}"
                            if len(singleLine) > 150:
                                data_item["FAI"] += f"{singleLine}\n"
                        else:
                            data_item["FAI"] += f"{singleLine}\n"
                    data_item["FAI"] = data_item["FAI"].strip()
                elif hAff == "UNKNOWN":
                    data_item["FAI"] += f"@[from_no_heading_section]\n{finalContent}"
            except IndexError:
                pass

        if config.in_process.use_artefact:
            data_item = check_and_get_manual_record(data_item)

        output_list.append(data_item)

    logger.info("Time cost: %.1fs", time.time()-start0)
    return output_list


def output_json(config, output_list):
    logger.info("Writing the output in JSON format.")
    json_output_cfg = config.json
    check_and_create_dirs(json_output_cfg.output_dir)
    ourput_file_path = os.path.join(json_output_cfg.output_dir, json_output_cfg.file_name)

    start0 = time.time()
    with open(ourput_file_path, "w", encoding="UTF-8") as f:
        for data_item in tqdm(output_list):
            column_name = config.name_style.mimic_cxr.section_name
            formatted_reocrd = {
                column_name.PID: data_item["PID"],
                column_name.SID: data_item["SID"],
                column_name.FINDINGS: data_item["FINDINGS"],
                column_name.IMPRESSION: data_item["IMPRESSION"],
                column_name.PFI: data_item["PFI"],
                column_name.FAI: data_item["FAI"],
                column_name.CLINICAL_INFO: data_item["CLINICAL_INFO"],
                column_name.PROCEDURE_INFO: data_item["PROCEDURE_INFO"],
                column_name.COMPARISON: data_item["COMPARISON"],
                column_name.ADDENDUM: data_item["ADDENDUM"],
                column_name.WET_READ: data_item["WET_READ"],
                column_name.UNKNOWN: data_item["UNKNOWN"],
            }
            f.write(json.dumps(formatted_reocrd))
            f.write("\n")
    logger.info("Time cost: %.2fs. Output: %s.", time.time()-start0, ourput_file_path)


def output_mysql(config, output_list):
    logger.info("Writing the output to mysql.")
    import pymysql

    start0 = time.time()
    mysql_cfg = config.mysql
    column_name = config.name_style.mimic_cxr.section_name

    db = pymysql.connect(
        host=str(mysql_cfg.host),
        port=int(mysql_cfg.port),
        user=str(mysql_cfg.user),
        password=str(mysql_cfg.password),
        db=str(mysql_cfg.db),
    )
    cursor = db.cursor()
    drop = f"DROP TABLE IF EXISTS `{mysql_cfg.table_name}` ;"
    cursor.execute(drop)
    cursor.connection.commit()
    flush = f"FLUSH TABLES `{mysql_cfg.table_name}` ;"
    cursor.execute(flush)
    cursor.connection.commit()
    create_table = f"""CREATE TABLE `{mysql_cfg.table_name}` (
    `{column_name.PID}` varchar(10) NOT NULL,
    `{column_name.SID}` varchar(10) NOT NULL,
    `{column_name.FINDINGS}` text,
    `{column_name.IMPRESSION}` text,
    `{column_name.PFI}` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
    `{column_name.FAI}` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
    `{column_name.CLINICAL_INFO}` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
    `{column_name.PROCEDURE_INFO}` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
    `{column_name.COMPARISON}` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
    `{column_name.ADDENDUM}` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
    `{column_name.WET_READ}` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
    `{column_name.UNKNOWN}` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
    PRIMARY KEY (`pid`,`sid`) USING BTREE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
    """
    cursor.execute(create_table)
    cursor.connection.commit()

    logger.info("Created TABLE `%s`.`%s`, writing data:", mysql_cfg.db, mysql_cfg.table_name)
    for data_item in tqdm(output_list):
        sql = f"""INSERT INTO `{mysql_cfg.db}`.`{mysql_cfg.table_name}`(
        `{column_name.PID}`,`{column_name.SID}`,
        `{column_name.FINDINGS}`,`{column_name.IMPRESSION}`,
        `{column_name.PFI}`,`{column_name.FAI}`,
        `{column_name.CLINICAL_INFO}`,`{column_name.PROCEDURE_INFO}`,
        `{column_name.COMPARISON}`,`{column_name.ADDENDUM}`,
        `{column_name.WET_READ}`,`{column_name.UNKNOWN}`)
        VALUES (
        '{data_item["PID"]}','{data_item["SID"]}',
        '{db.escape_string(data_item["FINDINGS"])}','{db.escape_string(data_item["IMPRESSION"])}',
        '{db.escape_string(data_item["PFI"])}','{db.escape_string(data_item["FAI"])}',
        '{db.escape_string(data_item["CLINICAL_INFO"])}','{db.escape_string(data_item["PROCEDURE_INFO"])}',
        '{db.escape_string(data_item["COMPARISON"])}','{db.escape_string(data_item["ADDENDUM"])}',
        '{db.escape_string(data_item["WET_READ"])}','{db.escape_string(data_item["UNKNOWN"])}');
        """
        cursor.execute(sql)

    cursor.connection.commit()
    cursor.close()
    db.close()
    logger.info("Time cost: %.2fs. Output: mysql: `%s`.`%s`.", time.time()-start0, mysql_cfg.db, mysql_cfg.table_name)


@hydra.main(version_base=None, config_path=config_path, config_name="data_preprocessing")
def main(config):
    """ The pre-processing aims to identify and split the sections (findings, impression, etc.) in raw mimic-cxr reports. """
    print(OmegaConf.to_yaml(config))

    dataList = read_raw_data(config)
    dataList = resolve(dataList)
    output_list = convert(config, dataList)
    if config.output.json:
        output_json(config, output_list)
    if config.output.mysql:
        output_mysql(config, output_list)


if __name__ == "__main__":
    sys.argv.append("data_preprocessing@_global_=mimic_cxr")
    main()  # pylint: disable=no-value-for-parameter
