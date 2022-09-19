import json
from hydra import compose
# pylint: disable=import-error
from data_preprocessing.utils_and_rules.utils import removePunctuation


class PrerequisiteResources:
    _heading_affiliation_map = {}
    _procedure_info_list = []
    _techniqueContent_lower = []
    _ignoreHeading_list = []
    _manually_preprcessed_record_list = []

    @staticmethod
    def _load_resources():
        config = compose(config_name="data_preprocessing", overrides=["data_preprocessing@_global_=mimic_cxr"])
        with open(config.rules.heading_affiliation_map, "r", encoding="UTF-8") as f:
            headingMap_list = f.readlines()
        for rawLine in headingMap_list:
            ele = rawLine.strip().split(":")
            if ele[1] == "":
                continue
            PrerequisiteResources._heading_affiliation_map[ele[0]] = ele[1]

        with open(config.rules.technique_procedureinfo, "r", encoding="UTF-8") as f:
            for line in f:
                PrerequisiteResources._procedure_info_list.append(line.strip())
                PrerequisiteResources._techniqueContent_lower.append(removePunctuation(line).strip().lower())

        with open(config.rules.ignore_heading, "r", encoding="UTF-8") as f:
            for line in f:
                PrerequisiteResources._ignoreHeading_list.append(line.strip())

        with open(config.rules.manually_processed_records, "r", encoding="UTF-8") as f:
            j = json.load(f)
            PrerequisiteResources._manually_preprcessed_record_list = j["RECORDS"]

    @staticmethod
    def get_heading_affiliation_map():
        if not PrerequisiteResources._heading_affiliation_map:
            PrerequisiteResources._load_resources()
        return PrerequisiteResources._heading_affiliation_map

    @staticmethod
    def get_procedure_info_list():
        if not PrerequisiteResources._procedure_info_list:
            PrerequisiteResources._load_resources()
        return PrerequisiteResources._procedure_info_list

    @staticmethod
    def get_techniqueContent_lower():
        if not PrerequisiteResources._techniqueContent_lower:
            PrerequisiteResources._load_resources()
        return PrerequisiteResources._techniqueContent_lower

    @staticmethod
    def get_ignoreHeading_list():
        if not PrerequisiteResources._ignoreHeading_list:
            PrerequisiteResources._load_resources()
        return PrerequisiteResources._ignoreHeading_list

    @staticmethod
    def get_manually_preprcessed_record_list():
        if not PrerequisiteResources._manually_preprcessed_record_list:
            PrerequisiteResources._load_resources()
        return PrerequisiteResources._manually_preprcessed_record_list


# Headings Affiliation Mapping

def mapHeading(heading: str):
    if heading == "UNKNOWN":
        return "UNKNOWN"
    try:
        heading_affiliation_map = PrerequisiteResources.get_heading_affiliation_map()
        return heading_affiliation_map[heading]
    except KeyError:
        return "to_be_defined"

###

# Procedure_Info Mapping


def isProcedureInfo_Findings_Heading(heading: str):
    procedure_info_list = PrerequisiteResources.get_procedure_info_list()
    return True if heading in procedure_info_list else False


def eqToPredefinedTechniqueContent(content: str):
    techniqueContent_lower = PrerequisiteResources.get_techniqueContent_lower()
    return True if removePunctuation(content).strip().lower() in techniqueContent_lower else False

###

# Ignore Heading List


def ignoreHeading(heading: str):
    ignoreHeading_list = PrerequisiteResources.get_ignoreHeading_list()
    if heading in ignoreHeading_list:
        return True
    else:
        return False

###

# Manually preprocessed records


def check_and_get_manual_record(data_item):
    """ If the corresponding manual record exists, then replace the ``data_item`` and retun the maunal record,
    otherwise return the original ``data_item``
    """
    manual_records = PrerequisiteResources.get_manually_preprcessed_record_list()
    for manual_record in manual_records:
        if data_item["SID"] == manual_record["SID"]:
            return manual_record
    return data_item

###


if __name__ == "__main__":
    # Move to the head of this file
    # import sys,os
    # sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from hydra import compose, initialize
    from omegaconf import OmegaConf

    with initialize(version_base=None, config_path="../../config"):
        print(PrerequisiteResources.get_heading_affiliation_map())
