
import re
import pandas as pd


def remove_tag_from_list(text_list):
    """ Remove ``@[tag]`` from the text of the list, and return a new list """
    new_list = []
    for text in text_list:
        regexPattern = r"@\[.+\]"
        res = re.sub(regexPattern, "", text)
        new_list.append(res.strip())
    return new_list


def load_mimic_cxr(file_path, section_name_cfg):
    """Load data from ``input_path``.
    ``section_name_cfg`` is the config load from ``config/name_style/mimic_cxr_section.yaml``
    """
    df = pd.read_json(file_path, orient="records", lines=True)
    df = df.sort_values(by=[section_name_cfg.PID, section_name_cfg.SID])
    pid_list = df.loc[:, section_name_cfg.PID].to_list()
    sid_list = df.loc[:, section_name_cfg.SID].to_list()
    findings_list = remove_tag_from_list(df.loc[:, section_name_cfg.FINDINGS].to_list())
    impression_list = remove_tag_from_list(df.loc[:, section_name_cfg.IMPRESSION].to_list())
    pfi_list = remove_tag_from_list(df.loc[:, section_name_cfg.PFI].to_list())
    fai_list = remove_tag_from_list(df.loc[:, section_name_cfg.FAI].to_list())
    return len(sid_list), pid_list, sid_list, findings_list, impression_list, pfi_list, fai_list


def load_mimic_cxr_bySection(input_path, target_section_cfg, section_name_cfg):
    """
    Return:
        data_size: int
        pid_list: list[str]
        sid_list: list[str]
        section_list: list[tuple]; the tuple is like (section_name:str, section_text_list:list)
    """
    data_size, pid_list, sid_list, findings_list, impression_list, pfi_list, fai_list = load_mimic_cxr(input_path, section_name_cfg)

    section_list = []
    if target_section_cfg.findings:
        section_list.append((section_name_cfg.FINDINGS, findings_list))
    if target_section_cfg.impression:
        section_list.append((section_name_cfg.IMPRESSION, impression_list))
    if target_section_cfg.provisional_findings_impression:
        section_list.append((section_name_cfg.PFI, pfi_list))
    if target_section_cfg.findings_and_impression:
        section_list.append((section_name_cfg.FAI, fai_list))

    return data_size, pid_list, sid_list, section_list


def load_i2b2(input_path, json_name_cfg):
    """ Each i2b2 document are expected to process as a whole rather than split into multiple sections like mimic-cxr.
    However, in order to adapt the input format of the subsequent scripts, we need to create the text_list with section name.
    Hence, we use `all` as a pseudo-section.
    """
    df = pd.read_json(input_path, orient="records", lines=True)
    # e.g. Sort `clinical-10` by its number `10`
    df = df.sort_values(by=json_name_cfg.id, key=lambda col: col.apply(lambda val: int(val.split("-")[-1])), ascending=True)
    id_list = df.loc[:, json_name_cfg.id].to_list()
    text_list = df.loc[:, json_name_cfg.text].to_list()
    section_list = [("all", text_list)]
    return id_list, section_list
