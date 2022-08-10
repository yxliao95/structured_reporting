import re

def isEmptyRow(contentRow: str) -> bool:
    return True if "" == contentRow else False


def isLineSeparator(contentRow: str) -> bool:
    # "_"*78 or "_"*81 or "p"+"_"*81
    # A line with more than 70 "_"
    return True if contentRow.replace("_", "") == "" else False


def hasLineFeed(content: str) -> bool:
    return True if content.endswith("\n") else False


def extract(contentRow: str) -> tuple:
    m = re.match(r"([^a-z0-9]+?):{1}(.*)", contentRow)
    if m:
        heading = m.group(1)
        content = m.group(2)
        return (heading.strip(), content)
    else:
        return ("", "")


def formatHeading(heading: str) -> str:
    return heading.replace(":", "")


def removePunctuation(content):
    punctuation = r"~!@#$%^&*()_+`{}|\[\]\:\";\-\\\='<>?,./，。、《》？；：‘“{【】}|、！@#￥%……&*（）——+=-"
    content = re.sub(f"[{punctuation}]+", "", content)
    return content

def isTechnique(contentRow:str):
    regex1 = r"((was|were) (reviewed .*comparison|compared to))|((chest radiograph[s]? (was|were|is|are) reviewed))|(the chest (was|were) reviewed)"
    m = re.search(regex1,contentRow)
    return True if m else False
        
def isComparison(contentRow:str):
    return True if "Comparison is made with" in contentRow else False