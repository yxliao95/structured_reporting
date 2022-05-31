import os, sys
from loguru import logger

LOG_ROOT = "./logs/"
FILE_PREFFIX = os.path.basename(__file__).removesuffix(".py")
LOG_FILE = LOG_ROOT + FILE_PREFFIX + ".log"

# Remove all handlers and reset stderr
logger.remove(handler_id=None)
logger.add(
    LOG_FILE,
    level="TRACE",
    mode="a",
    backtrace=False,
    diagnose=True,
    colorize=False,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)
logger.info("\r\n" + ">" * 29 + "\r\n" + ">>> New execution started >>>" + "\r\n" + ">" * 29)
# To filter log level: TRACE=5, DEBUG=10, INFO=20, SUCCESS=25, WARNING=30, ERROR=40, CRITICAL=50
logger.add(sys.stdout, level="INFO", filter=lambda record: record["level"].no < 40, colorize=True)
logger.add(sys.stderr, level="ERROR", backtrace=False, diagnose=True, colorize=True)

# Usage
# logger.trace("test")
# logger.info("infotest")
# logger.error("errorinfo")
# with logger.catch():
#     raise NameError("HiThere")
