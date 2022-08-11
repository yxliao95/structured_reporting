import time, random
import spacy, logging
from multiprocessing import Process

SPACY_NLP = spacy.load("en_core_web_md", disable=["ner"])
logger = logging.getLogger()


class SpacyProcess(Process):
    def __init__(self, progressId, input_pipe, input_text_list, input_sid_list):
        super().__init__()
        self.id = random.randint(100, 999)
        self.processId = progressId
        self.input_pipe = input_pipe
        self.input_text_list = input_text_list
        self.input_sid_list = input_sid_list

    def run(self):
        logger.debug(f"Spacy Process [{self.processId}-{self.id}] started.")
        time0 = time.time()
        text_tuples = [(text, {"sid": sid}) for text, sid in zip(self.input_text_list, self.input_sid_list)]
        spacy_output_list = []
        for doc, context in SPACY_NLP.pipe(text_tuples, as_tuples=True):
            sid = context["sid"]
            spacy_output_list.append({"sid": sid, "doc": doc})
        data = {"processId": self.id, "data": spacy_output_list}
        time1 = time.time()
        logger.debug(f"Spacy Process [{self.processId}-{self.id}] finished in {time1-time0}s")
        self.input_pipe.send(data)
