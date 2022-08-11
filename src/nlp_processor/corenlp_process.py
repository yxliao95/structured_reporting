import requests, time, random, logging
from multiprocessing import Process

logger = logging.getLogger()


class CorenlpProcess(Process):
    def __init__(self, request_url, progressId, input_pipe, input_text_list, input_sid_list):
        super().__init__()
        self.request_url = request_url
        self.id = random.randint(100, 999)
        self.processId = progressId
        self.input_pipe = input_pipe
        self.input_text_list = input_text_list
        self.input_sid_list = input_sid_list

    def run(self):
        logger.trace(f"CoreNLP Process [{self.processId}-{self.id}] started.")
        time0 = time.time()
        dataDict = {}  # A dict where key=sid, value=coreNLP output in json string
        for text, sid in zip(self.input_text_list, self.input_sid_list):
            corenlp_out_jsonStr = requests.post(self.request_url, data=text.encode()).text
            dataDict[sid] = corenlp_out_jsonStr
        time1 = time.time()
        data = {"processId": self.id, "data": dataDict}
        logger.trace(f"CoreNLP Process [{self.processId}-{self.id}] finished in {time1-time0}s")
        self.input_pipe.send(data)
