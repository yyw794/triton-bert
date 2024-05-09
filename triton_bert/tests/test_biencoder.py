from typing import List

from triton_bert.triton_bert import TritonBert
import numpy as np

class Biencoder(TritonBert):
    '''
    this is sentence sbert whose vector will be stored in milvus
    '''
    def __init__(self, triton_host:str, model: str, vocab:str, **kwargs):
        super().__init__(triton_host=triton_host, model=model, vocab=vocab, **kwargs)
        self.normalize_vector = True

    def proprocess(self, triton_output):
        if self.normalize_vector:
            #if you use IP, you must normalize the vector which is the same as cosine
            return [(x /np.linalg.norm(x)).tolist() for x in triton_output[0]]
        #milvus accept list type vector
        return triton_output[0].tolist()

if __name__ == "__main__":
    model = Biencoder(triton_host="xxx", model="sbert_onnx",
                       vocab="/Users/xxxx/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2")

    # batch inferences
    vectors = model(["基金的收益率是多少？", "我有个朋友的股票天天涨停"])
    # or
    # vectors = model.encodes(["基金的收益率是多少？", "我有个朋友的股票天天涨停"])
    assert len(vectors) == 2

    # single inference
    vector = model.encode("基金的收益率是多少？")
    assert vectors[0] == vector