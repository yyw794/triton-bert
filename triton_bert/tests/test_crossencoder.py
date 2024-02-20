from triton_bert.triton_bert import TritonBert
import numpy as np

class CrossEncoder(TritonBert):
    '''
    rank with text similarity
    '''
    def __init__(self, triton_host:str, model: str, vocab:str, **kwargs):
        super().__init__(triton_host=triton_host, model=model, vocab=vocab, **kwargs)

    def proprocess(self, triton_output):
        return np.squeeze(triton_output[0], axis=1).tolist()

    def __call__(self, query, text_pairs):
        #change user rank input into our input pairs
        texts = len(text_pairs)*[query]
        return self.predict(texts, text_pairs)

if __name__ == "__main__":
    model = CrossEncoder(triton_host="xx", model="xx", vocab="xx")
    model("小明借了小红500元", ['小红借了小明500元', '小明还了小红500元', '小明借了小红400元'])