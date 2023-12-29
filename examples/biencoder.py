from triton_bert.triton_bert import TritonBert
import numpy as np

class Biencoder(TritonBert):
    '''
    this is sentence sbert whose vector will be stored in milvus
    '''
    def __init__(self, triton_host:str, model: str, vocab:str, **kwargs):
        super().__init__(**kwargs)
        self.normalize_vector = False

    def proprocess(self, triton_output):
        if self.normalize_vector:
            #if you use IP, you must normalize the vector which is the same as cosine
            return [(x /np.linalg.norm(x)).tolist() for x in triton_output[0]]
        #milvus accept list type vector
        return triton_output[0].tolist()

if __name__ == "__main__":
    model = Biencoder(triton_host="30.171.160.44", model="sbert_onnx", vocab="/Users/yanyongwen712/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2")
    #model = Biencoder(triton_host="30.171.160.44", model="simcse_onnx", vocab="/Users/yanyongwen712/.cache/torch/sentence_transformers/simcse-chinese-roberta-wwm-ext")
    #model = Biencoder(triton_host="30.171.160.44", model="bge_onnx", vocab="/Users/yanyongwen712/.cache/torch/sentence_transformers/BAAI_bge-small-zh-v1.5")
    print(model("基金的收益率是多少？"))