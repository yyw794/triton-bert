from triton_bert.triton_bert import TritonBert
import numpy as np

class BgeM3Onnx(TritonBert):
    def proprocess(self, triton_output):
        return [(x /np.linalg.norm(x)) for x in triton_output[0]]

    def triton_infer(self, encoded_input):
        if encoded_input and 'input_ids' in encoded_input:
            encoded_input['token_type_ids'] = np.empty_like(encoded_input['input_ids'])
        
        return super().triton_infer(encoded_input)
    
if __name__ == "__main__":

    model = BgeM3Onnx(triton_host="30.171.160.44", model="bge_onnx", 
            vocab="hotchpotch/vespa-onnx-BAAI-bge-m3-only-dense")

    vectors = model(["基金的收益率是多少？", "我有个朋友的股票天天涨停", "今天的股票的收益率超高"])

    print(vectors[0] @ vectors[1].T)
    print(vectors[0] @ vectors[2].T)
    print(vectors[1] @ vectors[2].T)
