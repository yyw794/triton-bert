from triton_bert.model.bge_m3 import BgeM3Onnx

if __name__ == "__main__":

    model = BgeM3Onnx(triton_host="30.171.160.44", model="bge_onnx", 
            vocab="hotchpotch/vespa-onnx-BAAI-bge-m3-only-dense")

    vectors = model(["基金的收益率是多少？", "我有个朋友的股票天天涨停", "今天的股票的收益率超高"])

    print(vectors[0] @ vectors[1].T)
    print(vectors[0] @ vectors[2].T)
    print(vectors[1] @ vectors[2].T)
