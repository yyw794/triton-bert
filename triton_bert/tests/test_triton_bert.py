from triton_bert.triton_bert import TritonBert

if __name__ == "__main__":
    model = TritonBert(triton_host="xxx", model="sbert_onnx", vocab="/Users/xxxxx/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2")

    #use for batch inference
    vectors = model(["基金的收益率是多少？", "我有个朋友的股票天天涨停"])
    assert len(vectors) == 2
    vectors = model.encode(["基金的收益率是多少？", "我有个朋友的股票天天涨停"])
    assert len(vectors) == 2

    # use for single inference
    vector = model.encode("基金的收益率是多少？")
    assert vectors[0] == vector
