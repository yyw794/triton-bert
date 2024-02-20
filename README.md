It is easy to use bert in triton now.
Algorithm Engineer only need to focus to write proprocess function to make his model work.

# USAGES:
## Example 0:
Embedding model(Biencoder)
embedding output from model can be used directly.
so no need to override the proprocess function.

```python
from triton_bert.triton_bert import TritonBert

if __name__ == "__main__":
    model = TritonBert(triton_host="30.171.160.44", model="sbert_onnx", 
                       vocab="/Users/yanyongwen712/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2")

    # batch inferences
    vectors = model(["基金的收益率是多少？", "我有个朋友的股票天天涨停"])
    # or
    # vectors = model.encodes(["基金的收益率是多少？", "我有个朋友的股票天天涨停"])
    assert len(vectors) == 2

    # single inference
    vector = model.encode("基金的收益率是多少？")
    assert vectors[0] == vector
```

## Example 1:
Embedding model(Biencoder)
Embedding need normalized, override the proprecess
```python

from triton_bert.triton_bert import TritonBert
import numpy as np

class Biencoder(TritonBert):
    def __init__(self, triton_host:str, model: str, vocab:str, **kwargs):
        super().__init__(triton_host=triton_host, model=model, vocab=vocab, **kwargs)
        self.normalize_vector = True

    def proprocess(self, triton_output):
        if self.normalize_vector:
            #if you use IP, you must normalize the vector which is the same as cosine
            return [(x /np.linalg.norm(x)).tolist() for x in triton_output[0]]
        return triton_output[0].tolist()
```

## Example 2:
Rank model(CrossEncoder)
user query is compared the most similar top N results with each other, and find the most similar one.
```python
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
```

## Example 3
ChitChat Intention Detection.
```python
from triton_bert.triton_bert import TritonBert
import torch.nn.functional as F
import torch

class ChitchatIntentDetection(TritonBert):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_list = ["闲聊", "问答", "扯淡"]

    def proprocess(self, triton_output):
        logits = triton_output[0]
        label_ids = logits.argmax(axis=-1)
        logits = torch.tensor(logits)
        probs = F.softmax(logits, dim=1).numpy()
        ret = []
        for i, label_id in enumerate(label_ids):
            prob = probs[i][label_id]
            if label_id == 2 and prob < 0.8:
                label_id = 0
            ret.append({"category": self.label_list[label_id], "confidence": float(prob)})
        return ret


```

# run examples
## run triton server
```bash
# for example
docker run -d  --name triton-server   --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864  --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /home/yanyongwen712/triton_models:/models  nvcr.io/nvidia/tritonserver::22.08-py3 tritonserver --model-repository=/models  --model-control-mode=poll  --exit-on-error=false --log-verbose 1
# configure triton model folder
```
## prepare model for triton server
See the tests for more examples.

Example:
```python
from triton_bert.model_4_triton import Model4TritonServer

if __name__ == "__main__":
    pretrained_model = "/Users/yanyongwen712/.cache/torch/sentence_transformers/simcse-chinese-roberta-wwm-ext"
    model = Model4TritonServer(pretrained_model=pretrained_model)
    model.save_torchscript("model/simcse_model.pt")
    model.save_onnx("model/simcse_model.onnx")

```

## prepare PG with pgvector extension
...

## run example
```bash
# change triton server ip , triton model name and local transformer model folder
python retrieval_pgvector.py
```

