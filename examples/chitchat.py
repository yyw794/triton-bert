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