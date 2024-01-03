import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel

class Model4TritonServer(nn.Module):
    def __init__(self, pretrained_model:str, gpu_mode: bool=False):
        super(Model4TritonServer, self).__init__()
        self.model = BertModel.from_pretrained(pretrained_model, torchscript=True)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.device = torch.device("cuda" if gpu_mode and torch.cuda.is_available() else "cpu")
        self.gpu_mode = gpu_mode and torch.cuda.is_available()
        self.model = self.model.to(self.device)
        self.model.eval() # fix the Batch Normalization and Dropout

    def save_onnx(self, output_model:str="model.onnx", max_length:int=510):
        # reference: https://developer.aliyun.com/article/1258483
        query = max_length * "字"
        encoded_input = self.tokenizer(query, padding=True, truncation=True, max_length=max_length, return_tensors='pt')\
            .to(self.device)

        torch.onnx.export(self.model, (encoded_input['input_ids'],encoded_input['attention_mask'],encoded_input['token_type_ids']),
                          output_model,
                          do_constant_folding=True, # 常量折叠将使用一些算好的常量来优化一些输入全为常量的节点
                          input_names=['input_ids', "attention_mask", 'token_type_ids'],
                          output_names=['embedding'],
                          # {0: "batch_size", 1: "length"} or [0, 1] 都可以。前者表意性好
                          dynamic_axes={'input_ids': {0: "batch_size", 1: "length"}, 'attention_mask': [0, 1], 'token_type_ids': [0, 1],
                                        'embedding': {0: "batch_size"}},
                          opset_version=12)

    def save_torchscript(self, output_model: str = "model.pt", max_length: int = 510):
        """
        When tracing, we use an example input to record the actions taken and capture the the model architecture.
        This works best when your model doesn't have control flow.
        If you do have control flow, you will need to use the scripting approach
        """
        query = max_length*"草"
        encoded_input = self.tokenizer(query, max_length=max_length, truncation=True, padding=True, return_tensors='pt').to(self.device)
        traced_model = torch.jit.trace(self.model, [encoded_input['input_ids'], encoded_input['attention_mask'], encoded_input['token_type_ids']])
        # ATTENTION：执行时是device是cpu还是gpu，这个和运行环境耦合
        torch.jit.save(traced_model, output_model)


class ModelAveragePool(Model4TritonServer):
    def __init__(self, pretrained_model:str, gpu_mode: bool=False):
        super(ModelAveragePool, self).__init__(pretrained_model, gpu_mode)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            model_output = self.model.eval(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

class ModelCls(Model4TritonServer):
    def __init__(self, pretrained_model: str, gpu_mode: bool=False):
        super(ModelCls, self).__init__(pretrained_model, gpu_mode)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            model_output = self.model.eval(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        return model_output.last_hidden_state[:, 0]


if __name__ == "__main__":
    '''
    model = SentenceBert("/Users/yanyongwen712/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2")
    #model.save_torchscript("sbert_model.pt")
    model.save_onnx("sbert_model.onnx")
    '''

    model = ModelCls("/Users/yanyongwen712/.cache/torch/sentence_transformers/simcse-chinese-roberta-wwm-ext")
    model.save_torchscript("model/cpu/simcse_model.pt")
    model.save_onnx("model/cpu/simcse_model.onnx")

    '''
    model = ModelAveragePool("/Users/yanyongwen712/.cache/torch/sentence_transformers/BAAI_bge-small-zh-v1.5")
    model.save_onnx("model/cpu/bge_model.onnx")
    model.save_torchscript("model/cpu/bge_model.pt")
    '''
