from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class Model4TritonServer(nn.Module):
    def __init__(self, pretrained_model:Optional[str]=None, gpu_mode: bool=False, model=None, tokenizer=None, model_eval=True, torch_dtype='auto'):
        if pretrained_model is None and (model is None or tokenizer is None):
            raise Exception("pretrained_model and (model, tokenizer) must not be both None")
        super(Model4TritonServer, self).__init__()
        if model is None and pretrained_model is not None:
            self.model = AutoModel.from_pretrained(pretrained_model, torchscript=True, torch_dtype=torch_dtype)
        elif model is not None:
            self.model = model

        if tokenizer is None and pretrained_model is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        elif tokenizer is not None:
            self.tokenizer = tokenizer


        self.device = torch.device("cuda" if gpu_mode and torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        if model_eval:
            self.model.eval() # fix the Batch Normalization and Dropout

    def save_onnx(self, output_model:str="model.onnx", max_length:int=510, output_names:List[str]=['embedding'],
                  opset_version=12, do_constant_folding=True):
        # reference: https://developer.aliyun.com/article/1258483
        query = max_length * "字"
        encoded_input = self.tokenizer(query, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        inputs = [encoded_input['input_ids'], encoded_input['attention_mask']]

        input_names = ['input_ids', "attention_mask"]
        # {0: "batch_size", 1: "length"} or [0, 1] 都可以。前者表意性好
        dynamic_axes = {'input_ids': {0: "batch_size", 1: "length"}, 'attention_mask': {0: "batch_size", 1: "length"},
                        'embedding': {0: "batch_size"}}
        if 'token_type_ids' in encoded_input:
            inputs.append(encoded_input['token_type_ids'])
            input_names.append('token_type_ids')
            dynamic_axes['token_type_ids'] = [0, 1]

        torch.onnx.export(self.model, tuple(inputs),
                          output_model,
                          do_constant_folding=do_constant_folding, # 常量折叠将使用一些算好的常量来优化一些输入全为常量的节点
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes,
                          opset_version=opset_version)

    def save_torchscript(self, output_model: str = "model.pt", max_length: int = 510, strict=False):
        query = max_length*"草"
        encoded_input = self.tokenizer(query, max_length=max_length, truncation=True, padding=True, return_tensors='pt')
        inputs = [encoded_input['input_ids'], encoded_input['attention_mask']]
        # roberta has no token_type_ids because roberta has no NSP
        if 'token_type_ids' in encoded_input:
            inputs.append(encoded_input['token_type_ids'])
        #第一个参数实际调用的是self.model.forward函数
        traced_model = torch.jit.trace(self.model, tuple(inputs), strict=strict)
        # ATTENTION：执行时是device是cpu还是gpu，这个和运行环境耦合
        torch.jit.save(traced_model, output_model)
