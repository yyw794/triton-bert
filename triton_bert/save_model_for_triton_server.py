import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, convert_graph_to_onnx, BertModel, BertTokenizer
from pprint import pprint
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class SentenceBert(nn.Module):
    def __init__(self, model_path:str):
        super(SentenceBert, self).__init__()
        self.model = BertModel.from_pretrained(model_path, torchscript=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            model_output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def save_torchscript(self, output_model:str="model.pt", max_length:int=510):
        """
        When tracing, we use an example input to record the actions taken and capture the the model architecture.
        This works best when your model doesn't have control flow.
        If you do have control flow, you will need to use the scripting approach
        """
        query = max_length * "字"
        encoded_input = self.tokenizer(query, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        #pprint(encoded_input)
        traced_model = torch.jit.trace(self.model, [encoded_input['input_ids'].to(device),
                                                    encoded_input['token_type_ids'].to(device),
                                                    encoded_input['attention_mask'].to(device)])
        torch.jit.save(traced_model, output_model)

    def save_onnx(self, output_model:str="model.onnx", max_length:int=510):
        query = max_length * "字"
        encoded_input = self.tokenizer(query, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        torch.onnx.export(model, (encoded_input['input_ids'].to(device), encoded_input['attention_mask'].to(device),
                                  encoded_input['token_type_ids'].to(device)), output_model,
                          do_constant_folding=True,
                          input_names=['input_ids', "attention_mask", 'token_type_ids'],
                          output_names=['embedding'],
                          dynamic_axes={'input_ids': [0, 1], 'attention_mask': [0, 1], 'token_type_ids': [0, 1],
                                        'embedding': [0]},
                          opset_version=12)

class SimcseModel(nn.Module):
    """Simcse有监督模型定义"""
    def __init__(self, pretrained_model: str, pooling: str):
        super(SimcseModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.pooling = pooling
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            pprint(out.last_hidden_state[:, 0] )
            return out.last_hidden_state[:, 0]  # [batch, 768]

        if self.pooling == 'pooler':
            return out.pooler_output            # [batch, 768]

        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)    # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch, 768]

        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)    # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)    # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1) # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]

    def save_torchscript(self, output_model: str = "model.pt", max_length: int = 510):
        """
        When tracing, we use an example input to record the actions taken and capture the the model architecture.
        This works best when your model doesn't have control flow.
        If you do have control flow, you will need to use the scripting approach
        """
        query = max_length*"草"
        q_id = self.tokenizer(query, max_length=max_length, truncation=True, padding=True, return_tensors='pt')
        #pprint(q_id)
        traced_model = torch.jit.trace(self.bert, [q_id['input_ids'].to(device),
                                                    q_id['attention_mask'].to(device),
                                                    q_id['token_type_ids'].to(device)])
        torch.jit.save(traced_model, output_model)

    def save_onnx(self, output_model:str="model.onnx", max_length:int=510):
        query = max_length * "字"
        encoded_input = self.tokenizer(query, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        torch.onnx.export(self.model, (encoded_input['input_ids'].to(device), encoded_input['attention_mask'].to(device),
                                  encoded_input['token_type_ids'].to(device)), output_model,
                          do_constant_folding=True,
                          input_names=['input_ids', "attention_mask", 'token_type_ids'],
                          output_names=['embedding'],
                          dynamic_axes={'input_ids': [0, 1], 'attention_mask': [0, 1], 'token_type_ids': [0, 1],
                                        'embedding': [0]},
                          opset_version=12)


class Pytorch2Onnx(nn.Module):
    def __init__(self, model_path:str):
        super(Pytorch2Onnx, self).__init__()
        self.model = BertModel.from_pretrained(model_path, torchscript=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def save_onnx(self, output_model:str="model.onnx", max_length:int=510):
        query = max_length * "字"
        encoded_input = self.tokenizer(query, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        torch.onnx.export(self.model, (encoded_input['input_ids'].to(device), encoded_input['attention_mask'].to(device),
                                  encoded_input['token_type_ids'].to(device)), output_model,
                          do_constant_folding=True,
                          input_names=['input_ids', "attention_mask", 'token_type_ids'],
                          output_names=['embedding'],
                          dynamic_axes={'input_ids': [0, 1], 'attention_mask': [0, 1], 'token_type_ids': [0, 1],
                                        'embedding': [0]},
                          opset_version=12)

if __name__ == "__main__":
    '''
    model = SentenceBert("/Users/yanyongwen712/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2")
    #model.save_torchscript("sbert_model.pt")
    model.save_onnx("sbert_model.onnx")
    '''
    '''
    model = SimcseModel("/Users/yanyongwen712/.cache/torch/sentence_transformers/simcse-chinese-roberta-wwm-ext", "cls")
    #model.save_torchscript("simcse_model.pt", 100)
    model.save_onnx("simcse_model.onnx")
    '''
    Pytorch2Onnx("/Users/yanyongwen712/.cache/torch/sentence_transformers/BAAI_bge-small-zh-v1.5").save_onnx("bge_model.onnx")
