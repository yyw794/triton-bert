import torch
from torch import Tensor

from triton_bert.model_4_triton import Model4TritonServer
from FlagEmbedding.BGE_M3 import BGEM3ForInference

class BGEM3ForTritonInference(BGEM3ForInference):

    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor,
                return_dense: bool = True,
                return_sparse: bool = False,
                return_colbert: bool = False,
                return_sparse_embedding: bool = False):
        '''
        修改入参和返回格式
        '''
        assert return_dense or return_sparse or return_colbert, 'Must choose one or more from `return_colbert`, `return_sparse`, `return_dense` to set `True`!'

        last_hidden_state = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state

        output = {}
        if return_dense:
            dense_vecs = self.dense_embedding(last_hidden_state, attention_mask)
            output['dense_vecs'] = dense_vecs
        if return_sparse:
            sparse_vecs = self.sparse_embedding(last_hidden_state, input_ids,
                                                return_embedding=return_sparse_embedding)
            output['sparse_vecs'] = sparse_vecs
        if return_colbert:
            colbert_vecs = self.colbert_embedding(last_hidden_state, attention_mask)
            output['colbert_vecs'] = colbert_vecs

        if self.normlized:
            if 'dense_vecs' in output:
                output['dense_vecs'] = torch.nn.functional.normalize(output['dense_vecs'], dim=-1)
            if 'colbert_vecs' in output:
                output['colbert_vecs'] = torch.nn.functional.normalize(output['colbert_vecs'], dim=-1)
        '''
         PyTorch execute failure: output must be of type Tensor, List[str] or Tuple containing one of these two types. It should not be a List / Dictionary of Tensors or a Scalar
        '''
        #return output
        # temp code. only support dense vectors
        return torch.nn.functional.normalize(dense_vecs, dim=-1) if self.normlized else dense_vecs

if __name__ == "__main__":
    pretrained_model = "BAAI/bge-m3"
    n = BGEM3ForTritonInference(pretrained_model)

    model = Model4TritonServer(model=n.model, tokenizer=n.tokenizer)
    model.save_torchscript("model/bgem3_model.pt")
    model.save_onnx("model/bgem3_model.onnx")
