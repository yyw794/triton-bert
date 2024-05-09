import torch

from triton_bert.model.model_4_triton import Model4TritonServer

if __name__ == "__main__":
    model = Model4TritonServer(pretrained_model='hfl/chinese-bert-wwm-ext')
    model.save_torchscript("model/chinese-bert-wwm-ext.pt")
    model.save_onnx("model/chinese-bert-wwm-ext.onnx")