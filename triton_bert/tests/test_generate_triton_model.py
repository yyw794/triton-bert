from triton_bert.model_4_triton import Model4TritonServer

if __name__ == "__main__":
    pretrained_model = "/Users/yanyongwen712/.cache/torch/sentence_transformers/simcse-chinese-roberta-wwm-ext"
    model = Model4TritonServer(pretrained_model=pretrained_model)
    model.save_torchscript("model/simcse_model.pt")
    model.save_onnx("model/simcse_model.onnx")