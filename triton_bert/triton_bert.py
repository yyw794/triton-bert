
from typing import Optional, List, Union, Tuple
import tritonclient.grpc as grpcclient
from more_itertools.more import chunked
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
from tritonclient.utils import InferenceServerException, triton_to_np_dtype

class TritonBert:
    def __init__(self, model: str, vocab: str, triton_host: str="localhost", triton_grpc_port: int=8001, 
        model_max_len: int=512, padding: Optional[PaddingStrategy]=None, 
        truncation: TruncationStrategy=TruncationStrategy.LONGEST_FIRST):
        
        self.tokenizer = AutoTokenizer.from_pretrained(vocab)
        
        self.triton_url = f"{triton_host}:{triton_grpc_port}"
        self.connect_triton()
        
        self.model = model
        self.model_max_len = model_max_len
        self.padding = padding
        self.truncation = truncation
        self.parse_triton_model_config()

        

    def connect_triton(self):
        self.triton_client = grpcclient.InferenceServerClient(url=self.triton_url)

    def parse_triton_model_config(self):
        model_config = self.triton_client.get_model_config(self.model, as_json=True)
        self.model_config = model_config
        self.max_batch_size = int(model_config['config'].get('max_batch_size', 1))
        #assume the input dim is (sequence) or (batch, sequence)
        self.sequence_len =  int(model_config['config']['input'][0]['dims'][-1])
        self.dynamic_sequence = True if self.sequence_len == -1 else False
        if not self.padding:
            self.padding = PaddingStrategy.LONGEST if self.dynamic_sequence else PaddingStrategy.MAX_LENGTH
        #assume dynamic sequence is from [0, 512] typically
        self.model_max_len = self.model_max_len if self.dynamic_sequence else self.sequence_len
        self.model_max_sequence_len = self.model_max_len
        self.model_input_names = [_input['name'] for _input in model_config['config']['input']]
        self.model_input_data_types = [_input['data_type'].replace("TYPE_", "") for _input in model_config['config']['input']]
        self.model_output_names = [_output['name'] for _output in model_config['config']['output']]
        

    def triton_infer(self, encoded_input):
        if not encoded_input:
            return None

        batch = len(encoded_input['input_ids'])

        if self.padding == PaddingStrategy.MAX_LENGTH:
            max_sequence_len = self.model_max_len
        else:
            max_sequence_len = len(max(encoded_input['input_ids'], key=lambda x: len(x)))
        #assume bert model input dim is (batch, sequence)
        inputs = [grpcclient.InferInput(input_name, [batch, max_sequence_len], data_type) for input_name, data_type in zip(self.model_input_names, self.model_input_data_types)]
        outputs = [grpcclient.InferRequestedOutput(output_name) for output_name in self.model_output_names]
        # bert: ['input_ids', 'attention_mask', 'token_type_ids']
        # but, roberta has no token_type_ids
        encoded_input_keys = [x for x in ['input_ids', 'attention_mask', 'token_type_ids'] if x in encoded_input]
        for i, k in enumerate(encoded_input_keys):
            inputs[i].set_data_from_numpy(encoded_input[k].astype(triton_to_np_dtype(self.model_input_data_types[i])))
        try:
            triton_ret = self.triton_client.infer(model_name=self.model,inputs=inputs,outputs=outputs)
        except InferenceServerException as error:
            #if triton restart, we will miss the connection. so, we will trigger to reconnect again
            self.connect_triton()
            triton_ret = self.triton_client.infer(model_name=self.model,inputs=inputs,outputs=outputs)
        
        return [triton_ret.as_numpy(output_name) for output_name in self.model_output_names]

    def preprocess(self, texts, text_pairs=[]):
        if not texts:
            return
        if text_pairs:
            encoded_input = self.tokenizer(text=texts, text_pair=text_pairs, padding=self.padding, \
                truncation=self.truncation, max_length=self.model_max_len, return_tensors='np')
        else:
            encoded_input = self.tokenizer(text=texts, padding=self.padding, \
                truncation=self.truncation, max_length=self.model_max_len, return_tensors='np')
        return encoded_input

    def proprocess(self, triton_output):
        #raise NotImplementedError
        # triton_output[0] means we only get the first output. if you have two outputs, CHANGE THIS
        return triton_output[0].tolist()

    def _predict(self, texts, text_pairs=[]):
        if not texts:
            return []
        encoded_input = self.preprocess(texts, text_pairs)
        if not encoded_input:
            return []
        triton_output = self.triton_infer(encoded_input)
        if not triton_output:
            return []
        return self.proprocess(triton_output)

    def predict(self, texts, text_pairs=[]):
        outputs= []
        if text_pairs:
            for _texts, _text_pairs in zip(chunked(texts, self.max_batch_size), chunked(text_pairs, self.max_batch_size)):
                outputs.extend(self._predict(_texts, _text_pairs))
        else:
            for _texts in chunked(texts, self.max_batch_size):
                outputs.extend(self._predict(_texts))
        return outputs

    def __call__(self, texts: list, text_pairs: list=[]):
        if isinstance(texts, str):
            texts = [texts]
        return self.predict(texts, text_pairs)

    def encode(self, sentences: Union[str, List[str], Tuple[str]]):
        if isinstance(sentences, str):
            return self.__call__(sentences)[0]
        else:
            return self.__call__(sentences)







