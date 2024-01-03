It is easy to use bert in triton now.
Algorithm Engineer only need to focus to write proprocess function to make his model work.

pls see examples



# install dependency
poetry shell
poetry install

# run examples
## run triton server
```bash
# for example
docker run -d  --name triton-server   --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864  --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /home/yanyongwen712/triton_models:/models  nvcr.io/nvidia/tritonserver::22.08-py3 tritonserver --model-repository=/models  --model-control-mode=poll  --exit-on-error=false --log-verbose 1
# configure triton model folder
```
## prepare model for triton server
```bash
cd examples
python save_model_for_triton_server.py
# sftp put examples/model/cpu/xxx triton_server_model_folder
docker logs -f trition-server    
# check whether it is loaded successfully
```

## prepare PG with pgvector extension
...

## run example
```bash
# change triton server ip , triton model name and local transformer model folder
python retrieval_pgvector.py
```

