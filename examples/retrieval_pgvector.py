# reference: https://github.com/pgvector/pgvector-python
import os

from pgvector.sqlalchemy import Vector

from sqlalchemy import Column
from typing import Optional, List
# reference: https://sqlmodel.tiangolo.com/
from sqlmodel import select, Field, Session, SQLModel, create_engine

from pprint import pprint

from examples.biencoder import Biencoder

class Sentence(SQLModel, table=True):
    # 默认是serial
    id: Optional[int] = Field(default=None, primary_key=True)
    # 注意这里不需要再用 Field 的啰嗦用法
    sentence: str
    # Vector()内不加数字，就是不限定向量的维度。如果需要金陵
    embedding: List[float] = Field(sa_column=Column(Vector()))



if __name__ == "__main__":
    # TODO: CHANGE ME! local CACHE_FOLDER, remote triton_host, remote triton model name, local vocab folder
    CACHE_FOLDER = "/Users/yanyongwen712/.cache/torch/sentence_transformers"
    model = Biencoder(triton_host="30.171.160.44", model="simcse_onnx",
                      vocab=f"{CACHE_FOLDER}/sentence-transformers_all-MiniLM-L6-v2")  # dim 768
    #model = Biencoder(triton_host="30.171.160.44", model="bge_onnx", vocab=f"{CACHE_FOLDER}/BAAI_bge-small-zh-v1.5") # dim 384
    #model = Biencoder(triton_host="30.171.160.44", model="sbert_onnx", vocab=f"{CACHE_FOLDER}/simcse-chinese-roberta-wwm-ext") #
    # if you don't have local model, you can use these to download from hugging face
    #from sentence_transformers import SentenceTransformer
    #model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

    from urllib.parse import quote_plus #为了解决密码里有@这种情况
    password = os.environ.get("DM_DEV_DB_ADMIN_PASSWORD")
    # 注意开头是postgresql 不是 postgres
    engine = create_engine(f"postgresql://budevadmin:{quote_plus(password)}@D0PEIMDMINFO-postgresql.dbdev.paic.com.cn:3671/pivotoperation")
    # 当表不存在时，这个会自动创建表
    SQLModel.metadata.create_all(engine)

    # 闭包函数
    def insert_embeddings():
        sentences = ['小明借了100元给小红',
            '小红借了100元给小明',
            '小明打了小红，赔了100元',
                     '小明和小红一起赚了100元']

        embeddings = model.encode(sentences)

        with Session(engine) as session:
            for sentence, embedding in zip(sentences, embeddings):
                test_vector = Sentence(sentence=sentence, embedding=embedding)
                session.add(test_vector)
            session.commit()

    def retrieval(sentence: str, limit :int=5)->List[str]:

        with Session(engine) as session:
            ret = session.exec(select(Sentence).order_by(Sentence.embedding.l2_distance(model.encode(sentence)))
                               .limit(limit))
            similar_sentences = [s.sentence for s in ret.all()]

        return similar_sentences

    #insert_embeddings()
    pprint(retrieval('小明从小红那借走了100元'))
    
    