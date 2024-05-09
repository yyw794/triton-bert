import time
from typing import Union, List, Optional
from urllib.parse import quote_plus  # 为了解决密码里有@这种情况

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlmodel import SQLModel, create_engine, Session, select, Field

from triton_bert.triton_bert import TritonBert

import jsonlines
class PgvectorTriton():
    def __init__(self, db_user:str, db_password:str, db_instance:str, db_port:Union[str, int], db_schema:str,
                       create_table: bool,
                 triton_host: str, model: str, vocab: str,
                 table_model: SQLModel
                 ):
        self.engine = self._init_pg_engine(db_user, db_password, db_instance, db_port, db_schema, create_table)
        self.model = self._init_triton_model(triton_host, model, vocab)
        self.TABLE_MODEL = table_model

    def _init_pg_engine(self, db_user:str, db_password:str, db_instance:str, db_port:Union[str, int], db_schema:str,
                       create_table: bool=True):
        # 注意开头是postgresql 不是 postgres
        engine = create_engine(
            f"postgresql://{db_user}:{quote_plus(db_password)}@{db_instance}:{db_port}/{db_schema}")
        # 当表不存在时，这个会自动创建表
        if create_table:
            SQLModel.metadata.create_all(engine)
        self.engine = engine
        return engine

    def _init_triton_model(self, triton_host:str, model:str, vocab:str):
        model = TritonBert(triton_host=triton_host, model=model, vocab=vocab)
        self.model = model
        return model


    def insert_vectors(self, texts: List[str], **kwargs):
        with Session(self.engine) as session:
            for row in self._make_rows(texts, **kwargs):
                session.add(row)
            session.commit()

    def retrieval_vectors(self, texts: Union[str, List[str]], limit :int=10)->List[str]:
        # text -> vector
        str_input = False
        if isinstance(texts, str):
            str_input = True
            texts = [texts]
        embeddings = self.model.encode(texts)

        # retrieval from db
        with Session(self.engine) as session:
            rets = [session.exec(self._retrieval_with_select(embedding, limit)).all() for embedding in embeddings]
        return rets[0] if str_input else rets


    # NEED to OVERRIDE
    def load_texts(self, file_path):
        jsons = []
        with open(file_path, "r+", encoding="utf8") as f:
            for line in jsonlines.Reader(f):
                jsons.append(line)
        return jsons

    # NEED to OVERRIDE
    def _make_rows(self, texts: List[str], **kwargs):
        embeddings = self.model(texts)
        return [self.TABLE_MODEL(sentence=text, embedding=embedding) for text, embedding in zip(texts, embeddings) ]

    # NEED to OVERRIDE
    def _retrieval_with_select(self, embedding, limit: int):
        return select(self.TABLE_MODEL).order_by(self.TABLE_MODEL.embedding.l2_distance(embedding)).limit(limit)


