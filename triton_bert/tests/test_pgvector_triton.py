from typing import Optional, List

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlmodel import SQLModel, Field, select

from triton_bert.pgvector_triton import PgvectorTriton

class Sentence(SQLModel, table=True):
    # 默认是serial
    id: Optional[int] = Field(default=None, primary_key=True)
    # 注意这里不需要再用 Field 的啰嗦用法
    sentence: str
    # Vector()内不加数字，就是不限定向量的维度。
    embedding: List[float] = Field(sa_column=Column(Vector()))

class QA(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    question: str
    answer: str
    # Vector()内不加数字，就是不限定向量的维度。
    embedding: List[float] = Field(sa_column=Column(Vector()))


class PgvectorTritonCustomized(PgvectorTriton):
    TABLE_MODEL: QA
    def _make_rows(self, texts: List[str], **kwargs):
        embeddings = self.model(texts)
        return [self.TABLE_MODEL(answer=text, embedding=embedding, question=question) for text, embedding, question in zip(texts, embeddings, kwargs['questions']) ]


if __name__ == "__main__":
    '''
    TEST 1
    '''
    instance = PgvectorTriton(db_user="xx", db_password='xxx',
                              db_instance="xxx", db_port="3671",
                   db_schema="xx", create_table=True,
                   triton_host="xx", model="bge-m3",
                   vocab="/Users/xx/Codes/pingan_health_rag/models/bge-m3",
                              table_model=Sentence
                   )

    '''
    qas = instance.load_texts("dataset/medical_qa.jsonl")
    answers = [qa['answers'][0] for qa in qas]
    instance.insert_vectors(answers)
    '''

    recalls: List[Sentence] = instance.retrieval_vectors("我喉咙有些干")

    print(recalls[0].sentence)

    '''
    TEST 2
    '''
    instance = PgvectorTritonCustomized(db_user="xx", db_password='xxx',
                              db_instance="xx", db_port="3671",
                   db_schema="xx", create_table=True,
                   triton_host="xx", model="bge-m3",
                   vocab="/Users/xx/Codes/pingan_health_rag/models/bge-m3",
                              table_model=QA)

    qas = instance.load_texts("dataset/medical_qa.jsonl")
    answers = [qa['answers'][0] for qa in qas]
    questions = [qa['questions'][0][0] for qa in qas]
    instance.insert_vectors(answers, questions=questions)

    recalls: List[QA] = instance.retrieval_vectors("我工作时间长了，颈椎会不舒服")

    print(recalls[0].answer)