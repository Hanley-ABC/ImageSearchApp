from pymilvus import MilvusClient, DataType
import numpy as np
import time
from tqdm import tqdm
import feature_extractor as fe


def query_vector(collection_name, client, topk=3,query_vector=None, metric_type="COSINE",
                 output_fields=None):
    '''
    :param collection_name: 表名
    :param client: 库
    :param topk: 要查询多少个
    :param query_future: 直接传的向量，和模型一般只能传一个
    :param metric_type: 相似度判断方式
    :param output_fields: 输出的列需要哪些
    :return:
    '''
    t1 = time.time()
    res = client.search(
        collection_name=collection_name,
        anns_field="vector",
        data=[query_vector],
        limit=topk,
        search_params={"metric_type": metric_type},
        output_fields=output_fields
    )
    t2 = time.time()
    print(f'查询相似向量消耗时间{t2 - t1}')
    return res
