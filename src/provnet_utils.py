import pytz
from time import mktime
from datetime import datetime
import time
import psycopg2
from psycopg2 import extras as ex
import os.path as osp
import os
import copy
import logging
import torch
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric import *
from tqdm import tqdm
import networkx as nx
import numpy as np
import math
import copy
import time
import xxhash
import gc
import random
import csv
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score as ap_score,
    balanced_accuracy_score,
)

import re

from config import *
import hashlib
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

def stringtomd5(originstr):
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256() # TODO: check if we might remove it in the future
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()

def ns_time_to_datetime(ns):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00.000000000
    """
    dt = datetime.fromtimestamp(int(ns) // 1000000000)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s

def ns_time_to_datetime_US(ns):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00.000000000
    """
    tz = pytz.timezone('US/Eastern')
    dt = pytz.datetime.datetime.fromtimestamp(int(ns) // 1000000000, tz)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s

def time_to_datetime_US(s):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00
    """
    tz = pytz.timezone('US/Eastern')
    dt = pytz.datetime.datetime.fromtimestamp(int(s), tz)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')

    return s

def datetime_to_ns_time(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    timeStamp = timeStamp * 1000000000
    return timeStamp

def datetime_to_ns_time_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    tz = pytz.timezone('US/Eastern')
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp * 1000000000
    return int(timeStamp)

def datetime_to_timestamp_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    tz = pytz.timezone('US/Eastern')
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp
    return int(timeStamp)

def init_database_connection(cfg):
    if cfg.preprocessing.build_graphs.use_all_files:
        database_name = cfg.dataset.database_all_file
    else:
        database_name = cfg.dataset.database

    if cfg.database.host is not None:
        connect = psycopg2.connect(database = database_name,
                                   host = cfg.database.host,
                                   user = cfg.database.user,
                                   password = cfg.database.password,
                                   port = cfg.database.port
                                  )
    else:
        connect = psycopg2.connect(database = database_name,
                                   user = cfg.database.user,
                                   password = cfg.database.password,
                                   port = cfg.database.port
                                  )
    cur = connect.cursor()
    return cur, connect

def gen_nodeid2msg(cur, use_cmd=True, use_port=False):
    # node hash id to node label and type
    # {hash_id: index_id} and {index_id: {node_type:msg}}
    indexid2msg = {}

    # netflow
    sql = """
        select * from netflow_node_table;
        """
    cur.execute(sql)
    records = cur.fetchall()

    for i in records:
        hash_id = i[1]
        remote_ip = str(i[4])
        remote_port = str(i[5])
        index_id = i[-1] # int
        indexid2msg[hash_id] = index_id
        if use_port:
            indexid2msg[index_id] = {'netflow': remote_ip + ':' +remote_port}
        else:
            indexid2msg[index_id] = {'netflow': remote_ip}

    # subject
    sql = """
    select * from subject_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()

    for i in records:
        hash_id = i[1]
        path = str(i[2])
        cmd = str(i[3])
        index_id = i[-1]
        indexid2msg[hash_id] = index_id
        if use_cmd:
            indexid2msg[index_id] = {'subject': path + ' ' +cmd}
        else:
            indexid2msg[index_id] = {'subject': path}

    # file
    sql = """
    select * from file_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()

    for i in records:
        hash_id = i[1]
        path = str(i[2])
        index_id = i[-1]
        indexid2msg[hash_id] = index_id
        indexid2msg[index_id] = {'file': path}

    return indexid2msg #{hash_id: index_id} and {index_id: {node_type:msg}}

def std(t):
    t = np.array(t)
    return np.std(t)

def var(t):
    t = np.array(t)
    return np.var(t)

def mean(t):
    t = np.array(t)
    return np.mean(t)

def percentile_90(t):
    sorted_data = np.sort(t)
    Q = np.percentile(sorted_data, 90)
    return Q

def percentile_75(t):
    sorted_data = np.sort(t)
    Q = np.percentile(sorted_data, 75)
    return Q

def percentile_50(t):
    sorted_data = np.sort(t)
    Q = np.percentile(sorted_data, 50)
    return Q

def hashgen(l):
    """Generate a single hash value from a list. @l is a list of
    string values, which can be properties of a node/edge. This
    function returns a single hashed integer value."""
    hasher = xxhash.xxh64()
    for e in l:
        hasher.update(e)
    return hasher.intdigest()


def split_filename(path):
    '''
    Given a path, split it based on '/' and file extension.
    e.g.
        "/home/test/Desktop/123.txt" => "home test Desktop 123 txt"
    :param path: the name of the path
    :return: the split path name
    '''
    file_name, file_extension = os.path.splitext(os.path.basename(path))
    file_extension = file_extension.replace(".","")
    result = ' '.join(path.split('/')[1:-1]) + ' ' + file_name + ' ' + file_extension
    return result

def get_logger(name: str, filename: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("")
    logger.info(f"START LOGGING FOR SUBTASK: {name}")
    logger.info("")
    
    log("")
    log(f"START LOGGING FOR SUBTASK: {name}")
    log("")
    
    return logger

def get_all_files_from_folders(base_dir: str, folders: list[str]):
    paths = [os.path.abspath(os.path.join(base_dir, sub, f))
        for sub in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, sub)) and sub in folders
        for f in os.listdir(os.path.join(base_dir, sub))]
    paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return paths

def listdir_sorted(path: str):
    files = os.listdir(path)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f)))) # sorted by ascending number
    return files

def remove_underscore_keys(data, keys_to_keep=[], keys_to_rm=[]):
    for key in list(data.keys()):
        if (key in keys_to_rm) or (key.startswith('_') and key not in keys_to_keep):
            del data[key]
        elif isinstance(data[key], dict):
            data[key] = dict(data[key])
            remove_underscore_keys(data[key], keys_to_keep, keys_to_rm)
    return data

def compute_mcc(tp, fp, tn, fn):
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    
    if denominator == 0:
        return 0
    
    mcc = numerator / denominator
    return mcc

def classifier_evaluation(y_test, y_test_pred, scores):
    labels_exist = sum(y_test) > 0
    if labels_exist:
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    else:
        tn, fp, fn, tp = 1, 1, 1, 1  # only to not break tests

    fpr = fp/(fp+tn)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    fscore=2*(precision*recall)/(precision+recall)

    try:
        auc_val=roc_auc_score(y_test, scores)
    except: auc_val=float("nan")
    try:
        ap=ap_score(y_test, scores)
    except: ap=float("nan")
    try:
        balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    except: balanced_acc=float("nan")
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    lr_plus = sensitivity / (1 - specificity)
    dor = (tp * tn) / (fp * fn)
    mcc = compute_mcc(tp, fp, tn, fn)
    
    log(f'total num: {len(y_test)}')
    log(f'tn: {tn}')
    log(f'fp: {fp}')
    log(f'fn: {fn}')
    log(f'tp: {tp}')
    log('')

    log(f"ap: {ap}")
    log(f"precision: {precision}")
    log(f"recall: {recall}")
    log(f"fpr: {fpr}")
    log(f"fscore: {fscore}")
    log(f"accuracy: {accuracy}")
    log(f"balanced acc: {balanced_acc}")
    log(f"auc: {auc_val}")
    log(f"lr(+): {lr_plus}")
    log(f"dor: {dor}")
    log(f"mcc: {mcc}")

    stats = {
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "fpr": round(fpr, 7),
        "fscore": round(fscore, 5),
        "ap": round(ap, 5),
        "accuracy": round(accuracy, 5),
        "balanced_acc": round(balanced_acc, 5),
        "auc": round(auc_val, 5),
        "lr(+)": round(lr_plus, 5),
        "dor": round(dor, 5),
        "mcc": mcc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    return stats

def get_detected_attacks(cfg):
    cfg.dataset.attack_to_time_window

def get_indexid2msg(cur, use_cmd=True, use_port=False):
    indexid2msg = {}

    # netflow
    sql = """
        select * from netflow_node_table;
        """
    cur.execute(sql)
    records = cur.fetchall()

    log(f"Number of netflow nodes: {len(records)}")

    for i in records:
        remote_ip = str(i[4])
        remote_port = str(i[5])
        index_id = i[-1] # int
        if use_port:
            indexid2msg[index_id] = ['netflow', remote_ip + ':' +remote_port]
        else:
            indexid2msg[index_id] = ['netflow', remote_ip]

    # subject
    sql = """
    select * from subject_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()

    log(f"Number of process nodes: {len(records)}")

    for i in records:
        path = str(i[2])
        cmd = str(i[3])
        index_id = i[-1]
        if use_cmd:
            indexid2msg[index_id] = ['subject', path + ' ' +cmd]
        else:
            indexid2msg[index_id] = ['subject', path]

    # file
    sql = """
    select * from file_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()

    log(f"Number of file nodes: {len(records)}")

    for i in records:
        path = str(i[2])
        index_id = i[-1]
        indexid2msg[index_id] = ['file', path]

    return indexid2msg #{index_id: [node_type, msg]}

def tokenize_subject(sentence: str):
    new_sentence = re.sub(r'\\+', '/', sentence)
    return word_tokenize(new_sentence.replace('/', ' / '))
    # return word_tokenize(sentence.replace('/',' ').replace('=',' = ').replace(':',' : '))
def tokenize_file(sentence: str):
    new_sentence = re.sub(r'\\+', '/', sentence)
    return word_tokenize(new_sentence.replace('/',' / '))
def tokenize_netflow(sentence: str):
    return word_tokenize(sentence.replace(':',' : ').replace('.',' . '))

def log(msg: str, *args):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {msg}", *args)

def get_device(cfg):
    if cfg._use_cpu:
        return torch.device("cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu"):
        log("Warning: the device is CPU instead of CUDA")
    return device

def get_node_to_path_and_type(cfg):
    out_path = cfg.preprocessing.build_graphs._node_id_to_path
    out_file = os.path.join(out_path, "node_to_paths.pkl")
    
    if not os.path.exists(out_file):
        os.makedirs(out_path, exist_ok=True)
        cur, connect = init_database_connection(cfg)
        
        queries = {
            "file": "SELECT index_id, path FROM file_node_table;",
            "netflow": "SELECT index_id, src_addr, dst_addr, src_port, dst_port FROM netflow_node_table;",
            "subject": "SELECT index_id, path, cmd FROM subject_node_table;"
        }
        node_to_path_type = {}
        for node_type, query in queries.items():
            cur.execute(query)
            rows = cur.fetchall()
            for row in rows:
                if node_type == "netflow":
                    index_id, src_addr, dst_addr, src_port, dst_port = row
                    node_to_path_type[index_id] = {"path": f"{str(src_addr)}:{str(src_port)}->{str(dst_addr)}:{str(dst_port)}", "type": node_type}
                elif node_type == "file":
                    index_id, path = row
                    node_to_path_type[index_id] = {"path": str(path), "type": node_type}
                elif node_type == "subject":
                    index_id, path, cmd = row
                    node_to_path_type[index_id] = {"path": str(path), "type": node_type, "cmd": cmd}

        torch.save(node_to_path_type, out_file)
        connect.close()
        
    else:
        node_to_path_type = torch.load(out_file)
        
    return node_to_path_type

def get_all_filelist(filepath):
    file_paths = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            file_paths.append(file)
    return file_paths
