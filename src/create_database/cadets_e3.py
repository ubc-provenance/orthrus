import os
import re
import torch
from tqdm import tqdm
import hashlib
from config import *
from provnet_utils import *

def extract_subject_file_uuid(file_path, filelist):
    subject_uuid2path = {}
    file_uuid2path = {}

    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in (f):
                if "com.bbn.tc.schema.avro.cdm18.Subject" in line:
                    pattern = '{"com.bbn.tc.schema.avro.cdm18.Subject":{"uuid":"(.*?)"'
                    match_ans = re.findall(pattern, line)[0]
                    subject_uuid2path[match_ans] = None
                elif "com.bbn.tc.schema.avro.cdm18.FileObject" in line:
                    pattern = '{"com.bbn.tc.schema.avro.cdm18.FileObject":{"uuid":"(.*?)"'
                    match_ans = re.findall(pattern, line)[0]
                    file_uuid2path[match_ans] = None

    return subject_uuid2path, file_uuid2path

def store_netflow(file_path, cur, connect, index_id, filelist):
    # Parse data from logs
    netobjset = set()
    netobj2hash = {}
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if "NetFlowObject" in line:
                    try:
                        res = re.findall(
                            'NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":"(.*?)","localPort":(.*?),"remoteAddress":"(.*?)","remotePort":(.*?),',
                            line)[0]

                        nodeid = res[0]
                        srcaddr = res[2]
                        srcport = res[3]
                        dstaddr = res[4]
                        dstport = res[5]

                        nodeproperty = srcaddr + "," + srcport + "," + dstaddr + "," + dstport
                        hashstr = stringtomd5(nodeid)
                        netobj2hash[nodeid] = [hashstr, nodeproperty]
                        netobj2hash[hashstr] = nodeid
                        netobjset.add(hashstr)
                    except:
                        pass

    # Store data into database
    datalist = []
    net_uuid2hash = {}
    for i in netobj2hash.keys():
        if len(i) != 64:
            datalist.append([i] + [netobj2hash[i][0]] + netobj2hash[i][1].split(",") + [index_id])
            net_uuid2hash[i] = netobj2hash[i][0]
            index_id += 1

    sql = '''insert into netflow_node_table
                         values %s
            '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()

    return index_id, net_uuid2hash

def store_subject(file_path, cur, connect, index_id, filelist, subject_uuid2path):
    # Parse data from logs
    scusess_count = 0
    fail_count = 0
    subject_obj2hash = {}
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in (f):
                if "Event" in line:
                    subject_uuid = re.findall(
                        '"subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"},(.*?)"exec":"(.*?)",', line)
                    try:
                        if subject_uuid[0][0] in subject_uuid2path:
                            subject_obj2hash[subject_uuid[0][0]] = [subject_uuid2path[subject_uuid[0][0]], subject_uuid[0][-1]] #{uuid:[path, cmd]}
                        scusess_count += 1
                    except:
                        try:
                            subject_obj2hash[subject_uuid[0][0]] = "null"
                        except:
                            pass
                        fail_count += 1
    # Store into database
    datalist = []
    subject_uuid2hash = {}
    for i in subject_obj2hash.keys():
        if len(i) != 64:
            datalist.append([i] + [stringtomd5(i)] + subject_obj2hash[
                i] + [index_id])  # ([uuid, hashstr, path, cmdLine, index_id]) and hashstr=stringtomd5(uuid)
            subject_uuid2hash[i] = stringtomd5(i)
            index_id += 1

    sql = '''insert into subject_node_table
                         values %s
            '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()

    return index_id, subject_uuid2hash

def store_file(file_path, cur, connect, index_id, filelist, file_uuid2path):
    file_obj2hash = {}
    fail_count = 0
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if "Event" in line:
                    try:
                        object_uuid = \
                            re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"},', line)[0]
                        if object_uuid in file_uuid2path:
                            if '"predicateObjectPath":null,' not in line and '<unknown>' not in line:
                                object_path = re.findall('"predicateObjectPath":{"string":"(.*?)"}', line)
                                if len(object_path) == 0:
                                    file_obj_name = None
                                else:
                                    file_obj_name = object_path[0]
                            else:
                                file_obj_name = None
                            file_obj2hash[object_uuid] = file_obj_name  #{uuid: path}
                    except:
                        fail_count += 1

    datalist = []
    file_uuid2hash = {}
    for i in file_obj2hash.keys():
        if len(i) != 64:
            datalist.append([i] + [stringtomd5(i), file_obj2hash[i]] + [index_id])
            file_uuid2hash[i] = stringtomd5(i)
            index_id += 1

    sql = '''insert into file_node_table
                         values %s
            '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()

    return index_id, file_uuid2hash

def create_node_list(cur):
    nodeid2msg = {}

    # netflow
    sql = """
        select * from netflow_node_table;
        """
    cur.execute(sql)
    records = cur.fetchall()
    for i in records:
        nodeid2msg[i[1]] = i[-1]

    # subject
    sql = """
    select * from subject_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()
    for i in records:
        nodeid2msg[i[1]] = i[-1]

    # file
    sql = """
    select * from file_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()
    for i in records:
        nodeid2msg[i[1]] = i[-1]

    return nodeid2msg #{hash_id:index_id}

def write_event_in_DB(cur, connect, datalist):
    sql = '''insert into event_table
                         values %s
            '''
    ex.execute_values(cur,sql, datalist,page_size=10000)
    connect.commit()

def store_event(file_path, cur, connect, reverse, nodeid2msg, subject_uuid2hash, file_uuid2hash, net_uuid2hash, filelist):
    datalist = []
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line:
                    relation_type = re.findall('"type":"(.*?)"', line)[0]
                    if relation_type not in exclude_edge_type:
                        subject_uuid = re.findall('"subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"', line)
                        predicateObject_uuid = re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"', line)

                        if len(subject_uuid) > 0 and len(predicateObject_uuid) > 0:
                            if subject_uuid[0] in subject_uuid2hash and (predicateObject_uuid[0] in subject_uuid2hash or
                                                                         predicateObject_uuid[0] in file_uuid2hash or
                                                                         predicateObject_uuid[0] in net_uuid2hash):
                                event_uuid = re.findall('{"datum":{"com.bbn.tc.schema.avro.cdm18.Event":{"uuid":"(.*?)",', line)[0]
                                time_rec = re.findall('"timestampNanos":(.*?),', line)[0]
                                time_rec = int(time_rec)
                                subjectId = subject_uuid2hash[subject_uuid[0]]
                                if predicateObject_uuid[0] in file_uuid2hash:
                                    objectId = file_uuid2hash[predicateObject_uuid[0]]
                                elif predicateObject_uuid[0] in net_uuid2hash:
                                    objectId = net_uuid2hash[predicateObject_uuid[0]]
                                else:
                                    objectId = subject_uuid2hash[predicateObject_uuid[0]]
                                if relation_type in reverse:
                                    datalist.append(
                                        [objectId, nodeid2msg[objectId], relation_type, subjectId,
                                         nodeid2msg[subjectId], event_uuid, time_rec])
                                else:
                                    datalist.append(
                                        [subjectId, nodeid2msg[subjectId], relation_type, objectId,
                                         nodeid2msg[objectId], event_uuid, time_rec])

    sql = '''insert into event_table
                         values %s
            '''
    ex.execute_values(cur, sql, datalist, page_size=50000)
    connect.commit()

def main(cfg):
    raw_dir = cfg.dataset.raw_dir
    filelist = get_all_filelist(raw_dir)

    cur, connect = init_database_connection(cfg)

    subject_uuid2path, file_uuid2path = extract_subject_file_uuid(file_path=raw_dir,filelist=filelist)

    index_id = 0

    print("Processing netflow data")
    index_id, net_uuid2hash = store_netflow(file_path=raw_dir, cur=cur, connect=connect, index_id=index_id, filelist=filelist)

    print("Processing subject data")
    index_id, subject_uuid2hash = store_subject(file_path=raw_dir, cur=cur, connect=connect, index_id=index_id, filelist=filelist, subject_uuid2path=subject_uuid2path)

    print("Processing file data")
    index_id, file_uuid2hash = store_file(file_path=raw_dir, cur=cur, connect=connect, index_id=index_id, filelist=filelist, file_uuid2path=file_uuid2path)

    print("Extracting the node list")
    nodeid2msg = create_node_list(cur=cur)

    print("Processing the events")
    store_event(
        file_path=raw_dir,
        cur=cur,
        connect=connect,
        reverse=edge_reversed,
        nodeid2msg=nodeid2msg,
        subject_uuid2hash=subject_uuid2hash,
        file_uuid2hash=file_uuid2hash,
        net_uuid2hash=net_uuid2hash,
        filelist=filelist
    )
