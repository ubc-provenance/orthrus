import os
import re
import torch
from tqdm import tqdm
import hashlib
from config import *
from provnet_utils import *
def store_netflow(file_path, cur, connect, index_id, filelist):
    # Parse data from logs
    netobjset = set()
    netobj2hash = {}
    net_uuid2hash = {}
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if "NetFlowObject" in line:
                    try:
                        res = re.findall(
                            'NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":"(.*?)","localPort":(.*?),',
                            line)[0]

                        nodeid = res[0]
                        srcaddr = res[2]
                        srcport = res[3]

                        remote = re.findall(
                            '"remoteAddress":"(.*?)","remotePort":(.*?),',
                            line
                        )
                        if len(remote) > 0:
                            dstaddr = res[0][0]
                            dstport = res[0][1]
                        else:
                            dstaddr = None
                            dstport = None

                        # nodeproperty = srcaddr + "," + srcport + "," + str(dstaddr) + "," + str(dstport)
                        hashstr = stringtomd5(nodeid)
                        netobj2hash[nodeid] = [hashstr, [srcaddr, srcport, dstaddr, dstport]]
                        netobj2hash[hashstr] = nodeid
                        netobjset.add(hashstr)
                    except:
                        pass

    # Store data into database
    datalist = []

    for i in netobj2hash.keys():
        if len(i) != 64:
            datalist.append([i] + [netobj2hash[i][0]] + netobj2hash[i][1] + [index_id])
            net_uuid2hash[i] = netobj2hash[i][0]
            index_id += 1

    sql = '''insert into netflow_node_table
                            values %s
            '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()

    return index_id, net_uuid2hash

def store_subject(file_path, cur, connect, index_id, filelist):
    # Parse data from logs
    subject_obj2hash = {}
    subject_uuid2hash = {}
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in (f):
                if "schema.avro.cdm18.Subject" in line:
                    subject_uuid = re.findall(
                        'avro.cdm18.Subject":{"uuid":"(.*?)",(.*?)"path":"(.*?)"', line)
                    node_uuid = subject_uuid[0][0]
                    node_path = subject_uuid[0][2]
                    subject_cmd = re.findall(
                        ',"cmdLine":{"string":"(.*?)"},', line
                    )
                    if len(subject_cmd) == 0:
                        node_cmd = None
                    else:
                        node_cmd = subject_cmd[0][0]
                    subject_obj2hash[node_uuid] = [node_path, node_cmd]

    # Store into database
    datalist = []

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

def store_file(file_path, cur, connect, index_id, filelist):
    file_obj2hash = {}
    fail_count = 0
    file_uuid2hash = {}
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if "avro.cdm18.FileObject" in line:
                    Object_uuid = re.findall('avro.cdm18.FileObject":{"uuid":"(.*?)",(.*?)"path":"(.*?)"', line)
                    try:
                        file_obj2hash[Object_uuid[0][0]] = Object_uuid[0][-1]
                    except:
                        fail_count += 1

    datalist = []

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

    for file in tqdm(filelist):
        datalist = []
        with open(file_path + file, "r") as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line:
                    relation_type = re.findall('"type":"(.*?)"', line)[0]
                    if relation_type in rel2id:
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