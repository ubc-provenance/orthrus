## Install/setup postgres

- check if postgres is already install: `ls /etc/postgresql/`
- if empty, install it with `sudo apt-get update && sudo apt-get install postgresql`
- edit the file of the corresponding version: `sudo vi /etc/postgresql/{version}/main/pg_hba.conf`
- locate the line `local   all   postgres   peer` and switch it to `local   all   postgres   md5`
- restart the service with: `sudo service postgresql restart`
- run psql with the postgres user: `sudo -u postgres psql -p 5433`
- run the following steps, replacing `clearscope_e3` by the name you want (usually the dataset's name)

## Creating database
```commandline
create database clearscope_e3;

\connect clearscope_e3;

create table event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    event_uuid    varchar not null,
    timestamp_rec bigint,
    _id           serial
);
alter table event_table owner to postgres;
create unique index event_table__id_uindex on event_table (_id); grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;

create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    index_id  bigint,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);
alter table file_node_table owner to postgres;

create table netflow_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    index_id  bigint,
    constraint netflow_node_table_pk
        primary key (node_uuid, hash_id)
);
alter table netflow_node_table owner to postgres;

create table subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    path      varchar,
    cmd       varchar,
    index_id  bigint,
    constraint subject_node_table_pk
        primary key (node_uuid, hash_id)
);
alter table subject_node_table owner to postgres;
```
