## Creating database
```commandline
# execute the psql with postgres user
sudo -u postgres psql

# create the database
postgres=# create database database_name;

# switch to the created database
postgres=# \connect database_name;

# create the event table and grant the privileges to postgres
database_name=# create table event_table
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
database_name=# alter table event_table owner to postgres;
database_name=# create unique index event_table__id_uindex on event_table (_id); grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;

# create the file table
database_name=# create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    index_id  bigint,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);
database_name=# alter table file_node_table owner to postgres;

# create the netflow table
database_name=# create table netflow_node_table
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
database_name=# alter table netflow_node_table owner to postgres;

# create the subject table
database_name=# create table subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    path      varchar,
    cmd       varchar,
    index_id  bigint,
    constraint subject_node_table_pk
        primary key (node_uuid, hash_id)
);
database_name=# alter table subject_node_table owner to postgres;
```