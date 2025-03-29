# Uncompress DARPA TC files

At this stage, the data folder should contain the downloaded dataset files (.gz), the java client (tar.gz) and the schema files (.avdl, .avsc).

Then go back to the orthrus container within ```/home```, run following command to convert files:

```shell
./settings/scripts/uncompress_darpa_files.sh /data/
```

> This may take multiple hours based on the dataset.
