# Uncompress DARPA TC files

At this stage, the data folder should contain the downloaded dataset files (tar.gz), the java client (tar.gz) and the schema files (.avdl, .avsc).

If not installed, install maven with `sudo apt-get install maven`

Then go back to the orthrus repo within the `orthrus/settings/scripts/` folder and run this command after replacing the path:

```shell
./uncompress_darpa_files.sh /full/absolute/path/to/data_folder
```

**NOTE.** If using the docker environment (within the container), the path should be replaced as ```/data/``` 

> This may take multiple hours based on the dataset.
