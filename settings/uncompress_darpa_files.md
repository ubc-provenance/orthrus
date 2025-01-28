# Uncompress DARPA TC files

At this stage, the folder should contain the downloaded dataset files (tar.gz), the java client (tar.gz) and the schema/ folder.

If not installed, install maven with `sudo apt-get install maven`

Then go back to the orthrus repo within the `orthrus/settings/scripts/` folder and run this command after replacing the path:

```shell
./uncompress_darpa_files.sh /full/absolute/path/to/downloaded_files
```
