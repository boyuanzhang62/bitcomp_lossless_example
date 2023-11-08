# Bitcomp Utility

## Introduction
Bitcomp is a powerful tool for lossless file compression and decompression. It efficiently compresses files, ensuring that your data is stored in a compact format without any loss of information.

## Installation
Before using Bitcomp, make sure to install it on your system. Follow the installation guide provided with the software package.

## Usage Instructions

### Preparation

Modify the `Makefile` to change the `"/home/bozhan/repo/nvcomp/lib"` and `"/home/bozhan/repo/nvcomp/include/"` to your own installed nvcomp path. Then:

```bash
make -j
```


### Compression
To compress a file with Bitcomp, use the following command:

```bash
./bitcomp_example -c /path/to/file
```

This will compress the file located at `/path/to/file` using Bitcomp's lossless compression algorithm.

### Decompression
To decompress a file that has been compressed using Bitcomp, use the command:

```bash
./bitcomp_example -d /path/to/compressed/file
```

Here, `/path/to/compressed/file` is the path to the file that you want to decompress.

### Roundtrip Verification
For a roundtrip process (compress and then decompress a file), and to verify the integrity and correctness of the process, use:

```bash
./bitcomp_example -r /path/to/file
```

This command performs both compression and decompression on `/path/to/file`, allowing you to verify that the original file and the decompressed file are identical.
