#!/bin/bash

set -o errexit -o pipefail

RAW_DATASET_HASH=487a51d1789b174d13bdc4926e5fb300
export DATASET_DIR=${DATASET_DIR:-$PWD}

function download_dataset {
    curl -fSL \
        https://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz \
        -o$DATASET_DIR/raw-dataset.tar.gz
}

function extract_dataset {
    tar xzf $DATASET_DIR/raw-dataset.tar.gz -C $DATASET_DIR
    mkdir -p $DATASET_DIR/dataset
    find swb_ms98_transcriptions -type f -name '*-trans.text' -exec bash -c 'cut -d" " -f4- {} > $DATASET_DIR/dataset/`basename {}`' \;
}

[[ $( md5sum $DATASET_DIR/raw-dataset.tar.gz | cut -d' ' -f1 ) = $RAW_DATASET_HASH ]] || download_dataset
extract_dataset
