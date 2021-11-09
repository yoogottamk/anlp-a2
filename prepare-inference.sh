#!/bin/bash

set -o errexit -o pipefail

mkdir -p model
MODEL_WEIGHTS_HASH=bfab29506b433c4a2ab257a82ab4b859

function download_model {
    curl -fSL -k \
        https://web.iiit.ac.in/~yoogottam.khandelwal/model.tar.gz \
        -omodel/model.tar.gz
}

until [[ $( md5sum model/model.tar.gz | cut -d' ' -f1 ) = $MODEL_WEIGHTS_HASH ]]; do
    download_model
done
