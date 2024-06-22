#!/bin/bash

if [  ! -f /app/model_data/ggml-large-v3.bin ]; then
    mkdir /app/model_data
    git clone https://github.com/ggerganov/whisper.cpp.git
    cd whisper.cpp
    bash ./models/download-ggml-model.sh large-v3
    mv ./models/ggml-large-v3.bin /app/model_data
fi
