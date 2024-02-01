#!/bin/bash

# Script to convert ONNX models to TensorRT engines and start NVIDIA Triton Inference Server.

# Usage:
# Run this script to convert ONNX models to TensorRT engines and start Triton Inference Server.
# Use the flag --force-build to rebuild TensorRT engines even if they already exist.

# Prerequisites:
# - NVIDIA TensorRT must be installed.
# - NVIDIA Triton Inference Server must be installed and configured.

# Script Flow:
# 1. Convert YOLOv7 ONNX model to TensorRT engine with FP16 precision.
# 2. Convert YOLOv7 Quantized and Aware Training (QAT) ONNX model to TensorRT engine with INT8 precision.
# 3. Start Triton Inference Server with the converted models.
# Check if ONNX model files exist
if [[ ! -f "./models_onnx/yolov7/yolov7_end2end.onnx" || ! -f "./models_onnx/yolov7_qat/yolov7_qat_end2end.onnx" ]]; then
    echo "YOLOv7 ONNX model files not found. Attempting to download..."
    cd ./models_onnx
    bash ./download_models.sh
    cd ../
fi

# Check if ONNX model files exist
if [[ ! -f "./models_onnx/yolov7/yolov7_end2end.onnx" ]]; then
    echo "YOLOv7 ONNX model file not found: ./models_onnx/yolov7/yolov7_end2end.onnx"
    exit 1
fi

if [[ ! -f "./models_onnx/yolov7_qat/yolov7_qat_end2end.onnx" ]]; then
    echo "YOLOv7 QAT ONNX model file not found: ./models_onnx/yolov7_qat/yolov7_qat_end2end.onnx"
    exit 1
fi

# Check if force-build flag is set
force_build=false

if [[ "$1" == "--force-build" ]]; then
    force_build=true
fi

# Convert YOLOv7 ONNX model to TensorRT engine with FP16 precision if force flag is set or model does not exist
if [[ $force_build == true || ! -f "./models/yolov7/1/model.plan" ]]; then
    /usr/src/tensorrt/bin/trtexec \
        --onnx=./models_onnx/yolov7/yolov7_end2end.onnx \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:8x3x640x640 \
        --maxShapes=images:8x3x640x640 \
        --fp16 \
        --workspace=4096 \
        --saveEngine=./models/yolov7/1/model.plan

    # Check return code of trtexec
    if [[ $? -ne 0 ]]; then
        echo "Conversion of YOLOv7 ONNX model to TensorRT engine failed"
        exit 1
    fi
fi

# Convert YOLOv7 QAT ONNX model to TensorRT engine with INT8 precision if force flag is set or model does not exist
if [[ $force_build == true || ! -f "./models/yolov7_qat/1/model.plan" ]]; then
    /usr/src/tensorrt/bin/trtexec \
        --onnx=./models_onnx/yolov7_qat/yolov7_qat_end2end.onnx \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:8x3x640x640 \
        --maxShapes=images:8x3x640x640 \
        --fp16 \
        --int8 \
        --workspace=4096 \
        --saveEngine=./models/yolov7_qat/1/model.plan

    # Check return code of trtexec
    if [[ $? -ne 0 ]]; then
        echo "Conversion of YOLOv7 QAT ONNX model to TensorRT engine failed"
        exit 1
    fi
fi

# Start Triton Inference Server with the converted models
/opt/tritonserver/bin/tritonserver \
    --model-repository=/apps/models \
    --disable-auto-complete-config \
    --log-verbose=0
