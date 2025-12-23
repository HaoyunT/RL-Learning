#!/bin/bash
gcc main.c -o rknn_agent \
  -I/home/khadas/khadas/rknpu2/runtime/RK3588/Linux/librknn_api/include \
  -lrknnrt -lm