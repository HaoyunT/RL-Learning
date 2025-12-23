#!/bin/bash
gcc main.c -o actor_loop \
  -I/home/khadas/khadas/rknpu2/runtime/RK3588/Linux/librknn_api/include \
  -lrknnrt -lm