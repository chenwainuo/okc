#!/bin/bash
# usage: run_box.sh <logname> <timeout_s> <cmd...>   (runs on the box under the GPU lock, unbuffered, log to logs/tensorcore_<logname>.log)
name=$1; shift; to=$1; shift
mkdir -p /root/okc/logs
cd /root
timeout $to /root/gpu_lock.sh /root/okc_run.sh env PYTHONUNBUFFERED=1 "$@" > /root/okc/logs/tensorcore_$name.log 2>&1
echo "EXIT $?" >> /root/okc/logs/tensorcore_$name.log
