#!/bin/bash

# 清空 temp 目录
if [ -d "temp" ]; then
    rm -rf temp/*
    echo "temp 目录已清空"
else
    echo "temp 目录不存在"
fi