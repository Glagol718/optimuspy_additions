#!/bin/bash
mkdir -p opsc-bin
mkdir -p opsc-bin/artifacts
docker container rm ops-cli-files
docker create --name ops-cli-files ops-group/ops-cli:latest
docker cp ops-cli-files:/home/user/Lib/llvm-3.3.install/lib opsc-bin/artifacts/llvm-lib
docker cp ops-cli-files:/home/user/ops-build/lib opsc-bin/artifacts/ops-lib
docker cp ops-cli-files:/home/user/ops-build/bin opsc-bin/artifacts/ops-bin
mkdir -p opsc-bin/artifacts/temp
mkdir -p opsc-bin/artifacts/temp/lib
mv opsc-bin/artifacts/llvm-lib/* opsc-bin/artifacts/temp/lib/
mv opsc-bin/artifacts/ops-lib/* opsc-bin/artifacts/temp/lib/
mv opsc-bin/artifacts/ops-bin/* opsc-bin/artifacts/temp/
rm -rf opsc-bin/artifacts/llvm-lib
rm -rf opsc-bin/artifacts/ops-lib
rm -rf opsc-bin/artifacts/ops-bin
cd opsc-bin/artifacts/temp
tar cfJ ../../opsc.tar.xz ./*
