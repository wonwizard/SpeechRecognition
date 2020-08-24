# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/usr/bin/env bash

DATA_SET="LibriSpeech"
#DATA_ROOT_DIR="/datasets"
DATA_ROOT_DIR="/NAS/datasets"
DATA_DIR="${DATA_ROOT_DIR}/${DATA_SET}"
python utils/download_librispeech.py utils/librispeech.csv $DATA_DIR -e ${DATA_ROOT_DIR}/ --skip_download --skip_checksum