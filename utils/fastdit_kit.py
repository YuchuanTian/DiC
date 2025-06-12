# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, features_dir):
        # features_dir, _features/_labels
        L = os.listdir(features_dir)
        print(f'---> Folders in {features_dir}: {L}')
        for name in L:
            if name.endswith('_features'):
                self.features_dir = os.path.join(features_dir, name)
            elif name.endswith('_labels'):
                self.labels_dir = os.path.join(features_dir, name)


        self.features_files = sorted(os.listdir(self.features_dir), key=lambda x:int(x.split('_')[0])*8+int(x[-5]))[:-1]
        self.labels_files = sorted(os.listdir(self.labels_dir), key=lambda x:int(x.split('_')[0])*8+int(x[-5]))[:-1]
        assert len(self.features_files) == len(self.features_files) == 1281167 # ImageNet

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features).squeeze(0), torch.from_numpy(labels).squeeze(0)