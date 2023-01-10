import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, check_integrity
import os
from sklearn.model_selection import train_test_split


class UCIDataset(Dataset):
    def __init__(self, root, split='train', download=False):
        super(UCIDataset, self).__init__()
        self.root = root
        self.split = split

        if download and not self._check_integrity():
            self.downloads()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        for file in self.file_list:
            md5 = file[1]
            fpath = os.path.join(self.root, file[0].split('/')[-1])
            if not check_integrity(fpath, md5):
                return False
        return True

    def downloads(self):
        for file in self.file_list:
            md5 = file[1]
            download_url(file[0], self.root, file[0].split('/')[-1], md5)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)


class AdultDataset(UCIDataset):
    file_list = [
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
         '5d7c39d7b8804f071cdd1f2a7c460872'),
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
         '35238206dfdf7f1fe215bbb874adecdc'),
    ]

    def __init__(self, root, split='train', download=False, with_val=True):
        super(AdultDataset, self).__init__(root, split, download)

        if self.split == 'val':
            assert with_val

        if self.split in ['train', 'val']:
            self.data, self.labels = AdultDataset.preprocess_adult_to_binary_data(os.path.join(root, 'adult.data'))
            if with_val:
                data_train, data_val, labels_train, labels_val \
                    = train_test_split(self.data, self.labels, test_size=0.1, random_state=0)
                if split == 'train':
                    self.data, self.labels = data_train, labels_train
                elif split == 'val':
                    self.data, self.labels = data_val, labels_val
                else:
                    raise ValueError(split)
        else:
            self.data, self.labels = AdultDataset.preprocess_adult_to_binary_data(os.path.join(root, 'adult.test'))

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]

        return data, label

    @staticmethod
    def preprocess_adult_to_binary_data(data_file_name):

        attributes = [
            'age',
            'workclass',
            'fnlwgt',
            'education',
            'education-num',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital-gain',
            'capital-loss',
            'hours-per-week',
            'native-country'
        ]

        continuous_attributes = {
            'age': ['minor', 'very-young', 'young', 'middle-aged', 'senior'],
            # 'fnlwgt': 'Final-Weight',
            # 'education-num': 'Education-Num',
            'capital-gain': ['no-gain', 'small-gain', 'large-gain'],
            'capital-loss': ['no-loss', 'small-loss', 'large-loss'],  # 5k
            'hours-per-week': ['no-hours', 'mini-hours', 'half-hours', 'full-hours', 'more-hours', 'most-hours'],
        }

        discrete_attribute_options = [
            'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay',
            'Never-worked',
            'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th',
            '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool',
            'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent',
            'Married-AF-spouse',
            'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
            'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
            'Priv-house-serv', 'Protective-serv', 'Armed-Forces',
            'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried',
            'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black',
            'Female', 'Male',
            'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)',
            'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland',
            'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador',
            'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia',
            'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands',
            # 'Age', 'Final-Weight', 'Education-Num', 'Capital-Gain', 'Capital-Loss', 'Hours-Per-Week'
            *continuous_attributes['age'], *continuous_attributes['capital-gain'],
            *continuous_attributes['capital-loss'], *continuous_attributes['hours-per-week'],
        ]

        discrete_attribute_to_idx = {k: v for v, k in enumerate(discrete_attribute_options)}

        label_dict = {
            '>50K': 1.0,
            '<=50K': 0.0
        }

        def read_raw_data(filepath):
            with open(filepath, 'r') as f:
                data = f.readlines()

            for i in range(len(data)):
                if data[i].startswith('|') or len(data[i]) <= 2:
                    data[i] = None
                else:
                    data[i] = data[i].strip('\n').strip('.').strip().split(',')
                    data[i] = [d.strip() for d in data[i]]

            data = list(filter(lambda x: x is not None, data))

            return data

        def discard_missing_data(data):
            num_samples = len(data)
            for i in range(num_samples):
                if '?' in data[i]:
                    data[i] = None

            data = [sample for sample in data if sample is not None]
            return data

        def convert_sample_to_feature_vector(sample):
            D = len(discrete_attribute_options)
            vec = np.zeros(D)
            for i, attr_type in enumerate(attributes):
                if attr_type in ['education-num', 'fnlwgt']:
                    continue
                attr_value = sample[i]
                if attr_type in continuous_attributes:
                    attr_value = int(attr_value)
                    if attr_type == 'age':
                        if attr_value < 21:
                            attr_v = 'minor'
                        elif attr_value < 30:
                            attr_v = 'very-young'
                        elif attr_value < 45:
                            attr_v = 'young'
                        elif attr_value < 60:
                            attr_v = 'middle-aged'
                        else:
                            attr_v = 'senior'
                    elif attr_type == 'capital-gain':
                        if attr_value == 0:
                            attr_v = 'no-gain'
                        elif attr_value < 5_000:
                            attr_v = 'small-gain'
                        else:
                            attr_v = 'large-gain'
                    elif attr_type == 'capital-loss':
                        if attr_value == 0:
                            attr_v = 'no-loss'
                        elif attr_value < 5_000:
                            attr_v = 'small-loss'
                        else:
                            attr_v = 'large-loss'
                    elif attr_type == 'hours-per-week':
                        if attr_value == 0:
                            attr_v == 'no-hours'
                        elif attr_value <= 12:
                            attr_v == 'mini-hours'
                        elif attr_value <= 25:
                            attr_v == 'half-hours'
                        elif attr_value <= 40:
                            attr_v == 'full-hours'
                        elif attr_value < 60:
                            attr_v == 'more-hours'
                        else:
                            attr_v == 'most-hours'
                    else:
                        raise ValueError(attr_type)

                    vec_idx = discrete_attribute_to_idx[attr_v]
                else:
                    vec_idx = discrete_attribute_to_idx[attr_value]

                vec[vec_idx] = 1

            return vec

        def convert_data_to_feature_vectors(data):
            feat = [convert_sample_to_feature_vector(sample) for sample in data]
            return torch.tensor(feat).float()

        def get_labels(data):
            num_samples = len(data)
            labels = np.zeros(num_samples, dtype=np.float32)
            for i, sample in enumerate(data):
                labels[i] = label_dict[sample[-1]]

            return torch.tensor(labels).long()

        data = read_raw_data(data_file_name)
        data = discard_missing_data(data)

        feat = convert_data_to_feature_vectors(data)
        labels = get_labels(data)

        return feat, labels


class MONKsDataset(UCIDataset):
    file_list = [
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train',
         'fc1fc3a673e00908325c67cf16283335'),
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test',
         'de4255acb72fb29be5125a7c874e28a0'),
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.train',
         'f109ee3f95805745af6cdff06d6fbc94'),
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.test',
         '106cb9049ba5ccd7969a0bd5ff19681d'),
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.train',
         '613e44dbb8ffdf54d364bd91e4e74afd'),
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.test',
         '46815731e31c07f89422cf60de8738e7'),
    ]

    def __init__(self, root, style: int, split='train', download=False, with_val=False):
        super(MONKsDataset, self).__init__(root, split, download)
        self.style = style
        assert style in [1, 2, 3], style

        if self.split == 'val':
            assert with_val

        if self.split in ['train', 'val']:
            self.data, self.labels = MONKsDataset.preprocess_monks_to_binary_data(
                os.path.join(root, 'monks-{}.train'.format(style))
            )
            if with_val:
                data_train, data_val, labels_train, labels_val \
                    = train_test_split(self.data, self.labels, test_size=0.1, random_state=0)
                if split == 'train':
                    self.data, self.labels = data_train, labels_train
                elif split == 'val':
                    self.data, self.labels = data_val, labels_val
                else:
                    raise ValueError(split)
        else:
            self.data, self.labels = MONKsDataset.preprocess_monks_to_binary_data(
                os.path.join(root, 'monks-{}.test'.format(style))
            )

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]

        return data, label

    @staticmethod
    def preprocess_monks_to_binary_data(data_file_name):

        def read_raw_data(filepath):
            with open(filepath, 'r') as f:
                data = f.readlines()

            for i in range(len(data)):
                if len(data[i]) <= 2:
                    data[i] = None
                else:
                    data[i] = data[i].strip('\n').strip('.').strip().split(' ')
                    data[i] = [d for d in data[i]]
                    data[i] = data[i][:-1]

            data = list(filter(lambda x: x is not None, data))

            return data

        def convert_sample_to_feature_vector(sample):
            attribute_ranges = [3, 3, 2, 3, 4, 2]

            D = sum(attribute_ranges)
            vec = np.zeros(D)

            count = 0
            for i, attribute_range in enumerate(attribute_ranges):
                val = int(sample[i + 1]) - 1
                vec[count + val] = 1

                count += attribute_range

            return vec

        def convert_data_to_feature_vectors(data):
            feat = [convert_sample_to_feature_vector(sample) for sample in data]
            return torch.tensor(feat).float()

        def get_labels(data):
            num_samples = len(data)
            labels = np.zeros(num_samples, dtype=np.float32)
            for i, sample in enumerate(data):
                labels[i] = int(sample[0])

            return torch.tensor(labels).long()

        data = read_raw_data(data_file_name)

        feat = convert_data_to_feature_vectors(data)
        labels = get_labels(data)

        return feat, labels


class IrisDataset(UCIDataset):
    file_list = [
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
         '42615765a885ddf54427f12c34a0a070'),
    ]

    def __init__(self, root, split='train', download=False, with_val=False):
        super(IrisDataset, self).__init__(root, split, download)

        if self.split == 'val':
            assert with_val

        self.data, self.labels = IrisDataset.preprocess_iris_data(
            os.path.join(root, 'iris.data')
        )
        data_train, data_test, labels_train, labels_test \
            = train_test_split(self.data, self.labels, test_size=0.3, random_state=0)

        if self.split in ['train', 'val']:
            self.data, self.labels = data_train, labels_train
            if with_val:
                data_train, data_val, labels_train, labels_val \
                    = train_test_split(self.data, self.labels, test_size=0.2, random_state=0)
                if split == 'train':
                    self.data, self.labels = data_train, labels_train
                elif split == 'val':
                    self.data, self.labels = data_val, labels_val
                else:
                    raise ValueError(split)
        else:
            self.data, self.labels = data_test, labels_test

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]

        return data, label

    @staticmethod
    def preprocess_iris_data(data_file_name):

        label_dict = {
            'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 2
        }

        def read_raw_data(filepath):
            with open(filepath, 'r') as f:
                data = f.readlines()

            for i in range(len(data)):
                if len(data[i]) <= 2:
                    data[i] = None
                else:
                    data[i] = data[i].strip('\n').strip('.').strip().split(',')
                    data[i] = [d for d in data[i]]
                    data[i] = data[i]

            data = list(filter(lambda x: x is not None, data))

            return data

        def convert_sample_to_feature_vector(sample):
            vec = np.zeros(4)
            for i in range(4):
                vec[i] = float(sample[i])
            return vec

        def convert_data_to_feature_vectors(data):
            feat = [convert_sample_to_feature_vector(sample) for sample in data]
            return torch.tensor(feat).float()

        def get_labels(data):
            num_samples = len(data)
            labels = np.zeros(num_samples, dtype=np.int64)

            for i, sample in enumerate(data):
                labels[i] = label_dict[sample[-1]]

            return torch.tensor(labels).long()

        data = read_raw_data(data_file_name)

        feat = convert_data_to_feature_vectors(data)
        labels = get_labels(data)

        return feat, labels


class BreastCancerDataset(UCIDataset):
    file_list = [
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data',
         'd887db31e2b99e39c4b7fc79d2f36a3a'),
    ]

    def __init__(self, root, split='train', download=False, with_val=False):
        super(BreastCancerDataset, self).__init__(root, split, download)

        if self.split == 'val':
            assert with_val

        self.data, self.labels = BreastCancerDataset.preprocess_data(
            os.path.join(root, 'breast-cancer.data')
        )
        data_train, data_test, labels_train, labels_test \
            = train_test_split(self.data, self.labels, test_size=0.25, random_state=0)

        if self.split in ['train', 'val']:
            self.data, self.labels = data_train, labels_train
            if with_val:
                data_train, data_val, labels_train, labels_val \
                    = train_test_split(self.data, self.labels, test_size=0.2, random_state=0)
                if split == 'train':
                    self.data, self.labels = data_train, labels_train
                elif split == 'val':
                    self.data, self.labels = data_val, labels_val
                else:
                    raise ValueError(split)
        else:
            self.data, self.labels = data_test, labels_test

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]

        return data, label

    @staticmethod
    def preprocess_data(data_file_name):

        attributes = {
            'age': ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'],
            'menopause': ['lt40', 'ge40', 'premeno'],
            'tumor-size': ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49',
                           '50-54', '55-59'],
            'inv-nodes': ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30-32',
                          '33-35', '36-39'],
            'node-caps': ['yes', 'no'],
            'deg-malig': ['1', '2', '3'],
            'breast': ['left', 'right'],
            'breast-quad': ['left_up', 'left_low', 'right_up', 'right_low', 'central'],
            'irradiat': ['yes', 'no'],
        }

        label_dict = {
            'no-recurrence-events': 0,
            'recurrence-events': 1
        }

        def read_raw_data(filepath):
            with open(filepath, 'r') as f:
                data = f.readlines()

            for i in range(len(data)):
                if data[i].startswith('|') or len(data[i]) <= 2:
                    data[i] = None
                else:
                    data[i] = data[i].strip('\n').strip('.').strip().split(',')
                    data[i] = [d.strip() for d in data[i]]

            data = list(filter(lambda x: x is not None, data))

            return data

        def discard_missing_data(data):
            num_samples = len(data)
            for i in range(num_samples):
                if '?' in data[i]:
                    data[i] = None

            data = [sample for sample in data if sample is not None]
            return data

        def convert_sample_to_feature_vector(sample):
            D = sum([len(attributes[key]) for key in attributes])
            vec = np.zeros(D)

            count = 0
            for i, key in enumerate(attributes):
                val = attributes[key].index(sample[i + 1])
                vec[count + val] = 1

                count += len(attributes[key])

            return vec

        def convert_data_to_feature_vectors(data):
            feat = [convert_sample_to_feature_vector(sample) for sample in data]
            return torch.tensor(feat).float()

        def get_labels(data):
            num_samples = len(data)
            labels = np.zeros(num_samples, dtype=np.float32)
            for i, sample in enumerate(data):
                labels[i] = label_dict[sample[0]]

            return torch.tensor(labels).long()

        data = read_raw_data(data_file_name)
        data = discard_missing_data(data)

        feat = convert_data_to_feature_vectors(data)
        labels = get_labels(data)

        return feat, labels


if __name__ == '__main__':
    adult_train_set = AdultDataset('data-uci', split='train', download=True)
    adult_test_set = AdultDataset('data-uci', split='test')
    print('Adult')
    print(adult_train_set.data.shape)
    print(adult_train_set.labels.shape)

    monks1_train_set = MONKsDataset('data-uci', 1, split='train', download=True)
    monks1_test_set = MONKsDataset('data-uci', 1, split='test')
    monks2_train_set = MONKsDataset('data-uci', 2, split='train')
    monks2_test_set = MONKsDataset('data-uci', 2, split='test')
    print('\nMONKs')
    print(monks1_train_set.data.shape)
    print(monks1_train_set.labels.shape)
    print(monks1_test_set.data.shape)
    print(monks1_test_set.labels.shape)
    print(monks2_train_set.data.shape)
    print(monks2_train_set.labels.shape)
    print(monks2_test_set.data.shape)
    print(monks2_test_set.labels.shape)

    iris_train_set = IrisDataset('data-uci', split='train', download=True)
    iris_test_set = IrisDataset('data-uci', split='test')
    print('\nIris')
    print(iris_train_set.data.shape)
    print(iris_train_set.labels.shape)
    print(iris_test_set.data.shape)
    print(iris_test_set.labels.shape)

    bc_train_set = BreastCancerDataset('data-uci', split='train', download=True)
    bc_test_set = BreastCancerDataset('data-uci', split='test')
    print('\nBreast Cancer')
    print(bc_train_set.data.shape)
    print(bc_train_set.labels.shape)
    print(bc_test_set.data.shape)
    print(bc_test_set.labels.shape)

