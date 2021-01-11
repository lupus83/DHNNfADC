import argparse
from collections import Counter
from typing import Any, List, Optional, Dict
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset

from torchio.transforms import Pad, RandomAffine

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
import logging
import pdb
import models
import torch.optim as optim
from pathlib import Path

import patsy

from models import ModelFactory

LOG = logging.getLogger(__name__)

invalid_patIDs = [69, 112, 135, 162, 166, 167, 168, 178, 188, 205, 210, 384, 420, 422, 429, 443, 551, 668, 669, 679, 702, 722, 1009, 1092, 1168, 1188, 1226, 1241, 1245, 1352, 1408, 2002, 2010, 2018, 2022, 2073, 2123, 2130, 2146, 2180, 2184, 2187, 2234, 2263, 2274, 2301, 2304, 2315, 2332, 2378, 2379, 2389, 4005, 4051, 4054, 4061, 4072, 4114, 4168, 4184, 4187, 4199, 4212, 4214, 4268, 4275, 4277, 4310, 4332, 4351, 4380, 4381, 4388, 4419, 4430, 4499, 4513, 4543, 4556, 4607, 4624, 4641, 4706, 4713, 4741, 4750, 4767, 4799, 4813, 4842, 4845, 4855, 4874, 4899, 4947, 4960, 5097, 6033]    

def round_val(var):
    return '{0:.2f}'.format(round(100.0 * var, 2))

class AdniVolumeDataset(Dataset):

    def __init__(self, data, input_dim=64, masking="vol_without_bg", augment=False, heterog_exp=False, normalize_non_image_data=False, raw_dim=60, is_multiclass_exp=False):

        assert not (masking == 'concat' and augment), 'Augmentation is not supported for mask concat!'
        assert input_dim > raw_dim, 'Input dimension must be bigger then initial volume size!'
        
        self.data = data
        self.masking = masking
        self.input_dim = input_dim
        self.heterogeneous_experiment=heterog_exp
        self.pad = Pad(padding=int((input_dim-raw_dim)/2), padding_mode='edge')  # arg padding is before and after each dim
        if normalize_non_image_data:
            if is_multiclass_exp:
                self.normalize_non_img = [3.0, 374.0, 479.0, 213.0, 1.75121, 2.66921]
            else:
                self.normalize_non_img = [3.0, 300.0, 371.0, 213.0, 1.66565, 2.66921]
        else:
            self.normalize_non_img = None

        if augment:
            if self.masking == 'mask':
                self.transform = RandomAffine(
                    scales=(1,1),  # no scaling
                    degrees=45,  # +-45 degree in each dimension
                    translation=6,  # +-6mm offset in each dimension. Default pad mode is 'otsu'
                    image_interpolation='nearest',  # preserve image range
                    p=0.8)
            else:
                self.transform = RandomAffine(
                    scales=(1,1),  # no scaling
                    degrees=45,  # +-45 degree in each dimension
                    translation=6,  # +-6mm offset in each dimension. Default pad mode is 'otsu'
                    image_interpolation='linear',
                    p=0.8)
        else:
            self.transform = None

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        vol, non_image_data, label = self.data[index]

        # expand channel_dim
        if self.masking == 'concat':
            vol_t = vol  # shape 2, 60, 60, 60. vol[1] is in range [0|1]
        else:
            vol_t = vol[np.newaxis].astype(np.float32)  # shape 1, 60, 60, 60

        # normalize
        if self.masking != 'mask':
            min_val = np.amin(vol_t[0])
            max_val = np.amax(vol_t[0])
            vol_t[0] = (vol_t[0] - min_val) / (max_val - min_val)

        # pad to match input dimensions
        vol_t = self.pad(vol_t)

        assert vol_t[0].shape == (self.input_dim, self.input_dim, self.input_dim), 'Invalid input shape!'

        # augmentation
        if self.transform is not None:
            vol_t = self.transform(vol_t)

        # normalize non image data
        if self.normalize_non_img is not None:
            for i in range(len(self.normalize_non_img)):
                non_image_data[-(i+1)] = non_image_data[-(i+1)] / self.normalize_non_img[-(i+1)]
            assert max(non_image_data[-len(self.normalize_non_img):]) <= 1.0, f'Normalization failed on data {non_image_data}'

        if self.heterogeneous_experiment:
            return vol_t, np.array(non_image_data), np.array([label], dtype=np.float32)
        else:
            return vol_t, np.array([label], dtype=np.float32)


class AdniVolumeDataset_findlr(Dataset):

    def __init__(self, data, dim=64, masking="vol_without_bg", predicting=False, heterog_exp=False):
        self.data = data
        self.dim = dim
        self.masking = masking
        self.prediction = predicting
        self.heterogeneous_experiment=heterog_exp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vol, non_image_data, label = self.data[index]

        if (self.masking == "concat"):
            vol_t = add_random_padding(vol[0], prediction=self.prediction, dim=self.dim) / 255.
            vol_t = np.concatenate(
                [np.expand_dims(vol_t, axis=0), \
                np.expand_dims(add_random_padding(vol[1], prediction=self.prediction, dim=self.dim), axis=0)], \
                axis=0).astype(np.float32)
        elif (self.masking == "mask"):
            vol_t = add_random_padding(vol, prediction=self.prediction, dim=self.dim)
            vol_t = vol_t[np.newaxis].astype(np.float32)
        else:
            vol_t = add_random_padding(vol, prediction=self.prediction, dim=self.dim) / 255.
            vol_t = vol_t[np.newaxis].astype(np.float32)

        if self.heterogeneous_experiment:
            return vol_t, np.array(non_image_data), np.array([label], dtype=np.float32)
        else:
            return vol_t, label


class AiblVolumeDataset(Dataset):

    def __init__(self, data, input_dim=64, masking="vol_with_bg", raw_dim=60):
        self.data = data
        self.input_dim = input_dim
        self.masking = masking
        self.pad = Pad(padding=int((input_dim-raw_dim)/2), padding_mode='edge')  # arg padding is before and after each dim


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vol, label = self.data[index]

        # expand channel_dim
        if self.masking == 'concat':
            vol_t = vol  # shape 2, 60, 60, 60. vol[1] is in range [0|1]
        else:
            vol_t = vol[np.newaxis].astype(np.float32)  # shape 1, 60, 60, 60

        # normalize
        if self.masking != 'mask':
            min_val = np.amin(vol_t[0])
            max_val = np.amax(vol_t[0])
            vol_t[0] = (vol_t[0] - min_val) / (max_val - min_val)

        # pad to match input dimensions
        vol_t = self.pad(vol_t)

        assert vol_t[0].shape == (self.input_dim, self.input_dim, self.input_dim), 'Invalid input shape!'

        return vol_t, np.array([label], dtype=np.float32)


#   @param hippocampus  options: Left-Hippocampus, Right-Hippocampus (shape 60, 60, 60), Left-and-Right-Hippocampus (shape is then 114x60x60)
#   @param modality     options: vol_without_bg, vol_with_bg, mask, concat
#                       if concat, vol_with_bg and mask are concatenated
#
def load_aibl_volumes(path,
                    hippocampus: str = "Left-Hippocampus",
                    masking: str = "vol_without_bg",
                    binary_classification: bool = False,
                    just_baseline: bool = False,
                    debug: bool = False):
    if binary_classification:
        label_coding = {"NL": 0, "AD": 1}
    else:
        label_coding = {"NL": 0, "MCI": 1, "AD": 2}

    data = []
    patients = []
    viscodes = []
    counter = Counter()
    with h5py.File(str(path), "r") as hf:

        debug_counter = 0
        for image_uid, g in hf.items():

            label = g.attrs["DX"]
            if label not in label_coding:
                continue

            debug_counter += 1
            # if debugging just load one batch
            if (debug and debug_counter > 256):
                break

            patients.append(g.attrs["RID"])
            if just_baseline:
                viscodes.append(g.attrs["VISCODE"])
            label_code = label_coding[label]

            
            # select volume
            vols = g[hippocampus]
            # select masking
            if (masking == "concat"):

                # create new channel
                left_hippocampus = np.concatenate(
                                    [np.expand_dims(vols["vol_with_bg"][:], axis=0),
                                    np.expand_dims(vols["mask"][:], axis=0)],
                                    axis=0)
            else:
                left_hippocampus = vols[masking][:]

            data.append((left_hippocampus, label_code))

            counter[label] += 1

    if just_baseline:
        data_out = []
        patients_out = []

        # 1. filter patient ids
        pat_ids, index, counts = np.unique(patients, return_index=True, return_counts=True)

        # 2. append data of patients with just one visit
        patients_with_one_visit_indices = index[counts==1]
        for i in patients_with_one_visit_indices:
            data_out.append(data[i])
            patients_out.append(patients[i])

        # 3. find and append baseline or earlist visit
        for patient in pat_ids[counts > 1]:
            patient_visits_indices = [i for i in range(len(patients)) if patients[i] == patient]
            # 3.1
            patient_vscodes = [viscodes[i] for i in patient_visits_indices]
            earliest_vscode = 'bl' if 'bl' in patient_vscodes else sorted(patient_vscodes)[0]
            # 3.2
            local_idx = patient_vscodes.index(earliest_vscode)
            idx = patient_visits_indices[local_idx]
        
            data_out.append(data[idx])
            patients_out.append(patients[idx])
        lbls = np.array(data_out)[:, 1]
        eval_lbls = np.unique(lbls, return_counts=True)
        LOG.debug("Data has %d labels: %r", len(eval_lbls[0]), eval_lbls)
    else:
        data_out = data
        patients_out = patients
        LOG.debug("Data has %d labels: %r", len(counter), counter)
    return data_out, np.array(patients_out)

#   @param hippocampus  options: Left-Hippocampus, Right-Hippocampus (shape 60, 60, 60), Left-and-Right-Hippocampus (shape is then 114x60x60)
#   @param modality     options: vol_without_bg, vol_with_bg, mask, concat
#                       if concat, vol_with_bg and mask are concatenated
#   @return data, patient_ids(, visit_codes if True)
def load_adni_volumes(path,
                    hippocampus: str = "Left-Hippocampus",
                    masking: str = "vol_without_bg",
                    binary_classification: bool = False,
                    debug: bool = False,
                    visit_codes=False):
    if binary_classification:
        label_coding = {"CN": 0, "Dementia": 1}
    else:
        label_coding = {"CN": 0, "MCI": 1, "Dementia": 2}

    data = []
    patients = []
    counter = Counter()
    if visit_codes:
        viscodes = []
    with h5py.File(str(path), "r") as hf:

        debug_counter = 0
        for image_uid, g in hf.items():

            label = g.attrs["DX"]
            if label not in label_coding:
                continue

            debug_counter += 1
            # if debugging just load one batch
            if (debug and debug_counter > 256):
                break

            patients.append(g.attrs["RID"])
            if visit_codes:
                viscodes.append(g.attrs["VISCODE"])
            label_code = label_coding[label]

            # select volume
            vols = g[hippocampus]
            # select masking
            if (masking == "concat"):

                # create new channel
                left_hippocampus = np.concatenate(
                                    [np.expand_dims(vols["vol_with_bg"][:], axis=0),
                                    np.expand_dims(vols["mask"][:], axis=0)],
                                    axis=0)
            else:
                left_hippocampus = vols[masking][:]

            data.append((left_hippocampus, label_code))

            counter[label] += 1

    LOG.debug("Data has %d labels: %r", len(counter), counter)
    if visit_codes:
        return data, np.array(patients), np.array(viscodes)
    else:    
        return data, np.array(patients)


def split_stratified_by_label(
    data: List[Any],
    test_size: float,
    seed: int,
    ) -> Dict[str, List[Any]]:

    random_state = np.random.RandomState(seed)
    indices = np.arange(len(data))

    cv = StratifiedShuffleSplit(n_splits=1, test_size=2 * test_size, random_state=random_state)
    labels = np.array([d[-1] for d in data])
    train_idx, valid_test_idx = next(cv.split(indices, y=labels))

    cv.test_size = 0.5
    valid_idx, test_idx = next(cv.split(valid_test_idx, y=labels[valid_test_idx]))

    cv_data = {
        "train": [data[i] for i in train_idx],
        "valid": [data[i] for i in valid_test_idx[valid_idx]],
        "test":  [data[i] for i in valid_test_idx[test_idx]],
    }

    return cv_data

def split_by_group(
    data: List[Any],
    groups: np.ndarray,
    test_size: float,
    seed: int,
    visit_codes: Optional[np.ndarray] = None
    ) -> Dict[str, List[Any]]:

    assert len(data) == len(groups), "length of groups does not match number of samples"

    random_state = np.random.RandomState(seed)
    indices = np.arange(len(data))

    cv = GroupShuffleSplit(n_splits=1, test_size=2 * test_size, random_state=random_state)
    train_idx, valid_test_idx = next(cv.split(indices, groups=groups))

    cv.test_size = 0.5
    valid_idx, test_idx = next(cv.split(valid_test_idx, groups=groups[valid_test_idx]))

    cv_data = {
        "train": [data[i] for i in train_idx],
        "valid": [data[i] for i in valid_test_idx[valid_idx]],
        "test":  [data[i] for i in valid_test_idx[test_idx]],
    }
    if visit_codes is not None:
        ########## for each valid and test idx  #############
        for set_key, set_idx in {"valid" : valid_test_idx[valid_idx], "test": valid_test_idx[test_idx]}.items():
            
            # 1. First, extract subsets of data, groups, vscode.
            # -> Work only on this data, so indices are always with respect to subset.
            # 
            # 2. Save indices of patients with only one visit
            #
            # 3. For patients with more than one visit, save all indices of visits
            # 3.1 Find earliest visit in vs_codes array
            # 3.2 Use this index to map it onto the subset index
            # 
            data_filter = []
            # 1.
            sub_groups = [groups[i] for i in set_idx]
            sub_vscodes = [visit_codes[i] for i in set_idx]

            # filter patient ids
            pat_ids, index, counts = np.unique(sub_groups, return_index=True, return_counts=True)

            # 2.
            patients_with_one_visit_indices = index[counts==1]
            for i in patients_with_one_visit_indices:
                data_filter.append(i)

            # 3.
            for patient in pat_ids[counts > 1]:
                patient_visits_indices = [i for i in range(len(sub_groups)) if sub_groups[i] == patient]
                # 3.1
                patient_vscodes = [sub_vscodes[i] for i in patient_visits_indices]
                earliest_vscode = 'bl' if 'bl' in patient_vscodes else sorted(patient_vscodes)[0]
                # 3.2
                local_idx = patient_vscodes.index(earliest_vscode)
                idx = patient_visits_indices[local_idx]
            
                data_filter.append(idx)

            ds = [data[i] for i in set_idx]
            cv_data.update({set_key : [ds[i] for i in data_filter]})

    return cv_data

def split_data(
    data: List[Any],
    groups: Optional[np.ndarray] = None,
    visit_codes: Optional[np.ndarray] = None,
    test_size: float = 0.1,
    seed: int = 76253
    ) -> Dict[str, List[Any]]:

    if groups is None:
        data = split_stratified_by_label(data, test_size, seed)
        LOG.debug("Split data into %d sets: %r", len(data), {k: len(v) for k, v in data.items()})
    else: 
        data = split_by_group(data, groups, test_size, seed, visit_codes)
        LOG.debug("Split data into %d sets using %d groups: %r",
            len(data), len(np.unique(groups)), {k: len(v) for k, v in data.items()})

    return data


def add_random_padding(c_vol, prediction=False, dim=64):
    shape = np.array(c_vol.shape)
    if (shape[0] != 114):
        assert np.all(shape <= dim), "size of volume too big: {} > {}".format(dim, shape)
    else:  # left and right hippocampus
        assert (shape[0] < 2*dim or \
            shape[1] < dim or \
            shape[2] < dim), "size of volume too big: {} > {}".format(dim, shape)


    pad_before = []
    pad_after = []
    borders = dim - shape

    if (borders[0] < 0):  # cleanup if concatenated input
        borders[0] += 64

    for x in borders:
        if prediction:
            idx = int(x/2)
        else:
            idx = np.random.randint(x)
        pad_before.append(idx)
        pad_after.append(x - idx)

    pad_width = list(zip(pad_before, pad_after))
    pad_vol = np.pad(c_vol, pad_width, mode='constant')
    
    return pad_vol


def read_adnimerge_csv(filename: Path, column_image: str) -> pd.DataFrame:
    return pd.read_csv(filename, dtype={
        "ABETA": str,
        "ABETA_bl": str,
        "TAU": str,
        "TAU_bl": str,
        "PTAU": str,
        "PTAU_bl": str,
        column_image: str,
    }, low_memory=False)



def format_adnimerge_table(data: pd.DataFrame) -> pd.DataFrame:
    assert data.columns.is_unique, "data contains duplicate columns"
    assert not data.duplicated().any(), "data contains duplicate rows"

    df = data.copy(deep=True)

    # AGE
    df = df.assign(real_age=lambda x: x["AGE"] + x["Years_bl"])

    # Make Gender numeric
    df.loc[:, "PTGENDER"] = df.loc[:, "PTGENDER"].fillna("Unknown")

    # Ethnicity
    df.loc[:, "PTETHCAT"] = df.loc[:, "PTETHCAT"].fillna("Unknown")

    # Race
    df.loc[:, 'PTRACCAT'] = df.loc[:, 'PTRACCAT'].replace(
        {"Am Indian/Alaskan": "Unknown",
         "Hawaiian/Other PI": "Unknown"})

    # APOE
    miss_val = df.loc[:, "APOE4"].max() + 1
    df.loc[:, "APOE4"] = df.loc[:, "APOE4"].fillna(miss_val)

    # A/T/N status
    df.loc[:, "ATN_status"] = df.loc[:, "ATN_status"].fillna("Unknown")
    
    # CSF biomarkers
    csf_out_of_range = {
        "ABETA": {">1700": 1701, "<200": 199},
        "ABETA_bl": {">1700": 1701, "<200": 199},
        "TAU": {"<80": 79, ">1300": 1301},
        "TAU_bl": {"<80": 79, ">1300": 1301},
        "PTAU": {"<8": 7, ">120": 121},
        "PTAU_bl": {"<8": 7, ">120": 121},
    }
    df = df.replace(csf_out_of_range)
    # apply mapping to columns that actually exist
    for col in filter(lambda x: x in df.columns, csf_out_of_range.keys()):
        df.loc[:, col] = df.loc[:, col].astype(np.float32)
        assert df.loc[:, col].min() > 0, "min in {} is zero".format(col)

    with_missing = ["ABETA", "TAU", "PTAU", "FDG", "AV45"]
    missing_mask = df.loc[:, with_missing].isnull().add_suffix("_MISSING").astype(int)
    missing_as_zero = df.loc[:, with_missing].fillna(0.0)

    X = pd.concat((df.drop(with_missing, axis=1), missing_as_zero, missing_mask), axis=1)
    return X


def get_adni_feature_matrix(adnimerge_table: pd.DataFrame) -> pd.DataFrame:
    excluded = adnimerge_table.loc[:, ["real_age", "PTGENDER"]].isnull().sum(1) > 0
    excluded_idx = excluded[excluded].index
    adnimerge_table = adnimerge_table.drop(excluded_idx, axis=0)
    # print("Excluded entries:", len(excluded_idx))

    categorical = [
        "PTEDUCAT",
        "PTGENDER",
        "APOE4",
        "ABETA_MISSING",
        "TAU_MISSING",
        "PTAU_MISSING",
        "FDG_MISSING",
        "AV45_MISSING",
    ]
    continuous = [
        "real_age",
        "ABETA",
        "TAU",
        "PTAU",
        "FDG",
        "AV45",
    ]

    X_cat = adnimerge_table.loc[:, categorical]
    X_num = adnimerge_table.loc[:, continuous]
    X = pd.concat((X_cat, X_num), axis=1)

    Xt = patsy.dmatrix(
        "cr(real_age, df=4) + C(PTEDUCAT, Poly) + C(PTGENDER) + APOE4 "
        "+ ABETA + TAU + PTAU + FDG + AV45 "
        "+ C(ABETA_MISSING) + C(TAU_MISSING) + C(PTAU_MISSING) + C(FDG_MISSING) + C(AV45_MISSING)",
        data=X,
        return_type="dataframe",
        NA_action="raise").drop("Intercept", axis=1)

    LOG.debug("{} features after transformation:".format(Xt.shape[1]))
    for i, col in enumerate(Xt.columns):
       LOG.debug(f"{i:>3}\t{col}")
    return Xt

def load_heterogeneous_data(image_data_path: str,
                    non_image_data_path: str,
                    hippocampus: str = "Left-Hippocampus",
                    masking: str = "vol_without_bg",
                    debug: bool = False,
                    visit_codes=False,
                    label_coding: Dict[str, int]={"CN": 0, "MCI": 1, "Dementia": 2},
                    filter_unreliable_data=False):

    # load image data
    data = []
    patients = []
    img_ids = []
    labels = []
    if visit_codes:
        viscodes = []
    with h5py.File(image_data_path, "r") as hf:

        debug_counter = 0
        for image_uid, g in hf.items():

            label = g.attrs["DX"]
            if label not in label_coding:
                continue

            if filter_unreliable_data and g.attrs["RID"] in invalid_patIDs:
                continue

            debug_counter += 1
            # if debugging just load one batch
            if (debug and debug_counter > 256):
                break

            patients.append(g.attrs["RID"])
            if visit_codes:
                viscodes.append(g.attrs["VISCODE"])
            img_ids.append(image_uid)
            label_code = label_coding[label]

            
            # select volume
            vols = g[hippocampus]
            # select masking
            if (masking == "concat"):

                # create new channel
                left_hippocampus = np.concatenate(
                                    [np.expand_dims(vols["vol_with_bg"][:], axis=0),
                                    np.expand_dims(vols["mask"][:], axis=0)],
                                    axis=0)
            else:
                left_hippocampus = vols[masking][:]

            data.append(left_hippocampus)
            labels.append(label_code)


    # read non_imaging data
    non_image_table = read_adnimerge_csv(Path(non_image_data_path), column_image='IMAGEUID')

    # filter table such that only visits remain where all data is available
    img_ids_df = pd.DataFrame({'IMAGEUID': img_ids})
    non_image_filter = non_image_table.loc[:, 'IMAGEUID'].isin(
        img_ids_df.loc[:, 'IMAGEUID'])
    non_image_df = non_image_table.loc[non_image_filter, :].set_index('IMAGEUID')
    assert non_image_df.index.is_unique

    # format non_image data
    non_image_df = format_adnimerge_table(non_image_df)

    # select features and transform to numeric values
    non_image_features = get_adni_feature_matrix(non_image_df)

    # append non image data to final output
    counter = Counter()
    output_data = []
    output_patients = []
    if visit_codes:
        output_vscodes = []
    for i in range(len(data)):

        if img_ids[i] in non_image_features.index:  # only use visits for which non-imaging data is available
            
            row = non_image_features.loc[img_ids[i], :].astype(np.float32)
            output_data.append((data[i], row, labels[i]))
            output_patients.append(patients[i])
            if visit_codes:
                output_vscodes.append(viscodes[i])
            counter[labels[i]] += 1


    LOG.debug("Data has %d labels: %r", len(counter), counter)

    if visit_codes:
        return output_data, np.array(output_patients), np.array(output_vscodes)
    else:    
        return output_data, np.array(output_patients)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False, \
        help='True if you want to debug. Subset will be loaded. False otherwise.')
    parser.add_argument('--gpu', type=int, default=0, \
        help='Index of the graphics card you want to use.')
    parser.add_argument('--exp_name', type=str, default='', \
        help='Name of the experiment. A folder will be created with this name')
    parser.add_argument('--binary_classification', type=bool, default=False,
        help='True or False, depending on binary or multiclass classification')
    parser.add_argument('--net', type=str, default='',
        help='Possible Arguments: '+f"{ModelFactory().get_available_models()}")
    parser.add_argument('--weight_decay', type=float, default=0.001,
        help='Enter Weight decay as comma seperated number!')
    parser.add_argument('--max_lr', type=float, default=0.1,
        help='Enter maximum learning rate as comma seperated number!')
    
    return parser.parse_args()

def get_num_classes(binary: bool=True) -> pd.DataFrame:


    data, patients = load_adni_volumes(
        # PATH RERMOVED FOR PRIVACY, \
        hippocampus="Left-Hippocampus", masking="vol_with_bg", binary_classification=binary)

    split_seeds = [65602, 49303, 5779, 19079, 41305, 10441, 44240]

    df = pd.DataFrame(columns=['seed', 'dataset', 'class'])
    for seed in split_seeds:

        splits = split_data(data, groups=patients, seed=seed)

        for ds in splits.keys():

            for item in splits[ds]:

                df = df.append({'seed': seed, 'dataset': ds, 'class': item[1]}, ignore_index=True)

    return df

def get_num_hetero_classes() -> pd.DataFrame:


    data, patients, vscodes = load_heterogeneous_data(
        # PATH RERMOVED FOR PRIVACY, \
        # PATH RERMOVED FOR PRIVACY, \
        hippocampus="Left-Hippocampus", masking="vol_with_bg", debug=False, )

    split_seeds = [65602, 49303, 5779, 19079, 41305, 10441, 44240]

    df = pd.DataFrame(columns=['seed', 'dataset', 'class'])
    for seed in split_seeds:

        splits = split_data(data, groups=patients, seed=seed)

        for ds in splits.keys():

            for item in splits[ds]:

                df = df.append({'seed': seed, 'dataset': ds, 'class': item[2]}, ignore_index=True)

    return df


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          file_path='confusion_matrix.png',
                          scores={}):
# Source: https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
# adapted such that plot is saved

    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])

    stats_text = "\n"
    for key, value in scores.items():
        stats_text = stats_text + f'\n{key}: ' + round_val(value)
    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    # if sum_stats:
    #     #Accuracy is sum of diagonal divided by total observations
    #     accuracy  = (100.0 * np.trace(cf)) / float(np.sum(cf))

    #     #if it is a binary confusion matrix, show some more stats
    #     if len(cf)==2:
    #         #Metrics for Binary Confusion Matrices
    #         precision = (100.0 * cf[1,1]) / sum(cf[:,1])
    #         recall    = (100.0 * cf[1,1]) / sum(cf[1,:])
    #         f1_score  = 2*precision*recall / (precision + recall)
    #         stats_text = "\n\nAccuracy={:0.2f}\nPrecision={:0.2f}\nRecall={:0.2f}\nF1 Score={:0.2f}".format(
    #             accuracy,precision,recall,f1_score)
    #     else:
    #         stats_text = "\n\nAccuracy={:0.2f}".format(accuracy)
    # else:
    #     stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(file_path, dpi=300)

    #plt.show()

def get_seeds_to_net_dict(
    directory: str,
    model_args: Dict[Any, Any],
    seeds: int=[3707, 11730, 22229, 42732, 57446]) -> Dict[int, Path]:
    """
        model_args must contain unique values to the keys, when formatted so str.
        A bad example is 'x': 0.0, when there's also the option x=0.01, because
        'x0.0' in 'testStringx0.01' is True, although testStringx0.0 is sought after
    """
    directories = {}
    for model_dir in Path(directory).iterdir():
        
        if model_dir.is_dir():
            dir_str = str(model_dir)
            seed = dir_str[dir_str.index('Seed_')+5:]
            seed = int(seed[:seed.index('_')])

            booleans = [(k + str(v)) in dir_str for k, v in model_args.items()]
            if all(booleans):

                assert seed not in directories.keys(), 'Not enough attributs in model_args.' \
                + f'Found multiple options on seed {seed}: \n{str(directories[seed])}\n{dir_str}'

                directories[seed] = model_dir / 'checkpoints' / 'Best.pt'

    return directories

def save_concat_layer_weights(model_path, mask, ndim_non_img=31):

    in_channels = 2 if mask == 'concat' else 1
    model = models.ModelFactory().create_model(model_type='ConcatHNN', \
        args={'in_channels': in_channels, 'ndim_non_img': ndim_non_img})
    model.load_state_dict(torch.load(model_path))

    non_img_start_id = model.fc.in_features - ndim_non_img
    weights = model.fc.weight[:, non_img_start_id:]
    data = weights.detach().cpu().numpy()

    save_path = Path(model_path).parents[1]
    save_path = str(save_path) + 'coefficients_fcn.csv'
    np.savetxt(save_path, data.T, delimiter=';')

def get_epoch_latest_model(model_dir: Path):

    epochs = []
    for model in model_dir.iterdir():
        if 'epoch' in str(model):
            index = str(model).index('epoch_')+6
            epochs.append(str(model)[index:index+2])
    return sorted([int(x) for x in epochs])[-1]

def get_num_classes_to_csv():
    
    data, patients, vscodes =  load_heterogeneous_data(
        # PATH RERMOVED FOR PRIVACY, \
        # PATH RERMOVED FOR PRIVACY, \
        hippocampus="Left-Hippocampus", masking="vol_with_bg", debug=False, visit_codes=True)
   
    split_seeds = [3707, 11730, 22229, 42732, 57446]
    df_items = []
    for seed in split_seeds:
        split = split_data(data, patients, visit_codes=vscodes, seed=seed)

        for key, datas in split.items():
            uniq = np.unique(np.array(datas)[:,2], return_counts=True)

            for i in range(len(uniq[0])):
                df_items.append([seed, key, uniq[0][i], uniq[1][i], uniq[1][i] / len(datas)])

    df = pd.DataFrame(df_items, columns=['seed', 'dataset', 'label', 'count', 'relative_amount'])
    save_str = 'class_eval_splits.csv'
    df.to_csv(save_str)

def get_num_trainable_parameters(model: torch.nn.Module):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Trapezoid(optim.lr_scheduler._LRScheduler):

    def __init__(self,
                optimizer,
                n_iterations: int,
                max_lr: float,
                start_lr: Optional[float]=None,
                annihilate: bool=True,
                last_epoch: int=-1
                ):

        # if cyclic momentum would be implemented, according to Superconvergence paper
        # https://arxiv.org/abs/1708.07120
        # 0.85 as min val works just fine. Take that value!

        self.n_iters = n_iterations
        self.max_lr = max_lr
        if start_lr is None:
            self.start_lr = max_lr / 20
        else:
            self.start_lr = start_lr
        self.stop_warmup = int(0.1 * n_iterations)
        self.start_decline = int(0.8 * n_iterations)
        self.start_annihilate = int(0.95 * n_iterations) if annihilate else n_iterations

        super(Trapezoid, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch < self.stop_warmup:
            step_size = (self.max_lr - self.start_lr) / self.stop_warmup
            new_lr = self.start_lr + step_size * self.last_epoch
        elif self.last_epoch < self.start_decline:
            new_lr = self.max_lr
        elif self.last_epoch <= self.start_annihilate:
            step_size = (self.max_lr - self.start_lr) / (self.start_annihilate - self.start_decline)
            new_lr = self.max_lr - step_size * (self.last_epoch - self.start_decline)
        else:
            step_size = (self.start_lr - self.start_lr / 20) / (self.n_iters - self.start_annihilate)
            new_lr = self.start_lr - step_size * (self.last_epoch - self.start_annihilate)
            
        return [new_lr for group in self.optimizer.param_groups]


def get_lr_change(total_iterations, optimizer, lr, betas, weight_decay, restarts=20, momentum=0.9, epoch_annihilation=50):
    # used to evaluate behaviour visually
    model = models.ResNet()
    if optimizer == 'Adam':
        _optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        _scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(_optimizer, 100)
    elif optimizer == 'AdamW':
        _optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        _scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(_optimizer, 100)
    elif optimizer == 'SGD':
        _optimizer =  optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        _scheduler = optim.lr_scheduler.OneCycleLR(_optimizer, max_lr=lr, total_steps=total_iterations, anneal_strategy='linear', cycle_momentum=True)
    elif optimizer == 'Trapezoid':
        _optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        _scheduler = Trapezoid(_optimizer, n_iterations=total_iterations, max_lr=lr, annihilate=False)

    else:
        raise ValueError(optimizer)

    values = []
    for i in range(total_iterations):
        cur_lr = -1
        for pg in _optimizer.param_groups:
            cur_lr = pg['lr']
        values.append((cur_lr, i))
        _scheduler.step()

    return values



