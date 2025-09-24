import torch
import numpy as np



data_dict = torch.load("simple_data.pt")
data_dict["data"] = (data_dict["data"] - data_dict["data_mean"]) / data_dict["data_std"]
print("data_dict keys:", data_dict.keys())
# keep data with 'label' == '12' or 'label' == '13'
# I have a list of labels and i want to store the indices of the labels that are either 12 or 13
indices = torch.where((data_dict["tasks"] == 12) | (data_dict["tasks"] == 13))[0]

data_dict["data"] = data_dict["data"][indices]
data_dict["tasks"] = data_dict["tasks"][indices]