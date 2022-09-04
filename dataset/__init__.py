import importlib


def find_dataset_def(dataset):
    if dataset == "rpc":
        module_name = 'dataset.satmvsdataset'
        module = importlib.import_module(module_name)
    elif dataset == "pinhole":
        module_name = 'dataset.virdataset'
        module = importlib.import_module(module_name)
    else:
        raise Exception("Not implemented yet")
    return getattr(module, "MVSDataset")
