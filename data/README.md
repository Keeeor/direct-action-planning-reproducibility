# Data inputs and preprocessing

Raw and processed arrays are intentionally omitted from this anonymous package. Download the inputs under their applicable terms, place them in the paths below, and run the preprocessing scripts from the repository root.

## Azure Functions Trace 2019

- Source: <https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md>
- Download: <https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz>
- License: CC BY 4.0
- Expected directory: `data/raw/azure_functions_2019/`
- Expected files: `invocations_per_function_md.anon.d01.csv` through `invocations_per_function_md.anon.d14.csv`
- Preprocessing: `scripts/preprocess_azure_trace.py` and `scripts/preprocess_azure_bursty.py`

## GenTD26

- Upstream repository: <https://github.com/alibaba/clusterdata>
- The pinned upstream revision and expected raw filenames are recorded in `data/metadata/source_manifest.json`.
- Raw files are not redistributed by this package. Obtain them from the upstream source subject to its terms.
- Expected directory: `data/raw/gentd26/`
- Preprocessing: `scripts/preprocess_gentd26.py`

The preprocessing scripts fit scaling and chronological splits on the training partition only. They write generated arrays and metadata to ignored paths under `data/processed/` and `data/metadata/`.
