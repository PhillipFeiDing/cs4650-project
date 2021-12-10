# cs4650-project
The code repository for our CS4650 course project: Disaster Tweet prediction.

## ML-models
- All models are located under `ml-models/Traditional ML Method.ipynb`.

## NN-models
- The dataset can be found under `nn-models/data`.
- Our best performing RoBERTa model checkpoint is stored under `nn-models/models`. If you work with Git instead of Google Drive, you will need the [Large File Storage](https://git-lfs.github.com/) service.
- All codes from development is under `nn-models/src`.
- You can run the notebooks in `nn-models` directory. However, it is recommended that they are run under the [Google Colab](https://drive.google.com/drive/folders/1oWWWMtIImiUt7wCukfG1Zoz4IjiSiHHi) environment. You will see all instructions in the Jupyter notebook prompts: hyper-parameter settings, replicating results, and generating visualizations.
- To replicate all model results, first simply run everything before the "Train Model (No Pretrained Embedding)" in the `models.ipynb` file. Then,
  - to replicate vanilla LSTM, run everything forward until "5. Pretrained Embeddings (GloVe)"
  - to replicate all other models, run their corresponding section.