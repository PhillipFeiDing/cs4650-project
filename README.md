# cs4650-project
The code repository for our CS4650 course project: Disaster Tweet prediction.

## ML-models
- All models are located under `ml-models/Traditional ML Method.ipynb`.
- For the notebook, the first few cells tried to clean the data and do simple visualization, and then we tried to build BoW and TF-IDF feature input from the cleaned data. Afterwards, you can uncomment/comment the SMOTE section to see the model performance before/after SMOTE algorithm. The rest of the cells are just trying to apply different models to train on the data set. Finally, each model will have its F-1 scores, accuracy and confusion matrix printed out.

## NN-models
- The dataset can be found under `nn-models/data`.
- Our best performing RoBERTa model checkpoint is stored under `nn-models/models`. If you work with Git instead of Google Drive, you will need the [Large File Storage](https://git-lfs.github.com/) service.
- All codes from development is under `nn-models/src`.
- You can run the notebooks in `nn-models` directory. However, it is recommended that they are run under the [Google Colab](https://drive.google.com/drive/folders/1oWWWMtIImiUt7wCukfG1Zoz4IjiSiHHi) environment. You will see all instructions in the Jupyter notebook prompts: hyper-parameter settings, replicating results, and generating visualizations.
- To replicate all model results, first simply run everything before the "Train Model (No Pretrained Embedding)" in the `models.ipynb` file. Then,
  - to replicate vanilla LSTM, run everything forward until "5. Pretrained Embeddings (GloVe)"
  - to replicate all other models, run their corresponding section.