To run the codes, please follow the steps:
1. run the `reference_baseline.ipynb` notebook to train the original models and obtain the original result on the test set.
2. run the `original_bias.ipynb` notebook to see the bias of the original models.
3. run the `expand_data.ipynb` notebook to get the expanded data.
4. run the `retrain_model.ipynb` notebook to train the models on the expanded data.
5. run the `new_bias.ipynb` notebook to see the bias of the new models.

The `predict_demo.ipynb` notebook is to demonstrate loading the model and making predictions.

After running the files above, the prediction results (`.csv`) files and the model files (`.joblib`) will appear in the current folder. For the convenience of demonstration, the prediction results are in the `predictions` folder, and the original/new models are in the `models` folder.

- Prediction model file with bias exists: `models`
- Bias test before mitigation, including test methods and test results: `original_bias.ipynb`
- The model file with bias mitigated: `models`
- Bias test after mitigation, including test methods and test results: `new_bias.ipynb`

Checkout the `report.pdf` for more results.