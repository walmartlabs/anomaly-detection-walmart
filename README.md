# Anomaly Detection for Walmart's Online Pricing System

Code for the paper "On Anomaly Detection for Walmart's Online Pricing System," authored by Jagdish Ramakrishnan, Elham Shaabani, Chao Li, and Mátyás A. Sustik.

## Packages

This code works on Python 2.7. In order to run models, these python packages are needed:

- pyod
- scikit-learn
- xgboost

## Example Run

We can run the various models by providing the model name: gaussiannb, iforest, autoencoder, xgboost, rf, e.g.,

```bash
ipython train_evaluate_model.py -- --model autoencoder --input-filename input_data.pkl --output-filename autoencoder_output.pkl
```

However, the input files are not currently part of the repository, and thus, the above command will not run successfully. We plan to include some sample data files in the future.

## Contact

If you have questions, feel free to send an email to jramakrishnan@walmartlabs.com.
