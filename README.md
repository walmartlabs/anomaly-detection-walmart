# Anomaly Detection for Walmart's Online Pricing System

Code for the paper "On Anomaly Detection for Walmart's Online Pricing System."

## Packages

In order to run models, these python packages are needed:

- pyod
- scikit-learn
- xgboost

## Example Run

We can run the various models by providing the model name: gaussiannb, iforest, autoencoder, xgboost, rf, e.g.,

```bash
ipython train_evaluate_model.py -- --model autoencoder --input-filename input_data.pkl --output-filename autoencoder_output.pkl
```

## Contact

If you have questions, feel free to send an email to jramakrishnan@walmartlabs.com.
