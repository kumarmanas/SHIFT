##Heteroscedastic Spectral Normalized Gaussian Process based Trajectory Predictor

## Overview

This project deals with trajectory prediction using both aleatoric and epistemic uncertainty for better uncertainty calibration. We train both prior and posterior models with uncertainty quantification, using traffic rule (for prior model) and observed data (for posterior) for inference. The model uses Heteroscedastity for aleatoric uncertainty quantification and Spectral Normalized Gaussian Processes (SNGP) for epistemic uncertainty calculation. Finally, we use Monte Carlo sampling for posterior inference.

This code is for review only. After paper acceptance and patent processing, the code and paper are published as open source. 

## Files and Configuration

- **`posterior_HETSNGP.json` or `prior_HETSNGP.json`**: These files control the amount of data used during training, validation, and testing.
    - `train/NuscenesDataset.limit`, `val/NuscenesDataset.limit`, and `test/NuscenesDataset.limit` specify the fraction of the data to be used.
        - Example values: 
            - `train/NuscenesDataset.limit = 0.1`: Use 10% of the training data.
            - `train/NuscenesDataset.limit = 0.5`: Use 50% of the training data.
            - To use the full dataset, remove or comment out the `limit` line.

- **In-Distribution and Out-of-distribution Testing**: For specific location-based testing, use `train/NuscenesDataset.limit = ["location"]`. For example, to train on Boston-seaport and test on Singapore, use as: `train/NuscenesDataset.limit: ["boston-seaport"]`, `val/NuscenesDataset.limit: ["boston-seaport"]` and `test/NuscenesDataset.limit: "['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown']"` . 

## Automated Traffic Rules as Prior Functions
We provide both open and closed model option to the end user to create prior from traffic rule as natural language instruction. We used GPT-4 and Llama 3.3 in our work, however user are free to choose different variant of  GPT based model or other Llama variants from ollama [https://ollama.com/library] website or from hugging face. Also, other possible options can be Mistral-7B with fast inference time. 
For GPT-based models, go to the notebook and change the model name as desired.  Run notebook `Gpt4_based_Prior_Creation.ipynb` if you want to use a paid model for creating a prior model.  UI interface option is only supported by us for open source model for time-being.

If you use a Llama-based model, you have the option to make a UI interface. Just use the file name `Llama3_based_RAG_for_rule_prior.ipynb` to run the file, and you can open the UI interface at the end of the notebook run.

## Model Setup

### Prior Model Creation

1. **Edit Docker Entrypoint**:
    - Modify the entrypoint file in the Docker container to point to `prior_HETSNGP.json`. This switches the training from the default posterior model (which uses `posterior_HETSNGP.json`) to the prior model.
    
2. **Running Docker**:
    - After editing the entrypoint to use `prior_HETSNGP.json`, run your Docker container. Be sure to mount:
      - **Dataset path**: Nuscenes dataset.
      - **Results path**: CSV file for results.
      - **Model weights storage path**: Directory to store the model weights.

3. **Evaluate Prior Model**:
    - After running the prior model, check the results CSV to identify which model weight and epoch produced the best prior.

### Posterior Model Creation

1. **Update Posterior Configuration**:
    - After determining the best prior model (based on log file evaluation), update the [`docker/posterior_HETSNGP.json`](docker/posterior_HETSNGP.json) file with:
      - Set `prior_model` parameter to point to your best prior model weights
      - Configure `gp_posterior_temp` to balance rule-based vs observed data
      - Adjust `gp_mc_sample` for Monte Carlo sampling accuracy
2. **Dataset Processing Optimization**:
    - Set `"populate": true` in [`docker/posterior.json`](docker/posterior.json) to cache the Nuscenes dataset
    - Uses [`populate_cache()`](shift/main.py) function for faster data processing
    - Supports location-specific training with `NuscenesDataset.limit` parameter
    - Available locations: `boston-seaport`, `singapore-onenorth`, `singapore-hollandvillage`, `singapore-queenstown`
3. **Training Configuration**:
    - Use the [`Trainer`](shift/tf/model/training.py) class for model training
    - Models are saved in `exp_dir/models/current_time/name` directory
    - Logs are stored in `exp_dir/logs/current_time/name`
    - Enable checkpointing for model weights and optimizer state

4. **Saving Predictions**:
    - To save model predictions during the posterior model creation, enable `save_test_preds` in the code to ensure predictions are saved for later evaluation.

### Plotting Predictions

- Set `plot_predictions` to `true` to generate prediction plots.
- Ensure the posterior model is fully trained and predictions are saved before enabling this option.

## Key Parameters for Customization

- **`gp_posterior_temp`**: Controls the importance of rule-based knowledge over observed data in posterior model creation.
- **`extractor_posterior_temp`**: Adjusts the weight of the prior model in the feature extractor of the neural network.
- **`gp_mc_sample`**: Specifies the number of Monte Carlo samples for posterior inference.
- **Model Architecture**:
    - Uses TensorFlow 2.6.0 with spectral normalized layers
    - Implements heteroscedastic SNGP for uncertainty estimation
    - Configurable through `.gin` files in [`shift/configs/`](shift/configs/)

- **Training Parameters**:
    - Control training with command line arguments in [`shift/main.py`](shift/main.py):
      - `-c/--config`: Specify gin config file
      - `-j/--json`: Override gin config with JSON file
      - `-t/--test`: Run local test with reduced runtime
      - `-n/--notrain`: Evaluation only mode
      - `-k/--repetitions`: Number of training repetitions

## Example Usage
We use docker based setup for running our experiment, however you can adapt with just command line interface. Start by building docker image `docker build -t user_image_name .`. After that run your docker with  docker run -v your_dataset_location:/dataset_mounted in_config_file and other mounting of folder if needed.

1. Train a prior model by editing the Docker entrypoint to use `prior_HETSNGP.json`.
2. Once training is complete, evaluate the model by checking the results file for the best weights and epoch.
3. For final trajectory prediction, update `docker/posterior_HETSNGP.json` with the best prior model weights, set `"populate": true` for faster data processing, and fine-tune the model using the posterior parameters.

Refer to the JSON and `.gin` files for additional hyperparameter tuning to control model behavior and performance.

## Notes
- Refer to our paper Appendix for more detailed working process of **Rule-SNGP** from input traffic scene to prediction along with kernel approximation.
- Refer to the provided config files for more control over the training and inference process.
