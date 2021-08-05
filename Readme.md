# Neuronal Network Study

## Todo

- wandb

  ```python
  # Flexible integration for any Python script
  import wandb
  # 1. Start a W&B run
  wandb.init(project='Neuronal Network Study', entity='salversij')
  # 2. Save model inputs and hyperparameters
  config = wandb.config
  config.learning_rate = 0.01
  # Model training here
  # 3. Log metrics over time to visualize performance
  wandb.log({"loss": loss})
  ```
