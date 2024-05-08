# RBF-KAN
This code implements a Radial Basis Function (RBF) based Kolmogorov-Arnold Network (KAN) for function approximation. The network consists of several layers, including RBF kernel activation layers and ReLU layers.


![image](https://github.com/sidhu2690/RBF-KAN/assets/136654152/1122c3dc-76fa-4a73-bbcf-a4dc124cac8d)

## Overview

The code generates a synthetic dataset comprising a sinusoidal function with added noise. The dataset is then converted to PyTorch tensors and fed into the network for training. Training is performed using the Adam optimizer and mean squared error (MSE) loss.

After training, the network's performance is evaluated by plotting both the ground truth function and the predicted function on the same graph.

## Potential Improvements and Considerations

- **Hyper-parameter tuning:** Tune parameters such as the number of centers and learning rate using techniques like grid search or Bayesian optimization.
  
- **Initialization strategies:** Explore advanced initialization techniques like k-means clustering or gradient-based initialization for RBF centers and weights.

- **Regularization:** Employ techniques like L1/L2 regularization, dropout, or early stopping to prevent overfitting and improve generalization.

- **Architecture modifications:** Experiment with different layer configurations, types of layers (e.g., convolutional layers, attention mechanisms), and layer connections to better suit the problem domain.

- **Transfer learning:** Utilize pre-trained models or pre-trained RBF centers to expedite convergence and enhance performance.

- **Ensemble methods:** Combine multiple models using ensemble techniques like bagging or boosting to improve predictive performance.

- **Advanced optimization techniques:** Explore advanced optimization algorithms such as adaptive learning rate methods or second-order optimization methods to enhance convergence speed and final performance.

- **Interpretability and explainability:** Employ techniques like SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) to interpret and explain the behavior of the RBF-based KAN.

- **Domain-specific modifications:** Tailor the model architecture and modifications to specific application domains (e.g., time series forecasting, computer vision, natural language processing) to capture underlying patterns and relationships more effectively.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to enhance the functionality, documentation, or examples provided in this repository.

## License

This project is licensed under the [MIT License](LICENSE).
