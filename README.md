# ft_linear_regression

A simple linear regression project in Python designed to predict car prices based on mileage. This project uses gradient descent to optimize the linear regression model and includes options for data standardization and R² score calculation for model evaluation.

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example Commands](#example-commands)
- [Bonus Features](#bonus-features)

## Project Description

The purpose of `ft_linear_regression` is to create a basic linear regression model that predicts car prices based on mileage. This project supports both original and standardized data and visualizes the data points and regression line, allowing easy inspection of the model’s accuracy. The bonus R² score calculation feature provides a quantitative evaluation of the model's performance.

## Features

- **Data Standardization**: Optional standardization of the data for model training.
- **Visualization**: Generates scatter plots with regression lines.
- **Error Calculation**: Displays mean squared error at each training step.
- **Bonus Feature - R² Score**: Calculates the R² score to measure model accuracy.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/fuserwyn/ft_linear_regression.git
    cd ft_linear_regression
    ```

2. **Set up virtual environment and install dependencies**:
    ```bash
    bash install_venv_requirements.sh
    ```

## Usage

The program requires a CSV file (`data.csv`) with two columns: mileage and price (without headers). 

### Arguments
- **filename**: Path to the CSV file with the dataset (required).
- **--o**: Plot original data values.
- **--s**: Plot standardized data values.
- **--bonus**: Calculate and display the R² score as a bonus.

## Example Commands

- **Plot standardized data values**:
    ```bash
    python3 main.py data.csv --s
    ```

- **Plot standardized values with R² score**:
    ```bash
    python3 main.py data.csv --s --bonus
    ```

- **Plot original data values**:
    ```bash
    python3 main.py data.csv --o
    ```

- **Plot original values with R² score**:
    ```bash
    python3 main.py data.csv --o --bonus
    ```

## Bonus Features

### R² Score
Adding the `--bonus` flag calculates the R² score, providing a measure of the regression model's fit to the data. This metric ranges from 0 to 1, with a value closer to 1 indicating a more accurate model.

