import logging
from args_parser import args_parser
from data_loader import DataLoader
from linear_regression import LinearRegression
from plotter import Plotter

logger = logging.getLogger(__name__)


def model_training(args: tuple) -> LinearRegression:
    data_loader = DataLoader(args[0])
    model = LinearRegression(data_loader)
    model.train_model(learning_rate=0.01, print_error=False)
    if args_parser.bonus:
        r2_score = model.r_squared()
        logger.info("R² Score: %.5f", r2_score)
    return model


def plotting(model: LinearRegression):
    plotter = Plotter(model)
    if args_parser.graphics == "o":
        plotter.plot_value()
    if args_parser.graphics == "s":
        plotter.plot_standardized_values()


def main():
    plotting(model_training(args_parser.parse_args()))
    logger.info("Finish")


if __name__ == "__main__":
    main()