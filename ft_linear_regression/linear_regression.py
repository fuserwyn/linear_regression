from data_loader import DataLoader
import logging


logger = logging.getLogger(__name__)


class LinearRegression:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.value = data_loader.original_values
        self.learning_rate = None
        self.theta_0 = 0.0
        self.theta_1 = 0.0
        self.tmp_theta_0 = 1.0
        self.tmp_theta_1 = 1.0
        self.prev_mse = 0.0
        self.cur_mse = self.mean_square_error()
        self.delta_mse = self.cur_mse
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None

    def estimate_price(self, mileage):
        return self.tmp_theta_0 + (self.tmp_theta_1 * float(mileage))

    def mean_square_error(self):
        total_squared_error = 0
        count = 0

        for i, line in enumerate(self.value):
            if i > 0:  # Skip header
                predicted_price = self.estimate_price(line[0])
                actual_price = float(line[1])
                total_squared_error += (predicted_price - actual_price) ** 2
                count += 1

        return total_squared_error / count if count > 0 else float('inf')

    def get_gradient0(self):
        total_gradient = 0.0
        count = 0
        for i, line in enumerate(self.value):
            if i > 0:
                total_gradient += (self.estimate_price(line[0]) - float(line[1]))
                count += 1
        return (total_gradient / count) if count > 0 else 0

    def get_gradient1(self):
        total_gradient = 0.0
        count = 0
        for i, line in enumerate(self.value):
            if i > 0:
                total_gradient += (self.estimate_price(line[0]) - float(line[1])) * float(line[0])
                count += 1
        return (total_gradient / count) if count > 0 else 0

    def set_min_max(self):
        self.min_x = float('inf')
        self.max_x = float('-inf')
        self.min_y = float('inf')
        self.max_y = float('-inf')
        for i, line in enumerate(self.value):
            if i > 0:
                mileage = float(line[0])
                price = float(line[1])
                self.min_x = min(self.min_x, mileage)
                self.max_x = max(self.max_x, mileage)
                self.min_y = min(self.min_y, price)
                self.max_y = max(self.max_y, price)

    def standardize(self):
        self.set_min_max()
        for i, line in enumerate(self.value):
            if i > 0:
                line[0] = (float(line[0]) - self.min_x) / (self.max_x - self.min_x)
                line[1] = (float(line[1]) - self.min_y) / (self.max_y - self.min_y)
        self.data_loader.standardized_values = self.value.copy()

    def train_model(self, learning_rate, print_error):
        self.learning_rate = learning_rate
        self.standardize()
        while abs(self.delta_mse) > 0.0000001:
            self.theta_0 = self.tmp_theta_0
            self.theta_1 = self.tmp_theta_1
            self.tmp_theta_0 -= self.get_gradient0() * self.learning_rate
            self.tmp_theta_1 -= self.get_gradient1() * self.learning_rate
            self.prev_mse = self.cur_mse
            self.cur_mse = self.mean_square_error()
            self.delta_mse = self.cur_mse - self.prev_mse
            if print_error:
                logger.info(f"Current MSE: {self.cur_mse}")
        self.theta_1 = (self.max_y - self.min_y) * self.tmp_theta_1 / (self.max_x - self.min_x)
        self.theta_0 = self.min_y + ((self.max_y - self.min_y) * self.tmp_theta_0) + self.theta_1 * (1 - self.min_x)

    def r_squared(self):  # bonus part
        """ Calculate R² (coefficient of determination) to measure the precision of the regression model. """
        ss_total = 0
        ss_residual = 0
        mean_y = sum(float(line[1]) for line in self.value[1:]) / (len(self.value) - 1)

        for i, line in enumerate(self.value):
            if i > 0:
                actual_price = float(line[1])
                predicted_price = self.estimate_price(line[0])
                ss_total += (actual_price - mean_y) ** 2
                ss_residual += (actual_price - predicted_price) ** 2

        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else float('inf')
        return r_squared