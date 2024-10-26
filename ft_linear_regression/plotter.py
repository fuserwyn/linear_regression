import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Plotter:
    def __init__(self, model):
        self.model = model

    def plot_value(self):
        tmp_val = self.model.value[1:]
        mileage, price = zip(*[(float(row[0]), float(row[1])) for row in tmp_val])
        original_mileage = [self.model.min_x + (self.model.max_x - self.model.min_x) * m for m in mileage]
        original_price = [self.model.min_y + (self.model.max_y - self.model.min_y) * p for p in price]
        plt.title('Real values')
        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.plot(original_mileage, original_price, 'ro', label='Data Points')
        reg_line_x = [self.model.min_x, self.model.max_x]
        reg_line_y = [self.model.theta_0 + self.model.theta_1 * x for x in reg_line_x]
        logger.info(f"Regression Line Y-values: {reg_line_y}")
        plt.plot(reg_line_x, reg_line_y, label='Estimated Line', color='blue')
        axis_tuple = (
            self.model.min_x - abs(self.model.max_x * 0.1),
            self.model.max_x + abs(self.model.max_x * 0.1),
            min(reg_line_y) - abs(max(reg_line_y) * 0.1),
            max(reg_line_y) + abs(max(reg_line_y) * 0.1)
        )
        plt.axis(axis_tuple)
        plt.legend()
        plt.show()

    def plot_standardized_values(self):
        tmp_val = self.model.data_loader.standardized_values[1:]
        mileage, price = zip(*[(float(row[0]), float(row[1])) for row in tmp_val])
        plt.title('Standardized values')
        plt.xlabel('Standardized Mileage')
        plt.ylabel('Standardized Price')
        plt.plot(mileage, price, 'go', label='Standardized Data Points')
        reg_line_x = [0, 1]  # Standardized range is [0, 1]
        reg_line_y = [self.model.tmp_theta_0 + self.model.tmp_theta_1 * x for x in reg_line_x]
        plt.plot(reg_line_x, reg_line_y, label='Estimated Line (Standardized)', color='orange')
        plt.legend()
        plt.show()
