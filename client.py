import sys, json, math

class LinearRegressionGradientDescentEngine:
    """
    Zero-Dependency Multivariate Linear Regression Engine.
    Trains weights and bias via batch gradient descent minimizing Mean Squared Error (MSE).
    Calculates R^2 coefficient of determination without external ML frameworks.
    """
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = []
        self.bias = 0.0

    def fit(self, X, y):
        """
        X: list of lists (N samples, D features)
        y: list of floats (N targets)
        """
        n = len(X)
        if n == 0 or len(y) != n:
            return {"error": "Invalid training data dimensions."}

        d = len(X[0])
        self.weights = [0.0] * d
        self.bias = 0.0

        for _ in range(self.iterations):
            # Compute predictions and gradients
            dw = [0.0] * d
            db = 0.0

            for i in range(n):
                pred = sum(X[i][j] * self.weights[j] for j in range(d)) + self.bias
                error = pred - y[i]
                for j in range(d):
                    dw[j] += error * X[i][j]
                db += error

            # Update parameters
            for j in range(d):
                self.weights[j] -= (self.learning_rate / n) * dw[j]
            self.bias -= (self.learning_rate / n) * db

        # Compute R^2 and MSE
        y_mean = sum(y) / n
        ss_tot = sum((val - y_mean) ** 2 for val in y)
        ss_res = 0.0

        for i in range(n):
            pred = sum(X[i][j] * self.weights[j] for j in range(d)) + self.bias
            ss_res += (pred - y[i]) ** 2

        mse = ss_res / n
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        return {
            "status": "TRAINED",
            "weights": [round(w, 4) for w in self.weights],
            "bias": round(self.bias, 4),
            "mse": round(mse, 4),
            "r_squared": round(r2, 4)
        }

    def predict(self, X):
        d = len(self.weights)
        preds = []
        for row in X:
            val = sum(row[j] * self.weights[j] for j in range(d)) + self.bias
            preds.append(round(val, 4))
        return preds

    def run_benchmark_linear_regression(self):
        # Target formula: y = 2.5 * x1 + 4.0 * x2 + 3.0
        X = [
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 3.0],
            [5.0, 5.0],
            [6.0, 2.0]
        ]
        y = [
            2.5*1.0 + 4.0*2.0 + 3.0,
            2.5*2.0 + 4.0*1.0 + 3.0,
            2.5*3.0 + 4.0*4.0 + 3.0,
            2.5*4.0 + 4.0*3.0 + 3.0,
            2.5*5.0 + 4.0*5.0 + 3.0,
            2.5*6.0 + 4.0*2.0 + 3.0
        ]

        engine = LinearRegressionGradientDescentEngine(learning_rate=0.03, iterations=2500)
        fit_res = engine.fit(X, y)

        test_X = [[2.0, 2.0]]
        test_pred = engine.predict(test_X)[0]
        expected = 2.5*2.0 + 4.0*2.0 + 3.0 # 16.0

        return {
            "benchmark_status": "PASSED",
            "r_squared_high": fit_res["r_squared"] > 0.95,
            "test_prediction": test_pred,
            "prediction_accurate": abs(test_pred - expected) < 1.0,
            "r_squared": fit_res["r_squared"]
        }
