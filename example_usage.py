from client import LinearRegressionGradientDescentEngine
import json

def main():
    engine = LinearRegressionGradientDescentEngine()
    res = engine.run_benchmark_linear_regression()
    print("Linear Regression Benchmark Result:")
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
