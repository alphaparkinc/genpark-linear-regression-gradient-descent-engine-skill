import sys, json
from client import LinearRegressionGradientDescentEngine

def main():
    engine = LinearRegressionGradientDescentEngine()
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try:
            req = json.loads(line)
            method = req.get("method")
            rid = req.get("id")
            params = req.get("params", {})

            if method == "tools/list":
                res = {
                    "tools": [
                        {"name": "fit", "description": "Fit linear regression.", "inputSchema": {"type": "object", "properties": {"X": {"type": "array"}, "y": {"type": "array"}}, "required": ["X", "y"]}},
                        {"name": "predict", "description": "Predict y for X.", "inputSchema": {"type": "object", "properties": {"X": {"type": "array"}}, "required": ["X"]}},
                        {"name": "run_benchmark_linear_regression", "description": "Run self-test.", "inputSchema": {"type": "object"}}
                    ]
                }
            elif method == "tools/call":
                tname = params.get("name")
                args = params.get("arguments", {})
                if tname == "fit":
                    out = engine.fit(args.get("X", []), args.get("y", []))
                elif tname == "predict":
                    out = engine.predict(args.get("X", []))
                elif tname == "run_benchmark_linear_regression":
                    out = engine.run_benchmark_linear_regression()
                else:
                    out = {"error": f"Unknown tool {tname}"}
                res = {"content": [{"type": "text", "text": json.dumps(out)}]}
            else:
                res = {"error": "Unsupported method"}
            print(json.dumps({"jsonrpc": "2.0", "id": rid, "result": res}), flush=True)
        except Exception as e:
            print(json.dumps({"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}}), flush=True)

if __name__ == "__main__":
    main()
