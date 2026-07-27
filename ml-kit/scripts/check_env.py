import importlib
import platform

MODULES = [
    "numpy",
    "pandas",
    "polars",
    "sklearn",
    "matplotlib",
    "seaborn",
    "lightgbm",
    "xgboost",
    "catboost",
    "torch",
    "torchvision",
    "transformers",
    "datasets",
    "accelerate",
    "sentence_transformers",
    "cv2",
    "PIL",
    "tqdm",
    "optuna",
    "pyarrow",
    "scipy",
]


def main() -> None:
    print(platform.python_version(), platform.platform())
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            print(f"{name}: OK {getattr(mod, '__version__', 'unknown')}")
        except Exception as exc:
            print(f"{name}: FAIL {exc}")


if __name__ == "__main__":
    main()

