from setuptools import setup, find_packages

setup(
    name="ml_pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=1.10",
        "pyyaml",
        "torch",
        "transformers",
        "tqdm",
        "scikit-learn",
        "numpy"
    ],
    entry_points={
        'console_scripts': [
            'ml-pipeline=ml_pipeline.src:main',
        ],
    },
    author="Your Name",
    description="End-to-end LLM fine-tuning pipeline",
    include_package_data=True,
) 