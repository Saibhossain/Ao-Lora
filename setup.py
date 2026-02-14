from setuptools import setup, find_packages

setup(
    name="ao_lora",
    version="0.1.0",
    author="Md Saib Hossain",
    author_email="your.email@example.com",
    description="Activation-Orthogonal Low-Rank Adaptation (AO-LoRA) for Calibrated PEFT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Saibhossain/Ao-Lora.git",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.5.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)