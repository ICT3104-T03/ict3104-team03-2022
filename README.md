# ICT3104-team03-2022

Toyota Smart Home Project: Using Machine Learning to View Activities in a Home

## Install Requirements
### 1. Python.10
### 2. Pip 22.2.2
### 3. Anaconda3 2021.05

## Set up
### 1. Search Anaconda Prompt in Windows Search

### 2. Open the Anaconda Terminal, cd into directory and type in the commands below
```
cd C:\Users\~DirectoryLocation~\Desktop\CurrentProjects\ict3104-team03-2022
```

### 3. Create a new and clean virtual environment (May need to pip install virtualenv)

```
virtualenv new_env 
```

### 4. Step into the virtual environment

```
.\new_env\Scripts\activate
```

### 5. Create a new conda environment, press y to continue installation
```
conda create -n life python=3.8
```

### 6. Install pytorch library
Check if your laptop is Cuda enabled. [Nvidia](https://developer.nvidia.com/cuda-gpus#compute)
```
@@ CUDA 10.2 @@
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
@@ CUDA 11.3 @@
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch
@@ CUDA 11.6 @@
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.6 -c pytorch -c conda-forge
@@ CPU Only @@
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cpuonly -c pytorch
```

### 7. Create a new conda environment
```
conda activate life
```

### 8. Install the library dependencies
```
pip install -r requirements.txt
```

## Continue Development
### 1. Go into the root repository
```
cd to project directory
```
### 2. Activate the virtual environment
```
.\new_env\Scripts\activate
```

### 3. Activate the conda environment
```
conda activate life
```


### 11. Start Jupyter Notebook
```
jupyter notebook
```
