# ML-LSTM-sentimentanalysis
Here to implement a sequence model with a bidirectional LSTM. At most two layers of LSTM can be used, the model accuracy will be around 80%. 
**Data**
This project builds a sequence model with LSTM to conduct sentiment analysis on
SST-2 dataset with postive and negative sentiments.
**HOW TO RUN**
(This code was written and run on MacOS which doesn't support CUDA, for faster processing time, the end code is run via GPU-Colab. Please install from source if CUDA is needed on your end, or use https://colab.research.google.com/notebooks/gpu.ipynb) 
PREREQUISITES FOR PYTORCH IS FOR MAC

For other systems please view: https://pytorch.org/get-started/locally/

To run Pytorch, you have the meet the following prerequisites: 
### macOS Version[](https://pytorch.org/get-started/locally/#macos-version)

PyTorch is supported on macOS 10.10 (Yosemite) or above.

### Python[](https://pytorch.org/get-started/locally/#mac-python)

It is recommended that you use Python 3.5 or greater, which can be installed either through the Anaconda package manager (see  [below](https://pytorch.org/get-started/locally/#anaconda)),  [Homebrew](https://brew.sh/), or the  [Python website](https://www.python.org/downloads/mac-osx/).

### Package Manager[](https://pytorch.org/get-started/locally/#mac-package-manager)

To install the PyTorch binaries, you will need to use one of two supported package managers:  [Anaconda](https://www.anaconda.com/download/#macos)  or  [pip](https://pypi.org/project/pip/). Anaconda is the recommended package manager as it will provide you all of the PyTorch dependencies in one, sandboxed install, including Python.

1. Follow instructuions [above](https://pytorch.org/get-started/locally/) to install Pytorch
2. Open .py file in your preferred integrated developer environment (IDE)
3. Import and have Torch, Torchvision, Torchtext Matplotlib, etc installed and set up
4. Run import and Dataloader code to set up 
5. Only modify the Learning Rate, Weight Decay, and amount of Epochs while configuring for different training processes
6. GPU is built in as optional but strongly recommended
7. Run the model 
8. You should expect to see the average batch and validationa loss, as well as traning and validation accuracy percentage at each epoch
9. You should also expect to see overall test loss and test accuracy along with the avg batch loss per epoch graph
