# Experimental setup to examine the impact of adapter tuning of the BERT architecture with LexGLUE
In my bachelor thesis I investigated the effects of adding adapter modules to the BERT architecture. As part of this work, I conducted an empirical study comparing different training methods and analyzing the resulting differences. This repository contains the source code that implements the training of the BERT architecture using finetuning and adapter tuning. Adapter modules with a bottleneck architecture were used for adapter tuning. 

## Methodology behind this experimental setup
Due to the different loss values produced by the two training methods, in addition to using early-stopping to determine the model, a user-defined callback was developed to allow for timed training. Early-stopping determines the optimal model state by comparing training and evaluation loss. However, this creates challenges because different loss magnitudes are generated depending on the model variant resulting from the selected training method. This can lead to ambiguities in model comparison. Therefore, time-dependent training was introduced, where models are trained with both training methods over a period of 24 hours, for example, and then compared. This minimizes the variability in the comparisons. Nevertheless, the models determined by early-stopping are also stored, since these are of great importance in practical applications and a comparison is therefore essential.

## Installation & Usage
**Linux/macOS**
1. Open a terminal and navigate to the directory where you want to create your Python environment.
2. Use the command `python3 -m venv .env` to create a virtual Python environment named ".env".
3. Activate the virtual Python environment with the command source `.env/bin/activate`. You should now see an arrow (e.g. "(.env)") in front of the prompt in your terminal, indicating that the virtual environment is activated.
4. Use the command `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116` to install pytorch. Make sure that you select the cuda version. [Look here to see the documentation](https://pytorch.org/)
5. Use the command `pip install -r requirements.txt` to install the remaining dependencies.
6. Use the command `cd src` to enter the source folder.
7. Use the command `python -m run_experiment` to run the script.

**Windows**
1. Open a command prompt window and navigate to the directory where you want to create your Python environment.
2. Use the command `python -m venv .env` to create a virtual Python environment named ".env".
3. Activate the virtual Python environment with the command `.env\Scripts\activate.bat`. You should now see an arrow (e.g. "(.env)") in front of the prompt in your command prompt window, indicating that the virtual environment is activated.
4. Use the command `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116` to install pytorch. Make sure that you select the cuda version. [Look here to see the documentation](https://pytorch.org/)
5. Use the command `pip install -r requirements.txt` to install the remaining dependencies.
6. Use the command `cd src` to enter the source folder.
7. Use the command `python -m run_experiment` to run the script.
## Credits
Please check out the  [LexGLUE](https://github.com/coastalcph/lex-glue)-Repository, that provided scripts of the data preprocessing. 

## Contact
Dennis Mustafic - dennismustafic@gmail.com
