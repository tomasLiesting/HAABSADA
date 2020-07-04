# How to run HAABSADA
Data Augmentation extenstion on HAABSA. In order to use the code please follow the following steps:

1. Go to https://github.com/ofwallaart/HAABSA and follow the steps on the github page. There are some possible problems that you can come across when following his, which I hope to tackle as much as possible.
  - When running on windows, note that you need compatible c++ visual studio build tools (https://visualstudio.microsoft.com/downloads/). 
  - If you want to run on GPU (NVIDIA) note that you need compatible drivers, Cuda (v9.0) and cuDNN (v6.4)
  - If you need nltk files --> in your environment run `python` afterwards `import nltk` afterwards `ntlk.download(NLTK_FILE)`
  - Make sure your environment is active (with `Scripts/activate`) --> you should see the name of your environment next to the command line
2. Clone this repository using `git clone https://github.com/tomasLiesting/HAABSADA.git` and, in your file explorer, overwrite the files in HAABSA with the files in this repo
3. In your environment, cd to HAABSA, and again run `pip install -r requirements.txt`
4. Make sure you have access to the google translate api, if you want to use backtranslation. You have to create an account and get 300€ for free. This is done as follows:
  - `pip install --upgrade google-cloud-translate`
  - Afterwards go to this page https://cloud.google.com/docs/authentication/getting-started#windows
  - Create a service account
  - Create a new project
  - Create a service account with as role project owner
  - Download the json authentication (and place it somewhere you like)
  - In the environment run `$env:GOOGLE_APPLICATION_CREDENTIALS="[PATH]"` (where PATH is the path to your json authentication, e.g. `$env:GOOGLE_APPLICATION_CREDENTIALS="C:\Users\username\Downloads\my-key.json"`
5. In the environment, run `python main.py`


# How to modify parameters
In the file config.py, there are several Data Augmentation variables. These can be modified. To reproduce the 1% and 0.5% improvement on the 2015 and 2016 SemEval datasets, make sure that:
- EDA_type = 'adjusted'
- EDA_deletion =  0
- EDA_replacement = 1
- EDA_insertion = 1
- EDA_swap = 1
- EDA_pct = .2
- backtranslation_langs = 'None'
- use_word_mixup = 0
- original_multiplier = 3

Furthermore, in main.py make sure that:
 - loadData = True 
 - augment_data = True 
 - useOntology = True 
 - runLCRROTALT = True
For the rest all parameters in main.py should be False.
