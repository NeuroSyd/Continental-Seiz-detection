# Neureka-2020-Epilepsy-Challenge && Continental generalization of an AI system for clinical seizure recognition

The code implemented for 2020 Neureka-Epilepsy-Challenge paper and Continental generalization of an AI system for clinical seizure recognition.

Seizure Event Detection using minimum electrodes.

[Paper version 1 release here](https://www.researchgate.net/publication/350387463_Two-Channel_Epileptic_Seizure_Detection_with_Blended_Multi-Time_Segments_Electroencephalography_Spectrogram)

Please cite: https://www.biorxiv.org/content/10.1101/2021.03.07.433990v2

"Continental generalization of an AI system for clinical seizure recognition"

## Preprocessing
Load raw eeg data using STFT
```python
cd utils/
python load_data_elec_3s.py
python load_data_elec_5s.py
python load_data_elec_7s.py
```
Preprocessing the data with ICA
```python
cd utils/
python ICA_load_data_elec.py
```
## Model Training
```python
python main.py --mode=train
```
## Pretrained Model
Conv-LSTM pretrained model:
https://drive.google.com/file/d/1Tj2JZ_B5OqZrVILg15L_lPR2DKYiBoDS/view?usp=sharing
## Post Processing
Get raw results
```python
python main.py --mode=test
```
Get results based on threhold and apply average method.
```python
python main.py --mode=vote
```
Vote and discard short prediction
```python
cd post_process_code/
python overlap.py 
python discard.py 
python clean.py
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Better contact original contributor.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
