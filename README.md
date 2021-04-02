# Neureka-2020-Epilepsy-Challenge

The code implemented for 2020 Neureka-Epilepsy-Challenge paper.

Seizure Event Detection using minimum electrodes.

[Paper version 1 release here](https://github.com/NeuroSyd/Neureka-2020-Epilepsy-Challenge/blob/master/Two-Channel%20Epileptic%20Seziure%20Detection%20with%20Blended%20Multi-Time%20Segements%20Electroencephalography%20(EEG)%20Spectrogram.pdf)
Please cite: https://www.biorxiv.org/content/10.1101/2021.03.07.433990v1

## Preprocessing
Load raw eeg data using STFT
```python
cd utils/
python load_data_elec_3s.py
python load_data_elec_5s.py
python load_data_elec_7s.py
```
## Model Training
```python
python --mode=train
```
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
