# Neureka-2020-Epilepsy-Challenge

The code implemented for 2020 Neureka-Epilepsy-Challenge paper.
Seizure Event Detection using minimum electrodes



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
