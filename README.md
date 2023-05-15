The pickle file contains the datasets concerning the 88 days of rainfall measurements described in the article "". The data have been collected using sensors mounted on two parabolic dishes having a diameter of 60cm and 85cm, respectively. 

Open the datasets using the pickle library for python with the following commands:

```ruby
import pickle as pk 

datasetPath = "SRS_dataset_88days.pkl"  
with open(datasetPath, "rb") as f:  
   dataset = pk.load(f) 
   f.close()  
```

The *dataset* is a dictionary containing the 88 days SRS measurements expressed in mV for both dishes, the 88 days TBRG measurements expressed in mm/min, and the timestamps of the measurements.

The events are listed in the table which summarizes the dataset, displaying the ID assigned to each event, the total accumulation of rain in mm (h), the maximum rainfall
intensity in mm/h (max RI) observed by the TBRG, the number of rainy minutes (d), the freezing level in m. The elevation angle of the sensors with respect to the Turksat dataset is 29.2048°.

![image](https://github.com/cosmiclabunige/SRS_dataset_88_days/assets/114477377/9bbb4f3d-c98d-43a9-9fb2-d91c373d428c)

