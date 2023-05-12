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
intensity in mm/h (max RI) observed by the TBRG, and the number of rainy minutes (d).

![image](https://user-images.githubusercontent.com/114477377/223406835-db3ce7a1-69ca-491d-82c0-567ec7696dd5.png)

