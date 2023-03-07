The pickle file contains the datasets concerning the 88 days of rainfall measurements described in the article "". The data have been collected using sensors mounted on two parabolic dishes having a diameter of 60cm and 85cm, respectively.

Open the datasets using the pickle library for python with the following commands:

```ruby
import pickle as pk 

datasetPath = "SRS_dataset_88days.pkl"  
with open(datasetPath, "rb") as f:  
   dataset = pk.load(f) 
   f.close()  
```
|---------|-------|--------|----------|---------|
|**Event**|**Day**|  **h** |**max RI**|  **d**  |
| **ID**  |       |**[mm]**|**[mm/h]**|**[min]**|
|---------|-------|--------|----------|---------|


The *dataset* is a dictionary containing the 88 days SRS measurements expressed in mV for both dishes, the 88 days TBRG measurements expressed in mm/min, and the timestamps of the measurements.
