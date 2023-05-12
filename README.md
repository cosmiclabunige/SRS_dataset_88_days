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

```latex
\usepackage{multirow}
\begin{table*}[b]
    \centering
    \caption{Dataset: h=rainfall measured by the TBRG, max RI = maximum rainfall intensity measured by the TBRG, d = minutes of rainfall.}
    \label{tab:dataset}
    \begin{tabular}{|c|c|c|c|c||c|c|c|c|c||c|c|c|c|c|}
    \hline
    \textbf{Ev} & \multirow{2}{*}{\textbf{Day}} & \textbf{h}    & \textbf{max RI} & \textbf{d} &
    \textbf{Ev} & \multirow{2}{*}{\textbf{Day}} & \textbf{h}    & \textbf{max RI} & \textbf{d} & 
    \textbf{Ev} & \multirow{2}{*}{\textbf{Day}} & \textbf{h}    & \textbf{max RI} & \textbf{d} \\
    \textbf{ID}    &                               & \textbf{[mm]} & \textbf{[mm/h]} & \textbf{[min]} &  
    \textbf{ID}    &                               & \textbf{[mm]} & \textbf{[mm/h]} & \textbf{[min]} & 
    \textbf{ID}    &                               & \textbf{[mm]} & \textbf{[mm/h]} & \textbf{[min]} \\ \hline
                  1                   &	02-05-17                    &	    1.0     &       12.8      &  21            &
                  31	                &   23-07-17	                &	   0        &       0         &  0             & 
                  61                  &	14-12-17	                &	   0        &       0         &  0             \\
                  2                   &	03-05-17	                &      15.2     &       83.9      &  106           &
                  32	                &   25-07-17                    &	   0.8      &       14.0      &  4             &        
                  62                  &	15-12-17	                &	   0        &       0         &  0             \\
                  3	                &   04-05-17	                &       1.5     &       20.1      &  12            &
                  33                  & 	29-07-17	                &      0.4      &       2.9       &	13             &        
                  63                  &	16-12-17	                &	   0        &       0         &  0             \\
                  4	                &   05-05-17	                &       0       &         0       &  0             &
                  34	                &   06-08-17	                &      1.9      &       40.5      &  5             &         
                  64                  &	24-12-17	                &	   0        &       0         &  0             \\
                  5	                &   06-05-17	                &      24.4     &       30.8      &	319            &
                  35                  &	09-09-17	                &      52.1     &      146.8      &	328            &      
                  65                  &	25-12-17	                &      7.1      &       54.5      &	61            \\
                  6	                &   11-05-17		            &      0        &         0       &  0             &
                  36                  &	10-09-17	                &	   0        &       0         &  0             &      
                  66                  &	01-01-18	                &     32.4      &       50.5      &	369            \\
                  7	                &   12-05-17                    &	   0.4      &        3.8      &  20            &
                  37                  &	15-09-17	                &      13.5     &       20.5      &	170            &      
                  67                  &	02-01-18	                &	   0        &       0         &  0             \\
                  8                   &	14-05-17		            &      0        &         0       &  0             &
                  38                  &	18-09-17	                &      19.4     &      111.4      &	262            &       
                  68                  &	05-01-18	                &	   0        &       0         &  0             \\
                  9	                &   15-05-17                    &      0        &  		  0       &  0             &
                  39                  &	23-09-17	                &	   0        &       0         &  0             &       
                  69                  &	06-01-18	                &      8.3      &       42.3      &	116            \\
                  10	                &   19-05-17	                &	   3.6      &        6.5      & 83             &
                  40                  &	01-10-17	                &      1.9      &        11.3     &	47             &
                  70                  &	07-01-18	                &	   0        &       0         &  0             \\
                  11                  &	24-05-17                    &	   0        &       0         &  0             &
                  41                  &	03-10-17	                &      1.4      &         5.3     &	42             &
                  71                  &	09-01-18	                &      7.1      &       46.1      &	63             \\
                  12                  & 	25-05-17		            &      0        &       0         &  0             &
                  42                  &	20-10-17	                &	   0        &       0         &  0             &        
                  72                  &	12-01-18	                &	   0        &       0         &  0             \\
                  13	                &   26-05-17	                &	   0        &       0         &  0             &
                  43                  &	23-10-17	                &	   0        &       0         &  0             &        
                  73                  &	14-01-18	                &	   0        &       0         &  0             \\
                  14                  &	05-06-17		            &      0        &       0         &  0             &
                  44                  &	02-11-17	                &	   0        &       0         &  0             &
                  74                  &	15-01-18	                &      2.7      &       6.6       &	140            \\   
                  15                  &	06-06-17                    &	   0        &       0         &  0             &
                  45                  &	04-11-17	                &       9.2     &       65.9      &	 73           &            
                  75                  &	17-01-18	                &	   0        &       0         &  0             \\
                  16	                &   18-06-17	                &      0        &       0         &  0             &
                  46                  &	06-11-17	                &       3.5     &        2.8      &	139            &
                  76                  &	25-01-18	                &	   0        &       0         &  0             \\
                  17	                &   21-06-17		            &      0        &       0         &  0             & 
                  47                  &	07-11-17	                &       0.6     &        2.9      &	13             &
                  77                  &	27-01-18	                &     19.4      &       24.9      &	295            \\
                  18                  &	22-06-17		            &      0        &       0         &  0             & 
                  48                  &	11-11-17	                &	    0       &         0       &  0             &
                  78                  &	28-01-18	                &	   0        &       0         &  0             \\
                  19                  &	23-06-17		            &      0        &       0         &  0             &    
                  49                  &	12-11-17	                &       2.4     &       35.1      &	27            &
                  79                  &	29-01-18	                &	   0        &       0         &  0             \\
                  20	                &   25-06-17	                &      2.0      &       20.5      &  20            &    
                  50                  &	13-11-17	                &	   0        &         0       &  0             &
                  80                  &	04-02-18	                &      1.7      &       3.8       &	46             \\
                  21                  &	26-06-17	                &      0        &       0         &  0             &    
                  51                  &	14-11-17	                &	   0        &         0       &  0             &
                  81                  &	08-02-18	                &       4.7     &       20.1      &	63             \\
                  22	                &   27-06-17                    &	   0.2      &        2.4      &  5             &     
                  52                  &	15-11-17	                &	   0        &         0       &  0             & 
                  82                  &	04-08-18	                &       1.9     &        17.6     &	16             \\
                  23	                &   28-06-17		            &      0.7      &       21.1      &  7             &         
                  53                  &	18-11-17	                &	   0        &         0       &  0             &
                  83                  &	14-08-18	                &     35.5      &       121.7     &	151             \\
                  24	                &   29-06-17	                &      0        &       0         &  0             &
                  54                  &	25-11-17	                &     21.0      &       43.4      &	342            &
                  84                  &	04-04-19	                &     38.5      &       220.4     &	346             \\
                  25	                &   01-07-17		            &      0        &       0         &  0             &
                  55                  &	29-11-17	                &	   0        &       0         &  0             &
                  85                  &	07-04-19	                &      5.5      &        13.3     &	112             \\
                  26	                &   02-07-17                    &	   0        &       0         &  0             &         
                  56                  &	30-11-17	                &	   0        &       0         &  0             &
                  86                  &	11-04-19	                &      1.9      &        1.7      &	95             \\
                  27	                &   09-07-17                    &	   0        &       0         &  0             &        
                  57                  &	04-12-17	                &	   0        &       0         &  0             &
                  87                  &	13-04-19	                &      6.2      &        23.8     &	153             \\
                  28	                &   10-07-17                    &	   0        &       0         &  0             &         
                  58                  &	06-12-17	                &	   0        &       0         &  0             &
                  88                  &	25-04-19	                &      1.5      &       42.3      &	8              \\ 
                  29	                &   11-07-17	                &      22.4     &       218.0     &	61             &         
                  59                  &	08-12-17	                &	   0        &       0         &  0             &
                                      &                               &               &                 &                \\
                  30                  &	20-07-17	                &	   0        &       0         &  0             &
                  60                  &	09-12-17	                &	   0        &       0         &  0             & 
                                      &                               &               &                 &                \\ \hline

    \end{tabular}
\end{table*}
```