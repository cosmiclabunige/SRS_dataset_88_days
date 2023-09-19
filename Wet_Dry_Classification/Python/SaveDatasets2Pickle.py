import pickle
from pathlib import Path 

datasetPath = Path("Dataset")

index = [ 27, 28, 38, 39, 51, 54, 55, 56, 59, 62, 63, 64, 71, 72, 81, 84, 128, 141, 167, 170, 180,
         188, 190, 191, 192, 195, 206, 207, 210, 212, 214, 215, 220, 221, 222, 228, 236, 238, 240, 244, 246, 249, 253,
         256, 257, 21, 58, 60, 61, 73, 86, 89, 97, 127, 133, 136, 149, 151, 182, 184, 185, 189, 202,
         229, 235, 239, 242, 247, 255, 261, 265, 287, 297, 319, 322, 326, 328, 337]

index.sort()

radius = "85"
datasetPath85 =  datasetPath / f"Processed_SRS_{radius}.pkl"

radius = "60"
datasetPath60 = datasetPath / f"Processed_SRS_{radius}.pkl"

with open(datasetPath85, "rb") as f:
    datasetDict85 = pickle.load(f)
    f.close()

with open(datasetPath60, "rb") as f:
    datasetDict60 = pickle.load(f)
    f.close()

SRSmeasures_mV_85 = []
SRSmeasures_mV_60 = []
TBRGmeasures_mmh = []
Timestamps = []
for i in index:
    SRSmeasures_mV_85.append(datasetDict85["listOfSRSValuesInterpolated"][i])
    SRSmeasures_mV_60.append(datasetDict60["listOfSRSValuesInterpolated"][i])
    TBRGmeasures_mmh.append(datasetDict85["listOfTBRGValues"][i])
    Timestamps.append(datasetDict85["listOfSRSTimeStamps"][i])


path2save = datasetPath / "testset.pkl"
dataset = {"SRS_85": SRSmeasures_mV_85, "SRS_60": SRSmeasures_mV_60, "TBRG": TBRGmeasures_mmh, "Dates": Timestamps}
with open(path2save, "wb") as f:
    pickle.dump(dataset,f)
    f.close()

