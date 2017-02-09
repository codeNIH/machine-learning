import preprocessing.py as pre
import numpy as np
import os
import sys # for command line arguments

if(len(sys.argv)!=1):
    print('Need to give folder where patient folders are located')

MASTER_FOLDER = sys.argv[0]
all_patients = os.listdir(MASTER_FOLDER)
all_patients.sort()

for patient in all_patients: # Going to loop through each patient
    my_patient = pre.load_scan(MASTER_FOLDER + patient)
    my_patient_pixels = pre.get_pixels_hu(my_patient)
    pix_resampled, spacing = pre.resample(my_patient_pixels, my_patient, [1,1,1])
    segmented_lungs = pre.segment_lung_mask(pix_resampled, False)
    segmented_lungs_fill = pre.segment_lung_mask(pix_resampled, True)
    my_nodules = segmented_lungs_fill - segmented_lungs
    np.save(file=(os.path.join(MASTER_FOLDER, patient, 'img')), my_nodules)
