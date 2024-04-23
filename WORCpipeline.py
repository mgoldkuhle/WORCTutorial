# import neccesary packages
from WORC import BasicWORC
import os

# These packages are only used in analysing the results
import pandas as pd
import json
import fastr
import glob

### Data loading and preparation
script_path = os.getcwd()
data_path = '/mnt/share/01_followup_cleanedup'
# File in which the labels (i.e. outcome you want to predict) is stated
label_path = '/mnt/c/Users/mjgoldkuhle/ownCloud/LUMC/data/selected_features'
label_file = os.path.join(label_path, 'schwannoma_growths_t1ce.csv')

## add evaluation in the end?
add_evaluation = True

# ###################### refactor into class ############################

# class WORCPipeline:
#     def __init__(self, data_path, label_path, label_file, add_evaluation=True):
#         self.data_path = data_path
#         self.label_path = label_path
#         self.label_file = os.path.join(label_path, label_file)
#         self.add_evaluation = add_evaluation
#         self.patients = {}
#         self.dummy_image_path = '/mnt/share/01_followup_cleanedup/followup_part1/id_00015976/20080202/00015976_20080202_t1ce.nii.gz'
#         self.dummy_seg_path = '/mnt/share/01_followup_cleanedup/followup_part1/id_00015976/20080202/00015976_20080202_t1ce_seg.nii.gz'

#     def load_data(self):
#         for root, dirs, files in os.walk(self.data_path):
#             for file in files:
#                 if file.endswith('.nii.gz') and ('t1ce' in file):
#                     patient_id, date = tuple(file.split('_')[0:2])
#                     patient_id = 'id_' + patient_id
#                     file_path = os.path.join(root, file)
#                     self._add_file_to_patient(patient_id, date, file_path)

#     def _add_file_to_patient(self, patient_id, date, file_path):
#         if patient_id in self.patients:
#             if date in self.patients[patient_id]:
#                 self.patients[patient_id][date].append(file_path)
#             else:
#                 self.patients[patient_id][date] = [file_path]
#         else:
#             self.patients[patient_id] = {date: [file_path]}

#     def process_data(self):
#     # remove patients with less than 2 timepoints and find out which patient has the most timepoints
#         max_num_dates = 0
#         self.max_num_patient = ''
#         for patient in list(self.patients.keys()):
#             if len(self.patients[patient]) > max_num_dates:
#                 max_num_dates = len(self.patients[patient])
#                 self.max_num_patient = patient
#             if len(self.patients[patient]) < 2:
#                 del self.patients[patient]
#         # for now remove patients that have 0 measures from Yunjie's model until that is fixed. Loses 69/207 patients with 2+ measures.
#         growths = pd.read_csv(self.label_file)
#         # exclude patients that have no growth data
#         for patient in list(self.patients.keys()):
#             if patient not in list(growths['Patient']):
#                 del self.patients[patient]
#         max_dates = 3  # for now only include max 3 timepoints
#         # Create lists of dictionaries for each timepoint (except the last per patient), sequence (t1ce or t2) and segmentation (t1ce_seg or t2_seg)
#         self.timepoints_t1ce = []
#         self.timepoints_t1ce_seg = []
#         for i in range(max_dates):
#             timepoint_t1ce = {}
#             timepoint_t1ce_seg = {}
#             for patient, dates in self.patients.items():
#                 sorted_dates = sorted(dates)
#                 if len(dates) > i + 1:  # exclude last timepoint for each patient as it shall be predicted
#                     for file in dates[sorted_dates[i]]:
#                         if file.endswith('t1ce.nii.gz'):
#                             timepoint_t1ce[patient] = file
#                         elif file.endswith('t1ce_seg.nii.gz'):
#                             timepoint_t1ce_seg[patient] = file
#                 else:  # if patient doesn't have an image for this timepoint WORC needs a dummy file
#                     dummy_id = f"{patient}_Dummy"
#                     timepoint_t1ce[dummy_id] = self.dummy_image_path
#                     timepoint_t1ce_seg[dummy_id] = self.dummy_seg_path
#             self.timepoints_t1ce.append(timepoint_t1ce)
#             self.timepoints_t1ce_seg.append(timepoint_t1ce_seg)
#             if t1ce_count != t1ce_seg_count:
#                 print('Number of t1ce and t1ce_seg files do not match')

#             self.patient_list = list(timepoints_t1ce[0].keys())


# ###################### refactor into class ############################




# Create directory with patients as keys and list of dates as values
patients = {}
#ix = 0 # for testing
for root, dirs, files in os.walk(data_path):
    for file in files:
        # only use nifti files that contain either t1ce or t2 to exclude t1 that don't have segmentations
        if file.endswith('.nii.gz') and ('t1ce' in file): # for now only t1ce  (otherwise add: or 't2' in file)
            patient_id, date = tuple(file.split('_')[0:2])
            patient_id = 'id_' + patient_id
            file_path = os.path.join(root, file)

            # create nested dictionary like {patient_id: {date: [files]}}
            if patient_id in patients:
                if date in patients[patient_id]:
                    patients[patient_id][date].append(file_path)
                else:
                    patients[patient_id][date] = [file_path]
            else:
                patients[patient_id] = {date: [file_path]}


# remove dates where in the diameter csv the max_diameter is 0
diameters = pd.read_csv(os.path.join(label_path, 'schwannoma_diameters_t1ce.csv'))
diameters['date'] = diameters['date'].astype(str)
# count the total number of dates in patients
total_dates = sum(len(dates) for dates in patients.values())

# Ensure every date in "patients" dictionary has an entry in "diameters" DataFrame. this gets rid of some faulty scans or segmentations.
for patient, dates in list(patients.items()):
    for date in list(dates.keys()):
        if not ((diameters['patient'] == patient) & (diameters['date'] == date)).any():
            del patients[patient][date]
            if len(patients[patient]) == 0:
                del patients[patient]

# count the total number of dates in patients after removing faulty scans
total_dates_cleaned = sum(len(dates) for dates in patients.values())
print(f"Removed {total_dates - total_dates_cleaned} / {total_dates} scans that had to max_diameter measure.")


# remove patients with less than 2 timepoints and find out which patient has the most timepoints
max_num_dates = 0
max_patient = ''
for patient in list(patients.keys()):
    if len(patients[patient]) > max_num_dates:
        max_num_dates = len(patients[patient])
        max_patient = patient
    if len(patients[patient]) < 2:
        del patients[patient]

# for now remove patients that have 0 measures from Yunjie's model until that is fixed. Loses 69/207 patients with 2+ measures.
growths = pd.read_csv(label_file)
# exclude patients that have no growth data
for patient in list(patients.keys()):
    if patient not in list(growths['Patient']):
        del patients[patient]

# select how many timepoints to include in WORC
max_dates = 3  # for now only include max 3 timepoints. if patients have less than 3 timepoints, WORC imputes the missing data

# Create lists of dictionaries for each timepoint (except the last per patient), sequence (t1ce or t2) and segmentation (t1ce_seg or t2_seg)
timepoints_t1ce = []
timepoints_t1ce_seg = []
# timepoints_t2 = []
# timepoints_t2_seg = []
t1ce_count = 0
t1ce_seg_count = 0
# t2_count = 0
# t2_seg_count = 0

# dummy image and segmentations for patients that don't have an image for a timepoint. needed for WORC
dummy_image_path = '/mnt/share/01_followup_cleanedup/followup_part1/id_00015976/20080202/00015976_20080202_t1ce.nii.gz'
dummy_seg_path = '/mnt/share/01_followup_cleanedup/followup_part1/id_00015976/20080202/00015976_20080202_t1ce_seg.nii.gz'

for i in range(max_dates):
    timepoint_t1ce = {}
    timepoint_t1ce_seg = {}
    # timepoint_t2 = {}
    # timepoint_t2_seg = {}

    for patient, dates in patients.items():
        sorted_dates = sorted(dates)
        if len(dates) > i + 1:  # exclude last timepoint for each patient as it shall be predicted
            for file in dates[sorted_dates[i]]:
                if file.endswith('t1ce.nii.gz'):
                    timepoint_t1ce[patient] = file
                    t1ce_count += 1
                elif file.endswith('t1ce_seg.nii.gz'):
                    timepoint_t1ce_seg[patient] = file
                    t1ce_seg_count += 1
                # elif file.endswith('t2.nii.gz'):
                #     timepoint_t2[patient] = file
                #     t2_count += 1
                # elif file.endswith('t2_seg.nii.gz'):
                #     timepoint_t2_seg[patient] = file
                #     t2_seg_count += 1
        else:  # if patient doesn't have an image for this timepoint WORC needs a dummy file
            dummy_id = f"{patient}_Dummy"
            timepoint_t1ce[dummy_id] = dummy_image_path
            timepoint_t1ce_seg[dummy_id] = dummy_seg_path

    timepoints_t1ce.append(timepoint_t1ce)
    timepoints_t1ce_seg.append(timepoint_t1ce_seg)
    # timepoints_t2.append(timepoint_t2)
    # timepoints_t2_seg.append(timepoint_t2_seg)

if t1ce_count != t1ce_seg_count:
    print('Number of t1ce and t1ce_seg files do not match')
# if t2_count != t2_seg_count:
#     print('Number of t2 and t2_seg files do not match')

patient_list = list(timepoints_t1ce[0].keys())
# append patients from t2 to patient_list if they are not already in there
# for patient in timepoints_t2[0].keys():
#     if patient not in patient_list:
#         patient_list.append(patient)

### WORC

# Determine whether you would like to use WORC for binary_classification,
# multiclass_classification or regression
modus = 'binary_classification'

# Name of the label you want to predict
if modus == 'binary_classification':
    # Classification: predict a binary (0 or 1) label
    label_name = ['above_2mm']

elif modus == 'regression':
    # Regression: predict a continuous label
    label_name = ['growth']

elif modus == 'multiclass_classification':
    # Multiclass classification: predict several mutually exclusive binaru labels together
    label_name = ['imaginary_label_1', 'complement_label_1']

# Determine whether we want to do a coarse quick experiment, or a full lengthy
# one. Again, change this accordingly if you use your own data.
coarse = False

# Give your experiment a name
experiment_name = '09_schwannoma_t1ce_3timepoints_classification'

# Instead of the default tempdir, let's but the temporary output in a subfolder
# in the same folder as this script
tmpdir = os.path.join(script_path, 'WORC_' + experiment_name)
print(f"Temporary folder: {tmpdir}.")

# Create a WORC object
experiment = BasicWORC(experiment_name)

# append all images and segmentations to the WORC object
for i in range(max_dates):
    experiment.images_train.append(timepoints_t1ce[i])
    experiment.segmentations_train.append(timepoints_t1ce_seg[i])
    print(i)

# add labels to the WORC object
experiment.labels_from_this_file(label_file)
experiment.predict_labels(label_name)

# Set the types of images WORC has to process. Used in fingerprinting
# Valid quantitative types are ['CT', 'PET', 'Thermography', 'ADC']
# Valid qualitative types are ['MRI', 'DWI', 'US']

experiment.set_image_types(['MRI'] * max_dates)  # WORC needs the image type for each timepoint

# Use the standard workflow for your specific modus
if modus == 'binary_classification':
    experiment.binary_classification(coarse=coarse)
elif modus == 'regression':
    experiment.regression(coarse=coarse)
elif modus == 'multiclass_classification':
    experiment.multiclass_classification(coarse=coarse)

# Set the temporary directory
experiment.set_tmpdir(tmpdir)

# optional, adds run statistics, classification reports, ROC curves, etc.
if(add_evaluation):
    experiment.add_evaluation()

# Run the experiment!
experiment.execute()

# Locate output folder
outputfolder = fastr.config.mounts['output']
experiment_folder = os.path.join(outputfolder, 'WORC_' + experiment_name)

print(f"Your output is stored in {experiment_folder}.")

# Read the features for the first patient
# NOTE: we use the glob package for scanning a folder to find specific files
feature_files = glob.glob(os.path.join(experiment_folder,
                                       'Features',
                                       'features_*.hdf5'))
if len(feature_files) == 0:
    raise ValueError('No feature files found: your network has failed.')

feature_files.sort()
featurefile_p1 = feature_files[0]
features_p1 = pd.read_hdf(featurefile_p1)

# Read the overall peformance
performance_file = os.path.join(experiment_folder, 'performance_all_0.json')
if not os.path.exists(performance_file):
    raise ValueError(f'No performance file {performance_file} found: your network has failed.')
    
with open(performance_file, 'r') as fp:
    performance = json.load(fp)

# Print the feature values and names
print("Feature values from first patient:")
for v, l in zip(features_p1.feature_values, features_p1.feature_labels):
    print(f"\t {l} : {v}.")

# Print the output performance
print("\n Performance:")
stats = performance['Statistics']
for k, v in stats.items():
    print(f"\t {k} {v}.")