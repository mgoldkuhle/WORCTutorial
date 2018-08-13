import WORC
import fastr
import glob
import os

# NOTE: This example is just to condense the content of the notebook to a script you can use.

# NOTE: please give the below value the name of your account on Ubuntu, such that the home folder can be correctly found
username = 'worc'

# Create network and use simplest plugin for execution.
network = WORC.WORC('example')
network.fastr_plugin = 'LinearExecution'

# Below are the minimal inputs: you can also use features instead of images and segmentations.
# You need to change these to your sources: check the FASTR documentation.
# See the WORC Github Wiki for specifications of the various attributes, such as the ones used below

image_sources = glob.glob(('/home/{}/Documents/Data/STWStrategyMMD/*/image.nii.gz').format(username))
image_sources = [i.replace(('/home/{}').format(username), 'vfs://home') for i in image_sources]
image_sources = {os.path.basename(os.path.dirname(i)): i for i in image_sources}
network.images_train.append(image_sources)

segmentation_sources = glob.glob(('/home/{}/Documents/Data/STWStrategyMMD/*/mask.nii.gz').format(username))
segmentation_sources = [i.replace(('/home/{}').format(username), 'vfs://home') for i in segmentation_sources]
segmentation_sources = {os.path.basename(os.path.dirname(i)): i for i in segmentation_sources}
network.segmentations_train.append(segmentation_sources)

network.labels_train.append('vfs://home/Documents/WORCTutorial/Data/StrategyMMD/pinfo.txt')

# WORC passes a config.ini file to all nodes which contains all parameters.
# You can get the default configuration with the defaultconfig function
config = network.defaultconfig()

# This configparser object can be directly supplied to wROC: it will save
# it to a .ini file. You can interact with it as a dictionary and change fields.
# See the Github Wiki for all available fields and their explanation.
config['SampleProcessing']['SMOTE'] = 'False'
config['CrossValidation']['N_iterations'] = '5'
config['Genetics']['label_names'] = '[imaginary_label_1]'

# Add the configuration to the network.
network.configs.append(config)

# Build the network: the returned network.network object is a FASTR network.
network.build()

# Convert your appended sources to actual sources in the network.
network.set()

# Execute the network
network.execute()