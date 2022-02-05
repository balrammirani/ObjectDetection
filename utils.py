import os
import wget
import tarfile
# Function to Check if the path specified
# specified is a valid directory
def isEmpty(path):
    if os.path.exists(path) and not os.path.isfile(path):
  
        # Checking if the directory is empty or not
        if not os.listdir(path):
            return True
        else:
            return False
    else:
        print("The path is either for a file or not valid")
        print('Creating data directory')
        os.makedirs(path)
        return True

def setupdata(dataset_config):

    if isEmpty('data'):
        print('data not present.. downloading the default data')
        filename = wget.download(dataset_config['URL'],out = 'data')
        print(filename)
        my_tar = tarfile.open('data/trainval.tar.gz')
        my_tar.extractall('data')

    else:
        print('data already present... no need to download')