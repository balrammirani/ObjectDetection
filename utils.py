import os

# Function to Check if the path specified
# specified is a valid directory
def isEmpty(path):
    if os.path.exists(path) and not os.path.isfile(path):
  
        # Checking if the directory is empty or not
        if not os.listdir(path):
            if not os.path.exists('temp'):
                os.makedirs('temp')

            return True
        else:
            return False
    else:
        print("The path is either for a file or not valid")
        print('Creating data directory')
        if not os.path.exists('temp'):
            os.makedirs('temp')

        return True

def setupdata(dataset_config):

    if isEmpty('data'):
        print('data not present.. downloading the default data')
        filename = wget.download(dataset_config['URL'],out = 'temp')
        print(filename)
        my_tar = tarfile.open('temp/trainval.tar.gz')
        my_tar.extractall('data')

    else:
        print('data already present... no need to download')