import urllib.request, os, zipfile

URLS = {
    'https://github.com/alexander-g/INBD/releases/download/dataset_v1/DO_v1.zip' : 'dataset/DO.zip',
    'https://github.com/alexander-g/INBD/releases/download/dataset_v1/EH_v1.zip' : 'dataset/EH.zip',
    'https://github.com/alexander-g/INBD/releases/download/dataset_v1/VM_v1.zip' : 'dataset/VM.zip',
}

for url, destination_zipfile in URLS.items():
    destination_dir = os.path.dirname(destination_zipfile)
    print(f'Downloading {url} ...')
    with urllib.request.urlopen(url) as f:
        os.makedirs( destination_dir, exist_ok=True )
        open(destination_zipfile, 'wb').write(f.read())
    
    print(f'Extracting into {os.path.abspath(destination_dir)}')
    zipfile.ZipFile(destination_zipfile).extractall(destination_dir)
    os.remove(destination_zipfile)

print('Done')