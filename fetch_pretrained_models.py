import urllib.request, os

URLS = {
    'https://github.com/alexander-g/INBD/releases/download/pretrained_v1/DO_v1.zip' : 'checkpoints/INBD_DO/model.pt.zip',
    'https://github.com/alexander-g/INBD/releases/download/pretrained_v1/EH_v1.zip' : 'checkpoints/INBD_EH/model.pt.zip',
    'https://github.com/alexander-g/INBD/releases/download/pretrained_v1/VM_v1.zip' : 'checkpoints/INBD_VM/model.pt.zip',
}

for url, destination in URLS.items():
    print(f'Downloading {url} ...')
    with urllib.request.urlopen(url) as f:
        os.makedirs( os.path.dirname(destination), exist_ok=True )
        open(destination, 'wb').write(f.read())
