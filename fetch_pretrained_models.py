import urllib.request, os

URLS = {
    'https://github.com/alexander-g/INBD/releases/download/pretrained_v2/DO_v2.zip' : 'checkpoints/INBD_DO/model.pt.zip',
    'https://github.com/alexander-g/INBD/releases/download/pretrained_v2/EH_v2.zip' : 'checkpoints/INBD_EH/model.pt.zip',
    'https://github.com/alexander-g/INBD/releases/download/pretrained_v2/VM_v2.zip' : 'checkpoints/INBD_VM/model.pt.zip',
}

for url, destination in URLS.items():
    print(f'Downloading {url} ...')
    with urllib.request.urlopen(url) as f:
        os.makedirs( os.path.dirname(destination), exist_ok=True )
        open(destination, 'wb').write(f.read())
