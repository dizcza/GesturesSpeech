# coding=utf-8

import sys
import os
from tqdm import tqdm

if sys.version_info >= (3,):
    import urllib.request as urllib2
    import urllib.parse as urlparse
else:
    import urllib2
    import urlparse


def download_url(url, desc=None):
    """
    :param url: url to download a file
    :param desc: destination folder to download in
                 (downloads in current folder by default)
    :return: downloaded filename
    """
    url_response = urllib2.urlopen(url)

    scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
    fname = os.path.basename(path)
    if not fname:
        fname = 'downloaded.file'
    if desc:
        fname = os.path.join(desc, fname)

    with open(fname, 'wb') as f:
        meta = url_response.info()
        meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
        meta_length = meta_func("Content-Length")
        file_size = None
        if meta_length:
            file_size = int(meta_length[0])
        print("Downloading: {0} Size: {1} Kb".format(url, file_size // 1024))

        progress_bar = tqdm(total=file_size, leave=True)
        for curr_buffer in url_response:
            f.write(curr_buffer)
            progress_bar.update(len(curr_buffer))
        progress_bar.close()

    return fname


if __name__ == "__main__":
    fname = download_url("https://flashcart-helper.googlecode.com/files/UnRAR.exe")
    os.remove(fname)
