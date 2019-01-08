
# NetWalk
Implementation of NetWalk in paper KDD'18: NetWalk: A Flexible Deep Embedding Approach for Anomaly Detection in Dynamic Networks.

NetWalk: A Flexible Deep Embedding Approach for Anomaly Detection in Dynamic Networks. Wenchao Yu, *Wei Cheng, Charu Aggarwal, Kai Zhang, Haifeng Chen, Wei Wang. The Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (SIGKDD’18), 2018. (Oral, pdf, code)

Bib: @inproceedings{Yu:2018:NFD:3219819.3220024, author = {Yu, Wenchao and Cheng, Wei and Aggarwal, Charu C. and Zhang, Kai and Chen, Haifeng and Wang, Wei}, title = {NetWalk: A Flexible Deep Embedding Approach for Anomaly Detection in Dynamic Networks}, booktitle = {Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery &#38; Data Mining}, series = {KDD '18}, year = {2018}, isbn = {978-1-4503-5552-0}, location = {London, United Kingdom}, pages = {2672--2681}, numpages = {10}, url = {http://doi.acm.org/10.1145/3219819.3220024}, doi = {10.1145/3219819.3220024}, acmid = {3220024}, publisher = {ACM}, address = {New York, NY, USA}, keywords = {anomaly detection, clique embedding, deep autoencoder, dynamic network embedding}, }

Requirements for Run: We need python 3(2 also work) and tensorflow > 1.4

1). Install using the Homebrew package manager:

mac:

/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" export PATH="/usr/local/bin:/usr/local/sbin:$PATH" brew update brew install python@2  # Python 2 sudo pip install -U virtualenv  # system-wide install

Ubuntu:

sudo apt update sudo apt install python-dev python-pip sudo pip install -U virtualenv  # system-wide install

2). Create a new virtual environment by choosing a Python interpreter

virtualenv --system-site-packages -p python3.6 ./venv source ./venv/bin/activate pip install --upgrade pip

3). install tensorflow refer to https://www.tensorflow.org/install/

4). install packages pip install networkx pip install scipy pip install matplotlib pip install numpy pip install tqdm pip sklearn
