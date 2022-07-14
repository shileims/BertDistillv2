# install blobfuse
#sudo apt-get install blobfuse

temp_PATH=/mnt/resource/blobfusetmp       # or other place for cache
user=shilei                              # user name of the linux e.g. root

sudo mkdir $temp_PATH -p
sudo chown $user $temp_PATH
chmod 600 fuse_connectionzms.cfg
sudo blobfuse Z:/ShiLei/SS200M --tmp-path=$temp_PATH  --config-file=fuse_connectionzms.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 -o allow_other

