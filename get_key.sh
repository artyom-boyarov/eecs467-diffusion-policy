sudo gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys $1
sudo gpg --export $1 | sudo tee /etc/apt/trusted.gpg.d/$2.gpg
