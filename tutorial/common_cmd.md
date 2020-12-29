查看串口设备：
dmesg | grep ttyS*

查看端口状态：
netstat -tnl

完全卸载软件：
sudo apt-get --purge remove <programname>

安装deb软件：
sudo dpkg -i deb-name

卸载deb软件：
sudo dpkg -r deb-name
