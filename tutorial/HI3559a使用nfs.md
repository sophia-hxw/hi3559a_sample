## PC上的安装和配置
- 安装
```
sudo apt-get install nfs-kernel-server
```
- 共享文件夹
```
sudo mkdir /home/demo/hi3559a_share
```
- 配置文件
打开```/ets/exports```文件，在最后加上：
```
/home/demo/hi3559a_share *(rw,sync,no_subtree_check)
```
**NOTE:参数说明找百度哦**
- 重启服务
上面的配置完成之后，重启服务：
```
sudo /etc/init.d/rpcbind restart//重启 rpcbind
sudo /etc/init.d/nfs-kernel-server restart //重启 NFS
```

---

## 开发板上的安装和配置
- 安装客户端
```
sudo apt-get install nfs-common
```
- 挂载文件夹
```
mkdirs -p /home/hihope/hi3559av100
chmod 777 /home/hihope/hi3559av100
```

---

## 常用命令
- 挂载命令
```
sudo mount -t nfs -o nolock,nfsvers=3,vers=3 192.168.86.105:/home/demo/hi3559a_share /home/hihope/hi3559av100
```

- 卸载命令
```
umount /home/hihope/hi3559av100
df	//查看挂载信息
```
