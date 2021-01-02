## 前提
- 关闭win10防火墙
- 确认网口和网线正常   
    怎么确认就各显神通了啊~~~

## 桥接模式 
原理我也不太懂，就不多说了。  
我的理解是，PC和板子用网线连接，PC和虚拟机是逻辑连接，需要保证pc, ubuntu（vmware中的）和板子在同一个网段中。

- PC主机
【自动获取ip改为固定的ip地址】
ip: 192.168.86.189
netmask: 255.255.255.0

- vmware
【编辑->虚拟网络编辑器】
添加网络：名称VMnet0，类型自定义模式，其他默认；
VMnet信息：选择桥接模式（将虚拟机直接连接到外部网络），已桥接至，选择以太网的网卡，注意需要在win10中确认；

【虚拟机->设置->网络适配器->网络连接】
自定义(U):特定虚拟网络，下拉项选择上一步新增加的VMnet0；
** NOTE: 该配置需要关闭虚拟机的系统才能设置；需要联网时选择NAT模式(N) **

- vmware中的ubuntu
我新建了两个网络，分别用于连接www网络(名称是www)和做linux服务器(名称是Profile 1)连接PC和板子；

   - Profile 1
   ip: 192.168.86.105
   netmask: 255.255.255.0
   broadcast: 192.168.86.255

   - www
   自动获取ip

- 板子
【uboot中的修改，命令如下】
```
setenv ITEMNAME ITEMVALUE   //ITEMNAME是设置项名称，ITEMVALUE是设置项值
saveenv     //保存设置
```
| ITEMNAME | ITEMVALUE |
| - | - |
| ipaddr | 192.168.86.114 |
| netmask | 255.255.255.0 |
| gatewayip | 192.168.86.1 |
| serverip | 192.168.86.105 |
【板子系统中修改，我的文件是/etc/init.d/S80...】
ipaddr = 192.168.86.114
netmask = 255.255.255.0
gateway = 192.168.86.1

## 验证
- PC和vmware中ubuntu相互ping通

- PC和板子相互ping通

- vmware中ubuntu和板子相互ping通

## 其他
原本想放那些设置的图片出来的，可是md文件就不那么方便了，容我想一个方式先.....