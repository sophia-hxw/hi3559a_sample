- 文件复制
```
cp /lib/modules/4.2.0-27-generic/kernel/drivers/usb/serial/pl2303.ko /usr/src/linux-headers-4.2.0-27-generic/drivers/usb/serial
```
内核不同，可能路径不同

- 安装命令
```
$ modprobe usbserial

$ modprobe pl2303
```

- 验证
输入```lsmod | grep usbserial```可以看到```usbserial```信息说明安装成功；
输入```dmesg | tail```可以看到```usb pl2303```等信息亦说明安装成功；
