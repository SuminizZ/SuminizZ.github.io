---
title : "[Hadoop] Run Hadoop Cluster on Docker # 1 - Set up Hadoop on CentOS Container"
layout : post
categories : [de-hadoop]
img : de/docker.png
tags : [Hadoop, Hadoop Ecosystem, docker, mac]
toc : true
toc_sticky : true
---

<br/>

## 1. Download CentOS Image

<br/>

- (mac term) On your mac terminal, type the command line below to create new container with CentOS image (version 7 here)

```bash 
$ docker run --restart always --name [container_name] -dt centos:7
```

<br/>

- now you can see new centos image is created in your docker images list (Docker Dashboard)
<img src="https://user-images.githubusercontent.com/92680829/163716381-62d51c12-eb2d-4b79-82bf-1253e0d746aa.png" width="600">

<br/>

- new centos container is created with the name you set with the option ```--name [container_name]``` (here, my_centos)

- (mac term) execute the centos container that you've just created 
```bash 
$ docker exec -it my_centos_container /bin/bash
```

<br/>

- you can see the container list on run with the command ```docker ps``` 
<img width="906" alt="image" src="https://user-images.githubusercontent.com/92680829/163717913-f9c4b2f0-2c59-48c3-a2b4-26fab1381f75.png">


<br/>

- (mac term) execute docker 
```bash
$ docker exec -it [container_name] /bin/bash
```

<br/>

- after this command executed, you can see that your current serving environment is changed from base to root@[container_id]
<img width="500" alt="image" src="https://user-images.githubusercontent.com/92680829/163718158-2686975e-4958-4666-a618-db8a27aa7e91.png">

<br/>


## 2. Setting Hadoop Base on CentOS Image

<br/>

- (mac term) create new container that will be your hadoop base with the name 'hadoop_base' 
```bash
$ docker run -it --name hadoop_base -dt centos:7
```

<br/>

- (mac term) exec hadoop_base ```docker exec -it hadoop_base /bin/bash``` 
- (container) update yum packages and install all required libraries 
```bash
/* CentOS Container */
$ yum update
$ yum install wget -y
$ yum install vim -y
$ yum install openssh-server openssh-clients openssh-askpass -y
$ yum install java-1.8.0-openjdk-devel.x86_64 -y
```

<br/>

- **wget** : free software package for interacting with REST APIs to retrieve files using HTTP, HTTPS, FTP and FTPS
- **vim** : edit files at terminals
- **openssh-server openssh-clients openssh-askpass** : connectivity tool for remote login with the SSH protocol
- **java** : select the desired java version

- (container) type commands below to allow password-free interaction between containers (nodes of hadoop clusters) 
```bash
$ ssh-keygen -t rsa -P '' -f ~/.ssh/id_dsa
$ cat ~/.ssh/id_dsa.pub >> ~/.ssh/authorized_keys
```
```bash
$ ssh-keygen -f /etc/ssh/ssh_host_rsa_key -t rsa -N ""
$ ssh-keygen -f /etc/ssh/ssh_host_ecdsa_key -t ecdsa -N ""
$ ssh-keygen -f /etc/ssh/ssh_host_ed25519_key -t ed25519 -N "" 
```

<br/>

- (container) adding JAVA_HOME directory to PATH
```bash 
$ readlink -f /usr/bin/javac     ## check your java directory
/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.292.b10-1.el7_9.x86_64/bin/javac
$ vim ~/.bashrc      ## you can edit your PATH at terminal by using vim 
```
<br/>

- (vim) type 'i' to start writing mode and add your java direc (note! type except '/bin/javac' part)
```bash
.
.
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.322.b06-1.el7_9.aarch64
export PATH=$PATH:$JAVA_HOME/bin
.
.
```

<br/>

- result
- <img width="780" alt="image" src="https://user-images.githubusercontent.com/92680829/163719319-3608f010-c137-43fc-a13a-9c6337708abc.png">

<br/>

- (vim) to exit from writing mode, enter ```esc```
- (vim) to store the edit and exit from vim, type ```:w (store) -> :q (exit)```
- (container) make sure to actually execute the content of a file you've edited
```bash
$ source ~/.bashrc
```

<br/>

### Install Hadoop and Set Hadoop Configurations on CentOS Image

<br/>

- (container)
```bash
$ mkdir /hadoop_home       
$ cd /hadoop_home
$ wget https://archive.apache.org/dist/hadoop/common/hadoop-2.7.7/hadoop-2.7.7.tar.gz
## choose the hadoop version you want (here, hadoop-2.7.7)
$ tar -xvzf hadoop-2.7.7.tar.gz         ## unzip
```

<br/>

- (container) add HADOOP_HOME directory to your PATH
```bash
$ vim ~/.bashrc
```
<br/>

- (vim)
```bash
.
.
export HADOOP_HOME=/hadoop_home/hadoop-2.7.7
export HADOOP_CONFIG_HOME=$HADOOP_HOME/etc/hadoop
export PATH=$PATH:$HADOOP_HOME/bin
export PATH=$PATH:$HADOOP_HOME/sbin
## run sshd 
/usr/sbin/sshd          
.
.
```

<br/>

- result
- <img width="650" alt="image" src="https://user-images.githubusercontent.com/92680829/163719864-6efe2f89-f824-44b6-a6d4-0445811037a8.png">

<br/>

- (container) ```$ source ~/.bashrc```
- (container) create files (temp, namenode, datanode) in $HADOOP_HOME directory
```bash
$ mkdir /hadoop_home/tmp
$ mkdir /hadoop_home/namenode
$ mkdir /hadoop_home/datanode
```

<br/>

--- 

<br/>

Now, edit hadoop configurations with vim

(container)
```bash
$ cd $HADOOP_CONFIG_HOME
## create mapred-site.xml at $HADOOP_CONFIG_HOME direc
$ cp mapred-site.xml.template mapred-site.xml 
```

<br/>

#### 1) ***core-site.xml***

<br/>

(container) go to file *core-site.xml* 

```bash
vim $HADOOP_CONFIG_HOME/core-site.xml
```
(vim)
```html
<!-- core-site.xml -->
<configuration>
    <property>
        <name>hadoop.tmp.dir</name>
        <value>/hadoop_home/tmp</value>
    </property>

    <property>
        <name>fs.default.name</name>
        <value>hdfs://nn:9000</value>      <!-- nn : hostname of namenode, name as you wnat-->
        <final>true</final>
    </property>
</configuration>
```

<br/>

#### 2) ***hdfs-site.xml***

<br/>

```html
<!-- hdfs-site.xml -->
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>2</value>
        <final>true</final>
    </property>

    <property>
        <name>dfs.namenode.name.dir</name>
        <value>/hadoop_home/namenode</value>
        <final>true</final>
    </property>

    <property>
        <name>dfs.datanode.data.dir</name>
        <value>/hadoop_home/datanode</value>
        <final>true</final>
    </property>
</configuration>
```

<br/>

#### 3) ***mapred-site.xml***

<br/>

```html
<!-- mapred-site.xml -->
<configuration>

    <property>
        <name>mapred.job.tracker</name>
        <value>nn:9001</value>
    </property>

</configuration>
```
--- 

<br/>

- Finally, format namenode and commit the container to centos:hadoop image

    (container)
    ```bash
    $ hadoop namenode -format
    $ exit
    ```

    (mac term)

    ```bash
    $ docker commit -m "hadoop in centos" hadoop_base centos:hadoop
    ```


- docker commit -m [message] [container_name] [image_name] 


---
<br/>

Next posting, we will gonna create namenode and multiple datanodes with the created hadoop-base image file below (centos:hadoop)

<img width="700" alt="image" src="https://user-images.githubusercontent.com/92680829/163721073-aaf5ca3f-c66d-4e5f-be8b-485327ab1b1d.png">
