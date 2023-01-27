---
title : "[Hadoop] Run Hadoop Cluster on Docker # 2 - Create Hadoop Cluster and Set Up Hadoop Deamons on Docker Containers"
layout : post
categories : [de-hadoop]
img : de/docker.png
tags : [Hadoop, Hadoop Ecosystem, docker, mac]
toc : true
toc_sticky : true
---

<br/>

## 1. Create Hadoop Cluster : NameNode and 3 DataNodes

<br/>

### - Create Nodes and Connect them
- Previously, we made hadoop-base container where we install and set Hadoop configurations on CentOS image
- For this time, we will gonna make 4 nodes (single namenode and 3 datanodes) using previously set centos:hadoop image 
<br/>

- Firstly, let's make NameNode
- (mac term) 
```bash
$ docker run --it -h --name namenode -p 50070:50070 centos:7
```

<br/>

- **Port Forwarding**
    - connect localhost port 50070 on local PC to 50070 port of Docker containers
    - <img src="https://user-images.githubusercontent.com/92680829/169309361-b0d715ea-ee62-4e65-aa7e-1a0a7c30cf26.png" width="430">

<br/>

- Now let's make 3 Datanodes linked to Namenode container
```bash
$ docker run --it -h --name dn1 --link namenode:nn centos:7
$ docker run --it -h --name dn2 --link namenode:nn centos:7
$ docker run --it -h --name dn3 --link namenode:nn centos:7
```

<br/>


- Link three datanodes to namenode with option ```--link```
    - --link [container_name]:[alias] 
- This allows /etc/hosts file of slave containers to contain IP address of master container
- Any change on IP address of linked container is updated automatically 

<br/>

### - Store Information of DataNodes into NameNode

<br/>

- get IP addresses of all three datanodes with the command
- ```docker inspect [target_container] | grep IPAddress```

    <img src="https://user-images.githubusercontent.com/92680829/169314217-672a3cb3-28f1-4e1e-a99a-1aff0f90dc2c.png" width="520">

<br/>

- Now, add IP addresses of datanodes to /etc/hots file of NameNode 
- exec NameNode and edit /etc/hosts with **vim** like below
```bash
$ (mac term) docker exec -it namenode /bin/bash
$ (nn container) vim /etc/hosts 
``` 
<br/>
    <img src="https://user-images.githubusercontent.com/92680829/169540036-2c161b5a-32aa-4fc2-88a1-d29e72f1975b.png" width="380">

    - Note that all changes to hosts file are reset when you stop and restart the container, so make sure to re-edit the file every restart (preparing shell script file would be convenient)

<br/>

- Now let's add hostname of each datanode to **slaves** file 

    <img src="https://user-images.githubusercontent.com/92680829/169322013-d2563125-dd42-44e1-8f72-0f0964e0db15.png" width="200">
    
    - The $HADOOP_INSTALL/hadoop/conf directory contains some configuration files for Hadoop
    - **slaves** file is one of those
        - This file lists the hosts, one per line, where the Hadoop slave daemons (datanodes and tasktrackers) will run. By default this contains the single entry localhost
    - Ohter documentations are [<span style="color:blue">**here**</span>](https://cwiki.apache.org/confluence/display/HADOOP2/GettingStartedWithHadoop){:target="_blank"}

<br/>

## 2. Launch Hadoop Deamons on Docker Containers

<br/>

- You've just successfully created all the nodes required and connect them each other 
- Now let's actually run Hadoop deamons by activating ```start-all.sh``` script
```bash
$ (nn container) start-all.sh
```
<br/>

- If you type the command above, you'll see multiple warnings and questions like "Are you sure you want to continue connecting (yes/no)?" 
- Ignore all of them and answer yes
- When you finish all steps, you can finally see these lines below 
<img src="https://user-images.githubusercontent.com/92680829/169324484-7356cc90-8d37-4116-9267-2879e8a2ee05.png" width="700">

<br/>


- **Startup Scripts**
    - $HADOOP_INSTALL/hadoop/bin directory contains some scripts used to launch Hadoop DFS and Hadoop Map/Reduce daemons
    - start-all.sh is one of those
        - It starts all Hadoop daemons, the namenode, datanodes, the jobtracker and tasktrackers. 
        - Now Deprecated; use start-dfs.sh then start-mapred.sh instead
    - Ohter documentations are [<span style="color:blue">**here**</span>](https://cwiki.apache.org/confluence/display/HADOOP2/GettingStartedWithHadoop){:target="_blank"}


<br/>

### - Check Current Process Status 

<br/>

- (nn container) after executing the script, type ```ps -ef``` to check the current process status
<img src="https://user-images.githubusercontent.com/92680829/169325738-08e8cbfb-f591-4c7e-80c7-2596b86e98d9.png" width="700">

<br/>

- you'll see java process is running, but no detail is shown there
- So, you can alternatively use command ```jps``` : lists processes currently running on jvm
    - namenode : <img src="https://user-images.githubusercontent.com/92680829/169326533-83551620-69c4-41dd-8e2a-7eb850c69711.png" width="250">
    - datanode (dn2) : <img src="https://user-images.githubusercontent.com/92680829/169536745-98f91469-bb2c-4ef1-8067-2ccc9fb92b5b.png" width="180">

<br/>

- You can see all Hadoop deamons are set normally on each container for **HDFS and YARN Hadoop system**
    - Secondary NameNode, Resource Manager are running on namenode container
    - while DataNode and NodeManager are activatded on datanode containers
    - <img src="https://user-images.githubusercontent.com/92680829/169539052-13390366-25ec-471a-9e9c-8cbbeeaa1884.png" width="450">

<br/>

### - Start Hadoop on your Nodes

<br/>

- Now you can use **Hadoop commands** on your container terminal
    <br/>

    - 1) ```# hdfs dfsadmin -report``` : to check current status of Hadoop cluster
        - You'll find 'Live datanodes (3)' output, which shows that three datanodes (dn1, 2, 3) are currently running 
        - Also, you can drill down into the detail status of each datanode
            - <img src="https://user-images.githubusercontent.com/92680829/169541435-7dcc795a-51f1-418b-8123-f1449d24d657.png" width="320">

    <br/>


    - 2) ```# hdfs dfs -ls /``` or ```# hadoop fs -ls /``` : shows current file system 
        -  ```hadoop fs [args]```
            - FS relates to a generic file system which can point to any file systems like local, HDFS etc. So this can be used when you are dealing with different file systems such as Local FS, (S)FTP, S3, and others
        -  ```hdfs dfs [args]``` ( ```hadoop dfs [arg]``` has been deprecated)
            - specific to HDFS. would work for operation relates to only HDFS.

    <br/>

    - 3) ```hdfs dfs -mkdir /[filename]``` : literally makes dir
        - check if file was made by using previous command ```# hdfs dfs -ls /```
        - <img src="https://user-images.githubusercontent.com/92680829/169547448-9e145587-9d93-42f4-aedf-81f7f850cc59.png" width="550">


<br/>

### - Access to Admin Web Page

<br/>

- you can access to admin webpage deamon and see current status of Hadoop cluster on your local PC
- open the web browser and access to url 'localhost:[port number]' 
    - use the port number that you set for port mapping when creating master node (namenode)
    - example : localhost:50070 

    <br/>

    <img width="546" alt="image" src="https://user-images.githubusercontent.com/92680829/169549317-014099a0-5436-4f74-9404-4a68e56162ac.png">
    <img width="643" alt="image" src="https://user-images.githubusercontent.com/92680829/169550206-517a13bc-2f3d-41af-ae93-a8f3e099c3bc.png">
    
<br/>

- main page shows the summary of HDFS memory usages
- you can also browse the directories of HDFS system through 'Utilities' tap
    <img width="794" alt="image" src="https://user-images.githubusercontent.com/92680829/169551172-f2b3d425-ac61-49b9-9819-ca9276faa6ed.png">