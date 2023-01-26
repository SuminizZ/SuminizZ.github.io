---
title : "[Cloudera] Installing CDH thorugh Docker # 1 - Prepare Base Docker Image" 
categories : 
    - Hadoop Ecosystem
tags : [Cloudera, CDH, Hadoop, Hadoop Ecosystem, docker, Docker]
toc : true
toc_sticky : true
---

## CDH (Cloudera Distribution for Hadoop Platform)
- Hadoop Ecosystem requires multiple components and each component has complex version dependency to another, which makes it really hard for customers to manage entire hadoop cluster by themselves. 
- CDH is the most widely deployed distribution of Apache Hadoop as an end-to-end management tool for Hadoop system that allows integrated control over all necessary hadoop components . 
- It provides automated installation process, reduced deployment cost, real-time view of nodes, central console to process across clusters, and other range of tools needed to operate Hadoop cluster<br/>

    <img src="https://user-images.githubusercontent.com/92680829/177539404-d2d17371-fccb-4b16-a705-e26c58b8273e.png" width="500">


- Today, we will use 7.1.4 version of cloudera (open source)
    - [<span style="color:blue">URL for cloudera manager 7.1.4 installer</span>](https://archive.cloudera.com/cm7/7.1.4/){:target="_blank"} 

- Alternatively, you can just download cloudera docker image with a brief command ```docker pull cloudera/quickstart:latest```

## Creates Docker Image with Cloudera Manager Installed

- **!) Before start**, if you're using Mac (Apple M1 Silicon), then you need to build base Docker image on AMd64 architecture environment instead of ARM64 architecture
    - Refer to [<span style="color:blue">HERE</span>](http://127.0.0.1:4000/trouble%20shooting/tblshooting_build_amd64_docker_image_on_mac/){:target="_blank"} instead of this post

- **Steps**
    1. Create CentOS base Image - **centos:base**        
    2. Install Cloudera Manager 7.1.4 on centos:base image - **centos:CM**
    3. Set Haddop Cluster with one namenode (Hadoop 01, cloudera manager server) and three datanodes (Hadoop 02~04, cloudera mangaer agent)  ---> this part will be addressed next time
    <img src="https://user-images.githubusercontent.com/92680829/177546575-78dd1642-4f9f-4a3c-8e9f-4123461af330.png" width="650">
    

### **1. Create CentOS Base Image**
- (local terminal) first, create a new container named as centos_base with CentOS image (version 7 here)
```bash
$ docker run -it --name centos_base -dt centos:7
```
- execute centos_base container and install all the necessary basic components
    ```bash
    ## terminal
    $ docker exec -it centos_base /bin/bash

    ## container 
    $ yum update
    $ yum install wget -y
    $ yum install vim -y
    $ yum install openssh-server openssh-clients openssh-askpass -y
    $ yum install initscripts -y
    $ ssh-keygen -t dsa -P "" -f ~/.ssh/id_dsa                       ## create dsa key file with passphrases "" (empty)
    $ cat ~/.ssh/id_dsa.pub >> ~/.ssh/authorized_keys                ## append public key file on authorized_keys file
    $ ssh-keygen -f /etc/ssh/ssh_host_rsa_key -t rsa -N ""           ## rsa key with passphrases "" (empty)
    $ ssh-keygen -f /etc/ssh/ssh_host_ecdsa_key -t ecdsa -N ""       ## ecdsa key with passphrases "" (empty)
    $ ssh-keygen -f /etc/ssh/ssh_host_ed25519_key -t ed25519 -N ""   ## ed25519 key with passphrases "" (empty)
    ```
    - **wget** : software package for interacting with REST APIs to retrieve files using HTTP, HTTPS, FTP and FTPS
    - **vim** : can edit files on terminal 
    - **openssh-server openssh-clients openssh-askpass** : connectivity tool for remote login between computers with the SSH protocol 
        - **SSH protocl (Secure Shell Protocol)**
            - network communication protocol that enables two remote computers to share data or perform operations on each other
            - Communication between remote computers is encrypted by a pair of keys (private and public) for authentication instead of using passwords, which allows safer interation between computers even on insecrue, public networks
            - ssh-keygen : generate private - public key pairs 
                - Command Options ([<span style="color:blue">For more informations, here</span>](http://man.openbsd.org/OpenBSD-current/man1/ssh-keygen.1#NAME){:target="_blank"})
                    - -t :  Specifies the type (algorithm) of key to create, default by rsa but alternatively you can use other algorithms such as DSA, ECDSA and ED25519
                    - -b : Specifies the number of bits in the key to create. The default length is 3072 bits (RSA) or 256 bits (ECDSA)
                    - -p : Requests changing the passphrase (phrases to protect private key files) of a private key file
                        - -P : old passphrase 
                        - -N : new passphrase
                    - -f : Specifies the filename of the key file.

- (container) edit the "bashrc" file using vim 
    ```bash
    $ vim ~/.bashrc

    ## bashrc file
    $ /usr/sbin/sshd

    $ source ~/.bashrc   ## by sourcing it, you can relaod the file and execute the commands placed in there
    ```
    - add the command to activate sshd and exit (writing mode start - :i / overwrite - :w / exit - :q)
    - **sshd** (OpenSSH server process)
        - receives incoming connections using the SSH protocol and acts as the server for the protocol. 
        - It handles user authentication, encryption, terminal connections, file transfers, and tunneling.


### **2. Download Cloudera Manager Installer on centos:base image and Give Access Permission**
- (container) now, let's download cloudera managaer installer file on container
    ```bash
    $ wget https://archive.cloudera.com/cm7/7.1.4/cloudera-manager-installer.bin
    ```
- (container) allow execute permission to installer file and exit
    ```bash
    $ chmod u+x cloudera-manager-installer.bin
    
    $ exit
    ```
    - **chmod** (change mode)
        - command used to change the access permissions (file mode)
        - With a set of options, you can specify the classes of users to whom the permissions are applied and the types of access allowed (which permissions are to be granted or removed)
            - -u : user, only for file owner
            - \+ : operator that indicates you want to add the permission following behind
            - -x : execute permission (recursive, includes all sub-directory)
            - [<span style="color:blue">For more informations, here</span>](https://en.wikipedia.org/wiki/Chmod#Special_modes){:target="_blank"}

- (local terminal) now, commit the container (centos_base) with all the files and packages set into a docker image **centos:base**
    ```bash
    $ docker commit centos_base centos:base 
    ```
    <img src="https://user-images.githubusercontent.com/92680829/177571380-453b8fa9-4884-43dc-b11f-07db743c8d59.png" width="700">
    - you can see new docker image named centos:base has just been created
    


## References
- [https://docs.cloudera.com/cloudera-manager/7.5.4/concepts/cm-concepts.pdf](https://docs.cloudera.com/cloudera-manager/7.5.4/concepts/cm-concepts.pdf){:target="_blank"}
- [https://taaewoo.tistory.com/23?category=917407](https://taaewoo.tistory.com/23?category=917407){:target="_blank"}