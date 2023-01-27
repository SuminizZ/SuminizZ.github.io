---
title : "[Hadoop] Apache Hadoop - HDFS & Map Reduce"
layout : post
categories : [de-hadoop]
img : de/hadoop.png
tags : [Hadoop, Hadoop Ecosystem, Apache]
toc : true
toc_sticky : true
---

<br/>

## **Apache Hadoop**

<br/>

- hadoop is a framework that allows us to store and process large data sets in parallel and distributed fashion
- HDFS : Distributed File System - **for data storage**
- Map Reduce : Parallel & distributed **data processing (MapReduce)**
- Master/Slave Architecture : HDFS, YARN
    - <img src="https://user-images.githubusercontent.com/92680829/159821647-1285cb07-c5af-4d62-8da6-257497afef94.png"  width="550">

---

<br/>

## **HDFS**

<br/>

1. NameNode : Master
    - controls and distributes resources to each datanode
    - receives all report from datanodes 
    - records metadata (**information about data** blocks e.g. location of files stored, size of files, permissions, hierarchy) 
2. DataNode : Slave 
    - store actual data
    - serves read & write requests from clients

3. Secondary NameNode
    - receive first copy of fsimage
    - serves **Checkpointing** : combines editlog + fsimage and create newly updated fsimage and send it to NameNode
        - editlog : contains records of modifications in data
        - editlog (new) : retains all new changes untill the next checkpoint happens and send it to editlog
        - fsimage : contains total informations about data
        - happens periodically (default 1 per 1h)
    - <img src="https://user-images.githubusercontent.com/92680829/159822617-0653a4aa-bbd2-44d5-896b-0833515a36d6.png"  width="380">
    - makes NameNode more available

<br/>

### **HDFS Data Blocks**

<br/>

- how the data is actually stored in datanodes?
    - Each file is stored on several HDFS blocks (default size is 128MB, but last node will only contain the remaining size)
    - These datablocks are distributed across all the dataNodes in your Hadoop cluster

<br/>

### **Advantages of distributing data file**

<br/>

- This kind of distributing file system is **highly scalable (flexible)** and efficient without making resources wasted
- Save your time by processing data in a parallel manner
- **Fault Tolerance** : How Hadoop Adress DataNode Failure?
    - suppose one of the DataNode containing the datablocks crashed 
    - to prevent data loss from defective DataNode, we should makes multiple copies of data
        - Solution : **Replication Factor** - how many replicas are created per 1 datablock
            - Each of DataNode has different sets of copies of datablocks (thrice by default)
            - Even if two of the copis become defective, we still have 1 left intact
            - <img src="https://user-images.githubusercontent.com/92680829/159824849-e72774d3-ae2a-4de2-bce7-5b159d3d2bfb.png"  width="430">

<br/>

### **HDFS Writing Mechanism**

<br/>

- Step 1. **Pipeline Setup**
    - ClientNode sends write request about Block A to NameNode
    - NameNode sent IP addresses for DN 1,4,6 where block A will be copied
    - First, CN asks DN1 to be ready to copy block A, and sequentially DN1 ask the same thing to DN4 and DN4 to DN6 
    - This is the Pipeline!
        - <img src="https://user-images.githubusercontent.com/92680829/159825946-55b6ae8f-7c57-44cf-9838-f45f2aa88b37.png"  width="500">

<br/>

- Step2 : **Data Streaming** (Actual Writing) 
    - As the pipeline has been created, the client will push the data into the pipeline 
    - Client will copy the block (A) to DataNode 1 only. 
    - The remaining replication process is always done by **DataNodes sequentially**.
        - DN1 will connect to DN4 and DN4 will copy the block and DN4 will connect to DN6 and DN6 will copy the block

<br/>

- Step3 : **Acknowledgement Reversely**
    - each DN will confirm that each step of replication succeed 
    - DN6 -> DN4 -> DN1 -> Client -> NameNode -> Update Metadata
    - <img src="https://user-images.githubusercontent.com/92680829/159828057-361d2462-598c-459b-9aa3-06591fc64a61.png"  width="500"> 

- **Summary for Multiple Blocks** 
    - <img src="https://user-images.githubusercontent.com/92680829/159828251-c8f41f6a-99ea-4425-a18d-6ca8f3ea64eb.png"  width="520">

<br/>

### **HDFS Reading Mechanism**

<br/>

- <img src="https://user-images.githubusercontent.com/92680829/159828597-816c7875-e6b1-468b-94ca-e465e3184214.png"  width="600">

---

<br/>

## **Map Reduce : Parallel & Distributed Processing** 

<br/>

- Single file is splited into multiple parts and each is processed by one DataNode simultaneously
- This system allows much faster processing
- Map Reduce can be divided into 2 Distinct Steps
    1. Map Tasks
    2. Reduce Tasks
    - Example : Overall MapReduce Word Counting Process
    - <img src="https://user-images.githubusercontent.com/92680829/159829821-c74ca6ec-2759-4089-a28d-e68b1b0118c8.png"  width="600">

- 3 Major Parts of MapReduce Program
    - Mapper Code : how map tasks will process the data to prdouce the key-value pair
    - Reducer Code : combine intermediate key-value pair generated by Mapper and give the final aggregated output
    - Driver Code : specify all the job configurations (job name, input path, output path, etc..)
    

<br/>

### **YARN (Yet Another Resource Negotiator) : for MapReduce Process**

<br/>

- Cluster management component of Hadoop
- It includes Resource Manager, Node Manager, Containers, and Application Master
    - **Resource Manager (NameMode)**
        - major component that manages application and job scheduling for the batch processes
        - allocates first container for AppMaster
    - **Slave Nodes (DataNode)** 
        - Node Manager : 
            - control task distribution for each DataNode in cluster
            - report node status to RM : monitors container's resource usage(memory, cpu..) and report to RM
        - AppMaster : 
            - coordinates and manages individual application
            - only run during the application, terminated as soon as MapReduce job is completed
            - resource request : negotiate the resources from RM 
            - works with NM to monitor and execute the tasks
        - Container :
            - allocates a set of resources (ram, cpu..) on a single DataNode 
            - report MapReduce status to AppMater
            - scheduled by RM, monitored by NM 


<br/>

- **YARN Architecture**

<br/>

    - <img src="https://user-images.githubusercontent.com/92680829/159842914-d0031950-45ff-488c-904c-df9fc425d11e.png"  width="600">


<br/>

### **Hadoop Architecture : HDFS (Storage) & YARN (MapReduce)**

<br/>

- <img src="https://user-images.githubusercontent.com/92680829/159844204-ead81820-97a7-4d41-9216-a1d637544e6a.png"  width="500">

---

<br/>

## **Hadoop Cluster**

<br/>

- <img src="https://user-images.githubusercontent.com/92680829/159844804-52c3292f-0898-4591-a96a-be80977fc0ac.png"  width="500">


<br/>

### **Hadoop Cluster Modes**

<br/>

- Multi-Node Cluster Mode
    - one cluster cosists of multiple nodes
- Pseudo-Distributed Mode
- Standalone Mode 

<br/>

## **Hadoop Ecosystem**
- <img src="https://user-images.githubusercontent.com/92680829/159845221-238495fc-7b93-4183-87d5-efa7b3b46812.png"  width="600">
