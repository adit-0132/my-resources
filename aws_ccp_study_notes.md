# ☁️ AWS Certified Cloud Practitioner — High-Yield Study Notes

Other Resources that I used : 

[Fast Notes](https://github.com/Furkan-Gulsen/aws-certified-cloud-practitioner-certification-my-notes) for final day revision.
[Detailed Notes](https://github.com/kennethleungty/AWS-Certified-Cloud-Practitioner-Notes/tree/main) for those who wanna be 100% sure of their prep.

> Compiled from 23 Practice Exams | Optimized for Last-Day Revision --> Most Goated {I got around 50 questions from what's covered beneath}

---

## Table of Contents
1. [AWS Cloud Fundamentals & Value Proposition](#1-aws-cloud-fundamentals--value-proposition)
2. [AWS Global Infrastructure](#2-aws-global-infrastructure)
3. [Shared Responsibility Model](#3-shared-responsibility-model)
4. [IAM — Identity & Access Management](#4-iam--identity--access-management)
5. [EC2 — Elastic Compute Cloud](#5-ec2--elastic-compute-cloud)
6. [Storage Services](#6-storage-services)
7. [Database Services](#7-database-services)
8. [Networking & Content Delivery](#8-networking--content-delivery)
9. [Security Services](#9-security-services)
10. [Monitoring & Management](#10-monitoring--management)
11. [Cost Management & Billing](#11-cost-management--billing)
12. [AWS Support Plans](#12-aws-support-plans)
13. [Deployment, Automation & Developer Tools](#13-deployment-automation--developer-tools)
14. [Serverless & Containers](#14-serverless--containers)
15. [Migration Services](#15-migration-services)
16. [AWS Organizations & Governance](#16-aws-organizations--governance)
17. [Well-Architected Framework & Design Principles](#17-well-architected-framework--design-principles)

---

## 1. AWS Cloud Fundamentals & Value Proposition

### Key Concepts

**Cloud Computing Models**
- **IaaS (Infrastructure as a Service):** Customer manages OS, middleware, apps. AWS manages hardware, networking, virtualization. Example: Amazon EC2.
- **PaaS (Platform as a Service):** Customer manages only applications and data. Example: AWS Elastic Beanstalk.
- **SaaS (Software as a Service):** Vendor manages everything. Example: Gmail, Salesforce on AWS.
- ⚠️ **NaaS (Networking as a Service) does NOT exist** in AWS cloud computing models — it's a trap answer.

**Core Benefits of AWS Cloud**
- **Trade Capital Expense (CapEx) for Variable Expense (OpEx):** Pay only for what you use; no upfront hardware costs.
- **Economies of Scale:** AWS aggregates usage from millions of customers → lower variable costs for everyone. This is why AWS can offer lower pricing than running your own data center.
- **Stop Guessing Capacity:** Provision as much or as little as needed; scale on demand.
- **Increase Speed & Agility:** Resources available in minutes, not weeks. Developers can experiment faster.
- **Go Global in Minutes:** Deploy to any AWS Region worldwide with a few clicks.
- **No Data Center Maintenance:** Stop spending money on running and maintaining data centers.

**Pay-as-you-go Pricing**
- Linux EC2 instances are billed in **1-second increments** with a **minimum of 1 minute**.
- If you run an instance for 3 hours, 5 minutes, and 6 seconds → you're billed for **exactly that time**.
- There is **no charge when Lambda code is not running**.

### 💡 Exam Tips
- "Economies of scale" = AWS buys hardware in bulk → passes savings to customers. This is different from "pay-as-you-go."
- Agility = **speed of provisioning** (minutes vs. weeks on-premises) — not multi-region deployment or lower cost.
- "No long-term contract required" is a value proposition of AWS (vs traditional data centers with multi-year commitments).
- AWS has **lower variable costs AND lower upfront costs** compared to traditional data centers.

---

## 2. AWS Global Infrastructure

### Key Concepts

**Regions**
- A Region is a geographic area containing multiple, isolated Availability Zones (e.g., `us-east-1`, `ap-northeast-1`).
- Regions are completely isolated from each other for maximum fault tolerance and stability.
- To reduce latency for international users → **deploy EC2 instances in a Region closest to them**.
- Highest availability = deploy across **multiple Regions AND multiple Availability Zones**.

**Availability Zones (AZs)**
- Each Region has **2 or more** AZs (typically 3–6).
- AZs are physically separate data centers within a Region, connected by low-latency fiber.
- Purpose: **build resilient, highly available architectures** — if one AZ fails, another handles traffic.
- Multi-AZ = High Availability. Multi-Region = Highest Redundancy + Fault Tolerance.

**Edge Locations**
- Separate from Regions/AZs — there are **many more** Edge Locations than Regions.
- Used by **Amazon CloudFront (CDN)** to cache and serve content closer to end users.
- Also used by **AWS Shield** for DDoS protection at the edge.
- Edge Locations **do NOT host EC2 instances** — they only cache content and reduce latency.

**Global vs. Regional Services**

| Scope | Examples |
|-------|----------|
| **Global** | IAM, Amazon Route 53, Amazon CloudFront, AWS WAF |
| **Regional** | Amazon EC2, Amazon S3*, Amazon RDS, Amazon VPC |

> *S3 buckets are regional but data is automatically replicated across multiple AZs within that region.

### 💡 Exam Tips
- Edge Locations ≠ Regions ≠ AZs. Know what lives where.
- "Deploying across multiple AZs and Edge Locations" is a **wrong answer** for high availability — Edge Locations don't host applications.
- The correct answer for maximum availability is always **multiple Regions + multiple AZs**.
- `status.aws.amazon.com` → AWS Service Health Dashboard (shows status of ALL services in ALL regions).

---

## 3. Shared Responsibility Model

### Key Concepts

**The Core Principle**
AWS is responsible for **"security OF the cloud"** (the infrastructure). Customers are responsible for **"security IN the cloud"** (what they deploy on it). Responsibility **varies depending on the service used** — this is the single most important concept about this model.

**AWS Responsibilities (Always AWS)**
- Physical security of data centers
- Hardware, networking, and virtualization infrastructure
- Securing Regions, AZs, and Edge Locations
- Decommissioning old hardware (AWS destroys old storage devices per industry-standard practices)
- Securing the EC2 **hypervisor layer**
- Patching/maintaining **managed service** infrastructure (RDS, DynamoDB, Lambda, etc.)

**Customer Responsibilities (Always Customer)**
- IAM user/role/policy configuration
- Data encryption (client-side and server-side)
- Security group and NACL configuration (firewall rules)
- Patching guest OS on EC2 instances
- Application security and data management
- User security training and awareness

**Shared / Varies by Service**

| Area | AWS Responsibility | Customer Responsibility |
|------|--------------------|------------------------|
| **EC2 (IaaS)** | Hypervisor, hardware, AZ infrastructure | OS patching, app security, network config |
| **RDS (Managed)** | OS patching, DB engine, scaling | DB schema, data, access control, encryption settings |
| **Lambda (Serverless)** | Capacity management, OS maintenance, runtime | Code, access control, data |
| **DynamoDB (Managed)** | All infrastructure + scaling | Data, access policies |
| **Patch Management** | AWS patches infrastructure devices | Customer patches their EC2 OS and apps |
| **Configuration Mgmt** | AWS configures infrastructure devices | Customer configures OS, DB, apps |
| **Training** | AWS trains AWS employees | Customer trains their own employees |

**Inherited Controls (Customer Inherits from AWS)**
- Physical security controls (e.g., data center access restrictions)
- Environmental controls (e.g., power, HVAC, fire suppression)

### 💡 Exam Tips
- "Security of the IaaS services is AWS's responsibility" → **FALSE**. For IaaS (EC2), customers are responsible for OS and above.
- "Patching the guest OS is always AWS's responsibility" → **FALSE**. Guest OS = customer. AWS patches the hypervisor.
- "Scaling a DynamoDB database" → **AWS** (it's a fully managed service).
- **Patch management is a SHARED control** — AWS patches infrastructure, customer patches their OS/apps.
- Physical controls and environmental controls are fully inherited by the customer from AWS.

---

## 4. IAM — Identity & Access Management

### Key Concepts

**Core IAM Components**
- **IAM User:** A person or application with long-term credentials (username/password or access keys).
- **IAM Group:** A collection of users that all share the same permissions policy. Used to organize teams and assign permissions at scale.
- **IAM Role:** An entity that defines permissions for **temporary access** — assigned to AWS services, applications, or federated users. No long-term credentials.
- **IAM Policy:** A JSON document that defines allowed/denied actions on AWS resources. Can be attached to users, groups, or roles.

**Key IAM Best Practices**
- **Principle of Least Privilege:** Grant only the permissions users need, and nothing more. Apply policies only to users who require them.
- **Enable MFA (Multi-Factor Authentication):** Adds a second layer of verification. Supported for root and IAM users.
- **Use IAM Roles instead of long-term credentials** for applications and services.
- **Never use root account** for day-to-day tasks — only for initial setup.
- **Rotate access keys regularly** — AWS recommends NOT embedding access keys in code.
- **Use IAM Groups** to assign permissions to teams rather than individual users.

**Access Methods**
- **AWS Management Console:** Username + password (+ MFA optionally)
- **AWS CLI:** Requires **Access Keys** (Access Key ID + Secret Access Key)
- **AWS SDK:** Uses Access Keys programmatically from code

**IAM Credential Report**
- Downloadable report listing all IAM users and the status of their credentials (password age, MFA status, access key age, last used).
- Useful for **auditors** — simplest way to provide a compliance snapshot of all users.

**Federation**
- Allows users to sign in using **existing corporate credentials** (e.g., Active Directory, Google, SAML 2.0).
- Eliminates the need to create separate IAM accounts for every employee.

**Service Control Policies (SCPs)**
- Used within **AWS Organizations** to restrict what services and actions are allowed across entire accounts or OUs.
- SCPs are NOT IAM policies — they act as guardrails on top of IAM.

### 💡 Exam Tips
- **Groups vs. Organizations:** IAM Groups organize users within ONE account. AWS Organizations manages MULTIPLE accounts.
- AWS CLI needs **Access Keys**, not username/password.
- To check all users and credential status → use **IAM Credential Report** (not CloudTrail, not Trusted Advisor).
- IAM Role = **temporary access** with no long-term credentials. Perfect for EC2 instances accessing S3.
- **No permissions by default** — IAM users start with zero permissions and need explicit grants.
- MFA + strong password policies are both customer responsibilities under the shared model.

---

## 5. EC2 — Elastic Compute Cloud

### Key Concepts

**Scaling Types**
- **Vertical Scaling (Scale Up):** Replacing an instance with a bigger, more powerful one (more RAM, CPU). Requires downtime.
- **Horizontal Scaling (Scale Out):** Adding more instances of the same size to handle increased traffic. **This is the AWS-recommended approach.**

**EC2 Pricing Models**

| Model | Best For | Key Detail |
|-------|----------|------------|
| **On-Demand** | Short-term, unpredictable workloads; cannot be interrupted | Pay by second (Linux), no commitment |
| **Reserved Instances (RI)** | Steady-state, predictable workloads over 1–3 years | Up to 72% discount vs On-Demand; minimum 1-year commitment |
| **Spot Instances** | Fault-tolerant, flexible, interruptible workloads | Up to 90% discount; AWS can reclaim with 2-min notice |
| **Dedicated Instances** | Compliance needs; hardware not shared with other AWS customers | Higher cost; physical isolation at instance level |
| **Dedicated Hosts** | Bring Your Own License (BYOL); full server control | Most expensive; physical server dedicated to you |

**Choosing the Right Pricing Model — Common Exam Scenarios**

| Scenario | Answer |
|----------|--------|
| Short-term spike, cannot be interrupted | **On-Demand** |
| 3-year steady workload, minimize cost | **Reserved Instances** |
| Batch jobs, image processing, interruptible | **Spot Instances** |
| Need hardware not shared with others | **Dedicated Instances** |
| 2-month always-available workload | **On-Demand** (RIs need 1+ year; Spot can be interrupted) |
| Media transcoding, recovers quickly from failure | **Spot Instances** |

**AMI (Amazon Machine Image)**
- A pre-configured template used to launch EC2 instances.
- To replicate a configured instance → **create an AMI** from it, then launch new instances from that AMI.
- EBS Snapshot ≠ AMI: Snapshots are point-in-time backups of volumes; AMIs are full machine templates.

**EC2 Pricing Factors**
- Instance type (CPU, memory, storage configuration)
- Availability Zone where the instance is provisioned
- Purchasing model (On-Demand, Reserved, Spot)
- NOT affected by: number of S3 buckets, number of private IPs

**EC2 Billing**
- Linux: billed in **1-second increments**, minimum of **1 minute**
- You pay exactly for time used — 3h 5m 6s = billed for 3h 5m 6s

**Auto Scaling**
- Automatically adjusts the number of EC2 instances based on demand.
- Works across multiple **Availability Zones** (not multiple Regions automatically).
- Replaces unhealthy instances automatically.
- Key for both **elasticity** and **high availability**.

**Elastic Load Balancer (ELB)**
- Distributes incoming traffic across healthy targets (EC2 instances).
- **Improves reliability** by ensuring only healthy targets receive traffic.
- Works across multiple AZs.
- Types: Application Load Balancer (Layer 7/HTTP), Network Load Balancer (Layer 4/TCP), Classic.

### 💡 Exam Tips
- "Adding more RAM to an EC2 instance" = **vertical scaling** (wrong answer for horizontal).
- Spot Instances are NOT suitable for workloads that cannot be interrupted — use On-Demand.
- Reserved Instances require **at least a 1-year commitment** — a benefit AND a drawback.
- Reserved Instances discounts can be **shared across accounts** via consolidated billing (not just the master account).
- Dedicated Instances ≠ Dedicated Hosts: Dedicated Instances ensure no hardware sharing; Dedicated Hosts give full control of a physical server.
- EC2 Auto Scaling adds instances across AZs, but **not across Regions automatically**.

---

## 6. Storage Services

### Key Concepts

**Amazon S3 (Simple Storage Service)**
- Object storage — stores files (objects) in **buckets**.
- Automatically replicates data across **multiple AZs** within a region for durability (11 nines = 99.999999999%).
- Use cases: static website hosting, backup, media storage for CloudFront, log storage.
- **NOT** for hosting dynamic websites (no server-side scripting like PHP).
- Access control: via **IAM policies** and **S3 Bucket Policies**.
- **S3 Versioning:** Protects against accidental deletion by storing all object versions.
- **S3 Cross-Region Replication (CRR):** Replicates objects to another Region — for compliance or latency, not for accidental deletion protection.
- **S3 Transfer Acceleration:** Speeds up uploads to S3 using CloudFront edge network. For large file uploads.

**S3 Storage Classes**

| Class | Use Case | Retrieval |
|-------|----------|-----------|
| **S3 Standard** | Frequently accessed data | Instant |
| **S3 Intelligent-Tiering** | Unknown or changing access patterns | Instant (auto-moves between tiers) |
| **S3 Standard-IA** | Infrequent access, rapid retrieval | Instant |
| **S3 Glacier** | Archives, long-term storage; 5-hour retrieval | 1–5 hours |
| **S3 Glacier Deep Archive** | Compliance archives rarely accessed; lowest cost | Up to 12 hours |

> **Exam pattern:** "Compliance archive, only needed in legal/regulatory breach" → **S3 Glacier Deep Archive**. "Need within 5 hours" → **S3 Glacier**.

**Amazon EBS (Elastic Block Store)**
- Block storage attached to a single EC2 instance.
- Persists independently of the EC2 instance lifecycle.
- Data safety: **create EBS Snapshots** (stored in S3) + **encrypt at rest**.
- Not suitable for archival or shared access across multiple instances.

**Amazon EFS (Elastic File System)**
- Managed NFS (Network File System) — can be mounted by **multiple EC2 instances simultaneously**.
- Automatically scales up/down based on usage.

**Storage Quick Comparison**

| Feature | S3 | EBS | EFS |
|---------|-----|-----|-----|
| Type | Object | Block | File |
| Access | Any app/internet | Single EC2 | Multiple EC2s |
| Scales automatically | Yes | No (provisioned) | Yes |
| Best for | Backup, media, static sites | OS volumes, databases | Shared file access |
| AZ-level | Multi-AZ (auto) | Single AZ | Multi-AZ |

**AWS Snowball & Snowmobile (Data Transfer)**
- **Snowball:** Physical device for secure transfer of large amounts of data into/out of AWS. Also has built-in compute to process data locally before transfer.
- **Snowmobile:** A literal shipping container on a truck — for **Exabyte-scale** data migration (think 60+ Petabytes). Not for general use.
- **S3 Transfer Acceleration:** For uploading large files to S3 over the internet faster, using CloudFront edge.

**AWS Storage Gateway**
- Connects on-premises environments to AWS cloud storage seamlessly.
- Enables a **hybrid storage** architecture — on-premises apps access cloud storage as if it were local.

### 💡 Exam Tips
- S3 Versioning → protection from **accidental deletion**. S3 CRR → cross-region replication (different purpose).
- "60 Petabytes of data" → **Snowmobile**. "Large amounts" without Exabyte scale → **Snowball**.
- EBS is for a **single EC2 instance**. EFS is for **multiple EC2 instances sharing** the same file system.
- S3 automatically replicates across AZs — you do NOT need to configure this.
- S3 can host **static websites** only; dynamic websites need EC2.
- Storing most-frequently accessed data for a web app → **ElastiCache** (not EBS, not S3).

---

## 7. Database Services

### Key Concepts

**Amazon RDS (Relational Database Service)**
- Fully managed relational database. AWS handles patching, backups, and scaling infrastructure.
- Supported engines: **MySQL, PostgreSQL, Oracle, Microsoft SQL Server, MariaDB, Amazon Aurora**.
- NOT supported: **Teradata** (common trap answer).
- Benefits: lower administrative burden, resizable compute capacity.
- **NOT** for document/key-value data structures (that's DynamoDB).
- Supports **ACID transactions** — use for financial applications requiring data consistency.
- **Multi-AZ:** AWS automatically replicates to a standby instance in another AZ for high availability.
- **Read Replicas:** Read-only copies of the database to offload read traffic.

**Amazon Aurora**
- AWS's own MySQL/PostgreSQL-compatible relational database.
- **Scales storage capacity automatically** based on demand — no manual intervention.
- Up to 5x faster than standard MySQL on RDS.
- Managed by AWS.

**Amazon DynamoDB**
- Fully managed **NoSQL** database (key-value and document store).
- Automatically replicates across **multiple AZs** — highly available by default.
- Use when your application needs a NoSQL/key-value/document data model.
- AWS is responsible for scaling and operations.

**Amazon Redshift**
- Fully managed **data warehouse** service.
- Designed for **analytics and BI** on large datasets (petabyte scale).
- Use for: running complex queries on historical data, business intelligence.
- Not for transactional workloads — use RDS for OLTP.

**Amazon ElastiCache**
- Managed **in-memory caching** service (Redis or Memcached).
- Use for: storing frequently accessed data to reduce database load and latency.
- "Store most-frequently accessed data for optimal response time" → ElastiCache.

**Amazon DMS (Database Migration Service)**
- Migrates databases to AWS **without interrupting the source database**.
- Supports homogeneous migrations (MySQL to MySQL) and heterogeneous (Oracle to Aurora).

**Other DB Services**
- **Amazon Neptune:** Managed graph database.
- **Amazon CloudHSM:** Dedicated Hardware Security Module — for generating/managing your own encryption keys with full control. Not a database.

**DB Service Selection Guide**

| Need | Service |
|------|---------|
| Relational + ACID | RDS or Aurora |
| NoSQL key-value | DynamoDB |
| Data warehouse / analytics | Redshift |
| In-memory caching | ElastiCache |
| Graph database | Neptune |
| Migrate databases to AWS | DMS |
| Run custom DB software | EC2 (install it yourself) |

### 💡 Exam Tips
- Teradata is NOT an RDS-supported engine — memorize: MySQL, PostgreSQL, Oracle, SQL Server, MariaDB, Aurora.
- Redshift ≠ RDS: Redshift is for analytics (OLAP); RDS is for transactions (OLTP).
- DynamoDB automatically replicates across AZs — AWS handles all scaling.
- "Install and run custom relational database software" → **Amazon EC2** (not RDS, which is managed).
- ElastiCache is the answer for "reduce latency for frequently accessed data."
- "ACID transactions for a financial app" → **RDS** (not DynamoDB, which has eventual consistency by default).

---

## 8. Networking & Content Delivery

### Key Concepts

**Amazon VPC (Virtual Private Cloud)**
- A logically isolated section of the AWS cloud where you can launch resources in a defined virtual network.
- You control IP address ranges, subnets, route tables, gateways.
- **Security Groups:** Virtual firewalls at the **instance level**. Stateful (return traffic automatically allowed). Controls inbound/outbound traffic by IP/port.
- **Network ACLs (NACLs):** Firewalls at the **subnet level**. Stateless (must explicitly allow inbound AND outbound). Can be used to block IP address ranges.
- Both Security Groups and NACLs are used to **control network traffic in AWS**.

**Amazon Route 53**
- AWS's **DNS (Domain Name System)** service.
- Also used to **register domain names**.
- Supports **latency-based routing** to direct users to the closest Region.
- Route 53 is a **Global** service (not tied to a single Region).

**Amazon CloudFront (CDN)**
- A global **Content Delivery Network** that caches content at **Edge Locations** worldwide.
- Reduces latency by serving content from the nearest edge location to the user.
- Use cases: delivering static/dynamic web content, streaming video, delivering images/videos globally.
- **CloudFront cost factors:** Number of requests + Traffic distribution (geographic). NOT instance type or storage class.
- Integrates with S3, EC2, ELB as origins.
- Also provides **DDoS mitigation** at the edge via integration with AWS Shield.

**AWS Direct Connect**
- A **dedicated private network connection** from your on-premises data center to AWS.
- Uses industry-standard 802.1q VLANs.
- More reliable and consistent than VPN (no internet variability).
- For hybrid architectures requiring consistent, high-bandwidth connectivity.

**AWS Site-to-Site VPN**
- Encrypted connection over the **internet** using **IPSec** protocol between on-premises and AWS VPC.
- Uses a Virtual Private Gateway on the AWS side.
- Less expensive than Direct Connect but subject to internet variability.

**Direct Connect vs. VPN**

| Feature | Direct Connect | Site-to-Site VPN |
|---------|---------------|-----------------|
| Connection | Dedicated private line | Encrypted tunnel over internet |
| Protocol | 802.1q | IPSec |
| Latency | Consistent, low | Variable |
| Cost | Higher | Lower |
| Setup time | Weeks | Hours/days |

**AWS Transit Gateway**
- Hub-and-spoke network architecture to connect **thousands of VPCs** across multiple accounts and Regions.
- Simplifies complex VPC peering meshes — instead of N×(N-1)/2 peering connections, use one Transit Gateway.
- Use when: managing hundreds/thousands of VPCs across multiple accounts.

**VPC Peering**
- Direct connection between two VPCs. Works within and across Regions.
- Does NOT scale well for large numbers of VPCs (use Transit Gateway instead).

**Elastic Load Balancer (ELB)**
- Distributes traffic across healthy EC2 instances in multiple AZs.
- **Application Load Balancer (ALB):** Layer 7, HTTP/HTTPS routing. Best for web apps.
- **Network Load Balancer (NLB):** Layer 4, TCP/UDP. Ultra-high performance.
- Improves availability by routing only to healthy targets.

**AWS Global Accelerator**
- Routes traffic through AWS's global network to the nearest AWS endpoint.
- Improves performance for non-HTTP use cases (gaming, IoT, voice).
- Different from CloudFront (which is a CDN for caching content).

### 💡 Exam Tips
- CloudFront uses **Edge Locations**; EC2 and RDS use **AZs/Regions**. Edge Locations ≠ data centers for compute.
- Route 53 = DNS + domain registration + latency-based routing. It's **Global**, not regional.
- For thousands of VPCs → **Transit Gateway** (not VPC Peering, which doesn't scale).
- Direct Connect = private dedicated line (no internet). VPN = encrypted tunnel over the internet (uses IPSec).
- Security Groups are **stateful**; NACLs are **stateless** — this distinction is frequently tested.
- NACLs operate at **subnet level**; Security Groups at **instance level**.
- Both WAF and CloudFront have DDoS-related features. Shield is the dedicated DDoS service.

---

## 9. Security Services

### Key Concepts

**AWS Shield**
- Managed **DDoS (Distributed Denial of Service) protection** service.
- **Standard:** Free, automatic protection for all AWS customers against common DDoS attacks.
- **Advanced:** Paid tier with enhanced detection, 24/7 DDoS Response Team (DRT), and financial protection.
- Always-on detection and automatic inline mitigation.

**AWS WAF (Web Application Firewall)**
- Protects web applications from common exploits: **SQL injection, cross-site scripting (XSS)**, and other Layer 7 attacks.
- Can create rules to block specific IP addresses, geographic regions, or patterns.
- Integrates with CloudFront, ALB, API Gateway.
- WAF is for **application-layer attacks**; Shield is for **network-layer DDoS**.

**Amazon GuardDuty**
- Managed **threat detection** service that continuously monitors for malicious activity.
- Uses ML to analyze CloudTrail logs, VPC Flow Logs, and DNS logs.
- Detects: compromised credentials, unusual API calls, communication with malicious IPs.

**Amazon Inspector**
- Automated **security assessment** service for EC2 instances and container images.
- Checks for software vulnerabilities and unintended network exposure.
- Produces a detailed security findings report with severity ratings.

**AWS KMS (Key Management Service)**
- Managed service to **create, manage, and control encryption keys**.
- Integrated with most AWS services (S3, EBS, RDS, etc.).
- Keys are managed by AWS but controlled by the customer.

**AWS CloudHSM (Hardware Security Module)**
- **Dedicated hardware** device in AWS's data center where you can generate and manage your own encryption keys.
- Gives you **full control** over key management (you own the keys exclusively).
- More expensive than KMS; used for compliance requiring dedicated HSM.

**KMS vs. CloudHSM**

| Feature | AWS KMS | AWS CloudHSM |
|---------|---------|-------------|
| Key management | AWS-managed service | Customer-managed on dedicated hardware |
| Control | Shared | Full customer control |
| Cost | Lower | Higher |
| Compliance | Standard | FIPS 140-2 Level 3 |

**Amazon Macie**
- Uses ML to **discover and protect sensitive data** in S3 (PII, financial data).
- Alerts you to sensitive data exposures.

**AWS Artifact**
- A self-service portal for **on-demand access to AWS compliance reports and agreements**.
- Includes SOC reports, PCI DSS reports, ISO certifications, and agreements like BAA (HIPAA).
- Where auditors go to get AWS's compliance documentation.
- Also where you manage your **agreements with AWS**.

**AWS Certificate Manager (ACM)**
- Provision, manage, and deploy **SSL/TLS certificates** for HTTPS.
- Works with CloudFront, ALB, API Gateway.
- Route 53 can also deploy SSL certificates (for domains registered through Route 53).

**AWS Secrets Manager**
- Stores, rotates, and retrieves secrets (database credentials, API keys, passwords) securely.

**AWS Acceptable Use Policy**
- Document describing **prohibited uses** of AWS services — available publicly.
- Not a service — it's a policy document.

**DDoS Protection Summary**

| Layer | Service |
|-------|---------|
| Network/Transport (L3/L4) | AWS Shield |
| Application (L7) | AWS WAF |
| Infrastructure-level for EC2 | Security Groups + NACLs |

### 💡 Exam Tips
- **WAF** = SQL injection, XSS, application-layer attacks. **Shield** = DDoS (volumetric/network attacks).
- "Malicious use of AWS resources" → contact **AWS Abuse team** (even on Basic support).
- AWS **Artifact** = compliance documents and agreements. **Inspector** = security assessment of EC2/containers.
- **GuardDuty** = threat intelligence/detection. **Inspector** = vulnerability scanning.
- CloudHSM = you generate and manage your OWN keys on dedicated hardware.
- Free security resources available to everyone: AWS Security Blog, AWS Security Bulletins.
- Security Groups and NACLs both protect EC2 from DDoS — by filtering traffic at instance/subnet level.

---

## 10. Monitoring & Management

### Key Concepts

**Amazon CloudWatch**
- Monitoring service for AWS resources and applications.
- Collects metrics (CPU utilization, memory, disk, network), logs, and events.
- **CloudWatch Alarms:** Trigger actions based on metric thresholds (e.g., billing alarm when cost exceeds $2,000 → SNS notification).
- **Basic monitoring:** Metrics every **5 minutes**. **Detailed monitoring:** Every **1 minute** (additional cost).
- Use for: monitoring EC2 CPU/memory, setting billing alerts, triggering Auto Scaling.

**AWS CloudTrail**
- Records **all API calls made to AWS services** — who did what, when, and from where.
- Answers: "Who terminated those EC2 instances?" → CloudTrail.
- Logs include: caller identity, time of call, source IP, request parameters, response.
- Enabled by default for 90 days; create a Trail to store logs longer in S3.
- Use for: security auditing, compliance, troubleshooting, governance.

**AWS Config**
- Tracks and records **configuration changes** to AWS resources over time.
- Answers: "What did this resource look like at this point in time?" and "Does it comply with our rules?"
- **Config Rules:** Continuously evaluate resource configurations against desired settings.
- Use for: compliance auditing, resource configuration history, change tracking.

**CloudWatch vs. CloudTrail vs. Config**

| Service | What it answers | Use case |
|---------|-----------------|----------|
| **CloudWatch** | "How is my resource performing right now?" | Monitoring metrics, setting alarms |
| **CloudTrail** | "Who did what action and when?" | API auditing, security investigation |
| **AWS Config** | "What is/was the configuration of this resource?" | Compliance, configuration change tracking |

**AWS Trusted Advisor**
- Automated tool that analyzes your AWS environment and provides recommendations.
- Five categories: **Cost Optimization, Performance, Security, Fault Tolerance, Service Limits**.
- "Simplest way to check if security groups have unrestricted access" → **Trusted Advisor**.
- **Basic & Developer plans:** Access to **7 core checks** only (security + service limits).
- **Business & Enterprise plans:** Access to **all checks**.

**AWS Systems Manager**
- Provides operational insights and management of AWS resources at scale.
- Allows patching, automating tasks, and running commands across EC2 instances.

**AWS Health Dashboard (Personal Health Dashboard)**
- Shows alerts and notifications specific to **your AWS resources** when AWS events may affect them.
- Different from Service Health Dashboard (which is for all AWS services globally).

### 💡 Exam Tips
- **CloudTrail = API audit log** — use it anytime a question involves "who did/changed/deleted something."
- **CloudWatch = metrics and alarms** — use it for performance monitoring and billing alerts.
- **AWS Config = configuration compliance** — use it to track changes to resource configurations.
- Billing alerts can be set via: **CloudWatch Billing Alarm** + **SNS** OR **AWS Budgets**.
- CloudTrail does NOT delete resources when a threshold is exceeded (common trap answer).
- **Trusted Advisor** is the "simplest" way to review security group settings — not IAM console.
- Basic/Developer → 7 core Trusted Advisor checks. Business/Enterprise → all checks.

---

## 11. Cost Management & Billing

### Key Concepts

**AWS Pricing Fundamentals**
- Three fundamental drivers of cost in AWS: **Compute, Storage, Data Transfer (outbound)**.
- Inbound data transfer is generally **free**.
- Pay-as-you-go: no upfront costs, no termination fees (except Reserved Instances).

**Cost Optimization Tools**

| Tool | Purpose |
|------|---------|
| **AWS Cost Explorer** | Visualize, understand, and manage your costs and usage over time. View distribution of spending. |
| **AWS Budgets** | Set custom cost or usage budgets and receive alerts when thresholds are exceeded. |
| **AWS Cost & Usage Report (CUR)** | Most detailed billing report available. Shows past usage and costs. |
| **AWS Pricing Calculator** | Estimate costs before deploying. NOT for viewing past bills. |
| **Cost Allocation Tags** | Categorize costs by project, team, or department for chargeback/showback. |

**Consolidated Billing (via AWS Organizations)**
- One bill for multiple AWS accounts → simplified billing management.
- **Volume discounts:** Usage across all accounts is aggregated → potentially lower pricing tiers.
- **Reserved Instance sharing:** RIs purchased by one account can be used by others in the same organization.
- Consolidated billing itself is **free**.
- Does NOT give you 5× free tier; does NOT halve costs automatically.

**Reserved Instance Cost Sharing**
- When one account in an organization buys RIs, **all accounts** in the organization can receive the hourly cost benefit of those RIs (if their usage matches).
- This is a key advantage of consolidated billing beyond just getting one bill.

**TCO (Total Cost of Ownership) Analysis**
- When comparing on-premises vs. AWS costs, include: physical hardware, power, cooling, data center space, IT staff, maintenance.
- AWS eliminates most of these → significantly lower TCO.
- Software development and business analysis are NOT considered in TCO analysis.

**Cost Allocation Tags**
- Tags (key-value pairs) applied to AWS resources.
- Enable tracking costs by project, department, environment (dev/prod/test), etc.
- Must be activated in the Billing Console to appear in Cost & Usage Reports.

**AWS Savings Plans**
- Flexible pricing model offering lower prices in exchange for a consistent usage commitment ($/hour for 1 or 3 years).
- More flexible than RIs — applies across instance families, sizes, and Regions.

**CloudFront Cost Factors**
- Number of HTTP requests served
- Traffic distribution (geographic — data transfer rates differ by region)

### 💡 Exam Tips
- **Cost Explorer** = view and analyze past/current costs. **Pricing Calculator** = estimate future costs.
- **AWS Budgets** = proactive alerts when spending exceeds threshold.
- **CUR (Cost & Usage Report)** = most granular billing data; used for the past month's EC2 billing.
- Consolidated billing = **volume discounts + single bill** (not halved costs, not 5× free tier).
- "Track costs by department" → **Cost Allocation Tags**.
- Reserved Instances shared across accounts = via **Consolidated Billing** (not Cost Explorer, not Budgets).
- TCO analysis includes **physical hardware** costs — this is what makes on-premises expensive.
- For variable/unpredictable usage, AWS is more economical because you **launch and terminate based on demand**.

---

## 12. AWS Support Plans

### Key Concepts

**Support Plan Comparison**

| Feature | Basic | Developer | Business | Enterprise |
|---------|-------|-----------|----------|------------|
| Price | Free | $29/mo min | $100/mo min | $15,000/mo min |
| Tech support | None | Business hours (email) | 24/7 (phone + chat) | 24/7 (phone + chat) |
| Trusted Advisor checks | 7 core only | 7 core only | All checks | All checks |
| TAM | ❌ | ❌ | ❌ | ✅ |
| Support Concierge | ❌ | ❌ | ❌ | ✅ |
| Infra Event Management | ❌ | ❌ | Additional fee | ✅ Included |
| Response time (critical) | N/A | 12 business hrs | 1 hour | 15 minutes |

**Key Roles**

- **Technical Account Manager (TAM):** Primary point of contact for ongoing support needs — **Enterprise plan only**. Provides proactive guidance.
- **AWS Support Concierge:** Helps with **billing and account inquiries** — **Enterprise plan only**. Directs you to the right resources.
- **Infrastructure Event Management (IEM):** Short-term engagement for major launches — included in **Enterprise**; available for Business for additional cost.

**Accessing Support**
- **Basic plan:** Access to AWS documentation, whitepapers, forums, and AWS Health Dashboard.
- For malicious/abusive use of AWS resources → **AWS Abuse Team** (available regardless of support plan).
- For billing and account inquiries (without Enterprise plan) → **AWS Customer Service**.

### 💡 Exam Tips
- 24/7 phone + chat tech support starts at **Business** plan.
- TAM and Concierge = **Enterprise only**.
- Infrastructure Event Management included free = **Enterprise only** (Business pays extra).
- Basic + Developer = **7 core Trusted Advisor checks**. Business + Enterprise = **all checks**.
- "Quick billing/account guidance with Enterprise plan" → **Support Concierge** (not TAM, not Customer Service).

---

## 13. Deployment, Automation & Developer Tools

### Key Concepts

**AWS Management Console**
- Web-based graphical user interface (GUI) to manage AWS resources.
- The primary interface for visual resource management.

**AWS CLI (Command Line Interface)**
- Tool to manage AWS services via command line.
- Requires **Access Keys** (not username/password).

**AWS SDK (Software Development Kit)**
- Libraries/APIs for interacting with AWS services from code (Python, Java, Node.js, etc.).
- Used by developers to call AWS services programmatically.

**AWS CloudFormation**
- **Infrastructure as Code (IaC)** — define your entire AWS infrastructure in a template file (JSON/YAML).
- Automates provisioning and manages resource dependencies.
- Enables repeatable, consistent deployments across environments.

**AWS Elastic Beanstalk**
- PaaS offering — upload your code and Beanstalk handles: capacity provisioning, load balancing, auto scaling, monitoring.
- Faster deployment for developers without managing infrastructure.
- Supports: Java, .NET, PHP, Node.js, Python, Ruby, Go, Docker.

**AWS CloudFormation vs. Elastic Beanstalk**

| Feature | CloudFormation | Elastic Beanstalk |
|---------|---------------|-------------------|
| Type | IaC (any AWS resource) | PaaS (web apps) |
| Control | Full control over all resources | Managed environment |
| Use case | Full infrastructure automation | App deployment without infra management |

**AWS Auto Scaling**
- Dynamically adjusts compute capacity to match demand.
- Implementing **elasticity** as a cloud architecture principle.
- Works with: EC2, ECS tasks, DynamoDB, Aurora Replicas, Spot Fleets.

**AWS Developer Tools (DevOps)**
- **CodeCommit:** Git-based source code repository (like GitHub on AWS).
- **CodeDeploy:** Automates application deployments to EC2, Lambda, and **on-premises** servers.
- **CodePipeline:** Continuous integration/continuous delivery (CI/CD) pipeline orchestration.
- **AWS OpsWorks:** Configuration management using **Chef** and **Puppet**; can deploy to **on-premises** servers.
- **AWS X-Ray:** Distributed tracing for debugging and analyzing microservices performance.

**Other Tools**
- **Amazon Lightsail:** Simplified cloud platform for smaller workloads; easiest way to run WordPress/simple apps on AWS.
- **AWS Marketplace:** Buy/sell third-party software solutions that run on AWS.
- **AWS Quick Start:** Pre-built reference deployments for rapidly deploying popular IT solutions on AWS.

### 💡 Exam Tips
- "Deploy to on-premises servers" → **CodeDeploy** or **OpsWorks** (not CloudFormation, not Beanstalk).
- "Adjust compute dynamically to reduce cost" → **elasticity** (Auto Scaling is the implementation).
- AWS CLI → needs **Access Keys**, not username/password.
- SDK = call AWS from code (multiple programming languages). CLI = command-line terminal.
- "Rapidly deploy a popular IT solution immediately" → **AWS Quick Start** reference deployments.
- Elastic Beanstalk and CloudFormation are both deployment automation tools frequently tested together.
- Amazon Lightsail = simplest way to launch a WordPress site (not Elastic Beanstalk, not EC2 directly).

---

## 14. Serverless & Containers

### Key Concepts

**AWS Lambda**
- **Serverless compute** — run code without provisioning or managing servers.
- Pay only for compute time consumed (no charge when code is not running).
- Automatically scales — from 1 request to thousands per second.
- AWS manages: capacity, operating system, patching, scaling.
- Customer manages: code, access control, data.
- Maximum execution timeout: 15 minutes.
- **Lambda is NOT directly callable from any mobile app** — must go through API Gateway.
- Part of the AWS Serverless Platform.

**AWS Serverless Platform Services**
- AWS Lambda, Amazon S3, Amazon DynamoDB, Amazon API Gateway, Amazon SNS, Amazon SQS, AWS Step Functions, Amazon Kinesis.
- **NOT part of serverless:** Amazon EC2, Amazon EMR (these require server management).

**Amazon ECS (Elastic Container Service)**
- Managed **Docker container** orchestration service.
- Provides compute resources for containerized applications.

**AWS Fargate**
- **Serverless compute for containers** — run containers without managing EC2 instances.
- Works with both ECS and EKS.
- AWS manages the underlying infrastructure (OS patching, capacity).
- Customer manages: container images, code, data.

**Amazon EKS (Elastic Kubernetes Service)**
- Managed **Kubernetes** service on AWS.

**Lambda vs. EC2 (Shared Responsibility)**

| Responsibility | EC2 | Lambda |
|---------------|-----|--------|
| OS maintenance | Customer | AWS |
| Capacity management | Customer (+ Auto Scaling) | AWS |
| Application code | Customer | Customer |
| Access control | Customer | Customer |

### 💡 Exam Tips
- Lambda = **serverless**. EC2 = **server-based**. The key exam distinction is who manages OS/capacity.
- S3 and DynamoDB **scale automatically** without intervention — they're essentially serverless.
- Fargate = serverless **for containers** (you don't manage EC2, just containers).
- "After migrating EC2 to Lambda, what does AWS now manage?" → **Capacity management + OS maintenance**.
- AWS Lambda cannot be called directly from mobile apps (use API Gateway as the trigger).

---

## 15. Migration Services

### Key Concepts

**AWS Database Migration Service (DMS)**
- Migrates databases to AWS with **minimal downtime** to the source database.
- Works for homogeneous (same DB engine) and heterogeneous (different engines) migrations.
- Source stays fully operational during migration.

**AWS Server Migration Service (SMS)**
- Migrates on-premises VMs (VMware, Hyper-V, Azure) to AWS as AMIs.

**AWS Application Discovery Service**
- Helps plan migration by collecting data about on-premises data centers (servers, dependencies, usage).
- Not for actual migration — for discovery and planning.

**AWS Migration Hub**
- Central dashboard to track the progress of application migrations across multiple AWS and partner migration tools.

**Data Transfer Services**

| Service | Use Case | Scale |
|---------|----------|-------|
| **S3 Transfer Acceleration** | Speed up S3 uploads over internet | GBs–TBs |
| **AWS Snowball** | Offline bulk data transfer; built-in compute | Petabytes (up to ~80 TB per device) |
| **AWS Snowmobile** | Massive offline data transfer | **Exabyte scale** (100 PB per truck) |
| **AWS Direct Connect** | Ongoing dedicated private connection | Any |

**Migration Planning Support**
- **AWS Professional Services:** Helps evaluate and plan migration.
- **AWS Partner Network (APN):** Partner ecosystem that provides tools and services for migration.

### 💡 Exam Tips
- DMS = **database migration** without disrupting the source. Server Migration Service = VM migration.
- "60 Petabytes" or "Exabyte-scale" → **Snowmobile**. General large data → **Snowball**.
- Snowball: (1) built-in compute to process locally, (2) secure large data transfer — both are correct features.
- Application Discovery Service = planning; Migration Hub = tracking; DMS/SMS = executing.

---

## 16. AWS Organizations & Governance

### Key Concepts

**AWS Organizations**
- Service to centrally manage **multiple AWS accounts** under a single umbrella.
- Key benefits: consolidated billing + centralized access control.
- Enables creation of **Organizational Units (OUs)** to group accounts.

**Service Control Policies (SCPs)**
- Applied at the **Organization or OU level** (not individual IAM users).
- Restrict which AWS **services and actions** are available in member accounts.
- Act as guardrails — even if an IAM policy grants access, SCP can override it.
- SCPs do NOT grant permissions — they only restrict them.

**Consolidated Billing**
- All accounts in the organization share a single payment method.
- AWS aggregates usage across all accounts → **volume discounts** apply.
- Reserved Instances can be shared across accounts.

**Cost Allocation Tags**
- Applied to resources to track costs by project, department, or team.

**AWS Acceptable Use Policy**
- Policy document (not a service) describing what customers are **prohibited from doing** on AWS.
- Publicly available at aws.amazon.com/aup.

**AWS Artifact**
- Access AWS compliance reports (SOC, PCI, ISO), certifications, and agreements.
- Manage AWS agreements (BAA for HIPAA, NDA) through Artifact.

### 💡 Exam Tips
- SCPs are a feature of **AWS Organizations** — not IAM.
- SCPs restrict; they **don't grant** permissions. IAM policies grant permissions within those restrictions.
- Consolidated billing → volume discounts + single bill. It is **NOT** just for informational purposes.
- Reserved Instance discounts from one account → **all accounts** can benefit (not just master account).

---

## 17. Well-Architected Framework & Design Principles

### Key Concepts

**Six Pillars of the Well-Architected Framework**
1. **Operational Excellence** — Run and monitor systems to deliver business value; automate changes.
2. **Security** — Protect information and systems; implement least privilege, encrypt data, enable traceability.
3. **Reliability** — Ability to recover from failures, scale to meet demand, mitigate disruptions.
4. **Performance Efficiency** — Use computing resources efficiently; adapt to changing needs.
5. **Cost Optimization** — Avoid unnecessary costs; use right-sized resources.
6. **Sustainability** — Minimize environmental impact of running cloud workloads.

**Core Cloud Design Principles**

**Elasticity**
- Automatically provision resources based on demand changes.
- "Adjusting compute capacity dynamically to reduce cost" = implementing **elasticity**.
- Eliminates the need to guess future capacity.

**Decoupling (Loose Coupling)**
- Reduce inter-dependencies between application components.
- If one component fails, others continue to function.
- Use SQS, SNS to decouple services.
- "Advantages of decoupling" = reduces impact of failures on other components.

**Design for Failure**
- Assume everything will fail; build systems that survive failures.
- Multi-AZ deployments, health checks, Auto Scaling, and backups implement this.

**Horizontal Scaling (Think Parallel)**
- Add more instances rather than bigger instances.
- Improves fault tolerance and availability.

**Disposable Resources (vs. Fixed Servers)**
- Treat servers as temporary — terminate and replace rather than patch in place.
- Enables immutable infrastructure and faster recovery.

**Security Design Principles (7 Principles)**
- Enable traceability (CloudTrail, CloudWatch logs)
- Apply security at all layers
- Automate security best practices
- Protect data in transit and at rest
- Keep people away from data
- Prepare for security events
- **Use IAM roles for temporary access instead of long-term credentials**

**AWS Reliability Concepts**
- Automatically provisioning new resources to meet demand ✅
- Ability to recover quickly from failures ✅
- Applying least privilege ❌ (this is security, not reliability)
- All services being global ❌ (not true, and not a reliability feature)

### 💡 Exam Tips
- **Elasticity** = auto-scaling resources up/down. **Agility** = speed of provisioning (minutes vs. weeks).
- **Reliability** = recover from failures + auto-provision. Not "least privilege" (that's security).
- "Decoupling" → reduces failure cascade. SQS is the primary decoupling service.
- "Think parallel" = horizontal scaling, not parallel processing.
- Disposable resources + loose coupling are the two most-tested cloud design principles.
- Security design principles: use **IAM roles** (temporary) not long-term credentials + enable **traceability**.

---

## Quick Reference: Service → Purpose Cheat Sheet

| Service | One-Line Purpose |
|---------|-----------------|
| **CloudTrail** | Log all API calls (who did what) |
| **CloudWatch** | Monitor metrics and set alarms |
| **AWS Config** | Track resource configuration changes |
| **Trusted Advisor** | Best practice recommendations |
| **Shield** | DDoS protection |
| **WAF** | Block SQL injection, XSS |
| **GuardDuty** | Threat detection via ML |
| **Inspector** | Vulnerability scanning for EC2 |
| **Macie** | Sensitive data discovery in S3 |
| **KMS** | Managed encryption key service |
| **CloudHSM** | Your own keys on dedicated hardware |
| **Artifact** | Compliance reports and agreements |
| **Route 53** | DNS + domain registration |
| **CloudFront** | CDN via Edge Locations |
| **Direct Connect** | Private dedicated line to AWS |
| **Transit Gateway** | Hub for thousands of VPCs |
| **ElastiCache** | In-memory caching (Redis/Memcached) |
| **Redshift** | Data warehouse for analytics |
| **DynamoDB** | NoSQL key-value database |
| **Aurora** | MySQL/PostgreSQL-compatible, auto-scales |
| **DMS** | Migrate databases to AWS |
| **Snowball** | Offline bulk data transfer + local compute |
| **Snowmobile** | Exabyte-scale data transfer |
| **Storage Gateway** | Hybrid on-premises to cloud storage |
| **Lambda** | Serverless compute (event-driven) |
| **Fargate** | Serverless containers |
| **CloudFormation** | Infrastructure as Code |
| **Elastic Beanstalk** | PaaS app deployment |
| **Lightsail** | Simplified hosting for small apps |
| **CodeCommit** | Git source code repository |
| **CodeDeploy** | Automated deployment (incl. on-premises) |
| **Organizations** | Manage multiple AWS accounts |
| **Budgets** | Cost alerts and budget tracking |
| **Cost Explorer** | Visualize cost and usage history |
| **SNS** | Push notifications / pub-sub messaging |
| **SQS** | Message queue for decoupling |
| **Kinesis** | Real-time streaming data processing |

---

*Last updated from 23 Practice Exams | Good luck on your exam! ☁️*
