# ICT

ICT is the inter-cluster ML protocol. It supports the following:
1. Collects cluster info (machine type, data location)
2. Computes sync control (e.g., cluster 1 nccl, inter-cluster gloo, inter-cluster use mirror descent)

I am building a prototype that doesn't collect any information yet, but has fault tolerance on top of them.

## Test Case 1: 2 Clusters. 2 Replica Groups within a cluster

The kind of abstractions that we would need is the following:

1. Currently, let us define intra and inter-domain controllers. The controllers will currently use the Lighthouse as an abstraction.

2. For the intra domain controller, let us directly use TorchFT's setup (for now) + Error Bus. This error bus would give fault tolerance opportunities (TBD)

3. For the inter domain controller, 

1. For the local lighthouse, we can run as normal. We will use the TCPStore of the rank 0 replica group.

2. We need to introduce a "cluster_id" concept.

3. We want to make this recursive. So that 


Algorithm:

At the step before sync:
    Request Cluster ID information
    Get it through the Lighthouse



