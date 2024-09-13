First packets mirroring
***********************

The first_mirror subpackage enables simulations and analytical calculations of the first N packets mirroring feature found in the latest generation of SDN switches. This feature involves the switch's CPU or SDN controller receiving copies of the initial packets from each new flow in the switch's dataplane. The controller can then perform packet inspection and flow identification. By analyzing various aspects of the new flows, including the connection setup procedure, packet size, and time gaps between the first packets, the controller can determine the associated application, even for encrypted connections.

This capability is also valuable for early detection of elephant flows. By classifying flows based on their first packets, the need for mid-flow rerouting is eliminated. Furthermore, it ensures that for the majority of a flow's lifespan, it will be subject to traffic engineering mechanisms specifically designed for elephant flows, such as individual routing paths. Additionally, the controller can continuously learn and refine its detection models based on the stream of first packets from flows.

Tools
=====

.. toctree::
   :glob:
   :maxdepth: 1

   first_mirror/*
