First packets mirroring (new in 2.0)
************************************

.. toctree::
   :glob:
   :maxdepth: 2

   first_mirror/*

The framework can be also used to perform simulations and analytical calculations of flow's first packets mirroring feature. Such a mechanism can be implemented in a switch dataplane and mirror first N packets of a new flow to the switch's CPU or controller. By inspecting the beginning of new flows, the connection set-up procedure, and the size and gaps between the first packets of a flow, one can identify the application of a flow, even for encrypted connections. More advanced techniques can even single out malicious flows from benign flows by inspecting the same set of parameters.

Such a feature can be also used to perform detection of elephant flow candidates in their early stages. Classification with the first packets would be the most beneficial, as it would allow avoiding rerouting it in the middle. Moreover, it means that fir almost whole of its lifetime flow would be covered by elephant-specific TE mechanisms, for example individual routing path.
