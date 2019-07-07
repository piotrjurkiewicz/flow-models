# agh_201506

P. Jurkiewicz, G. Rzym and P. Boryło, "How Many Mice Make an Elephant? Modelling Flow Length and Size Distribution of Internet Traffic", arXiv:1809.03486, 2018. Available: http://arxiv.org/abs/1809.03486

Based on NetFlow records collected on the Internet-facing interface of the AGH University of Science and Technology network during the consecutive period of 30 days.

Dormitories, populated with nearly 8000 students, generated 69% of the traffic. The rest of the university (over 4000 employees) generated 31%. In the case of dormitories, 91% of traffic was downstream traffic (from the Internet).
In the case of rest of the university, downstream traffic made up 73% of the total traffic. Therefore, this model can also be considered as representative of residential traffic.

In order to reproduce all steps towards fitting this model, please follow the [Makefile](Makefile).

| Parameter | Value | Unit |
| - | -: | -: |
| Dataset name| agh_201506 | |
| Exporter | Cisco router | |
| L2 technology | Ethernet | |
| Sampling rate | none | |
| Active timeout | 300 | seconds |
| Inactive timeout | 15 | seconds|
| | | |
| Number of flows | 4 032 376 751 | flows |
| Number of packets | 316 857 594 090 | packets |
| Number of bytes | 275 858 498 994 998 | bytes |
| Average flow length | 78.578370 | packets |
| Average flow size | 68410.894128 | octets |
| Average packet size | 870.607188 | bytes |

| Flows of length up to | | Make up % | |
| -: | -: | -: | -: |
| (packets) | flows | packets | octets |
| | | | |
| 1 | 47.8326 | 0.6087 | 0.1047 |
| 2 | 65.3421 | 1.0544 | 0.1728 |
| 4 | 74.8933 | 1.4696 | 0.2537 |
| 8 | 84.1319 | 2.1958 | 0.4412 |
| 10 | 86.5086 | 2.4832 | 0.5343 |
| 100 | 97.3478 | 6.3895 | 2.4322 |
| 1 000 | 99.4544 | 14.3271 | 8.0737 |
| 10 000 | 99.8922 | 30.3569 | 22.8925 |
| 100 000 | 99.9896 | 67.8990 | 61.0966 |
| 1 000 000 | 99.9998 | 93.5945 | 92.1873 |

| Flows of size up to | | Make up % | |
| -: | -: | -: | -: |
| (octets) | flows | packets | octets |
| | | | |
| 64 | 4.3082 | 0.0548 | 0.0040 |
| 128 | 32.3376 | 0.4196 | 0.0424 |
| 256 | 56.8711 | 0.9477 | 0.1030 |
| 512 | 71.1101 | 1.4143 | 0.1780 |
| 1 024 | 79.0397 | 1.9054 | 0.2622 |
| 1 500 | 82.6085 | 2.2216 | 0.3275 |
| 4 096 | 89.7285 | 3.3288 | 0.5845 |
| 10 000 | 94.0991 | 4.6438 | 0.9875 |
| 100 000 | 98.4030 | 10.0929 | 2.9510 |
| 1 000 000 | 99.6050 | 21.0802 | 8.2231 |
| 10 000 000 | 99.9165 | 43.8534 | 21.7954 |
| 100 000 000 | 99.9876 | 69.9581 | 53.1475 |
| 1 000 000 000 | 99.9997 | 93.5540 | 90.2202 |
