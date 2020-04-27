export PYTHONPATH = $(CURDIR)/../../..
.SECONDARY:

.SECONDEXPANSION:

calculate_mix/%: ../mixtures/$$*
	mkdir -p $@
	cd $@; python3 -m flow_models.elephants.calculate -x $(*F) --save $(CURDIR)/../mixtures/$*

calculate_data/%: ../histograms/$$*_b0.csv
	mkdir -p $@
	cd $@; python3 -m flow_models.elephants.calculate -x $(*F) --save $(CURDIR)/../histograms/$*_b0.csv

simulate_mix/%: ../mixtures/$$*
	mkdir -p $@
	cd $@; python3 -m flow_models.elephants.simulate -x $(*F) -n 1000000000 -r 5 --seed 0 --save --affinity $(CURDIR)/../mixtures/$*

simulate_data/%: ../histograms/$$*_b0.csv
	mkdir -p $@
	cd $@; python3 -m flow_models.elephants.simulate -x $(*F) -n 1000000000 -r 5 --seed 0 --save --affinity $(CURDIR)/../histograms/$*_b0.csv

plot: simulate_mix/all
	python3 -m flow_models.elephants.plot --one simulate_mix/all/length simulate_mix/all/size
