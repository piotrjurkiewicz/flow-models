dumps:
	mkdir -p dumps
	cd dumps; zstd -d --stdout ~/nfcaps.tar.zst | tar --strip-components 1 --wildcards -xvf - 'nfcaps/2015-06-*'

cleaned: dumps
	mkdir -p cleaned
	for file in dumps/2015-06-*/nfcapd.* ; \
	do \
	    nice nfdump -y -r $$file -w cleaned/$${file##*/} 'router ip 10.156.119.1 and if 83 and duration < 1000000' ; \
	done

merged: cleaned
	PYTHONPATH=../../../flow_models nice pypy3 -m flow_models.merge -i nfcapd -o binary -I 15 -A 300 -O merged cleaned

sorted: merged
	PYTHONPATH=../../../flow_models nice python3 -m flow_models.sort -k merged/last merged/last_ms -O sorted merged

histograms/length.csv: sorted
	mkdir -p histograms
	PYTHONPATH=../../../flow_models nice pypy3 -m flow_models.hist -i binary -x length -b 12 sorted > histograms/length.csv

histograms/length_b0.csv: sorted
	mkdir -p histograms
	PYTHONPATH=../../../flow_models nice pypy3 -m flow_models.hist -i binary -x length -b 0 sorted > histograms/length_b0.csv

histograms/size.csv: sorted
	mkdir -p histograms
	PYTHONPATH=../../../flow_models nice pypy3 -m flow_models.hist -i binary -x size -b 12 sorted > histograms/size.csv

histograms/size_b0.csv: sorted
	mkdir -p histograms
	PYTHONPATH=../../../flow_models nice pypy3 -m flow_models.hist -i binary -x size -b 0 sorted > histograms/size_b0.csv

histograms: histograms/length.csv histograms/length_b0.csv histograms/size.csv histograms/size_b0.csv

mixtures: histograms
	PYTHONPATH=../../../flow_models python3 -m flow_models.fit -i 400 -U 6 -L 4 -y flows histograms/length.csv
	PYTHONPATH=../../../flow_models python3 -m flow_models.fit -i 400 -U 6 -L 4 -y packets histograms/length.csv
	PYTHONPATH=../../../flow_models python3 -m flow_models.fit -i 400 -U 6 -L 4 -y octets histograms/length.csv
	PYTHONPATH=../../../flow_models python3 -m flow_models.fit -i 400 -U 0 -L 8 -y flows histograms/size.csv
	PYTHONPATH=../../../flow_models python3 -m flow_models.fit -i 400 -U 0 -L 5 -y packets histograms/size.csv
	PYTHONPATH=../../../flow_models python3 -m flow_models.fit -i 400 -U 0 -L 5 -y octets histograms/size.csv

plots: mixtures
	mkdir -p plots
	cd plots; PYTHONPATH=../../../../flow_models python3 -m flow_models.plot --one -P hist comp --format pdf -x length ../histograms/length.csv ../mixtures/length
	cd plots; PYTHONPATH=../../../../flow_models python3 -m flow_models.plot --one -P hist comp --format pdf -x size ../histograms/size.csv ../mixtures/size

summary: plots
	PYTHONPATH=../../../flow_models python3 -m flow_models.summary histograms/length_b0.csv
	PYTHONPATH=../../../flow_models python3 -m flow_models.summary histograms/size_b0.csv
